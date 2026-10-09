"""BusinessEnv 实现：ai_to_c_server_react.orchestrator.chat_stream。

职责：
1. setup_once：worker 进程内幂等地完成
   - sys.path 注入 ai2c
   - bootstrap.apply_train_patches()（注入 Anthropic dispatch + 隔离 SESSION_DIR）
   - install_trace_bridge（agentic_train_offline.llm_bridge）
   - 注册主脑识别器（依靠 ReAct system prompt 的稳定前缀）
   - 注册 vLLM endpoint 转发（主脑请求改打训练 actor）
   - 注册 react 业务侧的 importer（已在 bridge 默认列表里，本 env 不需要追加）
   - 可选：把 SESSION_DIR 指向 tmpfs，避免 session 文件堆积（R3）

2. run：调用 [`ai_to_c_server_react.orchestrator.chat_stream`](../ai2c/ai_to_c_server_react/orchestrator.py:438)
   一次，得到完整 trajectory 的 (records, final_answer, env_info)。
   - 不传 session_id（业务自动 uuid，天然隔离）
   - 过滤掉子 agent 的 trace（保留 extra["role"] == "mainbrain"）
   - 过滤 toA-react 协议标记（__PASSTHROUGH_INTENT__、__RESULT_MAP__、__DEBUG_TRACE__）
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .ai_to_c_action_parser import AiToCReactActionParser
from .base import BusinessEnv, EnvRunResult

logger = logging.getLogger(__name__)

# 主脑（ReAct planner）的 system prompt 稳定识别特征。
# 来源：ai2c/ai_to_c_server_react/planner/react_reactor.py
DEFAULT_MAIN_BRAIN_SYSTEM_PROMPT_PATTERNS: Sequence[str] = (
    "你是保险域的 ReAct 规划助手",
)


# ---------------------------------------------------------------------------
# 主脑识别器
# ---------------------------------------------------------------------------


def make_default_main_brain_decider(
    patterns: Sequence[str] = DEFAULT_MAIN_BRAIN_SYSTEM_PROMPT_PATTERNS,
):
    """返回一个 messages → bool 的函数，命中 system prompt 前缀即视为主脑。

    判定逻辑：在 messages 中找 role=='system' 的消息，content 若以任一 pattern
    开头或包含该 pattern → 视为主脑调用；否则视为子 agent。
    """
    patterns_tuple = tuple(patterns)

    def _decider(messages: List[Dict[str, str]]) -> bool:
        if not messages:
            return False
        # 通常 system 在 messages[0]
        for m in messages:
            if m.get("role") != "system":
                continue
            content = str(m.get("content") or "")
            for pat in patterns_tuple:
                if pat and pat in content:
                    return True
            # 只看第一个 system 消息就够了；继续找下一个 system 也无妨
        return False

    return _decider


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class AiToCReactEnv(BusinessEnv):
    """ai_to_c_server_react 业务 env。

    Args:
        ai_to_c_repo_path: 业务包根（含 ``ai_to_c_server_react``）的绝对路径。
            None 时自动从本文件路径推导（假定项目结构同 agenticrl3）。
        main_brain_patterns: 主脑 system prompt 识别特征列表。
        session_dir_override: 若给定，启动时设置 AI_TO_C_REACT_SESSION_DIR。
        record_subagent_traces: True 时 records 同时保留子 agent 的 trace（用于 debug），
            训练时建议保持 False（默认）。
    """

    def __init__(
        self,
        ai_to_c_repo_path: Optional[str] = None,
        main_brain_patterns: Sequence[str] = DEFAULT_MAIN_BRAIN_SYSTEM_PROMPT_PATTERNS,
        session_dir_override: Optional[str] = None,
        record_subagent_traces: bool = False,
        vllm_chat_template_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(action_parser=AiToCReactActionParser(), **kwargs)
        self.ai_to_c_repo_path = ai_to_c_repo_path
        self.main_brain_patterns = tuple(main_brain_patterns)
        self.session_dir_override = session_dir_override
        self.record_subagent_traces = record_subagent_traces
        self._vllm_chat_template_kwargs = vllm_chat_template_kwargs

        self._setup_done = False
        # 保存 vLLM 配置 + 主脑识别器，供 run() 在每次调用时重新设置 ContextVar
        # （ContextVar 的 set 只对当前线程 context 生效，asyncio.to_thread 会拷贝新 context）
        self._vllm_base_url: Optional[str] = None
        self._vllm_model_name: Optional[str] = None
        self._main_brain_decider: Optional[Any] = None

    # ------------------------------------------------------------------
    # setup_once
    # ------------------------------------------------------------------

    def setup_once(
        self,
        vllm_base_url: Optional[str] = None,
        vllm_model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # 每次都更新存储的 URL，以便 run() 可在正确的线程 context 里重新注入
        if vllm_base_url:
            self._vllm_base_url = vllm_base_url
        if vllm_model_name:
            self._vllm_model_name = vllm_model_name

        if self._setup_done:
            # 仍允许刷新 vLLM override（训练里 actor URL 可能变）
            self._maybe_set_vllm_override(vllm_base_url, vllm_model_name)
            return

        # 1) sys.path 注入：ai2c + workspace 根（agentic_train_offline）
        repo_path = self.ai_to_c_repo_path or _auto_detect_repo_path()
        if repo_path and repo_path not in sys.path:
            sys.path.insert(0, repo_path)
            logger.info("[ai_to_c_env] inserted %s into sys.path", repo_path)
        # agentic_train_offline 与 ai2c 同级；Ray worker 偶发丢 PYTHONPATH 时兜底
        if repo_path:
            ws_root = str(Path(repo_path).resolve().parent)
            if ws_root and ws_root not in sys.path:
                sys.path.insert(0, ws_root)
                logger.info("[ai_to_c_env] inserted workspace root %s into sys.path", ws_root)

        # 2) bootstrap：注入 Anthropic dispatch + 隔离 SESSION_DIR
        from agentic_train_offline.bootstrap import apply_train_patches
        apply_train_patches(session_dir=self.session_dir_override)
        logger.info("[ai_to_c_env] apply_train_patches done (SESSION_DIR=%s)", self.session_dir_override)

        # 3) install bridge + 注册主脑识别器
        from agentic_train_offline.llm_bridge import (
            install_trace_bridge,
            set_main_brain_decider,
        )

        install_trace_bridge()
        self._main_brain_decider = make_default_main_brain_decider(self.main_brain_patterns)
        set_main_brain_decider(self._main_brain_decider)

        # 4) vLLM endpoint 注入（可选；训练时必填）
        self._maybe_set_vllm_override(vllm_base_url, vllm_model_name)

        self._setup_done = True
        logger.info(
            "[ai_to_c_env] setup_once done | vllm_url=%s | model=%s",
            vllm_base_url,
            vllm_model_name,
        )

    def _maybe_set_vllm_override(
        self, base_url: Optional[str], model: Optional[str]
    ) -> None:
        from agentic_train_offline.llm_bridge import set_vllm_override

        set_vllm_override(base_url, model, chat_template_kwargs=self._vllm_chat_template_kwargs)

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        history: Optional[List[Dict[str, Any]]] = None,
        ext_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> EnvRunResult:
        """同步执行一次 chat_stream，捕获主脑 trace。

        history: 历史会话轮次（与 trace_collect 的 history 字段含义一致），
            格式为 [{"role": "user"|"assistant", "content": "..."}]。
            非空时构造带历史的 SessionState 注入 chat_stream，与 trace_collect 行为对齐。
        """
        if not self._setup_done:
            # 兼容性：未提前 setup 时也能跑（只是没有 vLLM override，会用业务默认 endpoint）
            self.setup_once()

        # _vllm_override_var / _main_brain_decider_var 均是 contextvars.ContextVar：
        # asyncio.to_thread 会拷贝一个新 context，setup_once 里在主线程的 set() 对
        # 线程池线程不可见（get() 返回 default=None）。
        # 每次 run() 开头在当前线程 context 里重新注入，确保 bridge 正确路由到 vLLM。
        if self._vllm_base_url and self._vllm_model_name:
            self._maybe_set_vllm_override(self._vllm_base_url, self._vllm_model_name)
        if self._main_brain_decider is not None:
            from agentic_train_offline.llm_bridge import set_main_brain_decider
            set_main_brain_decider(self._main_brain_decider)

        # 延迟 import，避免在 setup_once 之前误 import 业务
        from agentic_train_offline.llm_bridge import TraceRecord, trace_scope
        from ai_to_c_server_react.core.models import is_thinking_step, is_passthrough_intent, is_result_map, is_debug_trace
        from ai_to_c_server_react.orchestrator import chat_stream

        records: List[TraceRecord] = []
        final_chunks: List[str] = []
        ts_start = time.time()
        status = "ok"
        err: Optional[str] = None
        session_id = f"rl_{uuid.uuid4().hex[:12]}"

        # 与 trace_collect 保持一致：有 history 时构造 pre_session 注入
        pre_session = _build_pre_session(session_id, history, ext_params) if history else None

        try:
            with trace_scope(records.append):
                for piece in chat_stream(
                    query,
                    session_id=session_id,
                    ext_params=ext_params or {},
                    **({"session": pre_session} if pre_session is not None else {}),
                ):
                    if is_thinking_step(piece):
                        continue
                    if is_passthrough_intent(piece):
                        continue
                    if is_result_map(piece):
                        continue
                    if is_debug_trace(piece):
                        continue
                    if piece.startswith("\n__SESSION_ID__"):
                        continue
                    final_chunks.append(piece)
        except Exception as exc:  # noqa: BLE001
            status = "error"
            err = f"{type(exc).__name__}: {exc}"
            logger.exception("[ai_to_c_env] chat_stream failed for query=%r", query[:80])
            # tail of traceback to env_info（避免太长）
            err = f"{err}\n{traceback.format_exc()[-800:]}"

        elapsed = time.time() - ts_start
        final_answer = "".join(final_chunks)

        # 过滤：默认只保留主脑 trace
        if self.record_subagent_traces:
            kept_records: List[TraceRecord] = list(records)
        else:
            kept_records = [
                r for r in records if _is_mainbrain_record(r)
            ]

        env_info: Dict[str, Any] = {
            "query": query,
            "status": status,
            "elapsed_sec": round(elapsed, 3),
            "num_total_calls": len(records),
            "num_mainbrain_calls": sum(1 for r in records if _is_mainbrain_record(r)),
            "session_id": session_id,
            "history_turns": len(history) if history else 0,
        }
        if err:
            env_info["error"] = err

        return EnvRunResult(records=kept_records, final_answer=final_answer, env_info=env_info)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _build_pre_session(
    session_id: str,
    history: Optional[List[Dict[str, Any]]],
    ext_params: Optional[Dict[str, Any]] = None,
) -> Any:
    """根据 history 列表构造 SessionState，与 trace_collect._build_session_with_history 逻辑一致。

    history 格式：[{"role": "user"|"assistant"|"system", "content": "..."}]
    注意：不要把本轮 query 放进 history，chat_stream 会自动 append。
    """
    try:
        from ai_to_c_server_react.core.models import ConversationTurn, SessionState
    except ImportError:
        # setup_once 之前调用或环境未就绪时，返回 None（兜底走无历史分支）
        logger.warning("[ai_to_c_env] _build_pre_session: import failed, skipping history injection")
        return None

    session = SessionState(session_id=session_id)
    turns = []
    for h in (history or []):
        if not isinstance(h, dict):
            continue
        content = str(h.get("content") or "").strip()
        if not content:
            continue
        role = str(h.get("role") or "assistant").strip().lower()
        if role not in ("user", "assistant", "system"):
            role = "assistant"
        turns.append(ConversationTurn(role=role, content=content))
    session.conversation_history = turns
    session.ext_params = dict(ext_params or {})
    return session


def _is_mainbrain_record(rec: Any) -> bool:
    """TraceRecord.extra["role"] == "mainbrain" → 主脑。

    若 extra 不含 role（如 bridge 未启用主脑识别器，旧行为），保守视为 True
    （让 SFT 场景也能复用本 env）。
    """
    extra = getattr(rec, "extra", None) or {}
    role = extra.get("role")
    if role is None:
        return True
    return role == "mainbrain"


def _auto_detect_repo_path() -> Optional[str]:
    """解析 ai2c 业务根目录。

    优先 AI2C_REPO_PATH；其次 ai2c_workspace 布局（research/verl_research/... → ../../../../ai2c）；
    再回退旧布局 parents[4]/ai2c。
    """
    import os

    env = os.environ.get("AI2C_REPO_PATH", "").strip()
    if env and Path(env).is_dir():
        return env
    here = Path(__file__).resolve()
    # envs → agentic → recipe → verl_research → research → ai2c_workspace
    for idx in (5, 4):
        candidate = here.parents[idx] / "ai2c"
        if candidate.is_dir():
            return str(candidate)
    return None