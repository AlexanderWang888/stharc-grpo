"""GenericBenchmarkEnv(BusinessEnv) —— 与 ai2c 业务完全解耦的 STHARC 环境实现。

设计依据：research/dataset_explore/IDEA2_IMPLEMENTATION_PLAN.md §5.2。

职责：
1. 自建一个业务无关的 ReAct 循环：system prompt（工具说明）+ history + query →
   反复调用 vLLM /chat/completions → 解析 Action → 执行工具 → 把 observation
   接回对话 → 直到 finish 或达到 max_steps。
2. 暴露覆盖全部 7 种 harness 操作的工具集（skill_create/modify/delete/load，
   memory_write/read/update/invalidate），工具执行时直接调用
   ``stharc.harness_tracker.get_tracker()`` 记录 write_events / read_set——
   不需要任何 monkeypatch（对比 ai2c 版依赖 ``ai2c_harness_patch`` 拦截生产代码）。
3. 状态存储：``stharc.generic_harness_state.GenericHarnessState``
   （进程内 dict，按 chain 隔离），通过 ext_params["_rl_session_id"] 定位沙箱。
4. ``USES_AI2C_HARNESS_PATCH = False``：告知 ``stharc.core.rollout_with_tracking``
   跳过 ai2c 专属的 monkeypatch / RL context 绑定步骤（见 core.py 注释）。

与 AiToCReactEnv 的关键差异：LLM 调用不走 agentic_train_offline.llm_bridge 的
monkeypatch 转发，而是直接对 vLLM 的 OpenAI 兼容 /chat/completions 端点发起同步
HTTP 请求（复用与 bridge._send_to_vllm 相同的请求结构，但完全自包含）。
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence

from .base import BusinessEnv, EnvRunResult
from .generic_action_parser import GenericReactActionParser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 本地 TraceRecord（与 agentic_train_offline.llm_bridge.TraceRecord 字段对齐，
# 但不 import 该模块 —— 保持与 ai2c 业务代码零耦合）
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _GenericTraceRecord:
    messages: List[Dict[str, str]]
    response: str
    model: Optional[str] = None
    ts_start: float = 0.0
    ts_end: float = 0.0


# ---------------------------------------------------------------------------
# 工具说明（system prompt 的一部分）
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = """You are an autonomous agent that solves tasks step by step using tools.

You have access to a persistent "harness" (skills + memory) that is shared across
all tasks in the current session. Use it to write down information you will need
later, and to read back information written by earlier tasks in this session.

Available tools:
  - skill_create(skill_id: str, content: str): create a new named skill/procedure note.
  - skill_modify(skill_id: str, content: str): overwrite an existing skill's content.
  - skill_delete(skill_id: str): permanently remove a skill. It can no longer be loaded afterwards.
  - skill_load(skill_id: str): read back a skill's content.
  - memory_write(key: str, value: str): store a new fact under `key`.
  - memory_update(key: str, value: str): overwrite an existing fact under `key`.
  - memory_invalidate(key: str, new_value: str = ""): mark the old value under `key` as stale/wrong;
      if `new_value` is given, it replaces the old value as the corrected fact, otherwise the key
      becomes unreadable (you must not use it anymore).
  - memory_read(key: str): read back the fact stored under `key`. May return NOT_FOUND or an
      INVALIDATED notice if the value has been invalidated with no replacement.
  - finish(answer: str): submit your final answer for the CURRENT task and stop.

Output format for EVERY step (no exceptions):
    Thought: <one or two sentences of reasoning about what to do next>
    Action: {"name": "<tool_name>", "arguments": {...}}

To submit your final answer, use:
    Action: {"name": "finish", "arguments": {"answer": "<your final answer>"}}

Rules:
  - Output exactly one Thought/Action pair per step. Do not output more than one Action.
  - `arguments` must be a valid JSON object matching the tool's parameters.
  - Keep answers concise: a bare fact, number, or short phrase, matching exactly what is asked.
  - Decide yourself whether and when to use harness tools; the same tool set is always available.
  - If a tool call fails or returns NOT_FOUND, adapt your plan instead of repeating the same call.
  - If a later answer depends on something stored in the harness, load/read it before finish;
    do not invent that content from parametric memory alone.
  - The system message may include a live catalog of skill/memory names currently stored (no contents).
"""

# 正常格式 few-shot：完整「存原值 → 同 key 读回」（与评测链不同源/不同 key）。
# 给小模型冷启动；不泄露评测 canonical key（secret_fact_* 等）。
_FORMAT_FEW_SHOT = """
Format examples (unrelated toy numbers; follow the same Thought/Action style):

Example A — a lasting fact appears; store the ORIGINAL value, then answer the side calculation:
User: The demo dock code is 7350. What do you get if you add 7 to it? Answer with the number only.
Assistant:
Thought: I may need the dock code later, so store the original value 7350. Side calc: 7350 + 7 = 7357.
Action: {"name": "memory_write", "arguments": {"key": "demo_dock_code", "value": "7350"}}
User: Observation: 已写入 memory['demo_dock_code']。
Assistant:
Thought: Now report the side-calculation answer.
Action: {"name": "finish", "arguments": {"answer": "7357"}}

Example B — the value is not in this question; read the same key you stored earlier:
User: Take the demo dock code, multiply it by 2, and answer with the number only.
Assistant:
Thought: The code is not in this question; read the key I stored before.
Action: {"name": "memory_read", "arguments": {"key": "demo_dock_code"}}
User: Observation: 7350
Assistant:
Thought: 7350 * 2 = 14700.
Action: {"name": "finish", "arguments": {"answer": "14700"}}
"""


def _format_harness_catalog(sandbox: Dict[str, Any]) -> str:
    """Inject live skill/memory name lists (no contents) into the user turn."""
    skills = sorted((sandbox.get("skills") or {}).keys())
    mem = sandbox.get("memory") or {}
    keys = sorted(k for k, v in mem.items() if not (isinstance(v, dict) and v.get("invalidated")))
    skill_s = ", ".join(f"'{s}'" for s in skills) if skills else "(none yet)"
    key_s = ", ".join(f"'{k}'" for k in keys) if keys else "(none yet)"
    return (
        "[Harness catalog — names only; load/read for content]\n"
        f"  skills currently stored: [{skill_s}]\n"
        f"  memory keys currently stored: [{key_s}]\n"
    )


class GenericBenchmarkEnv(BusinessEnv):
    """业务无关的 STHARC env：自建 ReAct 循环 + 8 个 harness 工具 + 直连 vLLM。

    Args:
        max_steps: 单个 task 最多允许多少个 ReAct 步（含最终 finish 步）。
        system_prompt: 覆盖默认工具说明 system prompt。
        temperature / max_tokens / request_timeout: vLLM 调用参数。
        chat_template_kwargs: 传给 vLLM /chat/completions 的 chat_template_kwargs
            （如 Qwen3 系列 ``{"enable_thinking": False}``），由 AgentLoop 同步注入
            （见 ``STHARCAgentLoop.__init__`` 对 ``_vllm_chat_template_kwargs`` 的赋值）。
    """

    # 告知 stharc.core.rollout_with_tracking 跳过 ai2c 专属 monkeypatch/RL-context 步骤
    USES_AI2C_HARNESS_PATCH: bool = False

    def __init__(
        self,
        max_steps: int = 8,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        request_timeout: float = 120.0,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        format_few_shot: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(action_parser=GenericReactActionParser(), **kwargs)
        self.max_steps = int(max_steps)
        # hydra 可传 format_few_shot=true/1；未传时读环境变量 STHARC_FORMAT_FEW_SHOT
        # 注意：不能 bool("0")——在 Python 里为 True
        if format_few_shot is None:
            raw_fs = os.environ.get("STHARC_FORMAT_FEW_SHOT", "0")
        else:
            raw_fs = format_few_shot
        if isinstance(raw_fs, bool):
            self.format_few_shot = raw_fs
        else:
            self.format_few_shot = str(raw_fs).strip().lower() in ("1", "true", "yes", "on")
        base = system_prompt or _DEFAULT_SYSTEM_PROMPT
        if self.format_few_shot:
            self.system_prompt = f"{base.rstrip()}\n{_FORMAT_FEW_SHOT}"
            logger.info("[GenericBenchmarkEnv] format_few_shot=ON (format demos appended to system)")
        else:
            self.system_prompt = base
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.request_timeout = float(request_timeout)
        self._vllm_chat_template_kwargs: Optional[Dict[str, Any]] = chat_template_kwargs

        # 由 STHARCAgentLoop 在探测到实际 vLLM 地址后直接赋值（同 AiToCReactEnv 约定）
        self._vllm_base_url: Optional[str] = None
        self._vllm_model_name: Optional[str] = None
        self._setup_done = False

    # ------------------------------------------------------------------
    # setup_once
    # ------------------------------------------------------------------

    def setup_once(
        self,
        vllm_base_url: Optional[str] = None,
        vllm_model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if vllm_base_url:
            self._vllm_base_url = vllm_base_url
        if vllm_model_name:
            self._vllm_model_name = vllm_model_name
        self._setup_done = True

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
        from ..stharc.generic_harness_state import get_sandbox
        from ..stharc.harness_tracker import get_tracker

        ext_params = ext_params or {}
        session_id = str(ext_params.get("_rl_session_id") or f"_adhoc_{id(self)}")
        frozen = bool(ext_params.get("_rl_frozen", False))
        sandbox = get_sandbox(session_id)
        tracker = get_tracker()

        # Catalog once per task, on system (not repeated on Observation turns).
        # Live names only — model still chooses whether/what to create.
        system_content = f"{self.system_prompt.rstrip()}\n\n{_format_harness_catalog(sandbox)}"
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_content}]
        for h in history or []:
            if isinstance(h, dict) and h.get("role") in ("user", "assistant") and h.get("content"):
                messages.append({"role": h["role"], "content": str(h["content"])})
        messages.append({"role": "user", "content": query})

        records: List[_GenericTraceRecord] = []
        final_answer = ""
        status = "ok"
        err: Optional[str] = None
        num_tool_calls = 0
        ts_start = time.time()

        try:
            for step in range(self.max_steps):
                step_ts_start = time.time()
                try:
                    response_text = self._call_llm(list(messages))
                except Exception as llm_exc:  # noqa: BLE001
                    # 与 AppWorldEnv 对齐：单任务多轮历史撞 vLLM max_model_len 时软结束
                    err_s = f"{type(llm_exc).__name__}: {llm_exc}"
                    low = err_s.lower()
                    if (
                        "maximum context length" in low
                        or "input_tokens" in low
                        or "context length" in low
                    ):
                        status = "context_overflow"
                        err = err_s
                        logger.warning(
                            "[GenericBenchmarkEnv] context overflow at step=%d query=%r",
                            step,
                            query[:80],
                        )
                        break
                    raise
                step_ts_end = time.time()

                records.append(
                    _GenericTraceRecord(
                        messages=[dict(m) for m in messages],
                        response=response_text,
                        model=self._vllm_model_name,
                        ts_start=step_ts_start,
                        ts_end=step_ts_end,
                    )
                )
                messages.append({"role": "assistant", "content": response_text})

                parsed = self.action_parser.parse(response_text)
                if not parsed.parse_ok:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "错误：未能解析出合法的 Action JSON，请严格按照 "
                                '"Thought: ...\\nAction: {\\"name\\":...,\\"arguments\\":{...}}" 格式输出。'
                            ),
                        }
                    )
                    continue

                action = parsed.parsed_action or {}
                name = str(action.get("name", "")).strip()

                if parsed.is_final:
                    final_answer = str((action.get("arguments") or {}).get("answer", "")).strip()
                    break

                num_tool_calls += 1
                obs = _execute_tool(
                    name=name,
                    arguments=action.get("arguments") or {},
                    sandbox=sandbox,
                    frozen=frozen,
                    tracker=tracker,
                    step=step,
                )
                messages.append({"role": "user", "content": f"Observation: {obs}"})
            else:
                status = "max_steps_exceeded"
        except Exception as exc:  # noqa: BLE001
            status = "error"
            err = f"{type(exc).__name__}: {exc}"
            logger.exception("[GenericBenchmarkEnv] run failed for query=%r", query[:80])

        elapsed = time.time() - ts_start
        env_info: Dict[str, Any] = {
            "query": query,
            "status": status,
            "elapsed_sec": round(elapsed, 3),
            "num_steps": len(records),
            "num_tool_calls": num_tool_calls,
            "session_id": session_id,
            "frozen": frozen,
            "truncation": {
                "context_overflow": status == "context_overflow",
                "max_steps_exceeded": status == "max_steps_exceeded",
                "max_steps": int(self.max_steps),
            },
        }
        if err:
            env_info["error"] = err

        return EnvRunResult(records=records, final_answer=final_answer, env_info=env_info)

    # ------------------------------------------------------------------
    # vLLM 调用
    # ------------------------------------------------------------------

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        import requests

        if not self._vllm_base_url:
            raise RuntimeError("GenericBenchmarkEnv: vllm_base_url 未设置，请先调用 setup_once")

        url = f"{self._vllm_base_url.rstrip('/')}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self._vllm_model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self._vllm_chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(self._vllm_chat_template_kwargs)

        resp = requests.post(url, json=payload, timeout=self.request_timeout)
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            # 默认的 HTTPError 只打 status code，看不到服务端返回的具体原因（比如
            # vLLM/verl 内部引擎返回的 404/400 body 里通常带 "detail" 字段，能
            # 直接定位是路由不对还是 model 名不对）。这里补上 body，方便排障。
            raise requests.exceptions.HTTPError(
                f"{exc} | url={url} | model={self._vllm_model_name!r} | "
                f"response_body={resp.text[:2000]!r}",
                response=resp,
            ) from exc
        data = resp.json()
        return str(data["choices"][0]["message"]["content"] or "")


# ---------------------------------------------------------------------------
# 工具执行（8 个 harness 操作 + finish 由 run() 主循环单独处理）
# ---------------------------------------------------------------------------


def _execute_tool(
    name: str,
    arguments: Dict[str, Any],
    sandbox: Dict[str, Any],
    frozen: bool,
    tracker: Any,
    step: int,
) -> str:
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        return f"错误：未知工具 {name!r}，请从可用工具列表中选择。"
    try:
        return handler(arguments, sandbox, frozen, tracker, step)
    except Exception as exc:  # noqa: BLE001
        return f"出错：{type(exc).__name__}: {exc}"


def _record_write(
    tracker: Any,
    op_type: str,
    target_id: str,
    sub_action: str,
    before: str,
    after: str,
    step: int,
) -> None:
    if tracker is None:
        return
    tracker.record_write(
        tool_name=op_type,
        target_id=target_id,
        args={"action": sub_action},
        before=before,
        after=after,
    )


def _record_read(tracker: Any, tool_name: str, target_id: str) -> None:
    if tracker is None:
        return
    # 无条件记录读事件（即便对象不存在 / 已被删除 / 已失效）：
    # 这是让 skill_delete / memory_invalidate 能吃到现有 compute_prospective_utility
    # 公式的关键设计——只要模型确实调用了读工具去核实，read_set 就会包含该
    # object_id，credit 就能按公式正常流转；若模型从未尝试读取（凭空回答），
    # 该 mutation 仍然拿不到 credit —— 这正是 research/dataset_explore/
    # IDEA2_IMPLEMENTATION_PLAN.md §4.2 描述的、仍待算法扩展解决的剩余盲区
    # （本实现只解决"读了但对象不存在"这一半，不解决"压根没读"这一半）。
    tracker.record_read(tool_name=tool_name, target_id=target_id)


def _tool_skill_create(args, sandbox, frozen, tracker, step) -> str:
    skill_id = str(args.get("skill_id", "")).strip()
    content = str(args.get("content", ""))
    if not skill_id:
        return "错误：缺少参数 skill_id"
    before = sandbox["skills"].get(skill_id, "")
    if not frozen:
        sandbox["skills"][skill_id] = content
    _record_write(tracker, "skill_manage", f"skill::{skill_id}", "create", before, content, step)
    return f"已创建技能 {skill_id!r}。"


def _tool_skill_modify(args, sandbox, frozen, tracker, step) -> str:
    skill_id = str(args.get("skill_id", "")).strip()
    content = str(args.get("content", ""))
    if not skill_id:
        return "错误：缺少参数 skill_id"
    before = sandbox["skills"].get(skill_id, "")
    if not before:
        return f"错误：技能 {skill_id!r} 不存在，无法 modify，请先用 skill_create。"
    if not frozen:
        sandbox["skills"][skill_id] = content
    _record_write(tracker, "skill_manage", f"skill::{skill_id}", "modify", before, content, step)
    return f"已更新技能 {skill_id!r}。"


def _tool_skill_delete(args, sandbox, frozen, tracker, step) -> str:
    skill_id = str(args.get("skill_id", "")).strip()
    if not skill_id:
        return "错误：缺少参数 skill_id"
    before = sandbox["skills"].get(skill_id)
    if before is None:
        return f"技能 {skill_id!r} 不存在，无需删除。"
    if not frozen:
        sandbox["skills"].pop(skill_id, None)
    _record_write(tracker, "skill_manage", f"skill::{skill_id}", "delete", before, "", step)
    return f"已删除技能 {skill_id!r}，之后将无法再加载。"


def _tool_skill_load(args, sandbox, frozen, tracker, step) -> str:
    skill_id = str(args.get("skill_id", "")).strip()
    if not skill_id:
        return "错误：缺少参数 skill_id"
    _record_read(tracker, "load_skill", f"skill::{skill_id}")
    content = sandbox["skills"].get(skill_id)
    if content is None:
        return f"NOT_FOUND: 技能 {skill_id!r} 不存在或已被删除。"
    return content


def _tool_memory_write(args, sandbox, frozen, tracker, step) -> str:
    key = str(args.get("key", "")).strip()
    value = str(args.get("value", ""))
    if not key:
        return "错误：缺少参数 key"
    entry = sandbox["memory"].get(key)
    before = entry.get("value", "") if entry else ""
    if not frozen:
        sandbox["memory"][key] = {"value": value, "invalidated": False}
    _record_write(tracker, "memory", f"mem::{key}", "add" if entry is None else "update", before, value, step)
    return f"已写入 memory[{key!r}]。"


def _tool_memory_update(args, sandbox, frozen, tracker, step) -> str:
    key = str(args.get("key", "")).strip()
    value = str(args.get("value", ""))
    if not key:
        return "错误：缺少参数 key"
    entry = sandbox["memory"].get(key)
    before = entry.get("value", "") if entry else ""
    if not frozen:
        sandbox["memory"][key] = {"value": value, "invalidated": False}
    _record_write(tracker, "memory", f"mem::{key}", "update", before, value, step)
    return f"已更新 memory[{key!r}]。"


def _tool_memory_invalidate(args, sandbox, frozen, tracker, step) -> str:
    key = str(args.get("key", "")).strip()
    new_value = args.get("new_value")
    if not key:
        return "错误：缺少参数 key"
    entry = sandbox["memory"].get(key)
    before = entry.get("value", "") if entry else ""
    if new_value is not None and str(new_value).strip():
        after = str(new_value)
        if not frozen:
            sandbox["memory"][key] = {"value": after, "invalidated": False}
        _record_write(tracker, "memory", f"mem::{key}", "invalidate", before, after, step)
        return f"已将 memory[{key!r}] 的旧值标记为失效，并写入更正后的新值。"
    if not frozen:
        sandbox["memory"][key] = {"value": "", "invalidated": True}
    _record_write(tracker, "memory", f"mem::{key}", "invalidate", before, "", step)
    return f"已将 memory[{key!r}] 标记为失效（无替代新值），此后不可再使用其旧值。"


def _tool_memory_read(args, sandbox, frozen, tracker, step) -> str:
    key = str(args.get("key", "")).strip()
    if not key:
        return "错误：缺少参数 key"
    _record_read(tracker, "memory_inject", f"mem::{key}")
    entry = sandbox["memory"].get(key)
    if entry is None:
        return f"NOT_FOUND: memory[{key!r}] 不存在。"
    if entry.get("invalidated"):
        return f"INVALIDATED: memory[{key!r}] 的旧值已被标记失效，不可信，当前无有效值。"
    return str(entry.get("value", ""))


_TOOL_HANDLERS = {
    "skill_create": _tool_skill_create,
    "skill_modify": _tool_skill_modify,
    "skill_delete": _tool_skill_delete,
    "skill_load": _tool_skill_load,
    "memory_write": _tool_memory_write,
    "memory_update": _tool_memory_update,
    "memory_invalidate": _tool_memory_invalidate,
    "memory_read": _tool_memory_read,
}
