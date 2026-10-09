"""AppWorldEnv(BusinessEnv) —— 接入真实 AppWorld 沙箱的 STHARC 环境实现。

设计依据：research/dataset_explore/idea2_datasets/gen_appworld.py 头部说明 +
与用户在 agentic_rl_harness_research_plan.md 讨论中确认的"AppWorld = 软依赖
（skill 复用只影响效率/鲁棒性，不影响正确性）"结论。

与 :class:`GenericBenchmarkEnv` 的关系：
    两者都复用同一套 ReAct 循环骨架（Thought/Action JSON）+
    :class:`~recipe.agentic.envs.generic_action_parser.GenericReactActionParser`
    （完全通用，不需要为 AppWorld 单独写 parser），也复用同一套
    ``stharc.generic_harness_state`` 内存态 sandbox 存 skill 内容（按
    ``ext_params["_rl_session_id"]`` 隔离/克隆，与 musique/textcraft 等数据集
    共享同一份 chain/frozen 语义，``run_frozen_chain`` 不需要为此改一行代码）。

    关键差异：AppWorld 的"工具"不是我们自定义的 8 个 harness 操作，而是
    "写 Python 代码在真实沙箱里执行"（``execute`` 工具，对应 AppWorld 官方
    CodeAct 风格）。skill_create/skill_modify/skill_delete/skill_load 仍然是
    我们自己的 harness 记账工具（用于 STHARC 信度分配），但多做一步：把 skill
    的 Python 源码 ``world.execute()`` 进当前任务的沙箱命名空间，让模型创建/
    加载完之后可以在后续 ``execute`` 步里直接调用该函数——对齐 SAGE
    (arXiv:2512.17102) "定义 skill 函数后立即可用" 的约定。

    ``memory_*`` 系列工具在这里不出现：AppWorld 本身没有独立于"技能代码"的
    "记忆事实"概念（persona/密码/邮件等都通过 apis 查询获得，不需要我们额外的
    memory harness），保持工具面与 SAGE 的 skill-only 设定一致。

正确性判分：本 env 不做任何字符串匹配——``env_info["appworld_success"]`` /
``env_info["appworld_eval"]`` 来自真实 ``world.evaluate().to_dict()``
（AppWorld 官方程序化 unit-test 判分器），交给
``recipe.agentic.reward.appworld_reward.AppWorldVerifierReward`` 使用。
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .base import BusinessEnv, EnvRunResult
from .generic_action_parser import GenericReactActionParser

logger = logging.getLogger(__name__)

# AppWorld 的 SafetyGuard 在 world.execute() 期间会全局 monkeypatch
# os.putenv / pathlib.Path.mkdir 等。同一 Ray AgentLoopWorker 进程内多个
# asyncio.to_thread(env.run) 并发时，一个 thread 启用 guard 会让其它 thread
# 的 AppWorld() 构造 / path_store.reload / setup_once 写环境变量直接
# PermissionError。因此整个 AppWorld 会话必须串行化到进程级锁。
_APPWORLD_PROCESS_LOCK = threading.RLock()
# SQLite 连接绑线程：AgentLoop 用 asyncio.to_thread 时线程池会换 thread，
# 复用上一轮缓存的 DB 会炸 "SQLite objects created in a thread..."。
# 所有 AppWorldEnv.run 本体固定丢到单线程 executor，保证同线程触库。
_APPWORLD_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="appworld_sqlite"
)
_APPWORLD_THREAD_ID: Optional[int] = None
_APPWORLD_THREAD_ID_LOCK = threading.Lock()
# 在首次使用时缓存一份"真实" OS hook，用于 execute 中途异常泄漏后恢复。
_SAFETY_RECOVERY: Any = None


def _run_on_appworld_thread(fn, /, *args, **kwargs):
    """Run ``fn`` on the dedicated AppWorld SQLite thread (reentrant-safe)."""
    global _APPWORLD_THREAD_ID
    with _APPWORLD_THREAD_ID_LOCK:
        dedicated = _APPWORLD_THREAD_ID
    if dedicated is not None and threading.get_ident() == dedicated:
        return fn(*args, **kwargs)

    def _call():
        global _APPWORLD_THREAD_ID
        with _APPWORLD_THREAD_ID_LOCK:
            _APPWORLD_THREAD_ID = threading.get_ident()
        return fn(*args, **kwargs)

    return _APPWORLD_THREAD_POOL.submit(_call).result()


def _restore_appworld_os_hooks() -> None:
    """Best-effort：若上次 execute 未 disable SafetyGuard，恢复 putenv/mkdir 等。"""
    global _SAFETY_RECOVERY
    try:
        from appworld.common.safety_guard import SafetyGuard
    except ImportError:
        return
    if _SAFETY_RECOVERY is None:
        _SAFETY_RECOVERY = SafetyGuard()
    try:
        _SAFETY_RECOVERY.disable()
    except Exception:  # noqa: BLE001
        pass


@dataclasses.dataclass
class _AppWorldTraceRecord:
    messages: List[Dict[str, str]]
    response: str
    model: Optional[str] = None
    ts_start: float = 0.0
    ts_end: float = 0.0


_DEFAULT_SYSTEM_PROMPT = """You are an autonomous agent operating inside AppWorld, a simulated world of \
everyday apps (Amazon, Spotify, Venmo, Gmail, Todoist, SimpleNote, Splitwise, FileSystem, Phone).

You interact with AppWorld by writing and executing Python code. Inside code you can call
`apis.<app_name>.<api_name>(...)`. You do not know the full API surface in advance — look it up
at runtime, e.g.:
    apis.api_docs.show_app_descriptions()
    apis.api_docs.show_api_descriptions(app_name="spotify")
    apis.api_docs.show_api_doc(app_name="spotify", api_name="show_song")
Most tasks require `apis.supervisor.show_profile()` / `apis.supervisor.show_account_passwords()`
first, then `apis.<app>.login(username=..., password=...)` to get an access_token for that app.
When you believe the task is fully done, you may (for QA-style tasks) call
`apis.supervisor.complete_task(answer=...)`; for action-style tasks just performing the requested
side effects is enough — there is no separate "submit" step beyond that.

You also have a persistent "skill library" shared across all tasks in the current session (but
NOT the underlying app/database state — every task in this session may run against a different
simulated user). Skills you save now may or may not still be useful for a later task in this
session; it is up to you to judge, name, and (re)use them well:
  - skill_create(skill_id: str, content: str): save a reusable Python function as a skill.
      `content` must be valid Python defining exactly one function (e.g. "def foo(x):\\n    ...").
      It is executed immediately, so the function becomes callable in this task right away.
  - skill_modify(skill_id: str, content: str): overwrite an existing skill's source code.
  - skill_delete(skill_id: str): permanently remove a skill. It can no longer be loaded afterwards.
  - skill_load(skill_id: str): load a previously saved skill's source code into this task's
      Python namespace (also returns the source text) so you can call it.

Available tools:
  - execute(code: str): run Python code in the AppWorld sandbox for the CURRENT task and return
      whatever it printed (use `print(...)` to see values — expressions alone are not echoed).
      State persists across multiple `execute` calls within the same task.
  - skill_create(skill_id: str, content: str)
  - skill_modify(skill_id: str, content: str)
  - skill_delete(skill_id: str)
  - skill_load(skill_id: str)
  - finish(): call this once you believe the task is complete, to end this task.

Output format for EVERY step (no exceptions):
    Thought: <one or two sentences of reasoning about what to do next>
    Action: {"name": "<tool_name>", "arguments": {...}}

To end the current task:
    Action: {"name": "finish", "arguments": {}}

Rules:
  - Output exactly one Thought/Action pair per step. Do not output more than one Action.
  - `arguments` must be a valid JSON object matching the tool's parameters (multi-line code goes
    into the JSON string with escaped newlines, e.g. "line1\\nline2").
  - If `execute` raises an error, read the traceback and adapt your code instead of repeating it.
"""

# 格式 few-shot：纠正「把 apis.* 当成工具名」的常见错误；演示必须经 execute 调 API。
# 用玩具场景，不泄露评测 task 答案。
_FORMAT_FEW_SHOT = """
Format examples (toy demo; follow the same Thought/Action style). CRITICAL:
  - Available tool names are ONLY: execute, skill_create, skill_modify, skill_delete, skill_load, finish.
  - NEVER invent a tool named like apis.xxx — always call APIs inside execute(code=...).

Example A — look up APIs via execute, then finish a simple QA task:
User: What is my first name according to the supervisor profile?
Assistant:
Thought: I must call supervisor APIs inside execute, not as a top-level tool name.
Action: {"name": "execute", "arguments": {"code": "print(apis.supervisor.show_profile())"}}
User: Observation: {'first_name': 'Ada', 'last_name': 'Demo', ...}
Assistant:
Thought: Profile shows first name Ada; submit via complete_task then finish.
Action: {"name": "execute", "arguments": {"code": "apis.supervisor.complete_task(answer='Ada')"}}
User: Observation: (no output; use print(...) to see values)
Assistant:
Thought: Answer submitted; end this task.
Action: {"name": "finish", "arguments": {}}

Example B — discover an app API doc, then login (still only via execute):
User: Log into Spotify with my account so later steps can use it.
Assistant:
Thought: Check Spotify login API signature through api_docs inside execute.
Action: {"name": "execute", "arguments": {"code": "print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))"}}
User: Observation: {'parameters': [{'name': 'username', ...}, {'name': 'password', ...}], ...}
Assistant:
Thought: Get passwords from supervisor, then call spotify.login inside execute.
Action: {"name": "execute", "arguments": {"code": "print(apis.supervisor.show_account_passwords())"}}
"""


class AppWorldEnv(BusinessEnv):
    """接入真实 AppWorld 沙箱的 STHARC env：ReAct 循环 + 代码执行 + skill harness 工具。

    Args:
        max_steps: 单个 task 最多允许多少个 ReAct 步（含最终 finish 步）。真实
            AppWorld 任务的合理步数远高于纯文本 QA 数据集（SAGE 论文里 Qwen2.5-32B
            平均 12~16 步），默认给到 30，留够裕度。
        appworld_root: 对应 appworld 包的 ``APPWORLD_ROOT`` 环境变量（指向本地缓存
            的 ``data/`` 目录所在的父目录，即 research/appworld_explore）。为 None
            时不设置，依赖调用方已在进程环境里配好。
        experiment_name_prefix: 传给 ``AppWorld(experiment_name=...)`` 的前缀，
            实际值会再拼接 session_id / frozen / uuid 保证并发安全、live/frozen 隔离。
        max_code_output_chars: execute 观测截断长度（对齐 SAGE 论文 Appendix E 的
            12000 字符截断约定，避免超长打印撑爆上下文）。
        format_few_shot: 是否把格式演示拼进 system prompt（也可用环境变量
            ``STHARC_FORMAT_FEW_SHOT=1``）。
        system_prompt / temperature / max_tokens / request_timeout /
        chat_template_kwargs: 语义同 :class:`GenericBenchmarkEnv`。
    """

    USES_AI2C_HARNESS_PATCH: bool = False

    def __init__(
        self,
        max_steps: int = 30,
        appworld_root: Optional[str] = None,
        experiment_name_prefix: str = "stharc_rl",
        max_code_output_chars: int = 12000,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1536,
        request_timeout: float = 180.0,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        format_few_shot: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(action_parser=GenericReactActionParser(), **kwargs)
        self.max_steps = int(max_steps)
        self.appworld_root = appworld_root
        self.experiment_name_prefix = str(experiment_name_prefix)
        self.max_code_output_chars = int(max_code_output_chars)
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
            logger.info("[AppWorldEnv] format_few_shot=ON (execute/API demos appended to system)")
        else:
            self.system_prompt = base
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.request_timeout = float(request_timeout)
        self._vllm_chat_template_kwargs: Optional[Dict[str, Any]] = chat_template_kwargs

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
        if self.appworld_root:
            desired = str(self.appworld_root)
            # 与 AppWorld 会话共用进程锁：SafetyGuard 启用期间写 os.environ 会触发
            # PermissionError: os.putenv not allowed。值已正确时跳过写入。
            with _APPWORLD_PROCESS_LOCK:
                if os.environ.get("APPWORLD_ROOT") != desired:
                    os.environ["APPWORLD_ROOT"] = desired
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
        ext_params = ext_params or {}
        task_id = str(ext_params.get("appworld_task_id") or "").strip()
        if not task_id:
            return EnvRunResult(
                records=[],
                final_answer="",
                env_info={
                    "status": "error",
                    "error": "AppWorldEnv.run: ext_params 缺少 'appworld_task_id'，"
                    "无法定位真实 AppWorld 任务（检查 gen_appworld.py 产出的 "
                    "task_chain[i].ext_params）。",
                    "query": query,
                },
            )
        try:
            import appworld  # noqa: F401,WPS433
        except ImportError as exc:
            return EnvRunResult(
                records=[],
                final_answer="",
                env_info={
                    "status": "error",
                    "error": f"appworld 包未安装，无法执行真实沙箱：{type(exc).__name__}: {exc}",
                    "query": query,
                    "appworld_task_id": task_id,
                },
            )
        return _run_on_appworld_thread(
            self._run_impl, query, history, ext_params, kwargs
        )

    def _run_impl(
        self,
        query: str,
        history: Optional[List[Dict[str, Any]]],
        ext_params: Dict[str, Any],
        kwargs: Dict[str, Any],
    ) -> EnvRunResult:
        from ..stharc.generic_harness_state import get_sandbox
        from ..stharc.harness_tracker import get_tracker

        session_id = str(ext_params.get("_rl_session_id") or f"_adhoc_{id(self)}")
        frozen = bool(ext_params.get("_rl_frozen", False))
        task_id = str(ext_params.get("appworld_task_id") or "").strip()
        sandbox = get_sandbox(session_id)
        tracker = get_tracker()

        from appworld import AppWorld  # noqa: WPS433 (lazy: 只在真正 rollout 时需要)

        experiment_name = (
            f"{self.experiment_name_prefix}_{'frozen' if frozen else 'live'}_{uuid4().hex[:10]}"
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        for h in history or []:
            if isinstance(h, dict) and h.get("role") in ("user", "assistant") and h.get("content"):
                messages.append({"role": h["role"], "content": str(h["content"])})
        messages.append({"role": "user", "content": query})

        records: List[_AppWorldTraceRecord] = []
        final_answer = ""
        status = "ok"
        err: Optional[str] = None
        num_tool_calls = 0
        appworld_eval: Dict[str, Any] = {}
        trunc_counter: Dict[str, int] = {
            "obs_char_trunc_count": 0,
            "obs_chars_dropped": 0,
        }
        ts_start = time.time()

        world: Any = None
        # 持锁覆盖构造 → execute → evaluate → __exit__。LLM HTTP 也会被串行化，
        # 换取进程内并发安全；跨 Ray worker 进程仍可并行。
        with _APPWORLD_PROCESS_LOCK:
            try:
                _restore_appworld_os_hooks()
                if self.appworld_root:
                    desired = str(self.appworld_root)
                    if os.environ.get("APPWORLD_ROOT") != desired:
                        os.environ["APPWORLD_ROOT"] = desired
                # timeout_seconds=None：禁用 AppWorld 内部基于 SIGALRM 的 timeout。
                # AgentLoop 用 asyncio.to_thread(env.run)，子线程里 signal.signal 会报
                # "signal only works in main thread"，导致几乎所有 execute 失败、reward 全 0。
                # 墙钟超时改由 stharc.per_rollout_timeout_sec 负责。
                world = AppWorld(
                    task_id=task_id,
                    experiment_name=experiment_name,
                    ground_truth_mode="full",
                    timeout_seconds=None,
                )
                world.__enter__()

                for step in range(self.max_steps):
                    step_ts_start = time.time()
                    try:
                        response_text = self._call_llm(list(messages))
                    except Exception as llm_exc:  # noqa: BLE001
                        # 不改 token 预算；仅在 vLLM 仍报 context 超限时软结束，保留 evaluate
                        err_s = f"{type(llm_exc).__name__}: {llm_exc}"
                        if "maximum context length" in err_s.lower() or "input_tokens" in err_s.lower():
                            status = "context_overflow"
                            err = err_s
                            logger.warning(
                                "[AppWorldEnv] context overflow at step=%d task_id=%r",
                                step,
                                task_id,
                            )
                            break
                        raise
                    step_ts_end = time.time()

                    records.append(
                        _AppWorldTraceRecord(
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
                        final_answer = str(
                            (action.get("arguments") or {}).get("answer", "")
                        ).strip()
                        break

                    num_tool_calls += 1
                    obs = _execute_tool(
                        name=name,
                        arguments=action.get("arguments") or {},
                        world=world,
                        sandbox=sandbox,
                        frozen=frozen,
                        tracker=tracker,
                        step=step,
                        max_output_chars=self.max_code_output_chars,
                        trunc_counter=trunc_counter,
                    )
                    messages.append({"role": "user", "content": f"Observation: {obs}"})
                else:
                    status = "max_steps_exceeded"

                # 无论是否显式 finish / 是否达到 max_steps，都按 AppWorld 官方评测惯例
                # 对"当前世界最终状态"打分一次（frozen 场景也要打分：反事实基线依然需要
                # 一个真实 reward，否则 b_j 恒为 0，回退到阶段一，见 core.run_frozen_chain）。
                try:
                    appworld_eval = world.evaluate().to_dict()
                except Exception as eval_exc:  # noqa: BLE001
                    appworld_eval = {
                        "success": False,
                        "error": f"evaluate_failed: {eval_exc}",
                    }
            except Exception as exc:  # noqa: BLE001
                status = "error"
                err = f"{type(exc).__name__}: {exc}"
                logger.exception("[AppWorldEnv] run failed for task_id=%r", task_id)
            finally:
                # close() 会 shutil.rmtree；必须先卸掉 SafetyGuard，否则 PermissionError
                _restore_appworld_os_hooks()
                if world is not None:
                    try:
                        world.__exit__(None, None, None)
                    except Exception as exit_exc:  # noqa: BLE001
                        logger.warning(
                            "[AppWorldEnv] world.__exit__ failed for task_id=%r: %s: %s",
                            task_id,
                            type(exit_exc).__name__,
                            exit_exc,
                        )
                    finally:
                        _restore_appworld_os_hooks()

        elapsed = time.time() - ts_start
        appworld_success = bool(appworld_eval.get("success", False))
        env_info: Dict[str, Any] = {
            "query": query,
            "status": status,
            "elapsed_sec": round(elapsed, 3),
            "num_steps": len(records),
            "num_tool_calls": num_tool_calls,
            "session_id": session_id,
            "frozen": frozen,
            "appworld_task_id": task_id,
            "appworld_success": appworld_success,
            "appworld_eval": appworld_eval,
            "truncation": {
                "obs_char_trunc_count": int(trunc_counter.get("obs_char_trunc_count", 0)),
                "obs_chars_dropped": int(trunc_counter.get("obs_chars_dropped", 0)),
                "context_overflow": status == "context_overflow",
                "max_steps_exceeded": status == "max_steps_exceeded",
                "max_code_output_chars": int(self.max_code_output_chars),
                "max_steps": int(self.max_steps),
            },
        }
        if err:
            env_info["error"] = err

        return EnvRunResult(records=records, final_answer=final_answer, env_info=env_info)

    # ------------------------------------------------------------------
    # vLLM 调用（与 GenericBenchmarkEnv 相同的直连 HTTP 实现，独立维护避免耦合）
    # ------------------------------------------------------------------

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        import requests

        if not self._vllm_base_url:
            raise RuntimeError("AppWorldEnv: vllm_base_url 未设置，请先调用 setup_once")

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
# 工具执行：execute（真实代码沙箱） + 4 个 skill harness 工具
# ---------------------------------------------------------------------------


def _execute_tool(
    name: str,
    arguments: Dict[str, Any],
    world: Any,
    sandbox: Dict[str, Any],
    frozen: bool,
    tracker: Any,
    step: int,
    max_output_chars: int,
    trunc_counter: Optional[Dict[str, int]] = None,
) -> str:
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        return f"错误：未知工具 {name!r}，请从可用工具列表中选择。"
    try:
        return handler(
            arguments,
            world,
            sandbox,
            frozen,
            tracker,
            step,
            max_output_chars,
            trunc_counter,
        )
    except Exception as exc:  # noqa: BLE001
        return f"出错：{type(exc).__name__}: {exc}"


def _truncate(
    text: str,
    max_chars: int,
    trunc_counter: Optional[Dict[str, int]] = None,
) -> str:
    if len(text) <= max_chars:
        return text
    dropped = len(text) - max_chars
    if trunc_counter is not None:
        trunc_counter["obs_char_trunc_count"] = int(
            trunc_counter.get("obs_char_trunc_count", 0)
        ) + 1
        trunc_counter["obs_chars_dropped"] = int(
            trunc_counter.get("obs_chars_dropped", 0)
        ) + int(dropped)
    return text[:max_chars] + f"\n...[observation truncated for display, {len(text)} chars total]"


def _tool_execute(args, world, sandbox, frozen, tracker, step, max_output_chars, trunc_counter=None) -> str:
    code = str(args.get("code", ""))
    if not code.strip():
        return "错误：缺少参数 code"
    # frozen 反事实基线：仍然真实执行代码本身（AppWorld 世界状态就是待评测对象），
    # frozen 只影响 skill harness 的写入是否落盘到 sandbox（见下方 skill_* 工具）。
    output = world.execute(code)
    text = str(output if output is not None else "")
    if not text.strip():
        text = "(no output; use print(...) to see values)"
    return _truncate(text, max_output_chars, trunc_counter)


def _record_write(tracker: Any, target_id: str, sub_action: str, before: str, after: str) -> None:
    if tracker is None:
        return
    tracker.record_write(
        tool_name="skill_manage",
        target_id=target_id,
        args={"action": sub_action},
        before=before or "",
        after=after or "",
    )


def _record_read(tracker: Any, target_id: str) -> None:
    if tracker is None:
        return
    # 无条件记录读事件（即便对象不存在 / 已被删除），与 GenericBenchmarkEnv 的
    # _tool_skill_load 保持一致——只要模型确实尝试加载，就能吃到 STHARC 信度分配
    # 公式；模型凭空写代码不尝试 skill_load，则该 skill_create 拿不到 prospective
    # credit（这是"软依赖"数据的预期行为：credit 来自"读了 + 效率变好"，不是
    # "读了 + 否则不可能做对"）。
    tracker.record_read(tool_name="load_skill", target_id=target_id)


def _tool_skill_create(args, world, sandbox, frozen, tracker, step, max_output_chars, trunc_counter=None) -> str:
    skill_id = str(args.get("skill_id", "")).strip()
    content = str(args.get("content", ""))
    if not skill_id:
        return "错误：缺少参数 skill_id"
    if not content.strip():
        return "错误：缺少参数 content"
    target_id = f"skill::{skill_id}"
    before = sandbox["skills"].get(skill_id, "")
    if not frozen:
        sandbox["skills"][skill_id] = content
    _record_write(tracker, target_id, "create", before, content)
    exec_msg = _exec_skill_into_world(world, content, max_output_chars, trunc_counter)
    return f"已创建技能 {skill_id!r} 并尝试立即执行以使其可用。{exec_msg}"


def _tool_skill_modify(args, world, sandbox, frozen, tracker, step, max_output_chars, trunc_counter=None) -> str:
    skill_id = str(args.get("skill_id", "")).strip()
    content = str(args.get("content", ""))
    if not skill_id:
        return "错误：缺少参数 skill_id"
    target_id = f"skill::{skill_id}"
    before = sandbox["skills"].get(skill_id)
    if before is None:
        return f"错误：技能 {skill_id!r} 不存在，无法 modify，请先用 skill_create。"
    if not frozen:
        sandbox["skills"][skill_id] = content
    _record_write(tracker, target_id, "modify", before, content)
    exec_msg = _exec_skill_into_world(world, content, max_output_chars, trunc_counter)
    return f"已更新技能 {skill_id!r} 并尝试立即执行以使其可用。{exec_msg}"


def _tool_skill_delete(args, world, sandbox, frozen, tracker, step, max_output_chars, trunc_counter=None) -> str:
    skill_id = str(args.get("skill_id", "")).strip()
    if not skill_id:
        return "错误：缺少参数 skill_id"
    target_id = f"skill::{skill_id}"
    before = sandbox["skills"].get(skill_id)
    if before is None:
        return f"技能 {skill_id!r} 不存在，无需删除。"
    if not frozen:
        sandbox["skills"].pop(skill_id, None)
    _record_write(tracker, target_id, "delete", before, "")
    return f"已删除技能 {skill_id!r}，之后将无法再加载（本任务内若已 exec 过其函数定义仍可调用，" \
           "但新任务/新沙箱里不会再出现）。"


def _tool_skill_load(args, world, sandbox, frozen, tracker, step, max_output_chars, trunc_counter=None) -> str:
    skill_id = str(args.get("skill_id", "")).strip()
    if not skill_id:
        return "错误：缺少参数 skill_id"
    target_id = f"skill::{skill_id}"
    _record_read(tracker, target_id)
    content = sandbox["skills"].get(skill_id)
    if content is None:
        return f"NOT_FOUND: 技能 {skill_id!r} 不存在或已被删除。"
    exec_msg = _exec_skill_into_world(world, content, max_output_chars, trunc_counter)
    return f"已加载技能 {skill_id!r} 并尝试在当前任务中执行其定义以使其可用。{exec_msg}\n源码:\n{content}"


def _exec_skill_into_world(
    world: Any,
    content: str,
    max_output_chars: int,
    trunc_counter: Optional[Dict[str, int]] = None,
) -> str:
    """把 skill 源码 exec 进当前任务的沙箱命名空间，让函数立即可调用。

    执行失败不阻断流程（模型可以用 skill_modify 修正），只把报错原样回显。
    """
    try:
        output = world.execute(content)
        text = str(output or "").strip()
        if text:
            return f"（执行输出：{_truncate(text, max_output_chars, trunc_counter)}）"
        return "（已在当前任务沙箱中定义，可直接调用）"
    except Exception as exc:  # noqa: BLE001
        return f"（警告：立即执行该 skill 源码时出错，可能需要 skill_modify 修正：{type(exc).__name__}: {exc}）"


_TOOL_HANDLERS = {
    "execute": _tool_execute,
    "skill_create": _tool_skill_create,
    "skill_modify": _tool_skill_modify,
    "skill_delete": _tool_skill_delete,
    "skill_load": _tool_skill_load,
}
