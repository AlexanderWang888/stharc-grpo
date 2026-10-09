"""BusinessEnv & ActionParser abstractions.

业务侧零改的关键：
- env 在 worker 进程内 setup_once 完成 sys.path 注入、bridge install、主脑识别器注册、
  vLLM endpoint 注入；幂等
- env.run 是一个**同步**函数（业务 chat_stream 也是同步生成器），在 AgentLoop 中通过
  ``asyncio.to_thread`` 放到线程池跑，ContextVar 自动拷贝以隔离 trace_scope

替换业务时，新建一个子类即可，trajectory / reward / agent_loop / trainer 全部 0 改。
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# ActionParser：把模型输出文本解析成结构化 action
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ParsedAction:
    """单步模型输出的结构化结果。

    Fields:
        parsed_action: 业务可识别的 action 结构（dict / None）；解析失败时为 None
        parse_ok: 解析是否成功（即模型遵循了输出格式）
        is_final: 是否是终止步（如 finish action / Final Answer）
        raw_text: 原始模型输出文本（便于日志与 reward 复现）
    """

    parsed_action: Optional[Dict[str, Any]] = None
    parse_ok: bool = False
    is_final: bool = False
    raw_text: str = ""


class ActionParser(abc.ABC):
    """业务无关的 action 解析协议。"""

    @abc.abstractmethod
    def parse(self, text: str) -> ParsedAction:
        """把单次 LLM 输出文本解析成 :class:`ParsedAction`。"""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Env run result：env.run 的返回值，一条 trajectory 的原始物料
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EnvRunResult:
    """env.run 返回的一条 trajectory 原始物料。

    Fields:
        records: 主脑的 LLM call trace 列表（List[TraceRecord]）；子 agent 的 trace
                 已经在 env 内被过滤掉（不会进 records）
        final_answer: 业务对外的最终回答（chat_stream 拼接出来的字符串），用于 reward
                      与日志；可能为空（表示业务异常或未给出 final answer）
        env_info: 任意 env 元信息（耗时、错误、step 数等），透传给 reward / metrics
    """

    records: List[Any] = dataclasses.field(default_factory=list)
    final_answer: str = ""
    env_info: Dict[str, Any] = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# BusinessEnv：业务抽象基类
# ---------------------------------------------------------------------------


class BusinessEnv(abc.ABC):
    """业务环境抽象。

    用法约定（与 AgenticAgentLoop 配合）：
        1. AgentLoop 在每条 rollout 第一次用到 env 时 await loop.run_in_executor(None,
           env.setup_once, vllm_base_url, vllm_model_name)；setup_once **必须幂等**
        2. AgentLoop 通过 ``await asyncio.to_thread(env.run, query=..., ...)`` 在线程池
         内跑业务同步生成器；ContextVar 会跨线程拷贝，trace_scope 自动隔离
        3. env 自身不持久化跨 rollout 的可变状态；若业务有 session_id 等会污染状态，
           env.run 内部自行隔离（如不传 session_id 让业务自动 uuid）

    Attributes:
        action_parser: 业务自带的 :class:`ActionParser`，由子类在 __init__ 设置
        USES_AI2C_HARNESS_PATCH: 类属性，默认 True（历史行为，ai2c 系 env 均依赖
            ``ai2c_harness_patch`` 做 monkeypatch + RL context 绑定）。与 ai2c 解耦的
            env（如 ``GenericBenchmarkEnv``）应在子类上覆盖为 False，让
            ``stharc.core.rollout_with_tracking`` 跳过 ai2c 专属的 patch/bind 步骤
            （HarnessTracker 拦截本身仍生效，由 env 自己在工具执行处调用
            ``get_tracker().record_write/record_read``）。
    """

    action_parser: ActionParser
    USES_AI2C_HARNESS_PATCH: bool = True

    def __init__(self, action_parser: ActionParser, **kwargs: Any) -> None:
        self.action_parser = action_parser
        # 子类可在 kwargs 里读各种业务专属配置（system prompt patterns 等）
        self._init_kwargs = kwargs

    # ------------------------------------------------------------------
    # 必须由子类实现
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def setup_once(
        self,
        vllm_base_url: Optional[str] = None,
        vllm_model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """worker 进程启动后调一次。**必须幂等**。

        典型操作：
            - sys.path 注入业务包根
            - install_trace_bridge（agentic_train_offline.llm_bridge）
            - 注册主脑识别器（system prompt 前缀）
            - 注册 vLLM endpoint 转发（主脑请求改打训练 actor 的 vLLM）
            - 注册业务模块的 importer（如有动态 importer）
            - 业务侧的轻量准备（如配置 SESSION_DIR=/tmp/...）
        """
        raise NotImplementedError

    @abc.abstractmethod
    def run(
        self,
        query: str,
        history: Optional[List[Dict[str, Any]]] = None,
        ext_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> EnvRunResult:
        """同步执行一次业务对话（= 一条 trajectory）。

        约束：
            - 必须**同步**（业务 chat_stream 通常是 sync generator）
            - 内部用 trace_scope 捕获主脑 LLM call → 写入 EnvRunResult.records
            - 子 agent 的 trace 不能进 records（由 bridge wrapper 标记 role 后过滤）
            - 异常不抛出，写入 env_info["error"] 即可（reward 侧给 0 分）
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 可选：默认实现已够用，子类一般不必重写
    # ------------------------------------------------------------------

    def teardown(self) -> None:
        """worker 退出前的清理（默认 no-op）。"""
        return None