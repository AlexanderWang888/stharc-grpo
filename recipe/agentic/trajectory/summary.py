"""TrajectorySummary / StepSummary —— reward 侧只认这个结构。

约束：
- 只描述「主脑这条 trajectory 干了啥」，业务无关
- TrajectoryBuilder 从 List[TraceRecord] + final_answer + env_info 构造出本结构
- Reward 组件只读本结构（+ teacher_trace 可选），不需要懂业务
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional


@dataclasses.dataclass
class StepSummary:
    """单次主脑 LLM call 的结构化摘要。

    Fields:
        step_idx: 第几步（从 0 开始）
        messages: 该次 call 的完整 messages（List[{"role","content"}]）
        response_text: 模型输出的原始文本
        parsed_action: ActionParser 解析后的结构（dict / None）
        parse_ok: 解析是否成功（模型是否遵循输出格式）
        is_final: 是否是终止步（finish / give_link_card action）
        action_name: 当 parse_ok 时的 action 名字（如 "search_general_kb" / "finish"）
        model: 该 call 使用的 model 名字（可选，便于排查）
        ts_start / ts_end: 调用时间戳（秒）
    """

    step_idx: int
    messages: List[Dict[str, str]] = dataclasses.field(default_factory=list)
    response_text: str = ""
    parsed_action: Optional[Dict[str, Any]] = None
    parse_ok: bool = False
    is_final: bool = False
    action_name: str = ""
    model: Optional[str] = None
    ts_start: float = 0.0
    ts_end: float = 0.0
    # 工具执行后的观测中是否包含错误（如 "错误：未知工具"、"错误：需要 skuid 与 query"、
    # parse 失败兜底文案等），parse_ok=True 但语义上执行失败的情况均在此记录。
    obs_error: bool = False
    # 工具服务端/执行层自身失败（下游服务异常、网络错误、工具 bug 等），非模型问题。
    tool_failure: bool = False
    # 无效步：工具调用执行成功但产出零有效信息（模型可控）。
    # 覆盖：resolve_products 真空跑、search 无结果、obs_error 类参数/前置校验错误。
    # 不含 tool_failure（服务端失败，非模型问题）。
    wasted_step: bool = False


@dataclasses.dataclass
class TrajectorySummary:
    """一条完整 trajectory 的结构化摘要（业务无关）。

    Fields:
        query: 用户原始 query
        steps: 每一步 :class:`StepSummary`
        final_answer: 业务对外的最终回答字符串（chat_stream 拼接出来）
        env_info: env.run 透传过来的 env 元信息（含可能的 error）

        # 训练相关（builder 同时产出，但仍归在 summary 里方便 reward 复用）
        prompt_text: 给 tokenizer 的初始 prompt 文本（第 0 步的 messages 拼接）
        response_text: 给 tokenizer 的完整 response 文本（assistant 段 + observation 段拼接）

        # 可选：teacher trajectory（用于 TeacherSimilarityReward）
        teacher_trace: 教师轨迹的关键文本（如多个 response_text 拼接），未提供时为 None
    """

    query: str = ""
    steps: List[StepSummary] = dataclasses.field(default_factory=list)
    final_answer: str = ""
    env_info: Dict[str, Any] = dataclasses.field(default_factory=dict)

    prompt_text: str = ""
    response_text: str = ""

    teacher_trace: Optional[str] = None

    # ------------------------------------------------------------------
    # 便利方法
    # ------------------------------------------------------------------

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def num_action_steps(self) -> int:
        """非 final 的步数。"""
        return sum(1 for s in self.steps if not s.is_final)

    @property
    def num_parse_ok(self) -> int:
        return sum(1 for s in self.steps if s.parse_ok)

    @property
    def has_error(self) -> bool:
        return bool(self.env_info.get("error"))

    @property
    def has_final_step(self) -> bool:
        return any(s.is_final for s in self.steps)

    def tool_call_success_rate(self) -> float:
        """非 final 的步里 parse_ok 比例（无非 final 步时返回 1.0）。"""
        action_steps = [s for s in self.steps if not s.is_final]
        if not action_steps:
            return 1.0
        return sum(1 for s in action_steps if s.parse_ok) / len(action_steps)

    @property
    def num_obs_errors(self) -> int:
        """工具执行后观测包含错误信息的非 final 步数。"""
        return sum(1 for s in self.steps if not s.is_final and s.obs_error)

    def obs_error_rate(self) -> float:
        """工具执行观测错误率（非 final 步中 obs_error 比例，无非 final 步时返回 0.0）。"""
        action_steps = [s for s in self.steps if not s.is_final]
        if not action_steps:
            return 0.0
        return sum(1 for s in action_steps if s.obs_error) / len(action_steps)

    @property
    def num_tool_failures(self) -> int:
        """工具服务端/执行层自身失败的非 final 步数（下游服务异常、网络错误等，非模型问题）。"""
        return sum(1 for s in self.steps if not s.is_final and s.tool_failure)

    def tool_failure_rate(self) -> float:
        """工具服务端失败率（非 final 步中 tool_failure 比例，无非 final 步时返回 0.0）。"""
        action_steps = [s for s in self.steps if not s.is_final]
        if not action_steps:
            return 0.0
        return sum(1 for s in action_steps if s.tool_failure) / len(action_steps)

    @property
    def num_wasted_steps(self) -> int:
        """无效步数（模型可控：resolve 真空跑、search 无结果、obs_error 类错误）。"""
        return sum(1 for s in self.steps if not s.is_final and s.wasted_step)

    def wasted_step_rate(self) -> float:
        """无效步率（非 final 步中 wasted_step 比例，无非 final 步时返回 0.0）。"""
        action_steps = [s for s in self.steps if not s.is_final]
        if not action_steps:
            return 0.0
        return sum(1 for s in action_steps if s.wasted_step) / len(action_steps)

    # ------------------------------------------------------------------
    # 序列化（便于落盘 debug / wandb artifact）
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "final_answer": self.final_answer,
            "env_info": dict(self.env_info),
            "num_steps": self.num_steps,
            "num_action_steps": self.num_action_steps,
            "num_parse_ok": self.num_parse_ok,
            "num_obs_errors": self.num_obs_errors,
            "num_tool_failures": self.num_tool_failures,
            "num_wasted_steps": self.num_wasted_steps,
            "has_error": self.has_error,
            "has_final_step": self.has_final_step,
            "steps": [
                {
                    "step_idx": s.step_idx,
                    "response_text": s.response_text,
                    "parsed_action": s.parsed_action,
                    "parse_ok": s.parse_ok,
                    "is_final": s.is_final,
                    "action_name": s.action_name,
                    "obs_error": s.obs_error,
                    "tool_failure": s.tool_failure,
                    "wasted_step": s.wasted_step,
                    "model": s.model,
                }
                for s in self.steps
            ],
        }