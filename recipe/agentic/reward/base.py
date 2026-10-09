"""Reward component abstraction.

设计原则：
- 所有内置 reward 只看 :class:`TrajectorySummary`（业务无关）
- 单一职责：每个组件只回答一件事 → 数值 ∈ [0, 1]（除特殊说明）
- :class:`WeightedReward` 通过 yaml 装配 + 加权 + 输出子项 metrics
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, Dict, Optional

from ..trajectory.summary import TrajectorySummary


@dataclasses.dataclass
class RewardContext:
    """传给 RewardComponent.compute 的上下文。

    Fields:
        summary: 标准化的 TrajectorySummary
        ground_truth: 数据集自带的 ground truth（dict / str / None），格式由数据集定
        extra_info: 数据集自带的额外信息（来自 parquet 行）
    """

    summary: TrajectorySummary
    ground_truth: Optional[Any] = None
    extra_info: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class RewardResult:
    """单个 RewardComponent.compute 的返回值。

    Fields:
        value: 数值，通常 ∈ [0, 1]
        info: 任意辅助信息（debug、子指标），会被聚合到 WeightedReward 的 metrics 里
    """

    value: float = 0.0
    info: Dict[str, Any] = dataclasses.field(default_factory=dict)


class RewardComponent(abc.ABC):
    """单个 reward 组件的协议。

    子类约定：
        - name: 组件名（用于日志与 metrics key 前缀）
        - compute(ctx) -> RewardResult: 同步函数；若组件需要异步外部调用（如 judge LLM），
          需在内部用 requests / 同步 HTTP 实现，**不要**让 compute 变成 async
          （AgentLoop 会通过 to_thread 整体放线程池跑）
    """

    name: str = "reward"

    def __init__(self, name: Optional[str] = None, enabled: bool = True, **kwargs: Any) -> None:
        if name:
            self.name = name
        self.enabled = enabled
        self._kwargs = kwargs

    @abc.abstractmethod
    def compute(self, ctx: RewardContext) -> RewardResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 组合器抽象（实际实现见 weighted.py）
# ---------------------------------------------------------------------------


class RewardCombiner(abc.ABC):
    """多 reward 组件组合协议。"""

    @abc.abstractmethod
    def compute(self, ctx: RewardContext) -> RewardResult:
        """返回 (总分, 各子项 metrics)；子项 metrics 放到 RewardResult.info 里。"""
        raise NotImplementedError