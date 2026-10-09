"""Reward components & weighted combiner.

Hydra 通过完整 ``_target_`` 路径实例化组件（如
``recipe.agentic.reward.components.AiToCCombinedReward``），不依赖本包
``__init__`` 的全量 re-export。这里只做轻量导出，避免旧版/半同步
``components.py`` 导致 ``import recipe.agentic.reward`` 直接炸掉。
"""

from .appworld_reward import AppWorldEfficientReward, AppWorldVerifierReward  # noqa: F401
from .base import RewardComponent, RewardContext, RewardResult  # noqa: F401
from .exact_match import ExactMatchReward  # noqa: F401
from .weighted import WeightedReward, build_weighted_reward_from_config  # noqa: F401

# 基础组件（旧版 components 也有）
from .components import (  # noqa: F401
    AiToCEfficientReward,
    JudgeReward,
    StepCountPenaltyReward,
    TaskCompletedReward,
    TeacherSimilarityReward,
    ToolCallSuccessReward,
)

# pairwise / combined（需完整同步后的 components.py）；缺失时不阻塞包导入
try:
    from .components import (  # noqa: F401
        AccuracyTrajectoryJudgeReward,
        AiToCCombinedReward,
        AiToCGatedJudgeReward,
        AiToCGatedPairwiseReward,
        GsbAbsoluteJudgeReward,
        GsbPairwiseJudgeReward,
        WastedStepPenaltyReward,
    )
except ImportError:
    AccuracyTrajectoryJudgeReward = None  # type: ignore[misc, assignment]
    AiToCCombinedReward = None  # type: ignore[misc, assignment]
    AiToCGatedJudgeReward = None  # type: ignore[misc, assignment]
    AiToCGatedPairwiseReward = None  # type: ignore[misc, assignment]
    GsbAbsoluteJudgeReward = None  # type: ignore[misc, assignment]
    GsbPairwiseJudgeReward = None  # type: ignore[misc, assignment]
    WastedStepPenaltyReward = None  # type: ignore[misc, assignment]

__all__ = [
    "AccuracyTrajectoryJudgeReward",
    "AiToCCombinedReward",
    "AiToCEfficientReward",
    "AiToCGatedJudgeReward",
    "AiToCGatedPairwiseReward",
    "AppWorldEfficientReward",
    "AppWorldVerifierReward",
    "ExactMatchReward",
    "GsbAbsoluteJudgeReward",
    "GsbPairwiseJudgeReward",
    "JudgeReward",
    "RewardComponent",
    "RewardContext",
    "RewardResult",
    "StepCountPenaltyReward",
    "TaskCompletedReward",
    "TeacherSimilarityReward",
    "ToolCallSuccessReward",
    "WastedStepPenaltyReward",
    "WeightedReward",
    "build_weighted_reward_from_config",
]
