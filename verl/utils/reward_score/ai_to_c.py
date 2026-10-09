"""verl reward_score 适配层：把 WeightedReward 包装成 verl 标准 compute_score。

verl 的 reward_score/__init__.py 在 GRPO trainer 里调用：
    res = default_compute_score(data_source, solution_str, ground_truth, extra_info)

ai_to_c 场景下 reward 在 AgentLoop 内部已经算好了（因为需要 TrajectorySummary），
AgentLoopOutput.reward_score 已包含最终加权分数。

但 verl 的 default_compute_score 仍会被调用（作为 fallback 或日志验证），
所以这里从 extra_info 取已经算好的 reward 值返回，不重复计算。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> float:
    """ai_to_c reward：从 extra_info 取 AgentLoop 已算好的分数。

    如果 extra_info 中没有 agentic_metrics（非 agentic 训练路径），退化为简单非空检查。
    """
    if extra_info is None:
        extra_info = {}

    # 优先取 AgentLoop 内已算好的分数
    agentic_metrics = extra_info.get("agentic_metrics", {})
    if isinstance(agentic_metrics, dict) and "reward/_total" in agentic_metrics:
        return float(agentic_metrics["reward/_total"])

    # 也可能在 extra_fields.agentic_metrics 里
    extra_fields = extra_info.get("extra_fields", {})
    if isinstance(extra_fields, dict):
        am = extra_fields.get("agentic_metrics", {})
        if isinstance(am, dict) and "reward/_total" in am:
            return float(am["reward/_total"])

    # fallback：简单非空给 1.0，空给 0.0
    if solution_str and solution_str.strip():
        return 1.0
    return 0.0