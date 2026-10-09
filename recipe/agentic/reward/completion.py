"""任务完成判定（ai_to_c GRPO reward 共用）。

transfer-human / transfer-planner 已弃用，不再视为合法完成方式。
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ..trajectory.summary import StepSummary, TrajectorySummary

# 已弃用：调用这些工具不算任务完成
DEPRECATED_TERMINAL_ACTIONS = frozenset({"transfer-human", "transfer-planner"})

# 合法终止动作
VALID_COMPLETION_ACTIONS = frozenset({"finish", "give_link_card"})


def _action_name_from_step(step: StepSummary) -> str:
    if step.action_name:
        return step.action_name.strip()
    pa = step.parsed_action
    if isinstance(pa, dict):
        return str(pa.get("name", "")).strip()
    return ""


def _step_uses_deprecated_terminal(step: StepSummary) -> bool:
    name = _action_name_from_step(step)
    if name in DEPRECATED_TERMINAL_ACTIONS:
        return True
    pa = step.parsed_action
    if not isinstance(pa, dict):
        return False
    if pa.get("parallel") and "calls" in pa:
        for call in pa.get("calls") or []:
            if isinstance(call, dict) and str(call.get("name", "")).strip() in DEPRECATED_TERMINAL_ACTIONS:
                return True
    return False


def uses_deprecated_terminal(summary: TrajectorySummary) -> bool:
    """轨迹中是否调用了已弃用的终止工具。"""
    return any(_step_uses_deprecated_terminal(s) for s in summary.steps)


def has_valid_completion_step(summary: TrajectorySummary) -> bool:
    """是否存在合法的终止步（finish / give_link_card）。"""
    for step in summary.steps:
        if not step.is_final:
            continue
        name = _action_name_from_step(step)
        if name in VALID_COMPLETION_ACTIONS:
            return True
    return False


def is_valid_task_completed(
    summary: TrajectorySummary,
    *,
    min_answer_len: int = 1,
) -> Tuple[bool, Dict[str, Any]]:
    """统一任务完成判定，供 AiToCEfficientReward / AiToCGatedJudgeReward 复用。"""
    info: Dict[str, Any] = {}

    if summary.has_error:
        info["no_error"] = 0
        return False, info
    info["no_error"] = 1

    answer_len = len((summary.final_answer or "").strip())
    ans_ok = answer_len >= min_answer_len
    info["answer_ok"] = int(ans_ok)
    info["answer_len"] = answer_len
    if not ans_ok:
        return False, info

    deprecated = uses_deprecated_terminal(summary)
    info["deprecated_terminal"] = int(deprecated)
    if deprecated:
        return False, info

    valid_final = has_valid_completion_step(summary)
    info["has_valid_final"] = int(valid_final)
    info["has_final_step"] = int(summary.has_final_step)
    if not valid_final:
        return False, info

    return True, info
