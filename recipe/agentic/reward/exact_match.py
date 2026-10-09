"""ExactMatchReward —— 业务无关的精确匹配 reward（供 GenericBenchmarkEnv 公开数据集使用）。

设计依据：research/dataset_explore/IDEA2_IMPLEMENTATION_PLAN.md §5.3。

判定逻辑：
    1. 数值题（ground_truth 可解析为 float）：从 final_answer 中抽取首个数字，
       按 ``numeric_tolerance`` 做浮点近似比较（textcraft 的比例/数量类答案用这个）。
    2. 文本题：归一化（大小写/首尾空白/标点）后做精确匹配；
       ``allow_substring=True``（默认）时额外允许"答案中包含归一化后的 ground_truth
       且总长度不超过 gt 长度 + slack_chars"，用于容忍模型在简短事实前后加了
       无关修饰语（如 "The answer is 14622." vs "14622"），同时避免过度宽松导致
       reward hacking（不允许无限堆答案硬凑）。

无 ground_truth 或空 final_answer 时返回 0.0（不是 NaN——公开数据集里 ground_truth
理论上总是存在，缺失视为数据异常，不应该 mask 掉整条 chain）。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .base import RewardComponent, RewardContext, RewardResult

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")


def _normalize(text: str, case_sensitive: bool = False, strip_punctuation: bool = True) -> str:
    s = text.strip()
    if not case_sensitive:
        s = s.lower()
    if strip_punctuation:
        s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _try_float(text: str) -> Optional[float]:
    """把字符串解析为 float，支持简单分数形式 "a/b"。"""
    s = text.strip()
    if not s:
        return None
    m = _NUM_RE.fullmatch(s)
    if not m:
        return None
    if "/" in s:
        num, _, den = s.partition("/")
        try:
            denom = float(den)
            if denom == 0:
                return None
            return float(num) / denom
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _extract_first_number(text: str) -> Optional[float]:
    m = _NUM_RE.search(text)
    if not m:
        return None
    return _try_float(m.group(0))


class ExactMatchReward(RewardComponent):
    """精确匹配 reward：文本归一化匹配 + 数值近似匹配。0/1 二值。"""

    name = "exact_match"

    def __init__(
        self,
        name: Optional[str] = None,
        enabled: bool = True,
        case_sensitive: bool = False,
        strip_punctuation: bool = True,
        numeric_tolerance: float = 1e-3,
        allow_substring: bool = True,
        substring_slack_chars: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, enabled=enabled, **kwargs)
        self.case_sensitive = bool(case_sensitive)
        self.strip_punctuation = bool(strip_punctuation)
        self.numeric_tolerance = float(numeric_tolerance)
        self.allow_substring = bool(allow_substring)
        self.substring_slack_chars = int(substring_slack_chars)

    def compute(self, ctx: RewardContext) -> RewardResult:
        gt = ctx.ground_truth
        if gt is None and isinstance(ctx.extra_info, dict):
            gt = ctx.extra_info.get("ground_truth")
        if gt is None or str(gt).strip() == "":
            return RewardResult(value=0.0, info={"skipped": "no_ground_truth"})

        pred = (ctx.summary.final_answer or "").strip()
        if not pred:
            return RewardResult(value=0.0, info={"skipped": "empty_answer", "ground_truth": str(gt)})

        gt_str = str(gt).strip()

        # 1) 数值近似匹配（ground_truth 本身是纯数字/分数时才走这条路径）
        gt_num = _try_float(gt_str)
        if gt_num is not None:
            pred_num = _extract_first_number(pred)
            if pred_num is None:
                return RewardResult(
                    value=0.0,
                    info={"mode": "numeric", "ground_truth": gt_str, "pred_num": None},
                )
            match = abs(pred_num - gt_num) <= self.numeric_tolerance * max(1.0, abs(gt_num))
            return RewardResult(
                value=1.0 if match else 0.0,
                info={
                    "mode": "numeric",
                    "ground_truth": gt_num,
                    "pred_num": pred_num,
                    "abs_diff": abs(pred_num - gt_num),
                },
            )

        # 2) 文本归一化匹配
        gt_norm = _normalize(gt_str, self.case_sensitive, self.strip_punctuation)
        pred_norm = _normalize(pred, self.case_sensitive, self.strip_punctuation)

        exact = gt_norm == pred_norm
        substring = False
        if not exact and self.allow_substring and gt_norm:
            substring = (
                gt_norm in pred_norm
                and len(pred_norm) <= len(gt_norm) + self.substring_slack_chars
            )
        match = exact or substring

        info: Dict[str, Any] = {
            "mode": "text",
            "ground_truth": gt_str,
            "gt_norm": gt_norm,
            "pred_norm": pred_norm,
            "exact": int(exact),
            "substring_match": int(substring),
        }
        return RewardResult(value=1.0 if match else 0.0, info=info)
