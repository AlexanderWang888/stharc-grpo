"""WeightedReward：加权组合多个 RewardComponent。

设计：
- 从 yaml 装配：components 列表，每个组件含 `_target_`、`weight`、`enabled` 和组件自身 kwargs
- 总分 = Σ (component.value * weight)  /  Σ weight   ← 归一化到 [0, 1] 区间附近
- 同时输出每个组件的 raw value + sub_metrics 到 info 里，便于 wandb 监控 reward hacking

yaml 示例（rl_plan_final.md §5.3）：

    reward:
      _target_: recipe.agentic.reward.WeightedReward
      components:
        - _target_: recipe.agentic.reward.components.TaskCompletedReward
          weight: 1.0
          enabled: true
        - _target_: recipe.agentic.reward.components.ToolCallSuccessReward
          weight: 0.3
        - _target_: recipe.agentic.reward.components.JudgeReward
          weight: 0.5
          enabled: false
          judge_url: "http://judge-llm:8000/v1/chat/completions"
        - _target_: recipe.agentic.reward.components.TeacherSimilarityReward
          weight: 0.2
          enabled: false
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from .base import RewardCombiner, RewardComponent, RewardContext, RewardResult

logger = logging.getLogger(__name__)


class WeightedReward(RewardCombiner):
    """加权 reward 组合器。

    Args:
        components: 已实例化的 :class:`RewardComponent` 列表（由 hydra/yaml 实例化后传入）
        weights: 与 components 同长度的权重列表；None 时全部取 1.0
        normalize: True 时返回 Σ(value*weight)/Σ(weight)；False 时直接返回 Σ(value*weight)
        clip_min / clip_max: 总分裁剪范围（None 不裁剪）
        log_prefix: metrics key 的统一前缀（默认 "reward/"）
    """

    def __init__(
        self,
        components: Sequence[RewardComponent],
        weights: Optional[Sequence[float]] = None,
        normalize: bool = True,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
        log_prefix: str = "reward/",
    ) -> None:
        self.components: List[RewardComponent] = list(components)
        if weights is None:
            self.weights: List[float] = [1.0] * len(self.components)
        else:
            if len(weights) != len(self.components):
                raise ValueError(
                    f"weights length {len(weights)} != components length {len(self.components)}"
                )
            self.weights = list(weights)
        self.normalize = normalize
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.log_prefix = log_prefix

    # ------------------------------------------------------------------

    def compute(self, ctx: RewardContext) -> RewardResult:
        import math

        weighted_sum = 0.0
        weight_total = 0.0
        info: Dict[str, Any] = {}
        has_nan = False

        for comp, w in zip(self.components, self.weights):
            if not getattr(comp, "enabled", True):
                continue
            try:
                res = comp.compute(ctx)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[reward] component %s raised: %s", comp.name, exc)
                res = RewardResult(value=0.0, info={"error": f"{type(exc).__name__}: {exc}"})

            v = float(res.value)
            if math.isnan(v):
                has_nan = True
            else:
                weighted_sum += v * w
                weight_total += abs(w)  # 兼容负权重场景

            # 落到 info
            info[f"{self.log_prefix}{comp.name}"] = v
            for k, sub_v in (res.info or {}).items():
                info[f"{self.log_prefix}{comp.name}/{k}"] = sub_v

        # 任一组件 NaN（如 pairwise 缺 baseline）→ 总分保持 NaN，供 GRPO mask。
        # 注意：Python 的 max(0.0, nan)==0.0，绝不能对 NaN 做 clip。
        if has_nan:
            total = float("nan")
        elif self.normalize and weight_total > 0:
            total = weighted_sum / weight_total
        else:
            total = weighted_sum

        if not math.isnan(total):
            if self.clip_min is not None:
                total = max(self.clip_min, total)
            if self.clip_max is not None:
                total = min(self.clip_max, total)

        info[f"{self.log_prefix}total"] = total
        info[f"{self.log_prefix}weight_total"] = weight_total
        return RewardResult(value=float(total), info=info)


# ---------------------------------------------------------------------------
# yaml-friendly 装配辅助（与 hydra.utils.instantiate 互补）
# ---------------------------------------------------------------------------


def build_weighted_reward_from_config(cfg: Any) -> WeightedReward:
    """从 OmegaConf / dict 构造 WeightedReward，避免硬依赖 hydra.utils.instantiate。

    cfg 形如：
        {
          "components": [
            {"_target_": "...", "weight": 1.0, "enabled": true, "...": ...},
            ...
          ],
          "normalize": true,
        }
    """
    import importlib

    components: List[RewardComponent] = []
    weights: List[float] = []
    for item in cfg.get("components", []):
        item = dict(item)  # 拷贝
        target = item.pop("_target_")
        weight = float(item.pop("weight", 1.0))
        module_path, _, cls_name = target.rpartition(".")
        mod = importlib.import_module(module_path)
        cls = getattr(mod, cls_name)
        components.append(cls(**item))
        weights.append(weight)

    return WeightedReward(
        components=components,
        weights=weights,
        normalize=bool(cfg.get("normalize", True)),
        clip_min=cfg.get("clip_min"),
        clip_max=cfg.get("clip_max"),
        log_prefix=cfg.get("log_prefix", "reward/"),
    )