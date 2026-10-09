"""AppWorld 专用 reward 组件——正确性来自真实 ``world.evaluate()``，不做任何字符串匹配。

设计依据：与用户在讨论中确认的结论——
    1. AppWorld/SAGE 的 scenario 结构（同 scenario 内 3 个 task 用不同 simulated
       user/DB，见 AppWorld 论文 Appendix C + 本地 appworld_explore 实测数据）
       保证 skill 复用永远是"软依赖"（只影响效率/鲁棒性，不影响正确性可解性）。
    2. 因此单用正确性信号，跨 task 的 ``delta_r = r_j - b_j`` 在 skill 有效复用时
       会恒等于 0（两条 chain 都能独立做对）；需要把"轮次/步数效率"也编码进
       reward，才能让 delta_r 反映"有没有真的用 skill 省了力气"。
    3. 具体做法复用 ``components.AiToCEfficientReward`` 已验证过的乘法结构
       （reward = correctness_gate × step_efficiency，两端 clip 到 [0,1]），但
       正确性门控换成 ``env_info["appworld_success"]``（真实 unit-test 判分），
       不能沿用 ``is_valid_task_completed``（要求非空文本答案——AppWorld 里大量
       "纯动作类"任务本来就没有文本答案，会被误判为未完成）。

两个组件：
    - :class:`AppWorldVerifierReward`：纯正确性（不含步数效率），用于只想看信度
      分配是否至少对硬信号（正确性本身）work 的诊断场景。
    - :class:`AppWorldEfficientReward`：正确性 × 步数效率（乘法结构），是
      ``stharc_grpo_appworld.yaml`` 里实际配置使用的组件——乘法结构保证"为省一步
      去赌一个错答案"永远不划算（答案错 → 整体直接归零，跟省了几步无关）。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import RewardComponent, RewardContext, RewardResult


def _appworld_eval_info(ctx: RewardContext) -> tuple:
    """从 env_info 里取出 (has_env_error, success, appworld_eval_dict)。"""
    env_info: Dict[str, Any] = ctx.summary.env_info or {}
    if env_info.get("error"):
        return True, False, {}
    appworld_eval = env_info.get("appworld_eval") or {}
    success = bool(env_info.get("appworld_success", appworld_eval.get("success", False)))
    return False, success, appworld_eval


def _partial_credit_fraction(appworld_eval: Dict[str, Any], success: bool) -> float:
    """passes / (passes+failures) 的软化正确性分数（partial_credit=True 时使用）。

    AppWorld 的 evaluate() 本身是若干条 unit-test 断言的集合（见
    ground_truth/evaluation.py），success 是"全部断言通过"，但 passes/failures
    的条数比例可以给训练早期一个更平滑的梯度（而不是纯 0/1）。
    """
    passes = appworld_eval.get("passes") or []
    failures = appworld_eval.get("failures") or []
    total = len(passes) + len(failures)
    if total <= 0:
        return 1.0 if success else 0.0
    return len(passes) / total


class AppWorldVerifierReward(RewardComponent):
    """AppWorld 纯正确性 reward：直接读 ``AppWorldEnv`` 写入的 ``appworld_success``。

    Args:
        partial_credit: True 时用 passes/(passes+failures) 做软化分数；
            False（默认）时严格 0/1（对齐 SAGE 论文的 TGC 判定口径）。
    """

    name = "appworld_verifier"

    def __init__(
        self,
        name: Optional[str] = None,
        enabled: bool = True,
        partial_credit: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, enabled=enabled, **kwargs)
        if isinstance(partial_credit, bool):
            self.partial_credit = partial_credit
        else:
            self.partial_credit = str(partial_credit).strip().lower() in ("1", "true", "yes", "on")

    def compute(self, ctx: RewardContext) -> RewardResult:
        has_error, success, appworld_eval = _appworld_eval_info(ctx)
        if has_error:
            return RewardResult(value=0.0, info={"skipped": "env_error"})

        value = _partial_credit_fraction(appworld_eval, success) if self.partial_credit else float(success)
        return RewardResult(
            value=value,
            info={
                "appworld_success": int(success),
                "num_tests": appworld_eval.get("num_tests"),
                "num_passes": len(appworld_eval.get("passes") or []),
                "num_failures": len(appworld_eval.get("failures") or []),
            },
        )


class AppWorldEfficientReward(RewardComponent):
    """AppWorld 专用：正确性门控 × 步数效率（乘法结构）。

    公式：
        reward = 0                                             若未通过 evaluate()
        reward = clip(1 - (n - target_steps) / (max_steps - target_steps), 0, 1)   否则

    其中 ``n`` 默认取 ``env_info["num_tool_calls"]``（AppWorldEnv 记录的真实工具调用
    步数：execute + skill_* 之和，不含 finish/格式重试步），可通过 ``step_metric``
    切换成 ``num_steps``（含 finish/重试）。

    Args:
        target_steps: 不超过该步数时步数效率给满分。AppWorld 真实任务的自然步数
            量级远高于纯文本 QA 数据集（SAGE 论文 Qwen2.5-32B 平均 12~16 步），
            默认给 8（比论文报告的"高效模型"均值略低，留出"确实更高效"的上升空间）。
        max_steps: 达到该步数时步数效率给 0 分（不代表任务失败——正确性门控是
            独立判据；两者都满足才能拿到接近 1.0 的总分）。应 <= env 的 max_steps
            （env 里跑不到这么多步就会被截断），否则效率项永远给不出 0 分以外的中间值。
        use_step_score: False 时效率项固定 1.0（等价于纯正确性 reward，但仍保留
            乘法结构 + env_info 兜底，方便做 ablation 对比）。
        partial_credit: 同 :class:`AppWorldVerifierReward`。
        step_metric: "num_tool_calls"（默认）或 "num_steps"。
    """

    name = "appworld_efficient"

    def __init__(
        self,
        name: Optional[str] = None,
        enabled: bool = True,
        target_steps: int = 8,
        max_steps: int = 30,
        use_step_score: bool = True,
        partial_credit: bool = False,
        step_metric: str = "num_tool_calls",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, enabled=enabled, **kwargs)
        if target_steps >= max_steps:
            raise ValueError(f"target_steps({target_steps}) must be < max_steps({max_steps})")
        self.target_steps = int(target_steps)
        self.max_steps = int(max_steps)
        # 勿用 bool("false")——在 Python 里为 True（hydra/env 常传字符串）
        if isinstance(use_step_score, bool):
            self.use_step_score = use_step_score
        else:
            self.use_step_score = str(use_step_score).strip().lower() in ("1", "true", "yes", "on")
        if isinstance(partial_credit, bool):
            self.partial_credit = partial_credit
        else:
            self.partial_credit = str(partial_credit).strip().lower() in ("1", "true", "yes", "on")
        self.step_metric = str(step_metric)

    def compute(self, ctx: RewardContext) -> RewardResult:
        has_error, success, appworld_eval = _appworld_eval_info(ctx)
        env_info: Dict[str, Any] = ctx.summary.env_info or {}

        if has_error:
            return RewardResult(
                value=0.0,
                info={"skipped": "env_error", "appworld_success": 0, "step_score": 0.0},
            )

        correctness = _partial_credit_fraction(appworld_eval, success) if self.partial_credit else float(success)

        if correctness <= 0.0:
            # 乘法结构：答案/世界状态不对，效率再高也是 0 分——避免"为省一步去赌一个
            # 错答案"划算，不需要额外调 λ 来防这个（讨论见模块 docstring）。
            return RewardResult(
                value=0.0,
                info={
                    "appworld_success": int(success),
                    "correctness": correctness,
                    "step_score": 0.0,
                    "num_tool_calls": env_info.get("num_tool_calls"),
                    "num_steps": env_info.get("num_steps"),
                },
            )

        n = int(env_info.get(self.step_metric, 0) or 0)
        if not self.use_step_score:
            step_score = 1.0
        elif n <= self.target_steps:
            step_score = 1.0
        elif n >= self.max_steps:
            step_score = 0.0
        else:
            step_score = 1.0 - (n - self.target_steps) / (self.max_steps - self.target_steps)

        total = correctness * step_score
        return RewardResult(
            value=total,
            info={
                "appworld_success": int(success),
                "correctness": correctness,
                "step_score": step_score,
                "num_tool_calls": env_info.get("num_tool_calls"),
                "num_steps": env_info.get("num_steps"),
                "step_metric": self.step_metric,
                "target_steps": self.target_steps,
                "max_steps": self.max_steps,
            },
        )
