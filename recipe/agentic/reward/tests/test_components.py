"""4 个 reward 组件的单测 + WeightedReward 装配单测。"""

from __future__ import annotations

import unittest
from typing import List
from unittest.mock import patch

from recipe.agentic.reward import (
    JudgeReward,
    RewardContext,
    TaskCompletedReward,
    TeacherSimilarityReward,
    ToolCallSuccessReward,
    WeightedReward,
    build_weighted_reward_from_config,
)
from recipe.agentic.reward.components import _parse_judge_score
from recipe.agentic.trajectory.summary import StepSummary, TrajectorySummary


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_summary(
    steps_spec: List[tuple],  # list of (parse_ok, is_final)
    final_answer: str = "ok",
    error: str = "",
    response_text: str = "",
    teacher_trace: str = "",
    query: str = "Q",
) -> TrajectorySummary:
    steps = [
        StepSummary(
            step_idx=i,
            messages=[],
            response_text=f"r{i}",
            parsed_action={"name": "x"} if parse_ok else None,
            parse_ok=parse_ok,
            is_final=is_final,
            action_name="finish" if is_final else "x",
        )
        for i, (parse_ok, is_final) in enumerate(steps_spec)
    ]
    return TrajectorySummary(
        query=query,
        steps=steps,
        final_answer=final_answer,
        env_info={"error": error} if error else {},
        prompt_text="P",
        response_text=response_text,
        teacher_trace=teacher_trace or None,
    )


# ---------------------------------------------------------------------------
# TaskCompletedReward
# ---------------------------------------------------------------------------


class TestTaskCompletedReward(unittest.TestCase):
    def test_all_satisfied(self):
        r = TaskCompletedReward()
        s = _make_summary([(True, False), (True, True)], final_answer="ok")
        res = r.compute(RewardContext(summary=s))
        self.assertEqual(res.value, 1.0)
        self.assertEqual(res.info["answer_ok"], 1)
        self.assertEqual(res.info["no_error"], 1)
        self.assertEqual(res.info["has_final_step"], 1)

    def test_empty_answer(self):
        r = TaskCompletedReward()
        s = _make_summary([(True, True)], final_answer="")
        res = r.compute(RewardContext(summary=s))
        self.assertEqual(res.value, 0.0)
        self.assertEqual(res.info["answer_ok"], 0)

    def test_with_error(self):
        r = TaskCompletedReward()
        s = _make_summary([(True, True)], error="net timeout")
        res = r.compute(RewardContext(summary=s))
        self.assertEqual(res.value, 0.0)
        self.assertEqual(res.info["no_error"], 0)

    def test_no_final_step(self):
        r = TaskCompletedReward()
        s = _make_summary([(True, False), (True, False)])
        res = r.compute(RewardContext(summary=s))
        self.assertEqual(res.value, 0.0)
        self.assertEqual(res.info["has_final_step"], 0)

    def test_require_final_step_false(self):
        r = TaskCompletedReward(require_final_step=False)
        s = _make_summary([(True, False)])  # 无 final
        res = r.compute(RewardContext(summary=s))
        self.assertEqual(res.value, 1.0)  # 因为放宽了 final 要求

    def test_min_answer_len(self):
        r = TaskCompletedReward(min_answer_len=10)
        s = _make_summary([(True, True)], final_answer="短")
        res = r.compute(RewardContext(summary=s))
        self.assertEqual(res.value, 0.0)


# ---------------------------------------------------------------------------
# ToolCallSuccessReward
# ---------------------------------------------------------------------------


class TestToolCallSuccessReward(unittest.TestCase):
    def test_all_ok(self):
        r = ToolCallSuccessReward()
        s = _make_summary([(True, False), (True, False), (True, True)])
        res = r.compute(RewardContext(summary=s))
        self.assertAlmostEqual(res.value, 1.0)
        self.assertEqual(res.info["num_action_steps"], 2)
        self.assertEqual(res.info["num_parse_ok"], 2)

    def test_partial(self):
        r = ToolCallSuccessReward()
        s = _make_summary([(True, False), (False, False), (True, True)])
        res = r.compute(RewardContext(summary=s))
        self.assertAlmostEqual(res.value, 0.5)
        self.assertEqual(res.info["num_action_steps"], 2)
        self.assertEqual(res.info["num_parse_ok"], 1)

    def test_no_action_step(self):
        """一步出 final → action_steps=0，empty_value=1.0 兜底。"""
        r = ToolCallSuccessReward()
        s = _make_summary([(True, True)])
        res = r.compute(RewardContext(summary=s))
        self.assertEqual(res.value, 1.0)
        self.assertEqual(res.info["num_action_steps"], 0)

    def test_custom_empty_value(self):
        r = ToolCallSuccessReward(empty_value=0.0)
        s = _make_summary([(True, True)])
        res = r.compute(RewardContext(summary=s))
        self.assertEqual(res.value, 0.0)


# ---------------------------------------------------------------------------
# JudgeReward
# ---------------------------------------------------------------------------


class TestJudgeReward(unittest.TestCase):
    def test_score_parsed_from_json(self):
        r = JudgeReward(judge_url="http://x/v1", judge_model="m")
        with patch.object(r, "_call_judge", return_value='{"score": 0.85}'):
            s = _make_summary([(True, True)], final_answer="A")
            res = r.compute(RewardContext(summary=s))
            self.assertAlmostEqual(res.value, 0.85)
            self.assertEqual(res.info["parse_ok"], 1)

    def test_score_clipped(self):
        r = JudgeReward(judge_url="http://x/v1", judge_model="m")
        with patch.object(r, "_call_judge", return_value='{"score": 1.7}'):
            res = r.compute(RewardContext(summary=_make_summary([(True, True)])))
            self.assertEqual(res.value, 1.0)
        with patch.object(r, "_call_judge", return_value='{"score": -0.2}'):
            res = r.compute(RewardContext(summary=_make_summary([(True, True)])))
            self.assertEqual(res.value, 0.0)

    def test_score_fallback_to_bare_number(self):
        r = JudgeReward(judge_url="http://x/v1", judge_model="m")
        with patch.object(r, "_call_judge", return_value="评分：0.6"):
            res = r.compute(RewardContext(summary=_make_summary([(True, True)])))
            self.assertAlmostEqual(res.value, 0.6)
            self.assertEqual(res.info["parse_ok"], 0)  # 兜底解析

    def test_judge_call_failure(self):
        r = JudgeReward(judge_url="http://x/v1", judge_model="m", default_on_error=0.0)
        with patch.object(r, "_call_judge", side_effect=RuntimeError("conn refused")):
            res = r.compute(RewardContext(summary=_make_summary([(True, True)])))
            self.assertEqual(res.value, 0.0)
            self.assertIn("error", res.info)

    def test_empty_answer_skipped(self):
        r = JudgeReward(judge_url="http://x/v1", judge_model="m")
        with patch.object(r, "_call_judge") as mock_call:
            s = _make_summary([(True, True)], final_answer="")
            res = r.compute(RewardContext(summary=s))
            self.assertEqual(res.value, 0.0)
            self.assertEqual(res.info["skipped"], "empty_answer")
            mock_call.assert_not_called()

    def test_must_have_url_when_enabled(self):
        with self.assertRaises(ValueError):
            JudgeReward(enabled=True, judge_url="", judge_model="m")
        # 关闭时不强制
        JudgeReward(enabled=False)

    def test_parse_score_helper(self):
        self.assertEqual(_parse_judge_score('{"score":0.5}'), (0.5, True))
        self.assertEqual(_parse_judge_score('prefix {"score": 0.3} suffix'), (0.3, True))
        v, ok = _parse_judge_score("分数 0.7")
        self.assertAlmostEqual(v, 0.7)
        self.assertFalse(ok)
        self.assertEqual(_parse_judge_score(""), (0.0, False))


# ---------------------------------------------------------------------------
# TeacherSimilarityReward
# ---------------------------------------------------------------------------


class TestTeacherSimilarityReward(unittest.TestCase):
    def test_basic_jaccard(self):
        r = TeacherSimilarityReward()
        s = _make_summary(
            [(True, True)],
            response_text="百万医疗 是 一种 保险",
            teacher_trace="百万医疗 是 长期 保险",
        )
        res = r.compute(RewardContext(summary=s))
        # tokens: {百,万,医,疗,是,一,种,保,险} vs {百,万,医,疗,是,长,期,保,险}
        # intersect = {百,万,医,疗,是,保,险} = 7
        # union = {百,万,医,疗,是,一,种,长,期,保,险} = 11
        self.assertAlmostEqual(res.value, 7 / 11)

    def test_no_teacher_returns_empty_value(self):
        r = TeacherSimilarityReward(empty_value=0.1)
        s = _make_summary([(True, True)], response_text="abc def", teacher_trace="")
        res = r.compute(RewardContext(summary=s))
        self.assertEqual(res.value, 0.1)
        self.assertEqual(res.info["skipped"], "below_min_token")

    def test_below_min_token(self):
        r = TeacherSimilarityReward(min_token=5)
        s = _make_summary([(True, True)], response_text="abc", teacher_trace="def ghi")
        res = r.compute(RewardContext(summary=s))
        self.assertEqual(res.value, 0.0)


# ---------------------------------------------------------------------------
# WeightedReward 装配
# ---------------------------------------------------------------------------


class TestWeightedReward(unittest.TestCase):
    def test_weighted_sum_normalized(self):
        wr = WeightedReward(
            components=[TaskCompletedReward(), ToolCallSuccessReward()],
            weights=[1.0, 0.5],
            normalize=True,
        )
        s = _make_summary([(True, False), (True, True)], final_answer="ok")
        res = wr.compute(RewardContext(summary=s))
        # task=1.0 * 1.0 + tool=1.0 * 0.5 = 1.5 / 1.5 = 1.0
        self.assertAlmostEqual(res.value, 1.0)
        # info 含子项
        self.assertIn("reward/task_completed", res.info)
        self.assertIn("reward/tool_call_success", res.info)
        self.assertIn("reward/total", res.info)

    def test_weighted_no_normalize(self):
        wr = WeightedReward(
            components=[TaskCompletedReward(), ToolCallSuccessReward()],
            weights=[1.0, 0.5],
            normalize=False,
        )
        s = _make_summary([(True, True)], final_answer="ok")
        res = wr.compute(RewardContext(summary=s))
        # 1.0*1.0 + 1.0*0.5 = 1.5
        self.assertAlmostEqual(res.value, 1.5)

    def test_clip(self):
        wr = WeightedReward(
            components=[TaskCompletedReward()],
            weights=[2.0],
            normalize=False,
            clip_max=1.0,
        )
        s = _make_summary([(True, True)], final_answer="ok")
        res = wr.compute(RewardContext(summary=s))
        self.assertEqual(res.value, 1.0)

    def test_disabled_component_skipped(self):
        tc = TaskCompletedReward()
        tcs = ToolCallSuccessReward(enabled=False)
        wr = WeightedReward(components=[tc, tcs], weights=[1.0, 1.0])
        s = _make_summary([(True, True)], final_answer="ok")
        res = wr.compute(RewardContext(summary=s))
        # 只有 tc 生效，1.0 / 1.0 = 1.0
        self.assertAlmostEqual(res.value, 1.0)
        # 子项里只有 tc 的 metrics
        self.assertIn("reward/task_completed", res.info)
        self.assertNotIn("reward/tool_call_success", res.info)

    def test_component_exception_isolated(self):
        class BoomReward(TaskCompletedReward):
            name = "boom"
            def compute(self, ctx):
                raise RuntimeError("boom")
        wr = WeightedReward(
            components=[TaskCompletedReward(), BoomReward()],
            weights=[1.0, 1.0],
        )
        s = _make_summary([(True, True)], final_answer="ok")
        res = wr.compute(RewardContext(summary=s))
        # task=1.0, boom=0 (异常兜底)，sum=1.0 / 2.0 = 0.5
        self.assertAlmostEqual(res.value, 0.5)
        self.assertIn("reward/boom/error", res.info)

    def test_build_from_config(self):
        cfg = {
            "components": [
                {
                    "_target_": "recipe.agentic.reward.components.TaskCompletedReward",
                    "weight": 1.0,
                },
                {
                    "_target_": "recipe.agentic.reward.components.ToolCallSuccessReward",
                    "weight": 0.3,
                    "empty_value": 0.5,
                },
            ],
            "normalize": True,
        }
        wr = build_weighted_reward_from_config(cfg)
        self.assertEqual(len(wr.components), 2)
        self.assertEqual(wr.weights, [1.0, 0.3])
        # 验证 kwargs 被传进去
        self.assertEqual(wr.components[1].empty_value, 0.5)


if __name__ == "__main__":
    unittest.main()