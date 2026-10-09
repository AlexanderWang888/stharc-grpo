"""STHARC：memory prospective 归因 + frozen 缓存按 step 失效。"""

from __future__ import annotations

import math
import unittest

from recipe.agentic.stharc.core import (
    SKILLRISE_DOC_BEGIN,
    SKILLRISE_DOC_END,
    apply_skillrise_document_to_task,
    compute_prospective_utility,
    compute_sage_skill_integrated,
    compute_skillrise_decoupled_credit,
    generate_skillrise_document,
    inject_skillrise_document,
)
from recipe.agentic.stharc.frozen_cache import load_frozen_rewards_cache


class TestMemoryProspectiveBonus(unittest.TestCase):
    def test_memory_write_gets_credit_when_later_task_injects(self):
        """task0 写 memory:memory，task1 被动注入读到 → prospective > 0。"""
        write_output = {
            "object_id": "memory:memory",
            "object_type": "memory_entry",
            "mutation_type": "create",
            "before": "",
            "after": "prefers A",
        }
        harness_mutations = [
            (0, {"op_type": "memory", "target_id": "memory:memory", "args": {}}, write_output),
        ]
        task_read_sets = [
            set(),  # task0：写时尚未注入
            {"memory:memory"},  # task1：format_for_system_prompt 注入
        ]
        task_rewards = [0.0, 1.0]
        frozen_rewards = [0.0, 0.0]

        bonus = compute_prospective_utility(
            harness_mutations=harness_mutations,
            task_read_sets=task_read_sets,
            task_rewards=task_rewards,
            frozen_rewards=frozen_rewards,
            gamma=0.9,
            attribution="gamma_decay",
        )
        # (1.0 - 0.0) * 0.9 ** 1
        self.assertAlmostEqual(bonus, 0.9)

    def test_memory_write_no_credit_without_read(self):
        write_output = {
            "object_id": "memory:user",
            "object_type": "memory_entry",
            "mutation_type": "create",
            "before": "",
            "after": "vip",
        }
        bonus = compute_prospective_utility(
            harness_mutations=[
                (0, {"op_type": "memory", "target_id": "memory:user", "args": {}}, write_output),
            ],
            task_read_sets=[set(), set()],
            task_rewards=[0.0, 1.0],
            frozen_rewards=[0.0, 0.0],
            gamma=0.9,
        )
        self.assertEqual(bonus, 0.0)

    def test_random_shuffle_breaks_correct_attribution(self):
        """Random-PU：正确路径有正 credit；打乱 oid 门控后应变 0（本例只有一个可读 oid）。"""
        mut_a = {
            "object_id": "mem::a",
            "object_type": "memory_entry",
            "mutation_type": "create",
            "before": "",
            "after": "a",
        }
        mut_b = {
            "object_id": "mem::b",
            "object_type": "memory_entry",
            "mutation_type": "create",
            "before": "",
            "after": "b",
        }
        harness_mutations = [
            (0, {"op_type": "memory", "target_id": "mem::a", "args": {}}, mut_a),
            (1, {"op_type": "memory", "target_id": "mem::b", "args": {}}, mut_b),
        ]
        # task2 只读 a；正确归因下 mut_a 拿到 credit
        task_read_sets = [set(), set(), {"mem::a"}]
        task_rewards = [0.0, 0.0, 1.0]
        frozen_rewards = [0.0, 0.0, 0.0]

        correct = compute_prospective_utility(
            harness_mutations, task_read_sets, task_rewards, frozen_rewards,
            gamma=0.9, attribution="gamma_decay",
        )
        self.assertGreater(correct, 0.0)

        # seed 选一个会把 a↔b 换掉的置换（对 {a,b} 只有恒等与对换两种）
        random_pu = compute_prospective_utility(
            harness_mutations, task_read_sets, task_rewards, frozen_rewards,
            gamma=0.9, attribution="random_shuffle", shuffle_seed=1,
        )
        # 对换后 gate(a)=b ∉ read_set → 总 bonus 应变 0；若 seed 恰好恒等则换 seed
        if random_pu == correct:
            random_pu = compute_prospective_utility(
                harness_mutations, task_read_sets, task_rewards, frozen_rewards,
                gamma=0.9, attribution="random_shuffle", shuffle_seed=0,
            )
        self.assertNotEqual(random_pu, correct)


class TestSageSkillIntegrated(unittest.TestCase):
    def _write(self, oid: str, mutation_type: str = "create") -> dict:
        return {
            "object_id": oid,
            "object_type": "memory_entry" if oid.startswith("mem::") else "skill",
            "mutation_type": mutation_type,
            "before": "",
            "after": "x",
        }

    def test_successful_generation_and_later_use_get_bonus(self):
        """默认自身与后来任务均成功，generation 和 usage 各加一。"""
        muts = [
            (0, {"op_type": "memory", "args": {"action": "add"}}, self._write("mem::a")),
        ]
        got = compute_sage_skill_integrated(
            harness_mutations=muts,
            task_read_sets=[set(), {"mem::a"}],
            task_rewards=[1.0, 1.0],
        )
        self.assertEqual(got.generation_bonus, [1.0, 0.0])
        self.assertEqual(got.usage_bonus, [0.0, 1.0])
        self.assertEqual(got.task_rewards_sage, [2.0, 2.0])
        self.assertAlmostEqual(got.chain_reward, 4.0)
        self.assertAlmostEqual(got.gen_bonus_sum, 1.0)
        self.assertAlmostEqual(got.use_bonus_sum, 1.0)

    def test_bonus_coef_scales_both_terms(self):
        muts = [
            (0, {"op_type": "memory", "args": {"action": "add"}}, self._write("mem::a")),
        ]
        got = compute_sage_skill_integrated(
            harness_mutations=muts,
            task_read_sets=[set(), {"mem::a"}],
            task_rewards=[1.0, 1.0],
            bonus_coef=0.5,
        )
        self.assertEqual(got.generation_bonus, [0.5, 0.0])
        self.assertEqual(got.usage_bonus, [0.0, 0.5])
        self.assertAlmostEqual(got.chain_reward, 3.0)

    def test_default_self_failure_gives_no_generation(self):
        muts = [
            (0, {"op_type": "memory", "args": {"action": "add"}}, self._write("mem::a")),
        ]
        got = compute_sage_skill_integrated(
            harness_mutations=muts,
            task_read_sets=[set(), {"mem::a"}],
            task_rewards=[0.0, 1.0],
        )
        self.assertEqual(got.generation_bonus, [0.0, 0.0])
        self.assertEqual(got.usage_bonus, [0.0, 1.0])
        self.assertEqual(got.task_rewards_sage, [0.0, 2.0])
        self.assertAlmostEqual(got.chain_reward, 2.0)

    def test_explicitly_disable_self_success_requirement(self):
        got = compute_sage_skill_integrated(
            harness_mutations=[
                (0, {"op_type": "memory", "args": {"action": "add"}}, self._write("mem::a")),
            ],
            task_read_sets=[set(), {"mem::a"}],
            task_rewards=[0.0, 1.0],
            require_self_success=False,
        )
        self.assertEqual(got.generation_bonus, [1.0, 0.0])
        self.assertEqual(got.usage_bonus, [0.0, 1.0])
        self.assertEqual(got.task_rewards_sage, [1.0, 2.0])
        self.assertAlmostEqual(got.chain_reward, 3.0)

    def test_no_bonus_without_read(self):
        muts = [
            (0, {"op_type": "memory", "args": {"action": "add"}}, self._write("mem::a")),
        ]
        got = compute_sage_skill_integrated(
            harness_mutations=muts,
            task_read_sets=[set(), set()],
            task_rewards=[1.0, 1.0],
        )
        self.assertEqual(got.generation_bonus, [0.0, 0.0])
        self.assertEqual(got.usage_bonus, [0.0, 0.0])
        self.assertEqual(got.task_rewards_sage, [1.0, 1.0])
        self.assertAlmostEqual(got.chain_reward, 2.0)

    def test_later_read_but_fail_gives_no_generation(self):
        muts = [
            (0, {"op_type": "skill_manage", "args": {"action": "create"}}, self._write("skill::s")),
        ]
        got = compute_sage_skill_integrated(
            harness_mutations=muts,
            task_read_sets=[set(), {"skill::s"}],
            task_rewards=[1.0, 0.0],
        )
        self.assertEqual(got.generation_bonus, [0.0, 0.0])
        self.assertEqual(got.usage_bonus, [0.0, 0.0])

    def test_generation_capped_at_one_across_multiple_later_uses(self):
        muts = [
            (0, {"op_type": "memory", "args": {"action": "add"}}, self._write("mem::a")),
        ]
        got = compute_sage_skill_integrated(
            harness_mutations=muts,
            task_read_sets=[set(), {"mem::a"}, {"mem::a"}],
            task_rewards=[1.0, 1.0, 1.0],
        )
        self.assertEqual(got.generation_bonus, [1.0, 0.0, 0.0])
        self.assertEqual(got.usage_bonus, [0.0, 1.0, 1.0])
        self.assertAlmostEqual(got.gen_bonus_sum, 1.0)
        self.assertAlmostEqual(got.use_bonus_sum, 2.0)

    def test_delete_does_not_count_as_write(self):
        muts = [
            (0, {"op_type": "memory", "args": {"action": "delete"}}, self._write("mem::a", "delete")),
        ]
        got = compute_sage_skill_integrated(
            harness_mutations=muts,
            task_read_sets=[set(), {"mem::a"}],
            task_rewards=[1.0, 1.0],
        )
        self.assertEqual(got.generation_bonus, [0.0, 0.0])
        self.assertEqual(got.usage_bonus, [0.0, 0.0])

    def test_modify_counts_as_write(self):
        muts = [
            (0, {"op_type": "skill_manage", "args": {"action": "modify"}}, self._write("skill::s", "modify")),
        ]
        got = compute_sage_skill_integrated(
            harness_mutations=muts,
            task_read_sets=[set(), {"skill::s"}],
            task_rewards=[1.0, 1.0],
        )
        self.assertEqual(got.generation_bonus[0], 1.0)
        self.assertEqual(got.usage_bonus[1], 1.0)

    def test_nan_task_reward_masks_chain(self):
        got = compute_sage_skill_integrated(
            harness_mutations=[],
            task_read_sets=[set(), set()],
            task_rewards=[0.0, float("nan")],
        )
        self.assertTrue(math.isnan(got.chain_reward))
        self.assertEqual(got.gen_bonus_sum, 0.0)


class TestSkillRiseDecoupledCredit(unittest.TestCase):
    def test_curate_discounts_all_later_upstream_rewards(self):
        """上游奖励前四项为 0、后两项为 1，curate 累积后来任务的折扣奖励。"""
        got = compute_skillrise_decoupled_credit([0.0, 0.0, 0.0, 0.0, 1.0, 1.0], gamma=0.6)
        self.assertEqual(got.solve_returns, [0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(got.curate_returns[0], 0.6**4 + 0.6**5)
        self.assertAlmostEqual(got.curate_returns[1], 0.6**3 + 0.6**4)
        self.assertAlmostEqual(got.curate_returns[2], 0.6**2 + 0.6**3)
        self.assertAlmostEqual(got.curate_returns[3], 0.6 + 0.6**2)
        self.assertAlmostEqual(got.curate_returns[4], 0.6)
        self.assertAlmostEqual(got.curate_returns[5], 0.0)
        self.assertAlmostEqual(got.curate_return_sum, sum(got.curate_returns))
        self.assertAlmostEqual(got.chain_reward, 2.0 + got.curate_return_sum)

    def test_last_task_has_no_curate(self):
        got = compute_skillrise_decoupled_credit([1.0], gamma=0.6)
        self.assertEqual(got.solve_returns, [1.0])
        self.assertEqual(got.curate_returns, [0.0])
        self.assertAlmostEqual(got.chain_reward, 1.0)

    def test_empty_chain(self):
        got = compute_skillrise_decoupled_credit([])
        self.assertEqual(got.solve_returns, [])
        self.assertEqual(got.curate_returns, [])
        self.assertEqual(got.chain_reward, 0.0)

    def test_nan_masks_chain(self):
        got = compute_skillrise_decoupled_credit([0.0, float("nan")])
        self.assertTrue(math.isnan(got.chain_reward))
        self.assertEqual(got.curate_return_sum, 0.0)


class TestSkillRiseInject(unittest.TestCase):
    def test_empty_document_leaves_query(self):
        self.assertEqual(inject_skillrise_document("Q?", ""), "Q?")
        self.assertEqual(inject_skillrise_document("Q?", "   "), "Q?")

    def test_wraps_document_before_query(self):
        out = inject_skillrise_document("Who founded X?", "store founder under key k")
        self.assertTrue(out.startswith(SKILLRISE_DOC_BEGIN))
        self.assertIn("store founder under key k", out)
        self.assertIn(SKILLRISE_DOC_END, out)
        self.assertTrue(out.endswith("Who founded X?"))

    def test_apply_updates_query_and_last_user(self):
        task = {
            "query": "hop2?",
            "raw_prompt": "hop2?",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hop2?"},
            ],
        }
        got = apply_skillrise_document_to_task(task, "use memory key founder")
        self.assertIn("use memory key founder", got["query"])
        self.assertIn("use memory key founder", got["raw_prompt"])
        self.assertIn("use memory key founder", got["messages"][-1]["content"])
        self.assertEqual(task["query"], "hop2?")

    def test_generate_falls_back_on_empty_or_error(self):
        self.assertEqual(
            generate_skillrise_document(
                lambda _m: "",
                prev_document="keep-me",
                task_query="q",
                trajectory_text="t",
                task_reward=0.0,
            ),
            "keep-me",
        )

        def _boom(_m):
            raise RuntimeError("llm down")

        self.assertEqual(
            generate_skillrise_document(
                _boom,
                prev_document="keep-me",
                task_query="q",
                trajectory_text="t",
                task_reward=1.0,
            ),
            "keep-me",
        )

    def test_generate_strips_wrapper_markers(self):
        wrapped = f"{SKILLRISE_DOC_BEGIN}\nnote about keys\n{SKILLRISE_DOC_END}"
        got = generate_skillrise_document(
            lambda _m: wrapped,
            prev_document="",
            task_query="q",
            trajectory_text="t",
            task_reward=0.0,
        )
        self.assertEqual(got, "note about keys")


class TestFrozenRewardsCache(unittest.TestCase):
    def test_accepts_same_step(self):
        raw = {
            "schema": "stharc_frozen_v3",
            "global_step": 12,
            "rewards": [0.1, 0.2, 0.3],
        }
        got = load_frozen_rewards_cache(raw, global_step=12, expected_len=3)
        self.assertEqual(got, [0.1, 0.2, 0.3])

    def test_rejects_different_step(self):
        raw = {
            "schema": "stharc_frozen_v3",
            "global_step": 12,
            "rewards": [0.1, 0.2, 0.3],
        }
        self.assertIsNone(
            load_frozen_rewards_cache(raw, global_step=13, expected_len=3)
        )

    def test_rejects_legacy_v2(self):
        raw = {"schema": "stharc_frozen_v2", "rewards": [0.1, 0.2]}
        self.assertIsNone(
            load_frozen_rewards_cache(raw, global_step=1, expected_len=2)
        )


if __name__ == "__main__":
    unittest.main()
