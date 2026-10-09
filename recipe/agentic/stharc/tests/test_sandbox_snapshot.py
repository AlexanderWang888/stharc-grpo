"""HarnessState.snapshot_for_log：cleanup 前沙箱落盘快照。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from recipe.agentic.stharc.harness_state import HarnessState


class TestSandboxSnapshot(unittest.TestCase):
    def test_snapshot_lists_skills_and_dumps_mutated(self):
        with tempfile.TemporaryDirectory() as tmp:
            skills_root = Path(tmp) / "skills"
            (skills_root / "skill_keep").mkdir(parents=True)
            (skills_root / "skill_keep" / "SKILL.md").write_text(
                "# keep\n", encoding="utf-8"
            )
            (skills_root / "skill_new").mkdir(parents=True)
            (skills_root / "skill_new" / "SKILL.md").write_text(
                "# new skill body\n", encoding="utf-8"
            )

            mem_dir = Path(tmp) / "memory"
            user_id = "u_chain0"
            (mem_dir / user_id).mkdir(parents=True)
            mem_obj = {"memory": ["note-a"], "user": []}
            (mem_dir / user_id / "memory.json").write_text(
                json.dumps(mem_obj), encoding="utf-8"
            )

            state = HarnessState(
                group_id="g1",
                chain_tag="chain_0",
                skills_root=skills_root,
                memory_user_id=user_id,
                provenance_registry={
                    "skill_new": [
                        {
                            "source_task": 0,
                            "action": {"op_type": "skill_manage", "target_id": "skill_new"},
                            "mutation_type": "create",
                        }
                    ]
                },
            )

            with mock.patch(
                "recipe.agentic.stharc.harness_state._resolve_memory_dir",
                return_value=mem_dir,
            ):
                snap = state.snapshot_for_log(mutated_skill_ids=["skill_new"])

            self.assertEqual(snap["skill_ids"], ["skill_keep", "skill_new"])
            self.assertEqual(snap["skill_count"], 2)
            self.assertEqual(snap["mutated_skill_ids"], ["skill_new"])
            # 默认只落变更 skill 全文，不落整库
            self.assertEqual(snap["skills"], {"skill_new": "# new skill body\n"})
            self.assertFalse(snap["include_all_skill_contents"])
            self.assertEqual(snap["memory"], mem_obj)
            self.assertEqual(
                snap["provenance_registry"]["skill_new"][0]["mutation_type"], "create"
            )

    def test_snapshot_full_skills_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            skills_root = Path(tmp) / "skills"
            for sid, body in (("a", "# a\n"), ("b", "# b\n")):
                (skills_root / sid).mkdir(parents=True)
                (skills_root / sid / "SKILL.md").write_text(body, encoding="utf-8")

            state = HarnessState(
                group_id="g1",
                chain_tag="chain_0",
                skills_root=skills_root,
                memory_user_id="nobody",
            )
            with mock.patch(
                "recipe.agentic.stharc.harness_state._resolve_memory_dir",
                return_value=Path(tmp) / "memory",
            ):
                snap = state.snapshot_for_log(include_all_skill_contents=True)

            self.assertEqual(snap["skills"], {"a": "# a\n", "b": "# b\n"})
            self.assertTrue(snap["include_all_skill_contents"])
            self.assertIsNone(snap["memory"])


if __name__ == "__main__":
    unittest.main()
