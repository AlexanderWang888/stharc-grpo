"""离线单测：跳过 verl 训练栈依赖，验证 AgenticAgentLoop.run 业务逻辑。

策略：
- 在 import agent_loop 之前，向 sys.modules 注入 stub 的 ``verl.experimental.agent_loop.agent_loop``
  + ``verl.utils.profiler``，让 agent_loop 模块能 import 起来
- 然后 __new__ + 手动初始化关键属性，跑 await loop.run(...)
- patch _send_to_vllm，用 smoke 同款 stub
"""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[4]


# ---------------------------------------------------------------------------
# 在 import agent_loop 之前注入 stub 依赖
# ---------------------------------------------------------------------------


def _install_verl_stubs() -> None:
    """注入 verl.experimental.agent_loop.agent_loop / verl.utils.profiler 的 stub。

    只暴露 agent_loop.py 实际用到的符号：AgentLoopBase / AgentLoopOutput / AgentLoopMetrics
    / register；simple_timer。
    """
    if "verl.experimental.agent_loop.agent_loop" in sys.modules:
        # 已经 import 过（说明真的装了 verl），不需要 stub
        return

    pkg_verl = sys.modules.setdefault("verl", types.ModuleType("verl"))
    pkg_verl.__path__ = []  # type: ignore[attr-defined]
    pkg_exp = sys.modules.setdefault("verl.experimental", types.ModuleType("verl.experimental"))
    pkg_exp.__path__ = []  # type: ignore[attr-defined]
    pkg_al_pkg = sys.modules.setdefault(
        "verl.experimental.agent_loop", types.ModuleType("verl.experimental.agent_loop")
    )
    pkg_al_pkg.__path__ = []  # type: ignore[attr-defined]
    pkg_utils = sys.modules.setdefault("verl.utils", types.ModuleType("verl.utils"))
    pkg_utils.__path__ = []  # type: ignore[attr-defined]

    # --- AgentLoopBase / Output / Metrics / register ---
    al_mod = types.ModuleType("verl.experimental.agent_loop.agent_loop")

    from pydantic import BaseModel

    class AgentLoopMetrics(BaseModel):
        generate_sequences: float = 0.0
        tool_calls: float = 0.0
        compute_score: float = 0.0
        num_preempted: int = -1

    class AgentLoopOutput(BaseModel):
        prompt_ids: List[int]
        response_ids: List[int]
        response_mask: List[int]
        reward_score: float = 0.0
        num_turns: int = 0
        metrics: AgentLoopMetrics
        extra_fields: Dict[str, Any] = {}

        # 兼容 verl 真实实现里的 model_dump
        def model_dump(self, *, exclude_unset: bool = False):
            return super().model_dump(exclude_unset=exclude_unset)

    class AgentLoopBase:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    def register(name: str):
        def deco(cls):
            return cls
        return deco

    al_mod.AgentLoopBase = AgentLoopBase
    al_mod.AgentLoopOutput = AgentLoopOutput
    al_mod.AgentLoopMetrics = AgentLoopMetrics
    al_mod.register = register
    sys.modules["verl.experimental.agent_loop.agent_loop"] = al_mod

    # --- verl.utils.profiler.simple_timer ---
    prof_mod = types.ModuleType("verl.utils.profiler")

    @contextmanager
    def simple_timer(name: str, metrics: Dict[str, float]):
        import time as _t
        t = _t.time()
        try:
            yield
        finally:
            metrics[name] = metrics.get(name, 0.0) + (_t.time() - t)

    prof_mod.simple_timer = simple_timer
    sys.modules["verl.utils.profiler"] = prof_mod


# ---------------------------------------------------------------------------
# FakeTokenizer（与 test_builder.py 同款）
# ---------------------------------------------------------------------------


class _FakeEncoding(dict):
    pass


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts: List[str] = []
        for m in messages:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            parts.append(f"[{role}]{content}[/{role}]\n")
        text = "".join(parts)
        if add_generation_prompt:
            text += "[ASSISTANT]"
        return text

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False):
        ids = [ord(c) for c in text]
        enc = _FakeEncoding(input_ids=ids)
        if return_offsets_mapping:
            enc["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return enc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentLoopOffline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # 让 ai2c 可 import
        sys.path.insert(0, str(REPO_ROOT / "ai2c"))
        sys.path.insert(0, str(REPO_ROOT))
        _install_verl_stubs()

    def test_agent_loop_run_end_to_end(self):
        # 此时再 import agent_loop（依赖已经 stub）
        from agentic_train_offline.llm_bridge import bridge as _bridge
        from recipe.agentic.agent_loop import AgenticAgentLoop, _instantiate_target
        from recipe.agentic.reward.weighted import build_weighted_reward_from_config
        from recipe.agentic.trajectory.builder import TrajectoryBuilder
        from recipe.agentic.scripts.smoke_env_builder import _make_stub_vllm

        cfg = SimpleNamespace(
            agentic={
                "env": {
                    "_target_": "recipe.agentic.envs.AiToCReactEnv",
                    "ai_to_c_repo_path": str(REPO_ROOT / "ai2c"),
                },
                "reward": {
                    "components": [
                        {
                            "_target_": "recipe.agentic.reward.components.TaskCompletedReward",
                            "weight": 1.0,
                        },
                        {
                            "_target_": "recipe.agentic.reward.components.ToolCallSuccessReward",
                            "weight": 0.3,
                        },
                    ],
                    "normalize": True,
                },
                "vllm_base_url": "http://stub-vllm/v1",
                "vllm_model_name": "qwen-32b-rl",
                "per_rollout_timeout_sec": 30,
            }
        )

        loop = AgenticAgentLoop.__new__(AgenticAgentLoop)
        loop.config = cfg
        loop.tokenizer = FakeTokenizer()
        loop.processor = None
        loop.rollout_config = SimpleNamespace(prompt_length=0, response_length=0)

        agentic_cfg = loop._resolve_agentic_config()
        loop.env = _instantiate_target(agentic_cfg["env"])
        loop.reward = build_weighted_reward_from_config(agentic_cfg["reward"])
        loop.vllm_base_url = agentic_cfg["vllm_base_url"]
        loop.vllm_model_name = agentic_cfg["vllm_model_name"]
        loop.per_rollout_timeout_sec = float(agentic_cfg["per_rollout_timeout_sec"])
        loop.builder = TrajectoryBuilder(tokenizer=loop.tokenizer, action_parser=loop.env.action_parser)
        loop.prompt_length = 0
        loop.response_length = 0
        loop.env.setup_once(loop.vllm_base_url, loop.vllm_model_name)

        # patch _send_to_vllm
        stub_vllm, call_idx = _make_stub_vllm()
        _bridge._send_to_vllm = stub_vllm

        # run
        out = asyncio.run(loop.run(
            sampling_params={},
            raw_prompt=[{"role": "user", "content": "百万医疗险是什么？"}],
            ext_params={},
            extra_info={"teacher_trace": ""},
        ))

        self.assertGreaterEqual(call_idx["n"], 1)
        self.assertGreater(len(out.prompt_ids), 0)
        self.assertGreater(len(out.response_ids), 0)
        self.assertEqual(len(out.response_ids), len(out.response_mask))
        self.assertGreaterEqual(out.reward_score, 0.0)
        self.assertLessEqual(out.reward_score, 1.0)
        self.assertGreaterEqual(out.num_turns, 1)
        self.assertEqual(out.extra_fields["agentic_metrics"]["agentic/env_status"], "ok")
        self.assertEqual(out.extra_fields["agentic_metrics"]["reward/task_completed"], 1.0)
        self.assertTrue(out.extra_fields["agentic_final_answer"])
        self.assertIn("agentic_summary", out.extra_fields)
        # 子项 metrics 完整
        self.assertIn("reward/_total", out.extra_fields["agentic_metrics"])
        # response_mask 既有 1 又有 0
        ss = sum(out.response_mask)
        self.assertGreater(ss, 0)
        self.assertLess(ss, len(out.response_mask))

    def test_extract_query_helper(self):
        from recipe.agentic.agent_loop import _extract_query

        self.assertEqual(_extract_query([{"role": "user", "content": "abc"}]), "abc")
        # 多轮：取最后一条 user
        self.assertEqual(
            _extract_query(
                [
                    {"role": "user", "content": "x"},
                    {"role": "assistant", "content": "y"},
                    {"role": "user", "content": "z"},
                ]
            ),
            "z",
        )
        self.assertEqual(_extract_query("plain"), "plain")
        self.assertEqual(_extract_query([]), "")
        # 没有 user → 退化到最后一条 content
        self.assertEqual(_extract_query([{"role": "system", "content": "s"}]), "s")


if __name__ == "__main__":
    unittest.main()