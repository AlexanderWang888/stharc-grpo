"""离线 smoke：跳过 verl AgentLoopBase 的 hydra/Ray 依赖，直接构造 AgenticAgentLoop 并跑 run。

通过给 AgenticAgentLoop 注入精简的 self.config + self.tokenizer + self.rollout_config，
避免真实启动 verl 训练栈。验证 AgentLoopOutput 各字段类型/值正确。

用法：

    python3 recipe/agentic/scripts/smoke_agent_loop.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("smoke_agent_loop")


# 在导入 recipe 前提升当前副本，支持从任意工作目录直接运行本脚本。
_repo_root = str(Path(__file__).resolve().parents[3])
sys.path[:] = [_repo_root] + [p for p in sys.path if p != _repo_root]

# 复用 smoke_env_builder 里的 FakeTokenizer + Stub vLLM
from recipe.agentic.scripts.smoke_env_builder import (
    FakeTokenizer, _configure_import_paths, _make_stub_vllm,
)


def _build_loop_instance():
    """绕开 AgentLoopBase.__init__ 的 hydra/Ray 依赖，构造一个 AgenticAgentLoop 实例。"""
    ai2c = _configure_import_paths()
    from agentic_train_offline.llm_bridge import bridge as _bridge
    from recipe.agentic.agent_loop import AgenticAgentLoop

    cfg = SimpleNamespace(
        agentic={
            "env": {
                "_target_": "recipe.agentic.envs.AiToCReactEnv",
                "ai_to_c_repo_path": str(ai2c),
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

    # 手动 __new__ + 设置必需属性（跳过 AgentLoopBase.__init__）
    loop = AgenticAgentLoop.__new__(AgenticAgentLoop)
    loop.config = cfg
    loop.tokenizer = FakeTokenizer()
    loop.processor = None
    loop.dataset_cls = None
    loop.apply_chat_template_kwargs = {}
    loop.mm_processor_kwargs = {}
    loop.system_prompt = []
    loop.rollout_config = SimpleNamespace(prompt_length=0, response_length=0)
    loop.server_manager = None

    # 跑 AgenticAgentLoop.__init__ 的核心逻辑（不能直接 call __init__，因为它会 super().__init__）
    # 直接复制关键初始化代码：
    agentic_cfg = loop._resolve_agentic_config()
    from recipe.agentic.agent_loop import _instantiate_target
    from recipe.agentic.reward.weighted import build_weighted_reward_from_config
    from recipe.agentic.trajectory.builder import TrajectoryBuilder

    loop.env = _instantiate_target(agentic_cfg["env"])
    loop.reward = build_weighted_reward_from_config(agentic_cfg["reward"])
    loop.vllm_base_url = agentic_cfg["vllm_base_url"]
    loop.vllm_model_name = agentic_cfg["vllm_model_name"]
    loop.per_rollout_timeout_sec = float(agentic_cfg["per_rollout_timeout_sec"])
    loop.builder = TrajectoryBuilder(tokenizer=loop.tokenizer, action_parser=loop.env.action_parser)
    loop.prompt_length = 0
    loop.response_length = 0
    loop.env.setup_once(loop.vllm_base_url, loop.vllm_model_name)

    # patch _send_to_vllm 走 stub
    stub_vllm, call_idx = _make_stub_vllm()
    _bridge._send_to_vllm = stub_vllm
    return loop, call_idx


async def _run():
    loop, call_idx = _build_loop_instance()

    raw_prompt = [
        {"role": "user", "content": "百万医疗险是什么？"},
    ]
    out = await loop.run(
        sampling_params={},
        raw_prompt=raw_prompt,
        ext_params={},
        extra_info={"teacher_trace": ""},
    )

    logger.info("==> AgentLoopOutput keys: %s", list(out.model_dump(exclude_unset=True).keys()))
    logger.info("    reward_score=%.4f", out.reward_score)
    logger.info("    num_turns=%d", out.num_turns)
    logger.info("    len(prompt_ids)=%d  len(response_ids)=%d  len(response_mask)=%d",
                len(out.prompt_ids), len(out.response_ids), len(out.response_mask))
    logger.info("    metrics: gen=%.4fs  tool_calls=%.0f  compute_score=%.4fs",
                out.metrics.generate_sequences, out.metrics.tool_calls, out.metrics.compute_score)
    logger.info("    extra_fields keys: %s", list(out.extra_fields.keys()))
    logger.info("    agentic_metrics sample:")
    for k in sorted(out.extra_fields["agentic_metrics"]):
        logger.info("        %s = %s", k, out.extra_fields["agentic_metrics"][k])

    # 断言
    assert call_idx["n"] >= 1
    assert len(out.prompt_ids) > 0
    assert len(out.response_ids) > 0
    assert len(out.response_ids) == len(out.response_mask)
    assert 0.0 <= out.reward_score <= 1.0
    assert out.num_turns >= 1
    assert out.extra_fields["agentic_metrics"]["agentic/env_status"] == "ok"
    assert out.extra_fields["agentic_metrics"]["reward/task_completed"] == 1.0
    assert out.extra_fields["agentic_final_answer"]
    assert "agentic_summary" in out.extra_fields
    logger.info("==> AGENT LOOP SMOKE PASSED")


def main() -> int:
    _configure_import_paths()
    asyncio.run(_run())
    return 0


if __name__ == "__main__":
    sys.exit(main())