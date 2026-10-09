"""离线 smoke 测试：完整跑通 bridge → Env → TrajectoryBuilder → TrajectorySummary。

不需要业务 LLM endpoint，也不需要 vLLM。做法：
  - patch agentic_train_offline.llm_bridge.bridge._send_to_vllm 为本地 stub（模拟主脑 2 步对话）
  - 通过 AiToCReactEnv 跑一次 chat_stream
  - 用 FakeTokenizer 喂给 TrajectoryBuilder 重建 trajectory
  - 校验 summary / mask 关键字段

用法：

    python3 recipe/agentic/scripts/smoke_env_builder.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("smoke_env_builder")


# ---------------------------------------------------------------------------
# FakeTokenizer（同 test_builder.py）
# ---------------------------------------------------------------------------


class _FakeEncoding(dict):
    pass


class FakeTokenizer:
    """一字符一 token；chat_template 用 [ROLE]content[/ROLE]\\n 形式。"""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
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
# Stub vLLM：根据 messages 长度返回不同 response，模拟两步 ReAct
# ---------------------------------------------------------------------------


_STEP_RESPONSES = [
    'Thought: 先查通用知识\nAction: {"name":"search_general_kb","arguments":{"query":"百万医疗"}}\n',
    'Thought: 已收集足够信息\nAction: {"name":"finish","arguments":{"answer":"百万医疗是一种保额上百万的医疗险..."}}\n',
]


def _make_stub_vllm():
    call_idx = {"n": 0}

    def _stub(messages, base_url, model, temperature=0.1, max_tokens=16384, timeout=120):
        idx = call_idx["n"]
        call_idx["n"] += 1
        resp = _STEP_RESPONSES[idx] if idx < len(_STEP_RESPONSES) else _STEP_RESPONSES[-1]
        # 模拟流式：逐 16 字符 yield
        for i in range(0, len(resp), 16):
            yield resp[i : i + 16]

    return _stub, call_idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _configure_import_paths() -> Path:
    """优先使用当前副本，同时保留业务仓和 bridge 的导入路径。"""
    repo_root = Path(__file__).resolve().parents[3]
    ws_root = repo_root.parent.parent
    ai2c = Path(__import__("os").environ.get("AI2C_REPO_PATH") or (ws_root / "ai2c"))
    paths = [repo_root]
    if (repo_root / "verl").is_dir():
        paths.append(repo_root / "verl")
    paths.extend((ai2c, ws_root))
    preferred = list(dict.fromkeys(str(p.resolve()) for p in paths))
    sys.path[:] = preferred + [p for p in sys.path if p not in preferred]
    return ai2c


def main() -> int:
    ai2c = _configure_import_paths()

    from agentic_train_offline.llm_bridge import bridge as _bridge
    from recipe.agentic.envs import AiToCReactEnv
    from recipe.agentic.reward import (
        RewardContext,
        TaskCompletedReward,
        TeacherSimilarityReward,
        ToolCallSuccessReward,
        WeightedReward,
    )
    from recipe.agentic.trajectory.builder import TrajectoryBuilder

    # 1) 创建 env + setup（包含 install_trace_bridge + 注册主脑识别器 + vLLM override）
    env = AiToCReactEnv(
        ai_to_c_repo_path=str(ai2c),
        # 不指向 tmpfs 也行；smoke test 跑完手动忽略
    )
    env.setup_once(
        vllm_base_url="http://stub-vllm/v1",
        vllm_model_name="qwen-32b-rl",
    )

    # 2) patch _send_to_vllm 为 stub
    stub_vllm, call_idx = _make_stub_vllm()
    _bridge._send_to_vllm = stub_vllm

    # 3) 跑 env.run
    logger.info("==> env.run start")
    query = "百万医疗险是什么？"
    env_result = env.run(query=query)
    logger.info("==> env.run done")
    logger.info("vllm calls=%d", call_idx["n"])
    logger.info("env_info=%s", json.dumps(env_result.env_info, ensure_ascii=False, indent=2))
    logger.info("num records (mainbrain only)=%d", len(env_result.records))
    logger.info("final_answer=%s", env_result.final_answer[:200])

    # 简单断言
    assert env_result.env_info["status"] == "ok", f"status != ok: {env_result.env_info}"
    assert call_idx["n"] >= 1, "stub_vllm was never called"
    assert len(env_result.records) >= 1, "no mainbrain trace captured"
    for rec in env_result.records:
        assert rec.extra.get("role") == "mainbrain", f"record role != mainbrain: {rec.extra}"
        assert rec.extra.get("via_vllm") is True

    # 4) 喂给 TrajectoryBuilder
    builder = TrajectoryBuilder(tokenizer=FakeTokenizer(), action_parser=env.action_parser)
    out = builder.build(env_result, query=query)

    logger.info("==> builder.build done")
    logger.info("summary.num_steps=%d", out.summary.num_steps)
    logger.info("summary.num_action_steps=%d", out.summary.num_action_steps)
    logger.info("summary.num_parse_ok=%d", out.summary.num_parse_ok)
    logger.info("summary.has_final_step=%s", out.summary.has_final_step)
    logger.info("summary.has_error=%s", out.summary.has_error)
    logger.info("len(prompt_ids)=%d", len(out.prompt_ids))
    logger.info("len(response_ids)=%d", len(out.response_ids))
    logger.info("sum(response_mask)=%d / %d", sum(out.response_mask), len(out.response_mask))

    # 关键断言
    assert out.summary.num_steps == len(env_result.records)
    assert out.summary.has_final_step, "最后一步必须是 finish/Final Answer"
    assert out.summary.num_parse_ok == out.summary.num_steps, "所有 stub response 都该 parse_ok"
    assert len(out.response_ids) == len(out.response_mask)
    # response_mask 不全 1（应该有 obs 段）也不全 0（应该有 assistant 段）
    if len(env_result.records) >= 2:
        s = sum(out.response_mask)
        assert 0 < s < len(out.response_mask), (
            f"mask 应有 assistant=1 + obs=0 混合，但 sum={s} len={len(out.response_mask)}"
        )

    # 5) 打印一段 summary，方便人眼看
    logger.info("==> summary.to_dict():")
    logger.info("%s", json.dumps(out.summary.to_dict(), ensure_ascii=False, indent=2)[:1500])

    # 6) Reward 链路：装配 WeightedReward 跑分（JudgeReward 关闭避免真实 HTTP）
    reward = WeightedReward(
        components=[
            TaskCompletedReward(),
            ToolCallSuccessReward(),
            TeacherSimilarityReward(empty_value=0.0),
        ],
        weights=[1.0, 0.3, 0.2],
        normalize=True,
    )
    rew_res = reward.compute(RewardContext(summary=out.summary))
    logger.info("==> reward.compute done | total=%.4f", rew_res.value)
    logger.info("    sub metrics:")
    for k in sorted(rew_res.info):
        logger.info("      %s = %s", k, rew_res.info[k])

    assert 0.0 <= rew_res.value <= 1.0
    # 至少 task_completed 给了分（因为 stub 跑出了 final_answer + finish）
    assert rew_res.info.get("reward/task_completed", 0) == 1.0

    logger.info("==> SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())