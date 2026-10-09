"""TrajectoryBuilder 核心单测（不依赖真实 HF tokenizer）。

测试用 FakeTokenizer：
- 一个字符 = 一个 token
- chat_template 把 messages 渲染为：``[ROLE]content[/ROLE]\n`` 串接
- 返回的 offset_mapping 是 [(i, i+1) for i in range(len(text))]
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List

from recipe.agentic.envs.base import ActionParser, EnvRunResult, ParsedAction
from recipe.agentic.trajectory.builder import TrajectoryBuilder


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeEncoding(dict):
    """让 enc["input_ids"] / enc["offset_mapping"] 都能直接用。"""


class FakeTokenizer:
    """一字符一 token 的极简 tokenizer。"""

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        assert tokenize is False, "FakeTokenizer only supports tokenize=False"
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
        ids = [ord(c) for c in text]  # 一字符一 token
        enc = _FakeEncoding(input_ids=ids)
        if return_offsets_mapping:
            enc["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return enc


class FakeRecord:
    """模拟 agentic_train_offline.llm_bridge.TraceRecord。"""

    def __init__(self, messages: List[Dict[str, str]], response: str, model: str = "fake-32b"):
        self.messages = messages
        self.response = response
        self.model = model
        self.ts_start = 0.0
        self.ts_end = 0.0


class FakeActionParser(ActionParser):
    """简单 parser：response 含 'FINAL' → is_final；含 'ACT:' → parse_ok。"""

    def parse(self, text: str) -> ParsedAction:
        if "FINAL" in text:
            return ParsedAction(
                parsed_action={"name": "finish", "arguments": {"answer": text}},
                parse_ok=True,
                is_final=True,
                raw_text=text,
            )
        if text.startswith("ACT:"):
            return ParsedAction(
                parsed_action={"name": "tool_a", "arguments": {}},
                parse_ok=True,
                is_final=False,
                raw_text=text,
            )
        return ParsedAction(parsed_action=None, parse_ok=False, is_final=False, raw_text=text)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrajectoryBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.parser = FakeActionParser()
        self.builder = TrajectoryBuilder(tokenizer=self.tokenizer, action_parser=self.parser)

    # -------- 测试 1：完整 roundtrip ----------------------------------

    def test_roundtrip_basic(self):
        """两步对话：第一步调用工具，第二步给出 final answer。"""
        sys_msg = {"role": "system", "content": "你是 ReAct"}
        user_msg = {"role": "user", "content": "你好"}
        step0_messages = [sys_msg, user_msg]

        step0_response = "ACT: search"

        # 第二步 messages = 第一步 + assistant(response0) + tool(observation)
        obs_msg = {"role": "tool", "content": "OBS_DATA"}
        step1_messages = step0_messages + [
            {"role": "assistant", "content": step0_response},
            obs_msg,
        ]
        step1_response = "FINAL: 答复"

        records = [
            FakeRecord(step0_messages, step0_response),
            FakeRecord(step1_messages, step1_response),
        ]
        env_result = EnvRunResult(records=records, final_answer="答复", env_info={"query": "你好"})

        out = self.builder.build(env_result, query="你好")

        # summary 字段
        self.assertEqual(out.summary.num_steps, 2)
        self.assertEqual(out.summary.num_action_steps, 1)
        self.assertEqual(out.summary.num_parse_ok, 2)
        self.assertTrue(out.summary.has_final_step)
        self.assertFalse(out.summary.has_error)

        # response_text 应该包含 assistant 段 + obs 段（一次）
        # step0 assistant 段 = "ACT: search"
        # observation 段 = chat_template([obs_msg]) = "[TOOL]OBS_DATA[/TOOL]\n"
        # step1 assistant 段 = "FINAL: 答复"
        expected_response = (
            "ACT: search"
            + "[TOOL]OBS_DATA[/TOOL]\n"
            + "FINAL: 答复"
        )
        self.assertEqual(out.summary.response_text, expected_response)

        # token 数量 = 字符数（FakeTokenizer 一字符一 token）
        self.assertEqual(len(out.response_ids), len(expected_response))
        self.assertEqual(len(out.response_mask), len(expected_response))

        # prompt = chat_template(step0_messages, add_generation_prompt=True)
        expected_prompt = (
            "[SYSTEM]你是 ReAct[/SYSTEM]\n"
            "[USER]你好[/USER]\n"
            "[ASSISTANT]"
        )
        self.assertEqual(out.summary.prompt_text, expected_prompt)
        self.assertEqual(len(out.prompt_ids), len(expected_prompt))

    # -------- 测试 2：assistant 段全部 mask=1 -------------------------

    def test_assistant_mask_all_one(self):
        sys_msg = {"role": "system", "content": "你是 ReAct"}
        user_msg = {"role": "user", "content": "?"}
        step0_messages = [sys_msg, user_msg]
        step0_response = "ACT: a"
        obs_msg = {"role": "tool", "content": "O"}
        step1_messages = step0_messages + [
            {"role": "assistant", "content": step0_response},
            obs_msg,
        ]
        step1_response = "FINAL"

        records = [
            FakeRecord(step0_messages, step0_response),
            FakeRecord(step1_messages, step1_response),
        ]
        env_result = EnvRunResult(records=records, final_answer="x")

        out = self.builder.build(env_result, query="?")

        # 拼出来的 response：
        #   "ACT: a"   (0..6)   ← assistant 段 → mask=1
        #   "[TOOL]O[/TOOL]\n"  ← obs 段 → mask=0
        #   "FINAL"             ← assistant 段 → mask=1
        text = out.summary.response_text
        ranges_obs = []
        # 找到 obs 段位置
        obs_chunk = "[TOOL]O[/TOOL]\n"
        idx = text.index(obs_chunk)
        ranges_obs.append((idx, idx + len(obs_chunk)))

        for i, m in enumerate(out.response_mask):
            in_obs = any(s <= i < e for s, e in ranges_obs)
            if in_obs:
                self.assertEqual(m, 0, f"char[{i}]={text[i]!r} 在 obs 段，应 mask=0")
            else:
                self.assertEqual(m, 1, f"char[{i}]={text[i]!r} 在 assistant 段，应 mask=1")

    # -------- 测试 3：observation 段全部 mask=0 -----------------------

    def test_observation_mask_all_zero(self):
        """3 步对话，两段 observation 都必须 mask=0。"""
        sys_msg = {"role": "system", "content": "sys"}
        step0_messages = [sys_msg, {"role": "user", "content": "q"}]
        step0_response = "ACT: r1"
        step1_messages = step0_messages + [
            {"role": "assistant", "content": step0_response},
            {"role": "tool", "content": "OBS1"},
        ]
        step1_response = "ACT: r2"
        step2_messages = step1_messages + [
            {"role": "assistant", "content": step1_response},
            {"role": "tool", "content": "OBS2"},
        ]
        step2_response = "FINAL bye"

        records = [
            FakeRecord(step0_messages, step0_response),
            FakeRecord(step1_messages, step1_response),
            FakeRecord(step2_messages, step2_response),
        ]
        env_result = EnvRunResult(records=records, final_answer="bye")
        out = self.builder.build(env_result, query="q")

        text = out.summary.response_text

        # 两段 observation 都该是 [TOOL]OBSx[/TOOL]\n
        obs1 = "[TOOL]OBS1[/TOOL]\n"
        obs2 = "[TOOL]OBS2[/TOOL]\n"
        i1 = text.index(obs1)
        i2 = text.index(obs2)

        # obs1 的所有字符 mask 应该都 0
        for k in range(i1, i1 + len(obs1)):
            self.assertEqual(out.response_mask[k], 0, f"obs1 char[{k}]={text[k]!r} 必须 mask=0")
        # obs2 同理
        for k in range(i2, i2 + len(obs2)):
            self.assertEqual(out.response_mask[k], 0, f"obs2 char[{k}]={text[k]!r} 必须 mask=0")

        # assistant 段（包括最后一步的 FINAL bye）应该 mask=1
        final_start = i2 + len(obs2)
        for k in range(final_start, len(text)):
            self.assertEqual(out.response_mask[k], 1, f"final assistant char[{k}] 必须 mask=1")

        # 也检查 summary
        self.assertEqual(out.summary.num_steps, 3)
        self.assertEqual(out.summary.num_action_steps, 2)
        self.assertTrue(out.summary.has_final_step)
        self.assertAlmostEqual(out.summary.tool_call_success_rate(), 1.0)

    # -------- 边界：空 records ----------------------------------------

    def test_empty_records(self):
        env_result = EnvRunResult(records=[], final_answer="", env_info={"error": "x"})
        out = self.builder.build(env_result, query="q")
        self.assertEqual(out.summary.num_steps, 0)
        self.assertTrue(out.summary.has_error)
        self.assertEqual(out.prompt_ids, [])
        self.assertEqual(out.response_ids, [])
        self.assertEqual(out.response_mask, [])

    # -------- 边界：parse_ok=False 也要正确统计 -----------------------

    def test_parse_ok_stats(self):
        sys_msg = {"role": "system", "content": "s"}
        step0_messages = [sys_msg, {"role": "user", "content": "q"}]
        # 这步既不 ACT: 开头也不含 FINAL → parse_ok=False, is_final=False
        step0_response = "我不知道"
        step1_messages = step0_messages + [
            {"role": "assistant", "content": step0_response},
            {"role": "tool", "content": "O"},
        ]
        step1_response = "FINAL ok"

        records = [
            FakeRecord(step0_messages, step0_response),
            FakeRecord(step1_messages, step1_response),
        ]
        env_result = EnvRunResult(records=records, final_answer="ok")
        out = self.builder.build(env_result, query="q")

        self.assertEqual(out.summary.num_steps, 2)
        self.assertEqual(out.summary.num_action_steps, 1)
        self.assertEqual(out.summary.num_parse_ok, 1)  # 只有 step1 parse_ok
        self.assertAlmostEqual(out.summary.tool_call_success_rate(), 0.0)  # action step 0/1


if __name__ == "__main__":
    unittest.main()