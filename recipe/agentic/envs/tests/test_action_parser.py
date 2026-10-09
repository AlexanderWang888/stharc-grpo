"""AiToCReactActionParser 单测。"""

from __future__ import annotations

import unittest

from recipe.agentic.envs.ai_to_c_action_parser import AiToCReactActionParser


class TestActionParser(unittest.TestCase):
    def setUp(self) -> None:
        self.p = AiToCReactActionParser()

    # ---------------- Action JSON 路径 ----------------

    def test_basic_action(self):
        text = (
            "Thought: 我先查一下知识库\n"
            'Action: {"name":"search_general_kb","arguments":{"query":"百万医疗"}}\n'
        )
        out = self.p.parse(text)
        self.assertTrue(out.parse_ok)
        self.assertFalse(out.is_final)
        self.assertEqual(out.parsed_action["name"], "search_general_kb")
        self.assertEqual(out.parsed_action["arguments"]["query"], "百万医疗")

    def test_finish_action_is_final(self):
        text = (
            "Thought: 已经够了\n"
            'Action: {"name":"finish","arguments":{"answer":"好的，您可以..."}}\n'
        )
        out = self.p.parse(text)
        self.assertTrue(out.parse_ok)
        self.assertTrue(out.is_final)
        self.assertEqual(out.parsed_action["name"], "finish")
        self.assertEqual(out.parsed_action["arguments"]["answer"], "好的，您可以...")

    def test_final_answer_path(self):
        text = "Thought: ok\nFinal Answer: 答案就是 42"
        out = self.p.parse(text)
        self.assertTrue(out.parse_ok)
        self.assertTrue(out.is_final)
        self.assertEqual(out.parsed_action["name"], "finish")
        self.assertEqual(out.parsed_action["arguments"]["answer"], "答案就是 42")

    def test_nested_json(self):
        text = (
            "Thought: 调用复杂工具\n"
            'Action: {"name":"skill_manage","arguments":{"action":"patch","old_string":"a","new_string":"b{c:1}"}}\n'
        )
        out = self.p.parse(text)
        self.assertTrue(out.parse_ok)
        self.assertEqual(out.parsed_action["name"], "skill_manage")
        self.assertEqual(out.parsed_action["arguments"]["action"], "patch")

    def test_action_no_trailing_newline(self):
        """Action 后没有 \\n，也应该解析成功（生成结束）"""
        text = 'Thought: x\nAction: {"name":"finish","arguments":{"answer":"end"}}'
        out = self.p.parse(text)
        self.assertTrue(out.parse_ok)
        self.assertTrue(out.is_final)

    def test_action_with_nested_braces_via_fallback(self):
        """嵌套花括号，但 _ACTION_RE 用非贪婪 + 单行匹配可能解析到不完整 JSON；
        测试兜底的平衡花括号匹配能拿到正确结果。"""
        text = 'Thought: a\nAction: {"name":"t","arguments":{"d":{"k":1}}}\n'
        out = self.p.parse(text)
        self.assertTrue(out.parse_ok)
        self.assertEqual(out.parsed_action["name"], "t")
        self.assertEqual(out.parsed_action["arguments"]["d"]["k"], 1)

    # ---------------- 解析失败路径 -----------------

    def test_empty_text(self):
        out = self.p.parse("")
        self.assertFalse(out.parse_ok)
        self.assertFalse(out.is_final)

    def test_no_action(self):
        out = self.p.parse("这只是普通文本，没有 Action 或 Final Answer")
        self.assertFalse(out.parse_ok)

    def test_bad_json(self):
        text = "Thought: x\nAction: {name:not_quoted}\n"
        out = self.p.parse(text)
        self.assertFalse(out.parse_ok)

    def test_action_without_name(self):
        text = 'Thought: x\nAction: {"arguments":{}}\n'
        out = self.p.parse(text)
        self.assertFalse(out.parse_ok)

    def test_multiline_finish_answer_lenient(self):
        """answer 含 literal 换行时，与 react_reactor level-3 lenient 对齐。"""
        text = (
            "Thought: 可以回答了\n"
            'Action: {"name":"finish","arguments":{"answer":"第一段\n\n第二段内容"}}'
        )
        out = self.p.parse(text)
        self.assertTrue(out.parse_ok)
        self.assertTrue(out.is_final)
        self.assertEqual(out.parsed_action["name"], "finish")
        self.assertEqual(out.parsed_action["arguments"]["answer"], "第一段\n\n第二段内容")

    def test_parallel_calls(self):
        text = (
            "Thought: 并行查\n"
            'Action: {"parallel":true,"calls":['
            '{"name":"search_general_kb","arguments":{"query":"a"}},'
            '{"name":"load_skill","arguments":{"skill_id":"x"}}'
            "]}\n"
        )
        out = self.p.parse(text)
        self.assertTrue(out.parse_ok)
        self.assertFalse(out.is_final)
        self.assertIn("calls", out.parsed_action)

    def test_nested_json_via_balanced_brackets(self):
        """跨行 Action JSON，平衡括号提取后 json.loads 成功。"""
        text = (
            "Thought: x\n"
            'Action: {"name":"t","arguments":{"d":{"k":1}}}\n'
        )
        out = self.p.parse(text)
        self.assertTrue(out.parse_ok)
        self.assertEqual(out.parsed_action["arguments"]["d"]["k"], 1)


if __name__ == "__main__":
    unittest.main()