"""ActionParser for GenericBenchmarkEnv 的通用 ReAct 输出格式。

与 ai_to_c_action_parser.py 的解析策略保持一致风格（平衡括号 → 裸 JSON → finish
lenient），但完全独立实现，不依赖任何 ai2c 业务模块，专供解耦的公开数据集 env 使用。

输出格式::

    Thought: <简述推理与下一步>
    Action: {"name":"<tool_name>","arguments":{...}}

终止（is_final=True）::

    Action: {"name":"finish","arguments":{"answer":"..."}}

或::

    Final Answer: <最终答案>
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from .base import ActionParser, ParsedAction

TERMINAL_TOOL_NAMES: frozenset = frozenset({"finish"})

_FINAL_ANSWER_RE = re.compile(r"Final\s*Answer\s*:\s*(.*)", re.DOTALL | re.IGNORECASE)


class GenericReactActionParser(ActionParser):
    """GenericBenchmarkEnv ReAct 输出解析器（业务无关，独立实现）。"""

    def parse(self, text: str) -> ParsedAction:
        if not isinstance(text, str) or not text.strip():
            return ParsedAction(parse_ok=False, raw_text=text or "")

        m_fa = _FINAL_ANSWER_RE.search(text)
        if m_fa:
            answer = m_fa.group(1).strip()
            return ParsedAction(
                parsed_action={"name": "finish", "arguments": {"answer": answer}},
                parse_ok=True,
                is_final=True,
                raw_text=text,
            )

        action_json = _parse_action_dict(text)
        if action_json is None:
            return ParsedAction(parse_ok=False, raw_text=text)

        name = str(action_json.get("name", "")).strip()
        if not name:
            return ParsedAction(parse_ok=False, raw_text=text)
        is_final = name in TERMINAL_TOOL_NAMES

        return ParsedAction(
            parsed_action=action_json,
            parse_ok=True,
            is_final=is_final,
            raw_text=text,
        )


# ---------------------------------------------------------------------------
# helpers（与 ai_to_c_action_parser.py 同构但独立维护，避免跨业务耦合）
# ---------------------------------------------------------------------------


def _parse_action_dict(text: str) -> Optional[Dict[str, Any]]:
    raw_json = _extract_action_json_after_colon(text)
    if raw_json is None:
        brace = re.search(r"\{[\s\S]*\}", text)
        if brace:
            try:
                obj = json.loads(brace.group(0))
                if isinstance(obj, dict) and obj.get("name"):
                    return obj
            except json.JSONDecodeError:
                pass
        return None
    try:
        obj = json.loads(raw_json)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    answer = _extract_finish_answer_lenient(text)
    if answer is not None:
        return {"name": "finish", "arguments": {"answer": answer}}
    return None


def _extract_balanced_json_object(s: str, start: int) -> Optional[str]:
    if start < 0 or start >= len(s) or s[start] != "{":
        return None
    depth = 0
    i = start
    n = len(s)
    in_str = False
    str_esc = False
    quote = ""
    while i < n:
        c = s[i]
        if in_str:
            if str_esc:
                str_esc = False
            elif c == "\\":
                str_esc = True
            elif c == quote:
                in_str = False
                quote = ""
        else:
            if c in ('"', "'"):
                in_str = True
                quote = c
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
        i += 1
    return None


def _extract_action_json_after_colon(text: str) -> Optional[str]:
    m = re.search(r"Action\s*:\s*\{", text, re.IGNORECASE)
    if not m:
        return None
    brace_at = m.end() - 1
    return _extract_balanced_json_object(text, brace_at)


def _extract_finish_answer_lenient(raw: str) -> Optional[str]:
    if '"finish"' not in raw:
        return None
    m_action = re.search(r"Action\s*:\s*(\{)", raw, re.IGNORECASE)
    if not m_action:
        return None
    raw_action = raw[m_action.start(1) :]

    m_ans = re.search(r'"answer"\s*:\s*"', raw_action)
    if not m_ans:
        return None
    start = m_ans.end()
    suffix = raw_action[start:]

    m_close: Optional[re.Match[str]] = None
    for m in re.finditer(r'"\s*\}\s*\}', suffix):
        m_close = m
    if m_close is None:
        return None

    raw_content = suffix[: m_close.start()]

    content = []
    i = 0
    while i < len(raw_content):
        c = raw_content[i]
        if c == "\\" and i + 1 < len(raw_content):
            nc = raw_content[i + 1]
            if nc == "n":
                content.append("\n")
            elif nc == "t":
                content.append("\t")
            elif nc == "r":
                content.append("\r")
            elif nc in ('"', "\\"):
                content.append(nc)
            else:
                content.append(c)
            i += 2
            continue
        content.append(c)
        i += 1

    result = "".join(content).strip()
    return result if result else None
