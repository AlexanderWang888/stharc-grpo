"""ActionParser for ai_to_c_server_react ReAct planner.

解析逻辑与 ``ai_to_c_server_react.planner.react_reactor._parse_action`` 对齐，
确保 reward / TrajectoryBuilder 与业务 runtime 对同一 LLM 输出给出一致判定。

输出格式（见 ``_build_system_prompt``）：

    Thought: <用中文简述推理与下一步>
    Action: <JSON，形如 {"name":"工具名","arguments":{...}}>

并行调用：

    Action: {"parallel":true,"calls":[{"name":"工具A","arguments":{...}}, ...]}

终止（任意终止工具 → is_final=True）：

    Action: {"name":"finish","arguments":{"answer":"..."}}
    Action: {"name":"transfer-human","arguments":{...}}
    ...

或：

    Final Answer: <对用户的完整中文答复>

设计原则：
- 解析策略与 react_reactor 一致（平衡括号 → 裸 JSON → finish lenient）
- TERMINAL_TOOL_NAMES 中的 action 与 "Final Answer:" 都视为 is_final=True
- 解析失败时 raw_text 保留，便于 reward / 日志
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .base import ActionParser, ParsedAction

# 终止工具集：命中这些 action name 即视为轨迹完成（is_final=True）
TERMINAL_TOOL_NAMES: frozenset = frozenset({
    "finish",
    "transfer-human",
    "transfer-planner",
    "give_link_card",
})

_FINAL_ANSWER_RE = re.compile(r"Final\s*Answer\s*:\s*(.*)", re.DOTALL | re.IGNORECASE)


class AiToCReactActionParser(ActionParser):
    """ai_to_c_server_react ReAct 输出解析器（与 react_reactor._parse_action 对齐）。"""

    def parse(self, text: str) -> ParsedAction:
        if not isinstance(text, str) or not text.strip():
            return ParsedAction(parse_ok=False, raw_text=text or "")

        # 1. Final Answer 路径（无 Action JSON）
        m_fa = _FINAL_ANSWER_RE.search(text)
        if m_fa:
            answer = m_fa.group(1).strip()
            return ParsedAction(
                parsed_action={"name": "finish", "arguments": {"answer": answer}},
                parse_ok=True,
                is_final=True,
                raw_text=text,
            )

        # 2. Action JSON 路径（与 react_reactor._parse_action 相同三级策略）
        action_json = _parse_action_dict(text)
        if action_json is None:
            return ParsedAction(parse_ok=False, raw_text=text)

        is_parallel = bool(action_json.get("parallel")) and "calls" in action_json
        if is_parallel:
            name = "parallel"
            calls = action_json.get("calls") or []
            is_final = any(
                str(c.get("name", "")).strip() in TERMINAL_TOOL_NAMES
                for c in calls
                if isinstance(c, dict)
            )
        else:
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
# helpers — 与 react_reactor.py 中同名函数保持语义一致
# ---------------------------------------------------------------------------


def _parse_action_dict(text: str) -> Optional[Dict[str, Any]]:
    """从模型输出中提取 Action JSON dict（对齐 react_reactor._parse_action）。"""
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
    """从 s[start] 处的 `{` 起，按括号深度与字符串状态截取完整 JSON 对象子串。"""
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
    """在 `Action:` 之后用平衡括号提取完整 JSON。"""
    m = re.search(r"Action\s*:\s*\{", text, re.IGNORECASE)
    if not m:
        return None
    brace_at = m.end() - 1
    return _extract_balanced_json_object(text, brace_at)


def _extract_finish_answer_lenient(raw: str) -> Optional[str]:
    """finish Action JSON 在 json.loads 失败时的启发式 answer 提取。"""
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

    content: List[str] = []
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
