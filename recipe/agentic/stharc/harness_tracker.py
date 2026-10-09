"""
HarnessTracker（research 侧实现）。

通过 ContextVar 注入；由 `ai2c_harness_patch` 在 ai2c `_execute_tool` 外包一层
调用 get_tracker() / record_*，不修改 ai2c 源码。
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

HARNESS_WRITE_OPS: frozenset = frozenset({"skill_manage", "memory"})
# load_skill：显式工具读；memory_inject：system prompt 被动注入（见 ai2c_harness_patch）；
# session_search / case_search：检索类读（object_id 为 _search_*，不与 memory 写对齐）。
HARNESS_READ_OPS: frozenset = frozenset(
    {"load_skill", "memory_inject", "session_search", "case_search"}
)

_tracker_var: ContextVar[Optional["HarnessTracker"]] = ContextVar(
    "_harness_tracker", default=None
)


@dataclass
class HarnessWriteEvent:
    step: int
    tool_name: str
    target_id: str
    args: Dict[str, Any]
    before_snapshot: str
    after_snapshot: str


@dataclass
class HarnessTracker:
    write_events: List[HarnessWriteEvent] = field(default_factory=list)
    read_set: Set[str] = field(default_factory=set)
    _step_counter: int = 0

    def record_write(
        self,
        tool_name: str,
        target_id: str,
        args: Dict[str, Any],
        before: str,
        after: str,
    ) -> None:
        self.write_events.append(
            HarnessWriteEvent(
                step=self._step_counter,
                tool_name=tool_name,
                target_id=str(target_id or ""),
                args=dict(args or {}),
                before_snapshot=before or "",
                after_snapshot=after or "",
            )
        )
        self._step_counter += 1

    def record_read(self, tool_name: str, target_id: str) -> None:
        if target_id:
            self.read_set.add(str(target_id))
        self._step_counter += 1


def set_tracker(tracker: HarnessTracker) -> Token:
    return _tracker_var.set(tracker)


def get_tracker() -> Optional[HarnessTracker]:
    return _tracker_var.get()


def clear_tracker(token: Token) -> None:
    _tracker_var.reset(token)


__all__ = [
    "HARNESS_READ_OPS",
    "HARNESS_WRITE_OPS",
    "HarnessTracker",
    "HarnessWriteEvent",
    "clear_tracker",
    "get_tracker",
    "set_tracker",
]
