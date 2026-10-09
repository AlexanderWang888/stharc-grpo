"""Hydra 入口：``recipe.agentic.stharc.entry.STHARCAgentLoop``.

与 ``agent_loop.py`` 等价（显式导出，兼容 Hydra _locate）。
"""

from __future__ import annotations

from recipe.agentic.agent_loop import STHARCAgentLoop

__all__ = ["STHARCAgentLoop"]
