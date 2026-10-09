"""Hydra 兼容入口：``recipe.agentic.stharc.agent_loop.STHARCAgentLoop``.

与 ``entry.py`` 等价；保留此模块名以兼容未同步的旧 yaml / 缓存配置。
"""

from __future__ import annotations

from recipe.agentic.agent_loop import STHARCAgentLoop

__all__ = ["STHARCAgentLoop"]
