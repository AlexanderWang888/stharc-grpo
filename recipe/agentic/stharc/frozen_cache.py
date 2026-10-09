"""STHARC frozen_rewards 磁盘缓存：同 step 内共享，跨 step 失效。"""

from __future__ import annotations

from typing import Any, List, Optional

# v2 — 仅 rewards，跨 step 不失效（已弃用）
# v3 — 含 global_step；同 step 内 G 条 chain 共享，跨 step 必须重算
FROZEN_REWARDS_SCHEMA = "stharc_frozen_v3"


def load_frozen_rewards_cache(
    raw: Any, *, global_step: int, expected_len: int
) -> Optional[List[float]]:
    """解析磁盘缓存；仅接受 v3 + 匹配 global_step，拒绝跨 step / 旧 schema。"""
    if not isinstance(raw, dict) or raw.get("schema") != FROZEN_REWARDS_SCHEMA:
        return None
    try:
        cached_step = int(raw.get("global_step", -999999))
    except (TypeError, ValueError):
        return None
    if cached_step != int(global_step):
        return None
    rewards = raw.get("rewards")
    if (
        isinstance(rewards, list)
        and len(rewards) == expected_len
        and all(isinstance(x, (int, float)) for x in rewards)
    ):
        return [float(x) for x in rewards]
    return None


def build_frozen_rewards_payload(rewards: List[float], global_step: int) -> dict:
    return {
        "schema": FROZEN_REWARDS_SCHEMA,
        "global_step": int(global_step),
        "rewards": list(rewards),
    }
