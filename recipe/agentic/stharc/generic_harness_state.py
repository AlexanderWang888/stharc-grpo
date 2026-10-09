"""GenericHarnessState —— 进程内内存态 harness 沙箱，供 GenericBenchmarkEnv 使用。

与 :mod:`recipe.agentic.stharc.harness_state` 的 ``HarnessState`` 暴露完全一致的
公开 API（``create_h_init`` / ``clone_for_chain`` / ``to_ext_params_patch`` /
``provenance_registry`` / ``snapshot_for_log`` / ``cleanup``），可通过
``stharc.harness_state_module`` 配置项在 ``agent_loop.py`` 中整体替换，
不需要改 ``core.py`` / ``agent_loop.py`` 的其余逻辑（见 §5 解耦设计）。

与 ai2c 版 ``HarnessState`` 的关键差异：
- ai2c 版从生产 skill 目录 + memory.json copytree 出 H_init（需要跨进程文件锁保证
  幂等），因为生产环境本身有一个"已存在"的技能库/记忆库需要作为起点快照。
- 本模块面向的是公开 benchmark / 合成数据集（musique / textcraft /
  credit_diagnostic_* / skill_delete / memory_invalidate）：这些任务的 skill /
  memory 内容都是 rollout 过程中由模型自己 create / write 出来的，H_init 天然是
  "空快照"，不存在需要跨进程共享的起点状态，因此不需要文件系统或跨进程锁，
  一个进程内的 dict 即可。

沙箱数据结构（``get_sandbox(session_id)`` 返回值）::

    {
      "skills": {skill_id: content_str, ...},
      "memory": {key: {"value": str, "invalidated": bool}, ...},
    }

调用约定（由 ``GenericBenchmarkEnv`` 负责）：
    每次 ``env.run(..., ext_params=...)`` 时，``ext_params["_rl_session_id"]``
    （由 ``to_ext_params_patch`` 注入）指向本次 rollout 应该操作的沙箱；
    ``ext_params["_rl_frozen"]`` 为 True 时，写操作只返回 mock ack，不落到沙箱里
    （对齐 ai2c 版 frozen chain 语义）。
"""

from __future__ import annotations

import threading
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

__all__ = ["GenericHarnessState", "HarnessState", "get_sandbox", "SANDBOX_REGISTRY"]

_REGISTRY_LOCK = threading.Lock()

# session_id -> {"skills": {...}, "memory": {...}}（进程内存活，由 GenericBenchmarkEnv 读写）
SANDBOX_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _new_sandbox() -> Dict[str, Any]:
    return {"skills": {}, "memory": {}}


def get_sandbox(session_id: str) -> Dict[str, Any]:
    """按 session_id 取（或懒创建）沙箱 dict。GenericBenchmarkEnv 工具执行时调用。"""
    with _REGISTRY_LOCK:
        sb = SANDBOX_REGISTRY.get(session_id)
        if sb is None:
            sb = _new_sandbox()
            SANDBOX_REGISTRY[session_id] = sb
        return sb


def _drop_sandbox(session_id: str) -> None:
    with _REGISTRY_LOCK:
        SANDBOX_REGISTRY.pop(session_id, None)


@dataclass
class GenericHarnessState:
    """单条 chain 的内存态 harness 状态封装（HarnessState 的解耦版）。

    Attributes:
        group_id:             GRPO group 标识符
        chain_tag:             本 chain 的标签（"h_init" / "chain_0" / "chain_frozen"）
        session_id:            本状态对应的沙箱 key（SANDBOX_REGISTRY 的 key）
        provenance_registry:   object_id -> list[{source_task, action, mutation_type}]
        skill_dep_graph:       预留字段（对齐 HarnessState，当前数据集未使用）
    """

    group_id: str
    chain_tag: str
    session_id: str
    provenance_registry: Dict[str, List[dict]] = field(default_factory=dict)
    skill_dep_graph: Dict[str, List[str]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def create_h_init(
        cls,
        group_id: str,
        base_user_id: str = "default",
        **kwargs: Any,
    ) -> "GenericHarnessState":
        """创建（或复用）本 group 的 H_init 空快照。

        与 ai2c 版不同：不需要文件锁幂等初始化，因为空快照在任何进程里创建的
        结果都一样（都是空 dict），不存在"谁先初始化"的竞态问题。
        """
        session_id = f"generic_hinit_{group_id}"
        get_sandbox(session_id)  # 确保存在（幂等，不覆盖已有内容）
        return cls(group_id=group_id, chain_tag="h_init", session_id=session_id)

    def clone_for_chain(self, chain_tag: str) -> "GenericHarnessState":
        """为某条 chain 从当前状态 deepcopy 一份独立沙箱。

        用 uuid 后缀生成新 session_id，避免同 group 同 chain_tag 的并发调用
        （如同一 batch 里 chain_g 重复采样）互相覆盖沙箱内容。
        """
        new_session_id = f"generic_{self.group_id}_{chain_tag}_{uuid4().hex[:8]}"
        src = get_sandbox(self.session_id)
        with _REGISTRY_LOCK:
            SANDBOX_REGISTRY[new_session_id] = deepcopy(src)
        return GenericHarnessState(
            group_id=self.group_id,
            chain_tag=chain_tag,
            session_id=new_session_id,
            provenance_registry={},
            skill_dep_graph=deepcopy(self.skill_dep_graph),
        )

    # ------------------------------------------------------------------
    # ext_params 注入
    # ------------------------------------------------------------------

    def to_ext_params_patch(self, frozen: bool = False) -> dict:
        """返回要注入到 ext_params 的字段，让 GenericBenchmarkEnv.run() 找到本 chain 的沙箱。

        Args:
            frozen: True 时 GenericBenchmarkEnv 的写工具只返回 mock ack，不落到沙箱里
                （对齐 ai2c 版 frozen chain 语义，用于 run_frozen_chain 反事实基线）。
        """
        return {
            "_rl_session_id": self.session_id,
            "_rl_frozen": bool(frozen),
        }

    # ------------------------------------------------------------------
    # 日志快照（cleanup 前调用）
    # ------------------------------------------------------------------

    def snapshot_for_log(
        self,
        *,
        mutated_skill_ids: Optional[List[str]] = None,
        include_all_skill_contents: bool = False,
    ) -> dict:
        """序列化本 chain 沙箱状态，供 chain trace JSONL 落盘。"""
        sandbox = get_sandbox(self.session_id)
        skills: Dict[str, str] = sandbox.get("skills", {})
        memory: Dict[str, Any] = sandbox.get("memory", {})

        touch_ids = {str(s).strip() for s in (mutated_skill_ids or []) if str(s).strip()}
        for oid in self.provenance_registry:
            oid_s = str(oid).strip()
            if oid_s and not oid_s.startswith("mem::"):
                touch_ids.add(oid_s)

        ids_to_dump = set(skills.keys()) if include_all_skill_contents else touch_ids
        skills_payload = {sid: skills[sid] for sid in sorted(ids_to_dump) if sid in skills}

        return {
            "group_id": self.group_id,
            "chain_tag": self.chain_tag,
            "session_id": self.session_id,
            "skill_ids": sorted(skills.keys()),
            "skill_count": len(skills),
            "mutated_skill_ids": sorted(touch_ids),
            "skills": skills_payload,
            "include_all_skill_contents": bool(include_all_skill_contents),
            "memory": deepcopy(memory),
            "provenance_registry": {
                oid: [
                    {
                        "source_task": e.get("source_task"),
                        "mutation_type": e.get("mutation_type"),
                        "op_type": (e.get("action") or {}).get("op_type"),
                        "target_id": (e.get("action") or {}).get("target_id"),
                    }
                    for e in entries
                ]
                for oid, entries in self.provenance_registry.items()
            },
        }

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """释放本 chain 的沙箱（训练结束后调用）。"""
        _drop_sandbox(self.session_id)

    def __repr__(self) -> str:
        return (
            f"GenericHarnessState(group={self.group_id!r}, chain={self.chain_tag!r}, "
            f"session={self.session_id!r})"
        )


# agent_loop.py 通过 ``stharc.harness_state_module`` 配置项动态 import 本模块后，
# 固定用 ``getattr(module, "HarnessState")`` 取实现类（与 ai2c 版
# ``recipe.agentic.stharc.harness_state.HarnessState`` 保持同名调用约定，
# 见 agent_loop.py 里 STHARCAgentLoop.run() 的导入逻辑）——这里补一个别名。
HarnessState = GenericHarnessState
