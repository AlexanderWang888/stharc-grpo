"""
HarnessState：STHARC 训练中每条 chain 独占的 harness 状态封装。

提供 snapshot（用于产生 H_init）和 clone_for_chain（用于每条 chain 独立起点）。

目录结构（默认 /tmp/rl_stharc/；训练脚本可经 STHARC_SANDBOX_DIR 指到大盘）：
    group_{group_id}/
      h_init/
        skills/          ← 从生产 skill 目录 copytree
      chain_0/
        skills/          ← 从 h_init/skills copytree
      chain_1/
        skills/
      chain_frozen/
        skills/
"""

from __future__ import annotations

import fcntl
import os
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ── 路径常量 ──────────────────────────────────────────────────────────────────
# parents: stharc(0) → agentic(1) → recipe(2) → verl_research(3) → research(4)
#          → ai2c_workspace(5)/ai2c/  （与主线共用业务代码，不再使用 ai2c_research）
def _default_production_skills_root() -> Path:
    here = Path(__file__).resolve()
    env = os.environ.get("AI2C_REPO_PATH", "").strip()
    if env:
        return Path(env) / "ai_to_c_server_react" / "skills" / "library"
    try:
        ws_ai2c = here.parents[5] / "ai2c" / "ai_to_c_server_react" / "skills" / "library"
        if ws_ai2c.is_dir():
            return ws_ai2c
    except IndexError:
        pass
    try:
        # 旧布局兼容（research 旁挂 ai2c_research）
        legacy = here.parents[4] / "ai2c_research" / "ai_to_c_server_react" / "skills" / "library"
        return legacy
    except IndexError:
        return Path("/tmp/rl_stharc/_missing_production_skills")


def _rl_sandbox_base() -> Path:
    """优先 STHARC_SANDBOX_DIR，其次 TMPDIR/rl_stharc，最后 /tmp/rl_stharc。"""
    env = os.environ.get("STHARC_SANDBOX_DIR", "").strip()
    if env:
        return Path(env)
    return Path(os.environ.get("TMPDIR") or "/tmp") / "rl_stharc"


# 延迟求值：避免模块 import 阶段因路径异常导致 HarnessState 类尚未定义
# （表现为 cannot import name 'HarnessState' 的半初始化模块）
_DEFAULT_PRODUCTION_SKILLS_ROOT: Path | None = None

RL_SANDBOX_BASE = _rl_sandbox_base()

__all__ = ["HarnessState", "RL_SANDBOX_BASE"]


def _production_skills_root() -> Path:
    global _DEFAULT_PRODUCTION_SKILLS_ROOT
    if _DEFAULT_PRODUCTION_SKILLS_ROOT is None:
        _DEFAULT_PRODUCTION_SKILLS_ROOT = _default_production_skills_root()
    return _DEFAULT_PRODUCTION_SKILLS_ROOT


@dataclass
class HarnessState:
    """单条 chain 的 harness 状态封装。

    Attributes:
        group_id:             GRPO group 标识符（字符串）
        chain_tag:            本 chain 的标签（如 "h_init", "chain_0", "chain_frozen"）
        skills_root:          本 chain 的 skill 沙箱目录
        memory_user_id:       本 chain 的专用 memory user_id
        provenance_registry:  object_id → list[{source_task, action, mutation_type}]
        skill_dep_graph:      技能依赖图（阶段二使用）
    """

    group_id: str
    chain_tag: str
    skills_root: Path
    memory_user_id: str
    provenance_registry: Dict[str, List[dict]] = field(default_factory=dict)
    skill_dep_graph: Dict[str, List[str]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def create_h_init(
        cls,
        group_id: str,
        base_user_id: str,
        source_skills_root: Optional[Path] = None,
    ) -> "HarnessState":
        """从生产 skill 目录创建 H_init 快照（每 group 调用一次）。

        多个 worker 并发调用时通过文件锁保证只有一个真正执行初始化，
        其余等待完成后直接使用已有结果（幂等）。

        Args:
            group_id:           GRPO group 标识
            base_user_id:       生产侧 user_id（用于复制 memory.json）
            source_skills_root: 生产 skill 库目录，None 时使用默认路径
        """
        src = (source_skills_root or _production_skills_root()).resolve()

        sandbox_root = RL_SANDBOX_BASE / f"group_{group_id}" / "h_init"
        sandbox_root.mkdir(parents=True, exist_ok=True)
        skills_dest = sandbox_root / "skills"
        h_init_user_id = f"{base_user_id}_stharc_hinit_{group_id}"

        # ── 文件锁：同 group 的 G 个 worker 并发调用时只初始化一次 ──────────
        lock_path = RL_SANDBOX_BASE / f"group_{group_id}.lock"
        with open(lock_path, "w") as _lf:
            fcntl.flock(_lf.fileno(), fcntl.LOCK_EX)
            try:
                # 拿到锁后重新检查：若已由其他 worker 初始化完成，直接复用
                if not skills_dest.exists():
                    if src.exists():
                        shutil.copytree(
                            src,
                            skills_dest,
                            dirs_exist_ok=True,
                            ignore_dangling_symlinks=True,
                        )
                    else:
                        skills_dest.mkdir(parents=True, exist_ok=True)
                    _copy_memory(base_user_id, h_init_user_id)
                    # H_init 重新创建 ⇒ 新一轮 rollout；清掉跨 step 残留的 frozen 缓存
                    _invalidate_frozen_rewards_cache(group_id)
            finally:
                fcntl.flock(_lf.fileno(), fcntl.LOCK_UN)

        return cls(
            group_id=group_id,
            chain_tag="h_init",
            skills_root=skills_dest,
            memory_user_id=h_init_user_id,
        )

    def clone_for_chain(self, chain_tag: str) -> "HarnessState":
        """为某条 chain 从当前状态 deepcopy 一份独立沙箱。

        通过 per-(group_id, chain_tag) 文件锁保证原子性：
        即使多个 worker 意外持有相同 (group_id, chain_tag)（如批次重复采样），
        也不会出现 rmtree 与 copytree 并发导致的 ENOENT 竞态。

        Args:
            chain_tag: 目标 chain 的标签（如 "chain_0", "chain_frozen"）
        """
        dest_root = RL_SANDBOX_BASE / f"group_{self.group_id}" / chain_tag / "skills"
        dest_root.parent.mkdir(parents=True, exist_ok=True)

        lock_path = RL_SANDBOX_BASE / f"group_{self.group_id}_{chain_tag}.lock"
        with open(lock_path, "w") as _lf:
            fcntl.flock(_lf.fileno(), fcntl.LOCK_EX)
            try:
                # 先清理旧内容（有则删），再复制
                if dest_root.exists():
                    shutil.rmtree(dest_root)

                if self.skills_root.exists():
                    shutil.copytree(
                        self.skills_root,
                        dest_root,
                        dirs_exist_ok=True,
                        ignore_dangling_symlinks=True,
                    )
                else:
                    dest_root.mkdir(parents=True, exist_ok=True)
            finally:
                fcntl.flock(_lf.fileno(), fcntl.LOCK_UN)

        # 构造新 user_id：替换 chain_tag 部分
        new_user_id = _derive_chain_user_id(self.memory_user_id, self.chain_tag, chain_tag)
        _copy_memory(self.memory_user_id, new_user_id)

        return HarnessState(
            group_id=self.group_id,
            chain_tag=chain_tag,
            skills_root=dest_root,
            memory_user_id=new_user_id,
            provenance_registry={},
            skill_dep_graph=deepcopy(self.skill_dep_graph),
        )

    # ------------------------------------------------------------------
    # ext_params 注入
    # ------------------------------------------------------------------

    def to_ext_params_patch(self, frozen: bool = False) -> dict:
        """返回要注入到 ext_params 的字段，让 chat_stream 使用本 chain 的沙箱。

        Args:
            frozen: True 时让 ai2c 侧把 harness 写操作改为 mock ack（不实际写盘）。
        """
        return {
            "_rl_skills_root":    str(self.skills_root),
            "_rl_memory_user_id": self.memory_user_id,
            "_rl_frozen":         frozen,
            "SKILL_MANAGE_ENABLED": True,
            "MEMORY_TOOL_ENABLED":  True,
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
        """序列化本 chain 沙箱状态，供 chain trace JSONL 落盘。

        默认不把整库 skill 全文写入（生产库可能很大），而是：
          - skill_ids：沙箱内全部 skill 目录名（清单）
          - skills：本 chain 改过的 skill 的最终 SKILL.md（+ 可选全量）
          - memory：最终 memory.json（解析后的对象；解析失败则存 raw 字符串）

        Args:
            mutated_skill_ids: 本 chain 通过 skill_manage 触及的 skill_id 列表。
            include_all_skill_contents: True 时把所有 skill 的 SKILL.md 写入
                ``skills``（体积大，仅调试用；可用环境变量
                ``STHARC_LOG_FULL_SKILLS=1`` 打开）。
        """
        import os as _os

        if not include_all_skill_contents:
            include_all_skill_contents = _os.environ.get(
                "STHARC_LOG_FULL_SKILLS", ""
            ).strip().lower() in ("1", "true", "yes")

        skill_ids = _list_skill_ids(self.skills_root)
        touch_ids = {
            str(s).strip()
            for s in (mutated_skill_ids or [])
            if str(s).strip()
        }
        # provenance 里也可能有 skill object_id（无 skill_manage 前缀）
        for oid in self.provenance_registry:
            oid_s = str(oid).strip()
            if oid_s and not oid_s.startswith("memory:"):
                touch_ids.add(oid_s)

        skills_payload: Dict[str, str] = {}
        ids_to_dump = set(skill_ids) if include_all_skill_contents else touch_ids
        for sid in sorted(ids_to_dump):
            content = _read_skill_md(self.skills_root, sid)
            if content is not None:
                skills_payload[sid] = content

        memory_payload = _read_memory_payload(self.memory_user_id)

        return {
            "group_id": self.group_id,
            "chain_tag": self.chain_tag,
            "skills_root": str(self.skills_root),
            "memory_user_id": self.memory_user_id,
            "skill_ids": skill_ids,
            "skill_count": len(skill_ids),
            "mutated_skill_ids": sorted(touch_ids),
            "skills": skills_payload,
            "include_all_skill_contents": include_all_skill_contents,
            "memory": memory_payload,
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
        """清理本 chain 的沙箱临时目录（训练结束后调用）。"""
        parent = RL_SANDBOX_BASE / f"group_{self.group_id}" / self.chain_tag
        if parent.exists():
            try:
                shutil.rmtree(parent)
            except Exception:
                pass

    def __repr__(self) -> str:
        return (
            f"HarnessState(group={self.group_id!r}, chain={self.chain_tag!r}, "
            f"skills={self.skills_root}, user_id={self.memory_user_id!r})"
        )


# ── 内部辅助函数 ──────────────────────────────────────────────────────────────

def _invalidate_frozen_rewards_cache(group_id: str) -> None:
    """删除 group 下的 frozen_rewards 缓存（H_init 重建时调用，避免跨 step 复用）。"""
    frozen_path = RL_SANDBOX_BASE / f"group_{group_id}" / "frozen_rewards.json"
    try:
        if frozen_path.exists():
            frozen_path.unlink()
    except OSError:
        pass


def _resolve_memory_dir() -> Path:
    """解析 MEMORY_DIR（优先 memory.config，与运行时 MemoryStore 一致）。"""
    try:
        import importlib
        if importlib.util.find_spec("ai_to_c_server_react.memory.config") is not None:
            import ai_to_c_server_react.memory.config as mem_cfg
            return Path(getattr(mem_cfg, "MEMORY_DIR", "/tmp/ai_to_c_memory"))
    except Exception:
        pass
    try:
        import importlib
        if importlib.util.find_spec("ai_to_c_server_react.config") is not None:
            import ai_to_c_server_react.config as _cfg
            return Path(getattr(_cfg, "MEMORY_DIR", "/tmp/ai_to_c_memory"))
    except Exception:
        pass
    return Path("/tmp/ai_to_c_memory")


def _list_skill_ids(skills_root: Path) -> List[str]:
    """列出沙箱 skills 目录下的 skill_id（子目录名）。"""
    root = Path(skills_root)
    if not root.is_dir():
        return []
    ids: List[str] = []
    try:
        for p in sorted(root.iterdir()):
            if p.is_dir() and not p.name.startswith("."):
                ids.append(p.name)
    except OSError:
        pass
    return ids


def _read_skill_md(skills_root: Path, skill_id: str) -> Optional[str]:
    """读取 skill 的 SKILL.md；不存在返回 None。"""
    sid = (skill_id or "").strip()
    if not sid:
        return None
    path = Path(skills_root) / sid / "SKILL.md"
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    except OSError:
        return None
    return None


def _read_memory_payload(memory_user_id: str) -> Optional[dict | str]:
    """读取最终 memory.json；优先返回解析后的 dict，失败则返回 raw 文本。"""
    import json as _json

    uid = (memory_user_id or "").strip()
    if not uid:
        return None
    path = _resolve_memory_dir() / uid / "memory.json"
    try:
        if not path.is_file():
            return None
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return _json.loads(raw)
    except (ValueError, TypeError):
        return raw


def _copy_memory(src_user_id: str, dst_user_id: str) -> None:
    """把 src_user_id 的 memory.json 复制给 dst_user_id（如源不存在则跳过）。"""
    memory_dir = _resolve_memory_dir()

    src = memory_dir / src_user_id / "memory.json"
    dst_dir = memory_dir / dst_user_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst_dir / "memory.json")


def _derive_chain_user_id(base_user_id: str, old_tag: str, new_tag: str) -> str:
    """从父 HarnessState 的 memory_user_id 推导新 chain 的 user_id。

    策略：若 old_tag 出现在 base_user_id 中则替换，否则追加 _new_tag。
    """
    if old_tag in base_user_id:
        return base_user_id.replace(old_tag, new_tag, 1)
    # h_init 特殊处理
    if "hinit" in base_user_id:
        return base_user_id.replace("hinit", new_tag.replace("_", ""), 1)
    return f"{base_user_id}_{new_tag}"
