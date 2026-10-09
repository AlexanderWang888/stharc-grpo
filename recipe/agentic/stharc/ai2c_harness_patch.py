"""Research-only：在不改 ai2c 源码的前提下，让 STHARC 沙箱 / HarnessTracker 生效。

通过幂等 monkeypatch（分两阶段，避免重依赖拖垮轻量 patch）：
  1) loader / manager / get_user_id —— 沙箱读写与 memory user
  2) ``_execute_tool`` —— HarnessTracker 记录 + ``_rl_frozen`` mock

由 ``rollout_with_tracking`` 在每次 task 前 ``bind_rl_context`` + ``apply_patches``。
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_patch_lock = threading.Lock()
_patched_core = False
_patched_execute = False

# 每条 chain / 每个 worker 线程各自的沙箱 root（并发安全）
_tls = threading.local()

# get_skill_library 按 root 分桶，避免原单例在多沙箱下互相踩
_lib_cache: Dict[str, Any] = {}
_lib_cache_lock = threading.Lock()


def _tls_skills_root() -> Optional[str]:
    root = getattr(_tls, "skills_root", None)
    return str(root) if root else None


def bind_rl_context(ext_params: Optional[Dict[str, Any]]) -> Optional[str]:
    """设置当前线程的 skills 沙箱 root，返回旧值（供 restore）。"""
    ep = ext_params or {}
    prev = getattr(_tls, "skills_root", None)
    raw = ep.get("_rl_skills_root")
    _tls.skills_root = str(raw) if raw else None
    return prev


def restore_rl_context(prev: Optional[str]) -> None:
    _tls.skills_root = prev


def _is_frozen(ext_params: Optional[Dict[str, Any]]) -> bool:
    v = (ext_params or {}).get("_rl_frozen")
    if isinstance(v, bool):
        return v
    return str(v or "").strip().lower() in ("1", "true", "yes")


def _target_id(tool_name: str, args: Dict[str, Any]) -> str:
    if tool_name == "skill_manage":
        return str(args.get("skill_id") or args.get("name") or "").strip()
    if tool_name == "memory":
        target = str(args.get("target", "memory")).strip().lower() or "memory"
        return f"memory:{target}"
    if tool_name == "load_skill":
        return str(args.get("skill_id") or "").strip()
    if tool_name in ("session_search", "case_search"):
        return f"_search_{tool_name}"
    return ""


def _skill_snapshot(skills_root: Optional[str], skill_id: str) -> str:
    sid = (skill_id or "").strip()
    if not sid or not skills_root:
        return ""
    path = Path(skills_root) / sid / "SKILL.md"
    try:
        return path.read_text(encoding="utf-8") if path.is_file() else ""
    except OSError:
        return ""


def _memory_snapshot(memory_store: Any, args: Dict[str, Any]) -> str:
    if memory_store is None:
        return ""
    target = str(args.get("target", "memory")).strip().lower() or "memory"
    try:
        if target == "user":
            entries = list(getattr(memory_store, "_user_entries", []) or [])
        else:
            entries = list(getattr(memory_store, "_memory_entries", []) or [])
        return "§".join(str(e) for e in entries)
    except Exception:
        return ""


def apply_ai2c_harness_patches() -> None:
    """幂等安装 patch。核心 patch 与 _execute_tool patch 可分开成功。"""
    global _patched_core, _patched_execute
    with _patch_lock:
        if not _patched_core:
            try:
                _install_core_patches()
                _patched_core = True
                logger.info("[stharc.ai2c_harness_patch] core patches installed")
            except Exception as exc:
                logger.warning("[stharc.ai2c_harness_patch] core install failed: %s", exc)
                raise
        if not _patched_execute:
            try:
                _install_execute_tool_patch()
                _patched_execute = True
                logger.info("[stharc.ai2c_harness_patch] _execute_tool patch installed")
            except Exception as exc:
                # 训练机通常能 import reactor；本地 py3.9 可能因无关 typing 失败
                logger.warning(
                    "[stharc.ai2c_harness_patch] _execute_tool patch deferred/failed: %s",
                    exc,
                )


def _install_core_patches() -> None:
    import ai_to_c_server_react.memory.store as mem_store
    import ai_to_c_server_react.skills.loader as loader
    import ai_to_c_server_react.skills.manager as manager

    # ── get_user_id ──────────────────────────────────────────────────────────
    _orig_get_user_id = mem_store.get_user_id

    def _patched_get_user_id(ext_params, session_id: str) -> str:
        ep = ext_params or {}
        rl_uid = str(ep.get("_rl_memory_user_id") or "").strip()
        if rl_uid:
            return rl_uid
        return _orig_get_user_id(ext_params, session_id)

    mem_store.get_user_id = _patched_get_user_id  # type: ignore[assignment]

    # ── format_for_system_prompt：被动注入视为 memory read ───────────────────
    # memory 没有独立 retrieve 工具；后续 task 通过 system prompt 注入消费
    # 先前写的 memory:{memory|user}。此处对齐 write 侧 object_id，供 prospective bonus。
    _orig_format_for_system_prompt = mem_store.MemoryStore.format_for_system_prompt

    def _patched_format_for_system_prompt(self) -> str:
        from recipe.agentic.stharc.harness_tracker import get_tracker

        result = _orig_format_for_system_prompt(self)
        tracker = get_tracker()
        if tracker is not None:
            # 与 _target_id("memory", ...) 约定一致：memory:memory / memory:user
            if getattr(self, "_memory_entries", None):
                tracker.record_read("memory_inject", "memory:memory")
            if getattr(self, "_user_entries", None):
                tracker.record_read("memory_inject", "memory:user")
        return result

    mem_store.MemoryStore.format_for_system_prompt = (  # type: ignore[assignment]
        _patched_format_for_system_prompt
    )

    # ── get_skill_library（按 root 分桶）─────────────────────────────────────
    _SkillLibrary = loader.SkillLibrary
    _default_skills_root = loader.default_skills_root
    _orig_clear_cache = loader.clear_cache

    def _patched_get_skill_library(root=None):
        override = _tls_skills_root()
        if root is None and override:
            root = override
        r = Path(root).resolve() if root else _default_skills_root().resolve()
        key = str(r)
        with _lib_cache_lock:
            lib = _lib_cache.get(key)
            if lib is None:
                lib = _SkillLibrary(r)
                _lib_cache[key] = lib
            return lib

    def _patched_clear_cache():
        with _lib_cache_lock:
            override = _tls_skills_root()
            if override:
                _lib_cache.pop(str(Path(override).resolve()), None)
            else:
                _lib_cache.clear()
        try:
            _orig_clear_cache()
        except Exception:
            pass

    loader.get_skill_library = _patched_get_skill_library  # type: ignore[assignment]
    loader.clear_cache = _patched_clear_cache  # type: ignore[assignment]

    # ── skill_manage：thread-local 覆盖 _SKILLS_DIR ───────────────────────────
    _orig_skills_dir = manager._SKILLS_DIR
    _orig_skill_manage = manager.skill_manage

    class _SkillsDirProxy:
        """让 ``_SKILLS_DIR / x`` 在有 TLS override 时指向沙箱。"""

        def _root(self) -> Path:
            override = _tls_skills_root()
            return Path(override).resolve() if override else Path(_orig_skills_dir)

        def __truediv__(self, other):
            return self._root() / other

        def __str__(self) -> str:
            return str(self._root())

        def __fspath__(self) -> str:
            return str(self._root())

        def __repr__(self) -> str:
            return f"_SkillsDirProxy({self._root()!r})"

        def resolve(self):
            return self._root().resolve()

        def exists(self) -> bool:
            return self._root().exists()

        def is_dir(self) -> bool:
            return self._root().is_dir()

        def __eq__(self, other) -> bool:
            try:
                return self._root() == Path(other)
            except Exception:
                return False

    manager._SKILLS_DIR = _SkillsDirProxy()  # type: ignore[assignment]

    def _patched_skill_manage(*args, **kwargs):
        explicit = kwargs.pop("skills_root", None)
        prev = getattr(_tls, "skills_root", None)
        if explicit:
            _tls.skills_root = str(explicit)
        try:
            return _orig_skill_manage(*args, **kwargs)
        finally:
            if explicit:
                _tls.skills_root = prev

    manager.skill_manage = _patched_skill_manage  # type: ignore[assignment]

    # 若 react_reactor 已加载，同步替换其 from-import 的 get_skill_library
    try:
        import sys

        reactor = sys.modules.get("ai_to_c_server_react.planner.react_reactor")
        if reactor is not None and hasattr(reactor, "get_skill_library"):
            reactor.get_skill_library = _patched_get_skill_library  # type: ignore[attr-defined]
    except Exception:
        pass


def _install_execute_tool_patch() -> None:
    import ai_to_c_server_react.planner.react_reactor as reactor
    import ai_to_c_server_react.skills.loader as loader

    from recipe.agentic.stharc.harness_tracker import (
        HARNESS_READ_OPS,
        HARNESS_WRITE_OPS,
        get_tracker,
    )

    # 同步 from-import 的 get_skill_library
    if hasattr(reactor, "get_skill_library"):
        reactor.get_skill_library = loader.get_skill_library  # type: ignore[attr-defined]

    _orig_execute_tool = reactor._execute_tool

    def _patched_execute_tool(
        name: str,
        arguments: Dict[str, Any],
        *,
        session: Any,
        gateway: Any,
        ext_params: Dict[str, str],
        user_query: str,
        planner_step: int = -1,
        memory_store: Optional[Any] = None,
        session_history: Optional[Any] = None,
        case_store: Optional[Any] = None,
    ) -> Tuple[str, Optional[str]]:
        n = (name or "").strip().lower()
        args = arguments if isinstance(arguments, dict) else {}
        ep = ext_params or {}

        prev_root = getattr(_tls, "skills_root", None)
        raw_root = ep.get("_rl_skills_root")
        if raw_root:
            _tls.skills_root = str(raw_root)
        skills_root = _tls_skills_root()

        tracker = get_tracker()
        try:
            if tracker is not None and n in HARNESS_READ_OPS:
                tid = _target_id(n, args)
                if tid:
                    tracker.record_read(n, tid)

            if n in HARNESS_WRITE_OPS and _is_frozen(ep):
                return (
                    json.dumps(
                        {
                            "success": True,
                            "message": f"frozen mock ack for {n}",
                            "frozen": True,
                            "target_id": _target_id(n, args),
                        },
                        ensure_ascii=False,
                    ),
                    None,
                )

            before = ""
            if tracker is not None and n in HARNESS_WRITE_OPS:
                if n == "skill_manage":
                    before = _skill_snapshot(skills_root, str(args.get("skill_id") or ""))
                elif n == "memory":
                    before = _memory_snapshot(memory_store, args)

            obs, final = _orig_execute_tool(
                name,
                arguments,
                session=session,
                gateway=gateway,
                ext_params=ext_params,
                user_query=user_query,
                planner_step=planner_step,
                memory_store=memory_store,
                session_history=session_history,
                case_store=case_store,
            )

            if tracker is not None and n in HARNESS_WRITE_OPS:
                if n == "skill_manage":
                    after = _skill_snapshot(skills_root, str(args.get("skill_id") or ""))
                elif n == "memory":
                    after = _memory_snapshot(memory_store, args)
                else:
                    after = ""
                tracker.record_write(n, _target_id(n, args), args, before, after)

            return obs, final
        finally:
            _tls.skills_root = prev_root

    reactor._execute_tool = _patched_execute_tool  # type: ignore[assignment]
