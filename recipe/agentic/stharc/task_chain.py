"""
task_chain.py — task chain 构建工具。

提供三种策略把单任务 query 列表组装成 task_chain（K 个任务的有序列表）：

  策略 1（build_task_chains_random）：
      随机 K 分组，最快上手，用于阶段一流程验证。

  策略 2（build_task_chains_by_skill_dep）：
      按 skill 依赖图排序，相关任务分在同一 chain，信号密度高。
      阶段二使用。

  策略 3（build_task_chains_from_traces）：
      从已有轨迹数据（filtered_traces.jsonl）中提取真实 task_chain。
      每条 trace 内的多轮对话作为一条 chain。

输出格式（parquet record list，供 prepare_dataset_stharc 使用）：
    [
      {
        "uid":        "chain_00001",   # G 条 chain 共享同一 uid → GRPO 同一 group
        "chain_id":   "chain_00001",
        "agent_name": "stharc",
        "task_chain": '[{"raw_prompt": ..., "ext_params": {...}}, ...]',  # JSON 字符串
      },
      ...  # 同一 chain_id 重复 G 次
    ]
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 策略 1：随机 K 分组
# ═══════════════════════════════════════════════════════════════════════════════

def build_task_chains_random(
    queries: List[Dict[str, Any]],
    K: int = 3,
    G: int = 4,
    seed: Optional[int] = 42,
) -> List[Dict[str, Any]]:
    """随机把 queries 分成 len(queries)/K 条 chain，每条 chain 重复 G 次。

    兼容两种输入格式：
      - 2w_query.jsonl：含 query / history / ext_params 字段
      - filtered_traces.jsonl：含 query / messages 字段（history 从 messages 提取）

    Args:
        queries: 原始 query 列表
        K:       chain 长度（每条 chain 包含 K 个任务）
        G:       每条 chain 并行数（GRPO group size）
        seed:    随机种子（None 表示不固定）

    Returns:
        parquet record list（uid 相同的 G 条为一个 GRPO group）
    """
    _validate_sizes(K=K, G=G)
    if seed is not None:
        random.seed(seed)
    shuffled = list(queries)
    random.shuffle(shuffled)

    chains = [shuffled[i : i + K] for i in range(0, len(shuffled) - K + 1, K)]
    records: List[Dict[str, Any]] = []

    for cid, chain_tasks in enumerate(chains):
        chain_id = f"chain_{cid:05d}"
        task_list = [_make_task_entry(q) for q in chain_tasks]
        for g in range(G):
            records.append(
                {
                    "uid":        chain_id,
                    "chain_id":   chain_id,
                    "chain_g":    g,
                    "agent_name": "stharc",
                    "task_chain": json.dumps(task_list, ensure_ascii=False),
                }
            )

    return records


# ═══════════════════════════════════════════════════════════════════════════════
# 策略 2：按 skill 依赖分组（阶段二）
# ═══════════════════════════════════════════════════════════════════════════════

def build_task_chains_by_skill_dep(
    queries: List[Dict[str, Any]],
    skill_dep_graph: Dict[str, List[str]],
    K: int = 5,
    G: int = 4,
    seed: Optional[int] = 42,
) -> List[Dict[str, Any]]:
    """按 skill 依赖图将相关任务分组到同一 chain。

    策略：
      1. 先把有明确 skill 关联的任务（query.skill_tag）按 skill 分组
      2. 每组内按 skill_dep_graph 拓扑排序（依赖 skill 先执行）
      3. 截断/填充到 K 个任务
      4. 无 skill 关联的任务随机填充剩余 chain

    注意：skill_groups 的迭代顺序影响上游去重效果。建议上游 skill 的数据先插入
    queries 列表（即先处理依赖方），确保上游 queries 在被下游借用前未被分配到
    自身 chain，从而让上游 queries 优先作为下游 chain 的前置上下文出现。

    Args:
        queries:        原始 query 列表（可含 skill_tag 字段）
        skill_dep_graph: {skill_id: [依赖的上游 skill_id]}
        K:              chain 长度
        G:              GRPO group size
        seed:           随机种子

    Returns:
        parquet record list
    """
    _validate_sizes(K=K, G=G)
    if seed is not None:
        random.seed(seed)

    # 按 skill_tag 分组
    skill_groups: Dict[str, List[Dict[str, Any]]] = {}
    ungrouped: List[Dict[str, Any]] = []
    for q in queries:
        tag = q.get("skill_tag") or q.get("skill_id") or ""
        if tag:
            skill_groups.setdefault(tag, []).append(q)
        else:
            ungrouped.append(q)

    chains: List[List[Dict[str, Any]]] = []

    # 全局已分配集合（object id）：保证每个 query 只出现在一条 chain 中，
    # 避免上游 skill queries 既作为下游 chain 的上下文前缀、又出现在其自身 chain 里。
    _assigned_ids: set = set()

    # 对有 skill 分组的任务：按依赖顺序排列
    for skill_id, group in skill_groups.items():
        # 把依赖 skill_id 的上游 skill 的任务排到前面（跨 skill_groups 查找上游任务），
        # 本 skill 自身的任务排在后面，使 chain 内先执行上游 skill 再执行当前 skill。
        # 只使用尚未被分配到其他 chain 的上游 queries，避免同一 query 重复出现。
        upstream_ids: List[str] = skill_dep_graph.get(skill_id, [])
        upstream_queries: List[Dict[str, Any]] = []
        for uid in upstream_ids:
            for q in skill_groups.get(uid, []):
                if id(q) not in _assigned_ids:
                    upstream_queries.append(q)
        # 本 skill 自身 queries 同样只取尚未分配的
        own_queries = [q for q in group if id(q) not in _assigned_ids]
        sorted_group = upstream_queries + own_queries
        # 切成若干条长度为 K 的 chain
        for i in range(0, max(1, len(sorted_group) - K + 1), K):
            chunk = sorted_group[i : i + K]
            if len(chunk) < K:
                # 从 ungrouped 里补齐，并移除已使用的条目避免重复
                need = K - len(chunk)
                fill = [q for q in ungrouped if id(q) not in _assigned_ids][:need]
                for q in fill:
                    ungrouped.remove(q)
                chunk = chunk + fill
            if len(chunk) == K:
                chains.append(chunk)
                for q in chunk:
                    _assigned_ids.add(id(q))
            else:
                logger.warning(
                    "build_task_chains_by_skill_dep: skill=%s 生成了长度 %d（< K=%d）的 chunk，已跳过",
                    skill_id, len(chunk), K,
                )

    # 剩余 ungrouped 随机分组（过滤已被 skill-dep chain 占用的条目）
    remaining_ungrouped = [q for q in ungrouped if id(q) not in _assigned_ids]
    random.shuffle(remaining_ungrouped)
    for i in range(0, len(remaining_ungrouped) - K + 1, K):
        chains.append(remaining_ungrouped[i : i + K])

    records: List[Dict[str, Any]] = []
    for cid, chain_tasks in enumerate(chains):
        chain_id = f"chain_{cid:05d}"
        task_list = [_make_task_entry(q) for q in chain_tasks]
        for g in range(G):
            records.append(
                {
                    "uid":        chain_id,
                    "chain_id":   chain_id,
                    "chain_g":    g,
                    "agent_name": "stharc",
                    "task_chain": json.dumps(task_list, ensure_ascii=False),
                }
            )

    return records


def _topo_sort_queries(
    queries: List[Dict[str, Any]],
    skill_id: str,
    dep_graph: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """简单拓扑：上游 skill 的任务排前面，本 skill（及无关）任务排后面。

    注意：queries 内所有条目的 skill_tag 已等于 skill_id，因此需要调用方
    自行合并上游 skill 的任务再传入，或直接使用 build_task_chains_by_skill_dep
    内联的跨 skill_groups 排序逻辑。此函数保留供外部单独调用。
    """
    upstream = set(dep_graph.get(skill_id, []))
    upstream_queries = [q for q in queries if q.get("skill_tag") in upstream]
    own_queries = [q for q in queries if q.get("skill_tag") not in upstream]
    return upstream_queries + own_queries


# ═══════════════════════════════════════════════════════════════════════════════
# 策略 3：从已有轨迹数据提取
# ═══════════════════════════════════════════════════════════════════════════════

def build_task_chains_from_traces(
    traces_jsonl_path: str,
    K: int = 3,
    G: int = 4,
    min_turns: int = 1,
    seed: Optional[int] = 42,
) -> List[Dict[str, Any]]:
    """从已有轨迹数据（filtered_traces.jsonl）中提取 task_chain。

    轨迹格式（filtered_traces.jsonl 每行）：
        {
          "messages": [{"role": "system", ...}, {"role": "user", ...}, ...],
          "response": "...",
          "query":    "用户问题",
          "query_id": "...",
        }

    策略：
      - 按 query_id 前缀（session_id）分组，同一 session 的多条 query 组成一条 chain
      - 若一个 session 的 query 数 < K，用随机填充
      - 若 > K，取前 K 条

    Args:
        traces_jsonl_path: filtered_traces.jsonl 路径
        K:                 chain 长度
        G:                 GRPO group size
        min_turns:         session 最少有多少条 query 才参与（过滤过短 session）
        seed:              随机种子

    Returns:
        parquet record list
    """
    _validate_sizes(K=K, G=G)
    if seed is not None:
        random.seed(seed)

    import re
    from pathlib import Path

    raw_lines = Path(traces_jsonl_path).read_text(encoding="utf-8").splitlines()
    all_rows = [json.loads(ln) for ln in raw_lines if ln.strip()]

    # 按 session_id 分组（query_id 格式：{session_id}_{turn_idx}_{variant}）
    sessions: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_rows:
        qid = row.get("query_id") or ""
        # 提取 session_id：去掉末尾 _r\d+ 和 _\d+_r\d+ 部分
        # e.g. "dingfengling.1_wmEDCeCgAA_..._0_1_r4" → "dingfengling.1_wmEDCeCgAA_..."
        session_id = re.sub(r"_\d+_r\d+$", "", qid)
        sessions.setdefault(session_id, []).append(row)

    # 过滤 & 提取 chain
    single_queries: List[Dict[str, Any]] = []  # 用于填充不足 K 条的 session
    session_chains: List[List[Dict[str, Any]]] = []

    for sid, rows in sessions.items():
        # 去重：同一 session 内相同 query_id 取第一条
        seen_queries: set = set()
        unique_rows = []
        for r in rows:
            q = r.get("query", "")
            if q not in seen_queries:
                seen_queries.add(q)
                unique_rows.append(r)

        tasks = [_trace_row_to_task(r) for r in unique_rows]
        single_queries.extend(tasks)

        if len(tasks) >= min_turns:
            session_chains.append(tasks[:K])

    # 补齐不足 K 的 session
    # _fill_pool 只包含 session 轮次 < min_turns 的任务（未被放入 session_chains 的任务），
    # 避免把已在 session_chains 中的任务重复用作填充，造成训练数据重叠。
    short_session_queries: List[Dict[str, Any]] = []
    for sid, rows in sessions.items():
        seen_qs: set = set()
        unique_rows = [r for r in rows if r.get("query", "") not in seen_qs and not seen_qs.add(r.get("query", ""))]
        tasks = [_trace_row_to_task(r) for r in unique_rows]
        if len(tasks) < min_turns:
            short_session_queries.extend(tasks)

    random.shuffle(short_session_queries)
    _fill_pool = list(short_session_queries)
    _fill_idx = 0  # 消耗指针，保证同一任务不被重复用于多条 chain

    chains: List[List[Dict[str, Any]]] = []
    for chain_tasks in session_chains:
        if len(chain_tasks) < K:
            need = K - len(chain_tasks)
            fill = _fill_pool[_fill_idx : _fill_idx + need]
            _fill_idx += len(fill)
            chain_tasks = chain_tasks + fill
        # 过滤仍不足 K 的 chain（_fill_pool 耗尽时 fill 为空），保持与策略1/2一致
        if len(chain_tasks) == K:
            chains.append(chain_tasks)
        else:
            logger.warning(
                "build_task_chains_from_traces: 丢弃长度不足的 chain（实际 %d < K=%d），"
                "fill_pool 已耗尽",
                len(chain_tasks), K,
            )

    # 若 session_chains 为空，退化为随机分组
    if not chains:
        random.shuffle(single_queries)
        for i in range(0, len(single_queries) - K + 1, K):
            chains.append(single_queries[i : i + K])

    records: List[Dict[str, Any]] = []
    for cid, chain_tasks in enumerate(chains):
        chain_id = f"chain_{cid:05d}"
        for g in range(G):
            records.append(
                {
                    "uid":        chain_id,
                    "chain_id":   chain_id,
                    "chain_g":    g,
                    "agent_name": "stharc",
                    "task_chain": json.dumps(chain_tasks, ensure_ascii=False),
                }
            )

    return records


def _trace_row_to_task(row: Dict[str, Any]) -> Dict[str, Any]:
    """把 filtered_traces.jsonl 的一行转成 task dict。

    raw_prompt 只用真实用户问题（row["query"]），不用完整 messages。
    完整 messages 含系统注入的「信息完整性自检」等中间轮次，不适合作为训练起点。
    env.run(query=..., history=...) 会重新构建完整 system prompt + tool list，
    并将 history 注入为前序对话上下文。

    history 提取规则（与 2w_query.jsonl 的 history 字段对齐）：
      - 若 row 有显式 history 字段（2w_query.jsonl 格式）→ 直接用
      - 否则仅保留 messages 中唯一、精确当前问题边界之前的对话
      - 边界缺失或不唯一时返回空历史，不使用位置切片推断
    """
    query = row.get("query", "")
    raw_prompt = [{"role": "user", "content": query}]
    ext_params: Dict[str, Any] = row.get("ext_params") or {}
    history = _get_history(row)
    out: Dict[str, Any] = {
        "raw_prompt": raw_prompt,
        "ext_params": ext_params,
        "history":    history,
        "_split_keys": sorted(task_identity_keys(row)),
        "source_dataset": _identity_namespace(row),
        "query_id":   row.get("query_id", ""),
    }
    ba = _get_baseline_answer(row)
    if ba:
        out["baseline_answer"] = ba
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 内部辅助
# ═══════════════════════════════════════════════════════════════════════════════

def _identity_namespace(task: Dict[str, Any]) -> str:
    metadata = task.get("ext_params") or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    for fields in (task, metadata):
        if isinstance(fields, dict):
            for name in ("source_dataset", "dataset", "data_source"):
                if fields.get(name):
                    return str(fields[name])
    return ""


def task_identity_keys(task: Dict[str, Any], source_namespace: str = "") -> set[str]:
    """Content/source identities, never auto-numbered chain_id/uid/task_id.

    Query equality is deliberately conservative: answers, replica numbers and
    mutable metadata cannot make a repeated question independent. Explicit
    source identities also connect reformulations of the same source task.
    """
    def digest(kind: str, value: Any) -> str:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return kind + ":" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    keys = set(task.get("_split_keys") or [])
    query = task.get("query")
    prompt = task.get("raw_prompt") or task.get("prompt")
    if not isinstance(query, str) or not query.strip():
        if isinstance(prompt, str):
            query = prompt
        elif isinstance(prompt, list):
            users = [m.get("content") for m in prompt
                     if isinstance(m, dict) and m.get("role") == "user"]
            query = users[-1] if users else None
    if isinstance(query, str) and query.strip():
        keys.add(digest("query", query.strip()))
    namespace = _identity_namespace(task)
    if not namespace and "source_dataset" not in task:
        namespace = source_namespace
    for metadata in (task, task.get("ext_params") or {}):
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        if not isinstance(metadata, dict):
            continue
        for field in ("query_id", "source_id", "source_task_id", "source_question_id",
                      "original_id", "musique_id", "appworld_task_id"):
            value = metadata.get(field)
            if value is not None and value != "":
                # Benchmark-specific keys already name their source namespace.
                scope = "" if field in ("musique_id", "appworld_task_id") else namespace
                keys.add(digest("source", [scope, field, value]))
    return keys


def record_task_keys(record: Dict[str, Any]) -> set[str]:
    """Identity union for a raw query or a raw/serialized complete chain."""
    keys = task_identity_keys(record)
    extra_info = record.get("extra_info")
    if isinstance(extra_info, dict):
        keys.update(extra_info.get("_split_keys") or [])
    if "task_chain" not in record:
        if not keys:
            raise ValueError("Task has no query/content or source identity")
        return keys
    tasks = record["task_chain"]
    if isinstance(tasks, str):
        tasks = json.loads(tasks)
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("task_chain must be a non-empty list")
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("task_chain entries must be objects")
        task_keys = task_identity_keys(task, _identity_namespace(record))
        if not task_keys:
            raise ValueError("Chain task has no query/content or source identity")
        keys.update(task_keys)
    return keys


def _validate_sizes(K: int = 1, G: int = 1) -> None:
    if isinstance(K, bool) or not isinstance(K, int) or K <= 0:
        raise ValueError("K must be a positive integer")
    if isinstance(G, bool) or not isinstance(G, int) or G <= 0:
        raise ValueError("G must be a positive integer")


def _get_raw_prompt(q: Dict[str, Any]) -> Any:
    """从 query dict 中提取 raw_prompt（只含当前 query 的单条 user 消息）。

    兼容多种输入格式：
      - q["raw_prompt"]  → 直接使用（已是 messages 格式）
      - q["query"]       → 包装成 [{"role": "user", "content": query}]
      - q["messages"]    → 不直接使用（含 system prompt 和中间轮次）

    注意：multi-turn 上下文通过 history 字段单独传递，不放进 raw_prompt。
    """
    if "raw_prompt" in q:
        return q["raw_prompt"]
    query = q.get("query")
    if not isinstance(query, str) or not query.strip():
        prompt = q.get("prompt")
        if isinstance(prompt, str):
            query = prompt
        elif isinstance(prompt, list):
            users = [m.get("content") for m in prompt
                     if isinstance(m, dict) and m.get("role") == "user"]
            query = users[-1] if users else ""
    return [{"role": "user", "content": query if isinstance(query, str) else ""}]


def _get_baseline_answer(q: Dict[str, Any]) -> Optional[str]:
    """从 query / extra_info 提取 baseline_answer（pairwise reward 用）。"""
    ba = q.get("baseline_answer")
    if ba is None and isinstance(q.get("extra_info"), dict):
        ba = q["extra_info"].get("baseline_answer")
    if ba is None:
        return None
    s = str(ba).strip()
    return s or None


def _make_task_entry(q: Dict[str, Any]) -> Dict[str, Any]:
    """构造 task_chain 中单个任务 dict，透传 baseline_answer（若有）。"""
    entry: Dict[str, Any] = {
        "raw_prompt": _get_raw_prompt(q),
        "ext_params": q.get("ext_params") or {},
        "history": _get_history(q),
        "_split_keys": sorted(task_identity_keys(q)),
        "source_dataset": _identity_namespace(q),
    }
    # Scoring metadata stays separate from the policy-visible prompt.
    for key in ("write_only", "ground_truth", "harness_op", "reward_model", "extra_info"):
        if key in q:
            entry[key] = dict(q[key]) if isinstance(q[key], dict) else q[key]
    ba = _get_baseline_answer(q)
    if ba:
        entry["baseline_answer"] = ba
    return entry


# ═══════════════════════════════════════════════════════════════════════════════
# 策略 4：从"idea2 构造格式"预构建的完整 chain 直接转换（K 完全由生成器决定）
# ═══════════════════════════════════════════════════════════════════════════════
#
# 依据 research/dataset_explore/IDEA2_IMPLEMENTATION_PLAN.md §3。
#
# 与策略 1/2/3 的关键区别：策略 1/2/3 都在这个文件内部把"零散 query 列表"按某个
# K 切分成 chain；策略 4 的输入本身就**已经是**一条条完整的、彼此长度可能不同的
# chain（由 research/dataset_explore/idea2_datasets/ 下的参数化生成器产出，见
# §2 的"内容参数决定链长"设计），本函数只做格式转换，不做任何切分/补齐，
# 因此天然不含、也不需要 K 参数。


def to_stharc_task_chain(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把"idea2 构造格式"的单条 record 转换成 task_chain.py 期望的任务 dict 列表。

    输入 record 格式（research/dataset_explore/idea2_datasets/ 生成器产出，字段与
    dataset_explore/idea2_constructed/*.json 手工样例一致）::

        {
          "uid": "...", "chain_id": "...", "data_source": "...",
          "task_chain": [
            {
              "task_id": "...", "query": "...", "ground_truth": "...",
              "harness_op": {"type": ..., "writes_oid": ..., "reads_oid": [...]},
              "reward_model": {"style": "verifiable", "ground_truth": "..."},
            },
            ...
          ],
          "provenance_answer_key": {oid: {...}},
        }

    输出格式（``_make_task_entry`` / ``rollout_with_tracking`` 期望）::

        [{"raw_prompt": [...], "ext_params": {...}, "history": [...],
          "ground_truth": "...", "extra_info": {"_provenance_debug": {...}}}, ...]

    职责边界（严格对齐 §3 的"转换层只做格式转换，不含业务逻辑"）：
        - ``query``  → 包装成 ``raw_prompt``
        - ``ground_truth`` → 放到任务 dict 顶层（``_extract_task_reward_extra`` 会
          读到，交给 ``RewardContext.ground_truth``，即 ``ExactMatchReward`` 用的字段）
        - ``harness_op`` / ``provenance_answer_key`` → 整条搬进
          ``extra_info["_provenance_debug"]``，不注入 env.run() 的策略 prompt。
          harness_op 可用于识别纯写任务的 outcome；其余 provenance 供离线诊断。
    """
    chain_id = str(record.get("chain_id") or record.get("uid") or "")
    data_source = record.get("data_source", "")
    provenance_answer_key = record.get("provenance_answer_key") or {}

    tasks = record.get("task_chain") or []
    if isinstance(tasks, str):
        tasks = json.loads(tasks)
    record_task_keys(record)  # Reject empty/malformed chains before conversion.
    out: List[Dict[str, Any]] = []
    for t in tasks:
        task = dict(t)
        namespace = _identity_namespace(t)
        if not namespace and "source_dataset" not in t:
            namespace = _identity_namespace(record)
        task["source_dataset"] = namespace
        entry = _make_task_entry(task)
        gt = t.get("ground_truth")
        if gt is None and isinstance(t.get("reward_model"), dict):
            gt = t["reward_model"].get("ground_truth")
        if gt is not None:
            entry["ground_truth"] = gt

        extra = entry.get("extra_info")
        extra = {} if extra is None else extra
        if not isinstance(extra, dict):
            raise ValueError("extra_info must be a mapping")
        debug = extra.get("_provenance_debug")
        debug = {} if debug is None else debug
        if not isinstance(debug, dict):
            raise ValueError("_provenance_debug must be a mapping")
        debug = dict(debug)
        debug.update(chain_id=chain_id, data_source=data_source, task_id=t.get("task_id"))
        if t.get("harness_op") is not None:
            debug["harness_op"] = t["harness_op"]
        if "provenance_answer_key" in record:
            debug["provenance_answer_key"] = provenance_answer_key
        entry["extra_info"] = {**extra, "_provenance_debug": debug}
        out.append(entry)
    return out


def build_grpo_records_from_prebuilt_chains(
    source_records: List[Dict[str, Any]],
    G: int = 4,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """把一批"idea2 构造格式"的完整 chain 转成 parquet record list（每条重复 G 次）。

    每个 ``source_records`` 元素已经是一条完整的、长度自定的 chain（由
    ``research/dataset_explore/idea2_datasets/`` 下的生成器产出）；本函数**不**
    做任何切分/分组/补齐，只负责：
        1. 调用 :func:`to_stharc_task_chain` 做格式转换
        2. 按 chain_id 重复 G 次（GRPO group）
        3. 序列化成 parquet 期望的 JSON 字符串列

    刻意不提供 K 参数——链长完全由 ``source_records`` 里每条记录自身的
    ``len(task_chain)`` 决定，调用方若想要变长分布，只需让生成器按内容参数
    （见 §2）采样即可，本函数不参与、也不允许固定链长。

    Args:
        source_records: idea2 构造格式的 record 列表（每条已是完整 chain）
        G:              GRPO group size（每条 chain 重复 G 次）
        seed:            若给定，先按该种子打乱 source_records 顺序（不影响单条
            chain 内部内容，只影响写盘顺序，便于复现调试）

    Returns:
        parquet record list（uid 相同的 G 条为一个 GRPO group）
    """
    records = list(source_records)
    _validate_sizes(G=G)
    if not records:
        raise ValueError("source_records must be non-empty")
    if seed is not None:
        random.seed(seed)
        random.shuffle(records)

    out: List[Dict[str, Any]] = []
    for i, rec in enumerate(records):
        chain_id = str(rec.get("chain_id") or rec.get("uid") or f"chain_{i:05d}")
        task_list = to_stharc_task_chain(rec)
        if not task_list:
            logger.warning(
                "build_grpo_records_from_prebuilt_chains: chain_id=%s 转换后为空，已跳过",
                chain_id,
            )
            continue
        task_chain_json = json.dumps(task_list, ensure_ascii=False)
        for g in range(G):
            out.append(
                {
                    "uid":         chain_id,
                    "chain_id":    chain_id,
                    "chain_g":     g,
                    "agent_name":  "stharc",
                    "data_source": rec.get("data_source", "stharc"),
                    "_split_keys": sorted(record_task_keys(rec)),
                    "task_chain":  task_chain_json,
                }
            )
    return out


def chain_length_stats(source_records: List[Dict[str, Any]]) -> Dict[str, float]:
    """统计一批 idea2 record 的链长（K = len(task_chain)）分布。

    对应 §6.3/§7 的"统计断言：同一批次里 chain 长度的标准差 > 0，防止不小心又写死"。
    调用方（如生成脚本 / pytest）应对返回的 ``std`` 做 ``assert std > 0`` 才算通过。
    """
    lengths = [len(rec.get("task_chain") or []) for rec in source_records]
    n = len(lengths)
    if n == 0:
        return {"n": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    mean = sum(lengths) / n
    var = sum((x - mean) ** 2 for x in lengths) / n
    return {
        "n": n,
        "min": float(min(lengths)),
        "max": float(max(lengths)),
        "mean": mean,
        "std": var ** 0.5,
    }


def _get_history(q: Dict[str, Any]) -> List[Dict[str, str]]:
    """Return explicit history or messages before a unique exact query boundary.

    Only a bare query or the current-question heading is recognized. Missing
    or ambiguous boundaries yield no history; there is no positional fallback.
    """
    # Explicit history is a caller-supplied prefix, never a trace heuristic.
    explicit_history = q.get("history")
    if explicit_history is not None:
        if not isinstance(explicit_history, list):
            raise ValueError("history must be a list")
        return [m for m in explicit_history if isinstance(m, dict)
                and m.get("role") in ("user", "assistant")
                and isinstance(m.get("content"), str)]

    msgs = q.get("messages") or []
    query = q.get("query")
    if not isinstance(msgs, list) or not isinstance(query, str) or not query.strip():
        return []
    query = query.strip()

    def matches(message: Any) -> bool:
        if not isinstance(message, dict) or message.get("role") != "user":
            return False
        content = message.get("content")
        if not isinstance(content, str):
            return False
        content = content.strip()
        if content == query:
            return True
        wrapper = re.fullmatch(r"## 用户当前问题[：:]?\s*\n\s*(.*)", content, re.DOTALL)
        return wrapper is not None and wrapper.group(1).strip() == query

    boundaries = [i for i, message in enumerate(msgs) if matches(message)]
    if len(boundaries) != 1:
        return []
    return [{"role": m["role"], "content": m["content"]}
            for m in msgs[:boundaries[0]] if isinstance(m, dict)
            and m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str)]
