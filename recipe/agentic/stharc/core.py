"""
STHARC-GRPO 核心函数：

  rollout_with_tracking       — 封装 env.run()，注入 HarnessTracker 拦截 harness 事件
  execute_harness_action      — 从 tracker 事件封装结构化 output 对象
  mark_provenance             — 更新 H_g.provenance_registry（原地修改）
  compute_prospective_utility — 计算 chain 的 prospective bonus
  compute_sage_skill_integrated — SAGE-style skill-integrated reward（外部对照臂）
  run_frozen_chain            — 运行冻结基准 chain，返回 frozen_rewards 列表

所有任务均使用上游任务奖励，不按 read/write 类型屏蔽或整形。
本发布副本采用新奖励协议，不是原 read-only 实验的等价复现。

SkillRise（credit=skillrise）：每题 i<K 后多一次 curate 生成 skill document，
solve 吃 r_i，curate 吃 Σ γ^{j-i} r_j；val 也走 curate。主报仍用 total_reward。
训练仍是整链一个 GRPO 分（= Σ r + Σ G_curate），不是官方按 (i, role) 分组。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from .harness_state import HarnessState

# ── harness 操作分类常量（与 ai2c 侧 harness_tracker.py 保持一致）─────────────
HARNESS_WRITE_OPS: frozenset = frozenset({"skill_manage", "memory"})
# memory_inject：MemoryStore.format_for_system_prompt 被动注入（与 write 的 memory:* 对齐）
HARNESS_READ_OPS: frozenset = frozenset(
    {"load_skill", "memory_inject", "session_search", "case_search"}
)


# ═══════════════════════════════════════════════════════════════════════════════
# rollout_with_tracking
# ═══════════════════════════════════════════════════════════════════════════════

def rollout_with_tracking(
    env: Any,
    task: Dict[str, Any],
    H_g: "HarnessState",
    frozen: bool = False,
) -> Tuple[Any, List[Dict[str, Any]], Set[str]]:
    """执行单个任务的 rollout，注入 HarnessTracker 拦截 harness 事件。

    Args:
        env:    AiToCReactEnv 实例（或兼容 .run(query, ext_params) 的对象）
        task:   {'raw_prompt': ..., 'ext_params': {...}}
        H_g:    本 chain 的 HarnessState
        frozen: 若 True，harness 写操作返回 mock ack（不实际写盘）

    Returns:
        (env_result, harness_actions_k, read_set_k)
        - env_result:        env.run() 的返回值
        - harness_actions_k: list[dict]，本任务中所有 harness 写操作的记录
        - read_set_k:        set[str]，本任务中所有 harness 读操作的 target_id

    解耦说明：
        ai2c 专属的 monkeypatch（``ai2c_harness_patch``）与 RL context 绑定
        （``bind_rl_context``/``restore_rl_context``）只在
        ``env.USES_AI2C_HARNESS_PATCH`` 为 True（默认，兼容旧行为）时执行。
        与 ai2c 解耦的 env（如 ``GenericBenchmarkEnv``，其
        ``USES_AI2C_HARNESS_PATCH = False``）自己在工具执行处调用
        ``get_tracker().record_write/record_read``，此处只需注入 tracker + 合并
        ext_params，不需要也不应该触发 ai2c 的 monkeypatch。
    """
    # research 侧 tracker（业务无关，两条路径都需要）
    from recipe.agentic.stharc.harness_tracker import (
        HarnessTracker,
        clear_tracker,
        set_tracker,
    )

    uses_ai2c_patch = bool(getattr(env, "USES_AI2C_HARNESS_PATCH", True))

    prev_root = None
    if uses_ai2c_patch:
        from recipe.agentic.stharc.ai2c_harness_patch import (
            apply_ai2c_harness_patches,
            bind_rl_context,
            restore_rl_context,
        )
        apply_ai2c_harness_patches()

    tracker = HarnessTracker()
    token = set_tracker(tracker)

    ext_params: Dict[str, Any] = {
        **(task.get("ext_params") or {}),
        **H_g.to_ext_params_patch(frozen=frozen),
    }
    # history：前序多轮对话上下文（与 2w_query.jsonl 的 history 字段对齐）
    # 两种来源：task["history"]（已在 task_chain 构建时提取），或兜底空列表
    history: list = task.get("history") or []

    if uses_ai2c_patch:
        prev_root = bind_rl_context(ext_params)
    try:
        query = _extract_query_from_task(task)
        env_result = env.run(query=query, ext_params=ext_params, history=history)
    finally:
        if uses_ai2c_patch:
            restore_rl_context(prev_root)
        clear_tracker(token)

    harness_actions_k: List[Dict[str, Any]] = [
        {
            "step":            ev.step,
            "op_type":         ev.tool_name,
            "target_id":       ev.target_id,
            "args":            ev.args,
            "before_snapshot": ev.before_snapshot,
            "after_snapshot":  ev.after_snapshot,
        }
        for ev in tracker.write_events
    ]
    read_set_k: Set[str] = tracker.read_set

    return env_result, harness_actions_k, read_set_k


def _extract_query_from_task(task: Dict[str, Any]) -> str:
    """从 task dict 中提取 query 字符串。"""
    raw = task.get("raw_prompt") or task.get("messages") or task.get("query", "")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list) and raw:
        for m in reversed(raw):
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content", ""))
        last = raw[-1]
        return str(last.get("content", "")) if isinstance(last, dict) else str(last)
    return str(raw)


# ═══════════════════════════════════════════════════════════════════════════════
# execute_harness_action
# ═══════════════════════════════════════════════════════════════════════════════

def execute_harness_action(
    action: Dict[str, Any],
    H_g: "HarnessState",
) -> Dict[str, Any]:
    """从 action 记录封装结构化 output 对象。

    harness 写操作已在 rollout_with_tracking 内部完成，此处不重复执行，
    只做 op_type → object_type/mutation_type 的语义映射。

    Args:
        action: harness_actions_k 中的单条记录
        H_g:    本 chain 的 HarnessState（暂未使用，保留供未来扩展）

    Returns:
        结构化 output dict，含 object_id / object_type / mutation_type / before / after
    """
    object_id = action["target_id"]

    if action["op_type"] == "skill_manage":
        object_type = "skill"
        sub_action = str((action.get("args") or {}).get("action", "")).lower()
        if sub_action == "create" or action.get("before_snapshot") == "":
            mutation_type = "create"
        elif sub_action == "delete":
            mutation_type = "delete"
        else:
            mutation_type = "modify"

    elif action["op_type"] == "memory":
        object_type = "memory_entry"
        sub_action = str((action.get("args") or {}).get("action", "")).lower()
        if sub_action == "add":
            mutation_type = "create"
        elif sub_action == "delete":
            mutation_type = "delete"
        else:
            mutation_type = "modify"

    else:
        raise ValueError(f"execute_harness_action: 未知 op_type={action['op_type']!r}")

    return {
        "object_id":     object_id,
        "object_type":   object_type,
        "mutation_type": mutation_type,
        "before":        action.get("before_snapshot", ""),
        "after":         action.get("after_snapshot", ""),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# mark_provenance
# ═══════════════════════════════════════════════════════════════════════════════

def mark_provenance(
    H_g: "HarnessState",
    output: Dict[str, Any],
    source: Tuple[int, Dict[str, Any]],
) -> None:
    """将 (task_k, action) 的 mutation 写入 H_g.provenance_registry（原地修改）。

    Args:
        H_g:    本 chain 的 HarnessState（原地修改 provenance_registry）
        output: execute_harness_action() 的返回值
        source: (k, action)，k 为任务编号
    """
    k, action = source
    oid = output["object_id"]
    if oid not in H_g.provenance_registry:
        H_g.provenance_registry[oid] = []
    H_g.provenance_registry[oid].append(
        {
            "source_task":   k,
            "action":        action,
            "mutation_type": output["mutation_type"],
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# compute_prospective_utility
# ═══════════════════════════════════════════════════════════════════════════════

def compute_prospective_utility(
    harness_mutations: List[Tuple[int, Dict[str, Any], Dict[str, Any]]],
    task_read_sets: List[Set[str]],
    task_rewards: List[float],
    frozen_rewards: List[float],
    gamma: float = 0.9,
    attribution: str = "gamma_decay",
    shuffle_seed: int = 0,
) -> float:
    """计算整条 chain 的 prospective bonus。

    公式（参考 research/stharc_grpo_implementation_plan.md §1.2）：
        prospective_utility(a_k) = Σ_{j>k} 𝟙[oid ∈ read_set_j] · (r_j - b_j) · γ^(j-k)

    Args:
        harness_mutations: list of (k, action, output)
        task_read_sets:    list[set]，长度 K，每个任务的 harness 读 object_id 集合
        task_rewards:      list[float]，长度 K，每个任务的 reward
        frozen_rewards:    list[float]，长度 K，基准 reward（阶段一可用 [0.0]*K）
        gamma:             跨任务折扣因子（默认 0.9）
        attribution:       归因策略：
                           - 'gamma_decay' / 'latest_only' / 'equal_split'：正确 provenance
                           - 'random_shuffle'：Random-PU 负对照——公式不变，但把 oid↔read
                             门控用的 object_id 做随机置换（抗 shaping 证据 E4）
        shuffle_seed:      仅 attribution='random_shuffle' 时使用，保证可复现

    Returns:
        prospective_bonus（标量）
    """
    K = len(task_rewards)
    if K == 0 or not harness_mutations:
        return 0.0

    # Random-PU：对出现过的 oid 做一次置换；门控时用 π(oid)∈read_set_j，打乱归因
    oid_perm: Dict[str, str] = {}
    if attribution == "random_shuffle":
        universe: Set[str] = set()
        for _, _, output in harness_mutations:
            if output.get("object_id"):
                universe.add(str(output["object_id"]))
        for rs in task_read_sets:
            universe.update(str(x) for x in rs)
        ordered = sorted(universe)
        shuffled = ordered[:]
        random.Random(int(shuffle_seed)).shuffle(shuffled)
        oid_perm = {a: b for a, b in zip(ordered, shuffled)}

    # 预计算 attribution 辅助结构
    latest_modifier: Dict[str, int] = {}
    modifier_count: Dict[str, int] = {}

    if attribution == "latest_only":
        for k, _, output in harness_mutations:
            oid = output["object_id"]
            if oid not in latest_modifier or k > latest_modifier[oid]:
                latest_modifier[oid] = k

    elif attribution == "equal_split":
        for _, _, output in harness_mutations:
            oid = output["object_id"]
            modifier_count[oid] = modifier_count.get(oid, 0) + 1

    prospective_bonus = 0.0

    for k, action, output in harness_mutations:
        oid = output["object_id"]
        gate_oid = oid_perm.get(str(oid), oid) if oid_perm else oid

        # latest_only：只为最新一次修改者记功
        if attribution == "latest_only" and latest_modifier.get(oid) != k:
            continue

        pu = 0.0
        for j in range(k + 1, K):
            if gate_oid in task_read_sets[j]:
                b_j = frozen_rewards[j] if j < len(frozen_rewards) else 0.0
                delta_r = task_rewards[j] - b_j
                if attribution == "equal_split":
                    delta_r /= max(modifier_count.get(oid, 1), 1)
                pu += delta_r * (gamma ** (j - k))

        prospective_bonus += pu

    return prospective_bonus


# ═══════════════════════════════════════════════════════════════════════════════
# compute_sage_skill_integrated
# ═══════════════════════════════════════════════════════════════════════════════

# create / modify 才算「生成了可复用 skill/memory」；delete / invalidate 不算。
_SAGE_WRITE_MUTATION_TYPES: frozenset = frozenset({"create", "modify"})


@dataclass(frozen=True)
class SageSkillIntegratedResult:
    """SAGE-style 每任务 R_k 与 bonus 分解。

    ``task_rewards_sage[k] = r_k + generation_bonus[k] + usage_bonus[k]``
    每个 bonus ∈ {0, 1}。``chain_reward = Σ R_k``（训练用）。
    主报仍用所有任务的上游 ``total_reward``，不含 bonus。
    """

    task_rewards_sage: List[float]
    generation_bonus: List[float]
    usage_bonus: List[float]
    chain_reward: float
    gen_bonus_sum: float
    use_bonus_sum: float


def _sage_mutation_type(output: Dict[str, Any], action: Dict[str, Any]) -> str:
    mt = output.get("mutation_type")
    if mt:
        return str(mt).lower()
    sub = str((action.get("args") or {}).get("action", "")).lower()
    if sub in ("create", "add"):
        return "create"
    if sub in ("delete",):
        return "delete"
    if sub in ("modify", "update"):
        return "modify"
    return ""


def _sage_is_success(r: float, success_threshold: float) -> bool:
    if r is None or math.isnan(float(r)):
        return False
    return float(r) + 1e-9 >= float(success_threshold)


def compute_sage_skill_integrated(
    harness_mutations: List[Tuple[int, Dict[str, Any], Dict[str, Any]]],
    task_read_sets: List[Set[str]],
    task_rewards: List[float],
    success_threshold: float = 1.0,
    bonus_coef: float = 1.0,
    require_self_success: bool = True,
) -> SageSkillIntegratedResult:
    """SAGE-style Skill-integrated Reward（接到我们的 harness tracker）。

    原文（K=2）::

        R1 = r1 + 1[r1=1] * 1[r2=1] * 1skill(q2|q1)
        R2 = r2 + 1[r2=1] * 1skill(q2|q1)

    默认 generation 需要生成任务自身成功，且至少一个后来任务成功复用其产物：

        used(j, k) = 任务 k 的 create/modify oid ∩ 任务 j 的 read_set 非空
        generation_k = 1[r_k 成功 ∧ ∃ j>k: r_j 成功 ∧ used(j,k)]
        usage_k      = 1[r_k 成功 ∧ ∃ i<k: used(k,i)]
        R_k          = r_k + c·generation_k + c·usage_k

    成功由上游任务奖励与 ``success_threshold`` 判定。
    ``bonus_coef``（默认 1）只缩放两个 +1，主报 total_reward 仍不含 bonus。
    可显式配置 ``require_self_success=False``，仅取消 generation 的自身成功门控。

    SkillRise 不在此函数：那是额外 curate 生成 + 解/编解耦信度（credit=skillrise）。
    """
    K = len(task_rewards)
    zeros = [0.0] * K
    if K == 0:
        return SageSkillIntegratedResult([], [], [], 0.0, 0.0, 0.0)
    if any(math.isnan(float(r)) for r in task_rewards):
        nan_list = [float("nan")] * K
        return SageSkillIntegratedResult(
            task_rewards_sage=nan_list,
            generation_bonus=list(zeros),
            usage_bonus=list(zeros),
            chain_reward=float("nan"),
            gen_bonus_sum=0.0,
            use_bonus_sum=0.0,
        )

    write_oids: List[Set[str]] = [set() for _ in range(K)]
    for k, action, output in harness_mutations:
        if k < 0 or k >= K:
            continue
        if _sage_mutation_type(output, action) not in _SAGE_WRITE_MUTATION_TYPES:
            continue
        oid = output.get("object_id")
        if oid:
            write_oids[k].add(str(oid))

    def _used(j: int, k: int) -> bool:
        if j < 0 or j >= K or k < 0 or k >= K:
            return False
        rs = task_read_sets[j] if j < len(task_read_sets) else set()
        if not rs or not write_oids[k]:
            return False
        return bool(write_oids[k] & {str(x) for x in rs})

    c = float(bonus_coef)
    generation_bonus = [0.0] * K
    usage_bonus = [0.0] * K
    for k in range(K):
        later_ok = any(
            _sage_is_success(task_rewards[j], success_threshold) and _used(j, k)
            for j in range(k + 1, K)
        )
        self_ok = (not require_self_success) or _sage_is_success(
            task_rewards[k], success_threshold
        )
        if later_ok and self_ok:
            generation_bonus[k] = c
        if _sage_is_success(task_rewards[k], success_threshold) and any(
            _used(k, i) for i in range(k)
        ):
            usage_bonus[k] = c

    task_rewards_sage = [
        float(task_rewards[k]) + generation_bonus[k] + usage_bonus[k] for k in range(K)
    ]
    return SageSkillIntegratedResult(
        task_rewards_sage=task_rewards_sage,
        generation_bonus=generation_bonus,
        usage_bonus=usage_bonus,
        chain_reward=float(sum(task_rewards_sage)),
        gen_bonus_sum=float(sum(generation_bonus)),
        use_bonus_sum=float(sum(usage_bonus)),
    )


SKILLRISE_DOC_BEGIN = "[Skill document — reusable notes from earlier tasks in this chain]"
SKILLRISE_DOC_END = "[End skill document]"

_SKILLRISE_CURATE_SYSTEM = """You curate a reusable skill document for later tasks in the same chain.

Rewrite the COMPLETE document. Keep transferable procedures (what to store in memory/skills, which keys to read later, how to extract facts). Drop instance-only fluff. Do not call tools. Output ONLY the document text."""


@dataclass(frozen=True)
class SkillRiseDecoupledResult:
    """SkillRise 解耦信度。

    solve_returns[i] = r_i
    curate_returns[i] = Σ_{j>i} γ^{j-i} r_j   （最后一题无 curate，为 0）
    chain_reward = Σ solve + Σ curate（训练用；主报仍用上游 total_reward）
    """

    solve_returns: List[float]
    curate_returns: List[float]
    chain_reward: float
    curate_return_sum: float


def compute_skillrise_decoupled_credit(
    task_rewards: List[float],
    gamma: float = 0.6,
) -> SkillRiseDecoupledResult:
    """SkillRise 解耦信度（style-on-ours）。

    官方::

        G_{i,solve}  = r_i
        G_{i,curate} = Σ_{j=i+1}^{K} γ^{j-i} r_j     (i < K)

    r_i 为每题的上游任务奖励，curate 累积所有后来任务的折扣奖励。
    γ 默认 0.6（原文），与 STHARC PU 的 0.9 分开。

    训练仍把整链打成一个 GRPO 分 ``Σ G_solve + Σ G_curate``；官方按
    (题号, solve/curate) 分组未做。
    """
    K = len(task_rewards)
    if K == 0:
        return SkillRiseDecoupledResult([], [], 0.0, 0.0)
    if any(math.isnan(float(r)) for r in task_rewards):
        nan_list = [float("nan")] * K
        return SkillRiseDecoupledResult(
            solve_returns=nan_list,
            curate_returns=nan_list,
            chain_reward=float("nan"),
            curate_return_sum=0.0,
        )

    g = float(gamma)
    solve_returns = [float(r) for r in task_rewards]
    curate_returns = [0.0] * K
    for i in range(K - 1):
        acc = 0.0
        power = g
        for j in range(i + 1, K):
            acc += power * float(task_rewards[j])
            power *= g
        curate_returns[i] = acc
    return SkillRiseDecoupledResult(
        solve_returns=solve_returns,
        curate_returns=curate_returns,
        chain_reward=float(sum(solve_returns) + sum(curate_returns)),
        curate_return_sum=float(sum(curate_returns)),
    )


def inject_skillrise_document(query: str, document: str) -> str:
    """把 skill document 垫到下一题 user query 前面。空文档不改 query。"""
    q = str(query or "")
    doc = str(document or "").strip()
    if not doc:
        return q
    return f"{SKILLRISE_DOC_BEGIN}\n{doc}\n{SKILLRISE_DOC_END}\n\n{q}"


def apply_skillrise_document_to_task(task: Dict[str, Any], document: str) -> Dict[str, Any]:
    """复制 task，把 S 注入 query / raw_prompt / messages，供下一题 solve。"""
    t = dict(task) if isinstance(task, dict) else {}
    doc = str(document or "").strip()
    if not doc:
        return t
    q = _extract_query_from_task(t)
    new_q = inject_skillrise_document(q, doc)
    t["query"] = new_q

    def _replace_last_user(msgs: List[Any]) -> List[Any]:
        out = [dict(m) if isinstance(m, dict) else m for m in msgs]
        for i in range(len(out) - 1, -1, -1):
            m = out[i]
            if isinstance(m, dict) and m.get("role") == "user":
                m["content"] = new_q
                break
        return out

    raw = t.get("raw_prompt")
    if isinstance(raw, str) and raw:
        t["raw_prompt"] = inject_skillrise_document(raw, doc) if raw == q else new_q
    elif isinstance(raw, list) and raw:
        t["raw_prompt"] = _replace_last_user(list(raw))
    msgs = t.get("messages")
    if isinstance(msgs, list) and msgs:
        t["messages"] = _replace_last_user(list(msgs))
    return t


def summarize_solve_trajectory(env_result: Any, max_chars: int = 4000) -> str:
    """把刚做完的 solve 轨迹压成 curate 用的短文本。"""
    parts: List[str] = []
    for rec in list(getattr(env_result, "records", None) or []):
        if isinstance(rec, dict):
            resp = rec.get("response") or rec.get("response_text") or ""
        else:
            resp = getattr(rec, "response", None) or getattr(rec, "response_text", "") or ""
        resp = str(resp).strip()
        if resp:
            parts.append(resp)
    text = "\n---\n".join(parts)
    fa = str(getattr(env_result, "final_answer", "") or "").strip()
    if fa:
        text = f"{text}\nFinal answer: {fa}" if text else f"Final answer: {fa}"
    if len(text) > int(max_chars):
        text = text[: int(max_chars)] + "\n...[truncated]..."
    return text


def format_skillrise_curate_messages(
    prev_document: str,
    task_query: str,
    trajectory_text: str,
    task_reward: float,
) -> List[Dict[str, str]]:
    prev = str(prev_document or "").strip() or "(empty)"
    q = str(task_query or "").strip() or "(missing query)"
    traj = str(trajectory_text or "").strip() or "(no trajectory)"
    user = (
        f"Previous skill document:\n{prev}\n\n"
        f"Just-finished task:\n{q}\n\n"
        f"Agent trajectory:\n{traj}\n\n"
        f"Upstream task reward: {float(task_reward):.4f}\n\n"
        "Rewrite the complete skill document now."
    )
    return [
        {"role": "system", "content": _SKILLRISE_CURATE_SYSTEM},
        {"role": "user", "content": user},
    ]


def generate_skillrise_document(
    generate_fn: Any,
    *,
    prev_document: str,
    task_query: str,
    trajectory_text: str,
    task_reward: float,
) -> str:
    """调用 generate_fn(messages) → 新文档。失败则沿用 prev_document。"""
    messages = format_skillrise_curate_messages(
        prev_document, task_query, trajectory_text, task_reward
    )
    try:
        raw = generate_fn(messages)
    except Exception:
        return str(prev_document or "").strip()
    text = str(raw or "").strip()
    if text.startswith(SKILLRISE_DOC_BEGIN):
        text = text[len(SKILLRISE_DOC_BEGIN) :].strip()
    if text.endswith(SKILLRISE_DOC_END):
        text = text[: -len(SKILLRISE_DOC_END)].strip()
    return text or str(prev_document or "").strip()


# ═══════════════════════════════════════════════════════════════════════════════
# run_frozen_chain
# ═══════════════════════════════════════════════════════════════════════════════

def run_frozen_chain(
    env: Any,
    task_chain: List[Dict[str, Any]],
    H_init: "HarnessState",
    reward: Any = None,
    builder: Any = None,
) -> Tuple[List[float], List[Any]]:
    """运行一条冻结 harness 的对照 chain，返回 frozen_rewards 与每个任务的 env_result。

    harness 写操作返回 mock ack（不实际写盘），H_frozen 始终保持 H_init 状态。

    **reward 必须与 live chain 使用同一套管道**（``builder.build`` + ``reward.compute``），
    否则 ``b_j`` 会恒为 0，prospective bonus 退化成阶段一。

    Args:
        env:        AiToCReactEnv 实例
        task_chain: K 个任务的列表
        H_init:     本 group 的 harness 起点快照
        reward:     与 STHARCAgentLoop.self.reward 相同的 reward 对象（必填，正式训练）
        builder:    与 STHARCAgentLoop.self.builder 相同的 TrajectoryBuilder（必填）

    Returns:
        frozen_rewards:   list[float]，长度 K
        frozen_env_results: list[Any]，每个任务的原始 env_result（含 records/final_answer）
    """
    if reward is None or builder is None:
        raise ValueError(
            "run_frozen_chain 需要传入与 live 一致的 reward 与 builder，"
            "否则无法计算正确的 frozen baseline（EnvRunResult 本身不含 reward 字段）"
        )

    from recipe.agentic.reward.base import RewardContext

    def _task_extra(task: Any) -> Dict[str, Any]:
        if not isinstance(task, dict):
            return {}
        extra: Dict[str, Any] = {}
        nested = task.get("extra_info") if isinstance(task.get("extra_info"), dict) else {}
        ba = task.get("baseline_answer")
        if ba is None:
            ba = nested.get("baseline_answer")
        if ba is not None and str(ba).strip():
            extra["baseline_answer"] = str(ba).strip()
        for k in ("ground_truth", "teacher_trace"):
            v = task.get(k, nested.get(k) if nested else None)
            if v is not None:
                extra[k] = v
        return extra

    H_frozen = H_init.clone_for_chain("chain_frozen")
    frozen_rewards: List[float] = []
    frozen_env_results: List[Any] = []

    try:
        for task in task_chain:
            env_result, harness_actions_k, _ = rollout_with_tracking(
                env, task, H_frozen, frozen=True
            )
            query = _extract_query_from_task(task)
            builder_out = builder.build(env_result, query, None)
            task_extra = _task_extra(task)
            reward_res = reward.compute(
                RewardContext(
                    summary=builder_out.summary,
                    ground_truth=task_extra.get("ground_truth"),
                    extra_info=task_extra,
                )
            )
            r_k = float(reward_res.value)
            frozen_rewards.append(r_k)
            frozen_env_results.append(env_result)
    finally:
        H_frozen.cleanup()

    return frozen_rewards, frozen_env_results
