"""Task-position credit used by the public paper training path.

Frozen outcomes and legacy curate/document helpers are deliberately not inputs.
"""
from __future__ import annotations

import math
import random
import re
from numbers import Integral, Real


def finite_rewards(values):
    """Reject incomplete or non-finite task outcomes before training."""
    values = list(values)
    if not values:
        raise ValueError("STHARC requires a non-empty task reward list")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in values):
        raise ValueError("STHARC task rewards must be finite real numbers")
    return [float(v) for v in values]


def is_write_only_task(task):
    """Use explicit reward metadata, or the legacy acknowledgement-only schema.

    A write operation alone, an 'ok' answer alone, or an empty read set is NOT
    sufficient. Legacy tasks must also explicitly request a finish: ok reply.
    Metadata is used only for scoring, never injected into model prompts.
    """
    if not isinstance(task, dict):
        raise ValueError("task must be a mapping")
    extra = task.get("extra_info")
    reward_model = task.get("reward_model")
    extra = {} if extra is None else extra
    reward_model = {} if reward_model is None else reward_model
    if not isinstance(extra, dict) or not isinstance(reward_model, dict):
        raise ValueError("extra_info and reward_model must be mappings")
    for source in (task, extra, reward_model):
        if "write_only" in source:
            if not isinstance(source["write_only"], bool):
                raise ValueError("write_only must be a boolean")
            return source["write_only"]
    debug = extra.get("_provenance_debug")
    debug = {} if debug is None else debug
    if not isinstance(debug, dict):
        raise ValueError("_provenance_debug must be a mapping")
    op = task.get("harness_op") or debug.get("harness_op") or {}
    if not isinstance(op, dict):
        raise ValueError("harness_op must be a mapping")
    if op.get("type") not in {
        "memory_write", "memory_update", "memory_invalidate",
        "skill_create", "skill_modify", "skill_delete",
    }:
        return False
    gt = task.get("ground_truth", extra.get("ground_truth", reward_model.get("ground_truth")))
    if not isinstance(gt, str) or gt.strip().lower() != "ok":
        return False
    prompt = task.get("raw_prompt") or task.get("messages") or task.get("query") or ""
    if isinstance(prompt, list):
        if any(not isinstance(m, dict) for m in prompt):
            raise ValueError("prompt messages must be mappings")
        prompt = "\n".join(str(m.get("content", "")) for m in prompt if m.get("role") == "user")
    return bool(re.search(r"\breply\s*(?:with)?\s*:\s*finish\s*:\s*ok\b", str(prompt), re.I))


def task_outcome(task, reward):
    value = finite_rewards([reward])[0]
    return 0.0 if is_write_only_task(task) else value


def compute_task_credit(harness_mutations, task_read_sets, task_rewards, *,
                        credit="pu", gamma=0.9, lam=0.25,
                        attribution="latest_only", shuffle_seed=0):
    """Return (task_returns, per-task bonus), without chain-scalar fallback.

    PU's bonus is C_k (unscaled); the other proxies return their additive bonus.
    latest_only selects the nearest strictly preceding writer for EACH read.
    gamma_decay/equal_split remain explicit attribution ablations, not defaults.
    """
    rewards = finite_rewards(task_rewards)
    K = len(rewards)
    if len(task_read_sets) != K:
        raise ValueError("read sets must match the complete task reward list")
    if credit not in {"pu", "sage", "skillrise"}:
        raise ValueError(f"Unknown STHARC credit: {credit}")
    if attribution not in {"latest_only", "random_shuffle", "gamma_decay", "equal_split"}:
        raise ValueError(f"Unknown STHARC attribution: {attribution}")
    if not math.isfinite(gamma) or not 0 <= gamma <= 1 or not math.isfinite(lam):
        raise ValueError("gamma must be in [0, 1] and lam must be finite")
    writes = [set() for _ in rewards]
    for k, action, output in harness_mutations:
        if isinstance(k, bool) or not isinstance(k, Integral) or not 0 <= k < K:
            raise ValueError("mutation task index is out of range")
        if not isinstance(output, dict):
            raise ValueError("mutation requires an object_id")
        oid = output.get("object_id")
        if not oid:
            raise ValueError("mutation requires an object_id")
        oid = str(oid)
        writes[k].add(oid)
    reads = [{str(oid) for oid in rs} for rs in task_read_sets]
    bonus = [0.0] * K
    if credit == "skillrise":
        for k in range(K):
            if writes[k]:
                bonus[k] = sum(gamma ** (j - k) * rewards[j] for j in range(k + 1, K))
    elif credit == "sage":
        for k in range(K):
            if rewards[k] > 0 and any(
                rewards[j] > 0 and writes[k] & reads[j] for j in range(k + 1, K)
            ):
                bonus[k] = 1.0
    else:
        universe = sorted(set().union(*writes, *reads))
        shuffled = universe[:]
        if attribution == "random_shuffle":
            random.Random(int(shuffle_seed)).shuffle(shuffled)
        gate = dict(zip(universe, shuffled))
        preceding = {}
        for j in range(K):
            # Update after reading: same-task writes cannot claim future credit.
            for oid, writers in preceding.items():
                if gate[oid] not in reads[j]:
                    continue
                selected = writers[-1:] if attribution in {"latest_only", "random_shuffle"} else writers
                divisor = len(selected) if attribution == "equal_split" else 1
                for k in selected:
                    bonus[k] += gamma ** (j - k) * rewards[j] / divisor
            for oid in sorted(writes[j]):
                preceding.setdefault(oid, []).append(j)
    scale = lam if credit == "pu" else 1.0
    returns = finite_rewards([r + scale * c for r, c in zip(rewards, bonus)])
    finite_rewards(bonus)
    return returns, bonus


def pack_task_credit(task_returns, response_task_ids, response_mask):
    """Validate and serialize unpadded production credit metadata."""
    returns = finite_rewards(task_returns)
    ids = list(response_task_ids)
    mask = list(response_mask)
    if len(ids) != len(mask):
        raise ValueError("response task ids and mask must have identical lengths")
    for task_id, active in zip(ids, mask):
        if isinstance(task_id, bool) or not isinstance(task_id, Integral):
            raise ValueError("response task ids must be integers")
        if active not in (0, 1) or not -1 <= task_id < len(returns):
            raise ValueError("invalid response task id or mask")
        if (active == 1) != (task_id >= 0):
            raise ValueError("only policy tokens may have task ids")
    return {"task_returns": returns, "response_task_ids": ids}