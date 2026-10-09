#!/usr/bin/env python3
"""独立 vLLM 跑 STHARC held-out val（不启动 Ray / verl PPO）。

计分与训练中 val 相同：所有任务均使用上游奖励，不按 read/write 类型屏蔽或整形。
链上 total_reward 为奖励总和，avg_task_reward 为 sum / K，空链为 0。
本发布副本采用新奖励协议，不是原 read-only 实验的等价复现。

用法（在 training_code_public 目录，先起好 vLLM）::

    CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \\
      --model /mnt/workspace/ckpt/Qwen3.5-9B --port 18100 \\
      --max-model-len 32768 --gpu-memory-utilization 0.85 --dtype bfloat16

    PYTHONPATH=. python recipe/agentic/scripts/eval_stharc_val_vllm.py \\
      --val-parquet data/stharc_musique_verbatim_mechlike_n1k/val.parquet \\
      --vllm-url http://127.0.0.1:18100/v1 \\
      --model /mnt/workspace/ckpt/Qwen3.5-9B
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

_HERE = Path(__file__).resolve()
_VERL_RESEARCH = _HERE.parents[3]
if str(_VERL_RESEARCH) not in sys.path:
    sys.path.insert(0, str(_VERL_RESEARCH))


def _as_list(tc: Any) -> List[Dict[str, Any]]:
    if isinstance(tc, str):
        tc = json.loads(tc)
    if tc is None:
        return []
    return [dict(t) if not isinstance(t, dict) else t for t in list(tc)]


def _ground_truth(task: Dict[str, Any]) -> Any:
    nested = task.get("extra_info") if isinstance(task.get("extra_info"), dict) else {}
    gt = task.get("ground_truth", nested.get("ground_truth") if nested else None)
    if gt is None:
        rm = task.get("reward_model")
        if isinstance(rm, dict):
            gt = rm.get("ground_truth")
    return gt


def _eval_one_chain(
    env: Any,
    reward: Any,
    task_chain: List[Dict[str, Any]],
    chain_id: str,
) -> Dict[str, Any]:
    from recipe.agentic.reward.base import RewardContext
    from recipe.agentic.stharc.core import rollout_with_tracking
    from recipe.agentic.stharc.generic_harness_state import GenericHarnessState
    from recipe.agentic.trajectory.summary import TrajectorySummary

    H_init = GenericHarnessState.create_h_init(
        group_id=f"base_eval_{chain_id}",
        base_user_id="default",
    )
    H_g = H_init.clone_for_chain(f"chain_{uuid.uuid4().hex[:8]}")
    task_rewards: List[float] = []
    for task in task_chain:
        env_result, harness_actions, _read_set = rollout_with_tracking(env, task, H_g, False)
        ctx = RewardContext(
            summary=TrajectorySummary(final_answer=str(getattr(env_result, "final_answer", "") or "")),
            ground_truth=_ground_truth(task),
            extra_info={"ground_truth": _ground_truth(task)},
        )
        r_k = float(reward.compute(ctx).value)
        task_rewards.append(r_k)
    total = float(sum(task_rewards))
    avg = total / len(task_chain) if task_chain else 0.0
    return {
        "chain_id": chain_id,
        "K": len(task_chain),
        "total_reward": total,
        "avg_task_reward": avg,
        "task_rewards": task_rewards,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-parquet", type=Path, required=True)
    ap.add_argument("--vllm-url", default="http://127.0.0.1:18100/v1")
    ap.add_argument("--model", default="/mnt/workspace/ckpt/Qwen3.5-9B")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    import pyarrow.parquet as pq

    from recipe.agentic.envs.generic_benchmark_env import GenericBenchmarkEnv
    from recipe.agentic.reward.exact_match import ExactMatchReward

    table = pq.read_table(args.val_parquet)
    df = table.to_pandas()
    rows: List[Tuple[str, List[Dict[str, Any]]]] = []
    for i, r in df.iterrows():
        cid = str(r.get("chain_id") or r.get("uid") or f"row_{i}")
        rows.append((cid, _as_list(r.get("task_chain"))))
    print(f"[eval-vllm] n_val={len(rows)} url={args.vllm_url} model={args.model}", flush=True)

    env = GenericBenchmarkEnv(
        max_steps=args.max_steps,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        request_timeout=600.0,
        chat_template_kwargs={"enable_thinking": False},
        format_few_shot=True,
    )
    env.setup_once(vllm_base_url=args.vllm_url, vllm_model_name=args.model)
    reward = ExactMatchReward(
        case_sensitive=False,
        strip_punctuation=True,
        numeric_tolerance=1e-3,
        allow_substring=True,
        substring_slack_chars=20,
    )

    t0 = time.time()
    results: List[Dict[str, Any]] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        for cid, tc in rows:
            rec = _eval_one_chain(env, reward, tc, cid)
            results.append(rec)
            print(
                f"  {rec['chain_id']} K={rec['K']} total={rec['total_reward']:.3f} "
                f"avg={rec['avg_task_reward']:.3f}",
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(_eval_one_chain, env, reward, tc, cid): cid
                for cid, tc in rows
            }
            for fut in as_completed(futs):
                rec = fut.result()
                results.append(rec)
                print(
                    f"  {rec['chain_id']} K={rec['K']} total={rec['total_reward']:.3f} "
                    f"avg={rec['avg_task_reward']:.3f}",
                    flush=True,
                )

    n = len(results)
    mean_tot = sum(x["total_reward"] for x in results) / n if n else 0.0
    mean_avg = sum(x["avg_task_reward"] for x in results) / n if n else 0.0
    out = {
        "n_val_chains": n,
        "avg_task_reward": mean_avg,
        "total_reward": mean_tot,
        "elapsed_sec": round(time.time() - t0, 1),
        "vllm_url": args.vllm_url,
        "model": args.model,
        "max_steps": args.max_steps,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "per_chain": results,
    }
    print(
        f"\n[eval-vllm] DONE n={n} avg_task={mean_avg:.4f} total={mean_tot:.4f} "
        f"sec={out['elapsed_sec']}",
        flush=True,
    )
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eval-vllm] wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
