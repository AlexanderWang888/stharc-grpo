#!/usr/bin/env python3
"""汇总 traces 里哪些 chain / task 发生了截断（只读检测，不改训练）。

用法：
  python3 recipe/agentic/scripts/summarize_truncation_traces.py \\
      --trace-dir /mnt/workspace/stharc/runs/public/rollout_logs/.../traces
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--limit", type=int, default=50, help="最多打印多少条被截断的 chain")
    args = ap.parse_args()

    root = Path(args.trace_dir)
    files = sorted(root.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no jsonl under {root}")

    n_chain = 0
    n_trunc_chain = 0
    n_trunc_task = 0
    examples = []

    for fp in files:
        with fp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                n_chain += 1
                tr = obj.get("truncation") or {}
                idxs = tr.get("truncated_task_idxs") or []
                # fallback: scan tasks
                if not idxs:
                    for t in obj.get("tasks") or []:
                        tt = t.get("truncation") or {}
                        if tt.get("any_trunc"):
                            idxs.append(t.get("task_idx"))
                hit = bool(tr.get("any_trunc_hit")) or bool(idxs)
                if not hit:
                    continue
                n_trunc_chain += 1
                n_trunc_task += len(idxs)
                if len(examples) < args.limit:
                    examples.append(
                        {
                            "file": fp.name,
                            "group_id": obj.get("group_id"),
                            "chain_g": obj.get("chain_g"),
                            "data_source": obj.get("data_source"),
                            "truncated_task_idxs": idxs,
                            "packing_trunc_hit": tr.get("packing_trunc_hit"),
                            "exec_trunc_hit": tr.get("exec_trunc_hit"),
                            "n_task_resp_trunc": tr.get("n_task_resp_trunc"),
                            "n_task_ctx_trunc": tr.get("n_task_ctx_trunc"),
                            "chain_resp_trunc": tr.get("chain_resp_trunc"),
                            "n_obs_char_trunc": tr.get("n_obs_char_trunc"),
                            "n_context_overflow": tr.get("n_context_overflow"),
                        }
                    )

    print(f"chains={n_chain} truncated_chains={n_trunc_chain} "
          f"rate={n_trunc_chain / max(n_chain, 1):.3f} truncated_task_mentions={n_trunc_task}")
    print("--- truncated examples ---")
    for ex in examples:
        print(json.dumps(ex, ensure_ascii=False))


if __name__ == "__main__":
    main()
