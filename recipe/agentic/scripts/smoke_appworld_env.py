#!/usr/bin/env python3
"""AppWorld 环境冒烟（不启 Ray/训练）。在训练机上先跑通这个再开 GRPO。

用法（在 training_code_public 目录）::

    export APPWORLD_ROOT=/mnt/workspace/agent_train/ai2c_workspace/research/appworld_explore
    python3 recipe/agentic/scripts/smoke_appworld_env.py \\
        --parquet data/stharc_appworld_smoke/train.parquet

可选：接真实 vLLM（验证 LLM 路径）::

    python3 recipe/agentic/scripts/smoke_appworld_env.py \\
        --parquet data/stharc_appworld_smoke/train.parquet \\
        --vllm-url http://127.0.0.1:8000/v1 \\
        --vllm-model /mnt/workspace/ckpt/Qwen3.5-9B \\
        --max-steps 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback


def _load_first_task(parquet_path: str) -> tuple[str, dict]:
    import pyarrow.parquet as pq

    df = pq.read_table(parquet_path, columns=["task_chain", "chain_g"]).to_pandas()
    for _, row in df.iterrows():
        if int(row.get("chain_g") or 0) != 0:
            continue
        tc = row["task_chain"]
        if isinstance(tc, str):
            tc = json.loads(tc)
        if not tc:
            continue
        t0 = tc[0]
        ep = t0.get("ext_params") or {}
        if isinstance(ep, str):
            ep = json.loads(ep) if ep.strip() else {}
        tid = str(ep.get("appworld_task_id") or "").strip()
        if not tid:
            continue
        raw = t0.get("raw_prompt") or t0.get("query") or ""
        if isinstance(raw, list):
            query = ""
            for m in raw:
                if isinstance(m, dict) and m.get("role") == "user":
                    query = str(m.get("content") or "")
                    break
        else:
            query = str(raw)
        return query, dict(ep)
    raise RuntimeError(f"no task with appworld_task_id in {parquet_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/stharc_appworld_smoke/train.parquet")
    ap.add_argument(
        "--appworld-root",
        default=os.environ.get(
            "APPWORLD_ROOT",
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../../../../appworld_explore")
            ),
        ),
    )
    ap.add_argument("--vllm-url", default="")
    ap.add_argument("--vllm-model", default="")
    ap.add_argument("--max-steps", type=int, default=1)
    ap.add_argument("--task-id", default="", help="override; default=first from parquet")
    args = ap.parse_args()

    print("=== AppWorld smoke ===")
    print("APPWORLD_ROOT =", args.appworld_root)
    tasks_dir = os.path.join(args.appworld_root, "data", "tasks")
    print("data/tasks exists =", os.path.isdir(tasks_dir))
    os.environ["APPWORLD_ROOT"] = args.appworld_root

    print("\n[1/4] import appworld ...")
    try:
        from appworld import AppWorld  # noqa: F401
        import appworld

        print("  ok:", getattr(appworld, "__file__", appworld))
    except Exception:
        traceback.print_exc()
        print(
            "\nFAIL: install GitHub main (NOT PyPI 0.1.3):\n"
            '  pip install "git+https://github.com/StonyBrookNLP/appworld.git"\n'
            "  appworld install\n"
            "  # if pydantic was downgraded: pip install 'pydantic>=2.12' 'fastapi>=0.115'"
        )
        return 2

    if args.task_id:
        query = "smoke query"
        ep = {"appworld_task_id": args.task_id}
    else:
        print("\n[2/4] load task from", args.parquet)
        query, ep = _load_first_task(args.parquet)
    tid = ep["appworld_task_id"]
    print("  task_id =", tid)
    print("  task dir exists =", os.path.isdir(os.path.join(tasks_dir, tid)))
    print("  query =", query[:120].replace("\n", " "))

    print("\n[3/4] AppWorld(task_id).execute + evaluate ...")
    try:
        with AppWorld(
            task_id=tid, experiment_name="stharc_manual_smoke", ground_truth_mode="full"
        ) as world:
            out = world.execute("print(1+1)")
            ev = world.evaluate().to_dict()
        print("  execute =", repr(out)[:200])
        print("  eval.success =", ev.get("success"), "keys =", list(ev)[:8])
    except Exception:
        traceback.print_exc()
        print("\nFAIL: AppWorld sandbox open/execute/evaluate")
        return 3

    if not args.vllm_url:
        print("\n[4/4] skip AppWorldEnv+vLLM (pass --vllm-url to enable)")
        print("\nPASS (sandbox only)")
        return 0

    print("\n[4/4] AppWorldEnv.run with vLLM ...")
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
    from recipe.agentic.envs.appworld_env import AppWorldEnv

    env = AppWorldEnv(max_steps=args.max_steps, appworld_root=args.appworld_root)
    env.setup_once(vllm_base_url=args.vllm_url, vllm_model_name=args.vllm_model)
    result = env.run(query=query, ext_params=ep)
    info = result.env_info or {}
    print("  status =", info.get("status"))
    print("  error  =", info.get("error"))
    print("  steps  =", info.get("num_steps"), "tools =", info.get("num_tool_calls"))
    print("  success=", info.get("appworld_success"))
    if info.get("status") == "error":
        print("\nFAIL: AppWorldEnv.run")
        return 4
    print("\nPASS (env + vLLM)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
