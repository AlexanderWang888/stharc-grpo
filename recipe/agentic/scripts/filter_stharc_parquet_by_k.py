#!/usr/bin/env python3
"""按 task_chain 长度过滤 STHARC parquet，供正式实验在「不截断 packing」预算内跑完。

示例：
  python3 recipe/agentic/scripts/filter_stharc_parquet_by_k.py \\
      --src data/stharc_public_smoke \\
      --dst data/stharc_public_smoke_k8 \\
      --max-k 8

若 val 没有 K<=max-k 的链（smoke 常见），默认回退保留最短 K 的那一组行，避免脚本中断。
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def chain_k(tc) -> int:
    if isinstance(tc, str):
        tc = json.loads(tc)
    return len(tc) if isinstance(tc, list) else 1


def filter_file(
    src: Path,
    dst: Path,
    max_k: int,
    *,
    fallback_shortest: bool = False,
) -> tuple[int, int, str]:
    table = pq.read_table(src)
    tcs = table.column("task_chain").to_pylist()
    ks = [chain_k(tc) for tc in tcs]
    keep = [i for i, k in enumerate(ks) if k <= max_k]
    note = f"K<={max_k}"
    if not keep:
        if not fallback_shortest:
            raise SystemExit(f"ERROR: no rows with K<={max_k} in {src}")
        min_k = min(ks)
        keep = [i for i, k in enumerate(ks) if k == min_k]
        note = f"FALLBACK shortest K={min_k} (no K<={max_k})"
        print(f"[warn] {src.name}: {note}; keeping {len(keep)} rows")
    filtered = table.take(pa.array(keep, type=pa.int64()))
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(filtered, dst)
    return len(tcs), len(keep), note


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="源目录（含 train.parquet / val.parquet）")
    ap.add_argument("--dst", required=True, help="输出目录")
    ap.add_argument("--max-k", type=int, required=True, help="保留 K<=max-k 的链")
    ap.add_argument(
        "--strict-val",
        action="store_true",
        help="val 无 K<=max-k 时直接报错（默认回退最短链）",
    )
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if dst.resolve() == src.resolve():
        raise SystemExit("ERROR: --dst must differ from --src")

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    for name in ("train.parquet", "val.parquet"):
        sp = src / name
        if not sp.exists():
            print(f"[skip] missing {sp}")
            continue
        fallback = name == "val.parquet" and not args.strict_val
        n0, n1, note = filter_file(sp, dst / name, args.max_k, fallback_shortest=fallback)
        print(f"[ok] {name}: {n0} -> {n1} rows ({note})")

    print(f"done -> {dst}")


if __name__ == "__main__":
    main()
