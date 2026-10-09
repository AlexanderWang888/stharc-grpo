#!/usr/bin/env python3
"""按 extra_info.data_source 过滤 STHARC parquet（机制消融用自造子集）。

示例（仅 diagnostic #1/#2）：
  python3 recipe/agentic/scripts/filter_stharc_parquet_by_source.py \\
      --src data/stharc_public_smoke \\
      --dst data/stharc_diagnostic_smoke \\
      --sources credit_diagnostic_memory,credit_diagnostic_skill_modify

示例（自造合成、去掉 musique/textcraft）：
  python3 recipe/agentic/scripts/filter_stharc_parquet_by_source.py \\
      --src data/stharc_public_smoke \\
      --dst data/stharc_synthetic_smoke \\
      --sources credit_diagnostic_memory,credit_diagnostic_skill_modify,skill_delete,memory_invalidate
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def filter_file(src: Path, dst: Path, sources: set[str]) -> tuple[int, int]:
    table = pq.read_table(src)
    extras = table.column("extra_info").to_pylist()
    keep = [
        i
        for i, ex in enumerate(extras)
        if (ex or {}).get("data_source", "") in sources
    ]
    if not keep:
        raise SystemExit(f"ERROR: no rows with data_source in {sorted(sources)} in {src}")
    filtered = table.take(pa.array(keep, type=pa.int64()))
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(filtered, dst)
    return len(extras), len(keep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument(
        "--sources",
        required=True,
        help="comma-separated data_source allowlist",
    )
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    sources = {s.strip() for s in args.sources.split(",") if s.strip()}
    if not sources:
        raise SystemExit("ERROR: empty --sources")
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
        n0, n1 = filter_file(sp, dst / name, sources)
        print(f"[ok] {name}: {n0} -> {n1} rows  sources={sorted(sources)}")

    print(f"done -> {dst}")


if __name__ == "__main__":
    main()
