"""把用户提供的 query 数据转换成 verl 训练用的 parquet 文件。

输入格式（jsonl，每行一条）：

    {
      "query": "百万医疗险是什么？",
      "ext_params": {"context_pin": "..."},        # 可选
      "ground_truth": "...",                       # 可选；通常 reward 用不上
      "teacher_trace": "Thought: ... Final ..."    # 可选，给 TeacherSimilarityReward 用
    }

输出 parquet 列（verl RLHFDataset 兼容）：

    data_source         "ai_to_c"
    prompt              List[{"role","content"}]
    raw_prompt          List[{"role","content"}]    ← 与 prompt 同；AgentLoop 用这个
    reward_model        {"style": "model", "ground_truth": ...}
    extra_info          dict (含 teacher_trace / ext_params 等所有原始字段)
    ext_params          dict
    ability             "agentic"
    index               int

用法：

    PYTHONPATH=verl:. python -m recipe.agentic.data.prepare_dataset \\
        --train-input train.jsonl --train-output train.parquet \\
        --val-input val.jsonl    --val-output   val.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger("prepare_dataset")


DATA_SOURCE = "ai_to_c"


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _normalize_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    query = str(row.get("query", "")).strip()
    if not query:
        raise ValueError(f"row {idx} missing 'query'")

    ext_params = row.get("ext_params") or {}
    if not isinstance(ext_params, dict):
        raise ValueError(f"row {idx}: ext_params must be dict, got {type(ext_params)}")

    teacher_trace = row.get("teacher_trace")
    ground_truth = row.get("ground_truth")
    # history: 历史会话轮次，格式同 trace_collect query_pool —— AiToCReactEnv.run() 会注入给 chat_stream
    history = row.get("history") or []
    if not isinstance(history, list):
        history = []

    messages = [{"role": "user", "content": query}]

    extra_info: Dict[str, Any] = {
        "index": idx,
        "query": query,
        "history": history,
        # ext_params 不放进 extra_info：顶层已有同名列（JSON 字符串），
        # RL loop 从 kwargs["ext_params"] 直接读，放进来会导致嵌套空 struct 无法写 parquet
    }
    if teacher_trace is not None:
        extra_info["teacher_trace"] = teacher_trace
    if ground_truth is not None:
        extra_info["ground_truth"] = ground_truth
    # 把原始 row 里其他用户字段也透传出来（不含上述已处理的）
    reserved = {"query", "ext_params", "teacher_trace", "ground_truth", "history"}
    for k, v in row.items():
        if k not in reserved and k not in extra_info:
            extra_info[k] = v

    return {
        "data_source": DATA_SOURCE,
        "prompt": messages,
        "raw_prompt": messages,
        "reward_model": {
            "style": "model",
            "ground_truth": ground_truth if ground_truth is not None else "",
        },
        "extra_info": extra_info,
        "ext_params": dict(ext_params),
        "ability": "agentic",
        "agent_name": "agentic",
        "index": idx,
    }


def _write_parquet(rows: List[Dict[str, Any]], out_path: str) -> None:
    try:
        import pandas as pd  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "prepare_dataset 需要 pandas + pyarrow，请先 pip install pandas pyarrow"
        ) from exc

    import pandas as pd

    # pyarrow 无法写顶层空 struct（如 ext_params={}），仅将其序列化为 JSON 字符串。
    # reward_model 固定 schema {"style": str, "ground_truth": str}，pyarrow 可正常写；
    # extra_info   含 history/index 等复杂结构，pyarrow 可正常写；
    # 两者均保留为原生 dict，让 verl 内部代码（rl_dataset.py / ray_trainer.py）直接 .get() 使用。
    # prompt / raw_prompt 必须保留为原生 list[dict]（tokenizer.apply_chat_template 要求）。
    _SERIALIZE_COLS = ("ext_params",)
    serialized_rows: List[Dict[str, Any]] = []
    for r in rows:
        sr: Dict[str, Any] = {}
        for k, v in r.items():
            if k in _SERIALIZE_COLS:
                sr[k] = json.dumps(v, ensure_ascii=False)
            else:
                sr[k] = v
        serialized_rows.append(sr)

    df = pd.DataFrame(serialized_rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("[prepare_dataset] wrote %d rows to %s", len(rows), out_path)


def convert(input_jsonl: str, output_parquet: str, offset: int = 0) -> int:
    items = _load_jsonl(input_jsonl)
    rows = [_normalize_row(it, offset + i) for i, it in enumerate(items)]
    _write_parquet(rows, output_parquet)
    return len(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare train/val parquet for ai_to_c agentic RL")
    ap.add_argument("--train-input", required=True, help="train jsonl path")
    ap.add_argument("--train-output", required=True, help="train parquet output path")
    ap.add_argument("--val-input", help="val jsonl path (optional)")
    ap.add_argument("--val-output", help="val parquet output path (optional)")
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    n_train = convert(args.train_input, args.train_output, offset=0)
    logger.info("train rows: %d", n_train)

    if args.val_input and args.val_output:
        n_val = convert(args.val_input, args.val_output, offset=n_train)
        logger.info("val rows: %d", n_val)
    return 0


# ---------------------------------------------------------------------------
# Minimal self-test（不需要 pandas/pyarrow）：仅验证 _normalize_row
# ---------------------------------------------------------------------------


def _self_test() -> None:
    row = _normalize_row(
        {
            "query": "百万医疗?",
            "ext_params": {"context_pin": "user42"},
            "teacher_trace": "x",
            "ground_truth": None,
            "custom_field": "y",
        },
        idx=0,
    )
    assert row["data_source"] == "ai_to_c"
    assert row["prompt"] == [{"role": "user", "content": "百万医疗?"}]
    assert row["raw_prompt"] == row["prompt"]
    assert row["ext_params"] == {"context_pin": "user42"}
    assert row["extra_info"]["teacher_trace"] == "x"
    assert row["extra_info"]["custom_field"] == "y"
    assert row["extra_info"]["ext_params"] == {"context_pin": "user42"}
    assert row["ability"] == "agentic"
    print("prepare_dataset self_test OK")




# ═══════════════════════════════════════════════════════════════════════════════
# STHARC chain 数据集构建
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_stharc_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """把 STHARC chain record 转成 verl parquet 行格式。

    输入 row 格式（来自 stharc.task_chain 的输出）：
        {
          "uid":        "chain_00001",
          "chain_id":   "chain_00001",
          "agent_name": "stharc",
          "task_chain": '[{"raw_prompt": [...], "ext_params": {...}}, ...]',
        }
    """
    chain_id = str(row.get("chain_id") or row.get("uid") or f"chain_{idx:05d}")
    uid = str(row.get("uid") or chain_id)
    chain_g = int(row.get("chain_g") or 0)
    task_chain_raw = row.get("task_chain") or "[]"

    # 从 task_chain 中取第一个任务的 raw_prompt 作为 parquet 的 prompt 列
    # （verl RLHFDataset 要求必须有 prompt 列）
    if isinstance(task_chain_raw, str):
        try:
            task_chain_list = json.loads(task_chain_raw)
        except (ValueError, TypeError):
            task_chain_list = []
    else:
        task_chain_list = list(task_chain_raw)

    if task_chain_list:
        first_task = task_chain_list[0]
        raw_prompt = first_task.get("raw_prompt") or first_task.get("messages") or []
        if isinstance(raw_prompt, str):
            raw_prompt = [{"role": "user", "content": raw_prompt}]
    else:
        raw_prompt = [{"role": "user", "content": ""}]

    extra_info: Dict[str, Any] = {
        "index":    idx,
        "chain_id": chain_id,
    }
    for k, v in row.items():
        if k not in {"uid", "chain_id", "agent_name", "task_chain"} and k not in extra_info:
            extra_info[k] = v

    return {
        "data_source": DATA_SOURCE,
        "uid":         uid,
        "chain_id":    chain_id,
        "chain_g":     chain_g,
        "prompt":      raw_prompt,
        "raw_prompt":  raw_prompt,
        "task_chain":  task_chain_raw if isinstance(task_chain_raw, str)
                       else json.dumps(task_chain_raw, ensure_ascii=False),
        "reward_model": {"style": "model", "ground_truth": ""},
        "extra_info":  extra_info,
        "ext_params":  json.dumps({}, ensure_ascii=False),
        "ability":     "agentic",
        "agent_name":  "stharc",
        "index":       idx,
    }


def _validate_paths(inputs: List[str], outputs: List[str]) -> None:
    """Reject aliases, including symlinks and existing hard links, before IO."""
    def same(a: str, b: str) -> bool:
        pa, pb = Path(a), Path(b)
        return pa.resolve() == pb.resolve() or (
            pa.exists() and pb.exists() and pa.samefile(pb)
        )

    for i, path in enumerate(inputs):
        if any(same(path, other) for other in inputs[:i]):
            raise ValueError("train and validation inputs must be different paths")
    for i, path in enumerate(outputs):
        if any(same(path, other) for other in inputs + outputs[:i]):
            raise ValueError("output paths must differ from inputs and each other")


def _dataset_keys(records: List[Dict[str, Any]]) -> set[str]:
    from recipe.agentic.stharc.task_chain import record_task_keys

    if not records:
        raise ValueError("dataset must be non-empty")
    keys: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("dataset records must be objects")
        keys.update(record_task_keys(record))
    return keys


def _validate_disjoint(train: List[Dict[str, Any]], val: List[Dict[str, Any]]) -> None:
    if _dataset_keys(train) & _dataset_keys(val):
        raise ValueError("train/validation overlap in task content or source identity")


def _validate_val_ratio(ratio: float) -> None:
    import math

    if not math.isfinite(ratio) or not 0 < ratio < 1:
        raise ValueError("val_ratio must be finite and strictly between 0 and 1")


def _split_stharc_records(
    records: List[Dict[str, Any]], val_ratio: float, seed: Optional[int],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split connected components; neither whole chains nor replicas can cross."""
    import random
    from recipe.agentic.stharc.task_chain import record_task_keys

    _validate_val_ratio(val_ratio)
    _dataset_keys(records)
    parents = list(range(len(records)))

    def root(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    owner: Dict[str, int] = {}
    for i, record in enumerate(records):
        for key in record_task_keys(record):
            if key in owner:
                parents[root(i)] = root(owner[key])
            else:
                owner[key] = i
    components: Dict[int, List[Dict[str, Any]]] = {}
    for i, record in enumerate(records):
        components.setdefault(root(i), []).append(record)
    groups = list(components.values())
    if len(groups) < 2:
        raise ValueError("cannot split: fewer than two independent task components")
    random.Random(seed).shuffle(groups)
    # Closest non-empty prefix; exact ratios are not possible for all group sizes.
    target = len(records) * val_ratio
    cumulative = 0
    candidates = []
    for i, group in enumerate(groups[:-1], 1):
        cumulative += len(group)
        candidates.append((abs(cumulative - target), i))
    cut = min(candidates)[1]
    val = [record for group in groups[:cut] for record in group]
    train = [record for group in groups[cut:] for record in group]
    _validate_disjoint(train, val)
    return train, val


def _build_stharc_records(
    input_jsonl: str, K: int, G: int, strategy: str, seed: int,
) -> List[Dict[str, Any]]:
    from recipe.agentic.stharc.task_chain import (
        build_task_chains_random, build_task_chains_from_traces,
        build_grpo_records_from_prebuilt_chains,
    )

    items = _load_jsonl(input_jsonl)
    _dataset_keys(items)
    if any("task_chain" in row for row in items):
        if not all("task_chain" in row for row in items):
            raise ValueError("cannot mix prebuilt chains and raw tasks")
        records = build_grpo_records_from_prebuilt_chains(items, G=G, seed=seed)
    elif strategy == "traces":
        records = build_task_chains_from_traces(input_jsonl, K=K, G=G, seed=seed)
    elif strategy == "random":
        records = build_task_chains_random(items, K=K, G=G, seed=seed)
    else:
        raise ValueError("unknown chain strategy")
    _dataset_keys(records)
    return records


def convert_stharc(
    input_jsonl: str,
    output_parquet: str,
    offset: int = 0,
    K: int = 3,
    G: int = 4,
    strategy: str = "traces",
    seed: int = 42,
) -> int:
    """把 query jsonl 转成 STHARC 训练用的 parquet 文件。

    Args:
        input_jsonl:    输入 jsonl 路径（raw queries 或 filtered_traces.jsonl）
        output_parquet: 输出 parquet 路径
        offset:         全局行偏移（train/val 拼接时使用）
        K:              chain 长度
        G:              GRPO group size（每条 chain 重复 G 次）
        strategy:       chain 构建策略（"random" / "traces"）
        seed:           随机种子

    Returns:
        实际写入的行数
    """
    _validate_paths([input_jsonl], [output_parquet])
    records = _build_stharc_records(input_jsonl, K, G, strategy, seed)

    rows = [_normalize_stharc_row(r, offset + i) for i, r in enumerate(records)]
    _write_parquet(rows, output_parquet)
    return len(rows)


def _parse_stharc_args(argv: List[str] | None = None) -> "argparse.Namespace":
    ap = argparse.ArgumentParser(
        description="Prepare STHARC chain train/val parquet for STHARC-GRPO"
    )
    ap.add_argument("--train-input",  required=True, help="train jsonl/traces path")
    ap.add_argument("--train-output", required=True, help="train parquet output path")
    ap.add_argument("--val-input",    help="独立 val jsonl/traces 路径；自动拆分时省略此参数")
    ap.add_argument("--val-output",   help="val parquet output path (optional)")
    ap.add_argument("--val-ratio",    type=float, default=0.05,
                    help="从 train-input 中按比例拆出 val（仅 val-input 未指定时生效，默认 0.05）")
    ap.add_argument("--K",            type=int, default=3, help="chain 长度（默认 3）")
    ap.add_argument("--G",            type=int, default=4, help="GRPO group size（默认 4）")
    ap.add_argument(
        "--strategy",
        choices=["random", "traces"],
        default="traces",
        help="chain 构建策略（random | traces）",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args(argv)


def main_stharc(argv: List[str] | None = None) -> int:
    """STHARC 数据集构建入口。

    用法::

        PYTHONPATH=verl:. python3 -m recipe.agentic.data.prepare_dataset stharc \\
            --train-input research/filtered_traces.jsonl \\
            --train-output /data/stharc_train.parquet \\
            --val-output   /data/stharc_val.parquet \\
            --val-ratio 0.05 \\
            --K 3 --G 4 --strategy traces

    val 优先级：
      1. 若同时指定 --val-input + --val-output → 使用独立 val 文件
      2. 若只指定 --val-output（无 --val-input）→ 从 train-input 按 --val-ratio 拆分
      3. 若都不指定 → 只生成 train
    """
    from recipe.agentic.stharc.task_chain import _validate_sizes

    args = _parse_stharc_args(argv)
    _validate_sizes(K=args.K, G=args.G)
    _validate_val_ratio(args.val_ratio)
    if args.val_input and not args.val_output:
        raise ValueError("val-input requires val-output")
    _validate_paths(
        [args.train_input] + ([args.val_input] if args.val_input else []),
        [args.train_output] + ([args.val_output] if args.val_output else []),
    )
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # ── 情况 1：独立 val 文件 ──────────────────────────────────────────
    if args.val_input and args.val_output:
        _validate_disjoint(_load_jsonl(args.train_input), _load_jsonl(args.val_input))
        train_records = _build_stharc_records(
            args.train_input, args.K, args.G, args.strategy, args.seed)
        val_records = _build_stharc_records(
            args.val_input, args.K, args.G, args.strategy, args.seed)
        _validate_disjoint(train_records, val_records)
        train_rows = [_normalize_stharc_row(r, i) for i, r in enumerate(train_records)]
        val_rows = [_normalize_stharc_row(r, len(train_rows) + i)
                    for i, r in enumerate(val_records)]
        _write_parquet(train_rows, args.train_output)
        _write_parquet(val_rows, args.val_output)
        logger.info("[stharc] train rows: %d  val rows: %d", len(train_rows), len(val_rows))
        return 0

    # ── 情况 2：从 train-input 按比例拆分 val ────────────────────────
    if args.val_output:
        all_records = _build_stharc_records(
            args.train_input, args.K, args.G, args.strategy, args.seed)
        train_records, val_records = _split_stharc_records(
            all_records, args.val_ratio, args.seed)

        train_rows = [_normalize_stharc_row(r, i)          for i, r in enumerate(train_records)]
        val_rows   = [_normalize_stharc_row(r, len(train_rows) + i) for i, r in enumerate(val_records)]

        _write_parquet(train_rows, args.train_output)
        _write_parquet(val_rows,   args.val_output)
        logger.info("[stharc] train rows: %d  val rows: %d (ratio=%.2f)",
                    len(train_rows), len(val_rows), args.val_ratio)
        return 0

    # ── 情况 3：只生成 train ──────────────────────────────────────────
    n_train = convert_stharc(
        args.train_input, args.train_output,
        offset=0, K=args.K, G=args.G, strategy=args.strategy, seed=args.seed,
    )
    logger.info("[stharc] train rows: %d  (no val generated)", n_train)
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# STHARC 预构建变长 chain（idea2 公开数据集）→ parquet
# ═══════════════════════════════════════════════════════════════════════════════
#
# 与 convert_stharc()（策略 1/2/3，内部按固定 K 切分零散 query 列表）的关键区别：
# 这里的输入本身就是一条条完整的、长度各异的 chain（由
# research/dataset_explore/idea2_datasets/ 下的生成器产出），本函数只做
# "record -> parquet 行"的格式转换 + G 次复制，不接受、也不需要 K 参数
# （对齐 IDEA2_IMPLEMENTATION_PLAN.md §0 的硬约束）。


def convert_stharc_prebuilt_records(
    source_records: List[Dict[str, Any]],
    output_parquet: str,
    offset: int = 0,
    G: int = 4,
    seed: Optional[int] = None,
    *,
    val_source_records: Optional[List[Dict[str, Any]]] = None,
    val_output_parquet: Optional[str] = None,
    val_ratio: float = 0.05,
) -> int:
    """Convert prebuilt chains, optionally writing an isolated validation set.

    Supply val_source_records for independently constructed validation chains,
    or only val_output_parquet to split by connected task components. Separate
    single-output calls cannot validate each other's inputs; use this paired
    interface whenever producing train and validation together. Returns the
    number of training rows (validation rows are not included).

    Args:
        source_records: idea2 生成器产出的 record 列表（每条已是完整变长 chain）
        output_parquet: 输出 parquet 路径
        offset:         全局行偏移（train/val 拼接时使用）
        G:              GRPO group size（每条 chain 重复 G 次）
        seed:           打乱 source_records 顺序用的随机种子（可选）

    Returns:
        实际写入的行数
    """
    from recipe.agentic.stharc.task_chain import build_grpo_records_from_prebuilt_chains

    _validate_val_ratio(val_ratio)
    if val_source_records is not None and not val_output_parquet:
        raise ValueError("val_source_records requires val_output_parquet")
    _validate_paths([], [output_parquet] + ([val_output_parquet] if val_output_parquet else []))
    _dataset_keys(source_records)
    if val_source_records is not None:
        _validate_disjoint(source_records, val_source_records)
    elif val_output_parquet:
        source_records, val_source_records = _split_stharc_records(source_records, val_ratio, seed)
    records = build_grpo_records_from_prebuilt_chains(source_records, G=G, seed=seed)
    rows = [_normalize_stharc_row(r, offset + i) for i, r in enumerate(records)]
    val_rows = None
    if val_source_records is not None:
        val_records = build_grpo_records_from_prebuilt_chains(val_source_records, G=G, seed=seed)
        _validate_disjoint(records, val_records)
        val_rows = [_normalize_stharc_row(r, offset + len(rows) + i)
                    for i, r in enumerate(val_records)]
    _write_parquet(rows, output_parquet)
    if val_rows is not None:
        _write_parquet(val_rows, val_output_parquet)
    return len(rows)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--self-test":
        _self_test()
        sys.exit(0)
    if len(sys.argv) >= 2 and sys.argv[1] == "stharc":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        sys.exit(main_stharc())
    sys.exit(main())