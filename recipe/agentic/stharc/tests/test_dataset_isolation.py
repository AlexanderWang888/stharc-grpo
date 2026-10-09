"""Low-dependency regressions: no model runtime or parquet engine required."""
import json
import os
import random
from collections import Counter

import pytest

from recipe.agentic.data import prepare_dataset as data
from recipe.agentic.stharc import task_chain as chain


def task(query, **fields):
    return {"query": query, **fields}


def record(*queries, **fields):
    return {"task_chain": [task(q) for q in queries], **fields}


def save(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return str(path)


@pytest.fixture
def writes(monkeypatch):
    calls = []
    monkeypatch.setattr(data, "_write_parquet", lambda rows, path: calls.append((rows, path)))
    return calls


def assert_isolated(train, val):
    assert train and val
    assert not data._dataset_keys(train) & data._dataset_keys(val)


def test_transitive_components_and_replicas():
    source = [record("a", "b"), record("b", "c"), record("c", "d"), record("z")]
    rows = chain.build_grpo_records_from_prebuilt_chains(source, G=3)
    state = random.getstate()
    train, val = data._split_stharc_records(rows, 0.25, 7)
    assert random.getstate() == state
    assert_isolated(train, val)
    assert sorted([len(train), len(val)]) == [3, 9]
    assert all(n == 3 for n in Counter(r["task_chain"] for r in train + val).values())
    assert (train, val) == data._split_stharc_records(rows, 0.25, 7)


@pytest.mark.parametrize("ratio", [0, 1, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_ratios(ratio):
    with pytest.raises(ValueError, match="val_ratio"):
        data._split_stharc_records([record("a"), record("b")], ratio, 42)


@pytest.mark.parametrize("rows", [[], [record("a")], [record("a", "b"), record("b", "c")]])
def test_unsplittable(rows):
    with pytest.raises(ValueError):
        data._split_stharc_records(rows, 0.5, 42)


@pytest.mark.parametrize("field", ["uid", "chain_id", "task_id"])
def test_auto_ids_are_not_source_ids(field):
    data._validate_disjoint([task("a", **{field: "0"})], [task("b", **{field: "0"})])


@pytest.mark.parametrize("field", ["query_id", "source_id", "source_task_id", "source_question_id", "original_id", "musique_id", "appworld_task_id"])
@pytest.mark.parametrize("nested", [False, True])
def test_real_source_ids_detect_rewording(field, nested):
    fields = {field: "same"}
    if nested:
        fields = {"ext_params": json.dumps(fields)}
    with pytest.raises(ValueError, match="overlap"):
        data._validate_disjoint([task("a", **fields)], [task("b", **fields)])


def test_content_identity_ignores_answers_and_whitespace():
    with pytest.raises(ValueError, match="overlap"):
        data._validate_disjoint([task(" a ", ground_truth="x")], [task("a", ground_truth="y")])


@pytest.mark.parametrize("scope", ["dataset", "source_dataset", "data_source"])
def test_namespace_inheritance(scope):
    left = {scope: "left", "task_chain": [task("a", source_id="0")]}
    right = {scope: "right", "task_chain": [task("b", source_id="0")]}
    data._validate_disjoint([left], [right])
    a, b = [chain.build_grpo_records_from_prebuilt_chains([r], G=1)[0] for r in (left, right)]
    data._validate_disjoint([a], [b])
    for raw, built in ((left, a), (right, b)):
        assert chain.record_task_keys(raw) <= chain.record_task_keys(built)


def test_explicit_empty_namespace_survives_conversion():
    raw = {"data_source": "parent", "task_chain": [task("a", source_id="0", source_dataset="")]}
    converted = chain.to_stharc_task_chain(raw)[0]
    assert converted["source_dataset"] == ""
    assert chain.task_identity_keys(converted) == chain.task_identity_keys(raw["task_chain"][0])


def test_record_identity_survives_normalization():
    raw = record("a", source_dataset="original", source_id="shared")
    built = chain.build_grpo_records_from_prebuilt_chains([raw], G=1)[0]
    normalized = data._normalize_stharc_row(built, 0)
    assert chain.record_task_keys(raw) <= chain.record_task_keys(normalized)
    with pytest.raises(ValueError, match="overlap"):
        data._validate_disjoint([normalized], [task("reworded", source_dataset="original", source_id="shared")])


def test_prompt_only_conversion():
    entry = chain.to_stharc_task_chain({"task_chain": [{"prompt": "question"}]})[0]
    assert entry["raw_prompt"] == [{"role": "user", "content": "question"}]


def msg(role, content):
    return {"role": role, "content": content}


@pytest.mark.parametrize("boundary", ["current", "  current  ", "## 用户当前问题\ncurrent", "## 用户当前问题：\ncurrent"])
def test_unique_history_boundary(boundary):
    prefix = [msg("user", "old"), msg("assistant", "old answer")]
    messages = [msg("system", "system"), *prefix, msg("user", boundary), msg("assistant", "SECRET")]
    assert chain._get_history(task("current", messages=messages)) == prefix
    assert chain._get_history(task("current", messages=messages[3:])) == []


@pytest.mark.parametrize("messages", [
    [msg("user", "prefix current suffix"), msg("assistant", "SECRET")],
    [msg("user", "current"), msg("assistant", "SECRET"), msg("user", "current")],
    [msg("user", "unrelated"), msg("assistant", "SECRET")],
    [None, 1, msg("user", None)],
    "not a list",
])
def test_ambiguous_history_fails_closed(messages):
    assert chain._get_history(task("current", messages=messages)) == []


def test_explicit_history_and_prebuilt_preservation():
    prefix = [msg("user", "old"), msg("assistant", "answer")]
    raw = task("current", history=[None, msg("tool", "ignored"), *prefix], baseline_answer="base", ground_truth="gold")
    entry = chain.to_stharc_task_chain({"task_chain": [raw]})[0]
    assert entry["history"] == prefix
    assert entry["baseline_answer"] == "base"
    assert entry["ground_truth"] == "gold"
    with pytest.raises(ValueError, match="history"):
        chain._get_history(task("current", history="invalid"))


@pytest.mark.parametrize("kind", ["random", "skill", "traces", "prebuilt"])
def test_all_builders_preserve_identity_and_history(tmp_path, kind):
    rows = [task(str(i), source_id=f"source{i}", data_source="dataset", query_id=f"s{i}_0_r1",
                 messages=[msg("user", "old"), msg("assistant", "old answer"), msg("user", str(i)), msg("assistant", "SECRET")]) for i in range(4)]
    if kind == "random":
        built = chain.build_task_chains_random(rows, K=1, G=2)
    elif kind == "skill":
        built = chain.build_task_chains_by_skill_dep(rows, {}, K=1, G=2)
    elif kind == "traces":
        built = chain.build_task_chains_from_traces(save(tmp_path / "traces.jsonl", rows), K=1, G=2)
    else:
        built = chain.build_grpo_records_from_prebuilt_chains([{"task_chain": [r]} for r in rows], G=2)
    assert data._dataset_keys(rows) <= data._dataset_keys(built)
    assert len(built) == 8
    for row in built:
        for entry in json.loads(row["task_chain"]):
            assert entry["history"] == [msg("user", "old"), msg("assistant", "old answer")]
    assert_isolated(*data._split_stharc_records(built, 0.5, 42))


@pytest.mark.parametrize("kind", ["random", "traces", "prebuilt"])
@pytest.mark.parametrize("independent", [False, True])
def test_cli_outputs_are_isolated(tmp_path, writes, kind, independent):
    rows = [record(str(i)) if kind == "prebuilt" else task(str(i), query_id=f"s{i}_0_r1") for i in range(8)]
    args = ["--train-input", save(tmp_path / "train.jsonl", rows[:4] if independent else rows),
            "--train-output", str(tmp_path / "train.parquet"),
            "--val-output", str(tmp_path / "val.parquet"),
            "--K", "1", "--G", "3", "--strategy", "traces" if kind == "traces" else "random"]
    if independent:
        args += ["--val-input", save(tmp_path / "val.jsonl", rows[4:])]
    assert data.main_stharc(args) == 0
    assert len(writes) == 2
    train, val = [call[0] for call in writes]
    assert_isolated(train, val)
    assert len(train) + len(val) == 24
    assert all(n == 3 for n in Counter(r["task_chain"] for r in train + val).values())
    assert [r["index"] for r in train + val] == list(range(24))


@pytest.mark.parametrize("independent", [False, True])
def test_paired_prebuilt_outputs(tmp_path, writes, independent):
    source = [record("a", "b"), record("b", "c"), record("z")]
    kwargs = {"val_source_records": [record("v")]} if independent else {}
    count = data.convert_stharc_prebuilt_records(
        source, str(tmp_path / "train.parquet"), offset=7, G=2, seed=5,
        val_output_parquet=str(tmp_path / "val.parquet"), **kwargs)
    train, val = [call[0] for call in writes]
    assert count == len(train)
    assert_isolated(train, val)
    assert [r["index"] for r in train + val] == list(range(7, 7 + len(train) + len(val)))
    assert all(n == 2 for n in Counter(r["task_chain"] for r in train + val).values())


@pytest.mark.parametrize("alias", ["same", "relative", "symlink", "hardlink"])
@pytest.mark.parametrize("conflict", ["inputs", "input_output", "outputs"])
def test_path_aliases_rejected(tmp_path, writes, alias, conflict):
    source = tmp_path / "source.jsonl"
    save(source, [task("a")])
    other = tmp_path / "alias.jsonl"
    if alias == "same":
        other = source
    elif alias == "relative":
        other = os.path.relpath(source)
    elif alias == "symlink":
        other.symlink_to(source)
    else:
        os.link(source, other)
    inputs, outputs = {
        "inputs": ([str(source), str(other)], []),
        "input_output": ([str(source)], [str(other)]),
        "outputs": ([], [str(source), str(other)]),
    }[conflict]
    with pytest.raises(ValueError, match="paths"):
        data._validate_paths(inputs, outputs)
    assert not writes


@pytest.mark.parametrize("overlap", ["content", "source", "record_source"])
def test_cli_rejects_source_overlap_before_build(tmp_path, writes, monkeypatch, overlap):
    if overlap == "content":
        left, right = task("same"), task("same", ground_truth="different")
    elif overlap == "source":
        left, right = task("a", source_id="same"), task("b", source_id="same")
    else:
        left, right = record("a", source_id="same"), record("b", source_id="same")
    def must_not_build(*args):
        pytest.fail("raw overlap must be rejected before construction or tail dropping")
    monkeypatch.setattr(data, "_build_stharc_records", must_not_build)
    with pytest.raises(ValueError, match="overlap"):
        data.main_stharc(["--train-input", save(tmp_path / "t.jsonl", [left]),
                          "--val-input", save(tmp_path / "v.jsonl", [right]),
                          "--train-output", str(tmp_path / "t.parquet"),
                          "--val-output", str(tmp_path / "v.parquet")])
    assert not writes
