"""CPU contract tests using real source AST, without importing distributed backends.

Run: python -m pytest verl/trainer/ppo/test_stharc_task_credit_offline.py -q
These tests do not replace full Ray/TransferQueue integration tests.
"""
import ast
import asyncio
import importlib.util
import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
PAPER_CREDIT_PATH = ROOT.parent / "recipe/agentic/stharc/paper_credit.py"


def load_paper_credit():
    spec = importlib.util.spec_from_file_location("public_paper_credit_offline", PAPER_CREDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PAPER = load_paper_credit()


def load_functions(relative, names, namespace=None, class_name=None):
    """Execute unchanged function bodies; defer annotations and drop decorators."""
    tree = ast.parse((ROOT / relative).read_text())
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    selected = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    assert {n.name for n in selected} == set(names)
    for node in selected:
        node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              *selected], type_ignores=[])
    env = dict(torch=torch, np=np, defaultdict=defaultdict)
    env.update(namespace or {})
    exec(compile(ast.fix_missing_locations(module), str(ROOT / relative), "exec"), env)
    return env


CORE = load_functions("trainer/ppo/core_algos.py", [
    "get_stharc_task_credit", "validate_stharc_task_credit", "compute_stharc_task_advantage",
    "compute_grpo_outcome_advantage", "agg_loss", "compute_policy_loss_vanilla",
])
ADV = CORE["compute_stharc_task_advantage"]
GET = CORE["get_stharc_task_credit"]


class Batch:
    def __init__(self, mask, credits=None, uids=None, nested=False):
        self.batch = {"response_mask": torch.tensor(mask), "token_level_rewards": torch.zeros_like(torch.tensor(mask)).float()}
        self.non_tensor_batch = {"uid": np.array(uids if uids is not None else ["u"] * len(mask), dtype=object)}
        if credits is not None:
            key = "extra_fields" if nested else "stharc_task_credit"
            self.non_tensor_batch[key] = [{"stharc_task_credit": c} for c in credits] if nested else credits

    def __len__(self):
        return len(self.non_tensor_batch["uid"])

    def select_idxs(self, indices):
        result = Batch([])
        result.batch = {k: v[indices] for k, v in self.batch.items()}
        result.non_tensor_batch = {k: np.asarray(v, dtype=object)[indices] for k, v in self.non_tensor_batch.items()}
        return result


EST = SimpleNamespace(GRPO="grpo", GAE="gae")
CORE["AdvantageEstimator"] = EST
STANDARD = load_functions("trainer/ppo/ray_trainer.py", ["compute_advantage"], {
    "core_algos": SimpleNamespace(**CORE), "AdvantageEstimator": EST,
})["compute_advantage"]
SYNC = load_functions("trainer/main_ppo_sync.py", ["compute_advantage_for_multi_trajectories"], {
    "core_algos": SimpleNamespace(**CORE), "compute_advantage": STANDARD,
})["compute_advantage_for_multi_trajectories"]


def credit(returns, ids):
    return {"task_returns": returns, "response_task_ids": ids}


@pytest.mark.parametrize("output", [None, [], "object", 7, {}, {"object_id": None}])
def test_mutation_output_requires_mapping_and_object_id(output):
    with pytest.raises(ValueError, match="mutation requires an object_id"):
        PAPER.compute_task_credit([(0, "write", output)], [set(), {"a"}], [0., 1.])


def test_paper_credit_flows_to_same_position_token_advantages():
    mutations = [(0, "write", {"object_id": "a"})]
    reads = [set(), {"a"}]
    credited, bonus = PAPER.compute_task_credit(mutations, reads, [0., 1.])
    assert credited == pytest.approx([0.225, 1.])
    assert bonus == pytest.approx([0.9, 0.])
    baseline, _ = PAPER.compute_task_credit([], reads, [0., 1.])
    mask = torch.tensor([[1, 0, 1], [1, 0, 1]])
    metadata = [PAPER.pack_task_credit(credited, [0, -1, 1], mask[0]),
                PAPER.pack_task_credit(baseline, [0, -1, 1], mask[1])]
    advantage, _ = ADV(mask, ["same", "same"], metadata, response_lengths=[3, 3])
    scale = 0.1125 / (0.1125 + 1e-6)  # population std of [0.225, 0.]
    torch.testing.assert_close(advantage, torch.tensor([[scale, 0., 0.], [-scale, 0., 0.]]))


def test_pu_clean_gate_is_not_applied_to_other_credit_methods():
    source = (ROOT.parent / "recipe/agentic/agent_loop.py").read_text()
    tree = ast.parse(source)
    loop = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "STHARCAgentLoop")
    run = next(n for n in loop.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "run")
    gate = next(n for n in ast.walk(run) if isinstance(n, ast.If)
                and isinstance(n.test, ast.BoolOp)
                and any(isinstance(v, ast.Attribute) and v.attr == "pu_require_clean"
                        for v in n.test.values))
    def gated(credit):
        return eval(compile(ast.Expression(gate.test), str(ROOT.parent / "recipe/agentic/agent_loop.py"), "eval"),
                    {"_math_pu": math}, {"self": SimpleNamespace(credit=credit, pu_require_clean=True),
                                           "chain_reward": 1., "total_reward": 1.})
    assert gated("pu")
    assert not gated("sage")
    assert not gated("skillrise")


def test_per_position_uid_isolation_tools_padding_and_truncation():
    credits = [credit([0, 10], [0, -1, 1]), credit([2, 4], [0]), credit([200, -3], [0, 1])]
    mask = torch.tensor([[1, 0, 1, 0], [1, 0, 0, 0], [1, 1, 0, 0]])
    adv, ret = ADV(mask, np.array(["a", "a", "b"]), credits, response_lengths=[3, 1, 2])
    expected = torch.tensor([[-1 / 1.000001, 0, 3 / 3.000001, 0],
                             [1 / 1.000001, 0, 0, 0], [0, 0, 0, 0]])
    torch.testing.assert_close(adv, expected)
    torch.testing.assert_close(ret, adv)


def test_constant_singleton_and_empty_responses():
    adv, _ = ADV(torch.zeros((2, 0)), np.array(["u", "u"]), [credit([4], []), credit([4], [])])
    assert adv.shape == (2, 0)
    adv, _ = ADV(torch.ones((2, 1)), np.array(["u", "u"]), [credit([4], [0]), credit([4], [0])])
    assert torch.count_nonzero(adv) == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1j, True, "1"])
def test_bad_returns_fail(value):
    with pytest.raises(ValueError):
        ADV(torch.ones((1, 1)), np.array(["u"]), [credit([value], [0])])


@pytest.mark.parametrize("item", [None, {}, credit([], [0]), credit([1], [1]),
                                  credit([1], [-2]), credit([1], [True]), credit([1], [0.0])])
def test_bad_credit_fails(item):
    with pytest.raises(ValueError):
        ADV(torch.ones((1, 1)), np.array(["u"]), [item])


@pytest.mark.parametrize("metadata", [
    {"uid": ["u"], "agent_name": ["stharc"]},
    {"uid": ["u"], "stharc_metrics": [{}]},
    {"uid": ["u", "u"], "stharc_task_credit": [credit([1], [0]), None]},
    {"uid": [], "data_source": ["stharc_test"]},
    {"uid": ["u"], "data_source": [None, "stharc_test"]},
    {"extra_fields": [{"stharc_task_credit": credit([1], [0])}]},
])
def test_missing_or_partial_metadata_never_falls_back(metadata):
    with pytest.raises(ValueError):
        GET(metadata)


def test_ordinary_metadata_and_nested_credit():
    assert GET({"uid": ["u"], "extra_fields": [None], "stharc_metrics": [None]}) is None
    c = credit([1], [0])
    assert GET({"uid": ["u"], "extra_fields": [{"stharc_task_credit": c}]}) == [c]


@pytest.mark.parametrize("kwargs", [
    {"response_lengths": [2]}, {"epsilon": 0}, {"epsilon": float("nan")},
    {"norm_adv_by_std_in_grpo": False}, {"rollout_index": []},
])
def test_options_validate(kwargs):
    with pytest.raises(ValueError):
        ADV(torch.ones((1, 1)), np.array(["u"]), [credit([1], [0])], **kwargs)


def test_task_counts_and_session_returns_must_match():
    with pytest.raises(ValueError):
        ADV(torch.ones((2, 1)), np.array(["u", "u"]), [credit([1], [0]), credit([2, 3], [0])])
    with pytest.raises(ValueError):
        ADV(torch.ones((2, 1)), np.array(["u", "u"]), [credit([1], [0]), credit([2], [0])], rollout_index=["s", "s"])
    with pytest.raises(ValueError):
        ADV(torch.ones((1, 1)), np.array(["u"]), [credit([1], [-1])])
    with pytest.raises(ValueError):
        ADV(torch.tensor([[float("nan")]]), np.array(["u"]), [credit([1], [0])])


def test_standard_entry_and_sync_session_deduplication():
    credits = [credit([0, 6], [0, 1]), credit([2, 2], [0, 1])]
    standard = STANDARD(Batch([[1, 1], [1, 1]], credits), "grpo")
    sync = SYNC(Batch([[1, 0], [1, 1], [1, 1]], [credit([0, 6], [0]), *credits], nested=True),
                ["u_s1_0", "u_s1_1", "u_s2_0"], "grpo")
    torch.testing.assert_close(sync.batch["advantages"][1:], standard.batch["advantages"])
    assert sync.batch["advantages"][0, 0] == standard.batch["advantages"][0, 0]
    assert sync.batch["advantages"][0, 1] == 0
    assert sync.batch["advantages"][1, 0] < 0 < sync.batch["advantages"][1, 1]


@pytest.mark.parametrize("kwargs", [{"adv_estimator": "gae"}, {"adv_estimator": "grpo", "config": {"use_kl_in_reward": True}}])
def test_standard_rejects_incompatible_algorithm(kwargs):
    with pytest.raises(ValueError):
        STANDARD(Batch([[1]], [credit([1], [0])]), **kwargs)


def test_standard_and_sync_response_length_validation():
    data = Batch([[1, 0]], [credit([1], [0])])
    data.batch["attention_mask"] = torch.ones((1, 3))
    with pytest.raises(ValueError):
        STANDARD(data, "grpo")
    data.batch.pop("attention_mask")
    data.non_tensor_batch["stharc_response_lengths"] = [2]
    with pytest.raises(ValueError):
        SYNC(data, ["u_s_0"], "grpo")


def test_ordinary_grpo_regression():
    data = Batch([[1, 0], [1, 1], [1, 0]], uids=["u", "u", "v"])
    data.batch["token_level_rewards"] = torch.tensor([[0., 0.], [2., 0.], [5., 0.]])
    result = STANDARD(data, "grpo")
    expected = torch.tensor([[-1 / (2 ** 0.5 + 1e-6), 0], [1 / (2 ** 0.5 + 1e-6)] * 2, [5 / 1.000001, 0]])
    torch.testing.assert_close(result.batch["advantages"], expected)


@pytest.mark.parametrize("length", [1.5, True, float("nan"), -1, 2])
def test_response_lengths_are_not_silently_coerced(length):
    with pytest.raises(ValueError):
        ADV(torch.ones((1, 1)), ["u"], [credit([1], [0])], response_lengths=[length])


@pytest.mark.parametrize("ids", [None, 1, True, "0"])
def test_malformed_token_maps_raise_value_error(ids):
    with pytest.raises(ValueError):
        ADV(torch.ones((1, 1)), ["u"], [credit([1], ids)])


def test_explicit_null_nested_credit_fails():
    with pytest.raises(ValueError):
        GET({"uid": ["u"], "extra_fields": [{"stharc_task_credit": None}]})


def test_ordinary_sync_uses_final_session_score():
    data = Batch([[1, 0], [1, 1], [1, 0]])
    data.batch["token_level_rewards"] = torch.tensor([[99., 0.], [0., 0.], [2., 0.]])
    result = SYNC(data, ["u_a_0", "u_a_1", "u_b_0"], "grpo")
    scale = 1 / (2 ** 0.5 + 1e-6)
    torch.testing.assert_close(result.batch["advantages"], torch.tensor([[-scale, 0], [-scale, -scale], [scale, 0]]))


def test_clipped_task_token_objective_and_microbatch_dp_reduction():
    functional = load_functions("utils/torch_functional.py", ["masked_sum", "masked_mean"])
    CORE["verl_F"] = SimpleNamespace(**functional)
    CORE["AlgoConfig"] = type("AlgoConfig", (), {})

    class Config(SimpleNamespace):
        def get(self, key, default=None):
            return getattr(self, key, default)

    config = Config(clip_ratio=0.2, clip_ratio_low=None, clip_ratio_high=None,
                    clip_ratio_c=float("inf"), global_batch_info={"global_batch_size": 3})
    mask = torch.tensor([[1, 1, 0], [1, 0, 0], [0, 0, 0]])
    advantages = torch.tensor([[1., -1., 0.], [-1., 0., 0.], [0., 0., 0.]])
    ratio = torch.tensor([[1.5, 0.5, 2.], [1.5, 2., 2.], [2., 2., 2.]])
    log_prob = ratio.log().requires_grad_()
    loss_fn = CORE["compute_policy_loss_vanilla"]
    loss, _ = loss_fn(torch.zeros_like(log_prob), log_prob, advantages, mask, "seq-mean-token-sum", config)
    # Tokens: -1.2 + 0.8 + 1.5. Include the empty sampled rollout in denominator G=3.
    torch.testing.assert_close(loss, torch.tensor(1.1 / 3))
    loss.backward()
    assert torch.count_nonzero(log_prob.grad[mask == 0]) == 0
    micro = [loss_fn(torch.zeros_like(log_prob[i:i+1]), log_prob[i:i+1], advantages[i:i+1],
                     mask[i:i+1], "seq-mean-token-sum", config)[0] for i in range(3)]
    torch.testing.assert_close(sum(micro), loss)
    config.global_batch_info["dp_size"] = 2
    shards = [loss_fn(torch.zeros_like(log_prob[s]), log_prob[s], advantages[s], mask[s],
                      "seq-mean-token-sum", config)[0] for s in (slice(0, 1), slice(1, 3))]
    torch.testing.assert_close(sum(shards) / 2, loss)


@pytest.mark.parametrize("lengths", [(2, 2), (2, 3), (0, 2)])
def test_sync_real_tensordict_serialization_and_writeback(lengths):
    import time
    import tensordict
    from packaging.version import parse as parse_version
    from tensordict import NonTensorStack, TensorDict

    convert = load_functions("utils/tensordict_utils.py", ["list_of_dict_to_tensordict"], {
        "tensordict": tensordict, "TensorDict": TensorDict,
        "NonTensorStack": NonTensorStack, "parse_version": parse_version,
    })["list_of_dict_to_tensordict"]
    response_to_nested = load_functions("workers/utils/padding.py", ["response_to_nested"])["response_to_nested"]
    rows = [{"uid": "u", "data_source": "stharc_test",
             "extra_fields": {"stharc_task_credit": credit([2 * i], [0] * n), "stharc_metrics": {"x": i}},
             "response_mask": torch.ones(n, dtype=torch.long), "rm_scores": torch.zeros(n)}
            for i, n in enumerate(lengths)]
    source = convert(rows)
    captured = {}

    def put(**kwargs):
        captured.update(kwargs)
        return kwargs["fields"]

    class Config(SimpleNamespace):
        def get(self, key, default=None):
            return getattr(self, key, default)

    class Meta:
        keys = ["u_a_0", "u_b_0"]
        partition_id = "test"

        def __len__(self):
            return 2

    EST.REMAX = "remax"
    method = load_functions("trainer/main_ppo_sync.py", ["_compute_advantage"], {
        "time": time, "tq": SimpleNamespace(kv_batch_get=lambda **kw: source, kv_batch_put=put),
        "core_algos": SimpleNamespace(AdvantageEstimator=EST),
        "DataProto": lambda **kw: SimpleNamespace(**kw), "TensorDict": TensorDict,
        "response_to_nested": response_to_nested, "compute_advantage_for_multi_trajectories": SYNC,
    }, class_name="PPOTrainer")["_compute_advantage"]
    owner = SimpleNamespace(config=SimpleNamespace(
        algorithm=Config(adv_estimator="grpo", use_kl_in_reward=False, gamma=1., lam=1.),
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=2))))
    result = method(owner, Meta(), {})
    assert captured["keys"] == Meta.keys
    assert result["advantages"].is_nested == (lengths[0] != lengths[1])
    for i, n in enumerate(lengths):
        expected = torch.full((n,), (-1 if i == 0 else 1) / (1 + 1e-6))
        torch.testing.assert_close(result["advantages"][i], expected)
        torch.testing.assert_close(result["returns"][i], expected)

