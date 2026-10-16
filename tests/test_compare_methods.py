"""The comparison script: original columns kept, per-trial and paired outputs, and the ceiling."""
import csv
import os
import random
import sys

import pytest

import experiments.compare_methods as compare
from src.damage import sample_fault_name
from src.policies import PolicyStore

ORIGINAL_COLUMNS = ["method", "gating", "budget", "coverage", "accuracy", "overclaim",
                    "probes", "aborts", "energy_j"]


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    out = tmp_path_factory.mktemp("compare")
    paths = {name: out / f"{name}.csv" for name in ("summary", "trials", "paired")}
    argv = ["compare_methods", "--seeds", "2", "--budgets", "6", "48", "--assessment", "scripted",
            "--out", str(paths["summary"]), "--trials-out", str(paths["trials"]),
            "--paired-out", str(paths["paired"]), "--bootstrap", "200", "--ceiling"]
    old = sys.argv
    sys.argv = argv
    try:
        compare.main()
    finally:
        sys.argv = old
    return {name: _read(path) for name, path in paths.items()}, paths


def test_summary_keeps_the_original_columns_first(outputs):
    _, paths = outputs
    with open(paths["summary"], newline="") as f:
        header = next(csv.reader(f))
    assert header[:len(ORIGINAL_COLUMNS)] == ORIGINAL_COLUMNS
    assert "level_brier" in header and "coverage_ceiling" in header


def test_one_trial_row_per_method_budget_and_seed(outputs):
    data, _ = outputs
    methods = {r["method"] for r in data["summary"]}
    assert len(data["trials"]) == len(methods) * 2 * 2
    # Every method saw the same damage for the same seed: this is the pairing.
    by_seed = {}
    for r in data["trials"]:
        by_seed.setdefault(r["seed"], set()).add(r["true_states"])
    assert all(len(states) == 1 for states in by_seed.values())


def test_paired_rows_compare_every_other_method_with_fixed(outputs):
    data, _ = outputs
    assert {r["reference"] for r in data["paired"]} == {"fixed"}
    assert "fixed" not in {r["method"] for r in data["paired"]}
    for r in data["paired"]:
        assert float(r["ci_low"]) <= float(r["mean_diff"]) <= float(r["ci_high"])


def test_passive_is_the_prior_only_floor(outputs):
    data, _ = outputs
    for r in data["summary"]:
        if r["method"] == "passive":
            assert float(r["probes"]) == 0.0 and float(r["coverage"]) == 0.0


def test_ceiling_matches_the_fixed_battery_when_the_budget_cannot_bind(outputs):
    data, _ = outputs
    fixed_unbounded = next(r for r in data["summary"] if r["method"] == "fixed" and r["budget"] == "48")
    assert float(fixed_unbounded["coverage_ceiling"]) == pytest.approx(float(fixed_unbounded["coverage"]))


# --- --seed-start and --policies -------------------------------------------------------------

def test_without_seed_start_the_seeds_are_0_to_n_minus_1(outputs):
    data, _ = outputs
    assert {int(r["seed"]) for r in data["trials"]} == {0, 1}
    assert compare.parse_args(["--assessment", "scripted"], "arm6").seed_start == 0
    assert list(compare.seed_range(0, 3)) == [0, 1, 2]


def test_seed_range_starts_at_seed_start():
    seeds = compare.seed_range(100, 200)
    assert seeds[0] == 100 and seeds[-1] == 299 and len(seeds) == 200


@pytest.fixture(scope="module")
def offset_trials(tmp_path_factory):
    out = tmp_path_factory.mktemp("offset")
    compare.main(["--seeds", "2", "--seed-start", "100", "--budgets", "6", "--assessment", "scripted",
                  "--out", str(out / "s.csv"), "--trials-out", str(out / "t.csv")])
    return _read(out / "t.csv")


def test_seed_start_runs_every_method_on_the_same_shifted_seeds(offset_trials):
    seeds_by_method = {}
    for r in offset_trials:
        seeds_by_method.setdefault(r["method"], []).append(int(r["seed"]))
    assert all(s == [100, 101] for s in seeds_by_method.values())
    # Default --risk-weights is [1.0]: one planner row plus the three baselines.
    assert set(seeds_by_method) == {"planner(risk=1)", "random", "fixed", "passive"}


def test_the_seed_value_not_its_position_sets_the_damage(arm, offset_trials):
    # Seed 100 must draw the damage of seed 100, for every method: the pairing is by seed value.
    _, _, damage, _ = arm
    rng = random.Random(100)
    names = ["base", "shoulder", "elbow", "wrist_1", "wrist_2", "wrist_3"]
    expected = ";".join(f"{n}={sample_fault_name(damage, rng)}" for n in names)
    assert {r["true_states"] for r in offset_trials if r["seed"] == "100"} == {expected}


@pytest.mark.parametrize("argv, message", [
    (["--seeds", "0"], "--seeds must be at least 1"),
    (["--seed-start", "-1"], "--seed-start must be 0 or more"),
    (["--bootstrap", "0"], "--bootstrap must be at least 1"),
    (["--policies", "anything"], "--policies is only used by the learned assessment"),
])
def test_invalid_arguments_are_rejected_with_a_clear_message(argv, message, capsys):
    with pytest.raises(SystemExit) as err:
        compare.parse_args(["--assessment", "scripted", *argv], "arm6")
    assert err.value.code == 2
    assert message in capsys.readouterr().err


def test_an_explicit_policies_folder_is_used_as_given(tmp_path):
    args = compare.parse_args(["--policies", str(tmp_path)], "arm6")
    assert args.policies == str(tmp_path)


def test_a_missing_policies_folder_is_rejected_not_replaced(tmp_path, capsys):
    with pytest.raises(SystemExit) as err:
        compare.parse_args(["--policies", str(tmp_path / "nope")], "arm6")
    assert err.value.code == 2
    assert str(tmp_path / "nope") in capsys.readouterr().err


def test_the_default_policies_folder_is_results_policies_platform(tmp_path, monkeypatch, capsys):
    # Point the repository root at an empty folder: the default must be its results/policies/arm6,
    # and since that folder is missing, the learned run must stop and name it.
    monkeypatch.setattr(compare, "REPO_ROOT", str(tmp_path))
    default = os.path.join(str(tmp_path), "results", "policies", "arm6")
    assert compare.parse_args(["--assessment", "scripted"], "arm6").policies == default
    with pytest.raises(SystemExit):
        compare.parse_args([], "arm6")
    assert default in capsys.readouterr().err


def test_missing_policies_lists_every_learned_slot_without_a_file(arm, tmp_path):
    platform, tiers, _, _ = arm
    store = PolicyStore(tmp_path)
    assert len(compare.missing_policies(store, platform, tiers)) == 18     # 6 joints x T1..T3
    for joint in platform.joints:
        for tier in ("T1", "T2", "T3"):
            (tmp_path / f"{joint.name}_{tier}.json").write_text("{}")
    (tmp_path / "elbow_T2.json").unlink()
    assert compare.missing_policies(store, platform, tiers) == [("elbow", "T2")]


def test_an_incomplete_policies_folder_stops_before_any_episode(tmp_path):
    (tmp_path / "base_T1.json").write_text("{}")
    out = tmp_path / "summary.csv"
    with pytest.raises(FileNotFoundError, match="17 trained policies missing") as err:
        compare.main(["--policies", str(tmp_path), "--seeds", "1", "--budgets", "6", "--out", str(out)])
    assert "elbow_T2.json" in str(err.value)
    assert not out.exists()


# --- --curve-out -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def curve_outputs(tmp_path_factory):
    out = tmp_path_factory.mktemp("curve")
    compare.main(["--seeds", "2", "--budgets", "4", "6", "--assessment", "scripted",
                  "--out", str(out / "s.csv"), "--trials-out", str(out / "t.csv"), "--curve-out", str(out / "c.csv")])
    return out


def _header(path):
    with open(path, newline="") as f:
        return next(csv.reader(f))


def test_curve_has_one_row_per_method_seed_and_k(curve_outputs):
    rows = _read(curve_outputs / "c.csv")
    assert len(rows) == 4 * 2 * (6 + 1)          # 4 methods, 2 seeds, k = 0 .. 6 (the largest budget)
    assert {int(r["k"]) for r in rows} == set(range(7))
    assert {r["budget"] for r in rows} == {"6"}
    assert _header(curve_outputs / "c.csv") == ["method", "gating", "budget", "seed", "k", *compare.CURVE_METRICS]


def test_curve_does_not_change_the_existing_outputs(curve_outputs):
    # Same columns as without --curve-out, and no new metric leaks into them.
    assert _header(curve_outputs / "s.csv") == ["method", "gating", "budget", *compare.METRICS]
    assert _header(curve_outputs / "t.csv") == ["method", "gating", "budget", "seed", *compare.METRICS, "true_states"]
    assert "time_s" not in compare.METRICS


def test_curve_rows_at_each_budget_equal_the_separately_run_trials(curve_outputs):
    curve = {(r["method"], r["seed"], int(r["k"])): r for r in _read(curve_outputs / "c.csv")}
    trials = _read(curve_outputs / "t.csv")
    for t in trials:
        c = curve[(t["method"], t["seed"], int(t["budget"]))]
        assert all(c[m] == t[m] for m in compare.METRICS), (t["method"], t["seed"], t["budget"])
    assert {t["budget"] for t in trials} == {"4", "6"}
