"""Evaluation scores: hand-checked values, zero-probability safety, ties, and the prior-only floor."""
import math
import random

import numpy as np
import pytest

from src.damage import sample_fault_name
from src.episode import run_episode
from src.evaluate import (EPS, belief_scores, brier, level_forecasts, log_loss, map_set,
                          paired_bootstrap, score_episode)
from src.planner import load_planner_config
from src.report import build_report


def test_brier_matches_a_hand_calculation():
    # (1 - 1)^2 = 0 and (0.5 - 0)^2 = 0.25, mean 0.125.
    assert brier([1.0, 0.5], [1, 0]) == pytest.approx(0.125)


def test_log_loss_matches_a_hand_calculation():
    # -(ln 0.8 + ln(1 - 0.3)) / 2
    assert log_loss([0.8, 0.3], [1, 0]) == pytest.approx(-(math.log(0.8) + math.log(0.7)) / 2)


def test_log_loss_is_finite_for_zero_and_one_probabilities():
    # A certain wrong forecast costs -ln(EPS), not infinity. A certain right one costs almost nothing.
    assert log_loss([0.0], [1]) == pytest.approx(-math.log(EPS))
    assert log_loss([1.0], [0]) == pytest.approx(-math.log(EPS))
    assert log_loss([1.0, 0.0], [1, 0]) < 1e-8


def test_level_forecasts_are_cumulative():
    belief = {"ok": 0.6, "weak": 0.3, "dead": 0.1}
    levels = {"ok": 3, "weak": 2, "dead": -1}
    assert level_forecasts(belief, levels, 4) == pytest.approx([0.9, 0.9, 0.9, 0.6])


def test_map_set_keeps_every_tied_hypothesis():
    assert map_set({"locked": 0.458, "missing_sensor": 0.458, "healthy": 0.084}) == ["locked", "missing_sensor"]
    assert map_set({"a": 0.7, "b": 0.3}) == ["a"]


def test_a_tie_gives_half_credit_whichever_tied_state_is_true():
    belief = {"healthy": 0.1, "locked": 0.45, "missing_sensor": 0.45}
    levels = {"healthy": 3, "locked": -1, "missing_sensor": -1}
    for truth in ("locked", "missing_sensor"):
        s = belief_scores(belief, levels, reference_level=-1, truth=truth, n_tiers=4)
        assert s["hypothesis_expected_accuracy"] == 0.5
        assert s["hypothesis_tie"]
    assert belief_scores(belief, levels, 3, "healthy", 4)["hypothesis_expected_accuracy"] == 0.0


def test_hypothesis_brier_matches_a_hand_calculation():
    s = belief_scores({"a": 0.7, "b": 0.3}, {"a": 0, "b": 0}, 0, "a", 1)
    assert s["hypothesis_brier"] == pytest.approx(0.3 ** 2 + 0.3 ** 2)


def _seed_where(damage, names, predicate):
    for s in range(1000):
        rng = random.Random(s)
        states = {n: sample_fault_name(damage, rng) for n in names}
        if predicate(states):
            return s
    raise AssertionError("no seed found")


def test_passive_scores_are_the_prior_only_forecast(arm):
    platform, tiers, damage, table = arm
    episode = run_episode(platform, tiers, damage, table, "passive", load_planner_config(), seed=3)
    score = score_episode(build_report(platform, episode), episode)
    expected = []
    for joint in platform.joints:
        truth = episode.true_states[joint.name]
        ref = table.reference_levels[joint.name][truth]
        expected.append(belief_scores(damage.prior, table.levels[joint.name], ref, truth, 4)["level_brier"])
    assert score["level_brier"] == pytest.approx(sum(expected) / len(expected))
    assert score["probes"] == 0


def test_probing_a_healthy_arm_beats_the_prior_on_every_proper_score(arm):
    platform, tiers, damage, table = arm
    names = [j.name for j in platform.joints]
    seed = _seed_where(damage, names, lambda st: all(h == "healthy" for h in st.values()))
    cfg = load_planner_config()
    probed = score_episode(*_report_and_episode(platform, tiers, damage, table, "planner", cfg, seed))
    prior = score_episode(*_report_and_episode(platform, tiers, damage, table, "passive", cfg, seed))
    assert probed["level_brier"] < prior["level_brier"]
    assert probed["level_log_loss"] < prior["level_log_loss"]
    assert probed["hypothesis_brier"] < prior["hypothesis_brier"]


def test_a_failed_locked_base_is_a_scored_tie(arm):
    # Locked and missing sensor predict the same outcome on every probe, so the base ends tied.
    platform, tiers, damage, table = arm
    names = [j.name for j in platform.joints]
    seed = _seed_where(damage, names, lambda st: st["base"] == "locked")
    report, episode = _report_and_episode(platform, tiers, damage, table, "planner", load_planner_config(), seed)
    base = belief_scores(episode.state.beliefs["base"], table.levels["base"], -1, "locked", 4)
    assert base["hypothesis_tie"] and base["hypothesis_expected_accuracy"] == 0.5
    # The existing argmax metric still reports the dict-order pick, unchanged.
    assert report["parts"][0]["most_likely_state"] == "locked"


def test_existing_score_keys_keep_their_meaning(arm):
    platform, tiers, damage, table = arm
    report, episode = _report_and_episode(platform, tiers, damage, table, "fixed", load_planner_config(), 3)
    score = score_episode(report, episode)
    parts = report["parts"]
    assert score["coverage"] == sum(p["determined"] for p in parts) / len(parts)
    assert score["hypothesis_accuracy"] == sum(
        p["most_likely_state"] == episode.true_states[p["joint"]] for p in parts) / len(parts)


def test_paired_bootstrap_of_identical_methods_is_exactly_zero():
    assert paired_bootstrap([0.5, 0.2, 0.9], [0.5, 0.2, 0.9]) == (0.0, 0.0, 0.0)


def test_paired_bootstrap_of_a_constant_shift_is_that_shift():
    mean, low, high = paired_bootstrap([1.5, 2.5, 3.5], [1.0, 2.0, 3.0])
    assert mean == pytest.approx(0.5) and low == pytest.approx(0.5) and high == pytest.approx(0.5)


def test_paired_bootstrap_pairs_by_seed_not_by_pooled_values():
    # The same values in a different order: the pooled means agree, but every pair differs.
    a, b = [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]
    mean, low, high = paired_bootstrap(a, b, resamples=500)
    assert mean == 0.0 and low < 0.0 < high


def test_paired_bootstrap_interval_is_reproducible_and_contains_the_mean():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.1, 1.0, 50), rng.normal(0.0, 1.0, 50)
    first = paired_bootstrap(a, b, seed=3)
    assert first == paired_bootstrap(a, b, seed=3)
    assert first[1] <= first[0] <= first[2]


def test_paired_bootstrap_rejects_mismatched_input():
    with pytest.raises(ValueError):
        paired_bootstrap([], [])
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0])


def _report_and_episode(platform, tiers, damage, table, method, cfg, seed):
    episode = run_episode(platform, tiers, damage, table, method, cfg, seed)
    return build_report(platform, episode), episode
