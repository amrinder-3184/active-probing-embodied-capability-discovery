"""Measurements to specification (milestone M1): prefix evaluation, probing time, safety events,
and per-tier confidence. All on the scripted (reference) table from conftest."""
import dataclasses
import random

import numpy as np
import pytest

from src.belief import prob_level_at_least
from src.damage import Fault, sample_fault_name
from src.episode import prefix_episodes, run_episode
from src.evaluate import score_episode
from src.physics import DT, make_plant, simulate
from src.planner import METHODS, load_planner_config
from src.policies import controller_for
from src.report import VIOLATION_TYPES, build_report
from src.tasks import skill_for

NAMES = ["base", "shoulder", "elbow", "wrist_1", "wrist_2", "wrist_3"]


def _states(damage, seed):
    rng = random.Random(seed)
    return {n: sample_fault_name(damage, rng) for n in NAMES}


def _first_seed(damage, predicate):
    return next(s for s in range(2000) if predicate(_states(damage, s)))


def _weak_shoulder_seed(damage):
    # The weak shoulder overloads on its hold (a real abort), if base and shoulder reach T3.
    return _first_seed(damage, lambda st: st["shoulder"] == "weak_motor" and st["base"] == "healthy")


# --- prefix evaluation -----------------------------------------------------------------------

LONG = 16
CHECK_K = (0, 1, 6, 7, 12, 16)   # empty, first probe, around the T0 -> T1 step, mid-run, the whole run


@pytest.mark.parametrize("method", METHODS)
def test_prefix_at_k_equals_a_separate_run_with_budget_k(arm, method):
    platform, tiers, damage, table = arm
    cfg = load_planner_config()
    seeds = list(range(10)) + [_weak_shoulder_seed(damage)]
    for seed in seeds:
        long = run_episode(platform, tiers, damage, table, method, dataclasses.replace(cfg, budget=LONG), seed)
        prefixes = list(prefix_episodes(long, table, damage, cfg))
        assert [p.budget for p in prefixes] == list(range(LONG + 1))
        for k in CHECK_K:
            short = run_episode(platform, tiers, damage, table, method, dataclasses.replace(cfg, budget=k), seed)
            replayed, separate = build_report(platform, prefixes[k]), build_report(platform, short)
            # Field by field: the scores, and the whole report they come from.
            assert score_episode(replayed, prefixes[k]) == score_episode(separate, short), (method, seed, k)
            assert replayed == separate, (method, seed, k)


def test_prefixes_past_the_end_of_the_log_are_the_whole_episode(arm):
    platform, tiers, damage, table = arm
    cfg = dataclasses.replace(load_planner_config(), budget=40)
    episode = run_episode(platform, tiers, damage, table, "fixed", cfg, seed=3)
    assert len(episode.log) < 40
    last = list(prefix_episodes(episode, table, damage, cfg))[-1]
    assert build_report(platform, last) == build_report(platform, episode)


def test_prefix_replay_does_not_change_the_episode(arm):
    platform, tiers, damage, table = arm
    cfg = load_planner_config()
    episode = run_episode(platform, tiers, damage, table, "planner", cfg, seed=5)
    before = build_report(platform, episode)
    list(prefix_episodes(episode, table, damage, cfg))
    assert build_report(platform, episode) == before


# --- probing time ----------------------------------------------------------------------------

def test_probing_time_is_the_sum_of_the_executed_skill_durations(arm):
    platform, tiers, damage, table = arm
    cfg = load_planner_config()
    seed = _weak_shoulder_seed(damage)
    episode = run_episode(platform, tiers, damage, table, "fixed", cfg, seed)
    report = build_report(platform, episode)
    assert report["time_s"] == sum(r.duration_s for r in episode.log)
    assert score_episode(report, episode)["time_s"] == report["time_s"]
    aborted = [r for r in episode.log if r.aborted]
    assert aborted, "this seed must contain an aborted probe"
    for r in episode.log:
        full = skill_for(platform, r.tier).n_steps * DT
        if r.aborted:
            assert 0.0 < r.duration_s < full      # an abort stops the skill early
        else:
            assert r.duration_s == pytest.approx(full)


def test_a_probe_on_a_missing_sensor_takes_no_time(arm):
    platform, tiers, damage, table = arm
    seed = _first_seed(damage, lambda st: st["base"] == "missing_sensor")
    episode = run_episode(platform, tiers, damage, table, "fixed", load_planner_config(), seed)
    base_runs = [r for r in episode.log if r.joint == "base"]
    assert base_runs and all(r.duration_s == 0.0 for r in base_runs)


def test_executed_duration_counts_the_steps_the_physics_ran():
    # The weak shoulder hold aborts on overload: its duration is the alive steps times DT.
    from src.platform_adapter import load_platform
    platform = load_platform()
    joint, skill = platform.joint("shoulder"), skill_for(platform, "T3")
    rollout = simulate(make_plant(joint, Fault(torque_scale=0.4), 1, gravity_on=True),
                       np.full(1, skill.budget_fraction * joint.tau_max_nm), controller_for(joint, 1.0, 1.0, 1),
                       skill.reference, skill.n_steps, platform.heat_coeff)
    from src.assessors import _executed_duration
    steps = int(rollout.alive[:, 0].sum())
    assert 0 < steps < skill.n_steps
    assert _executed_duration(rollout) == steps * DT


# --- safety events ---------------------------------------------------------------------------

def test_violation_types_are_the_ones_the_physics_reports():
    from src.platform_adapter import load_platform
    platform = load_platform()
    joint, skill = platform.joint("base"), skill_for(platform, "T0")
    rollout = simulate(make_plant(joint, Fault(), 1), np.full(1, 30.0), controller_for(joint, 1.0, 1.0, 1),
                       skill.reference, 5, platform.heat_coeff)
    assert tuple(rollout.violations) == VIOLATION_TYPES


def test_safety_event_counts_match_the_recorded_events(arm):
    platform, tiers, damage, table = arm
    cfg = load_planner_config()
    for seed in [_weak_shoulder_seed(damage), 0, 3]:
        for method in ("planner", "fixed", "random"):
            episode = run_episode(platform, tiers, damage, table, method, cfg, seed)
            report = build_report(platform, episode)
            score = score_episode(report, episode)
            for v in VIOLATION_TYPES:
                expected = sum(v in r.violations for r in episode.log)
                assert report["violations"][v] == expected
                assert score[f"violations_{v}"] == expected
            assert sum(report["violations"].values()) >= report["aborted_probes"]
    weak = run_episode(platform, tiers, damage, table, "fixed", cfg, _weak_shoulder_seed(damage))
    assert build_report(platform, weak)["violations"]["overload"] >= 1


# --- per-tier confidence ---------------------------------------------------------------------

def test_report_confidence_is_p_level_at_least_k_in_range_and_non_increasing(arm):
    platform, tiers, damage, table = arm
    cfg = load_planner_config()
    for seed in range(6):
        episode = run_episode(platform, tiers, damage, table, "planner", cfg, seed)
        for k, prefix in enumerate(prefix_episodes(episode, table, damage, cfg)):
            if k > len(episode.log):
                break
            for part in build_report(platform, prefix)["parts"]:
                belief, levels = prefix.state.beliefs[part["joint"]], prefix.state.levels[part["joint"]]
                values = [part["confidence"][t.name] for t in prefix.state.tiers]
                assert values == [prob_level_at_least(belief, levels, ti) for ti in range(len(values))]
                assert all(0.0 <= c <= 1.0 + 1e-12 for c in values)
                assert all(a >= b - 1e-12 for a, b in zip(values, values[1:]))
