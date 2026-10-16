"""End-to-end episodes on the real arm6 platform."""
import dataclasses
import json
import math
import random

import numpy as np
import pytest

from src.damage import sample_fault_name
from src.episode import SimulatedRobot, run_episode
from src.planner import METHODS, load_planner_config
from src.report import build_report


def true_states(damage, seed, joint_names):
    # Same draw as SimulatedRobot, so these are the states an episode with this seed will use.
    rng = random.Random(seed)
    return {name: sample_fault_name(damage, rng) for name in joint_names}


def first_seed_where(damage, joint_names, predicate):
    return next(s for s in range(1000) if predicate(true_states(damage, s, joint_names)))


def test_healthy_arm_reaches_the_top_tier_on_every_joint(arm):
    platform, tiers, damage, table = arm
    names = [j.name for j in platform.joints]
    seed = first_seed_where(damage, names, lambda st: all(h == "healthy" for h in st.values()))
    episode = run_episode(platform, tiers, damage, table, "planner", load_planner_config(), seed)
    report = build_report(platform, episode)
    assert all(p["level"] == 3 and p["determined"] for p in report["parts"])
    assert report["aborted_probes"] == 0


def test_a_locked_base_blocks_every_joint_above_it(arm):
    platform, tiers, damage, table = arm
    names = [j.name for j in platform.joints]
    seed = first_seed_where(damage, names, lambda st: st["base"] == "locked")
    episode = run_episode(platform, tiers, damage, table, "planner", load_planner_config(), seed)
    report = build_report(platform, episode)

    base, rest = report["parts"][0], report["parts"][1:]
    assert base["joint"] == "base" and base["level"] == -1 and base["determined"]
    for part in rest:
        assert part["tiers"]["T0"] == "blocked"   # chain gating: the base failed T0
        assert part["probes"] == 0                # so nothing above it was probed


def test_every_method_stays_within_the_budget_and_gives_a_serialisable_report(arm):
    platform, tiers, damage, table = arm
    cfg = dataclasses.replace(load_planner_config(), budget=12)
    for method in METHODS:
        episode = run_episode(platform, tiers, damage, table, method, cfg, seed=3)
        assert len(episode.log) <= 12
        json.dumps(build_report(platform, episode))


def test_passive_observer_runs_no_probes(arm):
    platform, tiers, damage, table = arm
    episode = run_episode(platform, tiers, damage, table, "passive", load_planner_config(), seed=3)
    assert episode.log == []


def test_same_seed_and_method_give_the_same_report(arm):
    platform, tiers, damage, table = arm
    cfg = load_planner_config()
    a = build_report(platform, run_episode(platform, tiers, damage, table, "random", cfg, seed=7))
    b = build_report(platform, run_episode(platform, tiers, damage, table, "random", cfg, seed=7))
    assert a == b


def _outcomes(episode):
    """(joint, tier, repeat) -> passed, from the probe log."""
    seen, out = {}, {}
    for r in episode.log:
        k = seen.get((r.joint, r.tier), 0)
        seen[(r.joint, r.tier)] = k + 1
        out[(r.joint, r.tier, k)] = r.passed
    return out


def test_per_probe_streams_give_every_method_the_same_outcome_for_the_same_probe(arm):
    # Pairing: the r-th run of a probe sees the same noise whichever method runs it, and in any order.
    platform, tiers, damage, table = arm
    cfg = load_planner_config()
    shared_probes = 0
    for seed in range(12):
        runs = {m: _outcomes(run_episode(platform, tiers, damage, table, m, cfg, seed))
                for m in ("planner", "fixed", "random")}
        for other in ("fixed", "random"):
            common = runs["planner"].keys() & runs[other].keys()
            shared_probes += len(common)
            assert all(runs["planner"][k] == runs[other][k] for k in common)
    assert shared_probes > 100


def test_same_probes_with_same_outcomes_give_the_same_final_belief(arm):
    # Bayes updates commute, so a different order of the same evidence gives the same belief.
    platform, tiers, damage, table = arm
    cfg = load_planner_config()
    compared = reordered = 0
    for seed in range(6):
        a = run_episode(platform, tiers, damage, table, "planner", cfg, seed)
        b = run_episode(platform, tiers, damage, table, "fixed", cfg, seed)
        if _outcomes(a) == _outcomes(b):
            compared += 1
            reordered += [(r.joint, r.tier) for r in a.log] != [(r.joint, r.tier) for r in b.log]
            for joint in a.state.beliefs:
                for h, p in a.state.beliefs[joint].items():
                    assert math.isclose(p, b.state.beliefs[joint][h], rel_tol=1e-12, abs_tol=1e-15)
    assert compared >= 3 and reordered >= 1   # exercised, including on a genuinely different order


def test_probe_streams_depend_on_the_probe_and_repeat_and_not_on_the_table_stream(arm):
    platform, tiers, damage, table = arm
    names = tuple(j.name for j in platform.joints)
    robot = SimulatedRobot(platform, tiers, table.assessors, damage, random.Random(0), None, names,
                           probe_seed=3000)
    draw = lambda rng: tuple(rng.random(4))
    assert draw(robot.rng_for("elbow", 2, 0)) == draw(robot.rng_for("elbow", 2, 0))
    assert draw(robot.rng_for("elbow", 2, 0)) != draw(robot.rng_for("elbow", 2, 1))
    assert draw(robot.rng_for("elbow", 2, 0)) != draw(robot.rng_for("wrist_1", 2, 0))
    # The model table seeds with SeedSequence([3000, joint, tier, hypothesis]); a robot run with
    # episode seed 3000 must not reuse those numbers.
    for j, t, h in [(1, 1, 1), (2, 2, 0), (0, 0, 1)]:
        table_draw = draw(np.random.default_rng(np.random.SeedSequence([3000, j, t, h])))
        assert table_draw != draw(robot.rng_for(names[j], t, h))


def test_shared_streams_use_one_generator_in_probe_order(arm):
    platform, tiers, damage, table = arm
    names = tuple(j.name for j in platform.joints)
    shared = np.random.default_rng(8)
    robot = SimulatedRobot(platform, tiers, table.assessors, damage, random.Random(0), shared, names)
    assert robot.rng_for("base", 0, 0) is shared and robot.rng_for("elbow", 3, 1) is shared
    cfg = load_planner_config()
    a = run_episode(platform, tiers, damage, table, "planner", cfg, 5, noise_streams="shared")
    b = run_episode(platform, tiers, damage, table, "planner", cfg, 5, noise_streams="shared")
    assert build_report(platform, a) == build_report(platform, b)


def test_unknown_noise_stream_mode_is_rejected(arm):
    platform, tiers, damage, table = arm
    with pytest.raises(ValueError, match="noise_streams"):
        run_episode(platform, tiers, damage, table, "fixed", load_planner_config(), 0, noise_streams="global")
