"""Training: evolution strategies, the healthy gate and the settings."""
import dataclasses
import math

import numpy as np

from src.platform_adapter import load_platform
from src.tasks import skill_for
from src.tiers import load_tiers
from src.training import _centred_ranks, load_training_config, train_skill, wilson_lower


def test_wilson_lower_bound_is_below_the_raw_rate():
    assert math.isclose(wilson_lower(100, 100, 1.96), 0.963, abs_tol=1e-3)
    assert wilson_lower(0, 100, 1.96) < 0.01
    assert wilson_lower(50, 100, 1.96) < 0.5 < 0.6
    assert wilson_lower(0, 0, 1.96) == 0.0


def test_centred_ranks_are_symmetric_and_order_preserving():
    fitness = _centred_ranks(np.array([3.0, -10.0, 0.5]))
    assert fitness.min() == -0.5 and fitness.max() == 0.5
    assert fitness[1] < fitness[2] < fitness[0]


def test_training_is_deterministic_for_a_seed(fast_settings):
    platform = load_platform()
    tier = {t.name: t for t in load_tiers()}["T1"]
    joint = platform.joint("base")
    a = train_skill(joint, tier, skill_for(platform, "T1"), platform, fast_settings, seed=11)
    b = train_skill(joint, tier, skill_for(platform, "T1"), platform, fast_settings, seed=11)
    assert (a.kp_multiplier, a.kd_multiplier) == (b.kp_multiplier, b.kd_multiplier)
    assert a.history == b.history


def test_learned_gains_stay_inside_their_clamps(fast_settings):
    platform = load_platform()
    tier = {t.name: t for t in load_tiers()}["T2"]
    record = train_skill(platform.joint("elbow"), tier, skill_for(platform, "T2"), platform,
                         fast_settings, seed=5)
    kp_low, kp_high = fast_settings.kp_bounds
    kd_low, kd_high = fast_settings.kd_bounds
    assert kp_low <= record.kp_multiplier <= kp_high
    assert kd_low <= record.kd_multiplier <= kd_high
    assert record.kp_multiplier <= 1.0          # stiffness may not exceed nominal


def test_the_gate_uses_the_wilson_bound(fast_settings):
    platform = load_platform()
    tier = {t.name: t for t in load_tiers()}["T1"]
    record = train_skill(platform.joint("base"), tier, skill_for(platform, "T1"), platform,
                         fast_settings, seed=2)
    gate = record.gate
    assert gate["lower_bound"] == round(wilson_lower(gate["successes"], gate["episodes"],
                                                     fast_settings.gate_z), 4)
    assert gate["trusted"] == (gate["lower_bound"] >= fast_settings.gate_min_lower_bound)


def test_the_real_settings_load():
    s = load_training_config()
    assert s.kp_bounds == (0.5, 1.0)
    assert dataclasses.is_dataclass(s)
    # All three pairwise distinct. (A chained a != b != c never compared training with gate.)
    assert len({s.seeds["training"], s.seeds["table"], s.seeds["gate"]}) == 3


def test_training_and_gate_seed_ranges_never_overlap():
    # Each (joint, tier) slot adds its index to the base seed, so the ranges must stay apart.
    s = load_training_config()
    platform = load_platform()
    slots = len(platform.joints) * len(load_tiers())
    training = set(range(s.seeds["training"], s.seeds["training"] + slots))
    gate = set(range(s.seeds["gate"], s.seeds["gate"] + slots))
    assert not training & gate
    assert s.seeds["table"] not in training | gate
