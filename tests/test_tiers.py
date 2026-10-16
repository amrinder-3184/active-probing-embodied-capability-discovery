import math

from src.tiers import Check, Tier, load_tiers


def test_min_check_passes_at_and_above_threshold():
    c = Check(metric="x", direction="min", threshold=0.5, noise_sigma=0.0)
    assert c.passes(0.5)
    assert c.passes(0.9)
    assert not c.passes(0.4)


def test_max_check_passes_at_and_below_threshold():
    c = Check(metric="x", direction="max", threshold=1.0, noise_sigma=0.0)
    assert c.passes(1.0)
    assert c.passes(0.2)
    assert not c.passes(1.5)


def test_nan_reading_is_a_fail():
    # A missing reading must never count as a pass, whichever direction the check uses.
    assert not Check("x", "min", 0.0, 0.0).passes(math.nan)
    assert not Check("x", "max", 1e9, 0.0).passes(math.nan)


def test_pass_probability_is_half_at_the_threshold_when_noisy():
    c = Check(metric="x", direction="max", threshold=1.0, noise_sigma=0.1)
    assert abs(c.pass_probability(1.0) - 0.5) < 1e-9
    assert c.pass_probability(0.5) > 0.99
    assert c.pass_probability(1.5) < 0.01


def test_pass_probability_is_0_or_1_without_noise():
    c = Check(metric="x", direction="min", threshold=1.0, noise_sigma=0.0)
    assert c.pass_probability(1.0) == 1.0
    assert c.pass_probability(0.99) == 0.0
    assert c.pass_probability(math.nan) == 0.0


def test_tier_needs_every_check():
    tier = Tier("T9", "test", (
        Check("a", "min", 1.0, 0.0),
        Check("b", "max", 1.0, 0.0),
    ))
    assert tier.passes({"a": 1.0, "b": 0.5})
    assert not tier.passes({"a": 1.0, "b": 2.0})
    assert not tier.passes({"a": 0.0, "b": 0.5})


def test_loaded_tiers_are_T0_to_T3_in_order():
    tiers = load_tiers()
    assert [t.name for t in tiers] == ["T0", "T1", "T2", "T3"]
    assert all(len(t.checks) >= 1 for t in tiers)


def test_vector_pass_and_margin_agree_with_the_scalar_checks():
    import numpy as np
    tier = Tier("T9", "test", (Check("a", "min", 1.0, 0.0), Check("b", "max", 2.0, 0.0)))
    metrics = {"a": np.array([1.5, 0.5, np.nan]), "b": np.array([1.0, 1.0, 1.0])}
    assert list(tier.passes_array(metrics)) == [True, False, False]
    assert tier.margin_array(metrics)[0] == 0.5          # tightest check: min(0.5, 1.0)
    assert np.isnan(tier.margin_array(metrics)[2])       # NaN reading stays NaN
