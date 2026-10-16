import random

from src.gating import State
from src.assessors import Prediction
from src.outcomes import Table
from src.planner import PlannerConfig, choose
from src.tiers import Tier

CFG = PlannerConfig(
    risk_weight=1.0, energy_weight=0.5, energy_scale_j=10.0,
    risk_weights={"overload": 0.5, "joint_limit": 0.5, "fall": 1.0},
    risk_cap=0.2, likelihood_floor=0.01,
    pass_confidence=0.95, max_repeats=2, gating="chain", budget=40,
)


def pred(pass_prob, violations=(), energy=0.0):
    """A prediction where every listed violation happens with certainty."""
    violation_prob = {v: 1.0 for v in violations}
    return Prediction(pass_prob=pass_prob, violation_prob=violation_prob,
                      p_any_violation=1.0 if violations else 0.0, energy_j=energy,
                      metrics={}, margin=0.0)


def make(joints, belief, predictions, levels=None):
    """One tier ("T0") and the given predictions, keyed by (joint, hypothesis)."""
    state = State(
        joint_names=tuple(joints),
        proximal={j: () for j in joints},
        tiers=[Tier("T0", "Alive", ())],
        levels={j: levels or {"ok": 1, "dead": -1} for j in joints},
        beliefs={j: dict(belief) for j in joints},
    )
    table = Table(
        predictions={(j, "T0", h): p for (j, h), p in predictions.items()},
        levels=state.levels,
        reference_levels=state.levels,
        trusted={},
        assessors={},
    )
    return state, table


def test_no_candidates_means_stop():
    state, table = make(["j"], {"ok": 0.5, "dead": 0.5}, {})
    assert choose("planner", state, [], table, CFG, random.Random(0)) is None


def test_passive_never_probes():
    state, table = make(["j"], {"ok": 0.5, "dead": 0.5},
                        {("j", "ok"): pred(1.0), ("j", "dead"): pred(0.0)})
    assert choose("passive", state, [("j", 0)], table, CFG, random.Random(0)) is None


def test_safety_veto_blocks_a_probe_that_is_too_likely_to_violate_a_limit():
    # Half the belief sits on a hypothesis where the probe overloads the joint. The cap is 0.2.
    state, table = make(["j"], {"ok": 0.5, "dead": 0.5},
                        {("j", "ok"): pred(1.0, ["overload"]), ("j", "dead"): pred(0.0)})
    assert choose("planner", state, [("j", 0)], table, CFG, random.Random(0)) is None
    assert choose("random", state, [("j", 0)], table, CFG, random.Random(0)) is None


def test_planner_still_runs_a_zero_information_probe_when_it_is_the_only_one_left():
    # Every hypothesis predicts the same outcome, so the information gain is zero and the energy
    # cost makes the utility negative. The planner must still run it: a tier can only be
    # concluded by probing it.
    state, table = make(["j"], {"ok": 0.5, "dead": 0.5},
                        {("j", "ok"): pred(0.7, energy=5.0), ("j", "dead"): pred(0.7, energy=5.0)})
    assert choose("planner", state, [("j", 0)], table, CFG, random.Random(0)) == ("j", 0)


def test_planner_prefers_the_probe_that_separates_the_hypotheses():
    belief = {"ok": 0.5, "dead": 0.5}
    state, table = make(["a", "b"], belief,
                        {("a", "ok"): pred(0.5), ("a", "dead"): pred(0.5),   # uninformative
                         ("b", "ok"): pred(1.0), ("b", "dead"): pred(0.0)})  # informative
    assert choose("planner", state, [("a", 0), ("b", 0)], table, CFG, random.Random(0)) == ("b", 0)


def test_risk_weight_makes_the_planner_avoid_a_risky_probe():
    belief = {"ok": 0.5, "dead": 0.5}
    # Probe a is more informative, but it has a 10% violation chance in both hypotheses.
    # Probe b is less informative and safe.
    predictions = {("a", "ok"): pred(1.0, ["overload"]), ("a", "dead"): pred(0.0, ["overload"]),
                   ("b", "ok"): pred(0.8), ("b", "dead"): pred(0.2)}
    state, table = make(["a", "b"], belief, predictions)
    cautious = PlannerConfig(**{**CFG.__dict__, "risk_weight": 10.0, "risk_cap": 1.0})
    assert choose("planner", state, [("a", 0), ("b", 0)], table, cautious, random.Random(0)) == ("b", 0)


def test_fixed_battery_runs_in_tier_order_then_joint_order():
    state, table = make(["a", "b"], {"ok": 0.5, "dead": 0.5},
                        {("a", "ok"): pred(1.0), ("a", "dead"): pred(0.0),
                         ("b", "ok"): pred(1.0), ("b", "dead"): pred(0.0)})
    candidates = [("b", 0), ("a", 0)]
    assert choose("fixed", state, candidates, table, CFG, random.Random(0)) == ("a", 0)
