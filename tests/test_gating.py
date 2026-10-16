"""Gating rules, tested on a small synthetic robot with two chains.

Chain A is a -> b -> c (a is nearest the base). Chain B is x -> y.
Hypotheses: "ok" reaches tier 1 (level 1), "dead" reaches nothing (level -1).
A belief of 1.0 or 0.0 is only a conclusion once a probe of that tier has run.
"""
from src.gating import State

LEVELS = {"ok": 1, "dead": -1}
PROXIMAL = {"a": (), "b": ("a",), "c": ("a", "b"), "x": (), "y": ("x",)}
JOINTS = ("a", "b", "c", "x", "y")


def make_state(beliefs, probed=(), gating="chain", max_repeats=2):
    """beliefs: joint -> P(ok). probed: (joint, tier index) pairs that have been probed once."""
    full = {j: {"ok": p, "dead": 1.0 - p} for j, p in beliefs.items()}
    for j in JOINTS:
        full.setdefault(j, {"ok": 0.5, "dead": 0.5})
    return State(
        joint_names=JOINTS,
        proximal=PROXIMAL,
        tiers=["T0", "T1"],
        levels={j: LEVELS for j in JOINTS},
        beliefs=full,
        gating=gating,
        confidence=0.95,
        max_repeats=max_repeats,
        probe_counts={key: 1 for key in probed},
    )


def test_a_failed_proximal_joint_blocks_the_joints_below_it_at_the_same_tier():
    s = make_state({"a": 0.0}, probed=[("a", 0)])        # a is dead, and T0 was probed
    assert s.status("a", 0) == "fail"
    assert s.status("b", 0) == "blocked"
    assert s.status("c", 0) == "blocked"


def test_chain_gating_does_not_block_the_other_chain():
    s = make_state({"a": 0.0}, probed=[("a", 0)])
    assert s.status("x", 0) == "ready"
    assert s.status("y", 0) == "wait"                    # y waits on x, not on a


def test_a_tier_is_tested_only_after_its_own_lower_tier_passes():
    s = make_state({"a": 1.0, "b": 0.5}, probed=[("a", 0), ("a", 1)])
    assert s.status("b", 1) == "wait"                    # b has not passed T0 yet
    assert s.status("b", 0) == "ready"


def test_an_undecided_proximal_joint_makes_the_one_below_wait():
    s = make_state({"a": 0.5, "b": 0.5})
    assert s.status("a", 0) == "ready"
    assert s.status("b", 0) == "wait"


def test_a_belief_alone_does_not_pass_a_tier():
    # The belief is certain that a reaches T0, but no probe has run, so the tier is still open.
    s = make_state({"a": 1.0})
    assert s.status("a", 0) == "ready"
    s.probe_counts[("a", 0)] = 1
    assert s.status("a", 0) == "pass"
    assert s.status("b", 0) == "ready"                   # so b may now be tested


def test_repeat_limit_with_no_conclusion_gives_unknown_and_unknown_spreads_down_the_chain():
    s = make_state({"a": 0.5}, max_repeats=2)
    s.probe_counts[("a", 0)] = 2                         # two probes, still undecided
    assert s.status("a", 0) == "unknown"
    assert s.status("b", 0) == "unknown"


def test_robot_gating_blocks_joints_in_other_chains():
    # y must pass T0 and stay undecided at T1, so that gating decides its T1 status, not belief.
    # "t0only" reaches tier 0 but not tier 1, which gives y that undecided T1.
    levels = {"ok": 1, "t0only": 0, "dead": -1}
    beliefs = {
        "a": {"ok": 0.0, "t0only": 0.0, "dead": 1.0},    # a fails T0
        "x": {"ok": 1.0, "t0only": 0.0, "dead": 0.0},
        "y": {"ok": 0.5, "t0only": 0.5, "dead": 0.0},
        "b": {"ok": 0.5, "t0only": 0.5, "dead": 0.0},
        "c": {"ok": 0.5, "t0only": 0.5, "dead": 0.0},
    }
    probed = {("a", 0): 1, ("x", 0): 1, ("x", 1): 1, ("y", 0): 1}
    s_robot = State(
        joint_names=JOINTS, proximal=PROXIMAL, tiers=["T0", "T1"],
        levels={j: levels for j in JOINTS}, beliefs=beliefs,
        gating="robot", probe_counts=probed,
    )
    assert s_robot.status("y", 0) == "pass"
    assert s_robot.status("y", 1) == "blocked"           # a, in chain A, failed T0

    s_chain = State(
        joint_names=JOINTS, proximal=PROXIMAL, tiers=["T0", "T1"],
        levels={j: levels for j in JOINTS}, beliefs=beliefs,
        gating="chain", probe_counts=probed,
    )
    assert s_chain.status("y", 1) == "ready"             # chain gating lets y go on


def test_an_untrusted_skill_is_never_used():
    # Its policy failed the healthy gate, so its tier is unknown even though the belief is certain.
    s = make_state({"a": 1.0}, probed=[("a", 0)])
    s.untrusted = frozenset({("a", 0)})
    assert s.status("a", 0) == "unknown"
    assert s.status("b", 0) == "unknown"                 # the joint above it inherits the unknown
