"""Which tier tests are allowed now, and the status of each tier of each joint.

Gating: a tier of a joint is tested only after the tiers it depends on have passed.
  chain mode (default): its own lower tiers, and the same tier on each joint below it in its chain.
  robot mode (whole-robot baseline): its own lower tiers, and the tier below on every other joint.

Status of a joint at a tier:
  pass / fail  concluded from the belief after at least one probe of that tier
               (see pass_confidence in configs/planner.yaml)
  blocked      a prerequisite failed, so the tier is never tested
  unknown      the repeat limit ran out, a prerequisite is unknown, or the learned policy
               for this tier failed the healthy gate
  ready        allowed now, not yet concluded
  wait         a prerequisite has not passed yet
"""
from dataclasses import dataclass, field

from .belief import prob_level_at_least


@dataclass
class State:
    joint_names: tuple
    proximal: dict          # joint -> joints below it in its chain
    tiers: list             # the tiers this platform uses, T0 first
    levels: dict            # joint -> {hypothesis: level}
    beliefs: dict           # joint -> {hypothesis: probability}
    gating: str = "chain"
    confidence: float = 0.95
    max_repeats: int = 2
    probe_counts: dict = field(default_factory=dict)  # (joint, tier index) -> probes run
    untrusted: frozenset = frozenset()                 # (joint, tier index) with no trusted assessor

    def status(self, joint, ti):
        # A skill whose learned policy failed the healthy gate is not used at all.
        if (joint, ti) in self.untrusted:
            return "unknown"
        # A tier is only concluded from a probe of that tier. The belief can be confident
        # without one (e.g. T0 evidence rules out a hypothesis that T2 would catch), and a
        # physical report must not call a tier passed on that basis alone.
        probed = self.probe_counts.get((joint, ti), 0)
        p_reach = prob_level_at_least(self.beliefs[joint], self.levels[joint], ti)
        if probed and p_reach >= self.confidence:
            return "pass"
        if probed and p_reach <= 1.0 - self.confidence:
            return "fail"
        if probed >= self.max_repeats:
            return "unknown"

        results = [self.status(joint, lower) for lower in range(ti)]
        results += [self.status(q, t) for q, t in self._prerequisites(joint, ti)]
        if any(r in ("fail", "blocked") for r in results):
            return "blocked"
        if "unknown" in results:
            return "unknown"
        if all(r == "pass" for r in results):
            return "ready"
        return "wait"

    def _prerequisites(self, joint, ti):
        if self.gating == "chain":
            return [(q, ti) for q in self.proximal[joint]]
        if ti == 0:
            return []
        return [(q, ti - 1) for q in self.joint_names if q != joint]
