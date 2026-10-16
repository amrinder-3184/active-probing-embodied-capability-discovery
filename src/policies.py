"""Control law and stored policy records.

A stage-1 policy is two gain multipliers per joint and skill:

    stiffness term = kp_mult * kp * error
    damping term   = kd_mult * kd * speed        (applied with a minus sign)

kp and kd are the joint's nominal PD gains from the platform config. The scripted PD controller is
this same law with both multipliers at 1. There is no feedforward, no gravity bias and no input about
the fault, so the policy cannot hide the damage it is meant to reveal.
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LinearFeedback:
    """Two gains per joint in the batch. torques(err, qd) returns (stiffness term, damping term)."""
    kp: np.ndarray
    kd: np.ndarray

    def torques(self, err, qd):
        return self.kp * err, -self.kd * qd


def controller_for(joint, kp_multiplier, kd_multiplier, n):
    """Controller for n copies of `joint`, with the given gain multipliers (scalars or (n,) arrays)."""
    kp_m = np.broadcast_to(np.asarray(kp_multiplier, dtype=float), (n,))
    kd_m = np.broadcast_to(np.asarray(kd_multiplier, dtype=float), (n,))
    return LinearFeedback(kp=joint.kp * kp_m, kd=joint.kd * kd_m)


@dataclass(frozen=True)
class PolicyRecord:
    """A trained policy, with the evidence for trusting it."""
    joint: str
    tier: str
    kp_multiplier: float
    kd_multiplier: float
    trusted: bool
    gate: dict                 # healthy-gate result: successes, episodes, lower bound
    training: dict             # the settings and seed used
    history: list = field(default_factory=list)  # mean reward per generation

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(raw):
        return PolicyRecord(**raw)


class PolicyStore:
    """Policies saved as one JSON file per (joint, tier)."""

    def __init__(self, root):
        self.root = Path(root)

    def _path(self, joint, tier):
        return self.root / f"{joint}_{tier}.json"

    def has(self, joint, tier):
        return self._path(joint, tier).exists()

    def put(self, record):
        self.root.mkdir(parents=True, exist_ok=True)
        self._path(record.joint, record.tier).write_text(json.dumps(record.to_dict(), indent=2))

    def get(self, joint, tier):
        path = self._path(joint, tier)
        if not path.exists():
            raise FileNotFoundError(
                f"no trained policy for {joint} {tier} in {self.root}. "
                "Run `python -m experiments.train_policies` first.")
        return PolicyRecord.from_dict(json.loads(path.read_text()))
