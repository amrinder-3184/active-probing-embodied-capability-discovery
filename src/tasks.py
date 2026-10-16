"""Skills: the reference motion and the readings for each tier's task.

A scripted probe and a learned policy run the same skill, so their readings are comparable,
and both are judged by the same tier checks in configs/tiers.yaml.
"""
import math
from dataclasses import dataclass

import numpy as np

from .physics import DT


def _min_jerk(s):
    """Smooth 0-to-1 profile with zero speed and acceleration at both ends."""
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def _reference(kind, params, duration, t):
    """Commanded angle in rad at time t (s), relative to the home pose."""
    if kind == "small_step":
        return math.radians(params["step_deg"])
    if kind == "smooth_move":
        return math.radians(params["move_deg"]) * _min_jerk(t / params["move_s"])
    if kind == "planned_path":
        # Out to the amplitude and back: a planned path, not a single move.
        amp = math.radians(params["amplitude_deg"])
        u = t / duration
        return amp * _min_jerk(2 * u) if u <= 0.5 else amp * _min_jerk(2 * (1 - u))
    if kind == "hold_under_load":
        return 0.0
    raise ValueError(f"unknown skill kind: {kind}")


@dataclass(frozen=True)
class Skill:
    tier: str
    kind: str
    duration_s: float
    params: dict
    budget_fraction: float   # overload budget as a fraction of the joint's nominal tau_max

    @property
    def n_steps(self):
        return round(self.duration_s / DT)

    def reference(self, t):
        return _reference(self.kind, self.params, self.duration_s, t)

    @property
    def holds_load(self):
        return self.kind == "hold_under_load"


def skill_for(platform, tier_name):
    """The skill the platform config defines for a tier.

    budget_fraction is the safety budget for this skill: a demand above it aborts the run. It
    defaults to the platform-wide probe budget.
    """
    spec = dict(platform.probes[tier_name])
    kind = spec.pop("kind")
    duration = spec.pop("duration_s")
    budget = spec.pop("budget_fraction", platform.probe_torque_limit_fraction)
    return Skill(tier=tier_name, kind=kind, duration_s=float(duration), params=spec,
                 budget_fraction=float(budget))


def readings(skill, rollout, nominal_tau_max, torque_constant):
    """Metric readings for each rollout, shape (n,). Aborted rollouts give NaN."""
    p = skill.params
    alive = rollout.alive
    if skill.kind == "small_step":
        values = {
            "response_ratio": rollout.q[-1] / math.radians(p["step_deg"]),
            "peak_current_a": np.where(alive, np.abs(rollout.tau), 0.0).max(axis=0) / torque_constant,
        }
    elif skill.kind == "smooth_move":
        values = {"final_error_deg": np.degrees(np.abs(rollout.ref[-1] - rollout.q[-1]))}
    elif skill.kind == "planned_path":
        err = rollout.ref - rollout.q
        values = {"rms_error_deg": np.degrees(np.sqrt((err**2).mean(axis=0)))}
    elif skill.kind == "hold_under_load":
        # Sag is measured over the last window of the hold, once the joint has settled.
        window = round(p["sag_window_s"] / DT)
        err = np.abs(rollout.ref[-window:] - rollout.q[-window:])
        values = {
            "sag_deg": np.degrees(err.max(axis=0)),
            "peak_torque_fraction": np.where(alive, np.abs(rollout.tau), 0.0).max(axis=0) / nominal_tau_max,
        }
    else:
        raise ValueError(f"unknown skill kind: {skill.kind}")

    n = rollout.q.shape[1]
    values["sensor_valid"] = np.ones(n)
    ok = ~rollout.aborted
    return {name: np.where(ok, value, np.nan) for name, value in values.items()}
