"""Assessors: where the evidence about a joint comes from.

Each tier is assessed by one assessor. Both return the same two shapes:
  Prediction  what the planner expects under one damage hypothesis (for the belief and the risk)
  Execution   what happens on the robot when a skill is run (the probe result)

ScriptedAssessor runs the PD reference controller (the T0 check and the reference probes).
LearnedAssessor runs a trained policy from the policy store.
"""
import warnings
from dataclasses import dataclass

import numpy as np

from .physics import DT
from .runner import noisy_readings, run_gains
from .tasks import skill_for


@dataclass(frozen=True)
class Prediction:
    pass_prob: float            # chance the tier passes, noise included
    violation_prob: dict        # violation name -> chance it happens
    p_any_violation: float      # chance of any violation
    energy_j: float             # expected energy proxy
    metrics: dict               # expected readings (NaN where there is no reading)
    margin: float               # distance to the tightest limit; positive is inside, -inf if nothing
    trusted: bool = True        # False: the learned policy did not pass the healthy gate


@dataclass(frozen=True)
class Execution:
    passed: bool
    violations: frozenset
    energy_j: float
    duration_s: float = 0.0     # time the skill actually ran: steps executed x DT (an abort stops it early)


def _executed_duration(rollout):
    """Seconds the single rollout ran before it ended or aborted (physics.simulate's alive history)."""
    return float(rollout.alive[:, 0].sum()) * DT


def _missing_prediction(tier, trusted=True):
    nan_metrics = {c.metric: float("nan") for c in tier.checks}
    nan_metrics["sensor_valid"] = 0.0
    return Prediction(pass_prob=0.0, violation_prob={}, p_any_violation=0.0, energy_j=0.0,
                      metrics=nan_metrics, margin=float("-inf"), trusted=trusted)


def _missing_execution():
    # Without an encoder the controller cannot close the loop, so nothing is run.
    return Execution(passed=False, violations=frozenset(), energy_j=0.0)


def _safe_margin(margins):
    """Mean margin over the rollouts that did run. -inf when none did."""
    finite = margins[~np.isnan(margins)]
    return float(finite.mean()) if finite.size else float("-inf")


class ScriptedAssessor:
    """The PD reference controller, with nominal parameters and no noise (for prediction)."""

    def __init__(self, platform):
        self.platform = platform

    def predict(self, joint, tier, fault, seed_key=None):
        if not fault.sensor_valid:
            return _missing_prediction(tier)
        skill = skill_for(self.platform, tier.name)
        rollout, values = run_gains(self.platform, joint, skill, 1.0, 1.0, fault,
                                    rng=None, n=1, spread=0.0, obs_noise=None)
        metrics = {k: float(v[0]) for k, v in values.items()}
        margin = float(tier.margin_array({k: np.array([v]) for k, v in metrics.items()})[0])
        return Prediction(
            pass_prob=tier.pass_probability(metrics),
            violation_prob={k: float(v[0]) for k, v in rollout.violations.items()},
            p_any_violation=float(rollout.aborted[0]),
            energy_j=float(rollout.energy_j[0]),
            metrics=metrics,
            margin=margin if np.isfinite(margin) else float("-inf"),
        )

    def execute(self, joint, tier, fault, rng):
        if not fault.sensor_valid:
            return _missing_execution()
        skill = skill_for(self.platform, tier.name)
        rollout, values = run_gains(self.platform, joint, skill, 1.0, 1.0, fault,
                                    rng=rng, n=1, spread=0.0, obs_noise=None)
        passed = bool(tier.passes_array(noisy_readings(values, tier, rng))[0])
        return Execution(
            passed=passed,
            violations=frozenset(k for k, v in rollout.violations.items() if v[0]),
            energy_j=float(rollout.energy_j[0]),
            duration_s=_executed_duration(rollout),
        )


class LearnedAssessor:
    """A trained policy from the store, run on randomised joints."""

    def __init__(self, platform, store, settings, base_seed):
        self.platform = platform
        self.store = store
        self.settings = settings
        self.base_seed = base_seed

    def _policy(self, joint, tier):
        return self.store.get(joint.name, tier.name)

    def predict(self, joint, tier, fault, seed_key):
        """Model table entry: K rollouts with randomised plants, input and reading noise."""
        record = self._policy(joint, tier)
        if not fault.sensor_valid:
            return _missing_prediction(tier, trusted=record.trusted)
        settings = self.settings
        k = settings.table_rollouts
        rng = np.random.default_rng(np.random.SeedSequence([self.base_seed, *seed_key]))
        skill = skill_for(self.platform, tier.name)
        rollout, values = run_gains(self.platform, joint, skill, record.kp_multiplier, record.kd_multiplier,
                                    fault, rng=rng, n=k, spread=settings.spread,
                                    obs_noise=settings.obs_noise)
        passed = tier.passes_array(noisy_readings(values, tier, rng)) & ~rollout.aborted
        successes = int(np.sum(passed))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN columns are expected
            metrics = {name: float(np.nanmean(v)) if np.any(~np.isnan(v)) else float("nan")
                       for name, v in values.items()}
        return Prediction(
            pass_prob=(successes + 1) / (k + 2),          # Laplace smoothing keeps 0 and 1 out
            violation_prob={name: float(np.mean(v)) for name, v in rollout.violations.items()},
            p_any_violation=float(np.mean(rollout.aborted)),
            energy_j=float(np.mean(rollout.energy_j)),
            metrics=metrics,
            margin=_safe_margin(tier.margin_array(values)),
            trusted=record.trusted,
        )

    def execute(self, joint, tier, fault, rng):
        """One rollout on the robot, with a fresh randomised plant and noise."""
        if not fault.sensor_valid:
            return _missing_execution()
        record = self._policy(joint, tier)
        skill = skill_for(self.platform, tier.name)
        rollout, values = run_gains(self.platform, joint, skill, record.kp_multiplier, record.kd_multiplier,
                                    fault, rng=rng, n=1, spread=self.settings.spread,
                                    obs_noise=self.settings.obs_noise)
        passed = bool(tier.passes_array(noisy_readings(values, tier, rng))[0])
        return Execution(
            passed=passed,
            violations=frozenset(k for k, v in rollout.violations.items() if v[0]),
            energy_j=float(rollout.energy_j[0]),
            duration_s=_executed_duration(rollout),
        )
