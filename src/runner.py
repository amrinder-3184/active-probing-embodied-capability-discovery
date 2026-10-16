"""Run one skill on a batch of joints. Shared by scripted probes and learned policies.

Two kinds of noise, kept separate:
  input noise   inside the control loop: what the controller sees (passed to simulate)
  reading noise on the metrics: what the tier check sees (added by noisy_readings)
"""
import numpy as np

from .physics import make_plant, simulate
from .tasks import readings
from .policies import controller_for


def run_skill(platform, joint, skill, controller, fault, rng, n, spread, obs_noise):
    """Simulate `skill` on n copies of `joint` with `fault`. Returns (rollout, readings)."""
    plant = make_plant(joint, fault, n, rng=rng, spread=spread, gravity_on=skill.holds_load)
    budget = np.full(n, skill.budget_fraction * joint.tau_max_nm)
    rollout = simulate(
        plant, budget, controller, skill.reference, skill.n_steps, platform.heat_coeff,
        obs_noise=obs_noise, rng=rng,
    )
    values = readings(skill, rollout, np.full(n, joint.tau_max_nm), platform.torque_constant_nm_per_a)
    return rollout, values


def run_gains(platform, joint, skill, kp_multiplier, kd_multiplier, fault, rng, n, spread, obs_noise):
    """Run the learned (or scripted, multipliers = 1) control law on n joints."""
    controller = controller_for(joint, kp_multiplier, kd_multiplier, n)
    return run_skill(platform, joint, skill, controller, fault, rng, n, spread, obs_noise)


def noisy_readings(values, tier, rng):
    """Add reading noise to each check's metric, as the real sensor would."""
    out = dict(values)
    for c in tier.checks:
        if c.noise_sigma > 0.0:
            size = np.shape(values[c.metric])
            out[c.metric] = values[c.metric] + rng.normal(0.0, c.noise_sigma, size)
    return out
