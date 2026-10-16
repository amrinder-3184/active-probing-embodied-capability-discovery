"""Train a learned skill by evolution strategies (ES), and gate it on healthy joints.

The policy has two parameters (see policies.py). ES perturbs them, scores each perturbed copy on
randomised healthy joints, and moves the parameters towards the copies that scored best.
"""
import math
from dataclasses import asdict, dataclass

import numpy as np

from .config import config_path, read_yaml
from .damage import Fault
from .policies import PolicyRecord
from .runner import noisy_readings, run_gains


@dataclass(frozen=True)
class TrainSettings:
    kp_bounds: tuple          # (low, high) clamp on the stiffness multiplier
    kd_bounds: tuple          # (low, high) clamp on the damping multiplier
    initial_multiplier: float
    population: int
    instances: int
    generations: int
    sigma: float
    learning_rate: float
    effort_weight: float
    abort_penalty: float
    spread: float
    obs_noise: tuple          # (sigma_angle_rad, sigma_speed_rad_per_s)
    gate_episodes: int
    gate_z: float
    gate_min_lower_bound: float
    table_rollouts: int
    identifiability_gap: float
    seeds: dict


def load_training_config(path=None):
    raw = read_yaml(path or config_path("training.yaml"))
    pol, ev, rnd, obs = raw["policy"], raw["evolution"], raw["randomisation"], raw["observation_noise"]
    gate, assess = raw["gate"], raw["assessment"]
    return TrainSettings(
        kp_bounds=tuple(float(x) for x in pol["kp_multiplier"]),
        kd_bounds=tuple(float(x) for x in pol["kd_multiplier"]),
        initial_multiplier=float(pol["initial_multiplier"]),
        population=int(ev["population"]),
        instances=int(ev["instances"]),
        generations=int(ev["generations"]),
        sigma=float(ev["sigma"]),
        learning_rate=float(ev["learning_rate"]),
        effort_weight=float(ev["effort_weight"]),
        abort_penalty=float(ev["abort_penalty"]),
        spread=float(rnd["spread"]),
        obs_noise=(math.radians(obs["angle_deg"]), math.radians(obs["speed_deg_s"])),
        gate_episodes=int(gate["episodes"]),
        gate_z=float(gate["wilson_z"]),
        gate_min_lower_bound=float(gate["min_lower_bound"]),
        table_rollouts=int(assess["rollouts"]),
        identifiability_gap=float(assess["identifiability_gap"]),
        seeds={k: int(v) for k, v in raw["seeds"].items()},
    )


def wilson_lower(successes, n, z):
    """Lower end of the Wilson score interval for a success rate. Safer than the raw rate for small n."""
    if n == 0:
        return 0.0
    p = successes / n
    z2 = z * z
    centre = p + z2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (centre - margin) / (1 + z2 / n)


def _rewards(rollout, values, tier, budget, settings):
    """One reward per run. Higher is better.

    The main term is the tier's primary metric in units of its own pass limit, so the policy is
    scored on the thing the tier checks. It is linear, not squared, so it still gives a gradient when
    the error is already tiny (a squared term made T1 gains drift to their bounds). The effort term
    discourages demands near the budget.
    """
    primary = tier.checks[0]
    scaled = values[primary.metric] / primary.threshold
    ratio = (rollout.demand / budget) ** 2
    effort = np.where(rollout.alive, ratio, 0.0).sum(axis=0) / np.maximum(rollout.alive.sum(axis=0), 1)
    reward = -scaled - settings.effort_weight * effort
    return np.where(rollout.aborted, -settings.abort_penalty, reward)


def _centred_ranks(scores):
    """Rank-based fitness in [-0.5, 0.5]. Robust to the scale of the rewards."""
    ranks = np.argsort(np.argsort(scores)).astype(float)
    return ranks / (len(scores) - 1) - 0.5


def gate_policy(joint, tier, skill, kp_multiplier, kd_multiplier, platform, settings, seed):
    """Run the policy on held-out randomised healthy joints and decide whether to trust it."""
    rng = np.random.default_rng(seed)
    n = settings.gate_episodes
    healthy = Fault()
    _, values = run_gains(platform, joint, skill, kp_multiplier, kd_multiplier,
                          healthy, rng, n, settings.spread, settings.obs_noise)
    passed = tier.passes_array(noisy_readings(values, tier, rng))
    successes = int(np.sum(passed))
    lower = wilson_lower(successes, n, settings.gate_z)
    return {
        "successes": successes,
        "episodes": n,
        "lower_bound": round(lower, 4),
        "trusted": bool(lower >= settings.gate_min_lower_bound),
    }


def train_skill(joint, tier, skill, platform, settings, seed):
    """Train the two gain multipliers for one (joint, skill). Returns a PolicyRecord."""
    rng = np.random.default_rng(seed)
    low = np.array([settings.kp_bounds[0], settings.kd_bounds[0]])
    high = np.array([settings.kp_bounds[1], settings.kd_bounds[1]])
    population, instances = settings.population, settings.instances
    half = population // 2
    budget = np.full(population * instances, skill.budget_fraction * joint.tau_max_nm)
    healthy = Fault()

    theta = np.full(2, settings.initial_multiplier)
    history = []
    for _ in range(settings.generations):
        draws = rng.standard_normal((half, 2))
        eps = np.concatenate([draws, -draws])            # antithetic pairs halve the variance
        candidates = np.clip(theta + settings.sigma * eps, low, high)
        batch = np.repeat(candidates, instances, axis=0)  # each candidate runs on several joints

        rollout, values = run_gains(platform, joint, skill, batch[:, 0], batch[:, 1], healthy,
                                    rng, len(batch), settings.spread, settings.obs_noise)
        reward = _rewards(rollout, values, tier, budget, settings).reshape(population, instances).mean(axis=1)

        fitness = _centred_ranks(reward)
        step = fitness @ eps / (population * settings.sigma)
        theta = np.clip(theta + settings.learning_rate * step, low, high)
        history.append(round(float(reward.mean()), 4))

    # The gate uses its own seed stream, offset by the same job index as the training seed,
    # so gate and training streams never overlap (configs/training.yaml, seeds).
    gate_seed = settings.seeds["gate"] + (seed - settings.seeds["training"])
    gate = gate_policy(joint, tier, skill, theta[0], theta[1], platform, settings, gate_seed)
    return PolicyRecord(
        joint=joint.name,
        tier=tier.name,
        kp_multiplier=round(float(theta[0]), 4),
        kd_multiplier=round(float(theta[1]), 4),
        trusted=gate["trusted"],
        gate=gate,
        training={"seed": int(seed), "settings": asdict(settings)},
        history=history,
    )
