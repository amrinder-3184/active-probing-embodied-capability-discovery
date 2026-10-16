"""Probe selection: the proposed planner and the three baselines.

All methods use the same State, gating, budget and safety veto. They differ only in which
allowed probe they choose, so a comparison measures the choice and nothing else.
"""
from dataclasses import dataclass

from .belief import expected_information_gain
from .config import config_path, read_yaml

METHODS = ("planner", "random", "fixed", "passive")


@dataclass(frozen=True)
class PlannerConfig:
    risk_weight: float
    energy_weight: float
    energy_scale_j: float
    risk_weights: dict
    risk_cap: float
    likelihood_floor: float
    pass_confidence: float
    max_repeats: int
    gating: str
    budget: int


def load_planner_config(path=None):
    raw = read_yaml(path or config_path("planner.yaml"))
    return PlannerConfig(
        risk_weight=float(raw["risk_weight"]),
        energy_weight=float(raw["energy_weight"]),
        energy_scale_j=float(raw["energy_scale_j"]),
        risk_weights={k: float(v) for k, v in raw["risk_weights"].items()},
        risk_cap=float(raw["risk_cap"]),
        likelihood_floor=float(raw["likelihood_floor"]),
        pass_confidence=float(raw["pass_confidence"]),
        max_repeats=int(raw["max_repeats"]),
        gating=raw["gating"],
        budget=int(raw["budget"]),
    )


@dataclass(frozen=True)
class Scored:
    joint: str
    ti: int
    info_gain: float      # bits
    risk: float           # expected weighted violation score under the belief
    p_violation: float    # chance of any violation under the belief
    energy: float         # expected energy, divided by energy_scale_j
    utility: float


def score(state, joint, ti, table, cfg):
    tier = state.tiers[ti]
    belief = state.beliefs[joint]
    preds = {h: table.get(joint, tier.name, h) for h in belief}

    info = expected_information_gain(
        belief, {h: p.pass_prob for h, p in preds.items()},
        state.levels[joint], cfg.likelihood_floor)
    risk = sum(b * sum(cfg.risk_weights.get(v, 0.0) * p for v, p in preds[h].violation_prob.items())
               for h, b in belief.items())
    p_violation = sum(b * preds[h].p_any_violation for h, b in belief.items())
    energy = sum(b * preds[h].energy_j for h, b in belief.items()) / cfg.energy_scale_j
    utility = info - cfg.risk_weight * risk - cfg.energy_weight * energy
    return Scored(joint, ti, info, risk, p_violation, energy, utility)


def choose(method, state, candidates, table, cfg, rng):
    """Return the (joint, tier index) to probe next, or None to stop.

    The utility sets the order of probes. It never skips an allowed one: a tier can only be
    concluded by probing it, so a probe with zero information is still needed for the report.
    """
    if method == "passive" or not candidates:
        return None
    scored = [score(state, j, ti, table, cfg) for j, ti in candidates]
    # Safety veto, applied to every method: a probe that is too likely to cause a violation is never run.
    safe = [s for s in scored if s.p_violation <= cfg.risk_cap]
    if not safe:
        return None

    if method == "planner":
        best = max(safe, key=lambda s: s.utility)
        return best.joint, best.ti
    if method == "random":
        pick = rng.choice(safe)
    elif method == "fixed":
        # Tier order first, then joint order from the base outwards.
        pick = min(safe, key=lambda s: (s.ti, state.joint_names.index(s.joint)))
    else:
        raise ValueError(f"unknown method: {method}")
    return pick.joint, pick.ti
