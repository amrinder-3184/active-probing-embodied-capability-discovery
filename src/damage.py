"""Damage hypotheses: what each fault does to a single joint in the simulator.

The hypotheses and their prior are in configs/damage.yaml.
"""
from dataclasses import dataclass

from .config import config_path, read_yaml


@dataclass(frozen=True)
class Fault:
    locked: bool = False          # motor cannot move the joint
    torque_scale: float = 1.0     # fraction of nominal torque the motor can deliver
    sensor_valid: bool = True     # False: no encoder reading
    stiffness_scale: float = 1.0  # fraction of the position-loop stiffness left


@dataclass(frozen=True)
class DamageModel:
    names: tuple     # hypothesis names, healthy first
    faults: dict     # name -> Fault
    prior: dict      # name -> probability


def load_damage(path=None):
    raw = read_yaml(path or config_path("damage.yaml"))
    names = tuple(raw["hypotheses"])
    faults = {name: Fault(**(spec or {})) for name, spec in raw["hypotheses"].items()}
    prior = {name: float(p) for name, p in raw["prior"].items()}
    if set(prior) != set(names):
        raise ValueError("prior must list exactly the hypotheses")
    if abs(sum(prior.values()) - 1.0) > 1e-9:
        raise ValueError("prior must sum to 1")
    return DamageModel(names=names, faults=faults, prior=prior)


def sample_fault_name(model, rng):
    """Draw one hypothesis from the prior. Used to build the simulated ground truth."""
    r = rng.random()
    cumulative = 0.0
    for name in model.names:
        cumulative += model.prior[name]
        if r < cumulative:
            return name
    return model.names[-1]  # guards against floating-point rounding at the top end
