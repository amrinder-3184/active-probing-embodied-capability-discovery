"""Tier definitions: the metrics each tier checks and what counts as a pass.

The numbers live in configs/tiers.yaml. This module only applies them. Scalar and array versions
are provided: the scripted probes use the scalar ones, and batched rollouts use the array ones.
"""
import math
from dataclasses import dataclass

import numpy as np

from .config import config_path, read_yaml


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass(frozen=True)
class Check:
    metric: str
    direction: str       # "min": reading must be >= threshold. "max": reading must be <= threshold.
    threshold: float
    noise_sigma: float   # sensor noise on this reading

    def passes_array(self, values):
        # NaN means the probe gave no usable reading (e.g. missing sensor). That is a fail.
        v = np.asarray(values, dtype=float)
        ok = v >= self.threshold if self.direction == "min" else v <= self.threshold
        return ok & ~np.isnan(v)

    def margin_array(self, values):
        """Signed distance to the limit: positive means inside it. NaN where there is no reading."""
        v = np.asarray(values, dtype=float)
        return v - self.threshold if self.direction == "min" else self.threshold - v

    def passes(self, reading):
        return bool(self.passes_array(reading))

    def pass_probability(self, predicted):
        """Chance the noisy reading passes, if the noise-free reading would be `predicted`."""
        if math.isnan(predicted):
            return 0.0
        margin = float(self.margin_array(predicted))
        if self.noise_sigma == 0.0:
            return 1.0 if margin >= 0.0 else 0.0
        return normal_cdf(margin / self.noise_sigma)


@dataclass(frozen=True)
class Tier:
    name: str    # "T0" ...
    label: str   # "Alive" ...
    checks: tuple

    def passes_array(self, metrics):
        # A tier passes only if every check passes.
        result = None
        for c in self.checks:
            ok = c.passes_array(metrics[c.metric])
            result = ok if result is None else result & ok
        return result

    def margin_array(self, metrics):
        # The tightest check decides how close the tier is to failing. NaN in any check stays NaN.
        return np.min(np.stack([c.margin_array(metrics[c.metric]) for c in self.checks]), axis=0)

    def passes(self, metrics):
        return bool(self.passes_array({k: np.asarray(v, dtype=float) for k, v in metrics.items()}))

    def pass_probability(self, metrics):
        # Checks use independent noise, so their pass chances multiply.
        p = 1.0
        for c in self.checks:
            p *= c.pass_probability(metrics[c.metric])
        return p


def load_tiers(path=None):
    raw = read_yaml(path or config_path("tiers.yaml"))["tiers"]
    return [
        Tier(name=name, label=spec["name"], checks=tuple(Check(**c) for c in spec["checks"]))
        for name, spec in raw.items()
    ]
