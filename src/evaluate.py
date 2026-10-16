"""Compare a physical state report with the hidden truth. Evaluation only.

The planner and the report never see the true damage. Checks, kept separate:
  hypothesis_accuracy  is the most likely damage state the true one? (the damage question)
  accuracy / overclaim the reported level against the reference level (the scripted controller's)
  coverage             how many joints got a determined level at all

Probabilistic scores of the final belief (docs/assessment.md, section 8, has the conventions):
  level_brier, level_log_loss
      For each joint and each tier index k, the forecast p_k = P(level >= k) under the final belief
      (model levels, as the planner uses them) is scored against y_k = 1[reference level >= k].
      Brier is the mean of (p_k - y_k)^2 over the tiers. Log-loss is the mean of
      -[y_k ln p_k + (1 - y_k) ln(1 - p_k)] in nats, with p_k clipped to [EPS, 1 - EPS].
  hypothesis_expected_accuracy, hypothesis_tie_rate
      The MAP set is every hypothesis within TIE_TOL of the largest belief. Expected accuracy is
      1[truth in set] / |set|: the accuracy of breaking ties uniformly at random. hypothesis_accuracy
      above breaks them by dict order instead, which favours whichever hypothesis is listed first.
  hypothesis_brier
      Multi-class Brier score of the belief against the true hypothesis: sum over h of
      (b_h - 1[h = truth])^2. Range [0, 2].
All scores are averaged over the joints. Lower is better for Brier and log-loss.

Cost and safety, per episode (from the report, so from the executed probes only):
  time_s               seconds the executed skills actually ran (an aborted skill counts until its abort)
  violations_<type>    probes that hit each safety limit (overload, joint_limit), cumulative

paired_bootstrap compares two methods run on the same seeds (so the same hidden damage): it takes
the per-seed differences and gives their mean with a percentile bootstrap interval over seeds.
"""
import math

import numpy as np

EPS = 1e-9       # probability clip for log-loss: one event costs at most -ln(EPS), about 20.7 nats
TIE_TOL = 1e-12  # beliefs this close to the maximum count as tied


def map_set(belief, tol=TIE_TOL):
    """Hypotheses whose belief is within `tol` of the largest one."""
    top = max(belief.values())
    return [h for h, p in belief.items() if p >= top - tol]


def level_forecasts(belief, levels, n_tiers):
    """P(level >= k) for k = 0 .. n_tiers - 1, from the belief and the per-hypothesis levels."""
    return [sum(p for h, p in belief.items() if levels[h] >= k) for k in range(n_tiers)]


def brier(forecasts, outcomes):
    return sum((p - y) ** 2 for p, y in zip(forecasts, outcomes)) / len(forecasts)


def log_loss(forecasts, outcomes, eps=EPS):
    total = 0.0
    for p, y in zip(forecasts, outcomes):
        p = min(max(p, eps), 1.0 - eps)
        total -= math.log(p) if y else math.log(1.0 - p)
    return total / len(forecasts)


def belief_scores(belief, levels, reference_level, truth, n_tiers):
    """Probabilistic and tie-aware scores of one joint's belief. See the module docstring."""
    forecasts = level_forecasts(belief, levels, n_tiers)
    outcomes = [1 if reference_level >= k else 0 for k in range(n_tiers)]
    tied = map_set(belief)
    return {
        "level_brier": brier(forecasts, outcomes),
        "level_log_loss": log_loss(forecasts, outcomes),
        "hypothesis_expected_accuracy": (truth in tied) / len(tied),
        "hypothesis_tie": len(tied) > 1,
        "hypothesis_brier": sum((p - (h == truth)) ** 2 for h, p in belief.items()),
    }


def paired_bootstrap(a, b, resamples=2000, level=0.95, seed=0):
    """Mean of a[i] - b[i] and its percentile bootstrap interval, resampling the pairs i.

    a and b hold one value per seed, in the same seed order. Returns (mean, low, high).
    The interval is reproducible: its resampling uses its own seeded generator.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or a.size == 0:  # no broadcasting: pairs must match one to one
        raise ValueError("need two equal-length, non-empty sequences of paired values")
    d = a - b
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(resamples, d.size))].mean(axis=1)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(d.mean()), float(low), float(high)


def score_episode(report, episode):
    rows = []
    hypothesis_hits = 0
    belief_rows = []
    state = episode.state
    n_tiers = len(state.tiers)
    for part in report["parts"]:
        joint = part["joint"]
        truth = episode.true_states[joint]
        reference = episode.reference_levels[joint][truth]
        rows.append((part["level"], part["determined"], reference))
        hypothesis_hits += part["most_likely_state"] == truth
        belief_rows.append(belief_scores(state.beliefs[joint], state.levels[joint], reference, truth, n_tiers))
    n = len(rows)

    def mean(key):
        return sum(r[key] for r in belief_rows) / n

    return {
        "coverage": sum(det for _, det, _ in rows) / n,
        "accuracy": sum(det and lvl == ref for lvl, det, ref in rows) / n,
        "overclaim": sum(lvl > ref for lvl, _, ref in rows) / n,
        "hypothesis_accuracy": hypothesis_hits / n,
        "probes": report["probes_used"],
        "aborts": report["aborted_probes"],
        "energy_j": report["energy_j"],
        "level_brier": mean("level_brier"),
        "level_log_loss": mean("level_log_loss"),
        "hypothesis_expected_accuracy": mean("hypothesis_expected_accuracy"),
        "hypothesis_tie_rate": mean("hypothesis_tie"),
        "hypothesis_brier": mean("hypothesis_brier"),
        "time_s": report["time_s"],
        **{f"violations_{v}": n for v, n in report["violations"].items()},
    }
