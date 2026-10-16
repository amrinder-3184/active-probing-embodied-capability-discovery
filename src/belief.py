"""Belief about one joint: a probability for each damage hypothesis.

The belief is updated with Bayes' rule after every probe. The planner measures information on
the capability level (what the report shows), not on the cause of the damage.
"""
import math


def clip(p, floor):
    return min(max(p, floor), 1.0 - floor)


def update(belief, pass_prob, passed, floor):
    """Bayes update after one probe outcome.

    pass_prob[h] is the chance the tier passes if hypothesis h holds. The floor stops one wrong
    model prediction from ruling a hypothesis out for good.
    """
    posterior = {}
    for h, prior in belief.items():
        p = clip(pass_prob[h], floor)
        posterior[h] = prior * (p if passed else 1.0 - p)
    total = sum(posterior.values())
    return {h: value / total for h, value in posterior.items()}


def prob_level_at_least(belief, levels, ti):
    """P(joint reaches tier index ti): summed over the hypotheses that reach it."""
    return sum(p for h, p in belief.items() if levels[h] >= ti)


def binary_entropy(p):
    """Entropy of a yes/no outcome, in bits."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def expected_information_gain(belief, pass_prob, levels, floor):
    """Expected bits about the capability level that one probe outcome reveals.

    This is the mutual information between the level L and the pass/fail outcome O:
        I(L; O) = H(O) - sum over l of P(l) * H(O | l)
    """
    mass = {}       # level -> P(level)
    mass_pass = {}  # level -> P(level and pass)
    for h, p in belief.items():
        lvl = levels[h]
        pp = clip(pass_prob[h], floor)
        mass[lvl] = mass.get(lvl, 0.0) + p
        mass_pass[lvl] = mass_pass.get(lvl, 0.0) + p * pp
    gain = binary_entropy(sum(mass_pass.values()))
    for lvl, m in mass.items():
        if m > 0.0:
            gain -= m * binary_entropy(mass_pass[lvl] / m)
    return gain
