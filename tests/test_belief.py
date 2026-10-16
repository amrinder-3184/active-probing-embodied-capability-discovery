import math

from src.belief import binary_entropy, expected_information_gain, prob_level_at_least, update


def test_consistent_evidence_moves_belief_towards_the_hypothesis_that_predicts_it():
    belief = {"ok": 0.5, "dead": 0.5}
    pass_prob = {"ok": 0.99, "dead": 0.01}
    after = update(belief, pass_prob, passed=True, floor=0.0)
    assert after["ok"] > 0.95
    assert math.isclose(sum(after.values()), 1.0)


def test_update_is_floored_so_one_wrong_prediction_cannot_rule_a_hypothesis_out():
    belief = {"ok": 0.5, "dead": 0.5}
    after = update(belief, {"ok": 1.0, "dead": 0.0}, passed=True, floor=0.01)
    assert after["dead"] > 0.0


def test_no_information_when_all_hypotheses_predict_the_same_outcome():
    belief = {"a": 0.3, "b": 0.7}
    levels = {"a": 1, "b": 3}
    gain = expected_information_gain(belief, {"a": 0.9, "b": 0.9}, levels, floor=0.0)
    assert abs(gain) < 1e-12


def test_information_when_the_outcome_separates_the_levels():
    belief = {"a": 0.5, "b": 0.5}
    levels = {"a": -1, "b": 3}
    gain = expected_information_gain(belief, {"a": 0.0, "b": 1.0}, levels, floor=0.0)
    assert math.isclose(gain, 1.0)                # one full bit: the outcome reveals the level


def test_information_is_bounded_by_the_entropy_of_the_outcome():
    belief = {"a": 0.25, "b": 0.25, "c": 0.5}
    levels = {"a": -1, "b": 1, "c": 3}
    pass_prob = {"a": 0.0, "b": 0.6, "c": 0.9}
    gain = expected_information_gain(belief, pass_prob, levels, floor=0.0)
    p_pass = sum(belief[h] * pass_prob[h] for h in belief)
    assert 0.0 <= gain <= binary_entropy(p_pass) + 1e-12


def test_binary_entropy_values():
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0
    assert math.isclose(binary_entropy(0.5), 1.0)


def test_prob_level_at_least_sums_the_hypotheses_that_reach_the_tier():
    belief = {"ok": 0.6, "weak": 0.3, "dead": 0.1}
    levels = {"ok": 3, "weak": 2, "dead": -1}
    assert math.isclose(prob_level_at_least(belief, levels, 2), 0.9)
    assert math.isclose(prob_level_at_least(belief, levels, 3), 0.6)
