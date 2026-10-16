"""Scripted and learned assessors: the two sources of evidence share one interface."""
import numpy as np

from src.assessors import Execution, LearnedAssessor, Prediction, ScriptedAssessor
from src.tiers import load_tiers


def _tier(name):
    return {t.name: t for t in load_tiers()}[name]


def test_scripted_and_learned_return_the_same_prediction_shape(arm, shoulder_store, fast_settings):
    platform, _, damage, _ = arm
    learned = LearnedAssessor(platform, shoulder_store, fast_settings, base_seed=1)
    scripted = ScriptedAssessor(platform)
    joint = platform.joint("shoulder")
    for assessor in (scripted, learned):
        pred = assessor.predict(joint, _tier("T1"), damage.faults["healthy"], seed_key=(0, 0, 0))
        assert isinstance(pred, Prediction)
        assert 0.0 <= pred.pass_prob <= 1.0
        assert 0.0 <= pred.p_any_violation <= 1.0


def test_a_trained_healthy_shoulder_passes_the_move_most_of_the_time(arm, shoulder_store, fast_settings):
    platform, _, damage, _ = arm
    learned = LearnedAssessor(platform, shoulder_store, fast_settings, base_seed=1)
    pred = learned.predict(platform.joint("shoulder"), _tier("T1"), damage.faults["healthy"], seed_key=(0, 0, 0))
    assert pred.pass_prob > 0.8
    assert pred.trusted


def test_a_missing_sensor_is_never_run_by_a_learned_policy(arm, shoulder_store, fast_settings):
    platform, _, damage, _ = arm
    learned = LearnedAssessor(platform, shoulder_store, fast_settings, base_seed=1)
    joint = platform.joint("shoulder")
    pred = learned.predict(joint, _tier("T2"), damage.faults["missing_sensor"], seed_key=(0, 1, 2))
    assert pred.pass_prob == 0.0 and pred.p_any_violation == 0.0 and pred.energy_j == 0.0
    result = learned.execute(joint, _tier("T2"), damage.faults["missing_sensor"], np.random.default_rng(0))
    assert result == Execution(passed=False, violations=frozenset(), energy_j=0.0)


def test_a_weak_shoulder_hold_is_predicted_to_violate_its_budget(arm, shoulder_store, fast_settings):
    platform, _, damage, _ = arm
    learned = LearnedAssessor(platform, shoulder_store, fast_settings, base_seed=1)
    pred = learned.predict(platform.joint("shoulder"), _tier("T3"), damage.faults["weak_motor"], seed_key=(0, 2, 1))
    assert pred.violation_prob["overload"] > 0.5
    assert pred.pass_prob < 0.5


def test_execution_reports_the_violations_it_hit(arm, shoulder_store, fast_settings):
    platform, _, damage, _ = arm
    learned = LearnedAssessor(platform, shoulder_store, fast_settings, base_seed=1)
    result = learned.execute(platform.joint("shoulder"), _tier("T3"), damage.faults["weak_motor"],
                             np.random.default_rng(4))
    assert isinstance(result, Execution)
    assert "overload" in result.violations or result.passed is False


def test_scripted_execution_on_a_locked_joint_fails(arm):
    platform, _, damage, _ = arm
    scripted = ScriptedAssessor(platform)
    result = scripted.execute(platform.joint("base"), _tier("T0"), damage.faults["locked"],
                              np.random.default_rng(0))
    assert result.passed is False
