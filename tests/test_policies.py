"""The control law and the policy store."""
import numpy as np
import pytest

from src.platform_adapter import load_platform
from src.policies import LinearFeedback, PolicyRecord, PolicyStore, controller_for


def test_zero_error_and_zero_speed_give_zero_torque():
    # No gravity bias and no feedforward: with nothing to correct, the policy asks for nothing.
    # A hidden bias would carry a hold and so hide a loss of stiffness.
    controller = LinearFeedback(kp=np.array([1500.0]), kd=np.array([50.0]))
    stiffness, damping = controller.torques(np.zeros(1), np.zeros(1))
    assert stiffness[0] == 0.0 and damping[0] == 0.0


def test_multipliers_scale_the_nominal_gains():
    platform = load_platform()
    joint = platform.joint("shoulder")
    controller = controller_for(joint, 0.5, 2.0, 1)
    assert controller.kp[0] == pytest.approx(0.5 * joint.kp)
    assert controller.kd[0] == pytest.approx(2.0 * joint.kd)
    stiffness, damping = controller.torques(np.array([0.01]), np.array([0.2]))
    assert stiffness[0] == pytest.approx(0.5 * joint.kp * 0.01)
    assert damping[0] == pytest.approx(-2.0 * joint.kd * 0.2)


def test_policy_record_round_trips_through_the_store(tmp_path):
    store = PolicyStore(tmp_path)
    record = PolicyRecord(joint="elbow", tier="T2", kp_multiplier=0.8, kd_multiplier=1.2,
                          trusted=True, gate={"successes": 99, "episodes": 100},
                          training={"seed": 3}, history=[-1.0, -0.5])
    store.put(record)
    assert store.has("elbow", "T2")
    assert store.get("elbow", "T2") == record


def test_missing_policy_says_how_to_train_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_policies"):
        PolicyStore(tmp_path).get("elbow", "T2")
