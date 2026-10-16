"""The batched physics: one model, used for a single joint and for a whole batch."""
import numpy as np

from src.damage import Fault
from src.physics import make_plant, simulate
from src.platform_adapter import load_platform
from src.policies import controller_for
from src.tasks import skill_for


def _run(platform, joint_name, skill_name, fault, n, kp=1.0, kd=1.0):
    joint = platform.joint(joint_name)
    skill = skill_for(platform, skill_name)
    plant = make_plant(joint, fault, n, gravity_on=skill.holds_load)
    budget = np.full(n, skill.budget_fraction * joint.tau_max_nm)
    return simulate(plant, budget, controller_for(joint, kp, kd, n), skill.reference,
                    skill.n_steps, platform.heat_coeff)


def test_a_batch_of_one_matches_a_batch_of_three():
    platform = load_platform()
    one = _run(platform, "elbow", "T2", Fault(), n=1)
    three = _run(platform, "elbow", "T2", Fault(), n=3)
    for k in range(3):
        assert np.array_equal(one.q[:, 0], three.q[:, k])
        assert np.array_equal(one.tau[:, 0], three.tau[:, k])


def test_a_locked_joint_in_the_batch_does_not_change_the_others():
    platform = load_platform()
    joint = platform.joint("elbow")
    skill = skill_for(platform, "T1")
    n = 2
    healthy_only = simulate(
        make_plant(joint, Fault(), n), np.full(n, skill.budget_fraction * joint.tau_max_nm),
        controller_for(joint, 1.0, 1.0, n), skill.reference, skill.n_steps, platform.heat_coeff)
    # Same two joints, but the second one is locked. Build it by hand: the batch takes one fault.
    plant = make_plant(joint, Fault(), n)
    plant = plant.__class__(**{**plant.__dict__, "locked": np.array([False, True])})
    mixed = simulate(
        plant, np.full(n, skill.budget_fraction * joint.tau_max_nm),
        controller_for(joint, 1.0, 1.0, n), skill.reference, skill.n_steps, platform.heat_coeff)
    assert np.array_equal(healthy_only.q[:, 0], mixed.q[:, 0])
    assert not mixed.aborted[0]


def test_an_aborted_joint_stops_moving():
    platform = load_platform()
    # A weak shoulder cannot hold its load, so its hold aborts on overload. After the abort the
    # recorded steps are not executed.
    rollout = _run(platform, "shoulder", "T3", Fault(torque_scale=0.4), n=1)
    assert rollout.violations["overload"][0]
    first_abort = int(np.argmax(~rollout.alive[:, 0]))
    assert not rollout.alive[first_abort:, 0].any()


def test_a_locked_joint_stalls_against_the_direction_it_is_asked_to_move():
    # The stalled motor pushes at full torque in the direction of the demand, and not at all
    # when nothing is demanded. Before the fix it always pushed at +tau_max.
    platform = load_platform()
    joint = platform.joint("base")
    n = 3

    class Constant:
        def torques(self, err, qd):
            return np.array([-5.0, 0.0, 5.0]), np.zeros(n)

    plant = make_plant(joint, Fault(locked=True), n)
    rollout = simulate(plant, np.full(n, 30.0), Constant(), lambda t: 0.0, 4, platform.heat_coeff)
    assert np.array_equal(rollout.tau[-1], [-joint.tau_max_nm, 0.0, joint.tau_max_nm])
    assert np.array_equal(rollout.q[-1], np.zeros(n))   # it still never moves


def test_the_controller_only_sees_error_and_speed():
    seen = []

    class Spy:
        def torques(self, err, qd):
            seen.append((err.shape, qd.shape))
            return err * 0.0, qd * 0.0

    platform = load_platform()
    joint = platform.joint("base")
    skill = skill_for(platform, "T1")
    plant = make_plant(joint, Fault(), 2)
    simulate(plant, np.full(2, 30.0), Spy(), skill.reference, 5, platform.heat_coeff)
    assert seen and all(e == (2,) and q == (2,) for e, q in seen)
