"""The scripted reference probes keep their known numbers and fault signatures.

The expected level map (see docs/design-decisions.md):
  healthy -> 3 on every joint, locked -> -1, missing_sensor -> -1,
  weak_motor -> 2 on the shoulder only, broken_link -> 1 on every joint.
"""
import math

import numpy as np

from src.damage import load_damage
from src.pipeline import build_model


def test_healthy_arm_reaches_top_tier_on_every_joint(arm):
    _, _, damage, table = arm
    assert all(table.levels[j]["healthy"] == 3 for j in table.levels)


def test_model_and_reference_agree_for_the_scripted_table(arm):
    _, _, damage, table = arm
    assert all(table.levels[j][h] == table.reference_levels[j][h]
               for j in table.levels for h in damage.names)


def test_locked_joint_fails_T0(arm):
    _, _, _, table = arm
    assert table.levels["shoulder"]["locked"] == -1
    assert table.get("shoulder", "T0", "locked").pass_prob == 0.0


def test_missing_sensor_is_never_moved(arm):
    _, _, _, table = arm
    pred = table.get("elbow", "T0", "missing_sensor")
    assert pred.metrics["sensor_valid"] == 0.0
    assert pred.energy_j == 0.0          # no motion, so no energy
    assert pred.p_any_violation == 0.0
    assert table.levels["elbow"]["missing_sensor"] == -1


def test_weak_shoulder_overloads_while_holding_its_load(arm):
    # 40% torque cannot hold the shoulder's 25 N*m load. The controller asks for more than the
    # hold's budget allows, so the run aborts on overload, which counts as a fail.
    _, _, _, table = arm
    pred = table.get("shoulder", "T3", "weak_motor")
    assert pred.violation_prob["overload"] == 1.0
    assert pred.pass_prob < 0.1
    assert table.levels["shoulder"]["weak_motor"] == 2


def test_weak_motor_on_unloaded_base_passes_everything(arm):
    _, _, _, table = arm
    assert table.levels["base"]["weak_motor"] == 3


def test_broken_link_fails_path_tracking_on_every_joint(arm):
    _, _, _, table = arm
    assert all(table.levels[j]["broken_link"] == 1 for j in table.levels)
    assert all(table.get(j, "T2", "broken_link").pass_prob == 0.0 for j in table.levels)


def test_hold_sag_is_gravity_over_stiffness(arm):
    # Healthy shoulder: sag = gravity / kp = 25 / 1500 rad, about 0.955 degrees.
    _, _, _, table = arm
    sag = table.get("shoulder", "T3", "healthy").metrics["sag_deg"]
    assert math.isclose(sag, math.degrees(25.0 / 1500.0), rel_tol=0.05)


def test_the_scripted_table_is_deterministic(arm, ):
    platform, tiers, damage, table = arm
    again = build_model(platform, tiers, damage, force_scripted=True)
    a = table.get("wrist_2", "T2", "broken_link")
    b = again.get("wrist_2", "T2", "broken_link")
    assert a.metrics == b.metrics and a.energy_j == b.energy_j


def test_moves_spend_energy(arm):
    _, _, _, table = arm
    assert table.get("shoulder", "T1", "healthy").energy_j > 0.0


def test_damage_model_prior_sums_to_one():
    damage = load_damage()
    assert math.isclose(sum(damage.prior.values()), 1.0)
    assert np.isclose(sum(damage.prior.values()), 1.0)
