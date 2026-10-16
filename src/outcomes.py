"""Table of predictions for the planner: each joint, tier and damage hypothesis.

Three things are kept apart (see docs/stage1-plan.md, section 3.7):
  model level      the level the assessors predict. The belief uses it.
  reference level  the level the scripted PD controller reaches, noise-free. Evaluation uses it.
  damage truth     the true hypothesis, which only the simulator knows.
"""
from dataclasses import dataclass

from .platform_adapter import usable_tiers


@dataclass
class Table:
    predictions: dict       # (joint, tier, hypothesis) -> Prediction
    levels: dict            # joint -> {hypothesis: model level}, -1 = none
    reference_levels: dict  # joint -> {hypothesis: reference level}, -1 = none
    trusted: dict           # (joint, tier) -> bool
    assessors: dict         # tier name -> assessor, also used for the robot's own runs

    def get(self, joint, tier, hypothesis):
        return self.predictions[(joint, tier, hypothesis)]


def _contiguous_level(passes):
    """Highest tier index reached with no gap below it. Returns -1 if the first tier fails."""
    level = -1
    for index, passed in enumerate(passes):
        if not passed:
            break
        level = index
    return level


def build_table(platform, tiers, damage, assessors, scripted):
    usable = usable_tiers(platform, tiers)
    predictions = {}
    levels = {}
    reference = {}
    trusted = {}

    for j_index, joint in enumerate(platform.joints):
        levels[joint.name] = {}
        reference[joint.name] = {}
        for h_index, h in enumerate(damage.names):
            fault = damage.faults[h]
            model_passes = []
            reference_passes = []
            for t_index, tier in enumerate(usable):
                # The reference is the scripted controller, judged on its noise-free readings.
                ref = scripted.predict(joint, tier, fault)
                assessor = assessors[tier.name]
                if assessor is scripted:
                    pred = ref  # the same computation, so do not run it twice
                else:
                    pred = assessor.predict(joint, tier, fault, seed_key=(j_index, t_index, h_index))
                predictions[(joint.name, tier.name, h)] = pred
                trusted[(joint.name, tier.name)] = pred.trusted
                model_passes.append(pred.pass_prob >= 0.5)
                reference_passes.append(
                    ref.p_any_violation == 0.0 and tier.passes(ref.metrics))
            levels[joint.name][h] = _contiguous_level(model_passes)
            reference[joint.name][h] = _contiguous_level(reference_passes)

    return Table(predictions=predictions, levels=levels, reference_levels=reference,
                 trusted=trusted, assessors=assessors)
