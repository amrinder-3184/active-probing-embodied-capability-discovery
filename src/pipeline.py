"""Wire the assessors together, and build the table the planner and the episodes use.

Which assessor a tier uses is set per platform in its `assessment:` section. A tier that is
`learned` needs a policy store. A missing policy stops the run with a message, rather than
silently falling back to the scripted controller.
"""
from .assessors import LearnedAssessor, ScriptedAssessor
from .outcomes import build_table
from .platform_adapter import usable_tiers


def build_assessors(platform, tiers, store=None, settings=None, force_scripted=False):
    """Map each usable tier name to its assessor. Returns (assessors, scripted)."""
    scripted = ScriptedAssessor(platform)
    learned = None
    if store is not None and settings is not None:
        learned = LearnedAssessor(platform, store, settings, base_seed=settings.seeds["table"])

    assessors = {}
    for tier in usable_tiers(platform, tiers):
        kind = "scripted" if force_scripted else platform.assessment.get(tier.name, "scripted")
        if kind == "scripted":
            assessors[tier.name] = scripted
        elif kind == "learned":
            if learned is None:
                raise ValueError(f"{tier.name} is learned, but no policy store was given. "
                                 "Run `python -m experiments.train_policies` first, "
                                 "or use the scripted assessment.")
            assessors[tier.name] = learned
        else:
            raise ValueError(f"unknown assessor kind for {tier.name}: {kind}")
    return assessors, scripted


def build_model(platform, tiers, damage, store=None, settings=None, force_scripted=False):
    assessors, scripted = build_assessors(platform, tiers, store, settings, force_scripted)
    return build_table(platform, tiers, damage, assessors, scripted)
