"""Assess every joint with the trained policies, and check the result against the scripted reference.

    python -m experiments.assess_policies
    python -m experiments.assess_policies --policies <dir> --out <file.md>

Prints and writes results/assessment_<platform>.md (or --out), reading the trained policies from
results/policies/<platform>/ (or --policies). Three parts:
  1. Identifiability with the scripted reference, run first. Which damage states no tier separates.
  2. Performance signature with the learned policies: pass probability of each tier under each damage
     state, and the mean margin to the limit (positive is inside it, so near zero means close to failing).
  3. Agreement: the model level from the learned assessors against the reference level from the
     scripted controller, for every joint and damage state.
"""
import argparse
import os

from src.config import REPO_ROOT
from src.damage import load_damage
from src.pipeline import build_model
from src.platform_adapter import load_platform, usable_tiers
from src.policies import PolicyStore
from src.tiers import load_tiers
from src.training import load_training_config


def unseparated_pairs(table, platform, tiers, damage, gap):
    """Per joint, the damage-state pairs that no tier separates by at least `gap` in pass probability."""
    usable = usable_tiers(platform, tiers)
    result = {}
    names = damage.names
    for joint in platform.joints:
        pairs = []
        for i, h1 in enumerate(names):
            for h2 in names[i + 1:]:
                separated = any(
                    abs(table.get(joint.name, t.name, h1).pass_prob - table.get(joint.name, t.name, h2).pass_prob) >= gap
                    for t in usable
                )
                if not separated:
                    pairs.append((h1, h2))
        result[joint.name] = pairs
    return result


def signature_rows(table, platform, tiers, damage):
    usable = usable_tiers(platform, tiers)
    header = ["joint", "damage"] + [f"{t.name} pass" for t in usable] + [f"{t.name} margin" for t in usable[1:]]
    rows = []
    for joint in platform.joints:
        for h in damage.names:
            preds = [table.get(joint.name, t.name, h) for t in usable]
            cells = [f"{p.pass_prob:.2f}" for p in preds]
            cells += [f"{p.margin:+.2f}" if p.margin != float("-inf") else "n/a" for p in preds[1:]]
            rows.append([joint.name, h] + cells)
    return header, rows


def agreement(table, platform, damage):
    rows = []
    matches = total = 0
    for joint in platform.joints:
        for h in damage.names:
            model = table.levels[joint.name][h]
            ref = table.reference_levels[joint.name][h]
            ok = model == ref
            matches += ok
            total += 1
            rows.append([joint.name, h, str(ref), str(model), "yes" if ok else "NO"])
    return rows, matches, total


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def parse_args(argv, platform_name):
    """Paths default to the stage-1 locations under results/, so no flags means the old behaviour."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policies", default=os.path.join(REPO_ROOT, "results", "policies", platform_name),
                        help="folder of trained policy JSON files")
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "results", f"assessment_{platform_name}.md"),
                        help="Markdown file to write")
    return parser.parse_args(argv)


def main(argv=None):
    platform = load_platform()
    args = parse_args(argv, platform.name)
    tiers = load_tiers()
    damage = load_damage()
    settings = load_training_config()
    store = PolicyStore(args.policies)
    gap = settings.identifiability_gap

    scripted = build_model(platform, tiers, damage, force_scripted=True)
    learned = build_model(platform, tiers, damage, store=store, settings=settings)

    out = [f"# Assessment: {platform.name}", ""]

    out += ["## 1. Identifiability with the scripted reference", "",
            f"Damage states that no tier separates by at least {gap} in pass probability, per joint. "
            "Empty means every pair is separable.", ""]
    scripted_pairs = unseparated_pairs(scripted, platform, tiers, damage, gap)
    out += [md_table(["joint", "not separable"],
                     [[j, ", ".join(f"{a} / {b}" for a, b in pairs) or "none"]
                      for j, pairs in scripted_pairs.items()]), ""]

    header, rows = signature_rows(learned, platform, tiers, damage)
    out += ["## 2. Performance signature with the learned policies", "",
            "Pass probability per tier under each damage state, and the mean margin to the limit "
            "(positive: inside it; near zero: close to failing).", "", md_table(header, rows), ""]

    learned_pairs = unseparated_pairs(learned, platform, tiers, damage, gap)
    out += ["Not separable with the learned policies:", "",
            md_table(["joint", "not separable"],
                     [[j, ", ".join(f"{a} / {b}" for a, b in pairs) or "none"]
                      for j, pairs in learned_pairs.items()]), ""]

    rows, matches, total = agreement(learned, platform, damage)
    out += ["## 3. Agreement: learned model level vs scripted reference level", "",
            f"Matches: {matches} of {total} (joint, damage state) pairs.", "",
            md_table(["joint", "damage", "reference level", "model level", "agree"], rows), ""]

    text = "\n".join(out)
    path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print(text)
    print(f"Wrote {os.path.relpath(path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
