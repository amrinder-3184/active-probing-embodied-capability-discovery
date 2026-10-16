"""Train one learned skill per (joint, tier), gate each one on healthy joints, and save it.

    python -m experiments.train_policies --workers 4

Each (joint, tier) is an independent job, so they run in parallel. Output: one JSON file per job in
results/policies/<platform>/. Re-running overwrites them. The seeds are fixed in configs/training.yaml.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

from src.config import REPO_ROOT
from src.platform_adapter import load_platform, usable_tiers
from src.policies import PolicyStore
from src.tasks import skill_for
from src.tiers import load_tiers
from src.training import load_training_config, train_skill


def _job(args):
    joint_name, tier_name, index, seed_base = args
    platform = load_platform()
    tiers = {t.name: t for t in load_tiers()}
    settings = load_training_config()
    seed = seed_base + index
    record = train_skill(platform.joint(joint_name), tiers[tier_name], skill_for(platform, tier_name),
                         platform, settings, seed)
    return record


def learned_jobs(platform, tiers, settings):
    jobs = []
    index = 0
    for joint in platform.joints:
        for tier in usable_tiers(platform, tiers):
            if platform.assessment.get(tier.name) == "learned":
                jobs.append((joint.name, tier.name, index, settings.seeds["training"]))
            index += 1
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    platform = load_platform()
    tiers = load_tiers()
    settings = load_training_config()
    store = PolicyStore(args.out or os.path.join(REPO_ROOT, "results", "policies", platform.name))
    jobs = learned_jobs(platform, tiers, settings)

    print(f"Training {len(jobs)} learned skills with {args.workers} worker(s)")
    print(f"{'joint':9s} {'tier':4s} {'kp_mult':>8s} {'kd_mult':>8s} {'healthy':>10s} {'lower':>7s} trusted")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(_job, jobs):
            store.put(record)
            g = record.gate
            print(f"{record.joint:9s} {record.tier:4s} {record.kp_multiplier:8.3f} {record.kd_multiplier:8.3f} "
                  f"{g['successes']:4d}/{g['episodes']:<4d}  {g['lower_bound']:6.3f} {record.trusted}")
    print(f"\nSaved to {os.path.relpath(store.root, REPO_ROOT)}")


if __name__ == "__main__":
    main()
