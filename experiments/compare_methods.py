"""Compare the probing methods over many simulated damage cases.

    python -m experiments.compare_methods --seeds 200 --budgets 6 9 12 18 24 40
    python -m experiments.compare_methods --risk-weights 0 0.25 1 4        # RQ2 sweep
    python -m experiments.compare_methods --gating robot                   # RQ3 baseline
    python -m experiments.compare_methods --trials-out t.csv --paired-out p.csv --ceiling
    python -m experiments.compare_methods --policies <dir> --seed-start 100 --seeds 200

Each row is one method at one budget, averaged over the seeds. Writes results/comparison.csv (or --out).
The same seeds are used for every method, so each method faces the same damage.

Seeds run from --seed-start (default 0) to --seed-start + --seeds - 1. Every method, budget and the
ceiling use these same seed values, so their damage and noise are matched seed by seed.

The learned assessment reads trained policies from --policies (default results/policies/<platform>).
The folder must hold a policy for every learned (joint, tier) before any episode runs; there is no
fallback to another folder or to the scripted controller.

The passive method runs no probes, so its belief stays at the prior: its row is the prior-only
forecast, the floor for every belief score.

Optional outputs (all off by default):
  --trials-out   one row per (method, budget, seed), every metric and the true damage states
  --paired-out   per-seed differences against the fixed battery, with a percentile bootstrap
                 interval over seeds (docs/assessment.md, section 8)
  --ceiling      coverage of the fixed battery with a budget that can never bind
                 (joints x tiers x max_repeats). A reference point, not a proven upper bound.
  --curve-out    every metric after each probe, k = 0 .. the largest budget, one row per
                 (method, seed, k). Computed by replaying the episodes run at the largest budget:
                 no choice depends on the budget, so the state after k probes is the result of a
                 run with budget k. Adds time and safety-violation counts (src/evaluate.py).
"""
import argparse
import csv
import dataclasses
import os
import statistics

from src.config import REPO_ROOT
from src.damage import load_damage
from src.episode import NOISE_STREAMS, prefix_episodes, run_episode
from src.evaluate import paired_bootstrap, score_episode
from src.pipeline import build_model
from src.planner import load_planner_config
from src.platform_adapter import load_platform, usable_tiers
from src.policies import PolicyStore
from src.report import build_report
from src.tiers import load_tiers
from src.training import load_training_config

# The first six are the original columns, in their original order. New columns are appended.
METRICS = ("coverage", "accuracy", "overclaim", "probes", "aborts", "energy_j",
           "hypothesis_accuracy", "hypothesis_expected_accuracy", "hypothesis_tie_rate",
           "hypothesis_brier", "level_brier", "level_log_loss")
CURVE_METRICS = METRICS + ("time_s", "violations_overload", "violations_joint_limit")  # --curve-out only
PAIRED_METRICS = ("coverage", "accuracy", "overclaim", "hypothesis_expected_accuracy",
                  "level_brier", "level_log_loss", "probes", "energy_j")
REFERENCE_METHOD = "fixed"


def _write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def seed_range(seed_start, count):
    """The seed values every method runs on, in order."""
    return range(seed_start, seed_start + count)


def missing_policies(store, platform, tiers):
    """(joint, tier) pairs the learned assessment needs but the store does not have."""
    return [(joint.name, tier.name)
            for joint in platform.joints
            for tier in usable_tiers(platform, tiers)
            if platform.assessment.get(tier.name) == "learned" and not store.has(joint.name, tier.name)]


def parse_args(argv, platform_name):
    """No new flags means the old behaviour: seeds 0 .. --seeds - 1, policies under results/."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=200, help="number of seeds")
    parser.add_argument("--seed-start", type=int, default=0,
                        help="first seed; seeds are seed-start .. seed-start + seeds - 1")
    parser.add_argument("--budgets", type=int, nargs="+", default=[6, 9, 12, 18, 24, 40])
    parser.add_argument("--risk-weights", type=float, nargs="+", default=[1.0],
                        help="planner risk weights to run (RQ2)")
    parser.add_argument("--gating", choices=["chain", "robot"], default=None,
                        help="override configs/planner.yaml (RQ3)")
    parser.add_argument("--assessment", choices=["learned", "scripted"], default="learned")
    parser.add_argument("--policies", default=None,
                        help="folder of trained policy JSON files for the learned assessment "
                             f"(default: results/policies/{platform_name})")
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "comparison.csv"))
    parser.add_argument("--trials-out", default=None, help="per-trial CSV (one row per method, budget, seed)")
    parser.add_argument("--paired-out", default=None, help="paired differences against the fixed battery")
    parser.add_argument("--curve-out", default=None,
                        help="metrics after every probe, k = 0 .. the largest budget (one row per method, seed, k)")
    parser.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples for --paired-out")
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--ceiling", action="store_true", help="add the unbounded-budget coverage reference")
    parser.add_argument("--noise-streams", choices=NOISE_STREAMS, default="per_probe",
                        help="per_probe: each probe run has its own seeded stream (paired). "
                             "shared: the stage-1 single stream, to reproduce earlier results.")
    args = parser.parse_args(argv)

    if args.seeds < 1:
        parser.error(f"--seeds must be at least 1, not {args.seeds}")
    if args.seed_start < 0:
        parser.error(f"--seed-start must be 0 or more (numpy seeds cannot be negative), not {args.seed_start}")
    if args.bootstrap < 1:
        parser.error(f"--bootstrap must be at least 1, not {args.bootstrap}")
    if args.policies is not None and args.assessment == "scripted":
        parser.error("--policies is only used by the learned assessment; drop it or use --assessment learned")
    if args.policies is None:
        args.policies = os.path.join(REPO_ROOT, "results", "policies", platform_name)
    if args.assessment == "learned" and not os.path.isdir(args.policies):
        parser.error(f"--policies folder does not exist: {args.policies}. "
                     "Train with `python -m experiments.train_policies --out <dir>` or pass the folder that holds them.")
    return args


def main(argv=None):
    platform = load_platform()
    args = parse_args(argv, platform.name)
    tiers = load_tiers()
    damage = load_damage()
    settings = load_training_config()
    base_cfg = load_planner_config()
    if args.gating:
        base_cfg = dataclasses.replace(base_cfg, gating=args.gating)
    store = PolicyStore(args.policies)
    if args.assessment == "learned":
        # Checked before any simulation, so a wrong folder fails at once and names what is missing.
        missing = missing_policies(store, platform, tiers)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} trained policies missing from {args.policies}: "
                + ", ".join(f"{j}_{t}.json" for j, t in missing)
                + ". Run `python -m experiments.train_policies --out <dir>` first.")
        print(f"Policies: {args.policies}")
    table = build_model(platform, tiers, damage, store=store, settings=settings,
                        force_scripted=args.assessment == "scripted")

    methods = [(f"planner(risk={w:g})", "planner", dataclasses.replace(base_cfg, risk_weight=w))
               for w in args.risk_weights]
    methods += [(m, m, base_cfg) for m in ("random", "fixed", "passive")]
    seeds = seed_range(args.seed_start, args.seeds)

    ceiling = None
    if args.ceiling:
        # Every (joint, tier) can be probed at most max_repeats times, so this budget never binds.
        unbounded = len(platform.joints) * len(usable_tiers(platform, tiers)) * base_cfg.max_repeats
        cfg = dataclasses.replace(base_cfg, budget=unbounded)
        ceiling = []
        for seed in seeds:
            episode = run_episode(platform, tiers, damage, table, REFERENCE_METHOD, cfg, seed,
                                  noise_streams=args.noise_streams)
            ceiling.append(score_episode(build_report(platform, episode), episode)["coverage"])

    rows = []
    trials = []
    curve = []
    curve_budget = max(args.budgets)
    curved = set()  # method labels whose curve is already recorded (a repeated budget is not replayed twice)
    by_key = {}  # (method label, budget) -> per-seed score dicts, in seed order
    for budget in args.budgets:
        for label, method, cfg in methods:
            cfg = dataclasses.replace(cfg, budget=budget)
            scores = []
            record_curve = args.curve_out and budget == curve_budget and label not in curved
            for seed in seeds:
                episode = run_episode(platform, tiers, damage, table, method, cfg, seed,
                                      noise_streams=args.noise_streams)
                score = score_episode(build_report(platform, episode), episode)
                scores.append(score)
                if args.trials_out:
                    trials.append({"method": label, "gating": cfg.gating, "budget": budget, "seed": seed,
                                   **{m: score[m] for m in METRICS},
                                   "true_states": ";".join(f"{j}={h}" for j, h in episode.true_states.items())})
                if record_curve:
                    for prefix in prefix_episodes(episode, table, damage, cfg):
                        s = score_episode(build_report(platform, prefix), prefix)
                        curve.append({"method": label, "gating": cfg.gating, "budget": budget, "seed": seed,
                                      "k": prefix.budget, **{m: s[m] for m in CURVE_METRICS}})
            if record_curve:
                curved.add(label)
            by_key[(label, budget)] = scores
            row = {"method": label, "gating": cfg.gating, "budget": budget}
            for metric in METRICS:
                row[metric] = statistics.fmean(s[metric] for s in scores)
            if ceiling is not None:
                row["coverage_ceiling"] = statistics.fmean(ceiling)
            rows.append(row)

    print(f"{'method':22s} {'gating':6s} {'budget':>6s} {'coverage':>9s} {'accuracy':>9s} "
          f"{'overclaim':>9s} {'probes':>7s} {'aborts':>7s} {'energy_j':>9s} "
          f"{'hyp_exp':>8s} {'lvl_brier':>9s} {'lvl_logl':>9s}")
    for r in rows:
        print(f"{r['method']:22s} {r['gating']:6s} {r['budget']:6d} {r['coverage']:9.3f} "
              f"{r['accuracy']:9.3f} {r['overclaim']:9.3f} {r['probes']:7.2f} "
              f"{r['aborts']:7.3f} {r['energy_j']:9.2f} "
              f"{r['hypothesis_expected_accuracy']:8.3f} {r['level_brier']:9.4f} {r['level_log_loss']:9.4f}")
    print(f"passive runs no probes: its row is the prior-only forecast. Noise streams: {args.noise_streams}.")
    if ceiling is not None:
        print(f"coverage ceiling (fixed battery, budget never binds): {statistics.fmean(ceiling):.3f}")

    out_path = args.out
    fields = ["method", "gating", "budget", *METRICS] + (["coverage_ceiling"] if ceiling is not None else [])
    _write_csv(out_path, fields, rows)
    print(f"\nWrote {os.path.relpath(out_path, REPO_ROOT)} ({args.seeds} seeds per row: "
          f"{seeds.start} to {seeds.stop - 1})")

    if args.trials_out:
        _write_csv(args.trials_out, ["method", "gating", "budget", "seed", *METRICS, "true_states"], trials)
        print(f"Wrote {os.path.relpath(args.trials_out, REPO_ROOT)} ({len(trials)} trials)")

    if args.curve_out:
        _write_csv(args.curve_out, ["method", "gating", "budget", "seed", "k", *CURVE_METRICS], curve)
        print(f"Wrote {os.path.relpath(args.curve_out, REPO_ROOT)} ({len(curve)} rows: k = 0 to {curve_budget})")

    if args.paired_out:
        paired = []
        for budget in args.budgets:
            reference = by_key[(REFERENCE_METHOD, budget)]
            for label, _, _ in methods:
                if label == REFERENCE_METHOD:
                    continue
                scores = by_key[(label, budget)]
                for metric in PAIRED_METRICS:
                    mean, low, high = paired_bootstrap(
                        [s[metric] for s in scores], [s[metric] for s in reference],
                        resamples=args.bootstrap, seed=args.bootstrap_seed)
                    paired.append({"method": label, "reference": REFERENCE_METHOD, "budget": budget,
                                   "metric": metric, "mean_diff": mean, "ci_low": low, "ci_high": high,
                                   "seeds": args.seeds, "resamples": args.bootstrap})
        _write_csv(args.paired_out, ["method", "reference", "budget", "metric", "mean_diff",
                                     "ci_low", "ci_high", "seeds", "resamples"], paired)
        print(f"Wrote {os.path.relpath(args.paired_out, REPO_ROOT)} "
              f"(differences against {REFERENCE_METHOD}, 95% percentile bootstrap over seeds)")


if __name__ == "__main__":
    main()
