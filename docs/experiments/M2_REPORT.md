# M2: reproducible reruns with the M1 measurements

Active Probing for Embodied Capability Discovery. Generated 2026-10-29. This report covers both pipelines; the scripted outputs are in `results/m2_scripted_2026-10-29/`.

## 1. Repository and environment

| Item | Value |
|---|---|
| Branch / commit | `main` at `1fd91b8` ("Implement active probing evaluation and measurement pipeline"); working tree clean before, between and after the runs |
| Tests at this commit | `pytest -q`: 125 passed (135.7 s), run before the experiments |
| Environment | Windows 10, CPython 3.11.9, numpy 2.4.6, PyYAML 6.0.3 (`requirements.txt`; `docs/testing.md` section 6) |
| Learned policies | `results/rerun_2026-10-15/policies/arm6/`, 18 files, all trusted (100/100 gate). Combined SHA-256 of the files sorted by name begins `26bc2c8044b3f90f`. Not retrained. |

## 2. Commands executed

Both runs were sequential (learned, validated, then scripted). Each was preceded by `New-Item -ItemType Directory <dir> -ErrorAction Stop`, which fails if the folder exists, and followed by appending the exit code to `run.log`.

```powershell
# Learned
.\.venv\Scripts\python.exe -m experiments.compare_methods `
  --assessment learned --policies results\rerun_2026-10-15\policies\arm6 `
  --seed-start 100 --seeds 200 --budgets 4 8 12 16 20 24 --risk-weights 1 `
  --gating chain --noise-streams per_probe --bootstrap 2000 --ceiling `
  --out        results\m2_learned_2026-10-29\comparison_learned.csv `
  --trials-out results\m2_learned_2026-10-29\trials_learned.csv `
  --paired-out results\m2_learned_2026-10-29\paired_learned.csv `
  --curve-out  results\m2_learned_2026-10-29\curve_learned.csv `
  2>&1 | Tee-Object results\m2_learned_2026-10-29\run.log
Add-Content results\m2_learned_2026-10-29\run.log "EXIT_CODE: $LASTEXITCODE" -Encoding utf8

# Scripted: identical except --assessment scripted, no --policies, *_scripted file names
#           and the results\m2_scripted_2026-10-29\ folder
```

## 3. Configuration

Seeds 100-299 (200); budgets 4, 8, 12, 16, 20, 24; methods planner (risk weight 1), random, fixed, passive; chain gating; per-probe noise streams; 2000 bootstrap resamples (generator seed 0); coverage ceiling on. These equal the settings of `results/full_2026-10-16/` and `results/full_scripted_2026-10-16/` (confirmed by the identical outputs in section 6).

## 4. Outputs and exit codes

| Pipeline | Folder | Exit code | Wall time | Files (bytes) |
|---|---|---|---|---|
| Learned | `results/m2_learned_2026-10-29/` | **0** | 28 min 52 s (12:07:52-12:36:44) | `comparison_learned.csv` (4,989), `trials_learned.csv` (1,183,279), `paired_learned.csv` (11,449), `curve_learned.csv` (3,196,385), `run.log` (3,656) |
| Scripted | `results/m2_scripted_2026-10-29/` | **0** | 26 min 31 s (12:37:31-13:04:02) | `comparison_scripted.csv` (4,695), `trials_scripted.csv` (1,178,712), `paired_scripted.csv` (11,585), `curve_scripted.csv` (3,179,400), `run.log` (3,614) |

Neither `run.log` contains a traceback, error or warning.

## 5. Row counts

Expected counts were derived from the code (`experiments/compare_methods.py`: `METRICS`, `PAIRED_METRICS`, `CURVE_METRICS`) before running.

| File | Expected rows x columns | Learned | Scripted |
|---|---|---|---|
| comparison | 24 x 16 (4 methods x 6 budgets) | 24 x 16 | 24 x 16 |
| trials | 4,800 x 17 (x 200 seeds) | 4,800 x 17 | 4,800 x 17 |
| paired | 144 x 9 (6 budgets x 3 methods x 8 metrics) | 144 x 9 | 144 x 9 |
| curve | 20,000 x 20 (4 methods x 200 seeds x k = 0..24) | 20,000 x 20 | 20,000 x 20 |

Curve columns: `method, gating, budget, seed, k`, the 12 existing metrics, and the 3 M1 metrics `time_s, violations_overload, violations_joint_limit`.

## 6. Validation

All checks passed for both pipelines.

- **Completeness:** every (method, budget) cell has seeds 100-299 exactly once; no duplicate keys in trials or curve; no empty or non-finite metric values; gating `chain` in every row.
- **Curve correctness:** k = 0..24 for each of the 800 (method, seed) pairs; all rows replayed from the budget-24 episodes. At k = budget, the curve equals the trials rows on all 12 metrics for all 4,800 rows. At k = 0 every method equals the prior-only (passive) row. Probes equal min(k, episode length); probes, time, energy, aborts and violations never decrease along k; aborts equal overload + joint-limit events in every row; passive has 0 probes and 0 s at every k. The prefix-equivalence tests in `tests/test_measurements.py` (part of the 125) cover the replay this relies on.
- **Definitions:** the 12 shared metrics were confirmed unchanged against the pre-M1 code (stash `stash@{0}` vs `1fd91b8`): `METRICS` identical; `src/evaluate.py` and `src/report.py` only gained lines; belief, gating, planner, physics, tasks, runner, outcomes, damage, tiers, policies, pipeline and configs are byte-identical.

**Reproducibility against the stored runs:**

| File | Learned vs `full_2026-10-16` | Scripted vs `full_scripted_2026-10-16` |
|---|---|---|
| trials | 4,800/4,800 rows identical (all 17 columns); **byte-identical** | 4,800/4,800; **byte-identical** |
| comparison | 24/24 rows identical; **byte-identical** | 24/24; **byte-identical** |
| paired | 144/144 rows identical; **byte-identical** | 144/144; **byte-identical** |
| curve | No stored counterpart (new in M1). Its rows at k = 4..24 equal the stored trials on all 12 metrics. | Same |
| time, overload, joint-limit events | Not recorded by the stored runs; cannot be compared | Same |

No discrepancies were found. Because the runs reuse seeds 100-299, they reproduce the October 16 results exactly; **they are not an independent confirmation on new seeds.**

**Preservation:** all 40 files in `full_2026-10-16`, `full_scripted_2026-10-16`, `pilot_2026-10-16` and `rerun_2026-10-15` have the same SHA-256 after both runs as before. `git status --short` was empty after each run; the new folders are git-ignored. No source, test, config or dependency file was modified, and no Git operation was performed.

## 7. Results by probe count (means over 200 seeds)

Coverage (determined levels) and level Brier score (lower is better) after k probes. Passive (prior only) Brier: 0.1083 learned, 0.1074 scripted, at every k.

| k | Learned: planner / fixed / random coverage | Learned: planner / fixed / random Brier | Scripted: planner / fixed / random coverage | Scripted: planner / fixed / random Brier |
|---|---|---|---|---|
| 0 | 0 / 0 / 0 | 0.1083 / 0.1083 / 0.1083 | 0 / 0 / 0 | 0.1074 / 0.1074 / 0.1074 |
| 4 | 0.045 / 0.045 / 0.046 | 0.0587 / 0.0587 / 0.0751 | 0.045 / 0.045 / 0.046 | 0.0579 / 0.0579 / 0.0732 |
| 8 | 0.095 / 0.095 / 0.148 | 0.0408 / 0.0407 / 0.0590 | 0.095 / 0.095 / 0.148 | 0.0367 / 0.0405 / 0.0550 |
| 12 | 0.145 / 0.145 / 0.273 | 0.0372 / 0.0372 / 0.0455 | 0.141 / 0.132 / 0.264 | 0.0330 / 0.0392 / 0.0414 |
| 16 | 0.303 / 0.222 / 0.408 | 0.0366 / 0.0366 / 0.0363 | 0.201 / 0.198 / 0.384 | 0.0277 / 0.0297 / 0.0303 |
| 20 | 0.413 / 0.413 / 0.516 | 0.0358 / 0.0358 / 0.0357 | 0.399 / 0.399 / 0.495 | 0.0249 / 0.0249 / 0.0266 |
| 24 | 0.704 / 0.704 / 0.704 | 0.0358 / 0.0358 / 0.0358 | 0.666 / 0.666 / 0.666 | 0.0249 / 0.0249 / 0.0249 |

The per-probe curve (`curve_*.csv`) holds every k from 0 to 24.

**Probing time** (simulated execution time of the executed skills, in seconds; descriptive, not tested): the scripted planner spends more time than fixed while its belief advantage lasts (k = 8: 8.3 vs 6.9; k = 10: 13.0 vs 10.6; k = 12: 17.3 vs 14.0; k = 14: 21.5 vs 19.5; k = 16: 24.9 vs 24.0), because its earlier, more informative probes are the longer path and hold skills. In the learned pipeline, planner and fixed time are within 0.5 s at every k. At k = 24 every probing method has run the same probes: 36.6 s (learned) and 35.9 s (scripted).

**Safety events** (counted from the physics' recorded events): only overloads occurred; there were **0 joint-limit events** in either pipeline. At k = 24 every probing method has 6 overloads (learned, 6 episodes) and 7 (scripted, 7 episodes), all from the 10 seeds with a weak shoulder. Random reaches them earlier: at k = 12, random has 2 (learned) and 3 (scripted) overloads while planner and fixed have 0; at k = 16, random 5 and 6 vs 1 for planner and fixed. These counts are too small for a statistical test.

## 8. RQ1: correct capability level with fewer probes

The statistical method is the one fixed before the October 16 experiments: paired differences over the 200 seeds, 95% percentile bootstrap (2000 resamples), a sign-flip randomisation test (20,000 flips; smallest reportable p = 5 x 10^-5), and Holm correction.

**Primary endpoint (pre-specified): normalised area under the level-Brier curve over budgets 4-24, Holm over 2 comparisons.**

| Pipeline | Comparison | Mean difference | 95% CI | 97.5% CI | Seeds lower / higher | Holm p |
|---|---|---|---|---|---|---|
| Learned | planner - fixed | +0.00002 | [0.00000, +0.00005] | [0.00000, +0.00006] | 0 / 99 | 0.0001 |
| Learned | planner - random | -0.00689 | [-0.00880, -0.00513] | [-0.00914, -0.00485] | 146 / 19 | 0.0001 |
| Scripted | planner - fixed | **-0.00239** | [-0.00334, -0.00151] | [-0.00348, -0.00143] | 139 / 5 | 0.0001 |
| Scripted | planner - random | **-0.00771** | [-0.00958, -0.00595] | [-0.00981, -0.00577] | 155 / 9 | 0.0001 |

Means (planner / fixed / random): learned 0.03951 / 0.03949 / 0.04640; scripted 0.03276 / 0.03515 / 0.04047.

- **Scripted:** the pre-specified hypothesis holds. The planner's area is lower than both baselines with Holm-adjusted intervals excluding 0. The effect against fixed is a 6.8% reduction; the upper end of its interval (-0.0014) lies inside the +/-0.002 margin set in advance, so it may be practically small. Against random it is a 19.0% reduction.
- **Learned:** no meaningful difference from fixed. The interval lies inside the +/-0.002 margin; the difference is detectable (the planner is slightly worse in 99 seeds) but negligible. The planner is better than random.

**Per budget** (primary family of 36: planner vs fixed and vs random, on coverage, accuracy and level Brier, at 6 budgets, Holm). Statistically supported differences:

- Scripted, planner - fixed: level Brier lower at budgets 8 (-0.0038 [-0.0060, -0.0016]), 12 (-0.0062 [-0.0091, -0.0036]) and 16 (-0.0020 [-0.0037, -0.0005]); coverage and accuracy +0.0092 [+0.0042, +0.0150] at budget 12. No difference at 4, 20 and 24 (identical in all 200 seeds).
- Learned, planner - fixed: only budget 16, where an energy-driven reordering raises coverage by +0.0808 [+0.0692, +0.0925] and accuracy by +0.0750, while level Brier is worse in 97 seeds by a negligible amount (mean +9.8 x 10^-7). Identical at budgets 4, 20 and 24.
- Both pipelines, planner - random: lower Brier at budgets 4, 8 and 12 (and 20 scripted), but **lower coverage and accuracy at budgets 8-20** (for example -0.128 learned and -0.123 scripted at budget 12).

**Secondary endpoint (pre-specified in the experiment proposal): probes until level Brier <= 0.05.**

| Pipeline | Comparison | Mean difference in probes | 95% CI | Seeds fewer / more probes | p |
|---|---|---|---|---|---|
| Learned | planner - fixed | 0.000 | [0.000, 0.000] | 0 / 0 | 1.0 |
| Learned | planner - random | -0.880 | [-1.185, -0.600] | 46 / 4 | 0.00005 |
| Scripted | planner - fixed | -0.400 | [-0.565, -0.255] | 27 / 0 | 0.00005 |
| Scripted | planner - random | -0.945 | [-1.245, -0.655] | 56 / 8 | 0.00005 |

Seeds that never reach the criterion within 24 probes count as 25. **This endpoint has a strong floor effect:** in 75 of 200 seeds the prior already meets it at k = 0, and 52 (learned) and 37 (scripted) seeds never reach it, so only 73 and 88 seeds can distinguish the methods. It supports the direction of the primary result but should be presented as weak supporting evidence.

**Conclusion for RQ1.** With informative probes (scripted pipeline), tier-gated active probing reaches a more accurate per-part capability estimate than the fixed battery and random probing, and reaches the Brier criterion with fewer probes than fixed (0.40 fewer on average, in 27 seeds fewer and none more). The advantage over fixed is modest, confined to budgets 8-16, and costs extra probing time (up to 3.3 s at budget 12). With the learned skills, whose probes carry little information, the planner behaves like the fixed battery. In both pipelines random probing completes more joints (coverage) at intermediate budgets. All results are simulation results on seeds 100-299.

## 9. RQ2 and RQ3

M2 does not answer them.

- **RQ2 (risk-weight trade-off):** only risk weight 1 was run. The new event counts show why a sweep here would be uninformative: 6-7 overload episodes in 200, none on joint limits, and no falls possible on the arm. A scenario where probe risk varies is needed first.
- **RQ3 (chain vs whole-robot gating; shared damage):** all runs use chain gating on a single chain; shared-damage inference is not implemented.

## 10. Limitations

- Same seeds as October 16: the reruns verify reproducibility and add measurements, not new statistical evidence.
- Simulation only; the planner's likelihood table and the robot's executions come from the same simulator (no model mismatch).
- `time_s` is the simulated duration of the executed skills; it excludes planning, resets and any real-robot overheads.
- "Wasted probes" is not measured; it needs an agreed definition.
- The secondary endpoint has the floor effect described in section 8.
- Results are local and git-ignored; they are traceable through commit `1fd91b8`, the policy fingerprint and this report, not through the repository.
- `results/pilot_2026-10-16/` was not used; it has no run log.

## 11. Confirmation

Both pipelines exited with code 0 and passed every validation check; the original result folders are unchanged (SHA-256 verified); no source code was modified; no commit, push, branch change or other Git operation was performed.
