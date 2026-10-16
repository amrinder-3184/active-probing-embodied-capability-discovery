# Assessment: results and what they mean

This document reports the stage-1 results. Every number is taken from `results/assessment_arm6.md` and `results/comparison_*.csv`, which the code regenerates. Commands to reproduce each result are given at the end of each section.

**Two sets of results.** Sections 2 to 7 are the stage-1 results as first documented: WSL2, unrecorded package versions, and one shared noise stream per episode. Reproduce them with `--noise-streams shared`. Section 8 is a rerun of 2026-10-15 on Windows, with pinned packages, per-probe noise streams, the new belief scores and paired intervals. Its numbers are new and are not edits of the earlier ones. Sections 2 to 4 were regenerated in that rerun and are unchanged.

## 1. What was measured

Three kinds of result:

1. **Performance signature.** For each joint and learned skill, the pass probability under each damage state, and the mean margin to the limit.
2. **Agreement.** For each joint and damage state, whether the model level (from the learned assessors) equals the reference level (from the scripted controller).
3. **Identifiability.** For each joint, which pairs of damage states no tier separates by at least 0.5 in pass probability.

Plus one benchmark: the four planning methods, compared over 100 seeds.

## 2. Performance signature (learned policies)

Pass probability is the Laplace-smoothed rate from 50 rollouts per cell. Healthy cells therefore read 0.98 rather than 1.00, because `(50+1)/(50+2) = 0.981`.

Margin is the mean signed distance to the tier's tightest check. For T1, T2 and T3's sag, that is in degrees. For T3 the torque check is a fraction of rated torque, so the margin is the smaller of the two. Positive is inside the limit. Negative values with a pass probability of 0.02 mean the skill fails almost always.

| Joint | Damage | T0 pass | T1 pass | T2 pass | T3 pass | T1 margin | T2 margin | T3 margin |
|---|---|---|---|---|---|---|---|---|
| base | healthy | 1.00 | 0.98 | 0.98 | 0.98 | +0.99 | +1.82 | +0.73 |
| base | locked | 0.00 | 0.02 | 0.02 | 0.02 | n/a | n/a | −0.20 |
| base | weak_motor | 1.00 | 0.98 | 0.98 | 0.98 | +0.99 | +1.83 | +0.73 |
| base | missing_sensor | 0.00 | 0.00 | 0.00 | 0.00 | n/a | n/a | n/a |
| base | broken_link | 1.00 | 0.79 | 0.73 | 0.98 | +0.09 | +0.09 | +0.75 |
| shoulder | healthy | 1.00 | 0.98 | 0.98 | 0.98 | +0.99 | +1.62 | +0.26 |
| shoulder | locked | 0.00 | 0.02 | 0.02 | 0.02 | n/a | n/a | −0.19 |
| shoulder | weak_motor | 1.00 | 0.98 | 0.98 | **0.27** | +0.99 | +1.62 | +0.37 |
| shoulder | missing_sensor | 0.00 | 0.00 | 0.00 | 0.00 | n/a | n/a | n/a |
| shoulder | broken_link | 0.69 | 0.60 | 0.02 | 0.02 | +0.02 | −1.59 | −7.38 |
| elbow | healthy | 1.00 | 0.98 | 0.98 | 0.98 | +0.99 | +1.67 | +0.40 |
| elbow | locked | 0.00 | 0.02 | 0.02 | 0.02 | n/a | n/a | −0.21 |
| elbow | weak_motor | 1.00 | 0.98 | 0.98 | 0.98 | +0.99 | +1.67 | +0.41 |
| elbow | missing_sensor | 0.00 | 0.00 | 0.00 | 0.00 | n/a | n/a | n/a |
| elbow | broken_link | 0.90 | **0.02** | 0.02 | 0.02 | −0.65 | −1.25 | −5.42 |
| wrist_1 | healthy | 1.00 | 0.98 | 0.98 | 0.98 | +0.99 | +1.82 | +0.61 |
| wrist_1 | locked | 0.00 | 0.02 | 0.02 | 0.02 | n/a | n/a | −0.21 |
| wrist_1 | weak_motor | 1.00 | 0.98 | 0.98 | 0.98 | +0.99 | +1.82 | +0.62 |
| wrist_1 | missing_sensor | 0.00 | 0.00 | 0.00 | 0.00 | n/a | n/a | n/a |
| wrist_1 | broken_link | 1.00 | **0.02** | 0.79 | 0.71 | −0.51 | +0.09 | +0.07 |
| wrist_2 | healthy | 1.00 | 0.98 | 0.98 | 0.98 | +1.00 | +1.84 | +0.65 |
| wrist_2 | locked | 0.00 | 0.02 | 0.02 | 0.02 | n/a | n/a | −0.20 |
| wrist_2 | weak_motor | 1.00 | 0.98 | 0.98 | 0.98 | +0.99 | +1.84 | +0.65 |
| wrist_2 | missing_sensor | 0.00 | 0.00 | 0.00 | 0.00 | n/a | n/a | n/a |
| wrist_2 | broken_link | 1.00 | 0.48 | 0.98 | 0.98 | +0.01 | +0.25 | +0.35 |
| wrist_3 | healthy | 1.00 | 0.98 | 0.98 | 0.98 | +0.99 | +1.86 | +0.62 |
| wrist_3 | locked | 0.00 | 0.02 | 0.02 | 0.02 | n/a | n/a | −0.19 |
| wrist_3 | weak_motor | 1.00 | 0.98 | 0.98 | 0.98 | +0.99 | +1.86 | +0.61 |
| wrist_3 | missing_sensor | 0.00 | 0.00 | 0.00 | 0.00 | n/a | n/a | n/a |
| wrist_3 | broken_link | 1.00 | 0.94 | 0.98 | 0.98 | +0.16 | +0.43 | +0.67 |

Reading the signature:

- **Locked and missing sensor are both 0.00 at T0.** They are indistinguishable on every tier. Both fail T0 and gate everything above it.
- **Weak motor shows only on the shoulder's T3** (0.27 pass, so detected in 73% of runs). The other joints pass every tier under a weak motor, as they must, since their loads are below the reduced torque.
- **Broken link is visible on T2 on the shoulder and elbow** (0.02 pass). It is visible on T1 on the elbow and wrist_1 (0.02 pass) and partly on wrist_2 (0.48), where the learned T1 is softer than the reference (section 3).
- **Base and wrist_3 broken link pass every tier** with pass probabilities of 0.73 to 0.98, so the learned policies miss this damage on those joints.

Reproduce: `python -m experiments.assess_policies`.

## 3. Agreement: model level against reference level

The model level is what the planner believes. The reference level is what the scripted controller reaches, noise-free. Agreement means the learned policies reach the same level as the simple controller.

| Joint | Damage | Reference | Model | Agree |
|---|---|---|---|---|
| base | healthy | 3 | 3 | yes |
| base | locked | −1 | −1 | yes |
| base | weak_motor | 3 | 3 | yes |
| base | missing_sensor | −1 | −1 | yes |
| base | broken_link | 1 | **3** | **no** |
| shoulder | healthy | 3 | 3 | yes |
| shoulder | locked | −1 | −1 | yes |
| shoulder | weak_motor | 2 | 2 | yes |
| shoulder | missing_sensor | −1 | −1 | yes |
| shoulder | broken_link | 1 | 1 | yes |
| elbow | healthy | 3 | 3 | yes |
| elbow | locked | −1 | −1 | yes |
| elbow | weak_motor | 3 | 3 | yes |
| elbow | missing_sensor | −1 | −1 | yes |
| elbow | broken_link | 1 | **0** | **no** |
| wrist_1 | healthy | 3 | 3 | yes |
| wrist_1 | locked | −1 | −1 | yes |
| wrist_1 | weak_motor | 3 | 3 | yes |
| wrist_1 | missing_sensor | −1 | −1 | yes |
| wrist_1 | broken_link | 1 | **0** | **no** |
| wrist_2 | healthy | 3 | 3 | yes |
| wrist_2 | locked | −1 | −1 | yes |
| wrist_2 | weak_motor | 3 | 3 | yes |
| wrist_2 | missing_sensor | −1 | −1 | yes |
| wrist_2 | broken_link | 1 | **0** | **no** |
| wrist_3 | healthy | 3 | 3 | yes |
| wrist_3 | locked | −1 | −1 | yes |
| wrist_3 | weak_motor | 3 | 3 | yes |
| wrist_3 | missing_sensor | −1 | −1 | yes |
| wrist_3 | broken_link | 1 | **3** | **no** |

**25 of 30 pairs agree.** All five disagreements are broken links, in two groups:

- **Missed damage (base, wrist_3): model 3, reference 1.** The learned policies pass a broken link on joints with little load. The signal is too weak for their T2 check to fail. This is the dangerous direction: it overclaims capability.
- **Policy gap (elbow, wrist_1, wrist_2): model 0, reference 1.** The learned T1 is softer than the nominal controller: its stiffness multiplier is about 0.51 on these three joints, half the reference's, so a broken link fails T1 where the reference passes. This direction underclaims capability, and it is a cost of learning rather than a hidden fault.

The shoulder agrees because its load makes the broken link fail T2 for both controllers.

Reproduce: `python -m experiments.assess_policies`.

## 4. Identifiability

Two damage states are **separable** if some tier's pass probability differs between them by at least 0.5. A pair that is not separable cannot be told apart by any tier with these skills, so the report cannot distinguish it.

Each joint has 10 pairs of damage states (5 choose 2), so 60 pairs in total.

**Scripted reference only** (the test that comes first, before any learned policy):

| Joint | Not separable |
|---|---|
| base | healthy / weak_motor |
| shoulder | none |
| elbow | healthy / weak_motor |
| wrist_1 | healthy / weak_motor |
| wrist_2 | healthy / weak_motor |
| wrist_3 | healthy / weak_motor |

5 of 60 pairs are not separable. The one shared pair is healthy against weak motor, which is expected: a weak motor only shows under load, and only the shoulder carries enough.

**Learned policies:**

| Joint | Not separable | Count |
|---|---|---|
| base | healthy / weak_motor, healthy / broken_link, locked / missing_sensor, weak_motor / broken_link | 4 |
| shoulder | locked / missing_sensor | 1 |
| elbow | healthy / weak_motor, locked / missing_sensor | 2 |
| wrist_1 | healthy / weak_motor, locked / missing_sensor | 2 |
| wrist_2 | healthy / weak_motor, healthy / broken_link, locked / missing_sensor, weak_motor / broken_link | 4 |
| wrist_3 | healthy / weak_motor, healthy / broken_link, locked / missing_sensor, weak_motor / broken_link | 4 |

17 of 60 pairs are not separable with the learned policies, against 5 with the scripted reference. The additional confusions come from the learned policies, not the physics:

- **healthy / broken_link** on base, wrist_2 and wrist_3 (the missed damage in section 3).
- **weak_motor / broken_link** on base, wrist_2 and wrist_3, for the same reason.

**Locked / missing sensor** is not separable on any joint, in either set. It is a real limit of the sensing model: both produce no usable motion, and no current tier differs between them. A check that reads the sensor without moving the joint would separate them. This is listed in [robotics-limitations.md](robotics-limitations.md).

## 5. Planning comparison

Four methods, same damage cases (seeds 0 to 99), same budget and gating:

- **planner**: expected information gain minus weighted risk and energy, with the safety veto. Three risk weights were run (0, 1, 4).
- **random**: an allowed probe chosen uniformly.
- **fixed**: the battery in tier order, joints from the base outwards.
- **passive**: no probes at all. A floor.

Metrics, averaged over 100 seeds:

- **Coverage**: share of joints whose level is determined.
- **Accuracy**: share of joints whose level is determined and equals the reference level.
- **Overclaim**: share of joints reported above the reference level.
- **Probes**: probes used. **Aborts**: probes that hit a safety limit. **Energy**: energy proxy in J per episode.

### 5.1 Learned pipeline (the stage-1 result)

| Budget | Method | Coverage | Accuracy | Overclaim | Probes | Aborts | Energy (J) |
|---|---|---|---|---|---|---|---|
| 12 | planner (risk 0, 1, 4) | 0.183 | 0.167 | 0.005 | 10.25 | 0.00 | 2.37 |
| 12 | random | 0.292 | 0.277 | 0.008 | 10.23 | 0.01 | 4.34 |
| 12 | fixed | 0.183 | 0.167 | 0.005 | 10.25 | 0.00 | 2.36 |
| 24 | planner (risk 0, 1, 4) | 0.673 | 0.640 | 0.020 | 15.76 | 0.03 | 7.08 |
| 24 | random | 0.683 | 0.647 | 0.020 | 15.73 | 0.02 | 7.18 |
| 24 | fixed | 0.682 | 0.645 | 0.022 | 15.81 | 0.02 | 7.17 |
| 40 | planner (risk 0, 1, 4) | 0.673 | 0.640 | 0.020 | 15.76 | 0.03 | 7.08 |
| 40 | random | 0.683 | 0.647 | 0.020 | 15.73 | 0.02 | 7.18 |
| 40 | fixed | 0.682 | 0.645 | 0.022 | 15.81 | 0.02 | 7.17 |
| any | passive | 0.000 | 0.000 | 0.000 | 0 | 0 | 0 |

The three planner rows are identical to the second decimal: the risk weight does not change the outcome here.

Reproduce: `python -m experiments.compare_methods --seeds 100 --budgets 12 24 40 --risk-weights 0 1 4` (about 21 minutes on the development machine).

### 5.2 Reference controller (scripted), for comparison

| Budget | Method | Coverage | Accuracy | Probes | Aborts | Energy (J) |
|---|---|---|---|---|---|---|
| 12 | planner | 0.183 | 0.183 | 10.28 | 0.00 | 1.95 |
| 12 | random | 0.283 | 0.280 | 10.16 | 0.01 | 3.04 |
| 12 | fixed | 0.173 | 0.173 | 10.28 | 0.00 | 1.63 |
| 24 | planner | 0.650 | 0.650 | 15.68 | 0.03 | 5.02 |
| 24 | random | 0.647 | 0.643 | 15.54 | 0.03 | 5.01 |
| 24 | fixed | 0.650 | 0.650 | 15.68 | 0.03 | 5.02 |

Scripted overclaim is 0.00 in every row, because the scripted model is the reference.

### 5.3 Whole-robot gating (RQ3 baseline), scripted

| Budget | Method | Coverage | Probes | Energy (J) |
|---|---|---|---|---|
| 12 | planner | 0.118 | 9.30 | 1.04 |
| 24 | planner | 0.443 | 13.84 | 2.91 |
| 40 | planner | 0.443 | 13.84 | 2.91 |

Chain gating reaches 0.65 coverage at budget 24, against 0.44 for whole-robot gating. That supports the first half of RQ3 for the scripted reference. The learned pipeline was not run with whole-robot gating.

## 6. Interpretation

**What the stage-1 pipeline establishes.**

- The learned skills are trustworthy on healthy joints: all 18 pass the gate, and healthy cells pass 98% of the time.
- The damage signatures are mostly where the reference puts them. The shoulder's weak motor and hold, and the shoulder's broken link, match exactly.
- The learned pipeline can be run end to end: train, gate, assess, plan, report and evaluate, with every step reproducible from a seed.

**What it does not establish.**

- **The planning method does not beat the fixed battery.** On a single chain, the set of allowed probes at any time is small. Usually one or two are allowed at once, so the order barely matters. The planner's advantage would need more choice, which means more chains (a quadruped) or more joints per chain.
- **The risk weight has no effect.** Aborts are rare (0.02 to 0.03 per episode), so the risk term never changes the order. A test at a higher abort rate would be needed for RQ2.
- **Overclaim is non-zero for the learned pipeline** (up to 0.022). It comes from the missed broken links on the base and wrist_3. The scripted reference never overclaims.

**Where the learned policies are weaker than the reference.** The policy gap has two sources:

1. The learned T1 gains are softer than the reference on the elbow and wrists (section 3, policy gap). A broken link fails them.
2. The learned T2 and T3 gains are at their clamps, so they cannot compensate for damage the reference would also fail. This is the intended effect of the clamp.

**What would change the conclusions.**

- A whole-arm model, so that joints interact and a shoulder fault changes the loads on the wrists.
- A stiffness probe at several poses, to find broken links on lightly loaded joints.
- A sensor-only check, to separate locked from missing sensor.
- A richer policy class, which could close the T1 gap without the stiffness clamp. The clamp is a stage-1 safety choice, and removing it would bring back the compensation effect.

## 7. How to reproduce everything

```bash
.venv/bin/python -m experiments.train_policies --workers 4     # 19 s: writes results/policies/arm6/
.venv/bin/python -m experiments.assess_policies                # 13 s: writes results/assessment_arm6.md
.venv/bin/python -m experiments.compare_methods --seeds 100 --budgets 12 24 40 --risk-weights 0 1 4 \
    --out results/comparison_learned.csv                       # ~21 min: writes the learned CSV
.venv/bin/python -m experiments.compare_methods --seeds 100 --budgets 12 24 40 --risk-weights 0 1 4 \
    --assessment scripted --out results/comparison_chain.csv   # scripted CSV
```

Results are in `results/`, which is git-ignored. The policy JSON files are regenerated by `train_policies`, so the repository does not carry them.

## 8. Rerun of 2026-10-15: belief scores and paired comparisons (new results)

Generated on Windows 10, CPython 3.11.9, numpy 2.4.6 ([testing.md](testing.md), section 6), with per-probe noise streams ([architecture.md](architecture.md), section 6). The artifacts are in `results/rerun_2026-10-15/` (git-ignored); its `MANIFEST.md` lists the commands. Retraining gave the 18 policies of [learning.md](learning.md) (two values differ in the last digit), and the assessment of sections 2 to 4 was reproduced exactly.

### 8.1 Conventions

| Quantity | Definition |
|---|---|
| Level events | For each joint and tier index k = 0 … 3: y_k = 1[reference level ≥ k]. The reference level is the scripted controller's level under the true damage (section 3), the target `accuracy` and `overclaim` already use. |
| Level forecast | p_k = P(level ≥ k) under the final belief, with the model levels the planner uses. A wrong model level (the learned broken links of section 3) therefore counts as a forecast error. |
| `level_brier` | Mean over k of (p_k − y_k)², averaged over joints. Range [0, 1], lower is better. |
| `level_log_loss` | Mean over k of −[y_k ln p_k + (1 − y_k) ln(1 − p_k)] in nats, with p_k clipped to [10⁻⁹, 1 − 10⁻⁹], so one event costs at most 20.7 nats. Probabilities of exactly 0 or 1 cannot give infinity. |
| Ties | The MAP set holds every hypothesis within 10⁻¹² of the largest belief. `hypothesis_expected_accuracy` = 1[truth in set] / set size: the accuracy of breaking ties at random. `hypothesis_tie_rate` is the share of joints whose MAP set has more than one member. `hypothesis_accuracy` (unchanged) breaks ties by dict order, which favours `locked` over `missing_sensor`. |
| `hypothesis_brier` | Σ_h (b_h − 1[h = truth])², averaged over joints. Range [0, 2]. |
| Prior-only forecast | The `passive` row: no probes, so the belief is the prior. |
| Coverage ceiling | Coverage of the fixed battery with a budget of 48 (joints × tiers × max_repeats), which cannot bind. A reference point, not a proven bound. |
| `time_s` | Seconds the executed skills actually ran, summed over the episode's probes: executed physics steps x DT, so an aborted skill counts until its abort and a missing sensor (never moved) counts 0. |
| `violations_overload`, `violations_joint_limit` | Number of probes in the episode that hit each safety limit, as reported by the physics. Cumulative over the probes run so far. |
| Per-tier confidence (report) | For each part and tier k, P(level >= k) under the current belief, the quantity the pass and fail rules threshold. Reported unrounded in the JSON report; the Markdown table is unchanged. |
| Curve (`--curve-out`) | Every metric after each probe k = 0 .. the largest budget. Computed by replaying the probe log of the run at the largest budget: no choice depends on the budget, so the state after k probes equals a run with budget k (tested field by field, and checked against the stored 2026-10-16 runs). |
| Paired difference | For each seed, method minus `fixed`, on the same damage and, with per-probe streams, the same noise for each probe. Mean over 100 seeds, with a 95% percentile bootstrap interval (2,000 resamples of seeds, generator seed 0). |

### 8.2 Results, 100 seeds

Budget 40 gives exactly the budget-24 numbers for every method, so it is not repeated. The three risk weights give identical rows except at budget 12 (learned), noted below.

**Learned pipeline** (coverage ceiling 0.680):

| Budget | Method | Coverage | Accuracy | Overclaim | Hyp. acc. (tie-aware) | Tie rate | Level Brier | Level log-loss | Aborts |
|---|---|---|---|---|---|---|---|---|---|
| 12 | planner (risk 0, 1) | 0.185 | 0.168 | 0.005 | 0.861 | 0.095 | 0.0424 | 0.2188 | 0.000 |
| 12 | planner (risk 4) | 0.187 | 0.168 | 0.003 | 0.861 | 0.095 | 0.0422 | 0.2168 | 0.000 |
| 12 | random | 0.295 | 0.278 | 0.008 | 0.855 | 0.083 | 0.0523 | 0.2456 | 0.010 |
| 12 | fixed | 0.187 | 0.168 | 0.003 | 0.861 | 0.095 | 0.0422 | 0.2168 | 0.000 |
| 24 | planner, random, fixed (all identical) | 0.680 | 0.647 | 0.018 | 0.868 | 0.095 | 0.0407 | 0.2545 | 0.020 |
| any | passive (prior only) | 0.000 | 0.000 | 0.000 | 0.792 | 0.000 | 0.1259 | 0.4175 | 0.000 |

**Scripted pipeline** (coverage ceiling 0.650):

| Budget | Method | Coverage | Accuracy | Overclaim | Hyp. acc. (tie-aware) | Tie rate | Level Brier | Level log-loss | Aborts |
|---|---|---|---|---|---|---|---|---|---|
| 12 | planner (risk 0, 1, 4) | 0.183 | 0.183 | 0.000 | 0.861 | 0.095 | 0.0383 | 0.1386 | 0.000 |
| 12 | random | 0.285 | 0.285 | 0.000 | 0.854 | 0.085 | 0.0477 | 0.1691 | 0.010 |
| 12 | fixed | 0.173 | 0.173 | 0.000 | 0.846 | 0.095 | 0.0452 | 0.1647 | 0.000 |
| 24 | planner, random, fixed (all identical) | 0.650 | 0.650 | 0.000 | 0.881 | 0.095 | 0.0301 | 0.1063 | 0.030 |
| any | passive (prior only) | 0.000 | 0.000 | 0.000 | 0.792 | 0.000 | 0.1262 | 0.4190 | 0.000 |

**Paired differences at budget 12** (95% bootstrap interval; for Brier and log-loss, negative means better):

| Pipeline | Comparison | Coverage | Level Brier | Level log-loss | Hyp. acc. (tie-aware) |
|---|---|---|---|---|---|
| learned | planner (risk 1) − fixed | −0.002 [−0.005, 0.000] | +0.0002 [0.0000, +0.0005] | +0.002 [0.000, +0.006] | 0.000 [0.000, 0.000] |
| learned | random − fixed | +0.108 [+0.082, +0.135] | +0.0101 [+0.0039, +0.0170] | +0.029 [−0.005, +0.062] | −0.006 [−0.015, +0.003] |
| learned | planner (risk 1) − random | −0.110 [−0.137, −0.083] | −0.0099 [−0.0169, −0.0038] | −0.027 [−0.059, +0.006] | +0.006 [−0.003, +0.015] |
| scripted | planner − fixed | +0.010 [+0.003, +0.018] | −0.0069 [−0.0113, −0.0032] | −0.026 [−0.040, −0.014] | +0.015 [+0.007, +0.025] |
| scripted | random − fixed | +0.112 [+0.087, +0.138] | +0.0025 [−0.0036, +0.0094] | +0.004 [−0.014, +0.024] | +0.008 [0.000, +0.018] |
| scripted | planner − random | −0.102 [−0.130, −0.077] | −0.0094 [−0.0158, −0.0040] | −0.031 [−0.049, −0.015] | +0.007 [−0.001, +0.015] |

The planner − random rows come from the trial CSVs, paired by seed in the same way, using `src.evaluate.paired_bootstrap`.

### 8.3 Findings

These are measured, in this simulation, with 100 seeds.

1. **At budgets 24 and 40, every probing method gives exactly the same result**, on every metric, in both pipelines, and every method reaches the coverage ceiling. With per-probe streams the same probe gives the same outcome under every method, so the identity is exact rather than within noise. Every allowed probe is run eventually, so the methods differ only in order, and order does not change the final belief.
2. **At budget 12, which method is better depends on the metric.** Random completes more joints (coverage +0.10 to +0.11 over the planner). The planner's final belief is better on the level Brier score (−0.009 to −0.010 against random), in both pipelines, with intervals that exclude zero. Coverage rewards finishing a joint's whole ladder, while the Brier score rewards knowledge spread over all joints and tiers.
3. **Against the fixed battery, the planner is better only in the scripted pipeline** (Brier −0.007, log-loss −0.026, coverage +0.010; intervals exclude zero). In the learned pipeline it is the same as or very slightly worse than fixed; the intervals touch zero.
4. **The risk weight changed the planner's choices in one setting only:** learned, budget 12, where risk 4 matched fixed exactly and risk 0 and 1 were slightly worse. In every other setting the three weights gave identical results. Aborts stay rare (at most 0.03 per episode).
5. **Probing beats the prior on every proper score**, by about 0.08 to 0.10 in level Brier and 0.16 to 0.31 nats in log-loss, with intervals that exclude zero.
6. **9.5% of joints end with a tied MAP set** (locked and missing sensor). With tie-aware scoring, hypothesis accuracy is 0.85 to 0.88 against 0.79 for the prior alone. That 0.79 is what argmax accuracy gives with no probes at all.

### 8.4 Comparison with the earlier documented results

| Setting (learned) | Section 5.1 (shared stream) | Section 8 (per-probe streams) |
|---|---|---|
| Budget 12: planner / random / fixed coverage | 0.183 / 0.292 / 0.183 | 0.185 / 0.295 / 0.187 |
| Budget 24: planner / random / fixed coverage | 0.673 / 0.683 / 0.682 | 0.680 / 0.680 / 0.680 |

The small differences between methods at budget 24 in section 5.1 came from the shared noise stream, which gave the same probe different noise under different methods. With pairing, they vanish. The qualitative conclusions of section 6 still hold, refined by finding 2: "the planner loses to random at budget 12" is true of coverage only.

### 8.5 What this does not establish

- The comparison is still the stage-1 design: the planner can only reorder a fixed set of probes. Whether active probing helps when probes can be skipped, or come in variants with different costs and risks, is untested (design decisions pending).
- The model is still well specified: the table and the robot's runs come from the same simulator. There is no misspecification test.
- Findings 2 and 3 are differences of 0.01 or less in Brier score, in simulation. They are statistically resolved, but small.
- The whole-robot gating comparison (section 5.3) was not rerun.
