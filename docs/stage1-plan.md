# Stage 1 plan, as built

This document is the stage-1 plan, updated to match what was built. Each section of the original plan is listed with its status. Where the build differs from the plan, the deviation and its reason are recorded. The robotics review is kept, with the decision made on each point.

## 1. Goal

Simulate a damaged joint, train a small policy with reinforcement learning for each skill (T1 move, T2 path, T3 hold) on healthy joints, run each policy on the damaged joint, and infer the damage and the highest tier the joint can still do from how the policy performs.

T0 (sensor and response check) stays scripted, because it is a diagnostic and it gates the learned skills.

## 2. Status of each plan item

| Plan section | Item | Status | Where |
|---|---|---|---|
| 3.1 | One batched physics model | Done | `src/physics.py` |
| 3.2 | Control law: pure feedback, no feedforward | Done, with tighter bounds (section 3) | `src/policies.py` |
| 3.3 | Stiffness fault scales only the position term | Done | `src/physics.py` |
| 3.4 | Training by evolution strategies, randomised healthy joints | Done, with changes (section 3) | `src/training.py` |
| 3.5 | Healthy gate with Wilson lower bound | Done. Threshold corrected to 92 of 100 (section 3) | `src/training.py` |
| 3.6 | Two assessors, one interface | Done | `src/assessors.py` |
| 3.6 | Model table: 50 rollouts, Laplace pass probability | Done | `src/assessors.py` |
| 3.6 | Truth execution under the true damage, fresh seed | Done | `src/episode.py` |
| 3.6 | Separate seed ranges | Done, after a fix (section 3) | `src/training.py`, `configs/training.yaml` |
| 3.6 | Margin to the threshold as the "direction" | Done, reported in the assessment | `experiments/assess_policies.py` |
| 3.7 | Model, reference and truth kept apart | Done | `src/outcomes.py`, `src/evaluate.py` |
| 3.8 | Conclusions from probes only | Done | `src/gating.py` |
| 3.9 | Horizontal scaling: independent jobs | Done for training (`--workers`). Table building is serial. | `experiments/train_policies.py` |
| 3.9 | Vertical scaling: interfaces at each layer | Done | [architecture.md](architecture.md), section 8 |
| 3.9 | No hard-coded numbers | Done, with three exceptions in code (see section 3) | `configs/` |
| 3.9 | Reproducibility: saved policy records | Done | `results/policies/arm6/*.json` |
| 3.10 | Layering with no upward imports | Done | [architecture.md](architecture.md), section 2 |
| 3.11 | Tests, items 1 to 4, 6 to 10 | Done | [testing.md](testing.md) |
| 3.11 | Test 5: ES matches a grid search | **Not done** | Planned, not built. |
| 3.11 | Test 7: a learned policy passes the gate on one joint and tier | Partly: the assessor test covers T1 with fast settings, not the gate on the real settings | `tests/test_assessors.py` |
| 3.12 | Acceptance for the experiment | Done (section 5) | `results/assessment_arm6.md` |
| 4 | Code changes table | Done, with additions (section 4) | — |
| 5 | Plain steps for students | Done | [README.md](../README.md) |

## 3. Deviations from the plan

Each deviation was made because a measured result required it, or because the plan was wrong about a number. The reason is given.

| Plan said | Built | Reason |
|---|---|---|
| Stiffness and damping multipliers clamped to [0.25, 4] | Stiffness [0.5, 1.0], damping [0.5, 4.0] | A stiffer policy hid a broken link on T2 (98% pass where the reference fails). See [learning.md](learning.md), section 1. |
| Population 32, 4 instances per candidate | Population 32, **8 instances** | With 4, aborts were too rare to estimate, and the shoulder hold settled at kp ≈ 2.9 with 12 to 15% aborts. |
| Reward: squared (metric / threshold)² | **Linear** metric / threshold | The squared form had no gradient near zero, and T1 gains drifted to their bounds under the effort term. |
| Gate needs about 91 of 100 successes | **92 of 100** | Computed from the Wilson bound: 91 gives 0.838, 92 gives 0.850. |
| "A policy truly at 0.9 passes about half the time" | **32%** | Computed exactly: `P(lower bound ≥ 0.85 given p = 0.90, n = 100) = 0.321`. |
| Hold budget not specified (50% in the code) | **70% for T3** | At 50%, the healthy shoulder hold passed 16% of randomised runs. Recorded as decision D14, for confirmation. |
| Skills use a single overload budget | Budget is **per skill** (`budget_fraction` in `platform_arm6.yaml`) | Needed for the hold budget. Keeps the budget explicit in the config. |
| Gate uses a separate seed stream | Gate seed was **training seed + 1**, which overlapped the next job | Found in review and fixed (commit `70ab4e4`). The trained policies and their gate results did not change. |
| Test 5: ES matches a grid search on the two-gain problem | **Not built** | Deferred. The determinism test is in place. Recorded in [testing.md](testing.md). |
| Reward per step | **Reward per run**: the primary metric over the run (final error, RMS or sag), plus the effort term, minus the abort penalty on abort | The metric is a property of the whole run, not of each step. |
| (not in plan) | Stiffness multiplier clamped to nominal | Added after the broken-link result. It is the most important change: without it, the learned policies show a broken link as healthy on T2. |

## 4. Changes to the code table

The plan's table listed the changes to make. These were built as planned, with these additions:

| Added | Reason |
|---|---|
| `src/tasks.py` (separated from the old `sim.py`) | Reference motions and readings are shared by probes and policies. |
| `src/runner.py` | One place that runs a skill on a batch and adds reading noise. |
| `experiments/assess_policies.py` | The acceptance checks: signatures, agreement and identifiability. |
| Per-skill budget | Needed for the hold budget. |
| `tests/test_physics.py`, `test_policies.py`, `test_training.py`, `test_assessors.py`, `test_probes.py` | Replace the old `test_sim.py`. |

`src/sim.py` was removed. Its physics is in `src/physics.py`, and its probes are in `src/tasks.py` and `src/runner.py`.

## 5. Acceptance (section 3.12 of the original plan)

| Criterion | Result | Met |
|---|---|---|
| Each trained skill is trusted, or reported as untrusted | 18 of 18 trusted. Each passed 100 of 100 healthy gate runs. | Yes |
| Signature table: one row per (joint, skill), one column per hypothesis | Produced: 30 rows of pass probability and margin | Yes |
| Agreement between model and reference level, reported not forced | 25 of 30 pairs agree. Five disagreements, all broken links, reported in [assessment.md](assessment.md), section 3 | Yes (reported) |
| Identifiability: pairs not separable on any skill, listed | Scripted: 5 of 60 pairs. Learned: 17 of 60 pairs. Listed in [assessment.md](assessment.md), section 4 | Yes |
| Identifiability checked first on the scripted controller | Yes. `assess_policies.py` runs it first. | Yes |
| Runtime: all 18 policies train in a few minutes with parallel workers | 19 s wall with 4 workers | Yes |

## 6. Robotics review and decisions

A subagent reviewed the first draft of this plan as a robotics researcher. The review's points and the decision on each are recorded here, as they were made. The current state of each limitation is in [robotics-limitations.md](robotics-limitations.md), section 7.

| Point | Decision |
|---|---|
| Locked joint "gets an overload abort on T3" | **Corrected.** At home a locked joint has zero demand, so it holds with no sag and passes T3 if run. It never reaches T3, because T0 fails and gates it. |
| Policy "no feedforward", but a reference-velocity term was present | **Accepted.** The policy is pure feedback on error and speed. The scripted PD is the same law. |
| Reward units | **Accepted.** Reward is (metric / threshold). Later changed to linear. |
| Truth equal to the PD level is controller performance, not damage | **Accepted.** Damage truth is the true hypothesis. Reference level is kept separately. |
| Budget versus clip: weak motors saturate between 0.4 and 0.5 of rated torque | **Kept by design.** A weak motor saturates before the overload limit. |
| T3 measures kp × gravity; compensation would hide stiffness loss | **Accepted as a limit.** See [robotics-limitations.md](robotics-limitations.md), section 3. |
| gravity_nm is one number at one hold pose | **Accepted as a limit.** |
| Friction is missing | **Accepted as a limit.** Future work. |
| No latency; velocity from a differenced encoder | **Accepted as a limit.** |
| Weak motor is a hard clip; a demagnetised motor draws more current | **Accepted as a limit.** |
| Abort must mean drive disable and brake | **Accepted as a hardware precondition.** |
| Sign-flipped encoder | **Accepted as a hardware precondition.** |
| "Penalised" over-budget requests are not limits | **Accepted.** Gains are hard-clamped. Overload is a hard abort. |
| Weak elbow and wrists are not separable | **Accepted as a finding.** |
| Locked against a sluggish policy | **Accepted as a known confusion.** |
| Noise counted twice | **Clarified.** Input noise and reading noise are separate sources. |
| ES is overkill for two parameters; use a grid | **Partly rejected.** ES stays, because RL is the goal. A grid search remains a planned check (not built). |
| No test that the policy cannot see the fault | **Accepted.** `test_the_controller_only_sees_error_and_speed`. |
| No test for locked on learned gains | **Accepted.** Locked joints fail T0, which gates the learned skills. |
| Gate of 20 rollouts is too noisy | **Accepted.** 100 gate runs with a Wilson bound. |
| Are model-table rollouts randomised? | **Decided: yes.** |
| Identifiability threshold, so "not forced" can fail | **Accepted.** A pair is separable if some tier differs by at least 0.5. |

## 7. Open questions for the TA

These are listed in [design-decisions.md](design-decisions.md), section 10, with defaults. The four that most change the results are:

1. Should the policy class stay linear, or move to an MLP after stage 1?
2. Is the scripted PD the right reference, or should a stronger search give the best achievable performance?
3. Is a Wilson lower bound of 0.85 the right gate?
4. For a robot with gravity compensation, which probe replaces T3?

## 8. Next steps

In order of their effect on the results:

1. **Whole-arm dynamics.** Joint coupling is the main structural limit. Without it, chain gating has no physical basis and the planner has little to choose between.
2. **A sensor-only check** to separate a locked joint from a missing sensor.
3. **A stiffness probe at several poses** for broken links on lightly loaded joints, and for robots with gravity compensation.
4. **Friction and latency** in the model, then re-run the gate (see [robotics-limitations.md](robotics-limitations.md)).
5. **The ES-versus-grid test**, so the trainer's optimum is checked, not assumed.
6. **A richer policy class** with the stiffness clamp kept, to see whether the T1 policy gap closes.
