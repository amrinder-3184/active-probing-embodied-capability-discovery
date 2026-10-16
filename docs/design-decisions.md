# Design decisions

This is the decision log for stage 1. Each decision has an ID, the value the code uses, the alternatives considered, the reason, and a status:

- **Implemented**: built as described.
- **Confirm**: the default is in the code, and the team or the TA must confirm or change it. These are also listed at the end.
- **Limitation**: a known gap, stated so that the report can say so.

Numbers are the values in `configs/`. Change a value there, not in code. Where a value affects the trained policies, the policies must be regenerated (`experiments/train_policies.py`).

## 1. Scope and platform

| ID | Decision | Value | Alternatives | Reason | Status |
|---|---|---|---|---|---|
| D1 | Platform | 6-DOF serial arm on a fixed base, one chain | Quadruped, humanoid, two-arm | The brief asks to start with an arm. An arm has no legs to balance, so the first version isolates the tier logic. | Implemented |
| D2 | Parts | Each joint is one part | Each link, each actuator-and-sensor pair | Joints are where the damage hypotheses act. | Implemented |
| D3 | Tiers on the arm | T0 to T3. T4 not implemented | Include a stand-up test | T4 is whole-body posture and needs legs. | Implemented |
| D4 | Simulator | Own batched single-joint model in numpy | MuJoCo, PyBullet, Gymnasium | Short enough to read, no heavy dependency, fast enough to run every probe under every hypothesis. | Implemented |
| D5 | Joint coupling | None: one joint at a time | Whole-arm dynamics | Keeps the model small. It is the main structural limit (A9 in [robotics-limitations.md](robotics-limitations.md)). | Limitation |
| D6 | Time step | 2 ms (500 Hz) | 1 kHz, 10 ms | 500 Hz is plausible for a joint servo loop. Learned policies would run slower in practice. | Implemented |

## 2. Tiers and thresholds

| ID | Decision | Value | Alternatives | Reason | Status |
|---|---|---|---|---|---|
| D7 | T0 checks | Sensor valid; response ≥ 0.5 of a 0.5° step; peak current ≤ 6 A | Current from a model | Response shows the actuator moves; current catches a stall. | Implemented |
| D8 | T1 check | Final error ≤ 1.0° after a 5° move in 1 s, held to 2 s | Peak error, settling time | Final error is what a user cares about. | Implemented |
| D9 | T2 check | RMS error ≤ 2.0° on a 20° out-and-back path over 4 s | Max error | RMS averages over the path, so one bad sample does not fail the path. | Implemented |
| D10 | T3 checks | Sag ≤ 1.5° over the last 1 s of a 3 s hold; peak torque ≤ 0.8 of rated | Torque only | Sag shows stiffness loss; torque shows excess demand. | Implemented |
| D11 | Threshold values | 0.5, 6 A, 1.0°, 2.0°, 1.5°, 0.8 | Measured from hardware | Hand-picked. There is no hardware. | **Confirm** |

## 3. Probes, budgets and safety

| ID | Decision | Value | Alternatives | Reason | Status |
|---|---|---|---|---|---|
| D12 | Probe sizes | 0.5° step; 5° move; 20° path; 3 s hold | Larger motions | Gentle by design (the brief says probes start gentle). | Implemented |
| D13 | Overload budget, T0 to T2 | 0.5 × rated torque | 0.4, 0.6 | Half of rated torque is gentle; a nominal joint's demand is far below it. | Implemented |
| D14 | Overload budget, T3 | **0.7 × rated torque** | 0.5 (the original) | At 0.5 the healthy shoulder hold passes 16% of randomised runs (see [model.md](model.md), section 9). At 0.7, it passes 100%, and the weak shoulder still fails 91%. | **Confirm** |
| D15 | Abort | Any violation ends the run and counts as a fail | Penalise and continue | An aborted run has no reliable readings, so a fail is the honest result. | Implemented |
| D16 | Joint-limit check | After each step, against `limits_deg` | Before each step | Checks the state the joint actually reached. | Implemented |
| D17 | Probe cost | Energy proxy, `∫(abs(τ·qd) + 0.002·τ²) dt` | Measured current | A proxy, not a measurement. | Limitation |

## 4. Damage hypotheses and prior

| ID | Decision | Value | Alternatives | Reason | Status |
|---|---|---|---|---|---|
| D18 | Hypotheses | healthy, locked, weak motor, missing sensor, broken link | More faults (encoder offset, friction, backlash) | The brief's four plus healthy. Encoder offset and friction are listed as future work. | Implemented |
| D19 | Locked | Stall torque `tau_max`, in the direction of the demand, when asked to move | Current proportional to command | Stall torque is what a motor draws when blocked. | Implemented |
| D20 | Weak motor | Torque capped at 40% | Lower torque constant (more current) | Simplest form. The current check therefore understates it (A6). | Limitation |
| D21 | Broken link | Stiffness scale 0.1 | 0.2 (tried first) | At 0.2, the wrists and base passed every tier, so they could not be identified. 0.1 separates them on T2. | Implemented |
| D22 | Prior | 0.8 healthy, 0.05 for each fault | Uniform; a fault-rate parameter | Most joints are healthy. The belief starts here, and the truth is drawn from it. | **Confirm** |
| D23 | Belief | Independent per joint | Shared-damage belief across joints | Simplest correct model. Shared damage is RQ3's second half. | Limitation |
| D24 | Likelihood | Per-check Gaussian pass probability, multiplied across checks; floored at 0.01 | Exact noise model per metric | A model-error floor, so one wrong prediction cannot rule a hypothesis out. | Implemented |

## 5. Gating and the evidence rule

| ID | Decision | Value | Alternatives | Reason | Status |
|---|---|---|---|---|---|
| D25 | Meaning of "below" | Closer to the base, in the same chain | Any tier, or the physical lower end of a chain | The brief says "the parts below it". The base supports everything above it. | **Confirm** |
| D26 | Gating rule (chain) | A tier of a joint is tested only when the same tier of each proximal joint has passed | Any tier of each proximal joint | Avoids confounded tests: a distal test on a faulty proximal joint is uninformative. | Implemented |
| D27 | Gating rule (robot) | A tier is tested on any joint only when every other joint has passed the tier below | — | The whole-robot baseline for RQ3. | Implemented |
| D28 | Evidence rule | A tier is concluded only after at least one probe of that tier | Belief alone | Without it, a T0 pass gave a 0.96 belief that T2 passed, with the true state a broken link. | Implemented |
| D29 | Conclusion thresholds | Pass when P ≥ 0.95; fail when P ≤ 0.05 | 0.9, 0.99 | Standard two-sided threshold. | Implemented |
| D30 | Repeat limit | After 2 probes without a conclusion, the tier is `unknown` | 3, 5 | Two probes confirm a deterministic fail under the floor. | Implemented |
| D31 | Ordering versus skipping | Utility orders probes but never skips an allowed one | Stop when no probe is worth its cost | A tier can only be concluded by probing it. Skipping would leave an untested tier `unknown` for no reason. | Implemented |

## 6. Planner

| ID | Decision | Value | Alternatives | Reason | Status |
|---|---|---|---|---|---|
| D32 | Information | Mutual information between capability level and pass/fail outcome, in bits | Information about the hypothesis | The report shows the level, so the planner should maximise information about the level. | Implemented |
| D33 | Risk | Expected weighted violation probability, weights fall 1.0, overload 0.5, joint limit 0.5 | Expected violation magnitude | Violations are discrete events in this model. | **Confirm** (weights) |
| D34 | Risk cap (all methods) | A probe is never run if its chance of any violation exceeds 0.2 | No veto; cap at 0.1 | A hard limit applies to every method, so baselines are as safe as the planner. | Implemented |
| D35 | Energy | Weight 0.5, scale 10 J | Weight 0 | Energy is a secondary cost. Its weight does not change the order here. | **Confirm** |
| D36 | Budget | 40 probes per episode | Per-tier budgets | A session limit. The planner usually stops before it, when no allowed probe is left. | Implemented |
| D37 | Baselines | random, fixed (tier order, then base outwards), passive (no probes) | A stronger baseline | Each baseline shares gating, budget and veto with the planner, so only the choice differs. | Implemented |
| D38 | Passive baseline | Zero probes: a floor | Free telemetry from normal operation | Passive observation has no telemetry model yet. | **Confirm** |

## 7. Learned skills

| ID | Decision | Value | Alternatives | Reason | Status |
|---|---|---|---|---|---|
| D39 | Policy form | Two gain multipliers, linear feedback on error and speed | MLP, feedforward, gravity bias | Gain-only policies cannot hide damage behind a feedforward (see [learning.md](learning.md), section 1). | Implemented |
| D40 | Learner | Evolution strategies | PPO, a grid search | Needs no gradient, and a few lines of numpy. A grid search is still useful as a check (not yet built). | Implemented |
| D41 | Stiffness bounds | [0.5, 1.0] | [0.25, 4.0] (tried first) | At up to 4, a stiffer policy hid a broken link on T2, which passed 98% of the time. | Implemented |
| D42 | Damping bounds | [0.5, 4.0] | [0.25, 4.0] | Damping cannot hide a stiffness loss, so its bounds are wider. | Implemented |
| D43 | Reward | Linear in the primary metric over its threshold, minus 0.05 × effort, minus 10 on abort | Squared metric | Squared metric had no gradient near zero error and drove gains to their bounds. | Implemented |
| D44 | Training | Population 32 (antithetic), 8 instances, 40 generations, σ 0.1, learning rate 0.1 | 4 instances | 4 instances gave noisy estimates when aborts are rare. | Implemented |
| D45 | Randomisation | ±10% on τ_max, I, g and stiffness | ±5%, ±20% | Gives a range of joints for training, and the gate tests robustness to it. | **Confirm** |
| D46 | Input noise | Angle σ 0.05°, speed σ 0.5°/s | No noise | The controller must not depend on perfect sensing. Values are assumed, not measured (A4). | **Confirm** |
| D47 | Healthy gate | Wilson 95% lower bound ≥ 0.85 on 100 episodes; needs 92 successes | Raw rate ≥ 0.9 | The lower bound accounts for sampling error. A raw 0.9 from 100 runs is trusted only 32% of the time. | **Confirm** |
| D48 | Untrusted skill | `unknown` everywhere it is used | Use it with a warning | An untrusted skill's evidence is not reliable, so it is not used. | Implemented |
| D49 | Model-table pass probability | Laplace-smoothed, (passes + 1)/(50 + 2) | Raw rate | Keeps 0 and 1 out of the belief. | Implemented |
| D50 | T0 | Scripted, not learned | Learned T0 | T0 is a diagnostic and gates the learned skills. Learning it would make the gate depend on the thing it gates. | **Confirm** |

## 8. Evaluation and reporting

| ID | Decision | Value | Alternatives | Reason | Status |
|---|---|---|---|---|---|
| D51 | Damage truth | The true hypothesis, sampled from the prior | — | The question is which damage is present. | Implemented |
| D52 | Reference level | The scripted controller's level, noise-free | The best achievable by any controller | The reference is what a simple controller does. A stronger search would give a higher reference. | **Confirm** |
| D53 | Model level | Highest tier in a contiguous ladder with predicted pass probability ≥ 0.5 | Expected level | The belief uses it, so it must say what the assessor predicts. | Implemented |
| D54 | Level determined | Top tier passed, or the next tier failed. A "blocked" next tier is a lower bound | Always report a level | A blocked tier was never tested. Calling its level determined would overclaim. | Implemented |
| D55 | Metrics | Coverage, accuracy, overclaim, hypothesis accuracy | Only accuracy | Overclaim is the dangerous error, so it is reported separately. | Implemented |

## 9. Seeds and reproducibility

| ID | Decision | Value | Alternatives | Reason | Status |
|---|---|---|---|---|---|
| D56 | Seed streams | Training 1000 + index; gate 2000 + index; table 3000 with SeedSequence per entry; truth SeedSequence([seed, 1, joint, tier, repeat]) per probe run (stage 1: one stream seed + 1); damage seed; policy seed + 2 | One global seed | Independent streams, so no two uses share numbers. A gate-stream overlap was found and fixed (commit `70ab4e4`). Per-probe truth streams (2026-10-15) make method comparisons paired: the same probe gets the same noise under every method. `noise_streams="shared"` keeps the stage-1 stream. See [architecture.md](architecture.md), section 6. | Implemented |
| D58 | Belief scores | Brier and log-loss (ε = 1e-9) of P(level ≥ k) against the reference level; tie-aware hypothesis accuracy; multi-class Brier | Argmax accuracy only | Argmax accuracy rewards the prior (0.79 with no probes) and breaks the locked / missing-sensor tie by dict order. Proper scores also penalise confident wrong beliefs. See [assessment.md](assessment.md), section 8. | Implemented |
| D59 | Method comparison | Paired differences against `fixed` over the same seeds, 95% percentile bootstrap over seeds (2000 resamples, seed 0) | Unpaired means | The methods share their seeds, so the per-seed difference removes the variation between damage cases. | Implemented |
| D57 | Index for training seeds | Counts every (joint, tier) slot in order, including T0 | Only learned slots | Keeps seeds stable if a tier's assessor changes. | Implemented |

## 10. Items to confirm

These are the decisions the TA or team must confirm. Each has a default in the code, so the work runs without them.

| Item | Decision ID | Default | Question |
|---|---|---|---|
| Pass thresholds | D11 | 0.5, 6 A, 1.0°, 2.0°, 1.5°, 0.8 | Are these the right limits for the robot the team chooses? |
| Hold budget | D14 | 0.7 × rated torque | Is 70% of rated torque acceptable for a loaded hold on the real robot? |
| Meaning of "below" | D25 | Closer to the base | Does the brief mean this, or any tier of any proximal joint? |
| Prior and fault rate | D22 | 0.8 / 0.05 each | What fault rate should the belief assume? |
| Risk weights | D33 | 1.0 / 0.5 / 0.5 | Which violations matter most? |
| Energy weight | D35 | 0.5 | Is energy a secondary cost, or should it decide the order? |
| Passive telemetry | D38 | None (floor) | What free data does normal operation give? |
| Randomisation | D45 | ±10% | Is ±10% a fair spread, or should it match measured tolerances? |
| Input noise | D46 | 0.05°, 0.5°/s | Can the team measure real encoder noise? |
| Healthy gate | D47 | 92 of 100 | Is 0.85 the right confidence bound? |
| T0 learned or scripted | D50 | Scripted | Should T0 ever be learned? |
| Reference controller | D52 | Scripted PD | Should the reference be the best achievable controller instead? |
| Gravity compensation | Section 3 of [robotics-limitations.md](robotics-limitations.md) | No compensation | Does the target robot compensate for gravity? If so, the hold probe must change. |
