# Learning: policies, training and the healthy gate

This document describes the learned skills in stage 1: what a policy is, how it is trained by evolution strategies, how a trained policy is judged, and what training produced. The reader is assumed to know what a controller is but not necessarily what evolution strategies are. Section 2 explains them in a few steps.

## 1. The policy

A stage-1 policy is a control law with two learned numbers per joint and skill:

```
torque = kp_mult * kp * (ref − q)  −  kd_mult * kd * qd
```

| Symbol | Meaning | Source |
|---|---|---|
| `kp`, `kd` | The joint's nominal PD gains | `configs/platform_arm6.yaml` |
| `kp_mult` | Learned multiplier on stiffness | Trained, clamped to **[0.5, 1.0]** |
| `kd_mult` | Learned multiplier on damping | Trained, clamped to **[0.5, 4.0]** |
| `ref`, `q`, `qd` | Commanded angle, measured angle, measured speed | The policy sees noisy versions (section 3) |

The scripted reference controller is this law with both multipliers fixed at 1.0. So a learned policy and the reference differ only in their two numbers.

### Why this form, and why these bounds

The policy has no feedforward term, no gravity bias, and no input about the fault. Each of these would hide damage:

- **A gravity bias** carries the load on a hold. A hold then looks perfect even when stiffness has dropped, because the bias does the work the stiffness should do.
- **A reference-velocity feedforward** cancels the lag that a softer loop produces during a move, so a broken link looks healthy on T2.
- **A stiffness multiplier above 1** lets the controller make up for a softer joint by pushing harder. A broken link then passes T2.

The last effect was observed, not only predicted. In the first learned run, the trainer pushed the stiffness multiplier to about 4 on T2 and T3. The broken link then passed T2 in 98% of runs, where the nominal controller fails it every time. The fix is the clamp `kp_mult ≤ 1.0`: a learned controller may soften a joint, but never stiffen it beyond its nominal design. Damping has no such effect on stiffness, so its clamp is wider.

The lower bounds stop the policy drifting into very soft regions. At `kp_mult = 0.25`, a broken link (scale 0.1) reduces effective stiffness to 2.5% of nominal, and the policy then fails T1 where the nominal controller passes. The 0.5 floor keeps that effect bounded, though it does not remove it (section 6).

## 2. Evolution strategies, in steps

Evolution strategies (ES) are a black-box search. They need no gradient, so they work with the simulator as it is. Here is the loop, with the values used in stage 1.

1. **Start** the mean `θ` at the nominal controller: `θ = (1.0, 1.0)`.
2. **Perturb.** Draw 16 random directions `ε` (population 32 in antithetic pairs, so for each `ε` also `−ε`). Candidate `i` is `θ + σ·ε_i`, with `σ = 0.1`, clipped to the bounds.
3. **Evaluate.** Score each candidate on 8 randomised healthy joints (`instances = 8`). The score of a candidate is the mean reward over those 8 runs.
4. **Rank.** Convert the 32 scores to centred ranks `f_i ∈ [−0.5, 0.5]`. Ranks make the update independent of the reward's scale, so one aborted run cannot dominate.
5. **Update.** `θ ← θ + α · (1 / (P·σ)) · Σ f_i · ε_i`, with learning rate `α = 0.1`, `P = 32`. Clip to the bounds.
6. **Repeat** for 40 generations.
7. **Gate.** Run the final `θ` on 100 held-out randomised healthy joints (section 4).

The antithetic pairs halve the variance of the update, because each direction is tried both ways. Rank fitness keeps the step size stable when rewards are on very different scales.

## 3. Reward, input noise and randomisation

**Reward** for one run:

```
if the run aborted:           R = −10                       (abort_penalty)
otherwise:  R = −(m / θ_tier) − 0.05 · E
where:
  m        the tier's primary metric: final error (T1), RMS error (T2), sag (T3)
  θ_tier   the tier's threshold for that metric (1.0°, 2.0°, 1.5°)
  E        mean over alive steps of (demand / budget)²
```

The main term is linear in the metric, in units of the tier's own pass limit. It was squared at first. The squared term has no gradient near zero error, and T1 errors are almost always near zero, so the effort term alone set the gains and drove them to their bounds. The linear form keeps a gradient at small errors.

The effort term is small (weight 0.05). It is the only term that penalises demands close to the overload budget before an abort happens. The abort penalty applies only when a run actually aborts.

**Input noise.** The controller sees the angle and speed with additive Gaussian noise, σ = 0.05° and σ = 0.5°/s. The physics does not change. This is noise on what the policy knows, not on what the joint does.

**Randomisation.** Each candidate's instances draw `tau_max`, `I`, `g` and the stiffness scale uniformly within ±10% of nominal. The policy must therefore work across a range of joints, not one.

## 4. The healthy gate

After training, the policy runs on 100 randomised healthy joints with reading noise, as the real sensors would give. A tier passes when its checks pass.

The gate does not use the raw success rate. It uses the **Wilson score lower bound** at 95% confidence (`z = 1.96`):

```
p̂ = successes / n
L  = ( p̂ + z²/(2n) − z · sqrt( p̂(1 − p̂)/n + z²/(4n²) ) ) / ( 1 + z²/n )
```

The policy is **trusted** if `L ≥ 0.85`. For `n = 100`, that means at least **92 successes**. The lower bound matters because a raw rate near 0.9 from 100 runs has real uncertainty: a policy that is truly at 0.90 still gets `L ≥ 0.85` only about a third of the time.

| True healthy success | Probability the gate trusts it (n = 100) |
|---|---|
| 0.80 | 0.001 |
| 0.90 | 0.321 |
| 0.92 | 0.593 |
| 0.95 | 0.937 |
| 0.98 | 1.000 |
| 1.00 | 1.000 |

Wilson values for `n = 100`: 85 successes → 0.767; 90 → 0.826; 92 → 0.850; 94 → 0.875; 96 → 0.902; 100 → 0.963.

A policy that is not trusted gives `unknown` for its tier everywhere in the planner. Its tier can never be passed or failed by evidence, so no damage conclusion is drawn from it.

## 5. The model table uses the same policy

The model table, which the planner uses for its belief, runs each trained policy 50 times per (joint, tier, hypothesis). Each rollout has a randomised plant, input noise, and reading noise. The pass probability is Laplace-smoothed:

```
pass_prob = (passes + 1) / (50 + 2)
```

The smoothing keeps 0 and 1 out of the belief, so one wrong prediction cannot make a hypothesis impossible. With 50 rollouts and all passing, the value is 51/52 = 0.981. That is why healthy joints show 0.98 in the assessment table, not 1.00.

## 6. Results of training

Trained on 18 (joint, tier) pairs: 6 joints × 3 learned tiers (T1, T2, T3). T0 is scripted. Each job ran 40 generations in about 4 seconds, and all 18 took 19 s wall with four workers. The records are in `results/policies/arm6/`, one JSON file per (joint, tier).

| Joint | Tier | kp_mult | kd_mult | Healthy success | Wilson lower | Trusted | Reward first → last |
|---|---|---|---|---|---|---|---|
| base | T1 | 0.500 (floor) | 1.306 | 100/100 | 0.963 | yes | −0.0079 → −0.0058 |
| base | T2 | 1.000 (cap) | 0.500 (floor) | 100/100 | 0.963 | yes | −0.1974 → −0.1013 |
| base | T3 | 0.500 (floor) | 1.759 | 100/100 | 0.963 | yes | −0.0155 → −0.0112 |
| shoulder | T1 | 0.944 | 1.302 | 100/100 | 0.963 | yes | −0.0064 → −0.0060 |
| shoulder | T2 | 1.000 (cap) | 0.500 (floor) | 100/100 | 0.963 | yes | −0.4019 → −0.2138 |
| shoulder | T3 | 1.000 (cap) | 1.692 | 100/100 | 0.963 | yes | −0.7014 → −0.6898 |
| elbow | T1 | 0.512 | 1.143 | 100/100 | 0.963 | yes | −0.0062 → −0.0055 |
| elbow | T2 | 1.000 (cap) | 0.500 (floor) | 100/100 | 0.963 | yes | −0.3631 → −0.1972 |
| elbow | T3 | 1.000 (cap) | 0.812 | 100/100 | 0.963 | yes | −0.4996 → −0.4964 |
| wrist_1 | T1 | 0.513 | 1.898 | 100/100 | 0.963 | yes | −0.0078 → −0.0058 |
| wrist_1 | T2 | 1.000 (cap) | 0.500 (floor) | 100/100 | 0.963 | yes | −0.1973 → −0.1005 |
| wrist_1 | T3 | 1.000 (cap) | 1.606 | 100/100 | 0.963 | yes | −0.1141 → −0.1119 |
| wrist_2 | T1 | 0.508 | 1.554 | 100/100 | 0.963 | yes | −0.0077 → −0.0063 |
| wrist_2 | T2 | 1.000 (cap) | 0.500 (floor) | 100/100 | 0.963 | yes | −0.1761 → −0.0917 |
| wrist_2 | T3 | 1.000 (cap) | 1.553 | 100/100 | 0.963 | yes | −0.0953 → −0.0925 |
| wrist_3 | T1 | 0.558 | 1.748 | 100/100 | 0.963 | yes | −0.0085 → −0.0058 |
| wrist_3 | T2 | 1.000 (cap) | 0.500 (floor) | 100/100 | 0.963 | yes | −0.1521 → −0.0765 |
| wrist_3 | T3 | 1.000 (cap) | 3.127 | 100/100 | 0.963 | yes | −0.0428 → −0.0380 |

All 18 are trusted, and each passed all 100 healthy gate runs.

### What the table says

- **T2 sits on both bounds on every joint** (stiffness at the cap, damping at the floor). The reward would prefer more stiffness and less damping, which the cap and the floor prevent. The cap is binding on T2 and on the shoulder hold, so those two skills are determined by the clamp, not by the reward.
- **T1 stiffness is 0.51 to 0.94 on five joints, and at the floor (0.50) on the base.** Damping is 1.14 to 1.90. The move is easy: the healthy final error is zero, so the gains are chosen by the effort term, not the tracking term.
- **The base hold drives stiffness to its floor (0.5)** because the base carries no load. The reward is then almost entirely effort, and the lowest gain that passes wins.
- **The shoulder hold sits at the stiffness cap** with damping near 1.7. The reward keeps improving up to the cap, so the cap is what stops it. The effort term and the overload aborts pull in opposite directions near the cap.
- **Rewards are small in magnitude for T1** (about −0.006 at the end), because healthy moves finish with near-zero error. For T2 and T3 the rewards are larger, because the tracking error is a real fraction of the threshold.

## 7. Settings that were changed during development

The trained values above depend on these settings. Each change was driven by a measured failure, and each is in `configs/training.yaml`.

| Setting | First value | Current value | Why it changed |
|---|---|---|---|
| Stiffness multiplier bounds | [0.25, 4.0] | [0.5, 1.0] | A stiffer policy hid a broken link on T2 (section 1). |
| Damping multiplier bounds | [0.25, 4.0] | [0.5, 4.0] | Separate bounds for the two gains, so the stiffness cap does not also limit damping. |
| Instances per candidate | 4 | 8 | With 4, rare aborts gave noisy estimates. The shoulder hold settled in a poor region with kp ≈ 2.9 and 12–15% aborts. |
| Reward shape | Squared metric | Linear metric | Squared metric had no gradient near zero error and drove T1 gains to their bounds. |
| Hold budget (T3, scripted and learned) | 0.5 | 0.7 | At 0.5, the healthy shoulder hold passed 16% of randomised runs (section 9 of [model.md](model.md)). |
| Gate seed | `training + 1` (overlapped) | `gate + index` | The gate stream overlapped the next job's training stream. Fixed in commit `70ab4e4`; the policies and their gate results did not change. |

Four of these changed the trained policies: the two sets of bounds, the instance count, the reward shape and the hold budget. The gate seed changed only the gate stream, and its results did not change. The stiffness cap was the most important, because it reversed a result: before it, the assessment showed a broken link as healthy on T2, which is the wrong answer.

## 8. Known behaviour and limits

- **The gain multipliers are not identifiable on the easy skills.** On T1 and on the base hold, many gain values pass, so ES settles on the one the effort term prefers. The pass result is what matters for assessment, not the gain value.
- **The stiffness clamp is binding on every T2 policy.** All six sit at the cap, so the reward's wish for more stiffness is never met. The cap is not a physical limit of the joint. It is a design choice for stage 1.
- **Training uses only healthy joints.** A policy is never shown a damaged joint during training. Its behaviour under damage is what the assessment measures.
- **Gain-only policies cannot express a different strategy.** For example, a policy cannot switch to a slower, more compliant motion when the joint is weak. That needs an observation of the fault, which stage 1 deliberately withholds. Richer policies are stage 2.
- **Eight instances is still a small sample.** The shoulder hold's reward has high variance from rare aborts. The gate, not the training reward, is the check on quality.

## 9. Tuning recipe

For a new platform or a changed skill, the sequence that produced the current settings:

1. Run the scripted reference over the randomised healthy population and read off the pass rate of each skill. If a skill passes under 90% of randomised healthy runs, no gain setting will pass it reliably, and the cause is in the physics or the budget. Fix that first (section 9 of [model.md](model.md) is an example).
2. Map the healthy pass rate over a grid of multipliers. This shows whether a feasible region exists and where it is. For the shoulder hold it was kp in [0.8, 1.5] and kd in [1.0, 2.5].
3. Train with `instances = 8`, `generations = 40`. Check the reward curve. If the mean reward is still falling at the last generation, raise `generations` before changing anything else.
4. Gate. A skill that is not trusted is a finding, not a failure to hide. Report it.
5. Run `assess_policies`. Check that a skill's healthy pass rate matches the gate, and that the damage signatures are where the reference puts them.
