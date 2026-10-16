# Model: physics, faults and skills

This document defines the simulated robot exactly: the equations, the parameters of each joint, the damage hypotheses and their physical effect, the skills (the tier tasks), and the readings each skill produces. Numbers are taken from `configs/platform_arm6.yaml` and `configs/damage.yaml`, and the scripted readings are measured from the code.

## 1. Joint dynamics

Each joint is a single rotational degree of freedom with inertia `I`, gravity load `g` (non-zero only in the hold skill), and a position-feedback controller. Time advances in steps of `dt = 0.002 s` (500 Hz).

The controller supplies two torque terms, a stiffness term `S` (on the position error) and a damping term `D` (on the speed):

```
S = kp_mult * kp * (ref - q_seen)          stiffness term
D = -kd_mult * kd * qd_seen                damping term
demand = stiffness_scale * S + D           torque asked for, before the motor cap
```

`q_seen` and `qd_seen` are the angle and speed the controller sees. They equal the true values plus input noise (section 5). `stiffness_scale` is 1 for a healthy joint. A stiffness fault scales only the `S` term, as a softer position loop would.

The applied torque and the motion follow:

```
tau     = clip(demand, -tau_max * torque_scale, +tau_max * torque_scale)     (unlocked)
qdd     = (tau - g) / I
qd(t+dt)= qd(t) + qdd * dt
q(t+dt) = q(t) + qd(t+dt) * dt                                             (semi-implicit Euler)
```

Energy is accumulated as a proxy:

```
energy += (|tau * qd| + heat_coeff * tau^2) * dt        heat_coeff = 0.002
```

## 2. Safety checks

Each step, in this order:

1. **Overload.** If `|demand| > budget`, the run aborts with violation `overload`. `budget = budget_fraction * tau_max_nominal`, where `budget_fraction` is 0.5 for T0 to T2 and 0.7 for T3 (section 6). The check uses the controller's request, not the capped torque, so a motor that saturates early is caught by the request, not by the cap.
2. **Locked.** A locked joint never moves. Its motor applies `tau_max · sign(demand)`: full torque in the direction it is asked to move, and zero when nothing is demanded. This is the stall model: a motor that is asked to move but cannot, draws full current. (Until 2026-10-15 the code applied `+tau_max` whatever the sign of the demand. Every reading uses the absolute torque, so no reading, prediction or level changed; see [testing.md](testing.md), section 7.)
3. **Joint limit.** After the update, if `q` is outside `limits_deg`, the run aborts with violation `joint_limit`.

An aborted run stops updating. Its readings are NaN, and its tier is a fail. A missing sensor (section 4) is never moved at all.

## 3. Joint parameters

Values from `configs/platform_arm6.yaml`. `tau_max` is the healthy motor limit. `g` is the load in the hold skill. The base carries no weight, since it turns about a vertical axis.

| Joint | Chain index | tau_max (N·m) | kp (N·m/rad) | kd (N·m·s/rad) | I (kg·m²) | g (N·m) | Limits (deg) |
|---|---|---|---|---|---|---|---|
| base | 0 | 60 | 1500 | 50 | 0.50 | 0.0 | ±170 |
| shoulder | 1 | 60 | 1500 | 100 | 2.00 | 25.0 | −90 … +90 |
| elbow | 2 | 40 | 1000 | 60 | 1.00 | 12.0 | −120 … +120 |
| wrist_1 | 3 | 20 | 600 | 20 | 0.20 | 1.5 | −160 … +160 |
| wrist_2 | 4 | 15 | 400 | 12 | 0.10 | 0.8 | −160 … +160 |
| wrist_3 | 5 | 10 | 300 | 8 | 0.05 | 0.2 | −180 … +180 |

Platform-wide: `torque_constant_nm_per_a = 5.0` (current = torque / 5), `probe_torque_limit_fraction = 0.5`, `heat_coeff = 0.002`.

**Randomisation.** Learned rollouts and the robot's own runs draw each of `tau_max`, `I`, `g` and the stiffness scale from a uniform distribution within ±10% of nominal (`randomisation.spread = 0.1`). The scripted reference uses nominal values only.

## 4. Damage hypotheses

Defined in `configs/damage.yaml`. Each joint has exactly one state at a time.

| Hypothesis | Parameter | Physical effect | Simulator effect | Prior |
|---|---|---|---|---|
| healthy | — | None | No change | 0.80 |
| locked | `locked: true` | The joint cannot move. The motor pushes at full torque when asked to move. | `q` and `qd` stay at zero. Stall torque is `tau_max`. | 0.05 |
| weak_motor | `torque_scale: 0.4` | Only 40% of rated torque is available. | `tau` is capped at `0.4 * tau_max`. Overload budget unchanged. | 0.05 |
| missing_sensor | `sensor_valid: false` | No encoder reading. | No motion at all. Metrics NaN, `sensor_valid = 0`. | 0.05 |
| broken_link | `stiffness_scale: 0.1` | The position loop is much softer. | `S` term multiplied by 0.1. Damping unchanged. | 0.05 |

The prior sums to 1. It is the belief each joint starts with, and the distribution the true state is drawn from.

## 5. Noise

Two independent noise sources, kept separate on purpose.

| Source | Where | Value | Used by |
|---|---|---|---|
| Input noise | Inside the control loop: what the controller sees | Angle σ = 0.05° (8.7×10⁻⁴ rad); speed σ = 0.5°/s | Learned rollouts (table and robot) and training. Not used by the scripted reference. |
| Reading noise | On each metric, before the tier check | Per check, from `configs/tiers.yaml` (table in section 7) | Scripted pass probability (Gaussian), learned pass probability (sampled), and the robot's readings. |

The scripted reference is deterministic: no input noise, no randomisation. Its pass probability is `Φ(margin / σ)`, where `Φ` is the normal CDF and `margin` is the signed distance to the threshold. For `sensor_valid`, `σ = 0` and the probability is 0 or 1.

## 6. Skills

A skill is the task a tier asks for. Scripted probes and learned policies run the same skill.

| Tier | Kind | Reference `ref(t)` (rad, relative to home) | Duration | Steps | Overload budget | Gravity |
|---|---|---|---|---|---|---|
| T0 | `small_step` | `0.5° ` for all `t` | 0.5 s | 250 | 0.5 · tau_max | none |
| T1 | `smooth_move` | `5° · s(t / 1.0 s)` for `t ≤ 1.0 s`, then `5°` | 2.0 s | 1000 | 0.5 · tau_max | none |
| T2 | `planned_path` | `20° · s(2t/4)` for `t ≤ 2 s`; `20° · s(2(1 − t/4))` after | 4.0 s | 2000 | 0.5 · tau_max | none |
| T3 | `hold_under_load` | `0` for all `t` | 3.0 s | 1500 | **0.7** · tau_max | `g` applied |

`s(x) = 10x³ − 15x⁴ + 6x⁵` for `0 ≤ x ≤ 1`, clamped outside. This is the minimum-jerk profile: zero speed and zero acceleration at both ends.

T2 is an out-and-back path, not a single move, so it tests coordination across the whole swing.

The hold budget is 0.7 rather than 0.5. At 0.5, a nominal shoulder aborts most healthy holds under randomisation, because its 25 N·m load leaves too little headroom for the overshoot (section 9). At 0.7 the healthy hold passes 100% of randomised runs, and the weak-motor shoulder still overloads in 91% of runs. This is a safety setting, flagged for confirmation in [design-decisions.md](design-decisions.md).

## 7. Readings and checks

Each skill produces readings. A tier passes when its checks, applied to its readings, all pass.

| Tier | Metric | Definition | Check | Threshold | Reading σ |
|---|---|---|---|---|---|
| T0 | `sensor_valid` | 1 if the encoder reads, else 0 | min | ≥ 1.0 | 0 |
| T0 | `response_ratio` | `q(T) / 0.5°` at the end of the step | min | ≥ 0.5 | 0.05 |
| T0 | `peak_current_a` | `max over alive steps of abs(tau) / 5.0` | max | ≤ 6.0 A | 0.1 |
| T1 | `final_error_deg` | `abs(ref(T) − q(T))` in degrees | max | ≤ 1.0° | 0.05 |
| T2 | `rms_error_deg` | RMS of `ref − q` over the path, in degrees | max | ≤ 2.0° | 0.05 |
| T3 | `sag_deg` | Max of `abs(ref − q)` over the last 1.0 s, in degrees | max | ≤ 1.5° | 0.05 |
| T3 | `peak_torque_fraction` | `max over alive steps of abs(tau) / tau_max` | max | ≤ 0.8 | 0.01 |

A run that aborts gives NaN for every reading, so its checks fail. A missing sensor gives `sensor_valid = 0` and nothing else.

## 8. Scripted readings

The reference controller's readings for every joint and hypothesis, measured from the code (noise-free). Columns: response (T0), peak current (T0, A), final error (T1, deg), RMS error (T2, deg), sag (T3, deg), peak torque fraction (T3), and any violation that stops the run.

| Joint | Hypothesis | Response | Current (A) | Final err (°) | RMS (°) | Sag (°) | Peak τ frac | Violation |
|---|---|---|---|---|---|---|---|---|
| base | healthy | 1.00 | 2.62 | 0.00 | 0.37 | 0.00 | 0.000 | — |
| base | locked | 0.00 | 12.00 | — | — | 0.00 | 0.000 | overload at T1, T2 |
| base | weak_motor | 1.00 | 2.62 | 0.00 | 0.37 | 0.00 | 0.000 | — |
| base | missing_sensor | — | — | — | — | — | — | none (no motion) |
| base | broken_link | 0.78 | 0.26 | 0.06 | 3.52 | 0.00 | 0.000 | — |
| shoulder | healthy | 1.00 | 2.62 | 0.00 | 0.77 | 0.95 | 0.481 | — |
| shoulder | locked | 0.00 | 12.00 | — | — | 0.00 | 0.000 | overload at T1, T2 |
| shoulder | weak_motor | 1.00 | 2.62 | 0.00 | 0.77 | — | — | **overload at T3** |
| shoulder | missing_sensor | — | — | — | — | — | — | none (no motion) |
| shoulder | broken_link | 0.52 | 0.26 | 0.53 | 5.70 | 9.45 | 0.427 | — |
| elbow | healthy | 1.00 | 1.75 | 0.00 | 0.69 | 0.69 | 0.344 | — |
| elbow | locked | 0.00 | 8.00 | — | — | 0.00 | 0.000 | overload at T1, T2 |
| elbow | weak_motor | 1.00 | 1.75 | 0.00 | 0.69 | 0.69 | 0.344 | — |
| elbow | missing_sensor | — | — | — | — | — | — | none (no motion) |
| elbow | broken_link | 0.56 | 0.17 | 0.42 | 5.35 | 6.83 | 0.307 | — |
| wrist_1 | healthy | 1.00 | 1.05 | 0.00 | 0.37 | 0.14 | 0.087 | — |
| wrist_1 | locked | 0.00 | 4.00 | — | — | 0.00 | 0.000 | overload at T1, T2 |
| wrist_1 | weak_motor | 1.00 | 1.05 | 0.00 | 0.37 | 0.14 | 0.087 | — |
| wrist_1 | missing_sensor | — | — | — | — | — | — | none (no motion) |
| wrist_1 | broken_link | 0.78 | 0.10 | 0.06 | 3.52 | 1.43 | 0.077 | — |
| wrist_2 | healthy | 1.00 | 0.70 | 0.00 | 0.33 | 0.11 | 0.061 | — |
| wrist_2 | locked | 0.00 | 3.00 | — | — | 0.00 | 0.000 | overload at T1, T2 |
| wrist_2 | weak_motor | 1.00 | 0.70 | 0.00 | 0.33 | 0.11 | 0.061 | — |
| wrist_2 | missing_sensor | — | — | — | — | — | — | none (no motion) |
| wrist_2 | broken_link | 0.81 | 0.07 | 0.04 | 3.23 | 1.15 | 0.055 | — |
| wrist_3 | healthy | 1.00 | 0.52 | 0.00 | 0.29 | 0.04 | 0.023 | — |
| wrist_3 | locked | 0.00 | 2.00 | — | — | 0.00 | 0.000 | overload at T1, T2 |
| wrist_3 | weak_motor | 1.00 | 0.52 | 0.00 | 0.29 | 0.04 | 0.023 | — |
| wrist_3 | missing_sensor | — | — | — | — | — | — | none (no motion) |
| wrist_3 | broken_link | 0.85 | 0.05 | 0.02 | 2.92 | 0.38 | 0.020 | — |

Reading the table:

- **Locked joints fail T0 first.** The response is zero on every joint. The stall current exceeds the 6 A limit on the base, shoulder and elbow (12, 12 and 8 A). On wrist_1 to wrist_3 (4, 3 and 2 A) the zero response alone is the failure. The locked base shows a zero sag on T3, which would pass, but T0 gates it, so T3 is never run.
- **Weak motor shows only on the shoulder.** The shoulder's 25 N·m load exceeds 0.4 × 60 = 24 N·m, so the hold overloads. The elbow's 12 N·m load is under 0.4 × 40 = 16 N·m, and every other joint carries less still.
- **Broken link shows on T2.** The final error after a 5° move is small (0.02–0.53°), so T1 passes on every joint. The RMS path error is 2.9–5.7°, so T2 fails everywhere.
- **Missing sensor is never moved.** Its readings are NaN and its energy is zero.
- **Hold sag scales with g / (kp · scale).** For the healthy shoulder, `25 / 1500 rad = 0.955°`, which matches the table.

## 9. Probe budget and headroom

The budget is a safety limit, so its headroom matters. For the healthy shoulder hold:

| Budget | Budget (N·m) | Headroom over g = 25 | Headroom over g = 27.5 (randomised max) | Healthy hold pass rate |
|---|---|---|---|---|
| 0.5 · 60 | 30 | 17% | 8% | 16% (84% aborts) |
| 0.6 · 60 | 36 | 31% | 24% | 99% |
| 0.7 · 60 | 42 | 40% | 35% | 100% |
| 0.8 · 60 | 48 | 48% | 43% | 100% |

Measured with nominal gains, 300 randomised rollouts per row. The weak-motor shoulder still fails 90–95% of holds at 0.6 to 0.8. The overshoot on the healthy shoulder reaches 30.06 N·m against a 26.6 N·m load, a 13% overshoot, which is what exhausts the 50% budget.

## 10. Known simplifications

These are stated again in [robotics-limitations.md](robotics-limitations.md), with what they change on hardware.

- No friction, backlash, encoder quantisation or latency.
- Gravity is one constant value per joint, at one hold pose.
- The weak motor is a torque cap. A real weak motor (demagnetised or underpowered) has a lower torque constant, so it draws more current per N·m. The current check here therefore understates it.
- The locked model draws stall current whenever the demand is non-zero, regardless of whether the joint is actually stalled.
- Each joint is simulated alone. Loads from distal joints are represented only by `g`.
