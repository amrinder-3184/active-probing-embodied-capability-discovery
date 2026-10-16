# Robotics limitations and hardware preconditions

This document lists what the simulation leaves out, what each omission does to the results on a real robot, and what must be true before any learned policy is run on hardware. Stage 1 is **simulation only**. Nothing in this repository drives a motor.

## 1. Summary

The simulation reproduces the tier logic, the gating, the Bayesian inference and the learned controller on one joint at a time. It does not reproduce friction, backlash, latency, encoder quantisation, the joint-to-joint coupling of a real arm, or the failure behaviour of real motors and brakes. Each omission changes the numbers. Some change the conclusions. This document says which.

## 2. Modelling assumptions and their consequences

| # | Assumption in the code | What a real robot does | Consequence | Severity |
|---|---|---|---|---|
| A1 | No friction. | Coulomb and viscous friction on every joint. | A steady error of about friction/kp appears, which looks like a softer joint. It is not in the hypothesis set, so it would be read as a broken link on T2 or as a sag on T3. | High |
| A2 | No backlash or compliance in the gears. | Gear backlash and joint compliance. | Small dead zones at reversal. Stiffness varies with load and pose, so the stiffness fault is not a single scale. | Medium |
| A3 | Controller acts every 2 ms, with no delay. | A servo loop at 1 to 10 kHz, but sensing and communication add 1 to 5 ms of latency. | A high-gain loop can go unstable with latency. The sim's gains would be too aggressive. | High |
| A4 | Speed is known with Gaussian noise (0.5°/s). | Speed is differenced from an encoder. At 500 Hz a 14-bit encoder gives about 11°/s per count. | The noise is not Gaussian, and it is much larger than modelled. The learned damping term would react to quantisation. A filtered velocity estimate is needed. | High |
| A5 | Gravity is one constant value per joint, at one hold pose. | Gravity torque depends on the whole chain and the pose. | The hold test (T3) checks one pose. A joint that is weak at other poses passes. | High |
| A6 | Weak motor is a torque cap at 40%. | A demagnetised or underpowered motor has a lower torque constant, so it draws more current per N·m. | The peak-current check understates the fault. The current threshold would not catch it. | Medium |
| A7 | Locked joint draws full stall torque whenever it is asked to move. | A stalled motor draws stall current, but a merely slow motor does not. Real locks depend on the mechanism (brake, pin, stiff drive), and each can look different on the current and response checks. | A slow joint could be read as locked, and a real lock might not match the sim's stall signature. | Medium |
| A8 | Missing sensor means the joint is never moved. | The controller may still move the joint open-loop, or the safety system may refuse. Depends on the robot. | The sim is conservative. A real robot might move a joint that the sim never moves. | Low |
| A9 | Each joint is simulated alone. Loads from distal joints are a fixed `g`. | Distal joints load proximal ones, and the loads change with pose. | The chain-gating rule assumes proximal joints are healthy while a distal joint is tested. On a real arm a fault upstream changes the loads downstream. | High |
| A10 | Stiffness fault scales only the position term. | A softer position loop also changes the damping and the phase margin. | The broken-link signature may differ from the real one: the real joint may oscillate rather than just lag. | Medium |
| A11 | No thermal model. The energy proxy is an integral with a fixed heat coefficient. | Motors heat up over repeated probes, and their torque limit falls. | Repeated probing can overheat a joint. The sim has no budget for heat. | Medium |
| A12 | Learned policies are trained and tested on the same simulator. | The policy meets a different plant. | The gate tests robustness to ±10% parameter spread, not to real modelling error. A trusted policy can still fail on hardware. | High |

## 3. The hold test and gravity compensation

The T3 hold measures `sag = g / (kp · scale)`. That relation holds only when the joint has no model-based gravity compensation. A robot that adds a gravity feedforward from its dynamic model holds the load with near-zero error, whatever the stiffness. The hold then passes on a broken link.

The stage-1 controller has no feedforward, so the test is valid for it. A real robot with compensation needs a different probe, for example the torque needed to hold at several poses, or a step response under a known load. The choice is open (see the gravity-compensation row in section 10 of [design-decisions.md](design-decisions.md)).

## 4. Safety

| # | Hazard | Why it matters | What the code does now | What hardware needs |
|---|---|---|---|---|
| S1 | A hold on a loaded shoulder releases if the torque is cut. | A 25 N·m load drops the arm. | Stops the simulated run, and the joint keeps its last state. On hardware, a drive that stops applying torque lets a loaded joint fall. | On abort: disable the drive and engage the brake or a support. Never cut torque alone on a loaded joint. |
| S2 | A sign-flipped or offset encoder turns the position loop into positive feedback. | The arm accelerates away from the command. | Not modelled. A sign flip would show as a negative response ratio on T0, but only once encoder faults are simulated. | Before any learned run: a low-torque direction check that confirms the joint moves in the commanded direction. |
| S3 | A learned policy asks for more torque than the joint can deliver safely. | Overload heats or damages the motor and gearbox. | The overload check uses the controller's request against the skill's budget, before the motor cap. Gains are clamped. | Hard torque and current limits in the drive, below the mechanical limit. The software budget is a second line, not the only one. |
| S4 | A stiffer policy hides a softer joint. | Damage goes unnoticed and the joint is used at full load. | The stiffness multiplier is capped at 1.0. | None beyond the clamp. Keep the clamp under review when richer policies are added. |
| S5 | A fault during a probe is not stopped in time. | Joint limits or a collision. | Joint limit aborts the simulated run after the step that exceeds it. | A hardware end-stop and a collision check, independent of software. |
| S6 | Repeated probing overheats a joint. | A joint that was fine on the first probe fails on the tenth. | Not modelled. | Temperature monitoring, and a probe-per-joint limit per session. |
| S7 | An emergency stop during a run. | The run must end safely. | Not modelled. | An emergency stop that disables all drives and engages all brakes, tested before any learned run. |

## 5. Preconditions for hardware

No learned policy should run on a robot until every item below is true and tested. They are ordered by how early they are needed.

1. **Emergency stop works** and removes drive power and engages brakes. Tested on the robot with no load.
2. **Encoder direction and zero are checked** at low torque on each joint (S2).
3. **Drive limits** for torque, current and speed are set below mechanical limits (S3).
4. **Abort behaviour** on each joint is defined: drive disable plus brake, or support. Verified on a loaded joint (S1).
5. **Joint limits** are set in the drive as well as in software (S5).
6. **Temperature monitoring** on each joint, with a session limit on probes (S6).
7. **Velocity estimate** is filtered, and its noise is measured on the real encoder. Replace A4 with that measurement.
8. **Latency** is measured and included in the control loop, and the gains are checked for stability with it (A3).
9. **Friction** is identified per joint and added to the simulator (A1), and the gate is re-run.
10. **Gravity** is measured at each hold pose the tests use (A5).

## 6. What the simulation can and cannot tell us

**Can tell us:**

- That the tier logic, gating, belief update and planner behave as specified, on a model of one joint.
- Which damage states are separable with the tests as designed, given the model (the identifiability analysis).
- That a learned controller can reach the reference performance on healthy joints in the model, and where it falls short.

**Cannot tell us:**

- Whether a real joint would show the same signature under a given fault. A1 to A10 change the signatures.
- Whether a learned policy is safe on a real robot. S1 to S7 are not modelled.
- How the belief and planner behave when the model is wrong. There is no model-error term in the likelihood, only a floor of 0.01 per hypothesis.
- Whether the thresholds in `configs/tiers.yaml` are the right ones for a real robot. They are hand-picked.

## 7. Items from the robotics review and their status

The plan's robotics review (see [stage1-plan.md](stage1-plan.md), section 6) raised these points. Their status now:

| Review point | Status |
|---|---|
| T3 measures kp × gravity; compensation would hide stiffness loss | Stated (section 3). Open: see section 10 of design-decisions.md. |
| gravity_nm is one number at one pose | Stated (A5). Not fixed. |
| Friction missing | Stated (A1). Not fixed. |
| No latency; velocity from a differenced encoder | Stated (A3, A4). Not fixed. |
| Weak motor is a hard clip, not a Kt effect | Stated (A6). Not fixed. |
| Abort must mean drive disable and brake | Stated (S1). Hardware precondition 4. |
| Sign-flipped encoder | Stated (S2). Hardware precondition 2. |
| "Penalised" limits are not limits | Fixed: gains are hard-clamped (S3, S4). |
| Locked joint on T3 | Corrected in the plan (locked joints fail T0 first). |
| Feedforward in the policy | Fixed: the policy has none. |
