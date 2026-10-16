# Documentation

These documents describe the stage-1 code: a simulated 6-DOF arm, with each joint assessed through tiers T0 to T3, by scripted probes and by learned policies.

## Reading order

Read these in order if you are new to the project. Each one builds on the one before.

| # | Document | Read it to learn |
|---|---|---|
| 1 | [architecture.md](architecture.md) | How the code is split into layers, what each module does, the data passed between them, and how a run is seeded and timed. |
| 2 | [model.md](model.md) | The simulated robot exactly: the equations, joint parameters, damage hypotheses, skills, readings, and the scripted numbers. |
| 3 | [learning.md](learning.md) | What a learned policy is, how it is trained by evolution strategies, how a policy is trusted, and what training produced. |
| 4 | [assessment.md](assessment.md) | The results: performance signatures, agreement with the reference, identifiability, and the planning comparison. |
| 5 | [testing.md](testing.md) | What each of the 125 tests guards, what is not tested, the tested environment, and how to run them. |
| 6 | [robotics-limitations.md](robotics-limitations.md) | What the simulation leaves out, what each omission changes on a real robot, and the hardware preconditions. |
| 7 | [design-decisions.md](design-decisions.md) | Every decision, its value, its alternatives, its reason, and which ones need confirmation. |
| 8 | [stage1-plan.md](stage1-plan.md) | The original plan, its status, the deviations from it, and the robotics review. |

## Status

- **Stage 1 (this version): implemented and run.** Learned skills are trained for every joint and tier T1 to T3. Assessment and planning comparisons are complete.
- **Not yet implemented:** a whole-arm model, a sensor-only check, an ES-versus-grid test, friction and latency, a richer policy class, and any legged platform. These are listed in [stage1-plan.md](stage1-plan.md), section 8.
- **Not yet confirmed by the team:** the items in [design-decisions.md](design-decisions.md), section 10.

## Conventions

- Every number in these documents is taken from a file in the repository, either a config in `configs/` or a result in `results/`. Where a number is computed from the code, the command that computes it is given.
- A result that depends on a random seed is reproducible, because every stream is seeded (see [architecture.md](architecture.md), section 6).
- Where the code and an older document disagree, the code is right, and the document is corrected. The corrections are recorded in [stage1-plan.md](stage1-plan.md), section 3.
- Stage-1 results are simulation results. [robotics-limitations.md](robotics-limitations.md) says what that does and does not establish.

## Where the results are

| Result | File | Regenerate with |
|---|---|---|
| Trained policies | `results/policies/arm6/*.json` | `python -m experiments.train_policies --workers 4` |
| Assessment (signatures, agreement, identifiability) | `results/assessment_arm6.md` | `python -m experiments.assess_policies` |
| Comparison, learned | `results/comparison_learned.csv` | `python -m experiments.compare_methods ...` (about 21 minutes) |
| Comparison, scripted | `results/comparison_chain.csv`, `results/comparison_robot.csv` | `python -m experiments.compare_methods ... --assessment scripted` |
| Rerun of 2026-10-15 (belief scores, paired intervals, per-trial rows; reported in [assessment.md](assessment.md), section 8) | `results/rerun_2026-10-15/` | The commands in its `MANIFEST.md` |

The `results/` folder is git-ignored, so the trained policies are regenerated on each machine rather than shared through the repository.
