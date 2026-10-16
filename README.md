# Active Probing for Embodied Capability Discovery

## Overview

This project investigates how a robot can actively discover its remaining capabilities after experiencing damage, degradation, or changes in its physical condition.

Instead of assuming that the robot's original capabilities remain unchanged, the system allows the robot to perform carefully selected probing actions and use the resulting observations to update its understanding of what it can and cannot do.

The project focuses on capability discovery as an active decision-making problem, where the robot must balance the expected information gained from an experiment against factors such as physical risk, energy consumption, and execution time.

## Problem Statement

Robots operating in real-world environments may experience actuator failures, sensor degradation, mechanical damage, or other changes that alter their capabilities.

A robot that relies only on its original capability model may attempt actions that are no longer feasible or may fail to exploit capabilities that remain available after damage.

The goal of this project is to develop an approach that enables a robot to actively probe its environment and body to discover its remaining capabilities. The robot should select informative experiments, observe their outcomes, and update its capability representation based on the acquired evidence.

## Objectives

- Represent the capabilities available to a robotic system.
- Model uncertainty about capabilities after damage or degradation.
- Select probing actions that provide useful information about uncertain capabilities.
- Incorporate factors such as expected information gain, physical risk, energy consumption, and execution time.
- Update the robot's capability model using observations from executed probes.
- Evaluate whether active probing can improve capability identification compared with passive or fixed testing strategies.

## High-Level Approach

The planned workflow is:

1. Define a hierarchical representation of robot capabilities.
2. Introduce simulated damage or capability degradation.
3. Identify capabilities whose status is uncertain.
4. Generate candidate probing actions.
5. Estimate the expected utility of each probe.
6. Select and execute an informative probe.
7. Observe the outcome.
8. Update the capability model.
9. Repeat until sufficient information about the robot's capabilities has been obtained.
10. Evaluate the resulting capability-discovery performance.

## Current Implementation

The current version runs in simulation on a 6-DOF arm. Each part is a joint, and each joint is assessed through the tiers T0 to T3.

- **Tier logic** (`src/tiers.py`): each tier is a set of checks with thresholds from `configs/tiers.yaml`.
- **Platform adapter** (`src/platform_adapter.py`, `configs/platform_arm6.yaml`): joints, chains, limits, skills and which assessor judges each tier. A new robot is a new YAML file.
- **Physics** (`src/physics.py`): one batched single-joint model. Scripted probes and learned policies run on it.
- **Skills** (`src/tasks.py`): the reference motion and readings for each tier's task.
- **Policies** (`src/policies.py`): a stage-1 policy is two gain multipliers per joint and skill.
- **Training** (`src/training.py`): evolution strategies on randomised healthy joints, plus a healthy gate.
- **Assessors** (`src/assessors.py`): the scripted reference and the learned policies, both with one interface.
- **Damage hypotheses** (`src/damage.py`, `configs/damage.yaml`): healthy, locked, weak motor, missing sensor, broken link.
- **Belief and planner** (`src/belief.py`, `src/planner.py`): a Bayesian belief per joint. Skills are ordered by expected information gain minus weighted risk and energy, with a safety veto.
- **Gating** (`src/gating.py`): a tier is tested only after the tiers below it pass, within the same chain. The whole-robot baseline is also available (`gating: robot`).
- **Report** (`src/report.py`): the physical state report, as Markdown and JSON.
- **Evaluation** (`src/evaluate.py`): hypothesis accuracy, level against the reference, and overclaim. The planner never sees the truth.

The plan and the robotics review are in [`docs/stage1-plan.md`](docs/stage1-plan.md). Design choices and their defaults are in [`docs/design-decisions.md`](docs/design-decisions.md). Items marked **Confirm** are open.

### Running

Requires Python 3.11 or newer with `venv`:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run the tests:

```bash
.venv/bin/python -m pytest
```

The steps, in order:

```bash
# 1. Teach each skill to a healthy joint (about 20 seconds with 4 workers)
.venv/bin/python -m experiments.train_policies --workers 4

# 2. Run each policy on each damage case and check it against the reference
.venv/bin/python -m experiments.assess_policies

# 3. One episode, with its physical state report
.venv/bin/python -m experiments.run_episode --method planner --seed 1

# 4. Compare the methods over many seeds
.venv/bin/python -m experiments.compare_methods --seeds 100 --budgets 12 24 40
```

Add `--assessment scripted` to steps 3 and 4 to use the reference controller only. Results go to `results/`, which is git-ignored except for `.gitkeep`.

On Windows, call the venv's Python directly; this works without activating the venv, so the PowerShell execution policy does not need changing:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m experiments.train_policies --workers 4
```

Options for evaluation and reproducibility (all optional; the defaults behave as before):

```bash
# per-trial rows, paired differences against fixed with bootstrap intervals, and the coverage ceiling
.venv/bin/python -m experiments.compare_methods --seeds 100 --budgets 12 24 40 \
    --trials-out results/trials.csv --paired-out results/paired.csv --ceiling
# reproduce results produced before 2026-10-15 (one shared noise stream instead of one per probe)
.venv/bin/python -m experiments.compare_methods --noise-streams shared
# assess policies from another folder, and write the report elsewhere
.venv/bin/python -m experiments.assess_policies --policies <dir> --out <file.md>
# compare with policies from a given folder, on seeds 100 to 299 instead of 0 to 199
.venv/bin/python -m experiments.compare_methods --policies <dir> --seed-start 100 --seeds 200
# every metric after each probe (k = 0 .. 24), plus probing time and safety events, from one run
.venv/bin/python -m experiments.compare_methods --budgets 24 --curve-out results/curve.csv
```

The tested environment and pinned versions are in [`docs/testing.md`](docs/testing.md), section 6. The metric conventions are in [`docs/assessment.md`](docs/assessment.md), section 8.

### Not yet implemented

- A whole-arm environment, with the joints interacting. This is the next step.
- A richer policy class (an MLP), and PPO.
- Friction, backlash, latency, and a weak motor modelled by its current.
- A stiffness probe at several poses, for a robot with gravity compensation.
- T4 (whole-body posture) and any legged platform.
- A shared-damage belief (RQ3, second half).
- A probe that reads the sensor without moving the joint, to separate a locked joint from a missing sensor.

## Project Structure

```text
g11-active-probing/
│
├── README.md
├── requirements.txt
├── pytest.ini
├── .gitignore
│
├── configs/                  tiers, platform, damage hypotheses, planner, training
│   ├── tiers.yaml
│   ├── platform_arm6.yaml
│   ├── damage.yaml
│   ├── planner.yaml
│   └── training.yaml
│
├── src/                      core implementation (module list in src/__init__.py)
│
├── experiments/
│   ├── train_policies.py     train and gate every learned skill
│   ├── assess_policies.py    signature, identifiability and agreement with the reference
│   ├── run_episode.py        one episode and its physical state report
│   └── compare_methods.py    methods compared over seeds and budgets
│
├── tests/                    tiers, physics, probes, policies, training, assessors, planner, episodes
│
├── results/                  experiment outputs (git-ignored)
│
└── docs/
    ├── stage1-plan.md        plan for learned skills, with the robotics review
    └── design-decisions.md   open choices and their defaults
```
