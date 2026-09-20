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

## Project Structure

```text
active-probing-embodied-capability-discovery/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   └── Core implementation
│
├── experiments/
│   └── Experimental scripts
│
├── configs/
│   └── Experiment configurations
│
├── results/
│   └── Experimental results
│
└── docs/
    └── Project documentation and literature notes
