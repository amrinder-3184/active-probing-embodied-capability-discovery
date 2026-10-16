"""Active probing for embodied capability discovery: core implementation.

Modules, in the order the layers build on each other:
  config            locate and read the YAML files in configs/
  tiers             what each tier checks and what counts as a pass
  damage            damage hypotheses and their prior
  platform_adapter  joints, chains, skills and assessors of one robot
  physics           batched single-joint dynamics (one model for probes and policies)
  tasks             skills: reference motion and readings for each tier
  policies          control law (two gains) and the policy store
  runner            run one skill on a batch of joints; reading noise
  training          evolution strategies and the healthy gate
  assessors         scripted reference and learned policies, one interface
  outcomes          table of predictions per joint, tier and hypothesis
  pipeline          wires assessors from the platform config and the policy store
  belief            Bayes update and expected information gain
  gating            which tier tests are allowed now, and each tier's status
  planner           probe ordering: the proposed method and the baselines
  episode           one probing run on a simulated robot with hidden damage
  report            the physical state report (uses no hidden truth)
  evaluate          compares a report with the hidden truth (evaluation only)
"""
