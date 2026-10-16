"""One probing episode: a simulated robot with hidden damage, probed by one method.

Random streams are kept separate, so every method sees the same damage for the same seed:
  seed       -> the true damage (python random)
  robot runs -> plant randomisation and noise (numpy), one stream per probe:
                SeedSequence([seed, TRUTH_STREAM, joint index, tier index, repeat]).
                The r-th run of a probe gets the same numbers whatever order a method runs it in,
                so two methods that run the same probes see the same outcomes (paired comparison).
                noise_streams="shared" restores the stage-1 behaviour, one stream default_rng(seed + 1)
                used in probe order, to reproduce results produced before this change.
  seed + 2   -> the random baseline's choices (python random)
"""
import dataclasses
import random
from dataclasses import dataclass

import numpy as np

from .belief import update
from .damage import sample_fault_name
from .gating import State
from .planner import choose
from .platform_adapter import usable_tiers

NOISE_STREAMS = ("per_probe", "shared")
TRUTH_STREAM = 1  # tag in the per-probe seed key, so robot runs never share a key with other streams


@dataclass
class ProbeRecord:
    step: int
    joint: str
    tier: str
    passed: bool
    aborted: bool
    violations: tuple
    energy_j: float
    duration_s: float = 0.0   # time the skill actually ran (shorter than the skill if it aborted)


@dataclass
class Episode:
    state: State
    log: list
    true_states: dict         # joint -> hypothesis actually present. For evaluation only.
    reference_levels: dict    # joint -> {hypothesis: reference level}. For evaluation only.
    method: str
    seed: int
    budget: int


class SimulatedRobot:
    """Hidden damage, run through the assessors. The planner only sees the results of its probes."""

    def __init__(self, platform, tiers, assessors, damage, damage_rng, rollout_rng, joint_names,
                 probe_seed=None):
        """rollout_rng: one shared stream used in probe order (stage-1 behaviour).
        probe_seed: if given, each run gets its own stream keyed by (probe_seed, joint, tier, repeat),
        and rollout_rng is not used.
        """
        self.platform = platform
        self.tiers = tiers
        self.assessors = assessors
        self.rollout_rng = rollout_rng
        self.probe_seed = probe_seed
        self.joint_index = {name: i for i, name in enumerate(joint_names)}
        self.true_states = {name: sample_fault_name(damage, damage_rng) for name in joint_names}
        self.faults = damage.faults

    def rng_for(self, joint_name, ti, repeat):
        if self.probe_seed is None:
            return self.rollout_rng
        key = [self.probe_seed, TRUTH_STREAM, self.joint_index[joint_name], ti, repeat]
        return np.random.default_rng(np.random.SeedSequence(key))

    def run(self, joint_name, ti, repeat=0):
        """Run tier ti on the joint. repeat counts earlier runs of the same probe in this episode."""
        tier = self.tiers[ti]
        fault = self.faults[self.true_states[joint_name]]
        assessor = self.assessors[tier.name]
        return assessor.execute(self.platform.joint(joint_name), tier, fault,
                                self.rng_for(joint_name, ti, repeat))


def run_episode(platform, tiers, damage, table, method, cfg, seed, noise_streams="per_probe"):
    if noise_streams not in NOISE_STREAMS:
        raise ValueError(f"noise_streams must be one of {NOISE_STREAMS}, not {noise_streams!r}")
    usable = usable_tiers(platform, tiers)
    names = tuple(j.name for j in platform.joints)
    tier_index = {t.name: i for i, t in enumerate(usable)}
    untrusted = frozenset(
        (joint, tier_index[tier])
        for (joint, tier), trusted in table.trusted.items()
        if not trusted and tier in tier_index
    )
    shared = noise_streams == "shared"
    robot = SimulatedRobot(platform, usable, table.assessors, damage,
                           damage_rng=random.Random(seed),
                           rollout_rng=np.random.default_rng(seed + 1) if shared else None,
                           joint_names=names,
                           probe_seed=None if shared else seed)
    policy_rng = random.Random(seed + 2)

    state = State(
        joint_names=names,
        proximal={n: platform.proximal(n) for n in names},
        tiers=usable,
        levels={n: table.levels[n] for n in names},
        beliefs={n: dict(damage.prior) for n in names},
        gating=cfg.gating,
        confidence=cfg.pass_confidence,
        max_repeats=cfg.max_repeats,
        untrusted=untrusted,
    )

    log = []
    for step in range(cfg.budget):
        candidates = [(n, ti) for n in names for ti in range(len(usable))
                      if state.status(n, ti) == "ready"]
        choice = choose(method, state, candidates, table, cfg, policy_rng)
        if choice is None:
            break
        joint, ti = choice
        repeat = state.probe_counts.get((joint, ti), 0)
        execution = robot.run(joint, ti, repeat)
        state.probe_counts[(joint, ti)] = repeat + 1

        tier = usable[ti]
        pass_prob = {h: table.get(joint, tier.name, h).pass_prob for h in state.beliefs[joint]}
        state.beliefs[joint] = update(state.beliefs[joint], pass_prob, execution.passed, cfg.likelihood_floor)
        log.append(ProbeRecord(
            step=step, joint=joint, tier=tier.name, passed=execution.passed,
            aborted=bool(execution.violations), violations=tuple(sorted(execution.violations)),
            energy_j=execution.energy_j, duration_s=execution.duration_s,
        ))

    return Episode(state=state, log=log, true_states=robot.true_states,
                   reference_levels=table.reference_levels,
                   method=method, seed=seed, budget=cfg.budget)


def prefix_episodes(episode, table, damage, cfg):
    """The episode as it stood after each of its first k probes, for k = 0 .. episode.budget.

    No method's choice depends on the budget (it only bounds the loop in run_episode), and the
    belief update is deterministic given the logged outcomes. So the state after k probes of this
    episode is the state a run with budget k would end in: replaying the first k log records from
    the prior, with the same table and likelihood floor, reproduces it. Past the end of the log,
    every further prefix is the whole episode.
    """
    final = episode.state
    tier_index = {t.name: i for i, t in enumerate(final.tiers)}
    state = dataclasses.replace(final, beliefs={n: dict(damage.prior) for n in final.joint_names},
                                probe_counts={})
    for k in range(episode.budget + 1):
        if 0 < k <= len(episode.log):
            r = episode.log[k - 1]
            ti = tier_index[r.tier]
            state.probe_counts[(r.joint, ti)] = state.probe_counts.get((r.joint, ti), 0) + 1
            pass_prob = {h: table.get(r.joint, r.tier, h).pass_prob for h in state.beliefs[r.joint]}
            state.beliefs[r.joint] = update(state.beliefs[r.joint], pass_prob, r.passed, cfg.likelihood_floor)
        snapshot = dataclasses.replace(state, beliefs={n: dict(b) for n, b in state.beliefs.items()},
                                       probe_counts=dict(state.probe_counts))
        yield Episode(state=snapshot, log=episode.log[:k], true_states=episode.true_states,
                      reference_levels=episode.reference_levels, method=episode.method,
                      seed=episode.seed, budget=k)
