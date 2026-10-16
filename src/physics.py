"""Batched single-joint dynamics: the one physics model behind scripted probes and learned policies.

Every array has shape (n,), one entry per simulated joint. One joint is a batch of size 1.
Batching lets training score a whole population of candidate policies in one pass.

Control law: the controller supplies two torque terms, a stiffness term (on the position error)
and a damping term (on the speed). The demanded torque is stiffness_scale * stiffness_term +
damping_term, so a stiffness fault weakens only the position feedback, as a softer position loop would.

Safety: a demand above the probe budget aborts the run (overload). Leaving the joint's range aborts
it (joint_limit). An aborted joint stops updating and keeps its last state.
"""
from dataclasses import dataclass

import numpy as np

DT = 0.002  # seconds per step: a 500 Hz servo loop


@dataclass(frozen=True)
class Plant:
    """Physical parameters of a batch of joints."""
    tau_max: np.ndarray          # N*m, healthy motor limit
    inertia: np.ndarray          # kg*m^2
    gravity: np.ndarray          # N*m, load pulling the joint down (zero unless the skill holds a load)
    low: np.ndarray              # rad, joint range relative to home
    high: np.ndarray
    torque_scale: np.ndarray     # fault: fraction of tau_max the motor can deliver
    stiffness_scale: np.ndarray  # fault: fraction of the position-feedback stiffness
    locked: np.ndarray           # fault: the joint cannot move (bool)

    @property
    def size(self):
        return len(self.tau_max)


@dataclass
class Rollout:
    ref: np.ndarray        # (T, n) commanded angle, rad
    q: np.ndarray          # (T, n) angle after each step, rad
    tau: np.ndarray        # (T, n) torque applied, N*m
    demand: np.ndarray     # (T, n) torque asked for, before the motor cap, N*m
    alive: np.ndarray      # (T, n) bool, the step was executed
    violations: dict       # name -> (n,) bool, one entry per violation type
    energy_j: np.ndarray   # (n,) energy proxy spent before the end or the abort

    @property
    def aborted(self):
        return np.logical_or.reduce(list(self.violations.values()))


def make_plant(joint, fault, n, rng=None, spread=0.0, gravity_on=False):
    """Plant for n copies of `joint` with `fault`.

    spread > 0 draws each physical parameter uniformly within +/- spread of its nominal value
    (domain randomisation). spread = 0 gives the nominal joint, with no random draws.
    `fault` needs torque_scale, stiffness_scale and locked attributes (see damage.py).
    """
    def draw(nominal):
        if spread == 0.0:
            return np.full(n, float(nominal))
        return float(nominal) * rng.uniform(1.0 - spread, 1.0 + spread, n)

    low, high = (np.radians(x) for x in joint.limits_deg)
    return Plant(
        tau_max=draw(joint.tau_max_nm),
        inertia=draw(joint.inertia),
        gravity=draw(joint.gravity_nm) if gravity_on else np.zeros(n),
        low=np.full(n, low),
        high=np.full(n, high),
        torque_scale=np.full(n, float(fault.torque_scale)),
        stiffness_scale=draw(1.0) * float(fault.stiffness_scale) if spread else np.full(n, float(fault.stiffness_scale)),
        locked=np.full(n, bool(fault.locked)),
    )


def simulate(plant, budget, controller, reference, n_steps, heat_coeff,
             obs_noise=None, rng=None, dt=DT):
    """Run the batch for n_steps.

    controller.torques(err, qd) -> (stiffness_term, damping_term), each of shape (n,).
      err is ref - q, qd is speed, both as the controller sees them (after any input noise).
    reference(t) -> commanded angle in rad, the same for every joint in the batch.
    budget: (n,) probe torque budget. A demand above it aborts the run.
    obs_noise: None, or (sigma_angle_rad, sigma_speed_rad_per_s) added to what the controller sees.
    """
    n = plant.size
    cap = plant.tau_max * plant.torque_scale
    q = np.zeros(n)
    qd = np.zeros(n)
    alive = np.ones(n, dtype=bool)
    overload = np.zeros(n, dtype=bool)
    joint_limit = np.zeros(n, dtype=bool)
    energy = np.zeros(n)

    ref_h = np.empty((n_steps, n))
    q_h = np.empty((n_steps, n))
    tau_h = np.empty((n_steps, n))
    demand_h = np.empty((n_steps, n))
    alive_h = np.empty((n_steps, n), dtype=bool)

    for k in range(n_steps):
        ref = np.full(n, float(reference(k * dt)))
        ref_h[k] = ref

        q_seen, qd_seen = q, qd
        if obs_noise is not None:
            sigma_q, sigma_qd = obs_noise
            q_seen = q + rng.normal(0.0, sigma_q, n)
            qd_seen = qd + rng.normal(0.0, sigma_qd, n)
        stiffness_term, damping_term = controller.torques(ref - q_seen, qd_seen)
        demand = plant.stiffness_scale * stiffness_term + damping_term
        demand_h[k] = demand

        over = alive & (np.abs(demand) > budget)
        overload |= over
        alive = alive & ~over
        alive_h[k] = alive

        # A locked joint never moves. Its motor pushes at full torque, in the direction it is asked
        # to move, while it is asked to move (sign(0) = 0, so no demand gives no torque).
        tau_locked = plant.tau_max * np.sign(demand)
        tau = np.where(plant.locked, tau_locked, np.clip(demand, -cap, cap))
        tau_h[k] = tau

        moving = alive & ~plant.locked
        qd_new = np.where(moving, qd + (tau - plant.gravity) / plant.inertia * dt, qd)
        q_new = np.where(moving, q + qd_new * dt, q)
        energy += np.where(alive, np.abs(tau * qd_new) + heat_coeff * tau * tau, 0.0) * dt

        out = alive & ((q_new < plant.low) | (q_new > plant.high))
        joint_limit |= out
        alive = alive & ~out

        q, qd = q_new, qd_new
        q_h[k] = q

    return Rollout(
        ref=ref_h, q=q_h, tau=tau_h, demand=demand_h, alive=alive_h,
        violations={"overload": overload, "joint_limit": joint_limit},
        energy_j=energy,
    )
