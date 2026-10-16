"""Platform adapter: the joints, chains, limits and probes of one robot.

To add a robot, write a new configs/platform_*.yaml. The tier logic does not change.
"""
from dataclasses import dataclass

from .config import config_path, read_yaml


@dataclass(frozen=True)
class Joint:
    name: str
    chain: str
    index: int            # position in its chain, 0 = nearest the base
    tau_max_nm: float
    kp: float             # position-loop stiffness, N*m/rad
    kd: float             # position-loop damping, N*m*s/rad
    inertia: float        # kg*m^2
    gravity_nm: float     # torque the joint must hold in the T3 probe
    limits_deg: tuple     # (low, high), relative to the home pose


@dataclass(frozen=True)
class Platform:
    name: str
    joints: tuple         # every joint, in config order
    chains: dict          # chain name -> joint names, from the base outwards
    probes: dict          # tier name -> probe spec, with a "kind" key
    assessment: dict      # tier name -> "scripted" or "learned"
    probe_torque_limit_fraction: float
    torque_constant_nm_per_a: float
    heat_coeff: float

    def joint(self, name):
        return next(j for j in self.joints if j.name == name)

    def proximal(self, name):
        """Joints below this one in its chain: the ones between it and the base."""
        chain = self.chains[self.joint(name).chain]
        return chain[: chain.index(name)]


def usable_tiers(platform, tiers):
    """Tiers that this platform has a probe for, in tier order."""
    return [t for t in tiers if t.name in platform.probes]


def load_platform(path=None):
    raw = read_yaml(path or config_path("platform_arm6.yaml"))
    chains = {name: tuple(joints) for name, joints in raw["chains"].items()}

    # Each joint must belong to exactly one chain, and each chain entry must be a joint.
    listed = [name for names in chains.values() for name in names]
    if sorted(listed) != sorted(raw["joints"]):
        raise ValueError("every joint must appear in exactly one chain")

    joints = []
    for name, spec in raw["joints"].items():
        chain_names = chains[spec["chain"]]
        joints.append(Joint(
            name=name,
            chain=spec["chain"],
            index=chain_names.index(name),
            tau_max_nm=float(spec["tau_max_nm"]),
            kp=float(spec["kp"]),
            kd=float(spec["kd"]),
            inertia=float(spec["inertia"]),
            gravity_nm=float(spec["gravity_nm"]),
            limits_deg=tuple(float(x) for x in spec["limits_deg"]),
        ))

    return Platform(
        name=raw["name"],
        joints=tuple(joints),
        chains=chains,
        probes=raw["probes"],
        assessment=dict(raw.get("assessment", {})),
        probe_torque_limit_fraction=float(raw["safety"]["probe_torque_limit_fraction"]),
        torque_constant_nm_per_a=float(raw["safety"]["torque_constant_nm_per_a"]),
        heat_coeff=float(raw["safety"]["heat_coeff"]),
    )
