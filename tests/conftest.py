import dataclasses

import pytest

from src.damage import load_damage
from src.pipeline import build_model
from src.platform_adapter import load_platform
from src.policies import PolicyStore
from src.tasks import skill_for
from src.tiers import load_tiers
from src.training import load_training_config, train_skill


@pytest.fixture(scope="session")
def arm():
    """The arm6 platform and the scripted (reference) table. Built once per test run."""
    platform = load_platform()
    tiers = load_tiers()
    damage = load_damage()
    table = build_model(platform, tiers, damage, force_scripted=True)
    return platform, tiers, damage, table


@pytest.fixture(scope="session")
def fast_settings():
    """Training settings small enough for tests. The real ones are in configs/training.yaml."""
    return dataclasses.replace(load_training_config(), population=16, instances=2, generations=12,
                               gate_episodes=40, table_rollouts=10)


@pytest.fixture(scope="session")
def shoulder_store(arm, fast_settings, tmp_path_factory):
    """Trained shoulder policies for T1 to T3, trained quickly into a temporary store."""
    platform, tiers, _, _ = arm
    by_name = {t.name: t for t in tiers}
    store = PolicyStore(tmp_path_factory.mktemp("policies"))
    for tier_name in ["T1", "T2", "T3"]:
        record = train_skill(platform.joint("shoulder"), by_name[tier_name],
                             skill_for(platform, tier_name), platform, fast_settings, seed=7)
        store.put(record)
    return store
