"""Locate and read the YAML files in configs/."""
import os

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO_ROOT, "configs")


def config_path(name):
    return os.path.join(CONFIG_DIR, name)


def read_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)
