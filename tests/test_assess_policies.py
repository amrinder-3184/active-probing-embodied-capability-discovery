"""The assessment script's path options. A full run needs all 18 trained policies, so it is an
experiment, not a test (docs/testing.md); these tests check the paths it uses."""
import os

import pytest

import experiments.assess_policies as assess
from src.config import REPO_ROOT


def test_no_flags_keep_the_stage1_paths():
    args = assess.parse_args([], "arm6")
    assert args.policies == os.path.join(REPO_ROOT, "results", "policies", "arm6")
    assert args.out == os.path.join(REPO_ROOT, "results", "assessment_arm6.md")


def test_flags_override_both_paths(tmp_path):
    args = assess.parse_args(["--policies", str(tmp_path / "p"), "--out", str(tmp_path / "a.md")], "arm6")
    assert args.policies == str(tmp_path / "p") and args.out == str(tmp_path / "a.md")


def test_the_policies_flag_is_where_the_policies_are_read_from(tmp_path):
    # An empty folder: the learned table must fail on that folder, and nothing is written.
    out = tmp_path / "assessment.md"
    with pytest.raises(FileNotFoundError, match="no trained policy") as err:
        assess.main(["--policies", str(tmp_path / "empty"), "--out", str(out)])
    assert str(tmp_path / "empty") in str(err.value)
    assert not out.exists()
