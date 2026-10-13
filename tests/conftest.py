"""Shared test setup: tests never write into the real logs/ folder, and never read the personal
config.local.yaml (so they test the committed config, the same as CI)."""
import os

import pytest

os.environ["JOB_AGENT_NO_LOCAL_CONFIG"] = "1"


@pytest.fixture(autouse=True)
def _isolated_apply_log(tmp_path, monkeypatch):
    import src.apply
    monkeypatch.setattr(src.apply, "LOG_PATH", tmp_path / "logs" / "apply.jsonl")
