"""Shared test setup: tests never write into the real logs/ folder."""
import pytest


@pytest.fixture(autouse=True)
def _isolated_apply_log(tmp_path, monkeypatch):
    import src.apply
    monkeypatch.setattr(src.apply, "LOG_PATH", tmp_path / "logs" / "apply.jsonl")
