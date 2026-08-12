"""Readiness-gate + docdistance.json tests - pure stdlib, no models load."""

import json
from pathlib import Path

import pytest

from docdistance import settings


@pytest.fixture(autouse=True)
def _reset():
    settings.reset()
    yield
    settings.reset()


def test_default_home_uses_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCDISTANCE_HOME", str(tmp_path))
    assert settings.default_home() == tmp_path


def test_default_home_falls_back_to_cwd(monkeypatch):
    monkeypatch.delenv("DOCDISTANCE_HOME", raising=False)
    assert settings.default_home() == Path.cwd()


def test_config_file_path_precedence(monkeypatch, tmp_path):
    monkeypatch.delenv("DOCDISTANCE_CONFIG", raising=False)
    monkeypatch.setenv("DOCDISTANCE_HOME", str(tmp_path))
    assert settings.config_file_path() == tmp_path / "docdistance.json"
    monkeypatch.setenv("DOCDISTANCE_CONFIG", str(tmp_path / "explicit.json"))
    assert settings.config_file_path() == tmp_path / "explicit.json"  # $DOCDISTANCE_CONFIG wins


def test_save_then_load_in_a_fresh_process_marks_ready(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCDISTANCE_HOME", str(tmp_path))
    settings.configure(
        modes=["wmd"], models_dir=str(tmp_path / "models"), sources={"mmbert": "hf"}
    )
    p = settings.save_config_file()
    assert p == tmp_path / "docdistance.json"
    data = json.loads(p.read_text())
    assert data["modes"] == ["wmd"] and data["sources"]["mmbert"] == "hf"

    settings.reset()  # a later process starts cold
    assert settings.is_ready("wmd") is True  # loads docdistance.json and sees the mode
    assert "wmd" in settings.get().modes


def test_require_ready_raises_clear_error_when_absent(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCDISTANCE_HOME", str(tmp_path))  # empty home, no docdistance.json
    with pytest.raises(settings.NotInitializedError) as ei:
        settings.require_ready("wmd")
    assert "docdistance init wmd" in str(ei.value)


def test_require_ready_distinguishes_modes(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCDISTANCE_HOME", str(tmp_path))
    settings.configure(modes=["wmd"])
    settings.save_config_file()
    settings.reset()
    settings.require_ready("wmd")  # provisioned, no raise
    with pytest.raises(settings.NotInitializedError):
        settings.require_ready("wmd-wrt-source")  # a different mode is not ready


def test_corrupt_config_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCDISTANCE_HOME", str(tmp_path))
    (tmp_path / "docdistance.json").write_text("{ not valid json")
    with pytest.raises(RuntimeError):
        settings.load_config_file()


def test_load_missing_file_returns_false(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCDISTANCE_HOME", str(tmp_path))
    assert settings.load_config_file() is False
