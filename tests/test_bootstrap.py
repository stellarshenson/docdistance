"""init / 3-way model resolution tests.

Local + HuggingFace paths are exercised with the heavy model imports and the HF fetch monkeypatched,
so they run in the light venv. The real S3 round-trip is opt-in (``DOCDISTANCE_S3_TEST=1``) and
skipped in CI - it hits ``s3://general-purpose/docdistance`` via the ``stellars-tech`` profile.
"""

import json
import os
from pathlib import Path

import pytest

from docdistance import bootstrap, config, settings


@pytest.fixture(autouse=True)
def _light(monkeypatch, tmp_path):
    """Reset state and stub the heavy model-extra import + HF token mapping."""
    settings.reset()
    monkeypatch.setattr("docdistance.encoders._require_models_extra", lambda: None)
    monkeypatch.setattr("docdistance.encoders._set_hf_token", lambda: None)
    monkeypatch.setenv("DOCDISTANCE_HOME", str(tmp_path / "home"))
    yield
    settings.reset()


def _fake_local_mirror(base: Path, keys) -> Path:
    """Build a local source dir with a model subdir per key (an empty IR file is enough to copy)."""
    for key in keys:
        sub = base / config.MODEL_REGISTRY[key]["subdir"]
        sub.mkdir(parents=True)
        (sub / "openvino_model.xml").write_text("<ir/>")
    return base


def test_scheme_and_split_s3():
    assert bootstrap._scheme("s3://bucket/key") == "s3"
    assert bootstrap._scheme("/some/local/dir") == "local"
    assert bootstrap._split_s3("s3://bucket/a/b/c") == ("bucket", "a/b/c")


def test_init_unknown_mode_raises():
    with pytest.raises(ValueError):
        bootstrap.init("bogus-mode")


def test_init_local_source_records_local_and_never_touches_hf(monkeypatch, tmp_path):
    def _boom(*a, **k):
        raise AssertionError("HuggingFace must not be reached when a local source has the model")

    monkeypatch.setattr(bootstrap, "_warm_hf", _boom)
    mirror = _fake_local_mirror(tmp_path / "mirror", ["mmbert", "sat"])

    summary = bootstrap.init("wmd", source=str(mirror), home=str(tmp_path / "home"))

    assert summary["mode"] == "wmd"
    assert summary["sources"] == {"mmbert": "local", "sat": "local"}
    cfg = Path(summary["config_file"])
    assert cfg.is_file()
    data = json.loads(cfg.read_text())
    assert data["modes"] == ["wmd"]
    assert data["sources"]["mmbert"] == "local"
    # the model was copied into the init mirror dir and recorded
    assert (Path(summary["models_dir"]) / "mmbert-openvino-int8" / "openvino_model.xml").is_file()


def test_init_hf_source_records_hf(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        bootstrap, "_warm_hf", lambda key, backend: seen.append(key) or f"/cache/{key}"
    )

    summary = bootstrap.init("wmd", home=str(tmp_path / "home"))  # no source -> HuggingFace

    assert summary["sources"] == {"mmbert": "hf", "sat": "hf"}
    assert set(seen) == {"mmbert", "sat"}


def test_init_wrt_source_provisions_all_four_models(monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap, "_warm_hf", lambda key, backend: f"/cache/{key}")
    summary = bootstrap.init("wmd-wrt-source", home=str(tmp_path / "home"))
    assert set(summary["sources"]) == {"mmbert", "sat", "reranker", "nli"}


def test_init_local_source_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        bootstrap.init("wmd", source=str(tmp_path / "does-not-exist"))


def test_s3_client_without_botocore_raises_clear_error(monkeypatch):
    """A missing botocore on an s3:// source gives a 'pip install docdistance[s3]' message, not a bare ImportError."""
    import sys

    from docdistance.encoders import ModelsNotInstalled

    for mod in ("botocore", "botocore.session", "botocore.config"):
        monkeypatch.setitem(sys.modules, mod, None)  # force ImportError even if installed
    with pytest.raises(ModelsNotInstalled, match=r"docdistance\[s3\]"):
        bootstrap._s3_client("some-profile", None, None)


def test_init_is_additive_across_modes(monkeypatch, tmp_path):
    """init wmd then (a fresh process) init wmd-wrt-source leaves BOTH modes ready, no clobber."""
    monkeypatch.setattr(bootstrap, "_warm_hf", lambda key, backend: f"/cache/{key}")
    home = str(tmp_path / "home")
    bootstrap.init("wmd", home=home)
    settings.reset()  # simulate a separate process
    bootstrap.init("wmd-wrt-source", home=home)

    data = json.loads((Path(home) / "docdistance.json").read_text())
    assert set(data["modes"]) == {"wmd", "wmd-wrt-source"}
    settings.reset()
    settings.require_ready("wmd")  # the first mode survives the second init
    settings.require_ready("wmd-wrt-source")


@pytest.mark.skipif(
    not os.getenv("DOCDISTANCE_S3_TEST"),
    reason="opt-in: hits real s3://general-purpose/docdistance via the stellars-tech profile",
)
def test_init_s3_source_downloads_the_mmbert_ir(monkeypatch, tmp_path):
    settings.reset()
    # isolate the S3 proof to mmBERT: skip the HF fallback for any model not staged on S3 (sat)
    monkeypatch.setattr(bootstrap, "_warm_hf", lambda key, backend: None)
    summary = bootstrap.init(
        "wmd",
        source="s3://general-purpose/docdistance",
        backend="openvino",
        aws_profile="stellars-tech",
        home=str(tmp_path / "home"),
    )
    assert summary["sources"]["mmbert"] == "s3"
    xml = Path(summary["models_dir"]) / "mmbert-openvino-int8" / "openvino_model.xml"
    assert xml.is_file()
