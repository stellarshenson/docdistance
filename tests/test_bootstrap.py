"""init / 3-way model resolution tests.

Local + HuggingFace paths are exercised with the heavy model imports and the HF fetch monkeypatched,
so they run in the light venv. The real S3 round-trip is opt-in (``DOCDISTANCE_S3_TEST=1``) and
skipped in CI - it hits ``s3://general-purpose/docdistance`` via the ``stellars-tech`` profile.
"""

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

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
    """Build a local source dir with a model subdir per key (an empty IR file is enough to copy).

    The ``sat`` subdir also holds a ``tokenizer.json``: a source folder that carries the tokenizer
    of the segmenter is complete (DEF-DEPS-17) when that tokenizer loads with its special token
    ids (DEF-DEPS-17-1); ``_record_tokenizer_loads`` replaces the load.
    """
    for key in keys:
        sub = base / config.MODEL_REGISTRY[key]["subdir"]
        sub.mkdir(parents=True)
        (sub / "openvino_model.xml").write_text("<ir/>")
        if key == "sat":
            (sub / "tokenizer.json").write_text("{}")
    return base


def _record_tokenizer_saves(monkeypatch) -> list[tuple[str, Path]]:
    """Replace the tokenizer save of ``init`` by a stub; return the list its calls are added to."""
    saves: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        bootstrap, "_save_tokenizer", lambda name, dest: saves.append((name, dest))
    )
    return saves


# the ids of the three special tokens SaT reads, as facebookAI/xlm-roberta-base holds them
_SAT_TOKEN_IDS = {"cls_token_id": 0, "sep_token_id": 2, "pad_token_id": 1}


def _record_tokenizer_loads(monkeypatch, load=lambda: SimpleNamespace(**_SAT_TOKEN_IDS)) -> list:
    """Replace the tokenizer load of ``init`` by ``load``; return the list its sources are added to.

    ``load`` takes no argument and returns the tokenizer or raises. The default returns a
    tokenizer that holds the three special token ids.
    """
    loads: list = []

    def from_pretrained(source):
        loads.append(source)
        return load()

    auto_tokenizer = SimpleNamespace(from_pretrained=from_pretrained)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=auto_tokenizer))
    return loads


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
    # the sat folder of the source carries a tokenizer that loads with its ids, so none is saved
    monkeypatch.setattr(bootstrap, "_save_tokenizer", _boom)
    _record_tokenizer_loads(monkeypatch)
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
    assert (Path(summary["models_dir"]) / "sat-3l-sm" / "tokenizer.json").is_file()


def test_init_local_source_without_the_sat_tokenizer_saves_it_beside_the_weights(
    monkeypatch, tmp_path
):
    """DEF-DEPS-17: a ``sat`` source folder without ``tokenizer.json`` gets the tokenizer from ``init``.

    wtpsplit loads the tokenizer of SaT by its Hub name, so a folder with the weights only left the
    first offline load without it. The save is called once, for ``sat`` only, with the tokenizer
    name and the destination folder, and while ``init`` holds the Hub library online: the flag is
    preset to offline here, as with ``HF_HUB_OFFLINE=1``.
    """
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    saves: list[tuple[str, Path]] = []
    offline_during: list[bool] = []

    def save(name: str, dest: Path) -> None:
        saves.append((name, dest))
        offline_during.append(constants.is_offline_mode())

    monkeypatch.setattr(bootstrap, "_save_tokenizer", save)
    mirror = _fake_local_mirror(tmp_path / "mirror", ["mmbert", "sat"])
    (mirror / "sat-3l-sm" / "tokenizer.json").unlink()

    summary = bootstrap.init("wmd", source=str(mirror), home=str(tmp_path / "home"))

    assert summary["sources"] == {"mmbert": "local", "sat": "local"}
    assert saves == [(config.SAT_TOKENIZER, Path(summary["models_dir"]) / "sat-3l-sm")]
    assert offline_during == [False]
    assert not list(mirror.rglob("tokenizer*"))  # the source folder is read, never written


def test_init_s3_source_without_the_sat_tokenizer_saves_it_beside_the_weights(
    monkeypatch, tmp_path
):
    """DEF-DEPS-17: the same holds when the ``sat`` folder comes from an S3 prefix."""

    def mirror_prefix(client, s3_base, name, dest: Path) -> bool:
        dest.mkdir(parents=True)
        (dest / "model.safetensors").write_bytes(b"weights")
        return True

    monkeypatch.setattr(bootstrap, "_s3_client", lambda profile, endpoint_url, region: object())
    monkeypatch.setattr(bootstrap, "_s3_prefix_to_dir", mirror_prefix)
    saves = _record_tokenizer_saves(monkeypatch)

    summary = bootstrap.init("wmd", source="s3://bucket/prefix", home=str(tmp_path / "home"))

    assert summary["sources"] == {"mmbert": "s3", "sat": "s3"}
    assert saves == [(config.SAT_TOKENIZER, Path(summary["models_dir"]) / "sat-3l-sm")]


def test_init_hf_source_makes_no_tokenizer_save(monkeypatch, tmp_path):
    """DEF-DEPS-17: from the HuggingFace Hub the SaT load of ``init`` puts the tokenizer in the cache."""
    monkeypatch.setattr(bootstrap, "_warm_hf", lambda key, backend: f"/cache/{key}")
    saves = _record_tokenizer_saves(monkeypatch)

    bootstrap.init("wmd", home=str(tmp_path / "home"))

    assert saves == []


def test_init_source_with_an_unavailable_sat_tokenizer_raises_and_names_it(monkeypatch, tmp_path):
    """DEF-DEPS-17: the tokenizer is in no place ``init`` reads - the error names it and both remedies.

    ``init`` raises before it writes ``docdistance.json``, so the mode is not marked ready over a
    segmenter that cannot load.
    """
    from docdistance.encoders import ModelsNotInstalled

    def unavailable(name: str, dest: Path) -> None:
        raise OSError("not in the HuggingFace cache and the Hub is not reachable")

    monkeypatch.setattr(bootstrap, "_save_tokenizer", unavailable)
    mirror = _fake_local_mirror(tmp_path / "mirror", ["mmbert", "sat"])
    (mirror / "sat-3l-sm" / "tokenizer.json").unlink()
    home = tmp_path / "home"

    with pytest.raises(ModelsNotInstalled) as raised:
        bootstrap.init("wmd", source=str(mirror), home=str(home))

    message = str(raised.value)
    assert "facebookAI/xlm-roberta-base" in message
    assert "source folder sat-3l-sm" in message  # remedy 1: copy the tokenizer files there
    assert "host with Hub access" in message  # remedy 2: run init once where the Hub answers
    assert not (home / "docdistance.json").exists()


def _load_that_raises():
    raise ValueError("expected value at line 1 column 1")


@pytest.mark.parametrize(
    ("load", "saved"),
    [
        (lambda: SimpleNamespace(**_SAT_TOKEN_IDS), False),
        (lambda: SimpleNamespace(**{**_SAT_TOKEN_IDS, "cls_token_id": None}), True),
        (lambda: SimpleNamespace(**{**_SAT_TOKEN_IDS, "sep_token_id": None}), True),
        (lambda: SimpleNamespace(**{**_SAT_TOKEN_IDS, "pad_token_id": None}), True),
        (_load_that_raises, True),
    ],
    ids=["three ids", "cls id None", "sep id None", "pad id None", "load raises"],
)
def test_init_replaces_a_sat_tokenizer_that_does_not_load_with_its_special_token_ids(
    load, saved, monkeypatch, tmp_path
):
    """DEF-DEPS-17-1: ``init`` loads the tokenizer a ``sat`` source folder carries and reads three ids.

    ``tokenizer.json`` and ``tokenizer_config.json`` as the Hub publishes them load beside the SaT
    ``config.json`` with ``cls_token_id``, ``sep_token_id`` and ``pad_token_id`` None, and a
    truncated ``tokenizer.json`` does not load. ``init`` passed both, and the first split raised
    ``TypeError``. A tokenizer with the three ids is kept; with one id None, or when the load
    raises, the tokenizer is saved over it as for a folder without one. The load reads the folder
    of the home, once.
    """
    saves = _record_tokenizer_saves(monkeypatch)
    loads = _record_tokenizer_loads(monkeypatch, load)
    mirror = _fake_local_mirror(tmp_path / "mirror", ["mmbert", "sat"])

    summary = bootstrap.init("wmd", source=str(mirror), home=str(tmp_path / "home"))

    sat_dir = Path(summary["models_dir"]) / "sat-3l-sm"
    assert loads == [sat_dir]
    assert saves == ([(config.SAT_TOKENIZER, sat_dir)] if saved else [])
    assert summary["sources"] == {"mmbert": "local", "sat": "local"}


def test_init_loads_no_tokenizer_from_a_sat_folder_without_one(monkeypatch, tmp_path):
    """DEF-DEPS-17-1: the load is made when the folder holds ``tokenizer.json``. Without the file
    there is nothing to load and the tokenizer is saved."""
    saves = _record_tokenizer_saves(monkeypatch)
    loads = _record_tokenizer_loads(monkeypatch)
    mirror = _fake_local_mirror(tmp_path / "mirror", ["mmbert", "sat"])
    (mirror / "sat-3l-sm" / "tokenizer.json").unlink()

    bootstrap.init("wmd", source=str(mirror), home=str(tmp_path / "home"))

    assert loads == []
    assert len(saves) == 1


def test_init_with_a_sat_tokenizer_that_does_not_load_and_none_to_save_raises(
    monkeypatch, tmp_path
):
    """DEF-DEPS-17-1: the folder holds a tokenizer without its ids and no place has one to save.

    ``init`` raises the error of a folder without a tokenizer and writes no ``docdistance.json``.
    Before, it wrote the file and the first split raised ``TypeError``.
    """
    from docdistance.encoders import ModelsNotInstalled

    def unavailable(name: str, dest: Path) -> None:
        raise OSError("not in the HuggingFace cache and the Hub is not reachable")

    monkeypatch.setattr(bootstrap, "_save_tokenizer", unavailable)
    _record_tokenizer_loads(monkeypatch, lambda: SimpleNamespace(**dict.fromkeys(_SAT_TOKEN_IDS)))
    mirror = _fake_local_mirror(tmp_path / "mirror", ["mmbert", "sat"])
    home = tmp_path / "home"

    with pytest.raises(ModelsNotInstalled) as raised:
        bootstrap.init("wmd", source=str(mirror), home=str(home))

    message = str(raised.value)
    assert "facebookAI/xlm-roberta-base" in message
    assert "loads with its special token ids" in message
    assert "host with Hub access" in message
    assert not (home / "docdistance.json").exists()


@pytest.mark.parametrize("in_cache", [True, False])
def test_save_tokenizer_reads_the_cached_snapshot_folder_and_else_the_hub_name(
    in_cache, monkeypatch, tmp_path
):
    """The tokenizer is loaded from its cached snapshot folder when it is cached, else by its name.

    ``init`` holds the Hub library online, and a load by name then sends a request also when every
    file is cached (transformers 5.0.0), so it raises on a host without Hub access. The model
    libraries are stubs that record the load source and the save destination. A file the Hub
    reported as absent is cached as an object that is not a path: it is read as not cached.
    """
    import sys
    from types import SimpleNamespace

    snapshot = tmp_path / "snapshot"
    cached = str(snapshot / "tokenizer.json") if in_cache else object()
    lookups: list[tuple[str, str]] = []
    loaded: list[object] = []
    saved: list[Path] = []

    def try_to_load_from_cache(repo_id: str, filename: str):
        lookups.append((repo_id, filename))
        return cached

    def from_pretrained(source):
        loaded.append(source)
        return SimpleNamespace(save_pretrained=saved.append)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(try_to_load_from_cache=try_to_load_from_cache),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained)),
    )

    bootstrap._save_tokenizer("org/name", tmp_path / "dest")

    assert lookups == [("org/name", "tokenizer.json")]
    assert loaded == [snapshot if in_cache else "org/name"]
    assert saved == [tmp_path / "dest"]


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


def test_init_holds_the_hub_library_online_and_puts_the_earlier_mode_back(monkeypatch, tmp_path):
    """DEF-DEPS-16: with ``HF_HUB_OFFLINE=1`` preset, ``init`` reaches the Hub and leaves the mode as it was.

    The Hub library reads the variable once, at import, into ``constants.HF_HUB_OFFLINE``; the
    test presets both. On 1.1.13 ``init`` removed the variable from the environment after that
    import: the flag stayed set while the models were fetched and the variable was absent afterwards.
    """
    from huggingface_hub import constants

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    during: list[bool] = []
    monkeypatch.setattr(
        bootstrap, "_warm_hf", lambda key, backend: during.append(constants.is_offline_mode())
    )

    bootstrap.init("wmd", home=str(tmp_path / "home"))

    assert during == [False, False]  # one fetch per model of the mode
    assert constants.is_offline_mode() is True
    assert os.environ["HF_HUB_OFFLINE"] == "1"


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
