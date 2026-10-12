"""One-call provisioning: pull a mode's models from local / S3 / HuggingFace.

``docdistance.init`` resolves every model a distance mode requires through one documented chain so
the distance commands can run offline:

    1. an ``s3://...`` source - mirror the model dir from the bucket (botocore, optional profile +
       endpoint for S3-compatible stores; in Lambda omit the profile for the execution-role chain)
    2. a local source dir - copy the model dir from it
    3. else the HuggingFace Hub - the always-available fallback

A source hit (1 or 2) copies the model directory. For the SaT segmenter it also saves the tokenizer
``facebookAI/xlm-roberta-base`` beside the weights when the copied folder holds no tokenizer that
loads with its special token ids, because wtpsplit otherwise loads that tokenizer by its Hub name:
from the HuggingFace cache when it is there, else from the Hub; when neither has it, ``init``
raises ``ModelsNotInstalled``. A source folder that carries the tokenizer files ``init`` saved
needs no Hub. The files as the Hub publishes them load beside the SaT ``config.json`` without
those ids, and ``init`` replaces them like a truncated ``tokenizer.json``. The model loads after
``init`` (``offline=True``, the default) make no Hub request.

It records what it did in ``docdistance.json`` (via :mod:`docdistance.settings`) so a later process
loads it and the readiness gate passes; the model loaders read the resolved paths back.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil

from loguru import logger

from docdistance import config, settings

__all__ = ["init"]


# --- source-scheme helpers -------------------------------------------------


def _scheme(uri: str) -> str:
    return "s3" if uri.startswith("s3://") else "local"


def _split_s3(uri: str) -> tuple[str, str]:
    """``s3://bucket/key/parts`` -> ``("bucket", "key/parts")``."""
    bucket, _, key = uri[len("s3://") :].partition("/")
    return bucket, key


def _s3_client(profile: str | None, endpoint_url: str | None, region: str | None):
    """A low-level botocore S3 client (path-style, for S3-compatible endpoints / custom regions)."""
    try:
        from botocore.config import Config
        import botocore.session
    except ImportError as exc:
        from docdistance.encoders import ModelsNotInstalled

        raise ModelsNotInstalled(
            "an s3:// source needs botocore - install the extra:  pip install 'docdistance[s3]'"
        ) from exc

    sess = botocore.session.Session(profile=profile) if profile else botocore.session.Session()
    return sess.create_client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region,
        config=Config(s3={"addressing_style": "path"}),
    )


def _fetch_to(uri: str, dest: str | Path, *, client=None) -> bool:
    """Fetch one S3 object or local file to ``dest``. True on success, False if absent."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if uri.startswith("s3://"):
        bucket, key = _split_s3(uri)
        try:
            body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        except Exception as exc:  # noqa: BLE001 - missing key / auth / endpoint
            logger.warning("S3 fetch miss {}: {}", uri, exc)
            return False
        dest.write_bytes(body)
        return True
    src = Path(uri)
    if not src.is_file():
        return False
    shutil.copyfile(src, dest)
    return True


def _s3_prefix_to_dir(client, s3_base: str, name: str, dest: Path) -> bool:
    """Mirror every object under ``<s3_base>/<name>/`` into ``dest``. True only if all copied.

    A non-empty listing is not proof of a usable mirror: the bucket policy may allow
    ``ListBucket`` and deny ``GetObject``, or a token may expire between the two calls.
    Reporting success on a partial copy marks the mode ready over a half-written model
    dir, so any per-object miss fails the whole prefix and lets the caller fall through.
    """
    bucket, key = _split_s3(s3_base.rstrip("/") + "/" + name + "/")
    try:
        resp = client.list_objects_v2(Bucket=bucket, Prefix=key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("S3 list miss {}: {}", key, exc)
        return False
    objs = resp.get("Contents") or []
    if not objs:
        return False
    dest.mkdir(parents=True, exist_ok=True)
    ok = True
    for o in objs:
        rel = o["Key"][len(key) :].lstrip("/")
        if not rel:
            continue
        ok = _fetch_to(f"s3://{bucket}/{o['Key']}", dest / rel, client=client) and ok
    if not ok:
        logger.warning("S3 mirror of {} incomplete - treating the prefix as a miss", key)
    return ok


# --- per-model resolution --------------------------------------------------


def _warm_hf(key: str, backend: str) -> str | None:
    """HuggingFace fallback: warm the cache for ``key``; return the local dir (None for sat / torch)."""
    reg = config.MODEL_REGISTRY[key]
    if key == "sat":  # wtpsplit downloads sat on first SaT(name) - warm it here
        import contextlib
        import io

        from wtpsplit import SaT

        with contextlib.redirect_stderr(io.StringIO()):
            SaT(reg["hf"])
        return None
    from huggingface_hub import snapshot_download

    repo = reg["torch"] if backend == "torch" else reg["hf"]
    return str(Path(snapshot_download(repo)))


def _mirror_one(
    key, s3_base, local_base, force_hf, models_dir: Path, client, backend
) -> tuple[str, str | None]:
    """Resolve one model: the source named by ``--source`` (S3 prefix XOR local dir), then HF.

    Returns ``(source_used, local_dir)``. ``init`` sets ``s3_base`` or ``local_base``, never both,
    so this is a 2-way chain, not the 3-way one older docs described.
    """
    name = config.MODEL_REGISTRY[key]["subdir"]
    dest = models_dir / name
    if not force_hf and s3_base and _s3_prefix_to_dir(client, s3_base, name, dest):
        return "s3", str(dest)
    if not force_hf and local_base:
        src = Path(local_base) / name
        if src.is_dir():
            if dest.resolve() != src.resolve():
                shutil.copytree(src, dest, dirs_exist_ok=True)
            return "local", str(dest)
    return "hf", _warm_hf(key, backend)


def _save_tokenizer(name: str, dest: Path) -> None:
    """Save the tokenizer ``name`` into ``dest``: from the HuggingFace cache, else from the Hub.

    A cached tokenizer is loaded from its snapshot folder. A load by name sends a request while the
    Hub library is online, also when every file is cached (transformers 5.0.0), and raises on a
    host without Hub access.
    """
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoTokenizer

    cached = try_to_load_from_cache(name, "tokenizer.json")  # a path only when the file is cached
    src = Path(cached).parent if isinstance(cached, str) else name
    AutoTokenizer.from_pretrained(src).save_pretrained(dest)


def _tokenizer_loads(folder: Path) -> bool:
    """Whether ``folder`` holds a tokenizer that loads with the special token ids SaT reads.

    The load is the one wtpsplit makes when :class:`docdistance.encoders.Segmenter` passes the
    folder. ``tokenizer.json`` and ``tokenizer_config.json`` as the Hub repository publishes them
    load beside the SaT ``config.json`` with ``cls_token_id``, ``sep_token_id`` and
    ``pad_token_id`` None, and the first split then raises ``TypeError`` (DEF-DEPS-17-1). A
    ``tokenizer.json`` that an interrupted write left truncated does not load at all.
    """
    if not (folder / "tokenizer.json").exists():
        return False
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(folder)
    except Exception:  # noqa: BLE001 - files that do not parse; the library has no one error type
        return False
    return None not in (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id)


def _add_sat_tokenizer(dest: Path) -> None:
    """Save the SaT tokenizer beside the weights in ``dest`` unless the source carried one that loads.

    wtpsplit loads the tokenizer by its Hub name whatever the path of the weights is, so a source
    folder with ``config.json`` and the weights only left the first offline load without a tokenizer
    (DEF-DEPS-17). :class:`docdistance.encoders.Segmenter` reads it from ``dest``. Tokenizer files
    in ``dest`` that do not load with their special token ids (:func:`_tokenizer_loads`) are
    replaced by the saved ones.
    """
    if _tokenizer_loads(dest):
        return
    try:
        _save_tokenizer(config.SAT_TOKENIZER, dest)
    except Exception as exc:  # not cached, and no answer from the Hub
        from docdistance.encoders import ModelsNotInstalled

        name = config.MODEL_REGISTRY["sat"]["subdir"]
        raise ModelsNotInstalled(
            f"the tokenizer of the segmenter, {config.SAT_TOKENIZER}, is not in the source folder "
            f"{name} in a form that loads with its special token ids, not in the HuggingFace "
            "cache and the Hub did not return it - run docdistance init --source once on a host "
            f"with Hub access and copy the tokenizer files it saves in <home>/models/{name} into "
            f"the source folder {name}"
        ) from exc


# --- public entrypoint -----------------------------------------------------


def init(
    mode: str = "wmd",
    *,
    source: str | None = None,
    backend: str = "openvino",
    aws_profile: str | None = None,
    aws_endpoint_url: str | None = None,
    aws_region: str | None = None,
    home: str | None = None,
) -> dict:
    """Provision the models ``mode`` requires and write ``docdistance.json``.

    ``mode`` is ``wmd`` (symmetric) or ``wmd-wrt-source`` (source-conditioned). ``source`` selects
    where to pull from: an ``s3://bucket/prefix`` base, a local dir, or ``None`` / ``"hf"`` for the
    HuggingFace Hub. Returns a summary naming which of the three served each model.
    """
    if mode not in config.MODE_MODELS:
        raise ValueError(f"unknown mode {mode!r}; choose one of {list(config.MODE_MODELS)}")

    home_path = Path(home) if home else settings.default_home()
    home_path.mkdir(parents=True, exist_ok=True)
    if home:  # pin so a later process resolves the same home; else docdistance.json falls to cwd
        os.environ[settings.ENV_HOME] = str(home_path)
    models_dir = home_path / "models"

    # accumulate onto any prior init in this home, so provisioning a second mode does not clobber
    # the first (init wmd then init wmd-wrt-source leaves both ready)
    try:
        settings.load_config_file()
    except RuntimeError:
        settings.reset()  # a corrupt prior config is replaced by this fresh init

    force_hf = source is None or source == "hf"
    s3_base = local_base = client = None
    if not force_hf:
        if _scheme(source) == "s3":
            s3_base = source
            client = _s3_client(aws_profile, aws_endpoint_url, aws_region)
        else:
            local_base = source
            if not Path(local_base).is_dir():
                raise FileNotFoundError(f"--source local dir not found: {local_base}")

    from docdistance.encoders import _hub_flag, _require_models_extra, _set_hf_token

    _require_models_extra()
    _set_hf_token()

    model_paths: dict[str, str] = {}
    sources: dict[str, str] = {}
    # init is the online step: the Hub library is held online while the models are fetched, also
    # when HF_HUB_OFFLINE was set before the library was imported (DEF-DEPS-16)
    with _hub_flag(False):
        for key in config.MODE_MODELS[mode]:
            used, path = _mirror_one(
                key, s3_base, local_base, force_hf, models_dir, client, backend
            )
            if key == "sat" and used != "hf":  # from the Hub the SaT load caches the tokenizer
                _add_sat_tokenizer(Path(path))
            sources[key] = used
            if path:
                model_paths[key] = path
            logger.info("model '{}' provisioned from {}", key, used)

    prior = settings.get()
    settings.configure(
        home=str(home_path),
        models_dir=str(models_dir),
        model_paths={**prior.model_paths, **model_paths},
        sources={**prior.sources, **sources},
    )
    settings.mark_ready(mode)  # appends to the modes loaded from any prior config
    cfg = settings.save_config_file()
    summary = {
        "mode": mode,
        "home": str(home_path),
        "models_dir": str(models_dir),
        "sources": sources,
        "config_file": str(cfg),
    }
    logger.success("docdistance init complete ({}): {}", mode, sources)
    return summary
