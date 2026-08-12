"""Runtime configuration and the init readiness gate.

``docdistance init`` provisions a mode's models (from local / S3 / HuggingFace) and records what it
did in a ``docdistance.json`` written to ``$DOCDISTANCE_HOME`` (else the current folder). A later
process - the ``distance`` CLI, a Lambda handler - loads that file, which marks the runtime ready so
the distance commands run; a mode that was never init'd raises :class:`NotInitializedError` with a
clear "run ``docdistance init <mode>``" message instead of a cryptic missing-model error.

Nothing here imports torch / openvino - the gate is pure stdlib so it loads cheaply.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
import json
import os
from pathlib import Path

from loguru import logger

ENV_HOME = "DOCDISTANCE_HOME"
ENV_CONFIG = "DOCDISTANCE_CONFIG"
CONFIG_FILENAME = "docdistance.json"


def default_home() -> Path:
    """Where ``init`` writes ``docdistance.json`` + the model mirror: ``$DOCDISTANCE_HOME`` else cwd."""
    env = os.environ.get(ENV_HOME)
    return Path(env) if env else Path.cwd()


@dataclass
class RuntimeConfig:
    """The init-resolved runtime state, persisted to ``docdistance.json``."""

    home: str = ""
    models_dir: str = ""
    modes: list[str] = field(default_factory=list)  # modes init has provisioned
    model_paths: dict[str, str] = field(default_factory=dict)  # model key -> resolved local dir
    sources: dict[str, str] = field(default_factory=dict)  # model key -> "local" | "s3" | "hf"


_RUNTIME = RuntimeConfig()
_LOADED = False  # whether a docdistance.json load was already attempted this process


def get() -> RuntimeConfig:
    """The active runtime config."""
    return _RUNTIME


def configure(**overrides) -> RuntimeConfig:
    """Override runtime settings in place (``None`` values are ignored)."""
    global _RUNTIME
    valid = {f.name for f in fields(RuntimeConfig)}
    changes = {k: v for k, v in overrides.items() if k in valid and v is not None}
    _RUNTIME = replace(_RUNTIME, **changes)
    return _RUNTIME


def reset() -> None:
    """Reset to built-in defaults and clear the load flag (test helper)."""
    global _RUNTIME, _LOADED
    _RUNTIME = RuntimeConfig()
    _LOADED = False


# --- readiness gate --------------------------------------------------------


class NotInitializedError(RuntimeError):
    """Raised when a distance mode runs before ``docdistance init <mode>`` provisioned it."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        super().__init__(
            f"mode '{mode}' is not initialized - run:  docdistance init {mode}  "
            f"(Python: docdistance.init('{mode}')) to provision its models first"
        )


def mark_ready(mode: str) -> None:
    """Record ``mode`` as provisioned for this process."""
    if mode not in _RUNTIME.modes:
        _RUNTIME.modes.append(mode)


def is_ready(mode: str) -> bool:
    """True once ``init`` ran for ``mode`` in this process, or a ``docdistance.json`` records it."""
    if mode in _RUNTIME.modes:
        return True
    load_config_file()  # lazily pick up a persisted init from an earlier process
    return mode in _RUNTIME.modes


def require_ready(mode: str) -> None:
    """Raise :class:`NotInitializedError` unless ``mode`` has been initialized."""
    if not is_ready(mode):
        raise NotInitializedError(mode)


# --- docdistance.json (the init-written runtime config) --------------------


def config_file_path(explicit: str | None = None) -> Path:
    """Where ``docdistance.json`` is written / read: an explicit path, then ``$DOCDISTANCE_CONFIG``,
    else ``<home>/docdistance.json`` (home = ``$DOCDISTANCE_HOME`` or the current folder)."""
    if explicit:
        return Path(explicit)
    env = os.environ.get(ENV_CONFIG)
    if env:
        return Path(env)
    return default_home() / CONFIG_FILENAME


def save_config_file(explicit: str | None = None) -> Path:
    """Persist the active runtime config to ``docdistance.json``; raises ``OSError`` if it cannot.

    Swallowing the write failure made ``init`` exit 0 having persisted nothing, so every later
    process reported the mode un-init'd with no clue why. A read-only home must fail loudly.
    """
    p = config_file_path(explicit)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(_RUNTIME), indent=2), encoding="utf-8")
    return p


def load_config_file(explicit: str | None = None) -> bool:
    """Load ``docdistance.json`` into the runtime config and mark its modes ready.

    Returns ``True`` when a file was found and loaded, ``False`` when none exists. A malformed file
    raises ``RuntimeError`` - a corrupt config must fail loudly, not silently look un-init'd.
    """
    global _LOADED
    p = config_file_path(explicit)
    if not p.is_file():
        _LOADED = True
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            f"docdistance config at {p} is unreadable - re-run docdistance init"
        ) from exc
    valid = {f.name for f in fields(RuntimeConfig)}
    configure(**{k: v for k, v in data.items() if k in valid})
    _LOADED = True
    logger.debug("loaded docdistance config from {} (modes: {})", p, _RUNTIME.modes)
    return True
