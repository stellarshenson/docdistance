"""Model-backed segmentation and embedding.

The heavy dependencies (torch, transformers, openvino, wtpsplit, huggingface_hub) are imported
lazily inside the functions, so the pure-numpy :mod:`docdistance.distance` core stays
importable - and unit-testable - without them.

Inference never downloads. The constructors set ``HF_HUB_OFFLINE=1``; a model missing from the
cache raises :class:`ModelsNotInstalled` pointing at ``docdistance install``. Downloading happens
only in :func:`download_models`, which the ``install`` CLI command calls (TQDM progress bars come
from huggingface_hub).
"""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path

from loguru import logger
import numpy as np

from docdistance import config

# keep transformers quiet before it is ever imported - it otherwise prints a model LOAD REPORT
# and advisory warnings (e.g. dropped LM-head keys) that leak past stderr redirection
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

EMBED_BATCH = 64
MAX_TOKENS = 128

_INSTALL_HINT = "model not found in cache - run:  docdistance install"
_EXTRA_HINT = "model dependencies missing - reinstall:  pip install --force-reinstall docdistance"
_GPU_HINT = (
    "GPU requested (--gpu) but GPU support is not secured - install the extra and a CUDA build:\n"
    "  pip install 'docdistance[gpu]'   (needs a CUDA-capable torch wheel + an NVIDIA driver)"
)


class ModelsNotInstalled(RuntimeError):
    """A required model is missing from the cache - run ``docdistance install``."""


class GpuNotAvailable(RuntimeError):
    """``--gpu`` was requested but the GPU extra is missing or no CUDA device is visible."""


def require_gpu() -> None:
    """Raise :class:`GpuNotAvailable` unless the ``[gpu]`` extra is installed AND a CUDA device is visible.

    The ``--gpu`` flag must fail loudly rather than silently fall back to CPU - this is the gate.
    ``accelerate`` is the ``[gpu]`` extra sentinel; ``torch.cuda.is_available()`` is the hardware check.
    """
    try:
        import accelerate  # noqa: F401  - the [gpu] extra sentinel
        import torch
    except ModuleNotFoundError as exc:
        raise GpuNotAvailable(_GPU_HINT) from exc
    if not torch.cuda.is_available():
        raise GpuNotAvailable(_GPU_HINT)


def _require_models_extra() -> None:
    try:
        import transformers
        import wtpsplit  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModelsNotInstalled(_EXTRA_HINT) from exc
    # belt-and-suspenders alongside the env vars: silence the LOAD REPORT / modeling logger
    import logging as _logging

    transformers.logging.set_verbosity_error()
    _logging.getLogger("transformers.modeling_utils").setLevel(_logging.ERROR)


def _set_hf_token() -> None:
    """Map the project's vault token (HF_AUTH_TOKEN, loaded from .env by config) to HF_TOKEN."""
    if os.environ.get("HF_AUTH_TOKEN") and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ["HF_AUTH_TOKEN"]
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class Segmenter:
    """SAT statement segmenter (wtpsplit ``sat-3l-sm``), CPU."""

    def __init__(self, offline: bool = True):
        _require_models_extra()
        _set_hf_token()
        if offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        with contextlib.redirect_stderr(io.StringIO()):
            from wtpsplit import SaT

            try:
                self._sat = SaT(config.SAT_MODEL)
            except Exception as exc:  # missing weights under offline mode
                raise ModelsNotInstalled(_INSTALL_HINT) from exc
        logger.debug("loaded SAT segmenter '{}'", config.SAT_MODEL)

    def split(self, text: str) -> list[str]:
        with contextlib.redirect_stderr(io.StringIO()):
            return [s.strip() for s in self._sat.split(text) if s.strip()]


def _length_order(tok, sents: list[str]) -> list[int]:
    """Indices of ``sents`` ordered by tokenized length (length-bucketing, E06-H25).

    Sorting same-length statements together makes each ``padding=True`` batch pad near its own
    length instead of the global max; with O(L^2) attention that cuts the padded-token compute.
    The encoders scatter results back to the input order, so embeddings stay row-aligned to ``sents``.
    """
    lengths = [len(ids) for ids in tok(sents, truncation=True, max_length=MAX_TOKENS)["input_ids"]]
    return sorted(range(len(sents)), key=lengths.__getitem__)


class OpenVINOEncoder:
    """mmBERT INT8 OpenVINO encoder (CPU). Mean-pooled, L2-normalized statement embeddings."""

    name = "openvino"

    def __init__(self, offline: bool = True):
        _require_models_extra()
        _set_hf_token()
        import openvino as ov
        from transformers import AutoTokenizer

        src = config.MMBERT_OPENVINO_LOCAL
        if not (src / "openvino_model.xml").exists():
            if offline:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from huggingface_hub import snapshot_download

            try:
                src = Path(snapshot_download(config.MMBERT_OPENVINO_HF))
            except Exception as exc:
                raise ModelsNotInstalled(_INSTALL_HINT) from exc
        core = ov.Core()
        model = core.read_model(str(src / "openvino_model.xml"))
        # 2nd input name is dropped to '74' (attention_mask) during conversion - feed positionally
        self._innames = [i.get_any_name() for i in model.inputs]
        self._cm = core.compile_model(model, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
        self._tok = AutoTokenizer.from_pretrained(str(src))
        logger.debug("loaded OpenVINO INT8 encoder from {}", src)

    def encode(self, sents: list[str]) -> np.ndarray:
        order = _length_order(self._tok, sents)  # length-bucket: pad near each batch's own length
        ordered = [sents[j] for j in order]
        out = []
        for i in range(0, len(ordered), EMBED_BATCH):
            batch = ordered[i : i + EMBED_BATCH]
            enc = self._tok(
                batch, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="np"
            )
            feeds = {self._innames[0]: enc["input_ids"], self._innames[1]: enc["attention_mask"]}
            hidden = self._cm(feeds)[self._cm.output(0)]
            mask = enc["attention_mask"][..., None].astype("float32")
            pooled = (hidden * mask).sum(1) / np.clip(mask.sum(1), 1, None)
            out.append(
                (pooled / (np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-9)).astype(
                    np.float32
                )
            )
        emb = np.concatenate(out, 0)
        result = np.empty_like(emb)
        result[order] = emb  # scatter back to input order - row i maps to sents[i]
        return result


class TorchEncoder:
    """mmBERT PyTorch encoder (GPU bf16 if available, else CPU fp32)."""

    name = "torch"

    def __init__(self, offline: bool = True, device: str | None = None):
        _require_models_extra()
        _set_hf_token()
        if offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self._torch = torch
        self._dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with contextlib.redirect_stderr(io.StringIO()):
            conf = AutoConfig.from_pretrained(config.MMBERT_TORCH_MODEL)
            conf.reference_compile = False  # avoid the ModernBERT first-forward torch.compile hang
            try:
                self._tok = AutoTokenizer.from_pretrained(config.MMBERT_TORCH_MODEL)
                enc = AutoModel.from_pretrained(
                    config.MMBERT_TORCH_MODEL, config=conf, attn_implementation="eager"
                )
            except Exception as exc:
                raise ModelsNotInstalled(_INSTALL_HINT) from exc
        dtype = torch.bfloat16 if self._dev == "cuda" else torch.float32
        self._enc = enc.to(self._dev).to(dtype).eval()
        logger.debug("loaded Torch encoder '{}' on {}", config.MMBERT_TORCH_MODEL, self._dev)

    def encode(self, sents: list[str]) -> np.ndarray:
        torch = self._torch
        order = _length_order(self._tok, sents)  # length-bucket: pad near each batch's own length
        ordered = [sents[j] for j in order]
        out = []
        with torch.no_grad():
            for i in range(0, len(ordered), EMBED_BATCH):
                batch = ordered[i : i + EMBED_BATCH]
                enc = self._tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=MAX_TOKENS,
                    return_tensors="pt",
                ).to(self._dev)
                hidden = self._enc(**enc).last_hidden_state.float()
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, dim=1)
                out.append(pooled.cpu().numpy().astype(np.float32))
        emb = np.concatenate(out, 0)
        result = np.empty_like(emb)
        result[order] = emb  # scatter back to input order - row i maps to sents[i]
        return result


def load_encoder(backend: str = "openvino", offline: bool = True, device: str | None = None):
    """Factory: return an encoder for ``backend`` in {openvino, torch}; ``device`` forces a torch device."""
    if backend == "openvino":
        return OpenVINOEncoder(offline=offline)
    if backend == "torch":
        return TorchEncoder(offline=offline, device=device)
    raise ValueError(f"unknown backend {backend!r}; choose 'openvino' or 'torch'")


def download_models(backend: str = "openvino") -> list[str]:
    """Download and cache the models for ``backend`` in {openvino, torch, both}.

    The only function that fetches from the Hub. huggingface_hub renders TQDM download bars.
    """
    _require_models_extra()
    _set_hf_token()
    os.environ.pop("HF_HUB_OFFLINE", None)  # force online for the install step

    logger.info("downloading SAT segmenter '{}'", config.SAT_MODEL)
    with contextlib.redirect_stderr(io.StringIO()):
        from wtpsplit import SaT

        SaT(config.SAT_MODEL)

    backends = ["openvino", "torch"] if backend == "both" else [backend]
    if "openvino" in backends:
        if (config.MMBERT_OPENVINO_LOCAL / "openvino_model.xml").exists():
            logger.info(
                "openvino INT8 encoder already present at {}", config.MMBERT_OPENVINO_LOCAL
            )
        else:
            logger.info("downloading '{}'", config.MMBERT_OPENVINO_HF)
            from huggingface_hub import snapshot_download

            snapshot_download(config.MMBERT_OPENVINO_HF)
    if "torch" in backends:
        logger.info("downloading '{}'", config.MMBERT_TORCH_MODEL)
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        conf = AutoConfig.from_pretrained(config.MMBERT_TORCH_MODEL)
        conf.reference_compile = False
        AutoTokenizer.from_pretrained(config.MMBERT_TORCH_MODEL)
        AutoModel.from_pretrained(
            config.MMBERT_TORCH_MODEL, config=conf, attn_implementation="eager"
        )

    logger.success("models ready for backend(s): {}", ", ".join(backends))
    return backends
