"""Model-backed segmentation and embedding.

The heavy dependencies (torch, transformers, openvino, wtpsplit, huggingface_hub) are imported
lazily inside the functions, so the pure-numpy :mod:`docdistance.distance` core stays
importable - and unit-testable - without them.

Inference never downloads. The constructors set ``HF_HUB_OFFLINE=1`` and resolve each model from the
init mirror (``docdistance.json`` model paths), then the local dev dir, then the HuggingFace cache;
a model missing from all three raises :class:`ModelsNotInstalled` pointing at ``docdistance init``.
Downloading happens only in :func:`docdistance.bootstrap.init`, which the ``init`` CLI command calls.
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

_INSTALL_HINT = "model not found in cache - run:  docdistance init"
_EXTRA_HINT = "model dependencies missing - reinstall:  pip install --force-reinstall docdistance"
_GPU_HINT = (
    "GPU requested (--gpu) but GPU support is not secured - install the extra and a CUDA build:\n"
    "  pip install 'docdistance[gpu]'   (needs a CUDA-capable torch wheel + an NVIDIA driver)"
)


class ModelsNotInstalled(RuntimeError):
    """A required model is missing from the cache - run ``docdistance init``."""


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


def _resolve_ov_dir(key: str, hf_repo: str, local_fallback: Path, offline: bool) -> Path:
    """Resolve an OpenVINO IR dir for ``key``: init mirror -> local dev dir -> HuggingFace snapshot.

    The init mirror (``docdistance.json`` model paths) wins so a provisioned S3 / local model is used
    offline; otherwise fall back to the dev ``models/`` dir, then download from the Hub.
    """
    from docdistance import settings

    mirror = settings.get().model_paths.get(key)
    if mirror and (Path(mirror) / "openvino_model.xml").exists():
        return Path(mirror)
    if local_fallback and (Path(local_fallback) / "openvino_model.xml").exists():
        return Path(local_fallback)
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(hf_repo))
    except Exception as exc:
        raise ModelsNotInstalled(_INSTALL_HINT) from exc


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable elementwise sigmoid."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax over ``axis``."""
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class Segmenter:
    """SAT statement segmenter (wtpsplit ``sat-3l-sm``), CPU."""

    def __init__(self, offline: bool = True):
        _require_models_extra()
        _set_hf_token()
        if offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from docdistance import settings

        sat_src = (
            settings.get().model_paths.get("sat") or config.SAT_MODEL
        )  # init mirror or HF name
        with contextlib.redirect_stderr(io.StringIO()):
            from wtpsplit import SaT

            try:
                self._sat = SaT(sat_src)
            except Exception as exc:  # missing weights under offline mode
                raise ModelsNotInstalled(_INSTALL_HINT) from exc
        logger.debug("loaded SAT segmenter '{}'", sat_src)

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

        src = _resolve_ov_dir(
            "mmbert", config.MMBERT_OPENVINO_HF, config.MMBERT_OPENVINO_LOCAL, offline
        )
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


class _Reranker:
    """Shared reranker pair-scoring: build the doc x source grid, length-bucket, sigmoid the logits."""

    def _pair_logits(self, a: list[str], b: list[str]) -> np.ndarray:
        raise NotImplementedError

    def score_grid(self, docs: list[str], sources: list[str]) -> np.ndarray:
        """Relevance grid ``R[i, j] = sigmoid(logit(docs[i], sources[j]))``; shape ``[n_doc, n_src]``."""
        a = [d for d in docs for _ in sources]
        b = [s for _ in docs for s in sources]
        logits = self._pair_logits(a, b)
        return _sigmoid(logits).reshape(len(docs), len(sources)).astype(np.float32)


class OpenVINOReranker(_Reranker):
    """bge-reranker-v2-m3 INT8 OpenVINO cross-encoder (CPU). Sigmoid relevance per statement pair."""

    name = "openvino"

    def __init__(self, offline: bool = True):
        _require_models_extra()
        _set_hf_token()
        import openvino as ov
        from transformers import AutoTokenizer

        src = _resolve_ov_dir(
            "reranker", config.RERANKER_OPENVINO_HF, config.RERANKER_OPENVINO_LOCAL, offline
        )
        core = ov.Core()
        model = core.read_model(str(src / "openvino_model.xml"))
        self._innames = [i.get_any_name() for i in model.inputs]
        self._cm = core.compile_model(model, "CPU", {"PERFORMANCE_HINT": "THROUGHPUT"})
        self._tok = AutoTokenizer.from_pretrained(str(src))
        logger.debug("loaded OpenVINO INT8 reranker from {}", src)

    def _pair_logits(self, a: list[str], b: list[str]) -> np.ndarray:
        # length-bucket the flat pair list (E06-H25): pad near each batch's own length, scatter back
        lens = self._tok(a, b, truncation=True, max_length=config.RERANK_MAX_TOKENS)["input_ids"]
        order = sorted(range(len(a)), key=lambda i: len(lens[i]))
        out = np.zeros(len(a), dtype=np.float32)
        for i in range(0, len(order), config.RERANK_PAIR_BATCH):
            sel = order[i : i + config.RERANK_PAIR_BATCH]
            enc = self._tok(
                [a[k] for k in sel],
                [b[k] for k in sel],
                padding=True,
                truncation=True,
                max_length=config.RERANK_MAX_TOKENS,
                return_tensors="np",
            )
            feeds = {self._innames[0]: enc["input_ids"], self._innames[1]: enc["attention_mask"]}
            out[np.asarray(sel)] = self._cm(feeds)[self._cm.output(0)].reshape(-1)
        return out


class TorchReranker(_Reranker):
    """bge-reranker-v2-m3 PyTorch cross-encoder (GPU fp16 if available, else CPU fp32)."""

    name = "torch"

    def __init__(self, offline: bool = True, device: str | None = None):
        _require_models_extra()
        _set_hf_token()
        if offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self._dev == "cuda" else torch.float32
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                self._tok = AutoTokenizer.from_pretrained(config.RERANKER_TORCH_MODEL)
                m = AutoModelForSequenceClassification.from_pretrained(config.RERANKER_TORCH_MODEL)
            except Exception as exc:
                raise ModelsNotInstalled(_INSTALL_HINT) from exc
        self._m = m.to(self._dev).to(dtype).eval()

    def _pair_logits(self, a: list[str], b: list[str]) -> np.ndarray:
        torch = self._torch
        lens = self._tok(a, b, truncation=True, max_length=config.RERANK_MAX_TOKENS)["input_ids"]
        order = sorted(range(len(a)), key=lambda i: len(lens[i]))
        out = np.zeros(len(a), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(order), config.RERANK_PAIR_BATCH):
                sel = order[i : i + config.RERANK_PAIR_BATCH]
                enc = self._tok(
                    [a[k] for k in sel],
                    [b[k] for k in sel],
                    padding=True,
                    truncation=True,
                    max_length=config.RERANK_MAX_TOKENS,
                    return_tensors="pt",
                ).to(self._dev)
                out[np.asarray(sel)] = self._m(**enc).logits.float().cpu().numpy().reshape(-1)
        return out


def _entail_index(id2label: dict) -> int:
    """The logit index labelled 'entailment' (mDeBERTa: 0); default 0 if labels are absent."""
    for k, v in id2label.items():
        if "entail" in str(v).lower():
            return int(k)
    return 0


class OpenVINONLI:
    """mDeBERTa-v3 MNLI/XNLI INT8 OpenVINO head (CPU). Returns P(entail) for premise -> hypothesis."""

    name = "openvino"

    def __init__(self, offline: bool = True):
        _require_models_extra()
        _set_hf_token()
        import json

        import openvino as ov
        from transformers import AutoTokenizer

        src = _resolve_ov_dir("nli", config.NLI_OPENVINO_HF, config.NLI_OPENVINO_LOCAL, offline)
        core = ov.Core()
        model = core.read_model(str(src / "openvino_model.xml"))
        self._innames = [i.get_any_name() for i in model.inputs]
        self._cm = core.compile_model(model, "CPU", {"PERFORMANCE_HINT": "THROUGHPUT"})
        self._tok = AutoTokenizer.from_pretrained(str(src))
        cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
        self._entail = _entail_index(cfg.get("id2label", {}))
        logger.debug("loaded OpenVINO INT8 NLI from {} (entail idx {})", src, self._entail)

    def entail(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(premises), config.RERANK_PAIR_BATCH):
            enc = self._tok(
                premises[i : i + config.RERANK_PAIR_BATCH],
                hypotheses[i : i + config.RERANK_PAIR_BATCH],
                padding=True,
                truncation=True,
                max_length=config.RERANK_MAX_TOKENS,
                return_tensors="np",
            )
            feeds = {self._innames[0]: enc["input_ids"], self._innames[1]: enc["attention_mask"]}
            logits = self._cm(feeds)[self._cm.output(0)]
            out.append(_softmax(logits, axis=1)[:, self._entail])
        return np.concatenate(out, 0).astype(np.float32)


class TorchNLI:
    """mDeBERTa-v3 MNLI/XNLI PyTorch head (GPU fp16 if available, else CPU fp32)."""

    name = "torch"

    def __init__(self, offline: bool = True, device: str | None = None):
        _require_models_extra()
        _set_hf_token()
        if offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self._dev == "cuda" else torch.float32
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                self._tok = AutoTokenizer.from_pretrained(config.NLI_TORCH_MODEL)
                m = AutoModelForSequenceClassification.from_pretrained(config.NLI_TORCH_MODEL)
            except Exception as exc:
                raise ModelsNotInstalled(_INSTALL_HINT) from exc
        self._m = m.to(self._dev).to(dtype).eval()
        self._entail = _entail_index({int(k): v for k, v in m.config.id2label.items()})

    def entail(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        torch = self._torch
        out = []
        with torch.no_grad():
            for i in range(0, len(premises), config.RERANK_PAIR_BATCH):
                enc = self._tok(
                    premises[i : i + config.RERANK_PAIR_BATCH],
                    hypotheses[i : i + config.RERANK_PAIR_BATCH],
                    padding=True,
                    truncation=True,
                    max_length=config.RERANK_MAX_TOKENS,
                    return_tensors="pt",
                ).to(self._dev)
                probs = torch.softmax(self._m(**enc).logits.float(), dim=1)
                out.append(probs[:, self._entail].cpu().numpy())
        return np.concatenate(out, 0).astype(np.float32)


def load_encoder(backend: str = "openvino", offline: bool = True, device: str | None = None):
    """Factory: return an encoder for ``backend`` in {openvino, torch}; ``device`` forces a torch device."""
    if backend == "openvino":
        return OpenVINOEncoder(offline=offline)
    if backend == "torch":
        return TorchEncoder(offline=offline, device=device)
    raise ValueError(f"unknown backend {backend!r}; choose 'openvino' or 'torch'")


def load_reranker(backend: str = "openvino", offline: bool = True, device: str | None = None):
    """Factory: return the grounding reranker for ``backend`` in {openvino, torch}."""
    if backend == "openvino":
        return OpenVINOReranker(offline=offline)
    if backend == "torch":
        return TorchReranker(offline=offline, device=device)
    raise ValueError(f"unknown backend {backend!r}; choose 'openvino' or 'torch'")


def load_nli(backend: str = "openvino", offline: bool = True, device: str | None = None):
    """Factory: return the grounding NLI head for ``backend`` in {openvino, torch}."""
    if backend == "openvino":
        return OpenVINONLI(offline=offline)
    if backend == "torch":
        return TorchNLI(offline=offline, device=device)
    raise ValueError(f"unknown backend {backend!r}; choose 'openvino' or 'torch'")
