"""Model-backed segmentation and embedding.

The heavy dependencies (torch, transformers, openvino, wtpsplit, huggingface_hub) are imported
lazily inside the functions, so the pure-numpy :mod:`docdistance.distance` core stays
importable - and unit-testable - without them.

Inference never downloads. With ``offline=True`` each constructor holds the HuggingFace Hub library
in offline mode while its model loads (:func:`_hub_offline`) and resolves the model from the init
mirror (``docdistance.json`` model paths), then the local dev dir, then the HuggingFace cache; a
model missing from all three raises :class:`ModelsNotInstalled` pointing at ``docdistance init``.
Downloading happens only in :func:`docdistance.bootstrap.init`, which the ``init`` CLI command calls.
"""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
from html.parser import HTMLParser
import io
import os
from pathlib import Path
import re
import sys
import threading

from loguru import logger
import numpy as np

from docdistance import config

# keep transformers quiet before it is ever imported - it otherwise prints a model LOAD REPORT
# and advisory warnings (e.g. dropped LM-head keys) that leak past stderr redirection
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# block OpenVINO's usage telemetry before openvino is ever imported - `import openvino` otherwise
# sends a Google Analytics event through a multiprocessing.Process, and under the forkserver start
# method (the Linux default from Python 3.14) that child re-imports the main module, so a script
# without a main guard runs twice. A None entry makes `import openvino_telemetry` raise ImportError,
# on which OpenVINO falls back to its bundled no-op stub (openvino.tools.ovc.telemetry_stub)
sys.modules.setdefault("openvino_telemetry", None)

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


_hub_lock = threading.Lock()
_hub_scopes = 0  # scopes of _hub_flag open now, on all threads
_hub_before = False  # the value of the flag before the first of them started


@contextlib.contextmanager
def _hub_flag(offline: bool) -> Iterator[None]:
    """Hold the offline flag of the HuggingFace Hub library at ``offline`` until the scope ends.

    The library reads ``HF_HUB_OFFLINE`` from the environment once, when it is imported, and
    :func:`_require_models_extra` imports it, so a change of the variable after that import changes
    nothing. Every request reads ``huggingface_hub.constants.HF_HUB_OFFLINE``, the flag this scope
    sets, whichever module imported the library first.

    The flag belongs to the process, so the scopes of all threads are counted under one lock: the
    value the flag had before the first scope started is put back when the last scope ends. Two
    loads that overlap on two threads both stay offline until the later one ends. A scope with the
    opposite value that overlaps (a model load during ``docdistance init``) sets the flag for both
    from its start until the last scope ends.
    """
    global _hub_scopes, _hub_before
    from huggingface_hub import constants

    with _hub_lock:
        if _hub_scopes == 0:
            _hub_before = constants.HF_HUB_OFFLINE
        _hub_scopes += 1
        constants.HF_HUB_OFFLINE = offline
    try:
        yield
    finally:
        with _hub_lock:
            _hub_scopes -= 1
            if _hub_scopes == 0:
                constants.HF_HUB_OFFLINE = _hub_before


def _hub_offline(offline: bool) -> contextlib.AbstractContextManager[None]:
    """Hold the HuggingFace Hub library in offline mode while a model loads (DEF-DEPS-14).

    With ``offline`` true the load reads the local cache only (:func:`_hub_flag`); the mode the
    library had is put back afterwards, so a later ``docdistance init`` in the same process still
    reaches the Hub. With ``offline`` false the mode of the library is left as it is.
    """
    return _hub_flag(True) if offline else contextlib.nullcontext()


def _resolve_ov_dir(key: str, hf_repo: str, local_fallback: Path) -> Path:
    """Resolve an OpenVINO IR dir for ``key``: init mirror -> local dev dir -> HuggingFace snapshot.

    The init mirror (``docdistance.json`` model paths) wins so a provisioned S3 / local model is used
    offline; otherwise fall back to the dev ``models/`` dir, then to the HuggingFace snapshot: the
    cached one when the caller holds the Hub library offline (:func:`_hub_offline`), a download
    otherwise.
    """
    from docdistance import settings

    mirror = settings.get().model_paths.get(key)
    if mirror and (Path(mirror) / "openvino_model.xml").exists():
        return Path(mirror)
    if local_fallback and (Path(local_fallback) / "openvino_model.xml").exists():
        return Path(local_fallback)
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
        from docdistance import settings

        sat_src = (
            settings.get().model_paths.get("sat") or config.SAT_MODEL
        )  # init mirror or HF name
        # an init mirror holds the tokenizer beside the weights (DEF-DEPS-17); without that file
        # wtpsplit loads the tokenizer by its Hub name, config.SAT_TOKENIZER, from the cache
        in_mirror = (Path(sat_src) / "tokenizer.json").exists()
        tokenizer = {"tokenizer_name_or_path": sat_src} if in_mirror else {}
        with contextlib.redirect_stderr(io.StringIO()), _hub_offline(offline):
            from wtpsplit import SaT

            try:
                self._sat = SaT(sat_src, **tokenizer)
            except Exception as exc:  # missing weights under offline mode
                raise ModelsNotInstalled(_INSTALL_HINT) from exc
        logger.debug("loaded SAT segmenter '{}'", sat_src)

    def split(self, text: str) -> list[str]:
        with contextlib.redirect_stderr(io.StringIO()):
            return [s.strip() for s in self._sat.split(text) if s.strip()]


# a heading or bullet marker matches with text after it or bare; a bare one leaves no text to embed.
# A bare numbered marker ("42.") stays prose: it can be the last word of a wrapped paragraph
_HEADING_RE = re.compile(r"^#{1,6}(?:\s+|$)")
_LIST_RE = re.compile(r"^(?:[-*+](?:\s+|$)|\d+[.)]\s+)")
# horizontal rule: 3 or more of one of - * _ with or without spaces between them
_RULE_RE = re.compile(r"^([-*_])(?:\s*\1){2,}$")


def split_markdown(text: str, prose_split) -> tuple[list[str], list[str]]:
    """Markdown-aware segmentation: one atomic statement per structural line, SaT only on prose.

    Each line is classified as heading, table row, list item, fenced-code line or prose. A
    structural line becomes one atomic statement: the reported text keeps the raw line, the
    embedded text strips the markup (heading hashes, list markers, table pipes). Fence delimiter
    lines, table separator rows (``|---|---|``) and marker-only lines carry no content and are
    dropped. A marker-only line is a bare heading marker (``###``), a bare bullet (``-``, ``*``,
    ``+``) or a horizontal rule (``---``, ``***``, ``___``, with or without spaces); it ends the
    prose paragraph before it. Prose paragraphs go through ``prose_split`` (the SaT segmenter's
    ``split``), where reported and embedded text coincide. Returns the parallel
    ``(reported, embedded)`` lists, row-aligned so ``embedded[i]`` is the encoder input for
    ``reported[i]``.
    """
    reported: list[str] = []
    embedded: list[str] = []
    prose: list[str] = []

    def flush_prose() -> None:
        if prose:
            for s in prose_split("\n".join(prose)):
                reported.append(s)
                embedded.append(s)
            prose.clear()

    def atomic(raw: str, core: str) -> None:
        core = core.strip()
        if core:  # a marker-only line embeds nothing - drop it
            reported.append(raw.rstrip())
            embedded.append(core)

    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            flush_prose()
            in_fence = not in_fence
            continue
        if in_fence:
            atomic(line, stripped)
            continue
        if not stripped or _RULE_RE.match(stripped):
            flush_prose()
            continue
        if _HEADING_RE.match(stripped):
            flush_prose()
            atomic(line, _HEADING_RE.sub("", stripped))
            continue
        if stripped.startswith("|"):
            flush_prose()
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(not c.strip("-: ") for c in cells):  # separator row, e.g. |---|---|
                continue
            atomic(line, " ".join(c for c in cells if c))
            continue
        marker = _LIST_RE.match(stripped)
        if marker:
            flush_prose()
            atomic(line, stripped[marker.end() :])
            continue
        prose.append(stripped)
    flush_prose()
    return reported, embedded


# the HTML reader: tags by what html_to_markdown writes for them. Every other tag is inline and
# leaves its text only
# tags dropped with what they hold. head is not one of them: a document may omit its end tag and
# the body tag, and the dropped region would then hold the whole document. title is dropped
# wherever it stands, since a document may omit the head tag around it
_HTML_DROPPED = frozenset({"script", "style", "template", "noscript", "svg", "title"})
_HTML_HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_HTML_LISTS = frozenset({"ul", "ol"})
_HTML_CELLS = frozenset({"td", "th"})
# tags whose text is a paragraph; hr holds no text and only ends the paragraph before it
_HTML_PARAGRAPHS = frozenset(
    {"p", "div", "blockquote", "section", "article", "dt", "dd", "figcaption", "caption", "hr"}
    | {"header", "footer", "main", "nav", "aside", "address", "form"}
    | {"figure", "details", "summary"}
)
# tags that open a block of their own kind: a heading line, a list-item line, a fenced block
_HTML_OWN_BLOCK = _HTML_HEADINGS | {"li", "pre"}
# tags that end the open block; a table row and its cells are handled apart from them
_HTML_BLOCKS = _HTML_OWN_BLOCK | _HTML_LISTS | _HTML_PARAGRAPHS


class _MarkdownWriter(HTMLParser):
    """The parser behind :func:`html_to_markdown`: one open block, one open table row, the lines."""

    def __init__(self) -> None:
        super().__init__()  # convert_charrefs is on: character references arrive decoded
        self.lines: list[str] = []
        self._text: list[str] = []  # text of the open block
        self._kind = "p"  # the open block: "p", "h1" to "h6", "li", "pre" or "cell"
        self._cells: list[str] = []  # closed cells of the open table row
        # the open lists, outermost first: the items opened so far in an ol, None for a ul
        self._lists: list[int | None] = []
        self._dropping = 0  # open elements of _HTML_DROPPED

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _HTML_DROPPED:
            self._dropping += 1
        if self._dropping:
            return
        if tag == "br":
            self._text.append("\n")
        elif tag == "tr":
            self.end_row()
        elif tag in _HTML_CELLS:
            self._end_block()
            self._kind = "cell"
        elif tag in _HTML_BLOCKS:
            self._break(tag, opens=tag if tag in _HTML_OWN_BLOCK else "p")
        if tag in _HTML_LISTS:
            self._lists.append(0 if tag == "ol" else None)
        elif tag == "li" and self._lists and self._lists[-1] is not None:
            self._lists[-1] += 1  # the number of this item; the start attribute is not read

    def handle_endtag(self, tag: str) -> None:
        if tag in _HTML_DROPPED:
            self._dropping = max(self._dropping - 1, 0)  # an end tag without a start drops nothing
        elif self._dropping:
            return
        elif tag in ("tr", "table"):
            self.end_row()
        elif tag in _HTML_BLOCKS:
            self._break(tag)
        if tag in _HTML_LISTS and self._lists:  # an end tag without a start closes no list
            self._lists.pop()

    def handle_data(self, data: str) -> None:
        if self._dropping:
            return
        # a paragraph is written on one line: a line break of its source text is whitespace, so
        # the line break a br adds is the only one its text holds
        self._text.append(data.replace("\n", " ") if self._kind == "p" else data)

    def _break(self, tag: str, opens: str = "p") -> None:
        """End the open block at the block tag ``tag`` and open a block of the kind ``opens``.

        A cell keeps every block inside it and a list item keeps the paragraphs inside it: there
        the tag is a space in the text and the block stays open.
        """
        if self._kind == "cell" or (self._kind == "li" and tag in _HTML_PARAGRAPHS):
            self._text.append(" ")
            return
        self._end_block()
        self._kind = opens

    def _end_block(self) -> None:
        """Write the open block as its markdown lines; the open block is a paragraph afterwards."""
        text, kind = "".join(self._text), self._kind
        self._text.clear()
        self._kind = "p"
        if kind == "pre":
            self.lines += ["```", *text.strip("\n").splitlines(), "```"]
        elif kind == "cell":
            self._cells.append(" ".join(text.replace("|", " ").split()))
        elif kind == "p":
            collapsed = (" ".join(line.split()) for line in text.split("\n"))
            lines = [line for line in collapsed if line]
            if lines:
                self.lines += [*lines, ""]
        elif text.split():
            marker = self._item_marker() if kind == "li" else "#" * int(kind[1])
            self.lines.append(" ".join([marker, *text.split()]))

    def _item_marker(self) -> str:
        """The marker of the open list item: its indentation, then its number or a dash.

        Two spaces per list around its own. The number and a full stop in an ``ol``; a dash in a
        ``ul`` and outside any list.
        """
        number = self._lists[-1] if self._lists else None
        return "  " * (len(self._lists) - 1) + ("-" if number is None else f"{number}.")

    def end_row(self) -> None:
        """End the open block and write the cells of the open table row as one pipe row."""
        self._end_block()
        if self._cells:
            self.lines.append("| " + " | ".join(self._cells) + " |")
            self._cells.clear()


def html_to_markdown(text: str) -> str:
    """Write the block structure of an HTML text as the markdown lines :func:`split_markdown` reads.

    The HTML reader is this conversion followed by the markdown reader, so the section scan and the
    fragment filters work on an HTML document as on a markdown one.

    - ``h1`` to ``h6``: the level in ``#``, a space, the text on one line
    - ``li``: a marker and the text on one line, indented by two spaces per list around its own,
      as a nested markdown list is. The marker is ``- `` in a ``ul``. In an ``ol`` it is the number
      of the item and a full stop (``1. ``, ``2. ``, ...): each ``ol`` counts its items from 1, a
      nested one as well, the ``start`` attribute is not read, and an item without text takes its
      number and writes no line. The text of an item ends where a list nested in it starts
    - ``tr``: ``| cell | cell |`` from its ``td`` and ``th`` texts; a ``|`` inside a cell becomes a
      space. No separator row is written: :func:`split_markdown` drops it
    - ``pre``: a fenced block with the lines of its text
    - ``p``, ``div``, ``blockquote``, ``section``, ``article``, ``dt``, ``dd``, ``figcaption``,
      ``caption``, ``header``, ``footer``, ``main``, ``nav``, ``aside``, ``address``, ``form``,
      ``figure``, ``details``, ``summary`` and text outside any block: a paragraph followed by a
      blank line
    - ``br``: a line break in the open block; ``hr``: the end of the paragraph before it
    - ``script``, ``style``, ``template``, ``noscript``, ``svg``, ``title`` (also outside a
      ``head``) and comments: dropped with what they hold. ``head`` is not dropped: a document may
      omit its end tag and the ``body`` tag. Of what a ``head`` holds, ``title``, ``script`` and
      ``style`` are dropped by themselves, and ``meta``, ``link`` and ``base`` hold no text
    - every other tag (``a``, ``em``, ``strong``, ``span``, ``code``, ``sup``, ``img``, ...): the
      text inside it only, so an ``img`` leaves nothing

    Character references are decoded. Whitespace inside a heading, a list item and a cell collapses
    to single spaces. A paragraph is written on one line with its whitespace collapsed to single
    spaces: the prose segmenter cuts at every line break, so the statements of an HTML document do
    not depend on how its source text is wrapped. A ``br`` is the only line break inside a
    paragraph, and a line left empty by two of them is removed. A ``p`` inside an ``li`` is text of
    that item, and every block inside a cell is text of that cell. U+FEFF, the byte order mark, is
    removed from the converted text.

    Malformed HTML raises nothing: the parser of the standard library is lenient, and an unclosed
    block ends at the next block or at the end of the text. The characters of a paragraph are not
    escaped, so a paragraph line that starts with a markdown marker is read by
    :func:`split_markdown` as that structure, and inline markup that a markdown document keeps as
    characters (``**bold**``) has no counterpart in the converted text. Text outside any block is
    one paragraph up to the next block, so a text without tags is converted to one line.
    """
    writer = _MarkdownWriter()
    writer.feed(text)
    writer.close()
    writer.end_row()
    return "\n".join(writer.lines).replace("\ufeff", "").strip("\n")


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

        with _hub_offline(offline):
            src = _resolve_ov_dir(
                "mmbert", config.MMBERT_OPENVINO_HF, config.MMBERT_OPENVINO_LOCAL
            )
            self._tok = AutoTokenizer.from_pretrained(str(src))
        core = ov.Core()
        model = core.read_model(str(src / "openvino_model.xml"))
        # 2nd input name is dropped to '74' (attention_mask) during conversion - feed positionally
        self._innames = [i.get_any_name() for i in model.inputs]
        self._cm = core.compile_model(model, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
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
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self._torch = torch
        self._dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with contextlib.redirect_stderr(io.StringIO()), _hub_offline(offline):
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

        with _hub_offline(offline):
            src = _resolve_ov_dir(
                "reranker", config.RERANKER_OPENVINO_HF, config.RERANKER_OPENVINO_LOCAL
            )
            self._tok = AutoTokenizer.from_pretrained(str(src))
        core = ov.Core()
        model = core.read_model(str(src / "openvino_model.xml"))
        self._innames = [i.get_any_name() for i in model.inputs]
        self._cm = core.compile_model(model, "CPU", {"PERFORMANCE_HINT": "THROUGHPUT"})
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
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self._dev == "cuda" else torch.float32
        with contextlib.redirect_stderr(io.StringIO()), _hub_offline(offline):
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

        with _hub_offline(offline):
            src = _resolve_ov_dir("nli", config.NLI_OPENVINO_HF, config.NLI_OPENVINO_LOCAL)
            self._tok = AutoTokenizer.from_pretrained(str(src))
        core = ov.Core()
        model = core.read_model(str(src / "openvino_model.xml"))
        self._innames = [i.get_any_name() for i in model.inputs]
        self._cm = core.compile_model(model, "CPU", {"PERFORMANCE_HINT": "THROUGHPUT"})
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
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self._dev == "cuda" else torch.float32
        with contextlib.redirect_stderr(io.StringIO()), _hub_offline(offline):
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
