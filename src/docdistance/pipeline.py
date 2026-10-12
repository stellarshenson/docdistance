"""High-level document-distance API.

Two entry styles:

- :class:`DocDistance` - load the models once, score many pairs (the pipeline-integration entry)
- :func:`semantic_distance` / :func:`source_conditioned_distance` - one-shot convenience that loads
  and scores in a single call

Inputs are raw text or a path to a text/markdown file (auto-detected). Every markdown ``# `` line
outside a fenced code block is stripped from files so titles do not count as statements; deeper
headings (``## `` and below) are kept, since they carry section content, and ``# `` comment lines
inside fences are kept verbatim. Files are read as UTF-8.

``input_format`` names the reader of every document of a call (:data:`INPUT_FORMATS`): ``plain``
gives the whole text to the prose segmenter; ``markdown`` and ``html`` read the block structure;
``auto``, the default, chooses one of the three per document from its file extension. With the
default the numbers for ``.md``, ``.markdown``, ``.html`` and ``.htm`` files can differ from 1.1.5,
which read every document with the plain reader; ``input_format="plain"`` returns them.
"""

from __future__ import annotations

from bisect import bisect_right
from importlib import metadata
import os
from pathlib import Path
import re
from stat import S_ISREG

from loguru import logger
import numpy as np

from docdistance import config, settings
from docdistance import distance as _core
from docdistance.distance import (
    ANISOTROPY_MIN_POOL,
    DEFAULT_THRESHOLD,
    MATCH_CUTOFF,
    SMD_MAX,
    SOURCE_SUPPORT_CUTOFF,
    STRUCTURE_THRESHOLD,
    DistanceResult,
    SourceConditionedResult,
    StructuralResult,
)
from docdistance.encoders import (
    Segmenter,
    html_to_markdown,
    load_encoder,
    load_nli,
    load_reranker,
    split_markdown,
)

# the values of ``input_format``. ``plain`` is the prose segmenter on the whole text, ``markdown``
# is :func:`~docdistance.encoders.split_markdown`, ``html`` is
# :func:`~docdistance.encoders.html_to_markdown` followed by the markdown reader, and ``auto``, the
# default, is one of the three, chosen per document by :func:`_reader_by_extension`
INPUT_FORMATS = ("plain", "markdown", "html", "auto")

# per-statement "changed" cutoff: a ground cost above the shipped closeness threshold (heuristic).
# Derived, so it tracks DEFAULT_THRESHOLD: the 0.725 -> 0.77 recalibration moves it 0.389 -> 0.325,
# so MORE flows are flagged "changed" on real documents.
DIFF_CHANGED_COST = (1.0 - DEFAULT_THRESHOLD) * SMD_MAX

# per-statement "no counterpart" cutoff on the [0, sqrt(2)] ground-cost scale: a cheapest match
# above 0.6 sits at closeness ~0.58, far below DIFF_CHANGED_COST's ~0.77 "same content" line -
# balanced OT still ships the statement's mass somewhere, but that target is an absorber forced
# by the marginals, not a counterpart. Annotation only: the transport plan and smd stay the exact
# balanced (metric) optimal transport.
NO_COUNTERPART_COST = 0.6

# "moved" tolerance (E15-H80). A statement reads moved when it is matched - in a one-to-one pair of
# partial_matching, ground cost below MATCH_CUTOFF - its matched_displacement is not 0, the norm
# abs(matched_displacement / n_matched) is above this tolerance, and its record is not
# no_counterpart. matched_displacement is the rank of the matched target among the matched targets
# minus the rank of the statement among the matched statements; n_matched is the number of pairs.
# These ranks do not change when a statement without a pair is added or removed, and no tie of the
# exact plan is read (DEF-TRANS-13). On a fully matched pair of equal statement counts the flag
# equals the E12-H61 conjunction displacement != 0 and abs(displacement_norm) > MOVED_TOLERANCE
# (1680 of 1680 permutations of 11 to 14 statements), with one exception: a rank step of exactly
# 5% of the count (1 of 20, 2 of 40) reads not moved, where target_index/n_b - index/n_a exceeds
# 0.05 at some positions through float rounding (183 of 190 and 765 of 780 single transpositions
# equal at 20 and 40 statements).
MOVED_TOLERANCE = 0.05

# bound, in entries, of each of the two memos of a DocDistance in a ``--pairs`` batch (E12-H67,
# E12-H68): the statements and embeddings of at most this many document files, and the grounding
# results of at most this many (document file, source file) pairs. Beyond the bound the least
# recently used entry is evicted. A sweep over all pairs of N documents encodes each document
# once while N is at most the bound (N + 1 with a source). An entry of the first memo holds
# 4 * dim bytes per statement, about 3 kB at 768 dimensions.
MEMO_SIZE = 64

# the three states of the one-to-one matching, one line each in the ``definitions`` block of both
# detail files
_MATCH_DEFINITIONS = {
    "kept": (
        "in a pair of the one-to-one matching of A and B statements: pair ground cost < "
        "match_cutoff on the embeddings as passed in, before any anisotropy pre-pass"
    ),
    "removed": "statement of A in no pair of the one-to-one matching",
    "added": "statement of B in no pair of the one-to-one matching",
}

# stamped into every detail file so a reader knows what wrote it and against which release
try:
    _VERSION = metadata.version("docdistance")
except metadata.PackageNotFoundError:  # source checkout without an installed distribution
    _VERSION = "unknown"

# cheapest-row-cost bands of the header summary; the last band is open-ended (> 0.6)
_COST_BANDS = ((0.1, "0-0.1"), (0.35, "0.1-0.35"), (0.6, "0.35-0.6"))
_TOP_CHANGED_N = 10


def _norm_text(text: str) -> str:
    """The ``identical`` normalisation: casefold + whitespace collapse."""
    return " ".join(text.casefold().split())


def _strip_titles(text: str) -> str:
    # hash-space stripping is title stripping only: inside a fenced code block a "# " line
    # is a bash/python comment, not a heading, so fence state gates the filter
    kept = []
    in_fence = False
    for ln in text.splitlines():
        stripped = ln.lstrip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            kept.append(ln)
            continue
        if not in_fence and ln.startswith("# "):
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


def _load_body(path: Path) -> str:
    # explicit utf-8: the default is the locale encoding, which silently mojibakes an
    # accented document on a cp1252 host and raises on an ASCII one
    return _strip_titles(path.read_text(encoding="utf-8"))


# extensions that make a non-resolving single-line argument look like a mistyped file path
_PATHLIKE_SUFFIXES = (".md", ".markdown", ".txt", ".rst", ".html", ".htm")


def _looks_like_path(doc: str) -> bool:
    # "<" opens a tag: every closing tag of a one-line HTML text holds "/", and it is no path
    if "\n" in doc or len(doc) >= 256 or "<" in doc:
        return False
    return "/" in doc or "\\" in doc or doc.lower().endswith(_PATHLIKE_SUFFIXES)


def _file_of(doc: str | Path) -> Path | None:
    """The file a document argument names; ``None`` when the argument is raw text."""
    if isinstance(doc, Path):
        return doc
    if isinstance(doc, str):
        try:
            p = Path(doc)
            if p.exists() and p.is_file():
                return p
        except OSError:
            pass
        if _looks_like_path(doc):
            logger.warning(
                "document argument {!r} looks like a file path but no such file exists; "
                "treating it as raw text",
                doc,
            )
        return None
    raise TypeError(f"document must be str or Path, got {type(doc).__name__}")


def _read(doc: str | Path) -> str:
    """Resolve a document argument: an existing file path is read, anything else is treated as text."""
    path = _file_of(doc)
    return doc.strip() if path is None else _load_body(path)


def _requested_format(markdown: bool, input_format: str) -> str:
    """The reader a call asks for: ``input_format``, or ``"markdown"`` when ``markdown`` is set.

    ``markdown=True`` alone equals ``input_format="markdown"``. Raises ``ValueError`` for a value
    outside :data:`INPUT_FORMATS`, and for ``markdown=True`` beside ``"plain"`` or ``"html"``,
    which name two readers. ``"auto"`` is the default of every call, so ``markdown=True`` beside
    it is ``markdown=True`` alone.
    """
    if input_format not in INPUT_FORMATS:
        raise ValueError(
            f"unknown input_format {input_format!r}; choose one of {', '.join(INPUT_FORMATS)}"
        )
    if not markdown:
        return input_format
    if input_format in ("plain", "html"):
        raise ValueError(
            f"markdown=True and input_format={input_format!r} name two readers; pass one of them"
        )
    return "markdown"


# the reader ``auto`` chooses for a file, by its extension in lower case. A file with another
# extension, a file without extension and raw text are read by the plain reader
_READER_BY_EXTENSION = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".html": "html",
    ".htm": "html",
}


def _reader_by_extension(path: Path | None) -> str:
    """The reader ``auto`` chooses for a document: ``plain``, ``markdown`` or ``html``.

    ``path`` is the file the document argument names, ``None`` for raw text, which reads
    ``plain``. The extension is the part of the file name from its last dot, compared without
    letter case (:data:`_READER_BY_EXTENSION`): ``a.md.txt`` reads ``plain``, and so does a name
    whose only dots stand at its start, such as ``.md``. The content of the file is not read.
    """
    if path is None:
        return "plain"
    return _READER_BY_EXTENSION.get(os.path.splitext(path)[1].lower(), "plain")


def _read_as(doc: str | Path, requested: str) -> tuple[str, str]:
    """The text a reader segments and the reader that segments it: ``plain``, ``markdown`` or ``html``.

    ``requested`` is a value of :data:`INPUT_FORMATS`. ``plain`` and ``markdown`` read the document
    as :func:`_read` does. ``auto`` chooses the reader with :func:`_reader_by_extension` and reads
    the same text for ``plain`` and ``markdown``. Under ``html`` the returned text is the output of
    :func:`~docdistance.encoders.html_to_markdown`, the text the markdown reader and the section
    scan then read. The title lines of a file are removed after that conversion: the ``h1`` of an
    HTML file is no statement, like the ``# `` line of a markdown file, and a ``# `` line inside
    a ``pre`` block is kept, like the one inside a fence.
    """
    if requested in ("plain", "markdown"):
        return _read(doc), requested
    path = _file_of(doc)
    reader = _reader_by_extension(path) if requested == "auto" else "html"
    text = doc if path is None else path.read_text(encoding="utf-8")
    if reader == "html":
        text = html_to_markdown(text)
    return (text.strip() if path is None else _strip_titles(text)), reader


def _memo_key(
    doc: str | Path, input_format: str, min_words: int, skip_table_fragments: bool
) -> tuple | None:
    """The memo key of a document argument, ``None`` when the argument does not name a file.

    ``(resolved path, mtime in ns, size, input_format, min_words, skip_table_fragments, reader of
    the name)``: the file identity, the three segmentation arguments and the reader that
    :func:`_reader_by_extension` chooses for the name as given, which decide the statements and
    the drop count. ``input_format`` is the requested value, so ``auto`` and the reader it chooses
    for a file are two entries. Two names of one file are two entries when their extensions
    choose two readers: a symbolic link ``notes.txt`` to ``a.md`` resolves to the path of
    ``a.md``, and ``auto`` reads each name with the reader of its own extension. A file that is
    rewritten has a new mtime or a new size, so its earlier entry is not served again. The limit:
    a rewrite of equal byte size inside one timestamp step of the file system keeps the key, and
    the memo then serves the earlier content. Raw text has no such identity and is never
    memoised; :func:`_read` decides what an argument without a key is.
    """
    try:
        path = Path(doc)
        stat = path.stat()
    except (OSError, ValueError, TypeError):  # no such file, or not usable as a path
        return None
    if not S_ISREG(stat.st_mode):  # a directory is read as raw text
        return None
    return (
        str(path.resolve()),
        stat.st_mtime_ns,
        stat.st_size,
        input_format,
        min_words,
        skip_table_fragments,
        _reader_by_extension(path),
    )


class _Memo:
    """Store of at most ``bound`` entries that evicts the least recently used; ``bound`` 0 keeps none."""

    def __init__(self, bound: int):
        self.bound = bound
        self._entries: dict = {}

    def get(self, key):
        """The entry under ``key``, which becomes the most recently used; ``None`` without one."""
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._entries[key] = entry  # last position; the first key is the least recently used
        return entry

    def put(self, key, entry) -> None:
        """Keep ``entry`` under ``key``, then evict the least recently used entry beyond the bound."""
        self._entries[key] = entry
        if len(self._entries) > self.bound:
            del self._entries[next(iter(self._entries))]


# any markdown heading line - the section scan tracks all levels present in the body
# (``# `` survives raw-text input; _load_body strips it from files)
_SECTION_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _statement_sections(body: str, statements: list[str]) -> list[str]:
    """Per statement, the nearest preceding heading chain (e.g. ``"Delivery options > Option A"``).

    One fence-aware scan of the body collects the heading lines (a ``# `` line inside a fenced
    code block is a comment, not a heading) and a whitespace-normalised copy of the text; each
    statement is then located by sequential search - segmentation preserves order and content,
    for both plain SaT and the markdown segmenter - and tagged with the chain of the last
    heading at or before its offset, so a heading statement carries its own chain. Empty string
    before the first heading; a statement the search cannot locate inherits the chain at the
    search cursor.
    """
    headings: list[tuple[int, str]] = []  # (normalised offset, heading chain)
    parts: list[str] = []
    length = 0  # running length of the normalised body
    stack: list[tuple[int, str]] = []  # (level, title) of the open headings
    in_fence = False
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
        m = None if in_fence else _SECTION_HEADING_RE.match(stripped)
        if m:
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2).strip()))
            headings.append((length, " > ".join(title for _, title in stack)))
        norm = " ".join(stripped.split())
        if norm:
            parts.append(norm)
            length += len(norm) + 1  # the joining space
    norm_body = " ".join(parts)
    offsets = [off for off, _ in headings]

    def chain_at(pos: int) -> str:
        k = bisect_right(offsets, pos)
        return headings[k - 1][1] if k else ""

    sections: list[str] = []
    cursor = 0
    for text in statements:
        needle = " ".join(text.split())
        pos = norm_body.find(needle, cursor) if needle else -1
        if pos < 0:
            sections.append(chain_at(cursor))
            continue
        sections.append(chain_at(pos))
        cursor = pos + len(needle)
    return sections


def _section_rows(sections: list[str]) -> list[tuple[str, list[int]]]:
    """Group statement indices by section, in first-appearance (document) order."""
    order: list[str] = []
    groups: dict[str, list[int]] = {}
    for i, sec in enumerate(sections):
        if sec not in groups:
            groups[sec] = []
            order.append(sec)
        groups[sec].append(i)
    return [(sec, groups[sec]) for sec in order]


def _is_table_fragment(text: str) -> bool:
    """A table fragment: carries a pipe, or is only table/separator punctuation (``-``, ``:``, ``|``)."""
    t = text.strip()
    return "|" in t or (bool(t) and not t.strip("-:| "))


def _filter_fragments(
    reported: list[str],
    embedded: list[str],
    *,
    min_words: int = 0,
    skip_table_fragments: bool = False,
) -> tuple[list[str], list[str], int]:
    """Drop fragment statements after segmentation, before encoding; returns survivors + drop count.

    ``min_words`` drops statements whose embedded text has fewer than that many words;
    ``skip_table_fragments`` drops statements whose reported text is a table fragment. A no-op at
    the defaults. Dropping renormalises the uniform transport marginals (each statement carries
    1/n of the mass) over the survivors, so the SMD is computed over the filtered set and differs
    from the unfiltered value.
    """
    if min_words <= 0 and not skip_table_fragments:
        return reported, embedded, 0
    keep_r: list[str] = []
    keep_e: list[str] = []
    for r, e in zip(reported, embedded):
        if min_words > 0 and len(e.split()) < min_words:
            continue
        if skip_table_fragments and _is_table_fragment(r):
            continue
        keep_r.append(r)
        keep_e.append(e)
    return keep_r, keep_e, len(reported) - len(keep_r)


def _seg_kwargs(input_format: str, min_words: int, skip_table_fragments: bool) -> dict:
    """Only the non-default segmentation kwargs - a default call keeps the original call shape.

    ``input_format`` is the reader a call asks for (:func:`_requested_format`); its default is
    ``"auto"``.
    """
    seg: dict = {}
    if input_format != "auto":
        seg["input_format"] = input_format
    if min_words > 0:
        seg["min_words"] = min_words
    if skip_table_fragments:
        seg["skip_table_fragments"] = True
    return seg


def _file_prefix(doc: str | Path) -> str:
    """The start of a log line about a document: its path as given and a colon when it names a file."""
    return f"{doc}: " if os.path.isfile(doc) else ""


def _warn_marker_statements(statements: list[str], doc: str | Path) -> None:
    """Log one warning when statements of the plain reader hold no letter and no digit (DEF-SEGM-12).

    The plain reader is the prose segmenter. It has no markdown rule, so a bare list or heading
    marker (``-``, ``###``) comes back as a statement of its own: 21 of the 70 statements of the ibm
    source article. Such statements repeat and each carries ``1/n`` of the transport mass. The
    warning names their count, the statement total and the two options that remove them: the
    markdown reader and the word filter. The statements are left as they are. When ``doc`` names a
    file the line starts with its path as given, so a ``--pairs`` run says which document it
    concerns; the line of a raw-text document carries no name.
    """
    n_markers = sum(not any(ch.isalnum() for ch in text) for text in statements)
    if n_markers:
        logger.warning(
            "{}{} of {} statements hold no letter and no digit (markdown markers such as '-' or "
            "'###') and are scored as statements; remove them with --markdown / markdown=True or "
            "--min-words / min_words",
            _file_prefix(doc),
            n_markers,
            len(statements),
        )


def _warn_small_pool(emb_a: np.ndarray, emb_b: np.ndarray) -> None:
    """Log one warning when an anisotropy pre-pass pool is below ``ANISOTROPY_MIN_POOL`` (ACC-GEOM-23).

    The pool is the statements of the two documents of a symmetric scoring call, the set
    :func:`~docdistance.distance.all_but_the_top` estimates the common direction from. Each such
    call has one pool and calls this once, so a detail call, which applies the pre-pass to that
    pool twice, logs once, and a ``--pairs`` run logs once per pair. The pre-pass still runs on the
    pool and every number is the one it gives. The source-conditioned calls log no such warning:
    their pool holds the source as well, and the minimum was measured on pools without one.
    """
    n_pooled = len(emb_a) + len(emb_b)
    if n_pooled < ANISOTROPY_MIN_POOL:
        logger.warning(
            "anisotropy pre-pass pool holds {} statements, below the minimum of {} pooled "
            "statements for estimating the common direction (E12-H65); the result is computed "
            "from this pool",
            n_pooled,
            ANISOTROPY_MIN_POOL,
        )


def _source_support(emb: np.ndarray, src_emb: np.ndarray) -> list[dict]:
    """Per statement, its nearest source statement and whether the source supports the statement.

    ``nearest_source_cost`` is the ground cost to the nearest source statement and ``unsupported``
    whether that cost exceeds ``SOURCE_SUPPORT_CUTOFF`` - the per-statement form of
    :func:`~docdistance.distance.unsupported_count`, a score and not a metric. Pass the embeddings
    before the anisotropy pre-pass: the cutoff holds on those only.
    """
    cost = _core.cost_matrix(emb, src_emb)
    return [
        {
            "nearest_source_index": int(j),
            "nearest_source_cost": round(float(row[j]), 4),
            "unsupported": bool(row[j] > SOURCE_SUPPORT_CUTOFF),
        }
        for row, j in zip(cost, cost.argmin(1))
    ]


def _align_statements(
    sents: list[str],
    emb: np.ndarray,
    src_sents: list[str],
    src_emb: np.ndarray,
    top_k: int,
    support: list[dict],
    sections: list[str] | None = None,
) -> list[dict]:
    """Per statement, its top-``k`` source statements by soft coverage weight.

    ``support`` carries each statement's nearest-source fields (:func:`_source_support`) onto its
    record. ``sections`` carries each statement's heading chain (empty strings when unknown).
    """
    secs = sections if sections is not None else [""] * len(sents)
    align = _core.coverage_alignment(emb, src_emb)  # [n, n_source], rows sum to 1
    k = min(top_k, align.shape[1])
    out = []
    for i, row in enumerate(align):
        top = np.argsort(row)[::-1][:k]
        matches = [
            {
                "source_index": int(j),
                "source_text": src_sents[j],
                "weight": round(float(row[j]), 4),
            }
            for j in top
        ]
        out.append(
            {"index": i, "text": sents[i], "section": secs[i], "matches": matches, **support[i]}
        )
    return out


def _build_source_map(
    sa: list[str],
    ea: np.ndarray,
    sb: list[str],
    eb: np.ndarray,
    ss: list[str],
    es: np.ndarray,
    *,
    anisotropy: bool = True,
    top_k: int = 3,
    n_dropped: dict | None = None,
    drop_rules: dict | None = None,
    sections_a: list[str] | None = None,
    sections_b: list[str] | None = None,
) -> dict:
    """A JSON-serializable map: each statement of A and B → its top-``k`` source statements.

    ``n_dropped`` / ``drop_rules`` record the pre-encoding fragment filter (zeros when off), so
    nothing is silently lost. ``sections_a`` / ``sections_b`` carry each statement's heading
    chain onto its record (empty strings when unknown). Anisotropy removal is applied the same
    way as the source-conditioned distance (over the pooled A/B/S statements), so the map
    reflects the same geometry the selection axis is scored on.
    Each record also names its nearest source statement (``nearest_source_index`` /
    ``nearest_source_cost``) and carries ``unsupported`` (:func:`_source_support`). These three are
    read on the embeddings as passed in, before the pre-pass, where ``SOURCE_SUPPORT_CUTOFF``
    holds; the header states the cutoff as ``source_support_cutoff`` and counts the unsupported
    statements per document in ``n_unsupported``, the same counts as ``n_unsupported_a`` /
    ``n_unsupported_b`` of the source-conditioned result.
    """
    # read before the pre-pass, on the embeddings SOURCE_SUPPORT_CUTOFF is calibrated on
    support_a, support_b = _source_support(ea, es), _source_support(eb, es)
    if anisotropy:
        fixed = _core.all_but_the_top({"a": ea, "b": eb, "s": es}, k=1)
        ea, eb, es = fixed["a"], fixed["b"], fixed["s"]
    return {
        "top_k": top_k,
        "anisotropy": anisotropy,
        "source_support_cutoff": SOURCE_SUPPORT_CUTOFF,
        "n_statements": {"a": len(sa), "b": len(sb), "source": len(ss)},
        "n_unsupported": {
            "a": sum(record["unsupported"] for record in support_a),
            "b": sum(record["unsupported"] for record in support_b),
        },
        "n_dropped": n_dropped or {"a": 0, "b": 0, "source": 0},
        "drop_rules": drop_rules or {"min_words": 0, "skip_table_fragments": False},
        "a": _align_statements(sa, ea, ss, es, top_k, support_a, sections=sections_a),
        "b": _align_statements(sb, eb, ss, es, top_k, support_b, sections=sections_b),
    }


def _statement_flows(
    sa: list[str], sb: list[str], plan: np.ndarray, cost: np.ndarray, eps: float = 1e-9
) -> list[dict]:
    """Per A statement, the B statements its transport mass flows to, by descending row-normalized weight."""
    out = []
    for i, row in enumerate(plan):
        total = row.sum()
        matches = [
            {
                "target_index": int(j),
                "target_text": sb[j],
                "weight": round(
                    float(row[j] / total), 4
                ),  # fraction of this statement's mass to B[j]
                "cost": round(float(cost[i, j]), 4),  # ground distance of the match
            }
            for j in np.argsort(row)[::-1]
            if row[j] > eps
        ]
        out.append({"index": i, "text": sa[i], "matches": matches})
    return out


def _build_semantic_details(
    sa: list[str],
    ea: np.ndarray,
    sb: list[str],
    eb: np.ndarray,
    *,
    anisotropy: bool = False,
    changed_cost: float = DIFF_CHANGED_COST,
    n_dropped: dict | None = None,
    drop_rules: dict | None = None,
    sections: list[str] | None = None,
    sections_b: list[str] | None = None,
) -> dict:
    """A JSON-serializable transport map: each statement of A → the B statements its OT mass flows to.

    The exact optimal-transport coupling behind the symmetric distance - ``weight`` is the fraction of
    statement A[i]'s mass landing on B[j], ``cost`` the ground distance of that match. Each flow carries
    a ``changed`` flag: whether the cheapest ground cost over the full cost row exceeds
    ``changed_cost`` (default ``DIFF_CHANGED_COST``) - the row minimum, not the heaviest match, because
    transport weight follows the marginal constraints while cost carries the content signal.
    ``no_counterpart`` marks flows whose cheapest cost exceeds ``NO_COUNTERPART_COST`` (no target is a
    plausible counterpart; the mass lands on an absorber), and ``identical`` whether the statement's
    normalised text (casefold + whitespace collapse) equals its cheapest-cost target's - so identical /
    edited-below-threshold / changed are three queryable states. All three are annotations: the plan
    and ``smd`` stay the exact balanced OT.
    The header is self-describing: ``generator`` / ``version`` stamp the writer, ``closeness`` restates
    ``smd`` on the CLI's closeness scale with its formula in ``closeness_definition``, ``thresholds``
    carries the cutoffs in force and ``definitions`` what each flag means against them. ``summary``
    aggregates the flows once for every consumer: cheapest-row-cost counts per band (``cost_bands``),
    ``n_changed`` / ``n_no_counterpart``, and ``top_changed`` - the statement indices with the highest
    cheapest-row cost, descending.
    Beside ``flows`` sits ``targets``, the target-side view: per B statement its cheapest incoming
    source (``best_source_index`` / ``min_incoming_cost``, the full cost-column minimum - the same
    row-vs-marginal reasoning as ``changed``) and ``total_incoming_weight``, the sum of the
    row-normalised flow weights landing on it - so added-content queries read directly instead of
    inverting the flows. ``sections`` carries each A statement's heading chain onto its flow record
    (empty strings when unknown) and ``sections_b`` does the same for the B-side ``targets``
    records; the ``sections`` header aggregates the flows per section - statement count, mean of
    the cheapest row cost and the share of statements at or below ``changed_cost``.
    ``n_dropped`` / ``drop_rules`` record the pre-encoding fragment filter (zeros when off) - dropped
    statements renormalise the marginals, so ``smd`` is computed over the filtered set.
    Anisotropy removal is applied the same way as :func:`~docdistance.distance.compute_distance`, so the
    map reflects the same geometry the distance is scored on.
    The one-to-one pairs of :func:`~docdistance.distance.partial_matching` (E13-H71) sit beside the
    plan: each flow carries ``match_index``, the B statement its A statement is paired with
    (``null`` without a pair), and ``match_status`` ``kept`` or ``removed``; each ``targets`` record
    carries ``match_source_index`` and ``match_status`` ``kept`` or ``added``; ``summary`` counts
    them in ``n_kept`` / ``n_removed`` / ``n_added``. The pairs are computed on the embeddings as
    passed in, before anisotropy removal, the geometry ``MATCH_CUTOFF`` is calibrated on, so they
    do not change with ``anisotropy``. They are an annotation that depends on that cutoff, not a
    distance: the plan and ``smd`` are untouched.
    """
    # matched before the pre-pass, on the embeddings MATCH_CUTOFF is calibrated on
    match_of_a = dict(_core.partial_matching(ea, eb).tolist())  # A index -> its paired B index
    match_of_b = {j: i for i, j in match_of_a.items()}
    if anisotropy:
        fixed = _core.all_but_the_top({"a": ea, "b": eb}, k=1)
        ea, eb = fixed["a"], fixed["b"]
    secs = sections if sections is not None else [""] * len(sa)
    secs_b = sections_b if sections_b is not None else [""] * len(sb)
    plan = _core.transport_plan(ea, eb)
    cost = _core.cost_matrix(ea, eb)
    flows = _statement_flows(sa, sb, plan, cost)
    min_costs: list[float] = []
    for i, flow in enumerate(flows):
        # the full-row minimum: a near-zero-cost counterpart clears the flags even when the
        # transport weight concentrates elsewhere
        min_cost = float(cost[i].min())
        min_costs.append(min_cost)
        flow["section"] = secs[i]
        flow["changed"] = bool(min_cost > changed_cost)
        flow["no_counterpart"] = bool(min_cost > NO_COUNTERPART_COST)
        # the cheapest-cost target, the same match ``changed`` is judged against
        flow["identical"] = _norm_text(sa[i]) == _norm_text(sb[int(cost[i].argmin())])
        flow["match_index"] = match_of_a.get(i)
        flow["match_status"] = "kept" if i in match_of_a else "removed"
    # target-side view: per B statement, its cheapest incoming source (full column minimum, the
    # mirror of the full-row minimum behind ``changed``) and the summed row-normalised inflow
    norm_plan = plan / plan.sum(axis=1, keepdims=True)  # rows as fractions, matching flow weights
    targets = [
        {
            "index": j,
            "text": sb[j],
            "section": secs_b[j],
            "best_source_index": int(cost[:, j].argmin()),
            "min_incoming_cost": round(float(cost[:, j].min()), 4),
            "total_incoming_weight": round(float(norm_plan[:, j].sum()), 4),
            "match_status": "kept" if j in match_of_b else "added",
            "match_source_index": match_of_b.get(j),
        }
        for j in range(len(sb))
    ]
    section_aggregates = []
    for sec, idxs in _section_rows(secs):
        costs = [min_costs[i] for i in idxs]
        section_aggregates.append(
            {
                "section": sec,
                "n_statements": len(idxs),
                "mean_cost": round(sum(costs) / len(costs), 4),
                "share_below_changed": round(
                    sum(c <= changed_cost for c in costs) / len(costs), 4
                ),
            }
        )
    cost_bands = {label: 0 for _, label in _COST_BANDS}
    cost_bands[f">{_COST_BANDS[-1][0]}"] = 0
    for c in min_costs:
        for edge, label in _COST_BANDS:
            if c <= edge:
                cost_bands[label] += 1
                break
        else:
            cost_bands[f">{_COST_BANDS[-1][0]}"] += 1
    smd_value = float((plan * cost).sum())
    return {
        "generator": "docdistance",
        "version": _VERSION,
        "smd": round(smd_value, 6),
        "closeness": round(_core.closeness(smd_value), 6),
        "closeness_definition": "1 - smd/sqrt(2)",
        "thresholds": {
            "changed_cost": changed_cost,
            "no_counterpart_cost": NO_COUNTERPART_COST,
            "match_cutoff": MATCH_CUTOFF,
        },
        "definitions": {
            "changed": "cheapest ground cost over the statement's full cost row > changed_cost",
            "no_counterpart": (
                "cheapest ground cost over the statement's full cost row > no_counterpart_cost"
            ),
            "identical": (
                "normalised text (casefold, whitespace collapsed) equals its "
                "cheapest-cost target's"
            ),
            "share_below_changed": (
                "share of a section's statements with cheapest row cost <= changed_cost"
            ),
            **_MATCH_DEFINITIONS,
        },
        "summary": {
            "cost_bands": cost_bands,
            "n_changed": sum(flow["changed"] for flow in flows),
            "n_no_counterpart": sum(flow["no_counterpart"] for flow in flows),
            "top_changed": sorted(range(len(min_costs)), key=lambda i: (-min_costs[i], i))[
                :_TOP_CHANGED_N
            ],
            "n_kept": len(match_of_a),
            "n_removed": len(sa) - len(match_of_a),
            "n_added": len(sb) - len(match_of_a),
        },
        "anisotropy": anisotropy,
        "n_statements": {"a": len(sa), "b": len(sb)},
        "n_dropped": n_dropped or {"a": 0, "b": 0},
        "drop_rules": drop_rules or {"min_words": 0, "skip_table_fragments": False},
        "sections": section_aggregates,
        "flows": flows,
        "targets": targets,
    }


def _build_structural_details(
    sa: list[str],
    ea: np.ndarray,
    sb: list[str],
    eb: np.ndarray,
    *,
    anisotropy: bool = False,
    n_dropped: dict | None = None,
    drop_rules: dict | None = None,
    sections_a: list[str] | None = None,
    sections_b: list[str] | None = None,
) -> dict:
    """A JSON-serializable structural (order) projection: per A statement, its aligned B statement + displacement.

    Content (``smd``) and structure (``order_gap``, the E11-H55 OPW order-gap) are reported separately;
    ``structure_closeness`` is the shipped SOTA readout ``closeness(order_gap)`` on the SMD scale
    (1 = same order). Each statement carries its crisp exact-EMD ``target`` and rank ``displacement`` from
    that alignment (the per-statement signal comes from the crisp coupling, not the soft OPW plan),
    plus ``displacement_norm`` - the target-relative position shift ``target_index/n_b - index/n_a``,
    which does not saturate when the documents differ in length. When the aligned pair's ground
    cost exceeds ``NO_COUNTERPART_COST`` the target is an absorber, not a location: the record
    carries ``no_counterpart: true`` with ``displacement_norm: null`` and ``moved: false``.
    ``moved`` is read on the one-to-one pairs of :func:`~docdistance.distance.partial_matching`
    (E15-H80), not on that alignment. ``matched_displacement`` is the rank of the statement's
    matched target among the matched targets minus the rank of the statement among the matched
    statements, ``null`` when the statement has no match, and ``moved`` is ``matched AND
    matched_displacement != 0 AND abs(matched_displacement / n_matched) > MOVED_TOLERANCE AND NOT
    no_counterpart``, with ``n_matched`` the number of pairs. A statement added or removed without
    a match changes no rank and flags nothing. On a document that repeats a statement this does
    not always hold: ``moved`` flagged a kept statement in 4 of 120 pure deletions of a
    62-statement document with 9 repeated statements (E15-H80, see
    :func:`~docdistance.distance.partial_matching`). The pairs are computed on the embeddings as
    passed in, before anisotropy removal, the geometry ``MATCH_CUTOFF`` is calibrated on, so
    ``matched_displacement`` does not change with ``anisotropy``. ``moved`` is an annotation that
    depends on that cutoff, not a distance.
    Each record also carries ``cost`` - the ground cost of the aligned pair, so moved-intact
    separates from vanished-with-absorption without joining the semantic file - and the heaviest
    target of the already-computed OT ``plan`` as ``semantic_target_index`` / ``semantic_target_cost``
    with ``targets_differ`` set when it disagrees with the structural ``target_index``; the
    disagreement locates moved-and-reworded content. Pure exposure: no extra solve, and neither
    distance changes.
    ``sections_a`` / ``sections_b`` carry each statement's heading chain onto its record as
    ``section`` / ``target_section`` (empty strings when unknown); the ``sections`` header
    aggregates the statements per A section - count, mean aligned-pair cost, share of statements
    at or below ``DIFF_CHANGED_COST``, median ``displacement_norm`` (null when every statement
    lacks a counterpart) and the share of counterpart-bearing statements whose target lies in a
    different section (null when none bears a counterpart).
    The header is self-describing: ``generator`` / ``version`` stamp the writer,
    ``structure_closeness_definition`` states the formula behind ``structure_closeness``,
    ``thresholds`` carries the cutoffs in force and ``definitions`` what each per-statement field
    means against them, with the three states of the matching (``kept``, ``removed``, ``added``).
    ``summary`` aggregates the statements once for every consumer -
    ``n_moved``, ``n_no_counterpart`` and the median ``displacement_norm`` over the
    counterpart-bearing statements (null when none bears a counterpart).
    ``n_dropped`` / ``drop_rules`` record the pre-encoding fragment filter (zeros when off) - dropped
    statements renormalise the marginals, so ``smd`` is computed over the filtered set.
    Anisotropy removal is applied as in :func:`_build_semantic_details`.
    """
    # matched before the pre-pass, on the embeddings MATCH_CUTOFF is calibrated on;
    # rows (A index, B index), ascending in the A index
    pairs = _core.partial_matching(ea, eb)
    n_matched = len(pairs)
    # per pair: rank of its B statement among the matched targets minus rank of its A statement
    # among the matched statements, which is the row of the pair
    target_ranks = np.argsort(np.argsort(pairs[:, 1]))
    matched_displacement = dict(
        zip(pairs[:, 0].tolist(), (target_ranks - np.arange(n_matched)).tolist())
    )
    if anisotropy:
        fixed = _core.all_but_the_top({"a": ea, "b": eb}, k=1)
        ea, eb = fixed["a"], fixed["b"]
    cost = _core.cost_matrix(ea, eb)
    plan = _core.transport_plan(ea, eb)
    align = _core.order_alignment(ea, eb)  # crisp exact-EMD alignment (diagonal tie-break)
    disp = _core.structure_displacement(ea, eb)  # rank shift; 0 = in place
    d_smd = float((plan * cost).sum())
    order_gap = _core.opw_gap(ea, eb)
    n_a, n_b = len(sa), len(sb)
    secs_a = sections_a if sections_a is not None else [""] * n_a
    secs_b = sections_b if sections_b is not None else [""] * n_b
    statements = []
    aligned_costs: list[float] = []
    for i in range(n_a):
        j = int(align[i])
        k = int(plan[i].argmax())  # heaviest semantic target of the already-computed OT plan
        shift = matched_displacement.get(i)  # None: the statement is in no pair
        aligned_costs.append(float(cost[i, j]))
        record = {
            "index": i,
            "text": sa[i],
            "section": secs_a[i],
            "target_index": j,
            "target_text": sb[j],
            "target_section": secs_b[j],
            "cost": round(float(cost[i, j]), 4),  # ground cost of the aligned pair
            "semantic_target_index": k,
            "semantic_target_cost": round(float(cost[i, k]), 4),
            "targets_differ": bool(k != j),
            "displacement": int(disp[i]),  # raw rank shift, kept for continuity; 0 = in place
            "matched_displacement": shift,  # rank shift among the matched pairs
        }
        if float(cost[i, j]) > NO_COUNTERPART_COST:
            # the aligned target absorbs the mass without being a counterpart: it names no
            # position, so the normalised displacement is undefined and the statement did not move
            record["displacement_norm"] = None
            record["moved"] = False
            record["no_counterpart"] = True
        else:
            record["displacement_norm"] = round(j / n_b - i / n_a, 4)
            record["moved"] = (
                shift is not None and shift != 0 and abs(shift / n_matched) > MOVED_TOLERANCE
            )
            record["no_counterpart"] = False
        statements.append(record)
    section_aggregates = []
    for sec, idxs in _section_rows(secs_a):
        costs = [aligned_costs[i] for i in idxs]
        recs = [statements[i] for i in idxs]
        norms = [r["displacement_norm"] for r in recs if r["displacement_norm"] is not None]
        with_target = [r for r in recs if not r["no_counterpart"]]
        section_aggregates.append(
            {
                "section": sec,
                "n_statements": len(idxs),
                "mean_cost": round(sum(costs) / len(costs), 4),
                "share_below_changed": round(
                    sum(c <= DIFF_CHANGED_COST for c in costs) / len(costs), 4
                ),
                "median_displacement_norm": round(float(np.median(norms)), 4) if norms else None,
                "share_target_other_section": round(
                    sum(r["target_section"] != r["section"] for r in with_target)
                    / len(with_target),
                    4,
                )
                if with_target
                else None,
            }
        )
    all_norms = [r["displacement_norm"] for r in statements if r["displacement_norm"] is not None]
    return {
        "generator": "docdistance",
        "version": _VERSION,
        "smd": round(d_smd, 6),
        "order_gap": round(order_gap, 6),
        "structure_closeness": round(_core.closeness(order_gap), 6),
        "structure_closeness_definition": "1 - order_gap/sqrt(2)",
        "thresholds": {
            "changed_cost": DIFF_CHANGED_COST,
            "no_counterpart_cost": NO_COUNTERPART_COST,
            "moved_tolerance": MOVED_TOLERANCE,
            "match_cutoff": MATCH_CUTOFF,
        },
        "definitions": {
            "displacement": "raw rank shift of the crisp alignment; 0 = in place",
            "displacement_norm": "target_index/n_b - index/n_a",
            "matched_displacement": (
                "rank of the statement's paired B statement among the paired B statements minus "
                "the rank of the statement among the kept A statements; null when not kept"
            ),
            "moved": (
                "matched AND matched_displacement != 0 AND "
                "abs(matched_displacement / n_matched) > moved_tolerance AND NOT no_counterpart; "
                "matched = kept, n_matched = number of kept statements"
            ),
            "no_counterpart": (
                "aligned-pair ground cost > no_counterpart_cost; "
                "displacement_norm null, moved false"
            ),
            "share_below_changed": (
                "share of a section's statements with aligned-pair cost <= changed_cost"
            ),
            **_MATCH_DEFINITIONS,
        },
        "summary": {
            "n_moved": sum(r["moved"] for r in statements),
            "n_no_counterpart": sum(r["no_counterpart"] for r in statements),
            "median_displacement_norm": round(float(np.median(all_norms)), 4)
            if all_norms
            else None,
        },
        "anisotropy": anisotropy,
        "n_statements": {"a": len(sa), "b": len(sb)},
        "n_dropped": n_dropped or {"a": 0, "b": 0},
        "drop_rules": drop_rules or {"min_words": 0, "skip_table_fragments": False},
        "sections": section_aggregates,
        "statements": statements,
    }


class DocDistance:
    """Reusable pipeline - construct once, then call :meth:`distance` per pair.

    Models load lazily on first use: the encoder + segmenter for ``wmd``, plus the reranker + NLI for
    ``wmd-wrt-source``. Each scoring method first checks the readiness gate, so an un-init'd mode
    raises :class:`~docdistance.settings.NotInitializedError` before any model load.

    ``memo_size`` above 0 switches on two memos (E12-H67, E12-H68), each bounded at that many
    entries with the least recently used entry evicted beyond it; the ``--pairs`` batches of the
    CLI pass :data:`MEMO_SIZE`. The first keeps the statements, the embeddings and the drop count
    of a document file under its :func:`_memo_key`, so a file that recurs across calls is
    segmented and encoded once. The second keeps the relevance grid and the entailment vector of
    a (document file, source file) pair, so the source-conditioned calls compute them once per
    pair. Both hold what is computed before the anisotropy pre-pass, which pools the documents of
    one call. A document given as raw text enters neither memo. The memos change no returned
    value while every rewrite of a file changes its mtime or its size (the limit is stated at
    :func:`_memo_key`). At the default ``memo_size=0`` both are off and every call reads and
    encodes its documents, as in 1.1.5.
    """

    def __init__(
        self,
        backend: str = "openvino",
        offline: bool = True,
        device: str | None = None,
        memo_size: int = 0,
    ):
        self.backend = backend
        self._offline = offline
        self._device = device
        self._segmenter = None
        self._encoder = None
        self._reranker = None
        self._nli = None
        # memo key -> (statements, embeddings, n_dropped, reader that ran)
        self._embed_memo = _Memo(memo_size)
        # (document memo key, source memo key) -> (relevance grid, entailment vector)
        self._grounding_memo = _Memo(memo_size)
        self._last_memo_key = None  # memo key of the latest embed_statements call; None: no memo
        self._last_n_dropped = 0  # fragment-filter drop count of the latest embed_statements call
        self.last_n_dropped: dict[
            str, int
        ] = {}  # per-document drop counts of the latest scoring call

    def _ensure_base(self) -> None:
        if self._encoder is None:
            self._segmenter = Segmenter(offline=self._offline)
            self._encoder = load_encoder(self.backend, offline=self._offline, device=self._device)

    @property
    def segmenter(self):
        """SAT segmenter, lazily loaded on first access (so direct use works, not only the methods)."""
        self._ensure_base()
        return self._segmenter

    @property
    def encoder(self):
        """Statement encoder, lazily loaded on first access."""
        self._ensure_base()
        return self._encoder

    def _ensure_grounding(self) -> None:
        self._ensure_base()
        if self._reranker is None:
            self._reranker = load_reranker(
                self.backend, offline=self._offline, device=self._device
            )
            self._nli = load_nli(self.backend, offline=self._offline, device=self._device)

    def embed_statements(
        self,
        doc: str | Path,
        *,
        markdown: bool = False,
        input_format: str = "auto",
        min_words: int = 0,
        skip_table_fragments: bool = False,
    ) -> tuple[list[str], np.ndarray]:
        """Segment a document and embed it, returning both the statement texts and their vectors.

        ``markdown=True`` switches to markdown-aware segmentation: each heading, table row, list
        item and fenced-code line is one atomic statement - the markup (hashes, pipes, list
        markers) is stripped from the text that is embedded while the raw line is the text
        reported - and only prose paragraphs go through SaT.

        ``input_format`` names the reader, one of :data:`INPUT_FORMATS`. ``"auto"``, the default,
        chooses it from the file extension (:func:`_reader_by_extension`): ``.md`` and
        ``.markdown`` the markdown reader, ``.html`` and ``.htm`` the HTML reader, every other
        file and raw text the plain reader. ``"plain"`` is the prose segmenter on the whole text,
        the reader of every document in 1.1.5. ``"markdown"`` equals ``markdown=True``. ``"html"``
        converts the text with :func:`~docdistance.encoders.html_to_markdown` and reads the
        result as markdown. One DEBUG line names the reader that ran and, for a file under
        ``"auto"``, that its extension chose it. An unknown value, and ``markdown=True`` beside
        ``"plain"`` or ``"html"``, raise ``ValueError`` before a model loads
        (:func:`_requested_format`).

        ``min_words`` / ``skip_table_fragments`` drop fragment statements after segmentation and
        before encoding (see :func:`_filter_fragments`). Filtering changes the SMD: every statement
        carries 1/n of the transport mass, so the marginals renormalise over the surviving
        statements. The drop count lands in ``self._last_n_dropped`` so callers can report it -
        nothing is silently lost.

        When the reader that ran is the plain one, also under ``"auto"``, statements that hold
        no letter and no digit after the filter (bare markers such as ``-`` or ``###``) are
        returned and embedded like every other statement, and one warning per document names
        their count (see :func:`_warn_marker_statements`).

        A document file met before under the same :func:`_memo_key` is served from the memo of the
        instance: the statements and the vectors are copies of the ones the first call computed,
        ``self._last_n_dropped`` is set to the count of that call and the marker warning is logged
        again. The copies keep the two documents of a pair distinct arrays, as two encodes are:
        the ground cost of one array against itself has an exactly zero diagonal, while two equal
        arrays leave the float32 rounding residue there (SMD 1.87e-4 for the ibm gold summary
        against itself).
        """
        requested = _requested_format(markdown, input_format)
        self._ensure_base()
        # taken before the file is read, so an entry never holds content older than its key
        key = (
            _memo_key(doc, requested, min_words, skip_table_fragments)
            if self._embed_memo.bound
            else None
        )
        self._last_memo_key = key
        entry = self._embed_memo.get(key)
        if entry is not None:
            statements, embeddings, self._last_n_dropped, reader = entry
            if reader == "plain":
                _warn_marker_statements(statements, doc)
            return list(statements), embeddings.copy()
        text, reader = _read_as(doc, requested)
        prefix = _file_prefix(doc)
        logger.debug(
            "{}read by the {} reader{}",
            prefix,
            reader,
            " (chosen by the file extension)" if requested == "auto" and prefix else "",
        )
        if reader == "plain":
            statements = self._segmenter.split(text)
            embed_texts = statements
        else:
            statements, embed_texts = split_markdown(text, self._segmenter.split)
        statements, embed_texts, n_dropped = _filter_fragments(
            statements, embed_texts, min_words=min_words, skip_table_fragments=skip_table_fragments
        )
        self._last_n_dropped = n_dropped
        if not statements:
            if n_dropped:
                raise ValueError(
                    f"document produced no statements ({n_dropped} dropped by the fragment filter)"
                )
            raise ValueError("document produced no statements")
        if reader == "plain":
            _warn_marker_statements(statements, doc)
        embeddings = self._encoder.encode(embed_texts)
        if key is not None:
            # copies: the caller owns what is returned and may change it in place
            self._embed_memo.put(key, (list(statements), embeddings.copy(), n_dropped, reader))
        return statements, embeddings

    def embed(
        self,
        doc: str | Path,
        *,
        markdown: bool = False,
        input_format: str = "auto",
        min_words: int = 0,
        skip_table_fragments: bool = False,
    ) -> np.ndarray:
        """Segment then embed a document into L2-normalized statement vectors ``[n, dim]``."""
        requested = _requested_format(markdown, input_format)
        seg = _seg_kwargs(requested, min_words, skip_table_fragments)
        return self.embed_statements(doc, **seg)[1]

    def _grounding(
        self, doc_sents: list[str], src_sents: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reranker relevance grid + per-statement entailment of the top-k fused source premise."""
        R = self._reranker.score_grid(doc_sents, src_sents)
        k = min(config.RERANK_TOP_K, len(src_sents))
        premises = [
            " ".join(src_sents[j] for j in np.argsort(R[i])[::-1][:k])
            for i in range(len(doc_sents))
        ]
        return R, self._nli.entail(premises, doc_sents)

    def _grounding_memoised(
        self,
        doc_key: tuple | None,
        src_key: tuple | None,
        doc_sents: list[str],
        src_sents: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`_grounding`, computed once per (document file, source file) pair of this instance.

        ``doc_key`` / ``src_key`` are the memo keys :meth:`embed_statements` took for the two
        documents. Both arrays are functions of the two statement lists only, which the keys
        identify. Each call returns copies, so a caller that changes one in place does not change
        what a later call returns. Without a key for either document (raw text, memo off) the
        arrays are computed and nothing is kept.
        """
        if doc_key is None or src_key is None:
            return self._grounding(doc_sents, src_sents)
        entry = self._grounding_memo.get((doc_key, src_key))
        if entry is None:
            entry = self._grounding(doc_sents, src_sents)
            self._grounding_memo.put((doc_key, src_key), entry)
        relevance, entail = entry
        return relevance.copy(), entail.copy()

    def semantic_distance(
        self,
        a: str | Path,
        b: str | Path,
        *,
        anisotropy: bool = False,
        threshold: float = DEFAULT_THRESHOLD,
        markdown: bool = False,
        input_format: str = "auto",
        min_words: int = 0,
        skip_table_fragments: bool = False,
    ) -> DistanceResult:
        settings.require_ready("wmd")
        requested = _requested_format(markdown, input_format)
        seg = _seg_kwargs(requested, min_words, skip_table_fragments)
        ea = self.embed(a, **seg)
        dropped_a = self._last_n_dropped
        eb = self.embed(b, **seg)
        self.last_n_dropped = {"a": dropped_a, "b": self._last_n_dropped}
        if anisotropy:
            _warn_small_pool(ea, eb)
        return _core.compute_distance(ea, eb, anisotropy=anisotropy, threshold=threshold)

    def semantic_distance_with_details(
        self,
        a: str | Path,
        b: str | Path,
        *,
        anisotropy: bool = False,
        threshold: float = DEFAULT_THRESHOLD,
        changed_cost: float = DIFF_CHANGED_COST,
        markdown: bool = False,
        input_format: str = "auto",
        min_words: int = 0,
        skip_table_fragments: bool = False,
    ) -> tuple[DistanceResult, dict]:
        """The symmetric distance result and the optimal-transport statement map, sharing one encode pass.

        ``changed_cost`` sets the per-flow ``changed`` cutoff on the cheapest row cost (default
        ``DIFF_CHANGED_COST``) and is emitted in the detail header's ``thresholds`` block; the
        distance result itself does not depend on it.
        """
        settings.require_ready("wmd")
        requested = _requested_format(markdown, input_format)
        seg = _seg_kwargs(requested, min_words, skip_table_fragments)
        sa, ea = self.embed_statements(a, **seg)
        dropped_a = self._last_n_dropped
        sb, eb = self.embed_statements(b, **seg)
        self.last_n_dropped = {"a": dropped_a, "b": self._last_n_dropped}
        if anisotropy:
            _warn_small_pool(ea, eb)
        result = _core.compute_distance(ea, eb, anisotropy=anisotropy, threshold=threshold)
        details = _build_semantic_details(
            sa,
            ea,
            sb,
            eb,
            anisotropy=anisotropy,
            changed_cost=changed_cost,
            n_dropped=self.last_n_dropped,
            drop_rules={"min_words": min_words, "skip_table_fragments": skip_table_fragments},
            sections=_statement_sections(_read_as(a, requested)[0], sa),
            sections_b=_statement_sections(_read_as(b, requested)[0], sb),
        )
        return result, details

    def structural_distance(
        self,
        a: str | Path,
        b: str | Path,
        *,
        anisotropy: bool = False,
        threshold: float = STRUCTURE_THRESHOLD,
        markdown: bool = False,
        input_format: str = "auto",
        min_words: int = 0,
        skip_table_fragments: bool = False,
    ) -> StructuralResult:
        settings.require_ready("wmd")
        requested = _requested_format(markdown, input_format)
        seg = _seg_kwargs(requested, min_words, skip_table_fragments)
        emb_a = self.embed(a, **seg)
        dropped_a = self._last_n_dropped
        emb_b = self.embed(b, **seg)
        self.last_n_dropped = {"a": dropped_a, "b": self._last_n_dropped}
        if anisotropy:
            _warn_small_pool(emb_a, emb_b)
        return _core.compute_structural(emb_a, emb_b, anisotropy=anisotropy, threshold=threshold)

    def structural_distance_with_details(
        self,
        a: str | Path,
        b: str | Path,
        *,
        anisotropy: bool = False,
        threshold: float = STRUCTURE_THRESHOLD,
        markdown: bool = False,
        input_format: str = "auto",
        min_words: int = 0,
        skip_table_fragments: bool = False,
    ) -> tuple[StructuralResult, dict]:
        """The structural (order) distance result and the per-statement displacement details, sharing one encode pass."""
        settings.require_ready("wmd")
        requested = _requested_format(markdown, input_format)
        seg = _seg_kwargs(requested, min_words, skip_table_fragments)
        sa, ea = self.embed_statements(a, **seg)
        dropped_a = self._last_n_dropped
        sb, eb = self.embed_statements(b, **seg)
        self.last_n_dropped = {"a": dropped_a, "b": self._last_n_dropped}
        if anisotropy:
            _warn_small_pool(ea, eb)
        result = _core.compute_structural(ea, eb, anisotropy=anisotropy, threshold=threshold)
        details = _build_structural_details(
            sa,
            ea,
            sb,
            eb,
            anisotropy=anisotropy,
            n_dropped=self.last_n_dropped,
            drop_rules={"min_words": min_words, "skip_table_fragments": skip_table_fragments},
            sections_a=_statement_sections(_read_as(a, requested)[0], sa),
            sections_b=_statement_sections(_read_as(b, requested)[0], sb),
        )
        return result, details

    def distance_wrt_source(
        self,
        a: str | Path,
        b: str | Path,
        source: str | Path,
        *,
        anisotropy: bool = True,
        markdown: bool = False,
        input_format: str = "auto",
        min_words: int = 0,
        skip_table_fragments: bool = False,
    ) -> SourceConditionedResult:
        settings.require_ready("wmd-wrt-source")
        requested = _requested_format(markdown, input_format)  # raises before a model loads
        self._ensure_grounding()
        seg = _seg_kwargs(requested, min_words, skip_table_fragments)
        sa, ea = self.embed_statements(a, **seg)
        dropped_a, key_a = self._last_n_dropped, self._last_memo_key
        sb, eb = self.embed_statements(b, **seg)
        dropped_b, key_b = self._last_n_dropped, self._last_memo_key
        ss, es = self.embed_statements(source, **seg)
        key_s = self._last_memo_key
        self.last_n_dropped = {"a": dropped_a, "b": dropped_b, "source": self._last_n_dropped}
        ra, enta = self._grounding_memoised(key_a, key_s, sa, ss)
        rb, entb = self._grounding_memoised(key_b, key_s, sb, ss)
        return _core.compute_source_conditioned(
            ea,
            eb,
            es,
            anisotropy=anisotropy,
            reranker_a=ra,
            reranker_b=rb,
            entail_a=enta,
            entail_b=entb,
        )

    def distance_wrt_source_with_map(
        self,
        a: str | Path,
        b: str | Path,
        source: str | Path,
        *,
        anisotropy: bool = True,
        top_k: int = 3,
        markdown: bool = False,
        input_format: str = "auto",
        min_words: int = 0,
        skip_table_fragments: bool = False,
    ) -> tuple[SourceConditionedResult, dict]:
        """The source-conditioned result and the statement-to-source alignment map, sharing one encode pass."""
        settings.require_ready("wmd-wrt-source")
        requested = _requested_format(markdown, input_format)  # raises before a model loads
        self._ensure_grounding()
        seg = _seg_kwargs(requested, min_words, skip_table_fragments)
        sa, ea = self.embed_statements(a, **seg)
        dropped_a, key_a = self._last_n_dropped, self._last_memo_key
        sb, eb = self.embed_statements(b, **seg)
        dropped_b, key_b = self._last_n_dropped, self._last_memo_key
        ss, es = self.embed_statements(source, **seg)
        key_s = self._last_memo_key
        self.last_n_dropped = {"a": dropped_a, "b": dropped_b, "source": self._last_n_dropped}
        ra, enta = self._grounding_memoised(key_a, key_s, sa, ss)
        rb, entb = self._grounding_memoised(key_b, key_s, sb, ss)
        result = _core.compute_source_conditioned(
            ea,
            eb,
            es,
            anisotropy=anisotropy,
            reranker_a=ra,
            reranker_b=rb,
            entail_a=enta,
            entail_b=entb,
        )
        smap = _build_source_map(
            sa,
            ea,
            sb,
            eb,
            ss,
            es,
            anisotropy=anisotropy,
            top_k=top_k,
            n_dropped=self.last_n_dropped,
            drop_rules={"min_words": min_words, "skip_table_fragments": skip_table_fragments},
            sections_a=_statement_sections(_read_as(a, requested)[0], sa),
            sections_b=_statement_sections(_read_as(b, requested)[0], sb),
        )
        return result, smap


def semantic_distance(
    a: str | Path,
    b: str | Path,
    *,
    backend: str = "openvino",
    anisotropy: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    offline: bool = True,
    device: str | None = None,
    markdown: bool = False,
    input_format: str = "auto",
    min_words: int = 0,
    skip_table_fragments: bool = False,
) -> DistanceResult:
    """Symmetric Statement Mover's Distance between documents ``a`` and ``b`` (loads models, then scores).

    ``markdown`` enables markdown-aware segmentation and ``input_format`` names the reader of
    every document (``plain``, ``markdown``, ``html``, or ``auto``, the default, which chooses it
    from the file extension); ``min_words`` /
    ``skip_table_fragments`` drop fragment statements before encoding, which renormalises the
    transport marginals and so changes the SMD (see :meth:`DocDistance.embed_statements`).
    """
    return DocDistance(backend=backend, offline=offline, device=device).semantic_distance(
        a,
        b,
        anisotropy=anisotropy,
        threshold=threshold,
        markdown=markdown,
        input_format=input_format,
        min_words=min_words,
        skip_table_fragments=skip_table_fragments,
    )


def structural_distance(
    a: str | Path,
    b: str | Path,
    *,
    backend: str = "openvino",
    anisotropy: bool = False,
    threshold: float = STRUCTURE_THRESHOLD,
    offline: bool = True,
    device: str | None = None,
    markdown: bool = False,
    input_format: str = "auto",
    min_words: int = 0,
    skip_table_fragments: bool = False,
) -> StructuralResult:
    """Structural (order) distance between documents ``a`` and ``b`` (loads models, then scores).

    ``markdown`` enables markdown-aware segmentation and ``input_format`` names the reader of
    every document (``plain``, ``markdown``, ``html``, or ``auto``, the default, which chooses it
    from the file extension); ``min_words`` /
    ``skip_table_fragments`` drop fragment statements before encoding, which renormalises the
    transport marginals and so changes the SMD (see :meth:`DocDistance.embed_statements`).
    """
    return DocDistance(backend=backend, offline=offline, device=device).structural_distance(
        a,
        b,
        anisotropy=anisotropy,
        threshold=threshold,
        markdown=markdown,
        input_format=input_format,
        min_words=min_words,
        skip_table_fragments=skip_table_fragments,
    )


def source_conditioned_distance(
    a: str | Path,
    b: str | Path,
    source: str | Path,
    *,
    backend: str = "openvino",
    anisotropy: bool = True,
    offline: bool = True,
    device: str | None = None,
    markdown: bool = False,
    input_format: str = "auto",
    min_words: int = 0,
    skip_table_fragments: bool = False,
) -> SourceConditionedResult:
    """Source-conditioned distance d(A, B | S): selection divergence + each document's distance to S.

    ``markdown`` enables markdown-aware segmentation and ``input_format`` names the reader of
    every document (``plain``, ``markdown``, ``html``, or ``auto``, the default, which chooses it
    from the file extension); ``min_words`` /
    ``skip_table_fragments`` drop fragment statements before encoding, which renormalises the
    transport marginals and so changes the SMD (see :meth:`DocDistance.embed_statements`).
    """
    return DocDistance(backend=backend, offline=offline, device=device).distance_wrt_source(
        a,
        b,
        source,
        anisotropy=anisotropy,
        markdown=markdown,
        input_format=input_format,
        min_words=min_words,
        skip_table_fragments=skip_table_fragments,
    )
