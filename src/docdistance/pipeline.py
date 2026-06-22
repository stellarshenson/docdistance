"""High-level document-distance API.

Two entry styles:

- :class:`DocDistance` - load the models once, score many pairs (the pipeline-integration entry)
- :func:`document_distance` / :func:`source_conditioned_distance` - one-shot convenience that loads
  and scores in a single call

Inputs are raw text or a path to a text/markdown file (auto-detected). A leading markdown ``# `` title
line is stripped from files so the document title does not count as a statement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from docdistance import distance as _core
from docdistance.distance import (
    DEFAULT_THRESHOLD,
    DistanceResult,
    SourceConditionedResult,
)
from docdistance.encoders import Segmenter, load_encoder


def _load_body(path: Path) -> str:
    lines = path.read_text().splitlines()
    return "\n".join(ln for ln in lines if not ln.startswith("# ")).strip()


def _read(doc: str | Path) -> str:
    """Resolve a document argument: an existing file path is read, anything else is treated as text."""
    if isinstance(doc, Path):
        return _load_body(doc)
    if isinstance(doc, str):
        try:
            p = Path(doc)
            if p.exists() and p.is_file():
                return _load_body(p)
        except OSError:
            pass
        return doc.strip()
    raise TypeError(f"document must be str or Path, got {type(doc).__name__}")


def _align_statements(
    sents: list[str], emb: np.ndarray, src_sents: list[str], src_emb: np.ndarray, top_k: int
) -> list[dict]:
    """Per statement, its top-``k`` source statements by soft coverage weight."""
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
        out.append({"index": i, "text": sents[i], "matches": matches})
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
) -> dict:
    """A JSON-serializable map: each statement of A and B → its top-``k`` source statements.

    Anisotropy removal is applied the same way as the source-conditioned distance (over the pooled
    A/B/S statements), so the map reflects the same geometry the selection axis is scored on.
    """
    if anisotropy:
        fixed = _core.all_but_the_top({"a": ea, "b": eb, "s": es}, k=1)
        ea, eb, es = fixed["a"], fixed["b"], fixed["s"]
    return {
        "top_k": top_k,
        "anisotropy": anisotropy,
        "n_statements": {"a": len(sa), "b": len(sb), "source": len(ss)},
        "a": _align_statements(sa, ea, ss, es, top_k),
        "b": _align_statements(sb, eb, ss, es, top_k),
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


def _build_transport_map(
    sa: list[str],
    ea: np.ndarray,
    sb: list[str],
    eb: np.ndarray,
    *,
    anisotropy: bool = False,
) -> dict:
    """A JSON-serializable transport map: each statement of A → the B statements its OT mass flows to.

    The exact optimal-transport coupling behind the symmetric distance - ``weight`` is the fraction of
    statement A[i]'s mass landing on B[j], ``cost`` the ground distance of that match. Anisotropy removal
    is applied the same way as :func:`~docdistance.distance.compute_distance`, so the map reflects the
    same geometry the distance is scored on.
    """
    if anisotropy:
        fixed = _core.all_but_the_top({"a": ea, "b": eb}, k=1)
        ea, eb = fixed["a"], fixed["b"]
    plan = _core.transport_plan(ea, eb)
    cost = _core.cost_matrix(ea, eb)
    return {
        "smd": round(float((plan * cost).sum()), 6),
        "anisotropy": anisotropy,
        "n_statements": {"a": len(sa), "b": len(sb)},
        "flows": _statement_flows(sa, sb, plan, cost),
    }


class DocDistance:
    """Reusable pipeline - construct once (models load here), then call :meth:`distance` per pair."""

    def __init__(self, backend: str = "openvino", offline: bool = True, device: str | None = None):
        self.backend = backend
        self.segmenter = Segmenter(offline=offline)
        self.encoder = load_encoder(backend, offline=offline, device=device)

    def embed_statements(self, doc: str | Path) -> tuple[list[str], np.ndarray]:
        """Segment a document and embed it, returning both the statement texts and their vectors."""
        statements = self.segmenter.split(_read(doc))
        if not statements:
            raise ValueError("document produced no statements")
        return statements, self.encoder.encode(statements)

    def embed(self, doc: str | Path) -> np.ndarray:
        """Segment then embed a document into L2-normalized statement vectors ``[n, dim]``."""
        return self.embed_statements(doc)[1]

    def distance(
        self,
        a: str | Path,
        b: str | Path,
        *,
        anisotropy: bool = False,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> DistanceResult:
        return _core.compute_distance(
            self.embed(a), self.embed(b), anisotropy=anisotropy, threshold=threshold
        )

    def distance_with_map(
        self,
        a: str | Path,
        b: str | Path,
        *,
        anisotropy: bool = False,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> tuple[DistanceResult, dict]:
        """The symmetric distance result and the optimal-transport statement map, sharing one encode pass."""
        sa, ea = self.embed_statements(a)
        sb, eb = self.embed_statements(b)
        result = _core.compute_distance(ea, eb, anisotropy=anisotropy, threshold=threshold)
        tmap = _build_transport_map(sa, ea, sb, eb, anisotropy=anisotropy)
        return result, tmap

    def distance_wrt_source(
        self,
        a: str | Path,
        b: str | Path,
        source: str | Path,
        *,
        anisotropy: bool = True,
    ) -> SourceConditionedResult:
        return _core.compute_source_conditioned(
            self.embed(a), self.embed(b), self.embed(source), anisotropy=anisotropy
        )

    def distance_wrt_source_with_map(
        self,
        a: str | Path,
        b: str | Path,
        source: str | Path,
        *,
        anisotropy: bool = True,
        top_k: int = 3,
    ) -> tuple[SourceConditionedResult, dict]:
        """The source-conditioned result and the statement-to-source alignment map, sharing one encode pass."""
        sa, ea = self.embed_statements(a)
        sb, eb = self.embed_statements(b)
        ss, es = self.embed_statements(source)
        result = _core.compute_source_conditioned(ea, eb, es, anisotropy=anisotropy)
        smap = _build_source_map(sa, ea, sb, eb, ss, es, anisotropy=anisotropy, top_k=top_k)
        return result, smap


def document_distance(
    a: str | Path,
    b: str | Path,
    *,
    backend: str = "openvino",
    anisotropy: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    offline: bool = True,
    device: str | None = None,
) -> DistanceResult:
    """Symmetric Statement Mover's Distance between documents ``a`` and ``b`` (loads models, then scores)."""
    return DocDistance(backend=backend, offline=offline, device=device).distance(
        a, b, anisotropy=anisotropy, threshold=threshold
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
) -> SourceConditionedResult:
    """Source-conditioned distance d(A, B | S): selection divergence + each document's distance to S."""
    return DocDistance(backend=backend, offline=offline, device=device).distance_wrt_source(
        a, b, source, anisotropy=anisotropy
    )
