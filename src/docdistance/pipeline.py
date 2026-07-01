"""High-level document-distance API.

Two entry styles:

- :class:`DocDistance` - load the models once, score many pairs (the pipeline-integration entry)
- :func:`semantic_distance` / :func:`source_conditioned_distance` - one-shot convenience that loads
  and scores in a single call

Inputs are raw text or a path to a text/markdown file (auto-detected). A leading markdown ``# `` title
line is stripped from files so the document title does not count as a statement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from docdistance import config, settings
from docdistance import distance as _core
from docdistance.distance import (
    DEFAULT_THRESHOLD,
    SMD_MAX,
    DistanceResult,
    SourceConditionedResult,
    StructuralResult,
)
from docdistance.encoders import Segmenter, load_encoder, load_nli, load_reranker

# per-statement "changed" cutoff: a ground cost above the shipped closeness threshold (heuristic)
DIFF_CHANGED_COST = (1.0 - DEFAULT_THRESHOLD) * SMD_MAX


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


def _build_semantic_details(
    sa: list[str],
    ea: np.ndarray,
    sb: list[str],
    eb: np.ndarray,
    *,
    anisotropy: bool = False,
) -> dict:
    """A JSON-serializable transport map: each statement of A → the B statements its OT mass flows to.

    The exact optimal-transport coupling behind the symmetric distance - ``weight`` is the fraction of
    statement A[i]'s mass landing on B[j], ``cost`` the ground distance of that match. Each flow carries
    a ``changed`` flag: whether its best-match (highest-weight) ground cost exceeds ``DIFF_CHANGED_COST``.
    Anisotropy removal is applied the same way as :func:`~docdistance.distance.compute_distance`, so the
    map reflects the same geometry the distance is scored on.
    """
    if anisotropy:
        fixed = _core.all_but_the_top({"a": ea, "b": eb}, k=1)
        ea, eb = fixed["a"], fixed["b"]
    plan = _core.transport_plan(ea, eb)
    cost = _core.cost_matrix(ea, eb)
    flows = _statement_flows(sa, sb, plan, cost)
    for flow in flows:
        matches = flow["matches"]
        # flows are weight-descending, so matches[0] is the best (highest-weight) match
        flow["changed"] = bool(matches and matches[0]["cost"] > DIFF_CHANGED_COST)
    return {
        "smd": round(float((plan * cost).sum()), 6),
        "anisotropy": anisotropy,
        "n_statements": {"a": len(sa), "b": len(sb)},
        "flows": flows,
    }


def _build_structural_details(
    sa: list[str],
    ea: np.ndarray,
    sb: list[str],
    eb: np.ndarray,
    *,
    anisotropy: bool = False,
) -> dict:
    """A JSON-serializable structural (order) projection: per A statement, its aligned B statement + displacement.

    Content (``smd``) and structure (``order_gap``, the E11-H55 OPW order-gap) are reported separately;
    ``structure_closeness`` is the shipped SOTA readout ``closeness(order_gap)`` on the SMD scale
    (1 = same order). Each statement carries its crisp exact-EMD ``target`` and rank ``displacement`` from
    that alignment (the per-statement signal comes from the crisp coupling, not the soft OPW plan).
    Anisotropy removal is applied as in :func:`_build_semantic_details`.
    """
    if anisotropy:
        fixed = _core.all_but_the_top({"a": ea, "b": eb}, k=1)
        ea, eb = fixed["a"], fixed["b"]
    cost = _core.cost_matrix(ea, eb)
    plan = _core.transport_plan(ea, eb)
    align = _core.order_alignment(ea, eb)  # crisp exact-EMD alignment (diagonal tie-break)
    disp = _core.structure_displacement(ea, eb)  # rank shift; 0 = in place
    d_smd = float((plan * cost).sum())
    order_gap = _core.opw_gap(ea, eb)
    statements = []
    for i in range(len(sa)):
        j = int(align[i])
        statements.append(
            {
                "index": i,
                "text": sa[i],
                "target_index": j,
                "target_text": sb[j],
                "displacement": int(disp[i]),  # position shift; 0 = in place
                "moved": bool(disp[i] != 0),
            }
        )
    return {
        "smd": round(d_smd, 6),
        "order_gap": round(order_gap, 6),
        "structure_closeness": round(_core.closeness(order_gap), 6),
        "anisotropy": anisotropy,
        "n_statements": {"a": len(sa), "b": len(sb)},
        "statements": statements,
    }


class DocDistance:
    """Reusable pipeline - construct once, then call :meth:`distance` per pair.

    Models load lazily on first use: the encoder + segmenter for ``wmd``, plus the reranker + NLI for
    ``wmd-wrt-source``. Each scoring method first checks the readiness gate, so an un-init'd mode
    raises :class:`~docdistance.settings.NotInitializedError` before any model load.
    """

    def __init__(self, backend: str = "openvino", offline: bool = True, device: str | None = None):
        self.backend = backend
        self._offline = offline
        self._device = device
        self._segmenter = None
        self._encoder = None
        self._reranker = None
        self._nli = None

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

    def embed_statements(self, doc: str | Path) -> tuple[list[str], np.ndarray]:
        """Segment a document and embed it, returning both the statement texts and their vectors."""
        self._ensure_base()
        statements = self._segmenter.split(_read(doc))
        if not statements:
            raise ValueError("document produced no statements")
        return statements, self._encoder.encode(statements)

    def embed(self, doc: str | Path) -> np.ndarray:
        """Segment then embed a document into L2-normalized statement vectors ``[n, dim]``."""
        return self.embed_statements(doc)[1]

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

    def semantic_distance(
        self,
        a: str | Path,
        b: str | Path,
        *,
        anisotropy: bool = False,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> DistanceResult:
        settings.require_ready("wmd")
        return _core.compute_distance(
            self.embed(a), self.embed(b), anisotropy=anisotropy, threshold=threshold
        )

    def semantic_distance_with_details(
        self,
        a: str | Path,
        b: str | Path,
        *,
        anisotropy: bool = False,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> tuple[DistanceResult, dict]:
        """The symmetric distance result and the optimal-transport statement map, sharing one encode pass."""
        settings.require_ready("wmd")
        sa, ea = self.embed_statements(a)
        sb, eb = self.embed_statements(b)
        result = _core.compute_distance(ea, eb, anisotropy=anisotropy, threshold=threshold)
        details = _build_semantic_details(sa, ea, sb, eb, anisotropy=anisotropy)
        return result, details

    def structural_distance(
        self,
        a: str | Path,
        b: str | Path,
        *,
        anisotropy: bool = False,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> StructuralResult:
        settings.require_ready("wmd")
        emb_a = self.embed(a)
        emb_b = self.embed(b)
        return _core.compute_structural(emb_a, emb_b, anisotropy=anisotropy, threshold=threshold)

    def structural_distance_with_details(
        self,
        a: str | Path,
        b: str | Path,
        *,
        anisotropy: bool = False,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> tuple[StructuralResult, dict]:
        """The structural (order) distance result and the per-statement displacement details, sharing one encode pass."""
        settings.require_ready("wmd")
        sa, ea = self.embed_statements(a)
        sb, eb = self.embed_statements(b)
        result = _core.compute_structural(ea, eb, anisotropy=anisotropy, threshold=threshold)
        details = _build_structural_details(sa, ea, sb, eb, anisotropy=anisotropy)
        return result, details

    def distance_wrt_source(
        self,
        a: str | Path,
        b: str | Path,
        source: str | Path,
        *,
        anisotropy: bool = True,
    ) -> SourceConditionedResult:
        settings.require_ready("wmd-wrt-source")
        self._ensure_grounding()
        sa, ea = self.embed_statements(a)
        sb, eb = self.embed_statements(b)
        ss, es = self.embed_statements(source)
        ra, enta = self._grounding(sa, ss)
        rb, entb = self._grounding(sb, ss)
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
    ) -> tuple[SourceConditionedResult, dict]:
        """The source-conditioned result and the statement-to-source alignment map, sharing one encode pass."""
        settings.require_ready("wmd-wrt-source")
        self._ensure_grounding()
        sa, ea = self.embed_statements(a)
        sb, eb = self.embed_statements(b)
        ss, es = self.embed_statements(source)
        ra, enta = self._grounding(sa, ss)
        rb, entb = self._grounding(sb, ss)
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
        smap = _build_source_map(sa, ea, sb, eb, ss, es, anisotropy=anisotropy, top_k=top_k)
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
) -> DistanceResult:
    """Symmetric Statement Mover's Distance between documents ``a`` and ``b`` (loads models, then scores)."""
    return DocDistance(backend=backend, offline=offline, device=device).semantic_distance(
        a, b, anisotropy=anisotropy, threshold=threshold
    )


def structural_distance(
    a: str | Path,
    b: str | Path,
    *,
    backend: str = "openvino",
    anisotropy: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    offline: bool = True,
    device: str | None = None,
) -> StructuralResult:
    """Structural (order) distance between documents ``a`` and ``b`` (loads models, then scores)."""
    return DocDistance(backend=backend, offline=offline, device=device).structural_distance(
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
