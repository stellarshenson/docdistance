"""Pure-numpy optimal-transport core for document distance.

No heavy ML dependencies - only numpy and POT (``ot``). Segmentation and embedding live in
``encoders.py``; this module operates on statement-embedding arrays: L2-normalized float32 of
shape ``[n_statements, dim]``. Every function here is deterministic and CPU-only, which is why
the unit tests can exercise it without loading a single model.

The distance is the exact Statement Mover's Distance (SMD) - optimal transport between two
statement clouds with the metric ground cost ``sqrt(2 - 2cos)`` (Euclidean on L2-normalized
vectors). ``wcd`` and ``rwmd`` are the cheap lower bounds (``WCD <= RWMD <= SMD``). The
source-conditioned helpers re-base the transport onto a common source ``S``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import ot

# orthogonal statement clouds -> closeness 0; cos >= 0 for these embeddings so distance lands in [0, sqrt(2)]
SMD_MAX = float(np.sqrt(2.0))

# closeness cutoff for the similar / not-similar verdict; heuristic, calibrate per corpus
# (measured boundary on the ibm-ai-adoption fixtures: min gold 72.7% vs max adversarial 72.2%)
DEFAULT_THRESHOLD = 0.725


def cost_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Ground cost ``sqrt(2 - 2cos)`` = Euclidean distance on L2-normalized rows (a metric)."""
    return ot.dist(X, Y, metric="euclidean")


def _uniform(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def _ab(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _uniform(len(X)), _uniform(len(Y))


def smd(X: np.ndarray, Y: np.ndarray) -> float:
    """The distance: exact Statement Mover's Distance via the network-simplex LP."""
    return float(ot.emd2(*_ab(X, Y), cost_matrix(X, Y)))


def wcd(X: np.ndarray, Y: np.ndarray) -> float:
    """Lower bound: distance between the mean-pooled statement clouds (whole-doc cosine)."""
    return float(np.linalg.norm(X.mean(0) - Y.mean(0)))


def rwmd(X: np.ndarray, Y: np.ndarray) -> float:
    """Lower bound: one-sided relaxation (greedy nearest-statement alignment)."""
    a, b = _ab(X, Y)
    C = cost_matrix(X, Y)
    return float(max((a * C.min(1)).sum(), (b * C.min(0)).sum()))


def closeness(d: float) -> float:
    """Map a distance to a 0-1 similarity: 1 = identical clouds, 0 = orthogonal."""
    return max(0.0, 1.0 - d / SMD_MAX)


def verdict(close: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    return "similar" if close >= threshold else "not similar"


def all_but_the_top(emb: dict[str, np.ndarray], k: int = 1) -> dict[str, np.ndarray]:
    """All-but-the-top anisotropy removal (Mu & Viswanath, ICLR 2018).

    Subtract the pooled mean and project out the top-``k`` principal components (via SVD of the
    mean-centered matrix), then re-L2-normalize. De-bunches the anisotropic mmBERT cosines and
    widens the distance dynamic range while preserving statement ordering. Operates over the
    pooled statements of all documents in ``emb`` so the common direction is shared.
    """
    keys = list(emb)
    pool = np.concatenate([emb[key] for key in keys], 0)
    centered = pool - pool.mean(0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    comps = Vt[:k]
    fixed = centered - centered @ comps.T @ comps
    fixed = fixed / (np.linalg.norm(fixed, axis=1, keepdims=True) + 1e-9)
    out: dict[str, np.ndarray] = {}
    off = 0
    for key in keys:
        m = len(emb[key])
        out[key] = fixed[off : off + m].astype(np.float32)
        off += m
    return out


# --- source-conditioned core (metric parts only; reranker/NLI grounding deferred to E02) ---


COVERAGE_TEMPERATURE = 0.1


def coverage_profile(
    X: np.ndarray, S: np.ndarray, temperature: float = COVERAGE_TEMPERATURE
) -> np.ndarray:
    """How document ``X``'s statements distribute over the source statements ``S``; a distribution, sums to 1.

    Each statement softly assigns to source statements by ``softmax(-cost / temperature)`` and the
    profile is the mean assignment over ``X``. A balanced-OT column marginal is forced uniform (the
    transport constraint), so it carries no per-document signal; this soft nearest-source histogram
    varies by document and captures which source content each one covers.
    """
    C = cost_matrix(X, S)
    A = np.exp(
        -(C - C.min(1, keepdims=True)) / temperature
    )  # subtract row-min for numerical stability
    A = A / A.sum(1, keepdims=True)
    return A.mean(0)


def selection_divergence(cov_a: np.ndarray, cov_b: np.ndarray, S: np.ndarray) -> float:
    """D_sel: metric OT between two coverage profiles over the shared source statements.

    Ground cost is ``sqrt(2 - 2cos)`` on the source-statement embeddings, so D_sel stays a metric -
    the guarantee conditioning on a fixed ``S`` buys back. Captures same source, different picks.
    """
    return float(ot.emd2(np.asarray(cov_a), np.asarray(cov_b), cost_matrix(S, S)))


def ungrounded_residual(X: np.ndarray, S: np.ndarray) -> float:
    """Per-document grounding proxy: the transport cost SMD(X, S) (distance of X to the source).

    A coarse, metric stand-in for the grounding residual - higher means the document drifts
    further from any source statement. The reranker + NLI grade that separates contradiction from
    omission is deferred to E02; this is the geometric distance-to-source only.
    """
    return smd(X, S)


@dataclass
class DistanceResult:
    """Symmetric distance result. ``smd`` is the distance; ``wcd``/``rwmd`` bound it below."""

    smd: float
    wcd: float
    rwmd: float
    closeness: float
    threshold: float
    verdict: str
    anisotropy: bool
    n_statements_a: int
    n_statements_b: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SourceConditionedResult:
    """Source-conditioned result: selection divergence plus each document's distance to the source."""

    d_sel: float
    residual_a: float
    residual_b: float
    closeness_a: float
    closeness_b: float
    n_statements_a: int
    n_statements_b: int
    n_statements_source: int
    coverage_a: list[float]
    coverage_b: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


def compute_distance(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    *,
    anisotropy: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
) -> DistanceResult:
    """Assemble a :class:`DistanceResult` from two statement-embedding arrays.

    ``anisotropy`` is off by default: all-but-the-top estimates the shared direction from a corpus,
    so over a single document pair it strips genuine shared meaning and distorts the scale. The
    validated nb04 verdict uses raw embeddings; enable anisotropy only over a pooled corpus.
    """
    n_a, n_b = len(emb_a), len(emb_b)
    if anisotropy:
        fixed = all_but_the_top({"a": emb_a, "b": emb_b}, k=1)
        emb_a, emb_b = fixed["a"], fixed["b"]
    d = smd(emb_a, emb_b)
    close = closeness(d)
    return DistanceResult(
        smd=d,
        wcd=wcd(emb_a, emb_b),
        rwmd=rwmd(emb_a, emb_b),
        closeness=close,
        threshold=threshold,
        verdict=verdict(close, threshold),
        anisotropy=anisotropy,
        n_statements_a=n_a,
        n_statements_b=n_b,
    )


def compute_source_conditioned(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    emb_source: np.ndarray,
    *,
    anisotropy: bool = True,
) -> SourceConditionedResult:
    """Assemble a :class:`SourceConditionedResult` from A, B and source statement embeddings.

    Anisotropy removal (all-but-the-top, k=1) is on by default: the source supplies the corpus the
    shared direction is estimated from, and removing it widens the ``D_sel`` dynamic range ~7.4x at
    0 ordinality violations (E04-H15). Pass ``anisotropy=False`` to opt out.
    """
    n_a, n_b, n_s = len(emb_a), len(emb_b), len(emb_source)
    if anisotropy:
        fixed = all_but_the_top({"a": emb_a, "b": emb_b, "s": emb_source}, k=1)
        emb_a, emb_b, emb_source = fixed["a"], fixed["b"], fixed["s"]
    cov_a = coverage_profile(emb_a, emb_source)
    cov_b = coverage_profile(emb_b, emb_source)
    res_a = ungrounded_residual(emb_a, emb_source)
    res_b = ungrounded_residual(emb_b, emb_source)
    return SourceConditionedResult(
        d_sel=selection_divergence(cov_a, cov_b, emb_source),
        residual_a=res_a,
        residual_b=res_b,
        closeness_a=closeness(res_a),
        closeness_b=closeness(res_b),
        n_statements_a=n_a,
        n_statements_b=n_b,
        n_statements_source=n_s,
        coverage_a=cov_a.tolist(),
        coverage_b=cov_b.tolist(),
    )
