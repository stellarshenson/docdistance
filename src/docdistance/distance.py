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

from docdistance import config

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


def transport_plan(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """The exact optimal-transport coupling behind :func:`smd`; shape ``[n_X, n_Y]`` - the transport map.

    ``T[i, j]`` is the probability mass moved from statement ``X[i]`` to statement ``Y[j]``. Row ``i``
    sums to the source marginal ``1/n_X`` and column ``j`` to ``1/n_Y``, and ``(T * cost_matrix(X, Y)).sum()``
    equals ``smd(X, Y)``. The network-simplex solution is sparse (at most ``n_X + n_Y − 1`` nonzeros), so
    most statements map to one or a few others - the interpretable statement-to-statement alignment the
    distance is built from.
    """
    return ot.emd(*_ab(X, Y), cost_matrix(X, Y))


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


# --- E11-H55 OPW order-gap: structural (order) distance + crisp per-statement alignment ---


# Su & Hua order-preserving Wasserstein defaults (E10/E11-H55)
OPW_LAMBDA1 = 50.0  # inverse-difference-moment weight
OPW_LAMBDA2 = 0.1  # entropic regularization
OPW_SIGMA = 1.0  # Gaussian temporal-prior bandwidth


def _round_to_polytope(P: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Project a nonnegative matrix onto the transport polytope U(a, b) (Altschuler et al., 2017).

    Scales rows then columns down to their marginal caps, then adds the rank-one deficit so both
    marginals hold exactly. This turns the plan into a valid coupling, which guarantees
    ``(P * D).sum() >= emd2(a, b, D)`` - without it the fixed-iteration Sinkhorn plan meets only the
    row marginal and its cost can dip below the SMD.
    """
    P = P * np.minimum(a / P.sum(1), 1.0)[:, None]
    P = P * np.minimum(b / P.sum(0), 1.0)[None, :]
    er = np.maximum(a - P.sum(1), 0.0)  # deficits are >= 0 by construction; clip float noise
    ec = np.maximum(b - P.sum(0), 0.0)
    total = er.sum()
    if total > 1e-300:
        P = P + np.outer(er, ec) / total
    return P


def _opw_plan(
    D: np.ndarray,
    lambda1: float = OPW_LAMBDA1,
    lambda2: float = OPW_LAMBDA2,
    sigma: float = OPW_SIGMA,
    iters: int = 100,
) -> np.ndarray:
    """Log-stabilized Sinkhorn OPW coupling for a precomputed ground cost ``D`` (private, folds D).

    The fixed-iteration loop ends on the ``u`` (row) update, so only the row marginal is met; for
    asymmetric shapes the column marginal is left badly unconverged at the shipped iteration count.
    The plan is therefore rounded onto U(a, b) so both marginals hold exactly and the plan is a valid
    coupling.
    """
    N, M = D.shape
    i = (np.arange(1, N + 1) / N)[:, None]
    j = (np.arange(1, M + 1) / M)[None, :]
    mid = np.abs(i - j) / np.sqrt(1 / N**2 + 1 / M**2)
    logP = -(mid**2) / (2 * sigma**2) - np.log(sigma * np.sqrt(2 * np.pi))  # KL temporal band
    S = lambda1 / ((i - j) ** 2 + 1)  # inverse difference moment
    logK = logP + (S - D) / lambda2
    logK = logK - logK.max()  # global shift: transport-invariant, prevents overflow
    K = np.exp(logK)
    a, b = np.full(N, 1.0 / N), np.full(M, 1.0 / M)
    u = np.ones(N) / N
    for _ in range(iters):
        v = b / (K.T @ u + 1e-300)
        u = a / (K @ v + 1e-300)
    return _round_to_polytope(u[:, None] * K * v[None, :], a, b)


def opw_plan(
    X: np.ndarray,
    Y: np.ndarray,
    lambda1: float = OPW_LAMBDA1,
    lambda2: float = OPW_LAMBDA2,
    sigma: float = OPW_SIGMA,
    iters: int = 100,
) -> np.ndarray:
    """Order-preserving Sinkhorn coupling ``T`` (shape ``[n_X, n_Y]``) - the plan E11-H55 computes then discards.

    A SOFT/entropic dense coupling (every cell carries mass), unlike the crisp network-simplex
    :func:`transport_plan`. Good for the aggregate order-gap magnitude, not for crisp per-statement
    pin-pointing - use :func:`order_alignment` for that.
    """
    return _opw_plan(cost_matrix(X, Y), lambda1, lambda2, sigma, iters)


def opw_cost(
    X: np.ndarray,
    Y: np.ndarray,
    lambda1: float = OPW_LAMBDA1,
    lambda2: float = OPW_LAMBDA2,
    sigma: float = OPW_SIGMA,
    iters: int = 100,
) -> float:
    """Order-preserving transport cost ``(opw_plan * cost_matrix).sum()``."""
    D = cost_matrix(X, Y)
    return float((_opw_plan(D, lambda1, lambda2, sigma, iters) * D).sum())


def opw_gap(
    X: np.ndarray,
    Y: np.ndarray,
    lambda1: float = OPW_LAMBDA1,
    lambda2: float = OPW_LAMBDA2,
    sigma: float = OPW_SIGMA,
    iters: int = 100,
) -> float:
    """Structural (order) distance ``max(0, opw_cost - smd)`` - the E11-H55 order-gap.

    Subtracting SMD cancels the content component, leaving the extra cost the order constraint forces.
    A translation-invariant SCORE (order-gap >= 0), NOT a metric. The OPW plan is rounded onto the
    transport polytope U(a, b) (see :func:`_round_to_polytope`), so it is a valid coupling and its cost
    is bounded below by the SMD; the ``max(0, ...)`` clamp only absorbs LP rounding noise. ``D`` is built
    once and reused across the OPW cost and the SMD.
    """
    D = cost_matrix(X, Y)
    cost = float((_opw_plan(D, lambda1, lambda2, sigma, iters) * D).sum())
    content = float(ot.emd2(*_ab(X, Y), D))
    return max(0.0, cost - content)


def _aligned_plan(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Exact-EMD coupling with an infinitesimal diagonal tie-break so ties resolve to the in-place map.

    Duplicate or near-duplicate statements make the ground cost degenerate, and the network simplex is
    then free to pick a swap among equal-cost optimal couplings, inventing displacement on statements
    that never moved. Adding an ``eps``-scaled positional cost ``|i/n - j/m|`` breaks those ties toward
    the minimal-shift coupling; ``eps`` is far below any genuine cost gap, so non-tied alignments are
    unchanged.
    """
    D = cost_matrix(X, Y)
    n, m = D.shape
    pos = np.abs((np.arange(1, n + 1) / n)[:, None] - (np.arange(1, m + 1) / m)[None, :])
    eps = 1e-6 * (float(D.max()) + 1.0)
    return ot.emd(*_ab(X, Y), D + eps * pos)


def order_alignment(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Per ``X`` statement, its aligned ``Y`` index - the crisp exact-EMD argmax (diagonal tie-break)."""
    return _aligned_plan(X, Y).argmax(1)


def structure_displacement(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Rank-based position shift from the crisp alignment; ``0`` = in place, nonzero = moved."""
    align = order_alignment(X, Y)
    order = np.argsort(np.argsort(align))
    return order - np.arange(len(order))


# --- source-conditioned core: metric selection axis + reranker/NLI grounding residual (D_grd) ---


COVERAGE_TEMPERATURE = 0.1


def coverage_alignment(
    X: np.ndarray, S: np.ndarray, temperature: float = COVERAGE_TEMPERATURE
) -> np.ndarray:
    """Per-statement soft assignment of ``X`` over the source ``S``; shape ``[n_X, n_S]``, each row sums to 1.

    Row ``i`` is statement ``i``'s ``softmax(-cost / temperature)`` distribution over the source
    statements - the alignment map of which source content each statement covers. The row-max is the
    soft nearest source. :func:`coverage_profile` is the mean of this matrix over ``X``.
    """
    C = cost_matrix(X, S)
    A = np.exp(
        -(C - C.min(1, keepdims=True)) / temperature
    )  # subtract row-min for numerical stability
    return A / A.sum(1, keepdims=True)


def coverage_profile(
    X: np.ndarray, S: np.ndarray, temperature: float = COVERAGE_TEMPERATURE
) -> np.ndarray:
    """How document ``X``'s statements distribute over the source statements ``S``; a distribution, sums to 1.

    Each statement softly assigns to source statements by ``softmax(-cost / temperature)`` and the
    profile is the mean assignment over ``X``. A balanced-OT column marginal is forced uniform (the
    transport constraint), so it carries no per-document signal; this soft nearest-source histogram
    varies by document and captures which source content each one covers.
    """
    return coverage_alignment(X, S, temperature).mean(0)


def selection_divergence(cov_a: np.ndarray, cov_b: np.ndarray, S: np.ndarray) -> float:
    """D_sel: metric OT between two coverage profiles over the shared source statements.

    Ground cost is ``sqrt(2 - 2cos)`` on the source-statement embeddings, so D_sel stays a metric -
    the guarantee conditioning on a fixed ``S`` buys back. Captures same source, different picks.
    """
    return float(ot.emd2(np.asarray(cov_a), np.asarray(cov_b), cost_matrix(S, S)))


def ungrounded_residual(X: np.ndarray, S: np.ndarray) -> float:
    """Per-document geometric grounding proxy: the transport cost SMD(X, S) (distance of X to S).

    A coarse, metric stand-in for the grounding residual - higher means the document drifts
    further from any source statement. The model-graded :func:`grounding_residual` (reranker + NLI)
    is the sharper signal; this is the geometric distance-to-source fallback when no models scored.
    """
    return smd(X, S)


def grounding_residual(reranker: np.ndarray, entail: np.ndarray) -> float:
    """D_grd residual (E03-H11 relevance-gated ungrounded mass): how much of a document is unsupported.

    ``mean_i (1 - entail_i) * (1 - max_j reranker[i, j])`` - each statement's ungrounded mass
    ``1 - P(entail)`` gated by ``1 - max relevance``, so a statement that strongly matches some source
    (likely grounded but not entailed by the fused premise) is down-weighted. Lower = better grounded.
    ``reranker`` is the ``[n_statements, n_source]`` relevance grid (or its per-statement max);
    ``entail`` the per-statement entailment probability.
    """
    reranker = np.asarray(reranker, dtype=np.float64)
    entail = np.asarray(entail, dtype=np.float64)
    max_rel = reranker.max(axis=1) if reranker.ndim == 2 else reranker
    return float(np.mean((1.0 - entail) * (1.0 - max_rel)))


def grounding_blend(
    d_sel: float, d_grd: float, *, alpha: float = config.GROUNDING_BLEND_ALPHA
) -> float:
    """E03-H14 two-axis blend ``alpha * d_sel + (1 - alpha) * d_grd`` (default alpha 0.75).

    The notebook min-maxes each axis over a candidate set before blending, so pass comparably-scaled
    inputs; for a single A-vs-B pair the raw ``d_sel`` / ``d_grd`` axes are the primary signal and
    this blend is the documented set-level aggregation.
    """
    return float(alpha * d_sel + (1.0 - alpha) * d_grd)


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
class StructuralResult:
    """Structural (order) distance result. ``order_gap`` is the E11-H55 OPW order-gap; ``structure_closeness`` its readout."""

    smd: float
    order_gap: float
    structure_closeness: float
    threshold: float
    verdict: str
    anisotropy: bool
    n_statements_a: int
    n_statements_b: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SourceConditionedResult:
    """Source-conditioned result: the selection axis, each document's grounding residual, distances to S.

    ``grd_a`` / ``grd_b`` are the reranker x NLI grounding residuals (E03-H11), present only when the
    grounding models scored; ``d_grd`` is the grounding-axis separation ``|grd_a - grd_b|``. They stay
    ``None`` on the metric-only path (no models), where ``residual_a`` / ``residual_b`` (geometric
    distance-to-source) carry the grounding proxy.
    """

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
    grd_a: float | None = None
    grd_b: float | None = None
    d_grd: float | None = None

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


def compute_structural(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    *,
    anisotropy: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
) -> StructuralResult:
    """Assemble a :class:`StructuralResult` from two statement-embedding arrays.

    The structural (order) axis: content is reported via ``smd``, structure via the E11-H55 OPW
    ``order_gap``, and ``structure_closeness = closeness(order_gap)`` is the SMD-scale readout
    (1 = same order). ``anisotropy`` is off by default for the same reason as
    :func:`compute_distance`.
    """
    n_a, n_b = len(emb_a), len(emb_b)
    if anisotropy:
        fixed = all_but_the_top({"a": emb_a, "b": emb_b}, k=1)
        emb_a, emb_b = fixed["a"], fixed["b"]
    d = smd(emb_a, emb_b)
    gap = opw_gap(emb_a, emb_b)
    close = closeness(gap)
    return StructuralResult(
        smd=d,
        order_gap=gap,
        structure_closeness=close,
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
    reranker_a: np.ndarray | None = None,
    reranker_b: np.ndarray | None = None,
    entail_a: np.ndarray | None = None,
    entail_b: np.ndarray | None = None,
) -> SourceConditionedResult:
    """Assemble a :class:`SourceConditionedResult` from A, B and source statement embeddings.

    Anisotropy removal (all-but-the-top, k=1) is on by default: the source supplies the corpus the
    shared direction is estimated from, and removing it widens the ``D_sel`` dynamic range ~7.4x at
    0 ordinality violations (E04-H15). Pass ``anisotropy=False`` to opt out.

    When the grounding arrays are supplied (``reranker_*`` the doc x source relevance grid,
    ``entail_*`` the per-statement entailment probability), the E03-H11 grounding residual is computed
    per document into ``grd_a`` / ``grd_b`` and ``d_grd = |grd_a - grd_b|``; without them those stay
    ``None`` and only the metric selection axis + geometric residuals are returned.
    """
    n_a, n_b, n_s = len(emb_a), len(emb_b), len(emb_source)
    if anisotropy:
        fixed = all_but_the_top({"a": emb_a, "b": emb_b, "s": emb_source}, k=1)
        emb_a, emb_b, emb_source = fixed["a"], fixed["b"], fixed["s"]
    cov_a = coverage_profile(emb_a, emb_source)
    cov_b = coverage_profile(emb_b, emb_source)
    res_a = ungrounded_residual(emb_a, emb_source)
    res_b = ungrounded_residual(emb_b, emb_source)
    grd_a = (
        grounding_residual(reranker_a, entail_a)
        if reranker_a is not None and entail_a is not None
        else None
    )
    grd_b = (
        grounding_residual(reranker_b, entail_b)
        if reranker_b is not None and entail_b is not None
        else None
    )
    d_grd = abs(grd_a - grd_b) if grd_a is not None and grd_b is not None else None
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
        grd_a=grd_a,
        grd_b=grd_b,
        d_grd=d_grd,
    )
