"""Pure-numpy optimal-transport core for document distance.

No heavy ML dependencies - only numpy, scipy and POT (``ot``). Segmentation and embedding live in
``encoders.py``; this module operates on statement-embedding arrays: L2-normalized float32 of
shape ``[n_statements, dim]``. Every function here is deterministic and CPU-only, which is why
the unit tests can exercise it without loading a single model.

The distance is the exact Statement Mover's Distance (SMD) - optimal transport between two
statement clouds with the metric ground cost ``sqrt(2 - 2cos)`` (Euclidean on L2-normalized
vectors). ``wcd`` and ``rwmd`` are the cheap lower bounds (``WCD <= RWMD <= SMD``). The
source-conditioned helpers re-base the transport onto a common source ``S``.
``partial_matching`` pairs statements one to one under a cost cutoff; ``matched_order_gap`` and
``unsupported_count`` are scores that depend on a calibrated cutoff, not metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import ot
from scipy.optimize import linear_sum_assignment

from docdistance import config

# orthogonal statement clouds -> closeness 0; cos >= 0 for these embeddings so distance lands in [0, sqrt(2)]
SMD_MAX = float(np.sqrt(2.0))

# closeness cutoff for the similar / not-similar verdict; heuristic, calibrate per corpus.
# Band on the shipped semantic_distance path (E12-H63: 11 ibm-ai-adoption fixtures, OpenVINO
# INT8): min gold 79.44% (haiku) vs max adversarial 74.74% (adv2-b), so any cutoff in
# [0.7474, 0.7944] separates the tiers. On the fp backend the band is [0.7566, 0.8027]; 0.760 to
# 0.790 separates the tiers on both. In-sample: the same 11 documents set it and are scored by
# it, and each edge is one document. The 79.7% / 75.0% this comment quoted before came from the
# dd.embed path of notebook 09 cell 10, not from semantic_distance. The leave-one-out and backend
# readings: docs/solution/wmd-docdistance-solution-sota.md, Calibration of the semantic verdict.
DEFAULT_THRESHOLD = 0.77

# closeness cutoff for the structural (order) verdict, placed on the ibm ladder of the E11
# scramble fixture (reports/E11-structure-sota-decision.json .scramble.ibm.H55), whose closeness
# rungs run 0.970 / 0.893 / 0.820 / 0.761 / 0.727 / 0.706 from lightest reorder to full scramble
# (wergeland bottoms at 0.720). The order-gap scale is compressed relative to the semantic
# one, so a semantic cutoff fires on almost nothing here: 0.725 catches only the deepest
# rung, 0.77 catches three of six. 0.90 puts the boundary between rung 1 (a light reorder,
# still "similar") and rung 2 (a genuine rearrangement). It is not a max-margin point: the
# thresholds that keep every ibm rung and every un-reordered ibm pair on its side are the band
# (0.892817, 0.906144], so 0.90 has 0.006144 of headroom below the nearest un-reordered ibm pair.
# The wergeland band is (0.908048, 0.973676], above the ibm band by 0.001904, so no cross-article
# band exists and 0.90 reads wergeland rung 2 as "similar" (E12-H59). Calibrate per corpus; the
# procedure: docs/solution/wmd-structure-distance-sota.md, Calibration of the structural verdict.
STRUCTURE_THRESHOLD = 0.90

# ground cost below which partial_matching pairs two statements, calibrated in E13-H71
# (reports/E13-H71-partial-matching-map-metrics.json) on 900 edited copies of the 10
# structure-fixture documents, labels kept / removed / added: chosen on half A of the documents,
# held-out macro-F1 0.9772 on half B, and the in-sample optimum over all 10. Any cutoff from 0.28
# to 0.44 keeps the macro-F1 of both halves at or above 0.90. Calibrated on raw embeddings, not on
# the output of all_but_the_top. A measured value: it is not the derived DIFF_CHANGED_COST of
# pipeline.py, (1 - DEFAULT_THRESHOLD) * sqrt(2) = 0.3253.
MATCH_CUTOFF = 0.325

# cost to the nearest source statement above which unsupported_count counts a statement,
# calibrated in E15-H78 (reports/E15-H78-raw-nearest-source-count-metrics.json): the mean of the
# cutoffs 0.3375 and 0.3550 chosen on the two calibration halves of the 7 gold-tier ibm summaries,
# statements of those summaries against statements of a second article. Held-out balanced accuracy
# 0.9412 / 0.9272; 0.9868 with the two articles exchanged and no refit. Valid on raw embeddings
# only: after all_but_the_top the held-out readings are 0.8168 / 0.7277.
SOURCE_SUPPORT_CUTOFF = 0.34625

# minimum number of pooled statements for the all_but_the_top estimate, measured in E12-H65
# (reports/E12-H65-cmr-pool-crossover-metrics.json) with the torch fp32 encoder, not the default
# OpenVINO INT8 one, on pools of 2, 3, 4, 6, 8 and 11 of the 11 ibm summaries. 71 is the smallest
# of those six pool sizes whose direction reads a mean cosine of at least 0.98 to the full-pool
# direction (0.9834) and keeps at least 90% of the full-pool d' (93.3%): 8 summaries, 71.3
# statements on average. The next smaller pool measured, 6 summaries with 53.9 statements, reads
# 0.9652 and 88.8%; no pool between 54 and 71 statements was measured. The rule is the statement
# count; on that fixture, whose summaries hold 1 to 13 statements, it is 8 documents. A pool
# below the minimum is still scored: the symmetric scoring calls of DocDistance (pipeline.py) log
# one warning that names its count.
ANISOTROPY_MIN_POOL = 71


def cost_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Ground cost ``sqrt(2 - 2cos)`` = Euclidean distance on L2-normalized rows (a metric)."""
    return ot.dist(X, Y, metric="euclidean")


def _uniform(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def _ab(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _uniform(len(X)), _uniform(len(Y))


class SolverNotOptimal(RuntimeError):
    """The exact solver returned without an optimal coupling, so its cost is not the distance."""


# POT's default cap on network-simplex iterations; the cap never drops below it, so a problem the
# default cap solves runs the same iterations to the same result
EMD_MIN_ITER = 100_000


def _emd_iter_cap(n: int, m: int) -> int:
    """Iteration cap of an ``n x m`` exact solve: one iteration per arc, never below POT's default.

    The iterations needed grow with the size: 3000 x 3000 random unit vectors need up to 138000,
    more than POT's fixed default of 100000 allows. Measured need per arc (POT 0.9.6, random unit
    vectors): 0.015 at 3000 x 3000, 0.65 at 60000 x 2 and exactly 1 at ``n x 1``. A problem with
    one statement on one side is solved at ``n`` iterations and not at ``n - 1``, so above 100000
    statements the cap has no spare iteration on that shape. The readings are printed by
    ``reports/E13-campaign/checks/solver-cap/cap_readings.py``.
    """
    return max(EMD_MIN_ITER, n * m)


def _exact_ot(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact network-simplex transport of ``a`` onto ``b`` under ground cost ``C``: ``(plan, cost)``.

    The one entry to the exact solver. POT only warns when the solver ends without an optimal
    coupling and still returns a result. That result is not optimal and can be infeasible, so its
    cost is not the distance and is not a bound of it: a 12 x 9 problem stopped after 1 iteration
    returned 58% of the mass at cost 0.687 against the distance 1.223, and a 3000 x 3000 problem
    stopped at POT's default cap returned a feasible coupling at 0.661948 against 0.661777. Any
    solver warning therefore raises :class:`SolverNotOptimal` instead.
    """
    n, m = C.shape
    cap = _emd_iter_cap(n, m)
    plan, log = ot.emd(a, b, C, numItermax=cap, log=True)
    if log["warning"] is not None:
        raise SolverNotOptimal(
            f"the exact solver stopped before optimality on a {n} x {m} problem "
            f"(iteration cap {cap}); POT reports: {log['warning']}"
        )
    return plan, log["cost"]


def smd(X: np.ndarray, Y: np.ndarray) -> float:
    """The distance: exact Statement Mover's Distance via the network-simplex LP."""
    return _exact_ot(*_ab(X, Y), cost_matrix(X, Y))[1]


def transport_plan(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """The exact optimal-transport coupling behind :func:`smd`; shape ``[n_X, n_Y]`` - the transport map.

    ``T[i, j]`` is the probability mass moved from statement ``X[i]`` to statement ``Y[j]``. Row ``i``
    sums to the source marginal ``1/n_X`` and column ``j`` to ``1/n_Y``, and ``(T * cost_matrix(X, Y)).sum()``
    equals ``smd(X, Y)``. The network-simplex solution is sparse (at most ``n_X + n_Y − 1`` nonzeros), so
    most statements map to one or a few others - the interpretable statement-to-statement alignment the
    distance is built from.
    """
    return _exact_ot(*_ab(X, Y), cost_matrix(X, Y))[0]


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
    pooled statements of all documents in ``emb`` so the common direction is shared. The estimate
    needs a pool of about :data:`ANISOTROPY_MIN_POOL` (71) statements (E12-H65).

    Which of the two steps carries the effect was measured per path, at ``k=1``:

    - Symmetric panels: the centering carries the gain. Centering and re-normalizing without the
      projection recovers 103.7% of the margin gain of the full pre-pass on the ibm panel (E12-H64)
      and 102.8% on the Wergeland pool (E14-H75). The projection isolate (first centered component
      removed, mean restored) reads -4.5% and -3.6%.
    - Source-conditioned path, ibm fixture, plain reader: centering without the projection reads
      0.5471x the ``DR(D_sel)`` of the full pre-pass and 1 of 28 selection-axis violations against
      0 of 28 (E14-H76). The difference is traced to 22 one-word markup statements that the plain
      reader returns from that source (``-`` 13 times, ``###`` 6, ``##`` 2, ``Overview`` 1): with
      them removed, centering alone reads 0.9165x at 0 of 28. This is one source and one reader
      setting, not a reading of the source-conditioned path in general.
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

    The gap of a document with itself is in general not zero (it is 0.0 for one statement) and grows
    with the statement count: ``opw_gap(P, P)`` reads 5.182e-03 at 30 statements and 6.786e-03 at 53
    (E13-H70, one document). On unequal statement counts the gap also includes the cost of the surplus
    statements, so a deletion that keeps the order of the remaining statements reads like a reorder:
    0.227 for 2 of 10 statements deleted (DEF-TRANS-10). :func:`matched_order_gap` is the reading over
    the statements both documents hold.
    """
    D = cost_matrix(X, Y)
    cost = float((_opw_plan(D, lambda1, lambda2, sigma, iters) * D).sum())
    content = _exact_ot(*_ab(X, Y), D)[1]
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
    return _exact_ot(*_ab(X, Y), D + eps * pos)[0]


def order_alignment(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Per ``X`` statement, its aligned ``Y`` index - the crisp exact-EMD argmax (diagonal tie-break)."""
    return _aligned_plan(X, Y).argmax(1)


def structure_displacement(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Rank-based position shift from the crisp alignment; ``0`` = in place, nonzero = moved.

    Both argsorts are ``kind="stable"`` so tied alignment targets (forced when ``len(X) > len(Y)``)
    rank by source position, and an order-preserving shrinkage yields zero displacement everywhere.
    """
    align = order_alignment(X, Y)
    order = np.argsort(np.argsort(align, kind="stable"), kind="stable")
    return order - np.arange(len(order))


# --- E13-H71 partial matching: one-to-one statement pairs + E13-H70 order gap over them ---


# weight of the position term in the partial-matching assignment cost (E15-H79), six orders of
# magnitude below MATCH_CUTOFF: it decides among matchings of equal ground cost and changed 0 of
# the 13223 matchings measured on documents without repeated statements
MATCH_POSITION_EPS = 1e-6


def partial_matching(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """One-to-one statement pairs of ``X`` and ``Y``; shape ``[n_pairs, 2]``, rows ``(i, j)`` ascending in ``i``.

    A unit-capacity assignment on the ground cost capped at :data:`MATCH_CUTOFF`: each statement is in
    at most one pair, and a pair whose ground cost is at or above the cutoff is left out, so a statement
    without a counterpart stays unmatched (E13-H71). The pairs annotate the two documents; they are
    not a distance and not a metric.

    The assignment cost carries the position term ``MATCH_POSITION_EPS * abs(i/n - j/m)``, positions
    counted from 0. The matching is deterministic: two calls return the same pairs. The term ranks
    the copies of a repeated statement by position: among matchings of exactly equal capped ground
    cost, one with the smallest sum of ``abs(i/n - j/m)`` is returned. The copies are not always
    paired in document order. ``abs`` is convex but not strictly convex: when every tied copy shifts
    the same way by more than the spacing of the copies, the in-order and the crossed pairing have
    the same position sum and the assignment solver decides. Example: 16 statements with one
    statement at positions 8 and 9, statements 11, 13 and 14 deleted, returns the pairs ``(8, 9)``
    and ``(9, 8)``. The term does not read which copy of a repeated statement was deleted. Measured
    limit (E15-H79, Refuted on this bar): on a 62-statement document with 9 repeated statements the
    matched statements were textually equal to the shorter document in 111 of 120 pure deletions,
    against a bar of 114. A cost difference above the weight is not ranked: the rows of one
    statement encoded inside two documents differed by up to 4.095e-03 in one embedding entry
    (E15-H79).
    """
    cost = cost_matrix(X, Y)
    n, m = cost.shape
    position = np.abs(np.arange(n)[:, None] / n - np.arange(m)[None, :] / m)
    rows, cols = linear_sum_assignment(
        np.minimum(cost, MATCH_CUTOFF) + MATCH_POSITION_EPS * position
    )
    kept = cost[rows, cols] < MATCH_CUTOFF  # the ground cost, without the position term
    return np.column_stack([rows[kept], cols[kept]])


def _order_gap_over(X: np.ndarray, Y: np.ndarray, pairs: np.ndarray) -> float | None:
    """:func:`opw_gap` of the statements named in ``pairs``, each side in its own document order.

    ``None`` when ``pairs`` is empty. A fully matched pair of equal statement counts goes to
    :func:`opw_gap` unmodified: POT sets the cost diagonal to exactly 0 only when both arguments are
    one array object, so indexed copies of ``(E, E)`` read a different last digit than ``(E, E)``.
    """
    if len(pairs) == 0:
        return None
    if len(pairs) == len(X) == len(Y):
        return opw_gap(X, Y)
    return opw_gap(X[pairs[:, 0]], Y[np.sort(pairs[:, 1])])


def matched_order_gap(X: np.ndarray, Y: np.ndarray) -> float | None:
    """Order gap over the statements both documents hold: :func:`opw_gap` of the matched sub-sequences (E13-H70).

    The statements paired by :func:`partial_matching` are kept, each document in its own order, and the
    shipped :func:`opw_gap` is read on them; ``None`` when no pair is matched. A SCORE, NOT a metric,
    like :func:`opw_gap`. With equal statement counts and every statement matched the value is exactly
    ``opw_gap(X, Y)``.

    A deletion that keeps the order of the remaining statements reads the order gap of the shorter
    document against a copy of itself, in general not zero: that floor grows with the statement count
    (see :func:`opw_gap`) and differs from ``opw_gap(Y, Y)`` on one array object by float32 rounding of
    the zero costs (at most 7.813e-06 measured, E15-H79). Measured on documents of 11 to 14 statements
    without repeated statements: 280 of 280 such deletions read at or below 0.005, where ``opw_gap``
    read 0 of 280. With repeated statements the limit of :func:`partial_matching` applies.
    """
    return _order_gap_over(X, Y, partial_matching(X, Y))


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
    return _exact_ot(np.asarray(cov_a), np.asarray(cov_b), cost_matrix(S, S))[1]


def ungrounded_residual(X: np.ndarray, S: np.ndarray) -> float:
    """Balanced transport cost from document ``X`` to the whole source ``S``: ``smd(X, S)``, a metric.

    Both marginals are uniform, so each source statement accepts ``1/n_S`` of the mass: a statement of
    a document shorter than the source keeps at most ``n_X/n_S`` of its mass on its nearest source
    statement. The cost therefore grows when a document covers only part of the source: 10 statements
    copied verbatim from a 70-statement source read 0.958 (DEF-TRANS-8). It is not the share of the
    document the source does not support. :func:`unsupported_count` (``n_unsupported_a`` /
    ``n_unsupported_b`` of the result) counts the statements with no near source statement
    (nearest-source cost above :data:`SOURCE_SUPPORT_CUTOFF`); it is not a reading of entailment.
    The model-graded reading is :func:`grounding_residual` (reranker + NLI; ``grd_a`` / ``grd_b``).
    """
    return smd(X, S)


def unsupported_count(X: np.ndarray, S: np.ndarray) -> int:
    """Count of statements of ``X`` whose nearest source statement costs more than :data:`SOURCE_SUPPORT_CUTOFF`.

    One-sided: each statement is compared with its nearest statement of ``S`` only, so a document that
    covers part of the source is not charged for the rest of it, and removing a supported statement
    leaves the count unchanged (E15-H78). A SCORE, NOT a metric. Pass raw embeddings: the cutoff is
    not valid after :func:`all_but_the_top`.

    The count reads distance to the source, not entailment by it, and does not replace
    :func:`grounding_residual`. A faithful summary can read above 0: 8 of the 77 statements of the 7
    gold-tier ibm summaries are counted (E15-H78).
    """
    return int((cost_matrix(X, S).min(1) > SOURCE_SUPPORT_CUTOFF).sum())


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
    """Structural (order) distance result. ``order_gap`` is the E11-H55 OPW order-gap; ``structure_closeness`` its readout.

    On unequal statement counts ``order_gap`` includes the cost of the surplus statements: a deletion
    that keeps the order of the remaining statements lowers ``structure_closeness`` and can turn
    ``verdict`` to ``not similar`` (DEF-TRANS-10). ``matched_order_gap`` is the order gap over the
    ``n_matched`` statement pairs both documents hold (:func:`matched_order_gap`), ``None`` when no
    pair is matched; it equals ``order_gap`` when the counts are equal and every statement is matched.
    For a document against itself both gaps are in general not zero and grow with the statement count
    (see :func:`opw_gap`). ``structure_closeness`` and ``verdict`` are read from ``order_gap`` only.
    """

    smd: float
    order_gap: float
    structure_closeness: float
    threshold: float
    verdict: str
    anisotropy: bool
    n_statements_a: int
    n_statements_b: int
    matched_order_gap: float | None
    n_matched: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SourceConditionedResult:
    """Source-conditioned result: the selection axis, each document's grounding residual, distances to S.

    ``grd_a`` / ``grd_b`` are the reranker x NLI grounding residuals (E03-H11), present only when the
    grounding models scored; ``d_grd`` is the grounding-axis separation ``|grd_a - grd_b|``. They stay
    ``None`` on the metric-only path (no models).

    ``residual_a`` / ``residual_b`` are the balanced transport cost from each document to the whole
    source (:func:`ungrounded_residual`). They grow when a document covers only part of the source -
    10 statements copied verbatim from a 70-statement source read 0.958 - so they are not the share of
    a document the source does not support. ``n_unsupported_a`` / ``n_unsupported_b`` are the count
    of statements with no near source statement: the nearest source statement costs more than
    :data:`SOURCE_SUPPORT_CUTOFF` (:func:`unsupported_count`), taken on the embeddings as passed in,
    before the anisotropy pre-pass. A score, not a metric, and not a reading of entailment: the
    model-graded reading is ``grd_a`` / ``grd_b`` (:func:`grounding_residual`).
    """

    d_sel: float
    residual_a: float
    residual_b: float
    closeness_a: float
    closeness_b: float
    n_statements_a: int
    n_statements_b: int
    n_statements_source: int
    n_unsupported_a: int
    n_unsupported_b: int
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

    The pool here is the statements of the two documents, and the minimum estimation pool is about
    :data:`ANISOTROPY_MIN_POOL` (71) pooled statements. The rule is the statement count. Measured in
    E12-H65 with the torch fp32 encoder, not the default OpenVINO INT8 one, on pools of 2, 3, 4, 6,
    8 and 11 of the 11 ibm summaries: 71 is the smallest of those six pool sizes whose direction
    reads a mean cosine of at least 0.98 to the full-pool direction (0.9834) and keeps at least 90%
    of the full-pool d' (93.3%). On that fixture this is 8 summaries, 71.3 statements on average; a
    pool of 2 of them holds 19 statements on average and reads a cosine of 0.8664. A smaller pool
    is scored all the same, and :class:`~docdistance.pipeline.DocDistance` logs a warning for it.
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
    threshold: float = STRUCTURE_THRESHOLD,
) -> StructuralResult:
    """Assemble a :class:`StructuralResult` from two statement-embedding arrays.

    The structural (order) axis: content is reported via ``smd``, structure via the E11-H55 OPW
    ``order_gap``, and ``structure_closeness = closeness(order_gap)`` is the SMD-scale readout
    (1 = same order). The verdict uses :data:`STRUCTURE_THRESHOLD`, not the semantic cutoff -
    the order-gap scale is compressed, so a semantic threshold never fires here.
    ``anisotropy`` is off by default for the same reason as :func:`compute_distance`, and the same
    minimum estimation pool applies: about :data:`ANISOTROPY_MIN_POOL` (71) pooled statements.

    ``n_matched`` counts the pairs of :func:`partial_matching` on the embeddings as passed in, the
    geometry :data:`MATCH_CUTOFF` is calibrated on, so the pairs are the same with ``anisotropy`` on
    and off. ``matched_order_gap`` is the order gap over those pairs in the geometry ``order_gap``
    uses.
    """
    n_a, n_b = len(emb_a), len(emb_b)
    pairs = partial_matching(emb_a, emb_b)
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
        matched_order_gap=_order_gap_over(emb_a, emb_b, pairs),
        n_matched=len(pairs),
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
    shared direction is estimated from, and removing it widens the ``D_sel`` dynamic range 6.34x at
    0 selection-axis violations on the 10 ibm triples (E14-H76; lever E04-H15). Pass
    ``anisotropy=False`` to opt out. The pool is the statements of A, B and the source together.
    The minimum estimation pool of about :data:`ANISOTROPY_MIN_POOL` (71) pooled statements was
    measured on pools of summaries without a source (E12-H65, see :func:`compute_distance`); on
    this path a pool below it logs no warning.

    When the grounding arrays are supplied (``reranker_*`` the doc x source relevance grid,
    ``entail_*`` the per-statement entailment probability), the E03-H11 grounding residual is computed
    per document into ``grd_a`` / ``grd_b`` and ``d_grd = |grd_a - grd_b|``; without them those stay
    ``None`` and only the metric selection axis + geometric residuals are returned.

    ``n_unsupported_a`` / ``n_unsupported_b`` (:func:`unsupported_count`) are taken on the embeddings
    as passed in, before the pre-pass: :data:`SOURCE_SUPPORT_CUTOFF` is valid on raw embeddings only
    (held-out balanced accuracy 0.9412 / 0.9272 raw, 0.8168 / 0.7277 after the pre-pass, E15-H78), so
    ``anisotropy`` on and off return the same counts.
    """
    n_a, n_b, n_s = len(emb_a), len(emb_b), len(emb_source)
    unsupported_a = unsupported_count(emb_a, emb_source)
    unsupported_b = unsupported_count(emb_b, emb_source)
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
        n_unsupported_a=unsupported_a,
        n_unsupported_b=unsupported_b,
        coverage_a=cov_a.tolist(),
        coverage_b=cov_b.tolist(),
        grd_a=grd_a,
        grd_b=grd_b,
        d_grd=d_grd,
    )
