"""Pure-core tests for the optimal-transport distance math.

No models are loaded - the functions in :mod:`docdistance.distance` operate on synthetic
L2-normalized embedding arrays, so the whole suite runs in the lightweight uv ``.venv``.
"""

import inspect
import json
import warnings

import numpy as np
import ot
import pytest

from docdistance import distance as d


def _emb(n: int, dim: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _anisotropic(n: int, seed: int, dim: int = 32, spread: float = 0.1) -> np.ndarray:
    """Unit vectors that share one direction, like raw statement embeddings: distinct rows cost about 0.7."""
    x = 1.0 / np.sqrt(dim) + spread * np.random.default_rng(seed).standard_normal((n, dim))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def _reworded(E: np.ndarray, scale: float, seed: int) -> np.ndarray:
    """Each row of ``E`` moved by gaussian noise of the given scale, then L2-normalized again."""
    noisy = E + scale * np.random.default_rng(seed).standard_normal(E.shape).astype(np.float32)
    return noisy / np.linalg.norm(noisy, axis=1, keepdims=True)


def _source(n: int = 12) -> np.ndarray:
    """A source of ``n`` statements that occupies the first 16 of 32 dimensions."""
    return np.pad(_emb(n, dim=16, seed=51), ((0, 0), (0, 16)))


def _off_source(n: int) -> np.ndarray:
    """``n`` statements orthogonal to every row of :func:`_source`: ground cost sqrt(2) to each."""
    return np.pad(_emb(n, dim=16, seed=52), ((0, 0), (16, 0)))


def test_identity_is_zero():
    X = _emb(8)
    assert d.smd(X, X) == pytest.approx(0.0, abs=1e-6)


def test_symmetry():
    X, Y = _emb(8, seed=1), _emb(6, seed=2)
    assert d.smd(X, Y) == pytest.approx(d.smd(Y, X), abs=1e-6)


def test_lower_bound_nesting():
    """WCD <= RWMD <= SMD - the Kusner et al. lower-bound nesting."""
    X, Y = _emb(10, seed=3), _emb(9, seed=4)
    w, r, s = d.wcd(X, Y), d.rwmd(X, Y), d.smd(X, Y)
    assert w <= r + 1e-6
    assert r <= s + 1e-6


def test_transport_plan_marginals_and_realizes_smd():
    """The coupling has uniform marginals 1/n and its cost-weighted sum equals SMD."""
    X, Y = _emb(7, seed=11), _emb(9, seed=12)
    T = d.transport_plan(X, Y)
    assert T.shape == (7, 9)
    assert (T >= -1e-9).all()
    assert np.allclose(T.sum(1), 1.0 / 7, atol=1e-6)  # row (source) marginal
    assert np.allclose(T.sum(0), 1.0 / 9, atol=1e-6)  # column (target) marginal
    realized = float((T * d.cost_matrix(X, Y)).sum())
    assert realized == pytest.approx(d.smd(X, Y), abs=1e-6)


def test_exact_solver_matches_the_bare_pot_call():
    """Below POT's default iteration cap the guarded solve returns the bare POT result bit for bit."""
    X, Y = _emb(40, seed=21), _emb(55, seed=22)
    a, b, C = d._uniform(40), d._uniform(55), d.cost_matrix(X, Y)
    assert d.smd(X, Y) == float(ot.emd2(a, b, C))
    assert np.array_equal(d.transport_plan(X, Y), ot.emd(a, b, C))


def _selection_divergence(X: np.ndarray, Y: np.ndarray) -> float:
    return d.selection_divergence(d.coverage_profile(X, Y), d.coverage_profile(Y, Y), Y)


@pytest.mark.filterwarnings("ignore:numItermax reached")
@pytest.mark.parametrize(
    "solve", [d.smd, d.transport_plan, d.opw_gap, d.order_alignment, _selection_divergence]
)
def test_exact_solver_raises_when_it_stops_at_the_iteration_cap(monkeypatch, solve):
    """DEF-TRANS-9: a solve that stops at its iteration cap raises at every exact-solver call site.

    The coupling POT returns at the cap is not optimal and can be infeasible, so its cost is not the
    distance and is not a bound of it.
    """
    monkeypatch.setattr(d, "_emd_iter_cap", lambda n, m: 1)
    with pytest.raises(d.SolverNotOptimal, match="numItermax reached"):
        solve(_emb(12, seed=23), _emb(9, seed=24))


def test_exact_solver_cap_scales_with_the_problem_size():
    """DEF-TRANS-9: 3000 statements per side solve to optimality, with no POT warning.

    Non-negative random unit vectors (pairwise cosine near 0.75, positive like the mmBERT statement
    cosines) need about 138000 simplex iterations at this size - above POT's default cap of 100000,
    which returns a non-optimal value with the warning ``numItermax reached before optimality``.
    """
    x = np.random.default_rng(0).random((6000, 768)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the POT warning fails the test
        d.smd(x[:3000], x[3000:])


def test_exact_solver_cap_is_never_below_the_pot_default():
    """DEF-TRANS-9: the floor of the cap is never below POT's own default, so a problem 1.1.5 solved gets no smaller cap.

    A 12 x 9 problem has 108 arcs; one iteration per arc alone would give it 108 iterations where
    1.1.5 allowed 100000. Above the floor the cap is one iteration per arc.
    """
    pot_default = inspect.signature(ot.emd).parameters["numItermax"].default
    assert d._emd_iter_cap(12, 9) == d.EMD_MIN_ITER
    assert d.EMD_MIN_ITER >= pot_default
    assert d._emd_iter_cap(400, 300) == 120_000


def test_solver_not_optimal_is_exported_from_the_package_root():
    import docdistance

    assert docdistance.SolverNotOptimal is d.SolverNotOptimal
    assert "SolverNotOptimal" in docdistance.__all__


def test_closeness_range():
    assert d.closeness(0.0) == pytest.approx(1.0)
    assert d.closeness(d.SMD_MAX) == pytest.approx(0.0)
    c = d.closeness(d.smd(_emb(5, seed=5), _emb(7, seed=6)))
    assert 0.0 <= c <= 1.0


def test_anisotropy_shapes_and_renormalization():
    emb = {"a": _emb(5, seed=7), "b": _emb(6, seed=8)}
    out = d.all_but_the_top(emb, k=1)
    assert out["a"].shape == emb["a"].shape
    assert out["b"].shape == emb["b"].shape
    norms = np.linalg.norm(np.concatenate([out["a"], out["b"]], 0), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_coverage_profile_is_a_distribution():
    X, S = _emb(8, seed=9), _emb(11, seed=10)
    cov = d.coverage_profile(X, S)
    assert cov.shape == (11,)
    assert cov.sum() == pytest.approx(1.0, abs=1e-6)
    assert (cov >= -1e-9).all()


def test_coverage_alignment_rows_sum_to_one_and_mean_is_profile():
    X, S = _emb(8, seed=9), _emb(11, seed=10)
    align = d.coverage_alignment(X, S)
    assert align.shape == (8, 11)
    assert np.allclose(align.sum(1), 1.0, atol=1e-6)
    assert (align >= -1e-9).all()
    assert np.allclose(align.mean(0), d.coverage_profile(X, S), atol=1e-6)


def test_selection_divergence_identity_and_symmetry():
    S = _emb(10, seed=11)
    cov_a = d.coverage_profile(_emb(7, seed=12), S)
    cov_b = d.coverage_profile(_emb(6, seed=13), S)
    assert d.selection_divergence(cov_a, cov_a, S) == pytest.approx(0.0, abs=1e-6)
    assert d.selection_divergence(cov_a, cov_b, S) == pytest.approx(
        d.selection_divergence(cov_b, cov_a, S), abs=1e-6
    )


def test_compute_distance_result():
    r = d.compute_distance(_emb(9, seed=14), _emb(8, seed=15))
    assert r.verdict in ("similar", "not similar")
    assert r.wcd <= r.rwmd + 1e-6
    assert r.rwmd <= r.smd + 1e-6
    assert set(r.to_dict()) >= {"smd", "closeness", "verdict", "threshold"}


def test_compute_structural_result():
    """compute_structural assembles a StructuralResult: order-gap axis, closeness readout, str verdict."""
    r = d.compute_structural(_emb(9, seed=14), _emb(8, seed=15))
    assert isinstance(r.verdict, str) and r.verdict in ("similar", "not similar")
    assert r.order_gap >= 0.0
    assert 0.0 <= r.structure_closeness <= 1.0
    assert r.smd >= 0.0
    assert r.n_statements_a == 9 and r.n_statements_b == 8
    assert r.n_matched == 0 and r.matched_order_gap is None  # two unrelated documents
    assert set(r.to_dict()) == {
        "smd",
        "order_gap",
        "structure_closeness",
        "threshold",
        "verdict",
        "anisotropy",
        "n_statements_a",
        "n_statements_b",
        "matched_order_gap",
        "n_matched",
    }
    assert d.StructuralResult(**r.to_dict()) == r  # to_dict round-trips


def test_compute_structural_reads_a_pure_deletion_on_the_matched_gap():
    """DEF-TRANS-10 / ACC-TRANS-54: 2 of 10 statements deleted, order kept.

    ``order_gap``, ``structure_closeness`` and ``verdict`` keep the shipped reading of the pair - a
    reorder-sized gap; ``matched_order_gap`` reads the 8 statements both documents hold.
    """
    A = _emb(10, seed=25)
    deleted = np.random.default_rng(0).choice(10, size=2, replace=False)
    B = A[np.setdiff1d(np.arange(10), deleted)]
    r = d.compute_structural(A, B)
    assert r.order_gap == d.opw_gap(A, B)
    assert r.structure_closeness == d.closeness(r.order_gap)
    assert r.verdict == d.verdict(r.structure_closeness, d.STRUCTURE_THRESHOLD)
    assert r.n_matched == 8
    assert r.matched_order_gap == d.opw_gap(B.copy(), B)  # the order gap of B against itself
    assert r.matched_order_gap < 1e-6
    assert r.order_gap > 0.01
    json.loads(json.dumps(r.to_dict()))  # the new fields are plain JSON values


def test_compute_structural_matches_on_raw_embeddings_under_anisotropy():
    """ACC-TRANS-76: the pairs come from the embeddings as passed in; their gap is read like ``order_gap``.

    The fixture shares one direction, like raw statement embeddings. On the raw embeddings every
    reworded statement costs below ``MATCH_CUTOFF`` to its original; after the pre-pass the same
    pairs cost more than the cutoff, so a matching taken there returns fewer pairs.
    """
    A = _anisotropic(8, seed=43)
    B = _reworded(A, 0.04, seed=44)[np.random.default_rng(45).permutation(8)]
    fixed = d.all_but_the_top({"a": A, "b": B}, k=1)
    assert (
        len(d.partial_matching(fixed["a"], fixed["b"])) < 8
    )  # the fixture tells the geometries apart

    off = d.compute_structural(A, B)
    on = d.compute_structural(A, B, anisotropy=True)
    assert off.n_matched == on.n_matched == 8
    assert off.matched_order_gap == off.order_gap  # fully matched, equal counts
    assert on.matched_order_gap == on.order_gap  # the same pair in the pre-pass geometry
    assert on.order_gap != off.order_gap

    # 2 statements of B removed: the 6 raw pairs, their gap in the pre-pass geometry
    B = B[[0, 1, 2, 4, 5, 7]]
    pairs = d.partial_matching(A, B)
    fixed = d.all_but_the_top({"a": A, "b": B}, k=1)
    on = d.compute_structural(A, B, anisotropy=True)
    assert on.n_matched == d.compute_structural(A, B).n_matched == len(pairs) == 6
    assert on.matched_order_gap == d.opw_gap(
        fixed["a"][pairs[:, 0]], fixed["b"][np.sort(pairs[:, 1])]
    )


def test_compute_source_conditioned_result():
    r = d.compute_source_conditioned(_emb(7, seed=16), _emb(6, seed=17), _emb(10, seed=18))
    assert r.n_statements_source == 10
    assert len(r.coverage_a) == 10
    assert r.d_sel >= 0.0
    assert 0.0 <= r.closeness_a <= 1.0
    assert r.grd_a is None and r.grd_b is None and r.d_grd is None  # no grounding without models
    assert (r.n_unsupported_a, r.n_unsupported_b) == (7, 6)  # unrelated documents: every statement
    json.loads(json.dumps(r.to_dict()))  # the counts are plain JSON integers


def test_unsupported_count_reads_statements_far_from_every_source_statement():
    """ACC-TRANS-67: 3 copies of source rows plus 2 rows orthogonal to the source read 2."""
    import docdistance

    S = _source()
    copies = S[[1, 5, 9]]
    assert d.unsupported_count(copies, S) == 0
    assert d.unsupported_count(np.concatenate([copies, _off_source(2)]), S) == 2
    assert docdistance.unsupported_count is d.unsupported_count
    assert "unsupported_count" in docdistance.__all__


def test_unsupported_counts_are_read_before_the_anisotropy_pre_pass():
    """ACC-TRANS-67: the counts are taken on the embeddings as passed in, so anisotropy on and off agree.

    The fixture shares one direction, like raw statement embeddings: 3 reworded source statements
    (cost about 0.25 to their original) and 2 statements of the same cloud that reword no source
    statement (cost above 0.5). After the pre-pass all 5 cost more than ``SOURCE_SUPPORT_CUTOFF``.
    """
    S = _anisotropic(12, seed=53)
    A = np.concatenate([_reworded(S[[1, 5, 9]], 0.04, seed=55), _anisotropic(2, seed=54)])
    B = S[[0, 2, 4]]
    fixed = d.all_but_the_top({"a": A, "b": B, "s": S}, k=1)
    assert (
        d.unsupported_count(fixed["a"], fixed["s"]) == 5
    )  # the fixture tells the geometries apart

    on = d.compute_source_conditioned(A, B, S)
    off = d.compute_source_conditioned(A, B, S, anisotropy=False)
    assert (on.n_unsupported_a, on.n_unsupported_b) == (2, 0)
    assert (off.n_unsupported_a, off.n_unsupported_b) == (2, 0)


def test_verbatim_extract_reads_no_unsupported_statement():
    """ACC-TRANS-68 / DEF-TRANS-8: 10 statements copied from a 70-statement source.

    The balanced residual moves 6/7 of each statement's mass to other source statements and reads
    far from 0; the count reads that every statement has its copy in the source.
    """
    S = _emb(70, seed=56)
    X = S[0::7][:10]
    r = d.compute_source_conditioned(X, X, S)
    assert r.n_unsupported_a == 0 and r.n_unsupported_b == 0
    assert r.residual_a > 0.5
    assert r.closeness_a == d.closeness(r.residual_a)


def test_unsupported_count_leaves_out_a_statement_exactly_at_the_cutoff(monkeypatch):
    """ACC-TRANS-67: the count is ``>``, so a statement whose nearest source cost equals the cutoff is supported.

    The cutoff is set to the nearest source cost of statement 1, bit for bit: that statement is not
    counted. One float32 step below, it is counted.
    """
    S = _source()
    X = np.concatenate([S[[4]], _reworded(S[[7]], 0.05, seed=57), _off_source(1)])
    nearest = d.cost_matrix(X, S).min(1)
    assert 0.0 < nearest[1] < nearest[2]  # a copy, a reworded statement, a foreign statement

    monkeypatch.setattr(d, "SOURCE_SUPPORT_CUTOFF", float(nearest[1]))
    assert d.unsupported_count(X, S) == 1  # the foreign statement only
    monkeypatch.setattr(
        d, "SOURCE_SUPPORT_CUTOFF", float(np.nextafter(nearest[1], np.float32(0.0)))
    )
    assert d.unsupported_count(X, S) == 2


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_unsupported_count_ignores_deleted_supported_statements(k):
    """ACC-TRANS-69: k supported statements removed leave the count at 0; k unsupported ones appended raise it by k."""
    S = _source()
    X = S[[0, 2, 3, 5, 7, 8, 10]]
    assert d.unsupported_count(X[k:], S) == 0
    assert d.unsupported_count(np.concatenate([X, _off_source(k)]), S) == k


def test_grounding_residual_matches_the_h11_formula():
    """mean_i (1 - entail_i) * (1 - max_j R[i,j]) - the E03-H11 relevance-gated residual."""
    R = np.array([[0.9, 0.1], [0.2, 0.3]])
    ent = np.array([1.0, 0.0])
    # statement 0: (1-1)*(1-0.9)=0 ; statement 1: (1-0)*(1-0.3)=0.7 ; mean = 0.35
    assert d.grounding_residual(R, ent) == pytest.approx(0.35, abs=1e-9)
    # accepts a precomputed per-statement max vector too
    assert d.grounding_residual(np.array([0.9, 0.3]), ent) == pytest.approx(0.35, abs=1e-9)


def test_grounding_residual_zero_when_fully_entailed():
    R = np.array([[0.4, 0.5], [0.1, 0.2]])
    assert d.grounding_residual(R, np.array([1.0, 1.0])) == pytest.approx(0.0, abs=1e-12)


def test_grounding_blend_is_the_h14_convex_combination():
    assert d.grounding_blend(0.2, 0.4) == pytest.approx(0.75 * 0.2 + 0.25 * 0.4)
    assert d.grounding_blend(1.0, 0.0, alpha=0.5) == pytest.approx(0.5)


def test_compute_source_conditioned_with_grounding_arrays():
    ea, eb, es = _emb(4, seed=1), _emb(3, seed=2), _emb(6, seed=3)
    rng = np.random.default_rng(0)
    ra, rb = rng.random((4, 6)), rng.random((3, 6))
    enta, entb = rng.random(4), rng.random(3)
    r = d.compute_source_conditioned(
        ea, eb, es, reranker_a=ra, reranker_b=rb, entail_a=enta, entail_b=entb
    )
    assert r.grd_a == pytest.approx(d.grounding_residual(ra, enta))
    assert r.grd_b == pytest.approx(d.grounding_residual(rb, entb))
    assert r.d_grd == pytest.approx(abs(r.grd_a - r.grd_b))


# --- calibration guards ----------------------------------------------------
# Both cutoffs are fitted on the 11 ibm-ai-adoption fixtures and are therefore in-sample.
# These pin them to the band the artefacts measured, so a future edit that moves a cutoff
# out of its separating range fails here instead of silently mis-classifying documents.


def test_default_threshold_separates_the_measured_fixture_band():
    """Semantic cutoff must split the tiers recorded in notebook 09 cell 10 (OpenVINO INT8):
    min gold 79.7% (haiku), max adversarial 75.0% (adv2-b)."""
    assert d.verdict(0.797, d.DEFAULT_THRESHOLD) == "similar"  # haiku, the closest gold
    assert (
        d.verdict(0.750, d.DEFAULT_THRESHOLD) == "not similar"
    )  # adv2-b, the furthest adversarial


def test_structure_threshold_splits_the_e11_scramble_ladder():
    """Structural cutoff must sit between rung 1 and rung 2 of the E11 scramble ladder.

    Ladder closeness (reports/E11-structure-sota-decision.json .scramble.ibm.H55), lightest
    reorder to full scramble: 0.9702 / 0.8928 / 0.8197 / 0.7609 / 0.7269 / 0.7058. The two-sided
    assertion is the point - a cutoff below rung 2 leaves real rearrangements reading "similar",
    and one above rung 1 flags a light reorder. A semantic cutoff fails the first: 0.725 catches
    only the deepest rung, 0.77 catches three of six.
    """
    assert d.verdict(0.9702, d.STRUCTURE_THRESHOLD) == "similar"  # rung 1, a light reorder
    assert d.verdict(0.8928, d.STRUCTURE_THRESHOLD) == "not similar"  # rung 2, a real one
