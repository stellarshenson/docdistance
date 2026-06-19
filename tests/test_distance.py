"""Pure-core tests for the optimal-transport distance math.

No models are loaded - the functions in :mod:`docdistance.distance` operate on synthetic
L2-normalized embedding arrays, so the whole suite runs in the lightweight uv ``.venv``.
"""

import numpy as np
import pytest

from docdistance import distance as d


def _emb(n: int, dim: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


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


def test_compute_source_conditioned_result():
    r = d.compute_source_conditioned(_emb(7, seed=16), _emb(6, seed=17), _emb(10, seed=18))
    assert r.n_statements_source == 10
    assert len(r.coverage_a) == 10
    assert r.d_sel >= 0.0
    assert 0.0 <= r.closeness_a <= 1.0
