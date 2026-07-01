"""Offline tests for the structural (order) feature (OPW order-gap + per-statement displacement).

No models load - the OPW plan/gap, the crisp alignment and ``_build_structural_details`` all take
synthetic L2-normalized embedding arrays, so the whole battery runs in the lightweight uv ``.venv``.
Covers the core OPW math, order vs content sensitivity, the crisp alignment/displacement recovery and
the JSON-serializable structural details dict shape.
"""

import json

import numpy as np
import pytest

from docdistance import distance as d
from docdistance import pipeline
from docdistance.pipeline import _build_structural_details


def _emb(n, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_structure_functions_reexported_from_package_root():
    import docdistance as dd

    for name in ("opw_plan", "opw_cost", "opw_gap", "order_alignment", "structure_displacement"):
        assert getattr(dd, name) is getattr(d, name)
        assert name in dd.__all__


# --- OPW plan ------------------------------------------------------------------------------------


def test_opw_plan_shape_finite_nonnegative():
    X, Y = _emb(7, seed=1), _emb(9, seed=2)
    T = d.opw_plan(X, Y)
    assert T.shape == (7, 9)
    assert np.isfinite(T).all()
    assert (T >= 0.0).all()


def test_opw_plan_marginals_are_uniform():
    """Sinkhorn coupling has uniform row/col marginals 1/n, 1/m (col needs more iters to converge)."""
    X, Y = _emb(7, seed=1), _emb(9, seed=2)
    T = d.opw_plan(X, Y, iters=500)
    assert np.allclose(T.sum(1), 1.0 / 7, atol=1e-3)  # source marginal
    assert np.allclose(T.sum(0), 1.0 / 9, atol=1e-3)  # target marginal


def test_opw_plan_different_lengths_do_not_crash():
    for na, nb in [(1, 5), (5, 1), (3, 8), (12, 4)]:
        T = d.opw_plan(_emb(na, seed=na), _emb(nb, seed=nb + 50))
        assert T.shape == (na, nb)
        assert np.isfinite(T).all()


# --- OPW gap -------------------------------------------------------------------------------------


def test_opw_gap_nonnegative_on_many_pairs():
    for s in range(40):
        na = int(np.random.default_rng(s).integers(3, 12))
        nb = int(np.random.default_rng(s + 500).integers(3, 12))
        g = d.opw_gap(_emb(na, seed=s), _emb(nb, seed=s + 1000))
        assert g >= 0.0  # the function clamps the tiny Sinkhorn negative to 0


def test_opw_gap_zero_on_identical():
    E = _emb(8, seed=3)
    assert d.opw_gap(E, E) == pytest.approx(0.0, abs=1e-6)


def test_opw_gap_different_lengths_do_not_crash():
    for na, nb in [(4, 9), (9, 4), (1, 6), (6, 1)]:
        g = d.opw_gap(_emb(na, seed=na), _emb(nb, seed=nb + 30))
        assert g >= 0.0


# --- order sensitivity (the killer test) ---------------------------------------------------------


def test_order_sensitivity_content_invariant_but_order_gap_positive():
    """Permuting rows leaves the content distance (SMD) unchanged but opens a positive order-gap."""
    E = _emb(10, seed=5)
    perm = np.random.default_rng(0).permutation(10)
    # content invariance: same multiset of statements -> SMD unchanged (~0)
    assert d.smd(E, E[perm]) == pytest.approx(d.smd(E, E), abs=1e-3)
    assert d.smd(E, E[perm]) == pytest.approx(0.0, abs=1e-3)
    # structure moved: the order-gap detects the reordering
    assert d.opw_gap(E, E[perm]) > 0.0


def test_order_gap_grows_with_displacement():
    """A full reverse displaces every statement further than a single adjacent swap -> larger gap."""
    E = _emb(10, seed=6)
    swap = np.arange(10)
    swap[0], swap[1] = 1, 0
    gap_swap = d.opw_gap(E, E[swap])
    gap_reverse = d.opw_gap(E, E[::-1])
    assert gap_swap > 0.0
    assert gap_reverse > gap_swap


# --- crisp alignment / displacement --------------------------------------------------------------


def test_alignment_and_displacement_identity():
    E = _emb(9, seed=7)
    assert np.array_equal(d.order_alignment(E, E), np.arange(9))
    assert np.array_equal(d.structure_displacement(E, E), np.zeros(9, dtype=int))


def test_displacement_recovers_a_cyclic_shift():
    """A known cyclic row shift is recovered exactly by the rank-based displacement."""
    n, shift = 10, 3
    E = _emb(n, seed=8)
    Y = np.roll(E, shift, axis=0)  # Y[j] = E[(j - shift) % n]
    align = d.order_alignment(E, Y)
    assert np.array_equal(align, (np.arange(n) + shift) % n)
    disp = d.structure_displacement(E, Y)
    expected = ((np.arange(n) + shift) % n) - np.arange(n)  # [+shift]*..., then wraps negative
    assert np.array_equal(disp, expected)
    assert (disp != 0).all()


# --- _build_structural_details dict shape and semantics ------------------------------------------


def test_build_structural_details_top_level_keys():
    sa, ea = [f"a{i}" for i in range(5)], _emb(5, seed=11)
    sb, eb = [f"b{i}" for i in range(4)], _emb(4, seed=12)
    diff = _build_structural_details(sa, ea, sb, eb)
    assert set(diff) == {
        "smd",
        "order_gap",
        "structure_closeness",
        "anisotropy",
        "n_statements",
        "statements",
    }
    assert diff["n_statements"] == {"a": 5, "b": 4}
    assert diff["anisotropy"] is False
    assert diff["smd"] >= 0.0
    assert diff["order_gap"] >= 0.0


def test_build_structural_details_statement_records_and_json_serializable():
    sa, ea = [f"a{i}" for i in range(5)], _emb(5, seed=13)
    sb, eb = [f"b{i}" for i in range(5)], _emb(5, seed=14)
    diff = _build_structural_details(sa, ea, sb, eb)
    assert len(diff["statements"]) == len(sa)
    for i, st in enumerate(diff["statements"]):
        assert set(st) == {
            "index",
            "text",
            "target_index",
            "target_text",
            "displacement",
            "moved",
        }
        assert st["index"] == i and st["text"] == sa[i]
        assert 0 <= st["target_index"] < len(sb)
        assert st["target_text"] == sb[st["target_index"]]
        assert st["moved"] == (st["displacement"] != 0)
    json.dumps(diff)  # whole dict round-trips through JSON


def test_build_structural_details_structure_closeness_matches_core_and_range():
    sa, ea = [f"a{i}" for i in range(6)], _emb(6, seed=15)
    sb, eb = [f"b{i}" for i in range(6)], _emb(6, seed=16)
    diff = _build_structural_details(sa, ea, sb, eb)
    assert 0.0 <= diff["structure_closeness"] <= 1.0
    assert diff["structure_closeness"] == pytest.approx(d.closeness(diff["order_gap"]), abs=1e-6)


def test_build_structural_details_identical_is_closeness_one_and_gaps_zero():
    s, e = [f"s{i}" for i in range(6)], _emb(6, seed=17)
    diff = _build_structural_details(s, e, s, e)
    assert diff["smd"] == pytest.approx(0.0, abs=1e-5)
    assert diff["structure_closeness"] == pytest.approx(1.0, abs=1e-6)
    for st in diff["statements"]:
        assert st["target_index"] == st["index"]  # each statement aligns to its twin
        assert st["displacement"] == 0
        assert st["moved"] is False


def test_build_structural_details_structure_closeness_decreases_as_order_gap_grows():
    """A real reorder drops closeness below 1, and a larger displacement drops it further."""
    s, e = [f"s{i}" for i in range(10)], _emb(10, seed=18)
    swap = np.arange(10)
    swap[0], swap[1] = 1, 0
    diff_swap = _build_structural_details(s, e, [s[i] for i in swap], e[swap])
    diff_reverse = _build_structural_details(s, e, s[::-1], e[::-1])
    assert diff_swap["structure_closeness"] < 1.0
    assert diff_reverse["structure_closeness"] < 1.0
    assert diff_swap["structure_closeness"] > diff_reverse["structure_closeness"]


def test_build_structural_details_content_edit_leaves_order_intact():
    """Replacing one B row with a far vector changes content only; every statement stays in place."""
    n, k = 6, 2
    s, e = [f"s{i}" for i in range(n)], _emb(n, seed=19)
    eb = e.copy()
    eb[k] = _emb(1, seed=999)[0]  # a far-away random statement in place of the twin
    diff = _build_structural_details(s, e, s, eb)
    for st in diff["statements"]:
        assert st["displacement"] == 0  # content changed, order did not
        assert st["moved"] is False


def test_structural_distance_with_details_returns_result_and_details(monkeypatch):
    """structural_distance_with_details shares one encode pass and returns the StructuralResult plus the details dict."""
    from docdistance import settings

    settings.reset()
    settings.mark_ready("wmd")
    docs = {
        "A": ([f"a{i}" for i in range(5)], _emb(5, seed=21)),
        "B": ([f"b{i}" for i in range(4)], _emb(4, seed=22)),
    }
    dd = pipeline.DocDistance()
    monkeypatch.setattr(dd, "_ensure_base", lambda: None)
    monkeypatch.setattr(dd, "embed_statements", lambda doc: docs[doc])

    result, diff = dd.structural_distance_with_details("A", "B")
    assert result.verdict in ("similar", "not similar")
    assert result.order_gap >= 0.0
    assert set(diff) == {
        "smd",
        "order_gap",
        "structure_closeness",
        "anisotropy",
        "n_statements",
        "statements",
    }
    assert diff["n_statements"] == {"a": 5, "b": 4}
    assert result.smd == pytest.approx(diff["smd"], abs=1e-5)
    assert result.order_gap == pytest.approx(diff["order_gap"], abs=1e-5)
    settings.reset()


# --- regressions: OPW plan validity and alignment tie-breaks -------------------------------------


def test_opw_cost_not_below_smd_on_asymmetric_shapes():
    """Regression: the rounded OPW plan is a valid coupling, so opw_cost - smd never dips negative.

    Before the U(a,b) rounding the fixed-iteration Sinkhorn plan met only the row marginal, so on
    ``n != m`` shapes ``opw_cost`` could fall ~0.15 below the SMD - a large error the max(0, .) clamp
    hid rather than rounding noise.
    """
    for seed in range(80):
        na = int(np.random.default_rng(seed).integers(2, 8))
        nb = int(np.random.default_rng(seed + 7).integers(2, 8))
        dim = int(np.random.default_rng(seed + 13).integers(2, 8))
        X, Y = _emb(na, dim, seed), _emb(nb, dim, seed + 99)
        assert d.opw_cost(X, Y) - d.smd(X, Y) >= -1e-9


def test_opw_plan_marginals_uniform_at_default_iters():
    """Regression: both marginals are exact at the shipped default iters, even for asymmetric shapes.

    The raw loop left the column marginal off by up to 0.12 on shapes like 8x2; rounding onto the
    transport polytope forces both marginals to 1/n, 1/m exactly.
    """
    for na, nb in [(8, 2), (2, 40), (5, 11), (12, 4)]:
        T = d.opw_plan(_emb(na, seed=na), _emb(nb, seed=nb + 5))
        assert np.allclose(T.sum(1), 1.0 / na, atol=1e-9)  # row marginal
        assert np.allclose(T.sum(0), 1.0 / nb, atol=1e-9)  # column marginal


def test_displacement_zero_on_identical_doc_with_duplicate_statements():
    """Regression: a duplicated statement pair no longer invents displacement on an unchanged document.

    Duplicate rows make the ground cost degenerate; the exact EMD used to be free to pick a swap on
    the tied rows, which the rank transform turned into phantom nonzero displacement. The diagonal
    tie-break resolves ties to the in-place map.
    """
    for seed in range(50):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((5, 16)).astype(np.float32)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        dup = int(rng.integers(0, 4))
        X[dup + 1] = X[dup]  # an interchangeable (duplicated) adjacent statement pair
        assert np.array_equal(d.structure_displacement(X, X), np.zeros(5, dtype=int))


def test_build_structural_details_duplicate_content_edit_does_not_leak_displacement():
    """Regression: rewriting a statement to duplicate another's content invents no displacement, no leak.

    The edited statement's new embedding equals another statement's, so the exact-EMD alignment had a
    tie; before the tie-break it mapped the edited and duplicated rows across each other, reporting
    nonzero displacement on the untouched neighbour.
    """
    rng = np.random.default_rng(77)
    base = rng.standard_normal((5, 16)).astype(np.float32)
    base = base / np.linalg.norm(base, axis=1, keepdims=True)
    ea = base.copy()
    eb = base.copy()
    eb[1] = base[3]  # statement 1 rewritten to restate statement 3's content
    sa = [f"a{i}" for i in range(5)]
    sb = [f"b{i}" for i in range(5)]
    diff = _build_structural_details(sa, ea, sb, eb)
    for st in diff["statements"]:
        assert st["displacement"] == 0  # only content changed, nothing moved
        assert st["moved"] is False
