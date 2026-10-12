"""Offline tests for the structural (order) feature (OPW order-gap + per-statement displacement).

No models load - the OPW plan/gap, the crisp alignment and ``_build_structural_details`` all take
synthetic L2-normalized embedding arrays, so the whole battery runs in the lightweight uv ``.venv``.
Covers the core OPW math, order vs content sensitivity, the crisp alignment/displacement recovery and
the JSON-serializable structural details dict shape.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from docdistance import distance as d
from docdistance import pipeline
from docdistance.pipeline import _build_structural_details

# 1680 permutations of the 10 base documents of the structure fixture (11 to 14 statements)
_REORDER_POOL = (
    Path(__file__).resolve().parents[1] / "data/processed/structure-fixture/reorder_pool.json"
)


def _emb(n, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_structure_functions_reexported_from_package_root():
    import docdistance as dd

    for name in (
        "opw_plan",
        "opw_cost",
        "opw_gap",
        "order_alignment",
        "structure_displacement",
        "partial_matching",
        "matched_order_gap",
    ):
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


def test_displacement_zero_on_monotone_shrinkage_with_shared_targets():
    """n_A > n_B forces tied crisp targets; stable rank keeps an order-preserving shrinkage at 0."""
    B = _emb(4, seed=9)
    targets = np.array([0, 0, 1, 1, 2, 2, 3, 3])  # monotone, every target shared
    A = B[targets]
    assert np.array_equal(d.order_alignment(A, B), targets)
    assert np.array_equal(d.structure_displacement(A, B), np.zeros(8, dtype=int))


# --- partial matching and the matched order gap (E13-H70, E13-H71, E15-H79) ----------------------


def _reworded(E, scale, seed):
    """Each row of ``E`` moved by gaussian noise of the given scale, then L2-normalized again."""
    noisy = E + scale * np.random.default_rng(seed).standard_normal(E.shape).astype(np.float32)
    return noisy / np.linalg.norm(noisy, axis=1, keepdims=True)


def _one_hot(ids, dim=8):
    """One axis vector per id: equal ids give identical rows, every ground cost is exactly 0 or sqrt(2)."""
    return np.eye(dim, dtype=np.float32)[ids]


def test_partial_matching_is_one_to_one_under_the_cutoff():
    """ACC-TRANS-47: 5 planted near-copies pair one to one; the 3 unrelated rows of each side stay unmatched."""
    X = _emb(8, seed=31)
    planted = [6, 1, 4, 0, 3]  # the X rows that have a near-copy in Y
    stacked = np.concatenate([_reworded(X[planted], 0.02, seed=32), _emb(3, seed=33)])
    where = [5, 0, 1, 6, 2, 3, 7, 4]  # Y position -> row of ``stacked``
    Y = stacked[where]

    pairs = d.partial_matching(X, Y)
    assert pairs.shape == (5, 2)
    assert len(set(pairs[:, 0].tolist())) == 5 and len(set(pairs[:, 1].tolist())) == 5
    assert (d.cost_matrix(X, Y)[pairs[:, 0], pairs[:, 1]] < d.MATCH_CUTOFF).all()
    assert pairs.tolist() == sorted([planted[k], where.index(k)] for k in range(5))


def _in_plane(degrees):
    """Unit rows in the plane at the given angles; the ground cost of two rows is ``2 * sin(angle / 2)``."""
    rad = np.deg2rad(degrees)
    return np.column_stack([np.cos(rad), np.sin(rad)]).astype(np.float32)


def test_partial_matching_caps_the_assignment_cost():
    """ACC-TRANS-47: the assignment runs on the cost capped at the cutoff, not on the ground cost.

    Ground costs ``[[0.261, 0.684], [1.351, 1.879]]``: ``X[0]`` and ``Y[0]`` are a pair below the cutoff,
    ``X[1]`` has no counterpart. Capped, the assignment keeps ``(0, 0)`` (0.261 + 0.325 against
    0.325 + 0.325). On the ground cost it returns ``(0, 1), (1, 0)`` (2.035 against 2.140), both at
    or above the cutoff, so no pair would be left.
    """
    X, Y = _in_plane([0, 100]), _in_plane([15, -40])
    assert d.partial_matching(X, Y).tolist() == [[0, 0]]


def test_partial_matching_leaves_out_a_pair_exactly_at_the_cutoff(monkeypatch):
    """ACC-TRANS-47: the pair filter is ``<``, so a pair whose ground cost equals the cutoff stays unmatched.

    The cutoff is set to the ground cost of the second pair, bit for bit: that pair is left out. One
    float32 step above, it is matched.
    """
    X, Y = _in_plane([0, 90]), _in_plane([5, 100])
    cost = d.cost_matrix(X, Y)
    assert (
        cost[0, 0] < cost[1, 1] < cost[1, 0]
    )  # 0.087 and 0.174; the crossed pairs cost 1.35 and 1.53

    monkeypatch.setattr(d, "MATCH_CUTOFF", float(cost[1, 1]))
    assert d.partial_matching(X, Y).tolist() == [[0, 0]]
    monkeypatch.setattr(d, "MATCH_CUTOFF", float(np.nextafter(cost[1, 1], np.float32(1.0))))
    assert d.partial_matching(X, Y).tolist() == [[0, 0], [1, 1]]


def test_matched_order_gap_ignores_a_pure_deletion():
    """ACC-TRANS-49 / DEF-TRANS-10: 2 of 12 statements deleted, order kept - the matched gap is the identity floor.

    The shipped ``opw_gap(A, B)`` reads the deletion as a reorder. The matched gap is the order gap of
    the 10 surviving statements against themselves, held as two arrays the way two documents arrive.
    POT sets the cost diagonal to exactly 0 only when both arguments are one array object, so the
    one-object reading ``opw_gap(B, B)`` differs from it by the float32 rounding of that diagonal
    (at most 7.813e-06 in the 120 deletion cases of E15-H79).
    """
    A = _emb(12, seed=35)
    B = A[[i for i in range(12) if i not in (3, 8)]]
    gap = d.matched_order_gap(A, B)
    assert gap == d.opw_gap(B.copy(), B)
    assert gap == pytest.approx(d.opw_gap(B, B), abs=1e-5)
    assert gap < d.opw_gap(A, B)
    assert d.opw_gap(A, B) > 0.01  # the shipped reading of the same pair is a reorder-sized gap


def test_matched_order_gap_equals_order_gap_when_fully_matched():
    """ACC-TRANS-50: equal counts and every statement matched - exactly the shipped ``opw_gap``.

    The identity pair is the case that tells a call on the unmodified arrays from a call on indexed
    copies: POT sets the cost diagonal to 0 only when both arguments are one array object.
    """
    X = _emb(10, seed=36)
    Y = _reworded(X[np.random.default_rng(37).permutation(10)], 1e-3, seed=38)
    assert len(d.partial_matching(X, Y)) == 10
    assert np.array_equal(d.matched_order_gap(X, Y), d.opw_gap(X, Y))
    assert np.array_equal(d.matched_order_gap(X, X), d.opw_gap(X, X))


def test_matched_order_gap_leaves_out_a_replaced_statement():
    """Equal counts with one statement unmatched: the matched gap is read on the 9 pairs, not on all 10 rows.

    Statement 4 of ``Y`` is replaced by an unrelated one, the order of the rest is kept. The equal
    statement counts alone do not send the pair to the shipped ``opw_gap(X, Y)``, which also reads the
    replaced statement.
    """
    X = _emb(10, seed=43)
    Y = X.copy()
    Y[4] = _emb(1, seed=143)[0]
    kept = [i for i in range(10) if i != 4]
    assert d.partial_matching(X, Y).tolist() == [[i, i] for i in kept]
    gap = d.matched_order_gap(X, Y)
    assert gap == d.opw_gap(X[kept], Y[kept])
    assert gap < 1e-6 < d.opw_gap(X, Y)


def test_matched_order_gap_still_reads_a_reorder_after_a_deletion():
    """ACC-TRANS-51: the survivors of a 2-statement deletion, reversed, read at least 10 times the deleted-only gap."""
    A = _emb(12, seed=35)
    deleted_only = A[[i for i in range(12) if i not in (3, 8)]]
    reordered = deleted_only[::-1]
    gap = d.matched_order_gap(A, reordered)
    assert gap > 10 * d.matched_order_gap(A, deleted_only)
    # the reorder reads as it does between the survivors alone (float32 rounding of the zero costs)
    assert gap == pytest.approx(d.opw_gap(deleted_only, reordered), abs=1e-3)


def test_partial_matching_no_statement_matched():
    """ACC-TRANS-52: two documents from orthogonal subspaces - no pair, no matched gap, n_matched 0."""
    X = np.pad(_emb(5, dim=16, seed=39), ((0, 0), (0, 16)))
    Y = np.pad(_emb(7, dim=16, seed=40), ((0, 0), (16, 0)))
    assert d.partial_matching(X, Y).shape == (0, 2)
    assert d.matched_order_gap(X, Y) is None
    result = d.compute_structural(X, Y)
    assert result.n_matched == 0 and result.matched_order_gap is None


def test_partial_matching_one_statement_against_twelve():
    """ACC-TRANS-53: a 1-statement document gives at most 1 pair; the gap is that of two 1-statement sub-sequences."""
    Y = _emb(12, seed=41)
    X = Y[[7]]
    assert d.partial_matching(X, Y).tolist() == [[0, 7]]
    assert d.matched_order_gap(X, Y) == d.opw_gap(X, X)
    assert len(d.partial_matching(_emb(1, seed=42), Y)) == 0  # an unrelated statement: no pair


def test_partial_matching_is_deterministic_among_equal_cost_pairs():
    """ACC-TRANS-83: the position term decides among equal-cost matchings.

    One-hot rows keep the ties exact on every BLAS: the ground cost between two copies of the
    repeated statement is exactly 0 at every position of the cost matrix.
    """
    A = _one_hot([1, 0, 2, 0, 0, 3])  # statement 0 at positions 1, 3 and 4
    B = A[[0, 1, 3, 4, 5]]  # position 2, another statement, deleted
    pairs = d.partial_matching(A, B)
    assert pairs.tolist() == [
        [0, 0],
        [1, 1],
        [3, 2],
        [4, 3],
        [5, 4],
    ]  # the copies in document order
    assert np.array_equal(d.partial_matching(A, B), pairs)  # two calls, the same pairs

    # a copy of the repeated statement deleted: 3 copies of A against 2 of B, every choice of 2 costs
    # the same. The position term returns the pairs that keep the matched statements of A in the
    # order of B; without it the assignment (scipy 1.17.1) returns the crossing pairs (1, 2), (2, 1)
    A = _one_hot([0, 0, 1, 0, 2, 3])
    B = A[1:]
    assert d.partial_matching(A, B).tolist() == [[0, 0], [2, 1], [3, 2], [4, 3], [5, 4]]


# --- _build_structural_details dict shape and semantics ------------------------------------------


def test_build_structural_details_top_level_keys():
    sa, ea = [f"a{i}" for i in range(5)], _emb(5, seed=11)
    sb, eb = [f"b{i}" for i in range(4)], _emb(4, seed=12)
    diff = _build_structural_details(sa, ea, sb, eb)
    assert set(diff) == {
        "generator",
        "version",
        "smd",
        "order_gap",
        "structure_closeness",
        "structure_closeness_definition",
        "thresholds",
        "definitions",
        "summary",
        "anisotropy",
        "n_statements",
        "n_dropped",
        "drop_rules",
        "sections",
        "statements",
    }
    assert diff["n_statements"] == {"a": 5, "b": 4}
    assert diff["n_dropped"] == {"a": 0, "b": 0}  # no filter active
    assert diff["drop_rules"] == {"min_words": 0, "skip_table_fragments": False}
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
            "section",
            "target_index",
            "target_text",
            "target_section",
            "cost",
            "semantic_target_index",
            "semantic_target_cost",
            "targets_differ",
            "displacement",
            "matched_displacement",
            "displacement_norm",
            "moved",
            "no_counterpart",
        }
        assert st["index"] == i and st["text"] == sa[i]
        assert 0 <= st["target_index"] < len(sb)
        assert st["target_text"] == sb[st["target_index"]]
        if st["no_counterpart"]:
            # an absorber target names no position: displacement is undefined
            assert st["displacement_norm"] is None
        else:
            expected = st["target_index"] / len(sb) - st["index"] / len(sa)
            assert st["displacement_norm"] == pytest.approx(expected, abs=1e-4)
        # two unrelated documents: no statement is matched, so none is moved
        assert st["matched_displacement"] is None and st["moved"] is False
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
        "generator",
        "version",
        "smd",
        "order_gap",
        "structure_closeness",
        "structure_closeness_definition",
        "thresholds",
        "definitions",
        "summary",
        "anisotropy",
        "n_statements",
        "n_dropped",
        "drop_rules",
        "sections",
        "statements",
    }
    assert diff["n_statements"] == {"a": 5, "b": 4}
    assert result.smd == pytest.approx(diff["smd"], abs=1e-5)
    assert result.order_gap == pytest.approx(diff["order_gap"], abs=1e-5)
    settings.reset()


def test_build_structural_details_header_self_describing_and_summary():
    """ACC-MAP-7 + ACC-MAP-8: the structural header stamps its writer, states the closeness formula
    and consolidated thresholds/definitions, and the summary block mirrors the semantic one -
    n_moved, n_no_counterpart and the median displacement_norm recomputed from the records."""
    ea = np.eye(12, dtype=np.float32)
    removed = {2, 5, 8, 11}
    kept = [i for i in range(12) if i not in removed]
    eb = ea[kept]
    sa = [f"a{i}" for i in range(12)]
    sb = [f"b{i}" for i in kept]
    diff = _build_structural_details(sa, ea, sb, eb)

    assert diff["generator"] == "docdistance"
    assert isinstance(diff["version"], str) and diff["version"]
    assert diff["structure_closeness_definition"] == "1 - order_gap/sqrt(2)"
    assert diff["thresholds"] == {
        "changed_cost": pipeline.DIFF_CHANGED_COST,
        "no_counterpart_cost": pipeline.NO_COUNTERPART_COST,
        "moved_tolerance": pipeline.MOVED_TOLERANCE,
        "match_cutoff": 0.325,
    }
    assert set(diff["definitions"]) == {
        "displacement",
        "displacement_norm",
        "matched_displacement",
        "moved",
        "no_counterpart",
        "share_below_changed",
        "kept",
        "removed",
        "added",
    }
    for text in diff["definitions"].values():
        assert isinstance(text, str) and text and "\n" not in text  # one line each
    # ACC-TRANS-20: the moved definition names the rank displacement and its norm
    assert diff["definitions"]["moved"].startswith(
        "matched AND matched_displacement != 0 AND "
        "abs(matched_displacement / n_matched) > moved_tolerance AND NOT no_counterpart"
    )

    summary = diff["summary"]
    recs = diff["statements"]
    assert summary["n_moved"] == sum(r["moved"] for r in recs)
    assert summary["n_no_counterpart"] == sum(r["no_counterpart"] for r in recs) == len(removed)
    norms = [r["displacement_norm"] for r in recs if r["displacement_norm"] is not None]
    assert summary["median_displacement_norm"] == pytest.approx(float(np.median(norms)), abs=1e-4)


def test_build_structural_details_summary_median_null_when_no_counterparts():
    """Every statement orthogonal to every target: no displacement_norm exists, the median is null."""
    ea = np.eye(6, dtype=np.float32)[:3]
    eb = np.eye(6, dtype=np.float32)[3:]
    diff = _build_structural_details(["a0", "a1", "a2"], ea, ["b0", "b1", "b2"], eb)
    assert diff["summary"]["n_no_counterpart"] == 3
    assert diff["summary"]["n_moved"] == 0
    assert diff["summary"]["median_displacement_norm"] is None


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


def test_build_structural_details_records_carry_aligned_pair_cost():
    """ACC-MAP-5: each record's ``cost`` equals cost_matrix[i, align[i]] rounded to 4 places."""
    sa, ea = [f"a{i}" for i in range(6)], _emb(6, seed=41)
    sb, eb = [f"b{i}" for i in range(4)], _emb(4, seed=42)
    diff = _build_structural_details(sa, ea, sb, eb)
    cost = d.cost_matrix(ea, eb)
    align = d.order_alignment(ea, eb)
    for i, st in enumerate(diff["statements"]):
        assert st["cost"] == round(float(cost[i, align[i]]), 4)


def test_build_structural_details_semantic_target_beside_structural_target():
    """ACC-MAP-6: the plan-argmax semantic target sits on the record with ``targets_differ``.

    Duplicated A content makes the exact EMD degenerate: the raw plan's vertex maps a0 -> b2 and
    a2 -> b0 while the crisp (diagonal tie-break) alignment keeps both in place - the disagreement
    that locates moved-and-reworded content, computed from the already-solved plan.
    """
    u = np.eye(8, dtype=np.float32)
    ea = np.stack([u[0], u[1], u[0]])  # a2 duplicates a0's content
    eb = np.stack([u[0], u[1], u[2]])
    sa = [f"a{i}" for i in range(3)]
    sb = [f"b{i}" for i in range(3)]
    diff = _build_structural_details(sa, ea, sb, eb)

    plan = d.transport_plan(ea, eb)
    cost = d.cost_matrix(ea, eb)
    for i, st in enumerate(diff["statements"]):
        k = int(plan[i].argmax())
        assert st["semantic_target_index"] == k
        assert st["semantic_target_cost"] == round(float(cost[i, k]), 4)
        assert st["targets_differ"] == (k != st["target_index"])
    # the degenerate rows disagree, the untied row does not
    by_index = {st["index"]: st for st in diff["statements"]}
    assert by_index[0]["targets_differ"] is True
    assert by_index[2]["targets_differ"] is True
    assert by_index[1]["targets_differ"] is False


def test_shrinkage_does_not_saturate_moved_and_marks_removed_statements():
    """DEF-TRANS-2: a document a third shorter must not flag every in-place statement as moved.

    A is 12 orthogonal statements; B keeps 8 of them in order (every third removed). The raw rank
    displacement shifts with the length change, but the target-relative ``displacement_norm`` of a
    kept statement stays within tolerance, so ``moved`` is quiet. The 4 removed statements align to
    absorbers above ``NO_COUNTERPART_COST``: displacement_norm null, moved False, no_counterpart True.
    """
    ea = np.eye(12, dtype=np.float32)
    removed = {2, 5, 8, 11}
    kept = [i for i in range(12) if i not in removed]
    eb = ea[kept]
    sa = [f"a{i}" for i in range(12)]
    sb = [f"b{i}" for i in kept]
    diff = _build_structural_details(sa, ea, sb, eb)

    # header states the definitions the flags are computed against
    assert diff["definitions"]["displacement_norm"] == "target_index/n_b - index/n_a"
    assert diff["thresholds"]["moved_tolerance"] == pipeline.MOVED_TOLERANCE
    assert diff["thresholds"]["no_counterpart_cost"] == pipeline.NO_COUNTERPART_COST

    st = {rec["index"]: rec for rec in diff["statements"]}
    for i in kept:
        rec = st[i]
        assert rec["no_counterpart"] is False  # a zero-cost twin exists in B
        assert rec["moved"] is False  # uniform shrinkage: still in place on the relative scale
        assert abs(rec["displacement_norm"]) <= 1.0 / 24.0 + 5e-5  # 4-decimal rounding quantum
        assert isinstance(rec["displacement"], int)  # raw rank shift kept for continuity
    for i in removed:
        rec = st[i]
        assert rec["no_counterpart"] is True  # the aligned target is an absorber, not a location
        assert rec["displacement_norm"] is None
        assert rec["moved"] is False


# --- moved read on matched pairs (E15-H80) -------------------------------------------------------


def _anisotropic(n, seed, dim=32, spread=0.1):
    """Unit vectors that share one direction, like raw statement embeddings: distinct rows cost about 0.7 at spread 0.1."""
    x = 1.0 / np.sqrt(dim) + spread * np.random.default_rng(seed).standard_normal((n, dim))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def _at_cost(base, toward, cost):
    """The unit vector at ground cost ``cost`` from ``base``, turned toward the orthogonal unit vector ``toward``."""
    angle = 2.0 * np.arcsin(cost / 2.0)
    return (np.cos(angle) * base + np.sin(angle) * toward).astype(np.float32)


def _moved(diff):
    """Indices of the statements that carry ``moved``."""
    return [st["index"] for st in diff["statements"] if st["moved"]]


def _matched_displacements(diff):
    return [st["matched_displacement"] for st in diff["statements"]]


@pytest.fixture
def structural_details(monkeypatch):
    """``structural_distance_with_details`` on stubbed statement embeddings: the public call, no model load."""
    from docdistance import settings

    settings.reset()
    settings.mark_ready("wmd")
    dd = pipeline.DocDistance()
    monkeypatch.setattr(dd, "_ensure_base", lambda: None)

    def run(sa, ea, sb, eb):
        docs = {"A": (sa, ea), "B": (sb, eb)}
        monkeypatch.setattr(dd, "embed_statements", lambda doc: docs[doc])
        return dd.structural_distance_with_details("A", "B")[1]

    yield run
    settings.reset()


# the worked example of docs/example-map-interpretations.md: statements 2 and 3 exchanged,
# statements 1 and 4 replaced
_WORKED_A = ["pagination", "tokens", "dashboard", "export", "offline", "memory leak"]
_WORKED_B = ["pagination", "dark mode", "export", "dashboard", "push notifications", "memory leak"]


def _worked_example():
    """Stub embeddings of the worked example.

    A replaced statement costs to its leftover target what that document reports: 0.5835 (A1 to B4)
    and 0.444 (A4 to B1), above ``MATCH_CUTOFF`` and below ``NO_COUNTERPART_COST``. Every other pair
    of distinct statements costs sqrt(2).
    """
    u = np.eye(8, dtype=np.float32)
    eb = np.stack(
        [u[0], _at_cost(u[4], u[6], 0.444), u[3], u[2], _at_cost(u[1], u[7], 0.5835), u[5]]
    )
    return u[:6], eb


def test_moved_reads_only_the_exchanged_pair_of_the_worked_example(structural_details):
    """ACC-TRANS-15 / ACC-TRANS-77: both exchanged statements carry ``moved``, the two replaced ones do not.

    A replaced statement has no pair below ``MATCH_CUTOFF``, so it holds no rank among the matched
    statements. Its exact-plan target is the leftover statement of B, a counterpart by
    ``NO_COUNTERPART_COST``, which the 1.1.5 flag read as a move: 4 of 6 flagged (E15-H80).
    """
    ea, eb = _worked_example()
    diff = structural_details(_WORKED_A, ea, _WORKED_B, eb)

    assert _moved(diff) == [2, 3]
    assert diff["summary"]["n_moved"] == 2
    assert _matched_displacements(diff) == [0, None, 1, -1, None, 0]
    # the shipped fields keep their reading: the leftover pairing and its rank shift
    assert [st["displacement"] for st in diff["statements"]] == [0, 3, 1, -1, -3, 0]
    assert [st["cost"] for st in diff["statements"]] == [0.0, 0.5835, 0.0, 0.0, 0.444, 0.0]
    assert not any(st["no_counterpart"] for st in diff["statements"])


def test_moved_quiet_when_one_statement_is_inserted_at_position_0():
    """ACC-TRANS-77: document A of the worked example against itself with one new statement in front - 0 of 6 moved.

    Every statement keeps its rank among the 6 matched statements. The target-relative position
    ``target_index/n_b - index/n_a`` shifts by ``(6 - i) / 42``, above the tolerance on 4 statements:
    the reading of the 1.1.5 flag (E12-H61).
    """
    u = np.eye(8, dtype=np.float32)
    diff = _build_structural_details(
        _WORKED_A, u[:6], ["billing", *_WORKED_A], u[[6, 0, 1, 2, 3, 4, 5]]
    )
    statements = diff["statements"]

    assert [st["target_index"] for st in statements] == [1, 2, 3, 4, 5, 6]
    assert sum(abs(st["displacement_norm"]) > pipeline.MOVED_TOLERANCE for st in statements) == 4
    assert _matched_displacements(diff) == [0] * 6
    assert _moved(diff) == [] and diff["summary"]["n_moved"] == 0


def test_moved_flags_every_statement_of_a_reversal():
    """ACC-TRANS-16: a fully reversed 12-statement document reads 12 of 12 moved."""
    s, e = [f"s{i}" for i in range(12)], _emb(12, seed=61)
    diff = _build_structural_details(s, e, s[::-1], e[::-1])
    assert _matched_displacements(diff) == [11 - 2 * i for i in range(12)]
    assert diff["summary"]["n_moved"] == 12


def test_moved_quiet_on_an_order_preserving_deletion(structural_details):
    """ACC-TRANS-17 / ACC-TRANS-78 / DEF-TRANS-13: 12 statements, 4 removed, order kept - 0 of 12 moved.

    The rows share one direction, like raw statement embeddings: every pair of distinct statements
    costs between ``MATCH_CUTOFF`` and ``NO_COUNTERPART_COST``. A removed statement is therefore
    not matched, and no record reads ``no_counterpart``. The exact plan splits the row of each
    removed statement into two equal parts of 1/24, and the 1.1.5 flag followed the target that
    ``argmax`` returned for that tie: for statement 8 one of the two targets reads moved and the
    other does not. On POT 0.9.6.post1 it flagged 4 of the 4 removed statements of this fixture.
    Read on matched pairs, a removed statement holds no rank and a kept one keeps its rank.
    """
    removed = [2, 5, 8, 11]
    kept = [i for i in range(12) if i not in removed]
    ea = _anisotropic(12, seed=5, spread=0.06)
    distinct = d.cost_matrix(ea, ea)[~np.eye(12, dtype=bool)]
    assert d.MATCH_CUTOFF < distinct.min() and distinct.max() < pipeline.NO_COUNTERPART_COST
    plan = d._aligned_plan(ea, ea[kept])
    tied = [np.flatnonzero(plan[i] > 1e-9) for i in removed]  # the targets that hold mass
    assert [len(targets) for targets in tied] == [2, 2, 2, 2]  # the tie of DEF-TRANS-13
    assert [bool(abs(j / 8 - 8 / 12) > pipeline.MOVED_TOLERANCE) for j in tied[2]] == [False, True]

    diff = structural_details([f"a{i}" for i in range(12)], ea, [f"a{i}" for i in kept], ea[kept])
    statements = diff["statements"]
    assert not any(st["no_counterpart"] for st in statements)
    assert [statements[i]["matched_displacement"] for i in kept] == [0] * 8
    assert [statements[i]["matched_displacement"] for i in removed] == [None] * 4
    assert _moved(diff) == [] and diff["summary"]["n_moved"] == 0


def test_matched_displacement_zero_on_kept_and_null_on_removed_statements():
    """ACC-MAP-60: B is A minus 2 rows - 0 on the 10 kept statements, null on the 2 removed."""
    removed = [3, 8]
    kept = [i for i in range(12) if i not in removed]
    s, e = [f"s{i}" for i in range(12)], _emb(12, seed=35)
    diff = _build_structural_details(s, e, [s[i] for i in kept], e[kept])
    assert _matched_displacements(diff) == [None if i in removed else 0 for i in range(12)]
    json.dumps(diff)  # null and integers: plain JSON values


def test_moved_reads_ranks_inside_the_matched_statements():
    """ACC-TRANS-77: 2 of 12 statements removed and 2 kept ones exchanged - only the exchanged pair is moved.

    Ranks are taken among the 10 matched statements of each document: statement 4 goes from rank 3
    to rank 7 and statement 9 from rank 7 to rank 3; the statements between them keep their rank.
    """
    s, e = [f"s{i}" for i in range(12)], _emb(12, seed=35)
    order = [0, 1, 2, 9, 5, 6, 7, 4, 10, 11]  # rows 3 and 8 removed, rows 4 and 9 exchanged
    diff = _build_structural_details(s, e, [s[i] for i in order], e[order])
    assert _matched_displacements(diff) == [0, 0, 0, None, 4, 0, 0, 0, None, -4, 0, 0]
    assert _moved(diff) == [4, 9] and diff["summary"]["n_moved"] == 2


def test_moved_tolerance_reads_the_rank_step_over_the_matched_count():
    """ACC-TRANS-79: one rank step of exactly 5% of the matched count is not above the tolerance.

    20 matched statements, two neighbours exchanged: ``abs(1 / 20) > 0.05`` is false at every
    position, so no statement is moved. The 1.1.5 form ``target_index/n_b - index/n_a`` exceeds 0.05
    at 7 of the 19 positions through float rounding (E15-H80: 183 of 190 single transpositions equal
    at 20 statements). With 19 matched statements one step is 1/19 and both neighbours are moved;
    one more statement without a match in each document does not change that reading.
    """
    e = _emb(22, seed=62)
    s = [f"s{i}" for i in range(22)]
    for first in range(19):
        order = list(range(20))
        order[first], order[first + 1] = order[first + 1], order[first]
        diff = _build_structural_details(s[:20], e[:20], [s[i] for i in order], e[order])
        assert _matched_displacements(diff)[first : first + 2] == [1, -1]
        # 2 statements hold a rank displacement and none is moved: the summary counts the flag
        assert _moved(diff) == [] and diff["summary"]["n_moved"] == 0

    order = [1, 0, *range(2, 19)]  # 19 statements, the first two exchanged
    diff = _build_structural_details(s[:19], e[:19], [s[i] for i in order], e[order])
    assert _moved(diff) == [0, 1]
    # 20 statements in each document, 19 of them matched: the step is still 1 of 19
    rows_a, rows_b = [*range(19), 20], [*order, 21]
    diff = _build_structural_details(
        [s[i] for i in rows_a], e[rows_a], [s[i] for i in rows_b], e[rows_b]
    )
    assert _matched_displacements(diff) == [1, -1, *[0] * 17, None]
    assert _moved(diff) == [0, 1]


def test_moved_false_on_a_matched_statement_without_counterpart():
    """ACC-TRANS-77: ``NOT no_counterpart`` - the shipped rule also holds for a matched statement.

    Ground costs ``[[1.158, 0.701, 0.300], [0.000, 1.656, 0.900]]``. The matching pairs ``A[0]`` with
    ``B[2]`` (0.300) and ``A[1]`` with ``B[0]`` (0.000): the pairs cross, so both statements hold a
    rank displacement. The exact plan puts 2/6 of the mass of ``A[0]`` on ``B[1]`` (0.701, above
    ``NO_COUNTERPART_COST``) and 1/6 on ``B[2]``, so its record reads ``no_counterpart``.
    """
    A, B = _in_plane([17.25, -53.5]), _in_plane([-53.5, 58.3, 0.0])
    diff = _build_structural_details(["a0", "a1"], A, ["b0", "b1", "b2"], B)
    first, second = diff["statements"]
    assert _matched_displacements(diff) == [1, -1]
    assert first["no_counterpart"] is True and first["moved"] is False
    assert second["no_counterpart"] is False and second["moved"] is True


def test_moved_needs_a_rank_displacement_at_any_tolerance(monkeypatch):
    """ACC-TRANS-77: ``matched_displacement != 0`` is a clause of its own.

    At the shipped tolerance the norm clause implies it. With the tolerance below 0 the norm clause
    holds for every matched statement, and a document against itself still reads 0 moved.
    """
    monkeypatch.setattr(pipeline, "MOVED_TOLERANCE", -1.0)
    s, e = [f"s{i}" for i in range(6)], _emb(6, seed=63)
    diff = _build_structural_details(s, e, s, e)
    assert _matched_displacements(diff) == [0] * 6
    assert _moved(diff) == []
    order = [1, 0, 2, 3, 4, 5]
    assert _moved(_build_structural_details(s, e, [s[i] for i in order], e[order])) == [0, 1]


def test_moved_equals_the_conjunction_on_fully_matched_pairs():
    """ACC-TRANS-79: 50 seeded permutations of a 12-statement document - ``moved`` equals the E12-H61 conjunction.

    The conjunction ``displacement != 0 and abs(displacement_norm) > MOVED_TOLERANCE`` is computed
    from the fields of the same record, statement for statement. One rank step is 1/12 here, above
    the tolerance; the 5% step of 20 and 40 statements is the stated exception.
    """
    rng = np.random.default_rng(64)
    s, e = [f"s{i}" for i in range(12)], _emb(12, seed=65)
    flagged = in_place = 0
    for draw in range(50):
        perm = np.arange(12)
        for _ in range(
            1 + draw % 4
        ):  # 1 to 4 exchanges: moved and in-place statements in one pair
            i, j = rng.choice(12, size=2, replace=False)
            perm[[i, j]] = perm[[j, i]]
        statements = _build_structural_details(s, e, [s[k] for k in perm], e[perm])["statements"]
        assert [st["matched_displacement"] for st in statements] == [
            st["displacement"] for st in statements
        ]  # fully matched: the ranks over the matched pairs are the ranks over all statements
        conjunction = [
            st["displacement"] != 0 and abs(st["displacement_norm"]) > pipeline.MOVED_TOLERANCE
            for st in statements
        ]
        assert [st["moved"] for st in statements] == conjunction
        flagged += sum(conjunction)
        in_place += len(conjunction) - sum(conjunction)
    assert flagged > 0 and in_place > 0


def test_moved_keeps_every_true_relocation_of_the_reorder_pool():
    """ACC-TRANS-18: over the 1680 permutations of the structure fixture no relocated statement loses its flag.

    Each base document of the pool is a set of distinct synthetic unit vectors, permuted as the pool
    records. ``lost`` counts the statements that changed position and that the 1.1.5 definition
    (``abs(target_index/n_b - index/n_a) > MOVED_TOLERANCE``, not ``no_counterpart``) flags while the
    record does not. E12-H61 read 13675 relocated statements, all of them flagged.
    """
    pool = json.loads(_REORDER_POOL.read_text(encoding="utf-8"))
    permutations = relocated = lost = false_fires = 0
    for seed, entries in enumerate(pool.values()):
        n = len(entries[0]["perm"])
        s, e = [f"s{i}" for i in range(n)], _emb(n, seed=70 + seed)
        for entry in entries:
            if entry["k"] == 0:  # the document in its own order
                continue
            perm = np.asarray(entry["perm"])
            statements = _build_structural_details(s, e, [s[j] for j in perm], e[perm])[
                "statements"
            ]
            moved = np.array([st["moved"] for st in statements])
            shipped = np.array(
                [
                    not st["no_counterpart"]
                    and abs(st["target_index"] / n - st["index"] / n) > pipeline.MOVED_TOLERANCE
                    for st in statements
                ]
            )
            truth = np.argsort(perm) != np.arange(n)  # the statement changed its position
            permutations += 1
            relocated += int(truth.sum())
            lost += int((shipped & truth & ~moved).sum())
            false_fires += int((moved & ~truth).sum())
    assert (permutations, relocated) == (1680, 13675)
    assert lost == 0 and false_fires == 0


def test_structural_match_fields_read_the_embeddings_as_passed_in():
    """ACC-TRANS-76: ``matched_displacement`` does not change with ``anisotropy``.

    The fixture shares one direction, like raw statement embeddings: each reworded statement costs
    below ``MATCH_CUTOFF`` to its original on the embeddings as passed in and above it after the
    pre-pass, where a matching returns no pair. ``no_counterpart`` stays a reading of the pre-pass
    geometry, like ``cost``, and ``moved`` follows it.
    """
    A = _anisotropic(8, seed=43)
    order = np.random.default_rng(45).permutation(8)
    B = _reworded(A, 0.04, seed=44)[order]
    fixed = d.all_but_the_top({"a": A, "b": B}, k=1)
    assert len(d.partial_matching(fixed["a"], fixed["b"])) == 0  # the geometries differ here

    sa, sb = [f"a{i}" for i in range(8)], [f"b{i}" for i in range(8)]
    off = _build_structural_details(sa, A, sb, B)
    on = _build_structural_details(sa, A, sb, B, anisotropy=True)
    expected = (np.argsort(order) - np.arange(8)).tolist()  # every statement changed its rank
    assert 0 not in expected
    assert _matched_displacements(off) == _matched_displacements(on) == expected
    assert _moved(off) == list(range(8))
    assert any(st["no_counterpart"] for st in on["statements"])
    assert _moved(on) == [st["index"] for st in on["statements"] if not st["no_counterpart"]]
