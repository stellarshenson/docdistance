"""Pure-logic tests for the pipeline's statement-map builders.

No models load - ``_build_source_map`` and ``_build_semantic_details`` take statement texts and
pre-computed embeddings, so the map shape and content are exercised on synthetic arrays in the
lightweight uv ``.venv``.
"""

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

from loguru import logger
import numpy as np
import pytest

from docdistance import pipeline, settings
from docdistance.pipeline import _build_semantic_details, _build_source_map


def _emb(n: int, dim: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_source_map_structure_and_top_k():
    sa, ea = [f"a{i}" for i in range(4)], _emb(4, seed=1)
    sb, eb = [f"b{i}" for i in range(3)], _emb(3, seed=2)
    ss, es = [f"s{i}" for i in range(6)], _emb(6, seed=3)
    m = _build_source_map(sa, ea, sb, eb, ss, es, anisotropy=False, top_k=2)

    assert m["n_statements"] == {"a": 4, "b": 3, "source": 6}
    assert len(m["a"]) == 4 and len(m["b"]) == 3
    first = m["a"][0]
    assert first["index"] == 0 and first["text"] == "a0"
    assert len(first["matches"]) == 2  # top_k
    # matches are ordered by descending weight, indices are valid source rows, text matches index
    weights = [match["weight"] for match in first["matches"]]
    assert weights == sorted(weights, reverse=True)
    for match in first["matches"]:
        assert 0 <= match["source_index"] < 6
        assert match["source_text"] == ss[match["source_index"]]


def test_source_map_top_k_clamped_to_source_size():
    sa, ea = ["a0"], _emb(1, seed=4)
    sb, eb = ["b0"], _emb(1, seed=5)
    ss, es = ["s0", "s1"], _emb(2, seed=6)
    m = _build_source_map(sa, ea, sb, eb, ss, es, anisotropy=False, top_k=10)
    assert len(m["a"][0]["matches"]) == 2  # clamped to n_source, not 10


def test_transport_map_structure_and_weights():
    sa, ea = [f"a{i}" for i in range(5)], _emb(5, seed=11)
    sb, eb = [f"b{i}" for i in range(4)], _emb(4, seed=12)
    m = _build_semantic_details(sa, ea, sb, eb, anisotropy=False)

    assert m["n_statements"] == {"a": 5, "b": 4}
    assert m["smd"] >= 0.0 and m["anisotropy"] is False
    assert len(m["flows"]) == 5  # one group per A statement
    first = m["flows"][0]
    assert first["index"] == 0 and first["text"] == "a0"
    assert first["matches"]  # every statement sends its mass somewhere
    assert isinstance(first["changed"], bool)  # per-flow content-edit flag
    weights = [mt["weight"] for mt in first["matches"]]
    assert weights == sorted(weights, reverse=True)  # descending weight
    assert sum(weights) == pytest.approx(1.0, abs=1e-3)  # row-normalized fractions
    for mt in first["matches"]:
        assert 0 <= mt["target_index"] < 4
        assert mt["target_text"] == sb[mt["target_index"]]
        assert mt["cost"] >= 0.0  # ground distance of the match
    for flow in m["flows"]:
        assert isinstance(flow["changed"], bool)


def test_transport_map_anisotropy_flag_recorded():
    sa, ea = [f"a{i}" for i in range(4)], _emb(4, seed=13)
    sb, eb = [f"b{i}" for i in range(4)], _emb(4, seed=14)
    m = _build_semantic_details(sa, ea, sb, eb, anisotropy=True)
    assert m["anisotropy"] is True
    assert len(m["flows"]) == 4


def test_transport_map_identical_maps_each_to_itself():
    """Identical documents: each statement maps to its twin at full weight and zero cost, changed False."""
    s, e = [f"s{i}" for i in range(5)], _emb(5, seed=21)
    m = _build_semantic_details(s, e, s, e, anisotropy=False)
    assert m["smd"] == pytest.approx(0.0, abs=1e-5)
    for flow in m["flows"]:
        best = flow["matches"][0]
        assert best["target_index"] == flow["index"]  # statement -> itself
        assert best["weight"] == pytest.approx(1.0, abs=1e-3)
        assert best["cost"] == pytest.approx(0.0, abs=1e-4)
        assert flow["changed"] is False  # a zero-cost twin match is not a content edit


def _basis(dim: int = 8) -> np.ndarray:
    """Orthogonal unit vectors - pairwise ground cost sqrt(2), identical rows cost 0."""
    return np.eye(dim, dtype=np.float32)


def test_changed_reads_the_cheapest_row_cost_not_the_heaviest_match():
    """DEF-TRANS-3: mass splitting onto a far absorber must not flag a statement with a cheap match.

    A0 and A1 both sit at B0 (A1 slightly perturbed); B1/B2 are orthogonal absorbers. The plan gives
    B0's capacity to A0, so A1's mass flows only to the absorbers at cost sqrt(2) - but its cost row
    holds a near-zero match, so ``changed`` (and ``no_counterpart``) stay False.
    """
    u = _basis()
    a1 = u[0] + 0.05 * u[3]
    ea = np.stack([u[0], a1 / np.linalg.norm(a1)])
    eb = np.stack([u[0], u[1], u[2]])
    m = _build_semantic_details(["a0", "a1"], ea, ["b0", "b1", "b2"], eb, anisotropy=False)

    split = m["flows"][1]
    # the old reading: the heaviest match is a far absorber
    assert split["matches"][0]["cost"] > pipeline.DIFF_CHANGED_COST
    # the new reading: the cheapest row cost (~0.05 to B0) clears both flags
    assert split["changed"] is False
    assert split["no_counterpart"] is False
    assert m["flows"][0]["changed"] is False


def test_no_counterpart_flags_statements_with_no_cheap_match():
    """DEF-TRANS-1: a statement orthogonal to every target is annotated, the plan and smd untouched."""
    u = _basis()
    ea = np.stack([u[0], u[1]])  # a1 has no counterpart in B
    eb = np.stack([u[0], u[2]])
    m = _build_semantic_details(["a0", "a1"], ea, ["b0", "b1"], eb, anisotropy=False)

    assert (
        m["thresholds"]["no_counterpart_cost"] == pipeline.NO_COUNTERPART_COST
    )  # threshold stated in the header
    assert m["flows"][0]["no_counterpart"] is False  # zero-cost twin
    assert m["flows"][1]["no_counterpart"] is True  # cheapest match at sqrt(2)
    assert m["flows"][1]["changed"] is True
    # the annotation does not perturb the exact balanced OT: half the mass moves at cost sqrt(2)
    assert m["smd"] == pytest.approx(np.sqrt(2.0) / 2.0, abs=1e-4)


def test_semantic_details_targets_view_inverts_flows():
    """ACC-MAP-4: ``targets`` lists every B statement with its best incoming source, the full
    cost-column minimum and the summed row-normalised inflow - matching an inversion of ``flows``."""
    from docdistance import distance as d

    sa, ea = [f"a{i}" for i in range(5)], _emb(5, seed=31)
    sb, eb = [f"b{i}" for i in range(3)], _emb(3, seed=32)
    m = _build_semantic_details(sa, ea, sb, eb, anisotropy=False)

    assert len(m["targets"]) == 3  # one record per B statement
    cost = d.cost_matrix(ea, eb)
    # invert the flows: per target, the incoming (weight, cost) entries
    incoming: dict[int, list[tuple[float, float]]] = {j: [] for j in range(3)}
    for flow in m["flows"]:
        for mt in flow["matches"]:
            incoming[mt["target_index"]].append((mt["weight"], mt["cost"]))
    for j, tgt in enumerate(m["targets"]):
        assert tgt["index"] == j and tgt["text"] == sb[j]
        assert tgt["section"] == ""  # no sections_b passed
        assert tgt["best_source_index"] == int(cost[:, j].argmin())
        assert tgt["min_incoming_cost"] == pytest.approx(float(cost[:, j].min()), abs=1e-3)
        # the full column minimum is never above any plan-supported incoming cost
        assert tgt["min_incoming_cost"] <= min(c for _, c in incoming[j]) + 1e-3
        assert tgt["total_incoming_weight"] == pytest.approx(
            sum(w for w, _ in incoming[j]), abs=1e-2
        )


def test_semantic_details_targets_identical_docs_and_sections_b():
    """Identical documents: each target's best source is its twin at zero cost, inflow 1."""
    s, e = [f"s{i}" for i in range(4)], _emb(4, seed=33)
    m = _build_semantic_details(s, e, s, e, anisotropy=False, sections_b=["X", "X", "Y", "Y"])
    assert [t["section"] for t in m["targets"]] == ["X", "X", "Y", "Y"]
    for j, tgt in enumerate(m["targets"]):
        assert tgt["best_source_index"] == j
        assert tgt["min_incoming_cost"] == pytest.approx(0.0, abs=1e-4)
        assert tgt["total_incoming_weight"] == pytest.approx(1.0, abs=1e-3)


def test_semantic_details_header_self_describing():
    """ACC-MAP-7: the header stamps its writer, restates the closeness formula and consolidates
    the thresholds and flag definitions in force."""
    from docdistance import distance as d

    sa, ea = [f"a{i}" for i in range(5)], _emb(5, seed=41)
    sb, eb = [f"b{i}" for i in range(4)], _emb(4, seed=42)
    m = _build_semantic_details(sa, ea, sb, eb, anisotropy=False)

    assert m["generator"] == "docdistance"
    assert isinstance(m["version"], str) and m["version"]
    assert m["closeness_definition"] == "1 - smd/sqrt(2)"
    assert m["closeness"] == pytest.approx(d.closeness(m["smd"]), abs=1e-5)
    assert m["thresholds"] == {
        "changed_cost": pipeline.DIFF_CHANGED_COST,
        "no_counterpart_cost": pipeline.NO_COUNTERPART_COST,
        "match_cutoff": 0.325,
    }
    assert set(m["definitions"]) == {
        "changed",
        "no_counterpart",
        "identical",
        "share_below_changed",
        "kept",
        "removed",
        "added",
    }
    for text in m["definitions"].values():
        assert isinstance(text, str) and text and "\n" not in text  # one line each


def test_semantic_details_summary_matches_recomputed_aggregates():
    """ACC-MAP-8: the header cost-band counts, changed/no-counterpart counts and top_changed
    ranking equal the same aggregates recomputed from the cost rows behind the flows."""
    from docdistance import distance as d

    sa, ea = [f"a{i}" for i in range(12)], _emb(12, seed=51)
    sb, eb = [f"b{i}" for i in range(9)], _emb(9, seed=52)
    m = _build_semantic_details(sa, ea, sb, eb, anisotropy=False)

    cost = d.cost_matrix(ea, eb)
    min_costs = [float(cost[i].min()) for i in range(12)]
    bands = {"0-0.1": 0, "0.1-0.35": 0, "0.35-0.6": 0, ">0.6": 0}
    for c in min_costs:
        if c <= 0.1:
            bands["0-0.1"] += 1
        elif c <= 0.35:
            bands["0.1-0.35"] += 1
        elif c <= 0.6:
            bands["0.35-0.6"] += 1
        else:
            bands[">0.6"] += 1
    summary = m["summary"]
    assert summary["cost_bands"] == bands
    assert sum(summary["cost_bands"].values()) == 12  # every statement lands in one band
    assert summary["n_changed"] == sum(f["changed"] for f in m["flows"])
    assert summary["n_no_counterpart"] == sum(f["no_counterpart"] for f in m["flows"])
    expected_top = sorted(range(12), key=lambda i: (-min_costs[i], i))[:10]
    assert summary["top_changed"] == expected_top
    assert len(summary["top_changed"]) == 10  # capped at top-10


def test_changed_cost_parameter_reflags_and_is_emitted():
    """ACC-TRANS-10: changed_cost=0.1 reflags a small edit that the default cutoff clears,
    lands in the header, and leaves the plan and smd untouched."""
    u = _basis()
    a1 = u[1] + 0.2 * u[3]  # a small edit of b1: cost ~0.2, between 0.1 and DIFF_CHANGED_COST
    ea = np.stack([u[0], a1 / np.linalg.norm(a1)])
    eb = np.stack([u[0], u[1]])
    sa, sb = ["a0", "a1"], ["b0", "b1"]

    default = _build_semantic_details(sa, ea, sb, eb, anisotropy=False)
    strict = _build_semantic_details(sa, ea, sb, eb, anisotropy=False, changed_cost=0.1)

    assert default["thresholds"]["changed_cost"] == pipeline.DIFF_CHANGED_COST
    assert strict["thresholds"]["changed_cost"] == 0.1
    # the small edit sits above 0.1 but below the default cutoff
    assert default["flows"][1]["changed"] is False
    assert strict["flows"][1]["changed"] is True
    assert strict["flows"][0]["changed"] is False  # a zero-cost twin clears any cutoff
    # summary counts follow the cutoff in force
    assert default["summary"]["n_changed"] == 0
    assert strict["summary"]["n_changed"] == 1
    # annotation only: the exact balanced OT is unchanged
    assert strict["smd"] == default["smd"]
    assert strict["flows"][1]["no_counterpart"] is False  # its own cutoff, not changed_cost


def test_identical_flag_normalised_text_equality_with_cheapest_target():
    """ACC-TRANS-10: ``identical`` compares casefolded, whitespace-collapsed text with the
    cheapest-cost target, so a pure formatting difference reads identical and a reworded twin
    does not."""
    u = _basis()
    ea = np.stack([u[0], u[1]])
    eb = np.stack([u[0], u[1]])
    m = _build_semantic_details(
        ["Hello  World", "a reworded statement"],
        ea,
        ["hello world", "the original statement"],
        eb,
        anisotropy=False,
    )
    assert m["flows"][0]["identical"] is True  # case + whitespace normalise away
    assert m["flows"][1]["identical"] is False  # same embedding row, different text
    for flow in m["flows"]:
        assert isinstance(flow["identical"], bool)


# --- the one-to-one matching on the semantic detail file (E13-H71) -------------------------------


def _anisotropic(n: int, seed: int, dim: int = 32, spread: float = 0.1) -> np.ndarray:
    """Unit vectors that share one direction, like raw statement embeddings: distinct rows cost about 0.7."""
    x = 1.0 / np.sqrt(dim) + spread * np.random.default_rng(seed).standard_normal((n, dim))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def _reworded(E: np.ndarray, scale: float, seed: int) -> np.ndarray:
    """Each row of ``E`` moved by gaussian noise of the given scale, then L2-normalized again."""
    noisy = E + scale * np.random.default_rng(seed).standard_normal(E.shape).astype(np.float32)
    return noisy / np.linalg.norm(noisy, axis=1, keepdims=True)


def _at_cost(base: np.ndarray, toward: np.ndarray, cost: float) -> np.ndarray:
    """The unit vector at ground cost ``cost`` from ``base``, turned toward the orthogonal unit vector ``toward``."""
    angle = 2.0 * np.arcsin(cost / 2.0)
    return (np.cos(angle) * base + np.sin(angle) * toward).astype(np.float32)


def _labels(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(n)]


def _flow_matches(m: dict) -> list[tuple]:
    return [(flow["match_index"], flow["match_status"]) for flow in m["flows"]]


def _target_matches(m: dict) -> list[tuple]:
    return [(tgt["match_source_index"], tgt["match_status"]) for tgt in m["targets"]]


def _match_counts(m: dict) -> tuple[int, int, int]:
    summary = m["summary"]
    return summary["n_kept"], summary["n_removed"], summary["n_added"]


def test_semantic_flows_carry_the_match():
    """ACC-MAP-55: A is B in another order plus one unrelated statement - that flow reads removed, every other flow kept."""
    eb = _emb(5, seed=61)
    ea = np.concatenate([eb[[3, 0]], _emb(1, seed=62), eb[[4, 1, 2]]])  # A[2] is the unrelated one
    m = _build_semantic_details(_labels("a", 6), ea, _labels("b", 5), eb, anisotropy=False)

    assert _flow_matches(m) == [
        (3, "kept"),
        (0, "kept"),
        (None, "removed"),
        (4, "kept"),
        (1, "kept"),
        (2, "kept"),
    ]
    for flow in m["flows"]:
        assert set(flow) == {
            "index",
            "text",
            "matches",
            "section",
            "changed",
            "no_counterpart",
            "identical",
            "match_index",
            "match_status",
        }


def test_semantic_targets_carry_added():
    """ACC-MAP-56: B is A in another order plus one unrelated statement - that target reads added, every other target kept."""
    ea = _emb(5, seed=63)
    eb = np.concatenate([ea[[2, 4, 0]], _emb(1, seed=64), ea[[3, 1]]])  # B[3] is the unrelated one
    m = _build_semantic_details(_labels("a", 5), ea, _labels("b", 6), eb, anisotropy=False)

    assert _target_matches(m) == [
        (2, "kept"),
        (4, "kept"),
        (0, "kept"),
        (None, "added"),
        (3, "kept"),
        (1, "kept"),
    ]
    for tgt in m["targets"]:
        assert set(tgt) == {
            "index",
            "text",
            "section",
            "best_source_index",
            "min_incoming_cost",
            "total_incoming_weight",
            "match_status",
            "match_source_index",
        }


def test_semantic_summary_counts_kept_removed_added():
    """ACC-MAP-57: 2 statements of A deleted and 1 statement inserted in B - 6 kept, 2 removed, 1 added."""
    ea = _emb(8, seed=65)
    eb = np.concatenate([ea[[0, 1, 3, 4]], _emb(1, seed=66), ea[[6, 7]]])  # rows 2 and 5 deleted
    m = _build_semantic_details(_labels("a", 8), ea, _labels("b", 7), eb, anisotropy=False)

    assert set(m["summary"]) == {
        "cost_bands",
        "n_changed",
        "n_no_counterpart",
        "top_changed",
        "n_kept",
        "n_removed",
        "n_added",
    }
    n_kept, n_removed, n_added = _match_counts(m)
    assert (n_kept, n_removed, n_added) == (6, 2, 1)
    assert n_kept + n_removed == m["n_statements"]["a"] == 8
    assert n_kept + n_added == m["n_statements"]["b"] == 7
    # the counts are those of the records
    assert [f["index"] for f in m["flows"] if f["match_status"] == "removed"] == [2, 5]
    assert [t["index"] for t in m["targets"] if t["match_status"] == "added"] == [4]


def test_semantic_match_is_one_to_one():
    """A statement of B is in at most one pair: of two statements of A near ``B[0]`` one is kept.

    Both cost rows hold a target below ``MATCH_CUTOFF`` (0.0 and about 0.05 to ``B[0]``), so neither
    flow reads ``changed``; the match is not the cheapest target of each row.
    """
    u = _basis()
    a1 = u[0] + 0.05 * u[3]
    ea = np.stack([u[0], a1 / np.linalg.norm(a1)])
    eb = np.stack([u[0], u[1], u[2]])
    m = _build_semantic_details(["a0", "a1"], ea, ["b0", "b1", "b2"], eb, anisotropy=False)

    assert [flow["changed"] for flow in m["flows"]] == [False, False]
    assert _flow_matches(m) == [(0, "kept"), (None, "removed")]
    assert _target_matches(m) == [(0, "kept"), (None, "added"), (None, "added")]
    assert _match_counts(m) == (1, 1, 2)


def test_semantic_target_match_is_the_paired_statement_not_the_cheapest_source():
    """``match_source_index`` names the paired statement, which can differ from ``best_source_index``.

    ``A[1]`` is the cheapest source of both targets (0.05 to ``B[0]``, 0.01 to ``B[1]``) and is paired
    with ``B[1]``; ``B[0]`` is paired with ``A[0]`` at 0.10.
    """
    u = _basis()
    ea = np.stack([_at_cost(u[0], -u[1], 0.15), u[0]])
    eb = np.stack([_at_cost(u[0], -u[1], 0.05), _at_cost(u[0], u[1], 0.01)])
    m = _build_semantic_details(["a0", "a1"], ea, ["b0", "b1"], eb, anisotropy=False)

    assert [tgt["best_source_index"] for tgt in m["targets"]] == [1, 1]
    assert _target_matches(m) == [(0, "kept"), (1, "kept")]
    assert _flow_matches(m) == [(0, "kept"), (1, "kept")]


def test_semantic_details_identical_documents_all_kept():
    """ACC-MAP-61: a document against itself - every flow kept, nothing removed, nothing added."""
    s, e = _labels("s", 7), _emb(7, seed=67)
    m = _build_semantic_details(s, e, s, e, anisotropy=False)
    assert _flow_matches(m) == [(i, "kept") for i in range(7)]
    assert _target_matches(m) == [(j, "kept") for j in range(7)]
    assert _match_counts(m) == (7, 0, 0)


def test_semantic_details_unrelated_documents_nothing_kept():
    """ACC-MAP-62: two documents from orthogonal subspaces - every flow removed, every target added."""
    ea = np.pad(_emb(5, dim=16, seed=68), ((0, 0), (0, 16)))
    eb = np.pad(_emb(7, dim=16, seed=69), ((0, 0), (16, 0)))
    m = _build_semantic_details(_labels("a", 5), ea, _labels("b", 7), eb, anisotropy=False)
    assert _match_counts(m) == (0, 5, 7)
    assert _flow_matches(m) == [(None, "removed")] * 5
    assert _target_matches(m) == [(None, "added")] * 7


def test_semantic_match_fields_read_the_embeddings_as_passed_in():
    """ACC-TRANS-76: the match fields do not change with ``anisotropy``.

    The fixture shares one direction, like raw statement embeddings: 6 statements of A have a
    reworded copy in B (cost below ``MATCH_CUTOFF`` on the embeddings as passed in), 2 have none and
    B holds 1 statement of its own. After the pre-pass a matching returns no pair.
    """
    from docdistance import distance as d

    ea = _anisotropic(8, seed=43)
    order = np.random.default_rng(45).permutation(8)[[0, 1, 2, 4, 5, 7]]
    eb = np.concatenate([_reworded(ea, 0.04, seed=44)[order], _anisotropic(1, seed=99)])
    fixed = d.all_but_the_top({"a": ea, "b": eb}, k=1)
    assert len(d.partial_matching(fixed["a"], fixed["b"])) == 0  # the geometries differ here

    off = _build_semantic_details(_labels("a", 8), ea, _labels("b", 7), eb, anisotropy=False)
    on = _build_semantic_details(_labels("a", 8), ea, _labels("b", 7), eb, anisotropy=True)
    assert on["smd"] != off["smd"]  # the distance is read in the pre-pass geometry
    assert _target_matches(off) == [(int(i), "kept") for i in order] + [(None, "added")]
    assert _flow_matches(on) == _flow_matches(off)
    assert _target_matches(on) == _target_matches(off)
    assert _match_counts(on) == _match_counts(off) == (6, 2, 1)


# --- the nearest source statement on the source map (E15-H78) ------------------------------------


def _support(records: list[dict]) -> list[tuple]:
    return [
        (r["nearest_source_index"], r["nearest_source_cost"], r["unsupported"]) for r in records
    ]


def test_source_map_unsupported_reads_the_source_support_cutoff():
    """ACC-MAP-73: ``unsupported`` is ``nearest_source_cost > SOURCE_SUPPORT_CUTOFF``.

    Two statements at ground cost 0.335 and 0.36 from their nearest source statement: the first is
    above ``MATCH_CUTOFF`` (0.325) and supported, the second unsupported. The call runs the default
    pre-pass, so both costs are those of the embeddings as passed in.
    """
    u = _basis()
    ea = np.stack([_at_cost(u[0], u[5], 0.335), _at_cost(u[1], u[6], 0.36)])
    m = _build_source_map(["a0", "a1"], ea, ["b0"], u[[2]], ["s0", "s1", "s2"], u[:3])

    assert m["anisotropy"] is True
    assert m["source_support_cutoff"] == 0.34625
    assert _support(m["a"]) == [(0, 0.335, False), (1, 0.36, True)]
    assert _support(m["b"]) == [(2, 0.0, False)]
    assert m["n_unsupported"] == {"a": 1, "b": 0}


def test_source_map_support_fields_are_read_before_the_pre_pass():
    """ACC-MAP-73: the nearest source statement is read on the embeddings as passed in.

    The fixture shares one direction, like raw statement embeddings: 3 reworded source statements
    and 2 statements that reword no source statement. After the pre-pass all 5 cost more than
    ``SOURCE_SUPPORT_CUTOFF``. The top-k ``matches`` stay a reading of the pre-pass geometry.
    """
    from docdistance import distance as d

    es = _anisotropic(12, seed=53)
    ea = np.concatenate([_reworded(es[[1, 5, 9]], 0.04, seed=55), _anisotropic(2, seed=54)])
    eb = es[[0, 2, 4]]
    fixed = d.all_but_the_top({"a": ea, "b": eb, "s": es}, k=1)
    assert d.unsupported_count(fixed["a"], fixed["s"]) == 5  # the geometries differ here

    args = (_labels("a", 5), ea, _labels("b", 3), eb, _labels("s", 12), es)
    on = _build_source_map(*args)  # the pre-pass is on by default
    off = _build_source_map(*args, anisotropy=False)
    assert _support(on["a"]) == _support(off["a"]) and _support(on["b"]) == _support(off["b"])
    assert [r["nearest_source_index"] for r in on["a"]][:3] == [1, 5, 9]
    assert [r["unsupported"] for r in on["a"]] == [False, False, False, True, True]
    assert on["n_unsupported"] == off["n_unsupported"] == {"a": 2, "b": 0}
    # every key the map carried before keeps its reading: the weights of the pre-pass geometry
    align = d.coverage_alignment(fixed["a"], fixed["s"])
    for i, record in enumerate(on["a"]):
        top = np.argsort(align[i])[::-1][:3]
        assert [mt["source_index"] for mt in record["matches"]] == top.tolist()
        assert [mt["weight"] for mt in record["matches"]] == [
            round(float(align[i, j]), 4) for j in top
        ]
    assert on["a"][0]["matches"] != off["a"][0]["matches"]


def test_distance_requires_init(monkeypatch, tmp_path):
    """The readiness gate raises before any model load when the mode was never init'd."""
    settings.reset()
    monkeypatch.setenv("DOCDISTANCE_HOME", str(tmp_path))  # empty home, no docdistance.json
    dd = pipeline.DocDistance()  # lazy: constructs without loading models
    with pytest.raises(settings.NotInitializedError):
        dd.semantic_distance("hello world", "goodbye world")
    settings.reset()


class _FakeReranker:
    """score_grid: statement 0 prefers source 0, the grid is constant per doc for a stable premise."""

    def score_grid(self, docs, sources):
        return np.array([[0.9, 0.1, 0.2] for _ in docs], dtype=np.float32)


class _FakeNLI:
    def __init__(self):
        self.premises = None

    def entail(self, premises, hypotheses):
        self.premises = list(premises)
        return np.full(len(premises), 0.5, dtype=np.float32)


def test_grounding_fuses_topk_source_into_the_premise():
    dd = pipeline.DocDistance()
    dd._reranker, dd._nli = _FakeReranker(), _FakeNLI()
    R, ent = dd._grounding(["a statement"], ["s0", "s1", "s2"])
    assert R.shape == (1, 3) and ent.shape == (1,)
    # argsort-desc of [0.9, 0.1, 0.2] is [0, 2, 1] -> top-3 premise "s0 s2 s1"
    assert dd._nli.premises == ["s0 s2 s1"]


def test_distance_wrt_source_wires_grounding_into_the_result(monkeypatch, tmp_path):
    settings.reset()
    settings.mark_ready("wmd-wrt-source")
    docs = {
        "A": (["a0", "a1"], _emb(2, seed=1)),
        "B": (["b0"], _emb(1, seed=2)),
        "S": (["s0", "s1", "s2"], _emb(3, seed=3)),
    }
    dd = pipeline.DocDistance()
    monkeypatch.setattr(dd, "_ensure_grounding", lambda: None)
    monkeypatch.setattr(dd, "embed_statements", lambda doc: docs[doc])
    dd._reranker, dd._nli = _FakeReranker(), _FakeNLI()

    r = dd.distance_wrt_source("A", "B", "S")
    assert r.grd_a is not None and r.grd_b is not None and r.d_grd is not None
    assert r.d_grd == pytest.approx(abs(r.grd_a - r.grd_b))
    settings.reset()


def test_source_map_marks_the_statement_foreign_to_the_source(monkeypatch):
    """ACC-MAP-73: document A holds 2 copies of source statements and 1 statement foreign to the source.

    The source occupies the first 16 of 32 dimensions and the foreign statement the other 16, so it
    costs sqrt(2) to every source statement. The header counts equal ``n_unsupported_a`` /
    ``n_unsupported_b`` of the result the same call returns.
    """
    settings.reset()
    settings.mark_ready("wmd-wrt-source")
    es = np.pad(_emb(12, dim=16, seed=51), ((0, 0), (0, 16)))
    foreign = np.pad(_emb(1, dim=16, seed=52), ((0, 0), (16, 0)))
    docs = {
        "A": (["a0", "a1", "a2"], np.concatenate([es[[4]], foreign, es[[9]]])),
        "B": (["b0", "b1"], es[[0, 2]]),
        "S": (_labels("s", 12), es),
    }
    dd = pipeline.DocDistance()
    monkeypatch.setattr(dd, "_ensure_grounding", lambda: None)
    monkeypatch.setattr(dd, "embed_statements", lambda doc: docs[doc])
    dd._reranker, dd._nli = _FakeReranker(), _FakeNLI()

    result, smap = dd.distance_wrt_source_with_map("A", "B", "S")
    assert set(smap) == {
        "top_k",
        "anisotropy",
        "source_support_cutoff",
        "n_statements",
        "n_unsupported",
        "n_dropped",
        "drop_rules",
        "a",
        "b",
    }
    for record in [*smap["a"], *smap["b"]]:
        assert set(record) == {
            "index",
            "text",
            "section",
            "matches",
            "nearest_source_index",
            "nearest_source_cost",
            "unsupported",
        }
    assert [r["unsupported"] for r in smap["a"]] == [False, True, False]
    assert [r["nearest_source_index"] for r in smap["a"]][::2] == [4, 9]
    assert smap["a"][1]["nearest_source_cost"] == 1.4142
    assert [r["nearest_source_index"] for r in smap["b"]] == [0, 2]
    assert smap["n_unsupported"] == {"a": 1, "b": 0}
    assert (result.n_unsupported_a, result.n_unsupported_b) == (1, 0)
    settings.reset()


class _NoStatementSegmenter:
    """Segmenter stub for a document that holds no statement."""

    def __init__(self, offline: bool = True):
        pass

    def split(self, text: str) -> list[str]:
        return []


def test_empty_document_raises_on_the_detail_paths(monkeypatch):
    """ACC-MAP-65: a document that segments to zero statements raises on the three detail calls."""
    settings.reset()
    settings.mark_ready("wmd")
    settings.mark_ready("wmd-wrt-source")
    monkeypatch.setattr(pipeline, "Segmenter", _NoStatementSegmenter)
    for loader in ("load_encoder", "load_reranker", "load_nli"):
        monkeypatch.setattr(pipeline, loader, lambda *args, **kwargs: None)
    dd = pipeline.DocDistance()

    with pytest.raises(ValueError, match="no statements"):
        dd.semantic_distance_with_details("one text", "another text")
    with pytest.raises(ValueError, match="no statements"):
        dd.structural_distance_with_details("one text", "another text")
    with pytest.raises(ValueError, match="no statements"):
        dd.distance_wrt_source_with_map("one text", "another text", "the source text")
    settings.reset()


def test_load_body_keeps_hash_comments_inside_fences(tmp_path):
    """DEF-SEGM-6: a bash comment inside a fenced block survives; H1 lines outside are stripped."""
    doc = tmp_path / "doc.md"
    doc.write_text(
        "# Title\n"
        "Some prose.\n"
        "```bash\n"
        "# install the package\n"
        "pip install docdistance\n"
        "```\n"
        "~~~python\n"
        "# a python comment\n"
        "print('hi')\n"
        "~~~\n"
        "# Second H1\n"
        "More prose.\n",
        encoding="utf-8",
    )
    body = pipeline._load_body(doc)
    assert "# install the package" in body  # fenced bash comment kept
    assert "# a python comment" in body  # tilde fence tracked too
    assert "# Title" not in body  # H1 outside fences still stripped
    assert "# Second H1" not in body


def test_read_warns_on_pathlike_missing_file(tmp_path, monkeypatch, capsys):
    """DEF-CLI-5: a mistyped path is still scored as raw text, but a warning names the fallback."""
    monkeypatch.chdir(tmp_path)  # ensure the path genuinely does not resolve
    from loguru import logger as _logger

    records = []
    sink_id = _logger.add(records.append, level="WARNING")
    try:
        assert pipeline._read("missing-file.md") == "missing-file.md"  # return value unchanged
        assert len(records) == 1
        assert "missing-file.md" in records[0]
        assert "raw text" in records[0]
        records.clear()
        assert pipeline._read("just some genuine prose to score") == (
            "just some genuine prose to score"
        )
        assert records == []  # no separator, no document extension: silent
    finally:
        _logger.remove(sink_id)


# --- ACC-GEOM-23: anisotropy pre-pass pool below the minimum estimation pool ---

# statement counts of the stub documents: A + B = 9 is below the minimum of 71 pooled statements,
# A + L67 = 71 is at it
_POOL_DOCS = {"A": 4, "B": 5, "L67": 67}

# the symmetric scoring calls: each applies the pre-pass to the pool of its two documents
_POOL_CALLS = [
    "semantic_distance",
    "semantic_distance_with_details",
    "structural_distance",
    "structural_distance_with_details",
]


@pytest.fixture
def pool_warnings():
    """The messages of the loguru warnings logged during the test, in order."""
    messages: list[str] = []
    sink_id = logger.add(lambda msg: messages.append(msg.record["message"]), level="WARNING")
    yield messages
    logger.remove(sink_id)


@pytest.fixture
def stub_pipeline(monkeypatch):
    """A DocDistance that returns the stub embeddings of ``_POOL_DOCS`` - no model loads."""
    settings.reset()
    settings.mark_ready("wmd")
    docs = {
        name: (_labels(name, n), _emb(n, seed=seed))
        for seed, (name, n) in enumerate(_POOL_DOCS.items(), start=60)
    }
    dd = pipeline.DocDistance()
    monkeypatch.setattr(dd, "embed_statements", lambda doc: docs[doc])
    yield dd
    settings.reset()


@pytest.mark.parametrize("call", _POOL_CALLS)
def test_symmetric_call_warns_once_on_a_pre_pass_pool_below_the_minimum(
    call, stub_pipeline, pool_warnings
):
    """ACC-GEOM-23: with anisotropy on, a pool below the minimum logs one warning that names the
    pooled statement count and the minimum. A pool at the minimum logs nothing, and neither does
    the small pool with anisotropy off."""
    score = getattr(stub_pipeline, call)
    score("A", "B", anisotropy=True)
    assert len(pool_warnings) == 1  # one per pool: the detail calls apply the pre-pass twice
    assert "holds 9 statements" in pool_warnings[0]
    assert "minimum of 71" in pool_warnings[0]

    pool_warnings.clear()
    score("A", "L67", anisotropy=True)
    score("A", "B", anisotropy=False)
    assert pool_warnings == []


def test_pool_warning_boundary_is_the_minimum(pool_warnings):
    """ACC-GEOM-23: 70 pooled statements are below the minimum, 71 are not."""
    pipeline._warn_small_pool(_emb(30), _emb(40))
    assert len(pool_warnings) == 1 and "holds 70 statements" in pool_warnings[0]
    pool_warnings.clear()
    pipeline._warn_small_pool(_emb(30), _emb(41))
    assert pool_warnings == []


@pytest.mark.parametrize("call", ["distance_wrt_source", "distance_wrt_source_with_map"])
def test_source_conditioned_call_logs_no_pool_warning(call, monkeypatch, pool_warnings):
    """ACC-GEOM-23: the source-conditioned calls run the pre-pass by default and log no pool warning.

    The pool of A, B and the source holds 6 statements, below the minimum of 71: the minimum was
    measured on pools without a source, so the warning is on the symmetric calls only.
    """
    settings.reset()
    settings.mark_ready("wmd-wrt-source")
    docs = {
        "A": (["a0", "a1"], _emb(2, seed=1)),
        "B": (["b0"], _emb(1, seed=2)),
        "S": (["s0", "s1", "s2"], _emb(3, seed=3)),
    }
    dd = pipeline.DocDistance()
    monkeypatch.setattr(dd, "_ensure_grounding", lambda: None)
    monkeypatch.setattr(dd, "embed_statements", lambda doc: docs[doc])
    dd._reranker, dd._nli = _FakeReranker(), _FakeNLI()

    getattr(dd, call)("A", "B", "S")
    assert pool_warnings == []
    settings.reset()


def test_pool_warning_changes_no_number(stub_pipeline, pool_warnings):
    """ACC-GEOM-23: the run proceeds - the result is the one the pre-pass gives on that pool."""
    emb_a, emb_b = stub_pipeline.embed_statements("A")[1], stub_pipeline.embed_statements("B")[1]
    result = stub_pipeline.semantic_distance("A", "B", anisotropy=True)
    assert len(pool_warnings) == 1
    assert result == pipeline._core.compute_distance(emb_a, emb_b, anisotropy=True)


# ------------------------------------------------- ACC-CLI-29..43: embed memo and grounding memo


def _vec(text: str, dim: int = 32) -> np.ndarray:
    seed = int(hashlib.sha1(text.encode()).hexdigest(), 16) % (2**32)
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _affinity(left: str, right: str) -> float:
    """A value in [0, 1] read from the two texts, so different texts give different values."""
    return float(_vec(left) @ _vec(right) + 1.0) / 2.0


class _SentenceSegmenter:
    def __init__(self, offline: bool = True):
        pass

    def split(self, text: str) -> list[str]:
        return [s.strip() for s in text.replace("\n", ". ").split(".") if s.strip()]


class _CountingEncoder:
    """One fixed unit vector per statement text; records every batch it encodes."""

    def __init__(self):
        self.batches: list[list[str]] = []

    def encode(self, statements: list[str]) -> np.ndarray:
        self.batches.append(list(statements))
        return np.stack([_vec(s) for s in statements])


class _CountingReranker:
    """A grid read from the texts, so each (document, source) pair has its own; records every call."""

    def __init__(self):
        self.grids: list[tuple[str, ...]] = []

    def score_grid(self, docs: list[str], sources: list[str]) -> np.ndarray:
        self.grids.append(tuple(docs))
        return np.array([[_affinity(d, s) for s in sources] for d in docs], dtype=np.float32)


class _TextNLI:
    def entail(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        return np.array([_affinity(p, h) for p, h in zip(premises, hypotheses)], dtype=np.float32)


@pytest.fixture
def models(monkeypatch):
    """Both modes ready; the segmenter and the three models are stubs that count their calls."""
    settings.reset()
    settings.mark_ready("wmd")
    settings.mark_ready("wmd-wrt-source")
    stubs = SimpleNamespace(encoder=_CountingEncoder(), reranker=_CountingReranker())
    monkeypatch.setattr(pipeline, "Segmenter", _SentenceSegmenter)
    monkeypatch.setattr(pipeline, "load_encoder", lambda *args, **kwargs: stubs.encoder)
    monkeypatch.setattr(pipeline, "load_reranker", lambda *args, **kwargs: stubs.reranker)
    monkeypatch.setattr(pipeline, "load_nli", lambda *args, **kwargs: _TextNLI())
    yield stubs
    settings.reset()


def _memoising() -> pipeline.DocDistance:
    """A ``DocDistance`` with both memos on, at the bound the ``--pairs`` batches use."""
    return pipeline.DocDistance(memo_size=pipeline.MEMO_SIZE)


def _write(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def _report(i: int) -> str:
    """A document of 3 statements; reports of different ``i`` share no statement."""
    return f"Report {i} opens. Report {i} continues. Report {i} closes."


def _reports(tmp_path: Path, n: int) -> list[str]:
    return [_write(tmp_path / f"report-{i}.md", _report(i)) for i in range(n)]


def _all_pairs(docs: list[str]) -> list[tuple[str, str]]:
    return [(a, b) for i, a in enumerate(docs) for b in docs[i + 1 :]]


# a heading, two prose statements, a bare list marker, a table row and a closing statement: the
# three segmentation arguments each give a different statement list
MEMO_DOC = (
    "## Plan\n\nThe first statement is long enough here. Short one.\n\n- \n\n"
    "| cell | value |\n\nThe closing statement is also long enough."
)
# the plain reader is named: the default reader of the .md file that holds MEMO_DOC is the
# markdown reader
_PLAIN = {"input_format": "plain"}
_SEGMENTATIONS = [
    _PLAIN,
    {"markdown": True},
    {**_PLAIN, "min_words": 3},
    {**_PLAIN, "skip_table_fragments": True},
]


def test_all_pairs_sweep_encodes_each_document_once(models, tmp_path):
    """ACC-CLI-29: the 55 pairs of 11 files fill 110 document slots. The memo encodes 11 documents,
    the default instance, which has no memo, 110, and all 55 results are equal."""
    pairs = _all_pairs(_reports(tmp_path, 11))
    assert len(pairs) == 55

    dd = _memoising()
    with_memo = [dd.semantic_distance(a, b) for a, b in pairs]
    assert len(models.encoder.batches) == 11

    models.encoder.batches.clear()
    dd_off = pipeline.DocDistance()
    assert [dd_off.semantic_distance(a, b) for a, b in pairs] == with_memo
    assert len(models.encoder.batches) == 110


def test_memo_key_is_the_resolved_path(models, tmp_path):
    """ACC-CLI-29: a ``Path``, its string and a spelling with ``..`` name one file - one encode."""
    doc = Path(_write(tmp_path / "report.md", _report(0)))
    (tmp_path / "sub").mkdir()
    dd = _memoising()
    for spelling in (doc, str(doc), str(tmp_path / "sub" / ".." / "report.md")):
        dd.embed_statements(spelling)
    assert len(models.encoder.batches) == 1


def test_memo_key_holds_the_segmentation_arguments(models, tmp_path):
    """ACC-CLI-29: one file under four segmentation settings is four entries. Each call returns the
    statements, the vectors and the drop count that the instance without a memo returns, on the
    first pass and on the second, which is served from the memo."""
    doc = _write(tmp_path / "plan.md", MEMO_DOC)
    dd_off = pipeline.DocDistance()
    expected = []
    for seg in _SEGMENTATIONS:
        statements, emb = dd_off.embed_statements(doc, **seg)
        expected.append((statements, emb.tobytes(), dd_off._last_n_dropped))
    assert all(expected[i] != expected[j] for i in range(4) for j in range(i))  # four readings
    assert [dropped for _, _, dropped in expected] == [0, 0, 3, 2]

    models.encoder.batches.clear()
    dd = _memoising()
    for _ in range(2):
        for seg, reading in zip(_SEGMENTATIONS, expected):
            statements, emb = dd.embed_statements(doc, **seg)
            assert (statements, emb.tobytes(), dd._last_n_dropped) == reading
    assert len(models.encoder.batches) == 4


def test_identity_pair_reads_the_same_with_and_without_the_memo(models, tmp_path):
    """ACC-CLI-30: a file against itself. Two encodes give two equal arrays, whose ground cost has a
    diagonal above zero; one array against itself has an exactly zero diagonal. The memo serves a
    copy per call, so the pair reads what two encodes read."""
    doc = _write(tmp_path / "long.md", " ".join(_report(i) for i in range(4)))
    dd, dd_off = _memoising(), pipeline.DocDistance()
    without_memo = dd_off.semantic_distance(doc, doc)
    assert without_memo.smd > 0.0  # the float32 floor of the diagonal
    assert dd.semantic_distance(doc, doc) == without_memo
    assert dd.structural_distance(doc, doc) == dd_off.structural_distance(doc, doc)
    assert len(models.encoder.batches) == 2 + 1 + 2  # the memo encoded the file once


def test_rewritten_file_is_encoded_again(models, tmp_path):
    """ACC-CLI-33: a file rewritten between two calls is segmented and encoded again - one new
    encode - and the pair reads the distance of the new content."""
    a, b = _reports(tmp_path, 2)
    dd = _memoising()
    first = dd.semantic_distance(a, b)
    assert len(models.encoder.batches) == 2

    _write(Path(b), _report(7) + " Report 7 adds one statement.")
    second = dd.semantic_distance(a, b)
    assert len(models.encoder.batches) == 3  # a from the memo, b encoded again
    assert second.n_statements_b == 4 and second != first
    assert second == pipeline.DocDistance().semantic_distance(a, b)


def test_memo_key_is_taken_before_the_file_is_read(models, tmp_path, monkeypatch):
    """ACC-CLI-33: a file rewritten between the read and the store of one call.

    The reader returns the old content and the file then holds the new one. The entry is kept under
    the key taken before the read, the identity of the old file, so the next call misses it and
    reads the new content. A key taken after the read would name the new file and serve the old
    statements for it. ``_read`` is the read of a named ``plain`` or ``markdown`` reader, so the
    plain reader is named; the read of the default ``auto`` is patched in
    ``test_memo_key_is_taken_before_an_html_file_is_read``.
    """
    doc = Path(_write(tmp_path / "report.md", _report(1)))
    read = pipeline._read

    def read_then_rewrite(document):
        text = read(document)
        monkeypatch.setattr(pipeline, "_read", read)  # the rewrite happens once
        _write(doc, _report(2) + " Report 2 adds one statement.")
        return text

    monkeypatch.setattr(pipeline, "_read", read_then_rewrite)
    dd = _memoising()
    assert dd.embed_statements(doc, **_PLAIN)[0][0] == "Report 1 opens"
    assert dd.embed_statements(doc, **_PLAIN)[0][0] == "Report 2 opens"
    assert len(models.encoder.batches) == 2


def test_new_mtime_alone_and_new_size_alone_each_miss_the_memo(models, tmp_path):
    """ACC-CLI-33: the key holds the mtime and the size. A rewrite of equal byte length with a
    later mtime is encoded again, and so is a longer rewrite that carries the earlier mtime."""
    doc = Path(_write(tmp_path / "report.md", _report(1)))
    dd = _memoising()
    dd.embed_statements(doc)
    before = doc.stat()

    _write(doc, _report(2))  # the same byte length
    os.utime(doc, ns=(before.st_atime_ns, before.st_mtime_ns + 10**9))
    assert doc.stat().st_size == before.st_size
    assert dd.embed_statements(doc)[0][0] == "Report 2 opens"

    _write(doc, _report(3) + " More.")
    os.utime(doc, ns=(before.st_atime_ns, before.st_mtime_ns + 10**9))  # the mtime of report 2
    assert dd.embed_statements(doc)[0][0] == "Report 3 opens"
    assert len(models.encoder.batches) == 3


def test_embed_memo_evicts_beyond_its_bound(models, tmp_path):
    """ACC-CLI-34: at a bound of 3 the fourth document evicts the first, which is encoded again on
    its next use; the fourth is still served from the memo."""
    docs = _reports(tmp_path, 4)
    dd = pipeline.DocDistance(memo_size=3)
    for doc in docs:
        dd.embed_statements(doc)
    assert len(models.encoder.batches) == 4
    dd.embed_statements(docs[3])
    assert len(models.encoder.batches) == 4
    dd.embed_statements(docs[0])
    assert len(models.encoder.batches) == 5
    assert models.encoder.batches[-1] == models.encoder.batches[0]


def test_embed_memo_evicts_the_least_recently_used_entry(models, tmp_path):
    """ACC-CLI-34: at a bound of 2, a document served from the memo after the second one was
    encoded is the more recently used of the two. The third document evicts the second."""
    first, second, third = _reports(tmp_path, 3)
    dd = pipeline.DocDistance(memo_size=2)
    for doc in (first, second, first, third):
        dd.embed_statements(doc)
    assert len(models.encoder.batches) == 3
    dd.embed_statements(first)  # still held
    assert len(models.encoder.batches) == 3
    dd.embed_statements(second)  # evicted
    assert len(models.encoder.batches) == 4


def test_raw_text_is_never_memoised(models, tmp_path):
    """ACC-CLI-35: the same raw text scored twice is encoded twice per document, and the text of a
    memoised file given as raw text is encoded, not served."""
    dd = _memoising()
    for _ in range(2):
        dd.semantic_distance(_report(1), _report(2))
    assert len(models.encoder.batches) == 4

    dd.embed_statements(_write(tmp_path / "report.md", _report(3)))
    dd.embed_statements(_report(3))
    assert len(models.encoder.batches) == 6


def test_directory_argument_is_read_as_raw_text_on_every_call(models, tmp_path, pool_warnings):
    """ACC-CLI-35: a string that names a directory is raw text for the reader. It gets no memo
    key: each call encodes it and logs the reader's warning."""
    folder = tmp_path / "folder"
    folder.mkdir()
    dd = _memoising()
    for _ in range(2):
        dd.embed_statements(str(folder))
    assert len(models.encoder.batches) == 2
    assert len([w for w in pool_warnings if "looks like a file path" in w]) == 2


def test_document_of_another_type_raises_the_reader_error(models):
    """ACC-CLI-35: an argument that is neither ``str`` nor ``Path`` gets no memo key; the reader
    raises its ``TypeError`` as before."""
    with pytest.raises(TypeError, match="document must be str or Path, got int"):
        _memoising().embed_statements(5)


def test_raw_text_that_is_no_valid_path_gets_no_memo_key(models):
    """ACC-CLI-35: raw text with a NUL character cannot be a path. It is read as raw text, as
    without the memo."""
    text = "Report 1 opens\x00. Report 1 closes."
    statements, _ = _memoising().embed_statements(text)
    assert statements == pipeline.DocDistance().embed_statements(text)[0]
    assert len(statements) == 2


def test_memo_off_takes_no_key(models, tmp_path, monkeypatch):
    """ACC-CLI-32: at the default ``memo_size=0`` a call reads no file identity - the off setting
    is the path without the memo - and every call encodes."""

    def no_key(*args):
        raise AssertionError("the memo key is taken with the memo off")

    monkeypatch.setattr(pipeline, "_memo_key", no_key)
    doc = _write(tmp_path / "report.md", _report(1))
    dd = pipeline.DocDistance()
    for _ in range(2):
        dd.embed_statements(doc)
    assert len(models.encoder.batches) == 2


def test_drop_count_is_restored_on_a_memo_hit(models, tmp_path):
    """ACC-CLI-36: document A drops 3 statements under ``min_words=3``, document B none. Scored a
    second time, both from the memo, the pair reports the same counts."""
    a = _write(tmp_path / "plan.md", MEMO_DOC)
    b = _write(tmp_path / "report.md", "Report 1 opens here. Report 1 closes here.")
    dd = _memoising()

    dd.embed_statements(a, min_words=3)
    assert dd._last_n_dropped == 3
    dd.semantic_distance(a, b, min_words=3)
    assert dd.last_n_dropped == {"a": 3, "b": 0}
    assert len(models.encoder.batches) == 2

    dd.semantic_distance(a, b, min_words=3)  # both documents from the memo
    assert dd.last_n_dropped == {"a": 3, "b": 0}
    dd.semantic_distance(b, a, min_words=3)
    assert dd.last_n_dropped == {"a": 0, "b": 3}
    assert len(models.encoder.batches) == 2


def test_marker_warning_is_logged_on_a_memo_hit(models, tmp_path, pool_warnings):
    """DEF-SEGM-12 with ACC-CLI-36: a call logs one marker warning for a document that holds such
    statements, whether the document was encoded or served from the memo. The warning belongs to
    the plain reader, which is named for the ``.md`` file."""
    doc = _write(tmp_path / "plan.md", MEMO_DOC)
    dd = _memoising()
    for _ in range(3):
        dd.embed_statements(doc, **_PLAIN)
    assert len(models.encoder.batches) == 1
    assert len(pool_warnings) == 3
    # the same line on the encode and on both hits, naming the file
    assert all(w.startswith(f"{doc}: 1 of 6 statements hold no letter") for w in pool_warnings)
    assert len(set(pool_warnings)) == 1

    pool_warnings.clear()
    for _ in range(2):
        dd.embed_statements(doc, markdown=True)  # the markdown reader logs none
    assert pool_warnings == []


def test_embed_memo_serves_copies(models, tmp_path):
    """ACC-CLI-41 on the embed memo: a caller that changes the returned list or array in place,
    after the encode or after a hit, does not change what a later hit returns."""
    doc = _write(tmp_path / "report.md", _report(1))
    dd = _memoising()
    statements, emb = dd.embed_statements(doc)
    kept_statements, kept_emb = list(statements), emb.copy()

    for _ in range(2):  # first what the encode returned, then what a hit returned
        statements.clear()
        emb[:] = 0.0
        statements, emb = dd.embed_statements(doc)
        assert statements == kept_statements
        assert np.array_equal(emb, kept_emb)
    assert len(models.encoder.batches) == 1


def _wrt_sweep(dd, pairs, source, **kwargs):
    return [dd.distance_wrt_source(a, b, source, **kwargs) for a, b in pairs]


def test_wrt_source_sweep_computes_each_grounding_once(models, tmp_path):
    """ACC-CLI-38: 55 pairs of 11 files against one source fill 110 (document, source) slots. The
    memo computes 11 relevance grids and encodes 12 files; without it 110 grids and 165 encodes.
    The 55 results are equal, and the map call is served from the same memo."""
    pairs = _all_pairs(_reports(tmp_path, 11))
    source = _write(tmp_path / "source.md", " ".join(_report(i) for i in range(0, 22, 2)))

    dd = _memoising()
    with_memo = _wrt_sweep(dd, pairs, source)
    assert len(models.reranker.grids) == 11
    assert len(set(models.reranker.grids)) == 11  # one per document
    assert len(models.encoder.batches) == 12
    maps = [dd.distance_wrt_source_with_map(a, b, source) for a, b in pairs]
    assert len(models.reranker.grids) == 11
    assert [result for result, _ in maps] == with_memo

    models.reranker.grids.clear()
    models.encoder.batches.clear()
    dd_off = pipeline.DocDistance()
    assert _wrt_sweep(dd_off, pairs, source) == with_memo
    assert len(models.reranker.grids) == 110
    assert len(models.encoder.batches) == 165
    assert [dd_off.distance_wrt_source_with_map(a, b, source) for a, b in pairs] == maps


def test_grounding_memo_serves_copies(models):
    """ACC-CLI-41: a grid or an entailment vector changed in place by the caller, after the
    computation or after a hit, does not change what a later hit returns."""
    dd = _memoising()
    dd._ensure_grounding()
    doc, source = ["Report 1 opens", "Report 1 closes"], ["Report 1 opens", "Report 2", "Report 3"]
    kept_grid, kept_entail = dd._grounding(doc, source)
    models.reranker.grids.clear()

    grid, entail = dd._grounding_memoised(("doc",), ("source",), doc, source)
    for _ in range(2):  # first what the computation returned, then what a hit returned
        grid[:] = 0.0
        entail[:] = 0.0
        grid, entail = dd._grounding_memoised(("doc",), ("source",), doc, source)
        assert np.array_equal(grid, kept_grid)
        assert np.array_equal(entail, kept_entail)
    assert len(models.reranker.grids) == 1


def test_grounding_memo_evicts_beyond_its_bound(models, tmp_path):
    """ACC-CLI-42: three documents against one source at a bound of 2. The grid of the third
    document evicts the grid of the first, which is computed again on its next use; at a bound of
    3 it is not."""
    first, second, third = _reports(tmp_path, 3)
    source = _write(tmp_path / "source.md", _report(1) + " " + _report(5))
    calls = [(first, second), (second, third), (first, third)]

    _wrt_sweep(pipeline.DocDistance(memo_size=2), calls, source)
    assert [grid[0] for grid in models.reranker.grids] == [
        "Report 0 opens",
        "Report 1 opens",
        "Report 2 opens",
        "Report 0 opens",  # evicted by the third document
    ]

    models.reranker.grids.clear()
    _wrt_sweep(pipeline.DocDistance(memo_size=3), calls, source)
    assert len(models.reranker.grids) == 3


def test_grounding_of_raw_text_is_never_memoised(models, tmp_path):
    """ACC-CLI-35 on the grounding memo: a pair with a raw-text document or a raw-text source is
    computed on every call, and each call reads what the instance without a memo reads."""
    a, b = _reports(tmp_path, 2)
    source = _write(tmp_path / "source.md", _report(1) + " " + _report(5))
    dd, dd_off = _memoising(), pipeline.DocDistance()

    for text in (_report(3), _report(4), _report(3)):  # raw document against a file source
        assert dd.distance_wrt_source(text, b, source) == dd_off.distance_wrt_source(
            text, b, source
        )
    # per call: the memo computes the raw document, the other instance both documents
    assert len(models.reranker.grids) == (3 + 1) + 3 * 2

    models.reranker.grids.clear()
    for text in (_report(5), _report(6), _report(5)):  # file documents against a raw source
        assert dd.distance_wrt_source(a, b, text) == dd_off.distance_wrt_source(a, b, text)
    assert len(models.reranker.grids) == 3 * 2 + 3 * 2


def test_grounding_memo_key_holds_both_documents_and_the_segmentation(models, tmp_path):
    """ACC-CLI-38: the grounding of a pair is kept under the memo keys of the document and of the
    source. A rewritten source, a rewritten document and a new segmentation setting are each
    computed again and read what the instance without a memo reads."""
    a, b = _write(tmp_path / "plan.md", MEMO_DOC), _write(tmp_path / "report.md", _report(1))
    source = _write(tmp_path / "source.md", _report(1) + " " + _report(5))
    dd, dd_off = _memoising(), pipeline.DocDistance()

    def grids_computed(**seg) -> int:
        """Score the pair on both instances, compare, and return the grids the memo computed."""
        models.reranker.grids.clear()
        result = dd.distance_wrt_source(a, b, source, **seg)
        computed = len(models.reranker.grids)
        assert result == dd_off.distance_wrt_source(a, b, source, **seg)
        return computed

    assert grids_computed() == 2
    assert grids_computed() == 0
    assert grids_computed(min_words=3) == 2  # other statements of A, B and the source
    assert grids_computed(min_words=3) == 0
    _write(Path(source), _report(1) + " " + _report(6) + " Report 6 adds one statement.")
    assert grids_computed() == 2  # both documents against the new source
    _write(Path(b), _report(2) + " Report 2 adds one statement.")
    assert grids_computed() == 1  # document B alone


# ------------------------------ ACC-SEGM-93, ACC-SEGM-94, ACC-SEGM-91: the option input_format

# the scoring calls that take ``markdown``, with the number of documents each takes
_FORMAT_METHODS = {
    "embed_statements": 1,
    "embed": 1,
    "semantic_distance": 2,
    "semantic_distance_with_details": 2,
    "structural_distance": 2,
    "structural_distance_with_details": 2,
    "distance_wrt_source": 3,
    "distance_wrt_source_with_map": 3,
}
_FORMAT_ONE_SHOTS = {
    "semantic_distance": 2,
    "structural_distance": 2,
    "source_conditioned_distance": 3,
}
_FORMAT_CALLS = [("method", name) for name in _FORMAT_METHODS] + [
    ("one-shot", name) for name in _FORMAT_ONE_SHOTS
]

# prose with one line that starts with a dash: the plain reader returns the bare "-" as a statement
# (1 of 3)
ONE_DASH_DOC = "Adoption held steady.\n-\nBarriers remain.\n"


def _format_markdown(i: int) -> str:
    """The markdown text of document ``i`` of :func:`_format_twins`."""
    return (
        f"## Section {i}\n\nReport {i} opens here. Report {i} goes on.\n\n"
        f"- point {i} one\n- point {i} two\n"
    )


def _format_twins(tmp_path: Path, n: int) -> tuple[list[str], list[str]]:
    """``n`` documents, each written as a markdown file and as an HTML file.

    A heading, a paragraph of two statements and two list items. Documents of different index
    share no statement.
    """
    md_files, html_files = [], []
    for i in range(n):
        md_files.append(_write(tmp_path / f"doc-{i}.md", _format_markdown(i)))
        html_files.append(
            _write(
                tmp_path / f"doc-{i}.html",
                f"<h2>Section {i}</h2>\n<p>Report {i} opens here. Report {i} goes on.</p>\n"
                f"<ul>\n<li>point {i} one</li>\n<li>point {i} two</li>\n</ul>\n",
            )
        )
    return md_files, html_files


def _comparable(value):
    """``value`` with every array replaced by its bytes, so two results compare with ``==``."""
    if isinstance(value, np.ndarray):
        return value.tobytes()
    if isinstance(value, tuple):
        return tuple(_comparable(item) for item in value)
    return value


def test_every_call_that_takes_markdown_takes_input_format_with_the_default_plain():
    """ACC-SEGM-94: the eight methods of ``DocDistance`` and the three one-shot functions are every
    public callable of the module with a ``markdown`` parameter, and each has ``input_format``
    with the default ``"auto"``. The name of this test is from the build whose default was
    ``"plain"``."""
    import inspect

    calls = [getattr(pipeline.DocDistance, name) for name in _FORMAT_METHODS]
    calls += [getattr(pipeline, name) for name in _FORMAT_ONE_SHOTS]
    public = [
        member
        for owner in (pipeline, pipeline.DocDistance)
        for name, member in vars(owner).items()
        if inspect.isfunction(member)
        and member.__module__ == pipeline.__name__
        and not name.startswith("_")
        and "markdown" in inspect.signature(member).parameters
    ]
    assert set(public) == set(calls)
    for call in calls:
        assert inspect.signature(call).parameters["input_format"].default == "auto"


@pytest.mark.parametrize(("kind", "name"), _FORMAT_CALLS)
def test_input_format_reaches_the_reader_of_every_document(kind, name, models, tmp_path):
    """ACC-SEGM-94, ACC-SEGM-86: each call on HTML files with ``input_format="html"``, with
    ``"auto"`` and without the option returns what it returns on the markdown twins with
    ``markdown=True``: result, statements, vectors and the detail dict with its section fields.
    The default ``auto`` reads the markdown files as ``markdown=True`` does. The plain reader
    returns something else for the files of both kinds."""
    if kind == "method":
        call, n_docs = getattr(pipeline.DocDistance(), name), _FORMAT_METHODS[name]
    else:
        call, n_docs = getattr(pipeline, name), _FORMAT_ONE_SHOTS[name]
    md_files, html_files = _format_twins(tmp_path, n_docs)

    expected = _comparable(call(*md_files, markdown=True))
    assert _comparable(call(*html_files, input_format="html")) == expected
    assert _comparable(call(*html_files, input_format="auto")) == expected
    assert _comparable(call(*md_files, input_format="auto")) == expected
    assert _comparable(call(*md_files, input_format="markdown")) == expected
    assert _comparable(call(*html_files)) == expected
    assert _comparable(call(*md_files)) == expected
    assert _comparable(call(*html_files, input_format="plain")) != expected
    assert _comparable(call(*md_files, input_format="plain")) != expected


@pytest.mark.parametrize(("kind", "name"), _FORMAT_CALLS)
def test_default_reader_of_every_call_is_the_plain_reader_without_a_reader_extension(
    kind, name, models, tmp_path
):
    """ACC-SEGM-94, ACC-SEGM-93: each call without ``input_format`` on ``.txt`` files and on raw
    text returns what it returns with ``input_format="plain"``. The texts are the markdown
    documents of ``_format_twins``, so the markdown reader returns something else for them, and
    ``markdown=True`` alone returns what ``input_format="markdown"`` returns: on these documents
    the flag, not the extension, chooses the markdown reader."""
    if kind == "method":
        call, n_docs = getattr(pipeline.DocDistance(), name), _FORMAT_METHODS[name]
    else:
        call, n_docs = getattr(pipeline, name), _FORMAT_ONE_SHOTS[name]
    texts = [_format_markdown(i) for i in range(n_docs)]
    txt_files = [_write(tmp_path / f"doc-{i}.txt", text) for i, text in enumerate(texts)]

    for docs in (txt_files, texts):
        plain = _comparable(call(*docs, input_format="plain"))
        assert _comparable(call(*docs)) == plain
        by_value = _comparable(call(*docs, input_format="markdown"))
        assert by_value != plain
        assert _comparable(call(*docs, markdown=True)) == by_value


@pytest.fixture
def model_loads(monkeypatch):
    """Both modes ready; returns the list that the load of the segmenter and of each model adds
    its name to."""
    settings.reset()
    settings.mark_ready("wmd")
    settings.mark_ready("wmd-wrt-source")
    loads: list[str] = []
    for name in ("Segmenter", "load_encoder", "load_reranker", "load_nli"):
        monkeypatch.setattr(
            pipeline, name, lambda *args, _name=name, **kwargs: loads.append(_name)
        )
    yield loads
    settings.reset()


@pytest.mark.parametrize(
    "reader",
    [{"input_format": "pdf"}, {"markdown": True, "input_format": "plain"}]
    + [{"markdown": True, "input_format": "html"}],
    ids=["unknown value", "markdown flag and plain", "markdown flag and html"],
)
@pytest.mark.parametrize(("kind", "name"), _FORMAT_CALLS)
def test_reader_error_of_every_call_is_raised_before_a_model_loads(
    kind, name, reader, model_loads
):
    """ACC-SEGM-94: an unknown ``input_format``, and ``markdown=True`` beside ``"plain"`` or
    ``"html"``, raise ``ValueError`` in each of the eleven calls with no segmenter and no model
    loaded. The two source-conditioned methods loaded all four before they read the argument."""
    if kind == "method":
        call, n_docs = getattr(pipeline.DocDistance(), name), _FORMAT_METHODS[name]
    else:
        call, n_docs = getattr(pipeline, name), _FORMAT_ONE_SHOTS[name]
    with pytest.raises(ValueError, match="input_format"):
        call(*[_report(i) for i in range(n_docs)], **reader)
    assert model_loads == []


@pytest.mark.parametrize(
    "name",
    ["semantic_distance_with_details", "structural_distance_with_details"]
    + ["distance_wrt_source_with_map"],
)
def test_detail_records_of_html_files_carry_the_heading_of_their_section(name, models, tmp_path):
    """ACC-SEGM-86: the section scan of a details call reads the converted text of an HTML
    document. Every record of document A and of document B names the ``h2`` of its document. The
    HTML is stored under ``.txt`` names, for which the extension rule chooses the plain reader, so
    the scan gets the converted text only when the explicit reader reaches it."""
    n_docs = _FORMAT_METHODS[name]
    _, html_files = _format_twins(tmp_path, n_docs)
    txt_files = [
        _write(tmp_path / f"html-{i}.txt", Path(html).read_text(encoding="utf-8"))
        for i, html in enumerate(html_files)
    ]
    _, details = getattr(pipeline.DocDistance(), name)(*txt_files, input_format="html")
    if name == "distance_wrt_source_with_map":
        sections_a = {record["section"] for record in details["a"]}
        sections_b = {record["section"] for record in details["b"]}
    elif name == "semantic_distance_with_details":
        sections_a = {record["section"] for record in details["flows"]}
        sections_b = {record["section"] for record in details["targets"]}
    else:
        sections_a = {record["section"] for record in details["statements"]}
        sections_b = {record["target_section"] for record in details["statements"]}
    assert (sections_a, sections_b) == ({"Section 0"}, {"Section 1"})


def test_memo_holds_one_entry_per_requested_input_format(models, tmp_path):
    """ACC-SEGM-91: the memo key carries the requested value. ``markdown=True`` and
    ``input_format="markdown"`` name one entry; ``auto`` on the same markdown file is a second
    entry with the same statements, which a call without the option shares, and ``plain`` a
    third. A second pass encodes nothing."""
    doc = _write(tmp_path / "plan.md", MEMO_DOC)
    dd = _memoising()
    readings, keys = {}, {}
    requests = {
        "flag": {"markdown": True},
        "markdown": {"input_format": "markdown"},
        "auto": {"input_format": "auto"},
        "default": {},
        "plain": {"input_format": "plain"},
    }
    for label, kwargs in requests.items():
        readings[label] = dd.embed_statements(doc, **kwargs)[0]
        keys[label] = dd._last_memo_key
    assert len(models.encoder.batches) == 3
    assert keys["flag"] == keys["markdown"] and keys["default"] == keys["auto"]
    assert len({keys["markdown"], keys["auto"], keys["plain"]}) == 3
    assert [keys[label][3] for label in ("markdown", "auto", "plain")] == [
        "markdown",
        "auto",
        "plain",
    ]
    assert readings["auto"] == readings["markdown"] != readings["plain"]

    for kwargs in requests.values():
        dd.embed_statements(doc, **kwargs)
    assert len(models.encoder.batches) == 3


@pytest.mark.parametrize("target_first", [True, False], ids=["target first", "link first"])
def test_memo_reads_two_names_of_one_file_with_the_reader_of_each_name(
    target_first, models, tmp_path
):
    """ACC-SEGM-93, ACC-SEGM-91: a symbolic link ``notes.txt`` to ``plan.md`` gives one resolved
    path two names. ``auto`` reads each name with the reader of its own extension, and the memo
    key holds that reader, so each name is one entry: both passes return what the instance
    without a memo returns, in either order of the two names."""
    target = Path(_write(tmp_path / "plan.md", MEMO_DOC))
    link = tmp_path / "notes.txt"
    link.symlink_to(target)
    docs = [target, link] if target_first else [link, target]
    dd_off = pipeline.DocDistance()
    expected = [dd_off.embed_statements(doc)[0] for doc in docs]
    assert expected[0] != expected[1]  # the markdown reader and the plain reader
    models.encoder.batches.clear()

    dd = _memoising()
    for _ in range(2):
        assert [dd.embed_statements(doc)[0] for doc in docs] == expected
    assert len(models.encoder.batches) == 2


def test_memo_key_is_taken_before_an_html_file_is_read(models, tmp_path, monkeypatch):
    """ACC-SEGM-91: the read of ``test_memo_key_is_taken_before_the_file_is_read`` on the path of
    the HTML reader. The file is rewritten between the read and the store of one call; the entry
    is kept under the key taken before the read, so the next call reads the new content."""
    doc = Path(_write(tmp_path / "report.html", "<p>Report 1 opens. Report 1 closes.</p>\n"))
    read_as = pipeline._read_as

    def read_then_rewrite(document, requested):
        result = read_as(document, requested)
        monkeypatch.setattr(pipeline, "_read_as", read_as)  # the rewrite happens once
        _write(doc, "<p>Report 2 opens. Report 2 closes. Report 2 adds one statement.</p>\n")
        return result

    monkeypatch.setattr(pipeline, "_read_as", read_then_rewrite)
    dd = _memoising()
    assert dd.embed_statements(doc, input_format="html")[0][0] == "Report 1 opens"
    assert dd.embed_statements(doc, input_format="html")[0][0] == "Report 2 opens"
    assert len(models.encoder.batches) == 2


def test_marker_warning_on_a_memo_hit_follows_the_reader_that_ran(models, tmp_path, pool_warnings):
    """ACC-SEGM-91: under ``auto`` the entry holds the reader that ran. A ``.txt`` file, read as
    plain text, logs the marker warning on the encode and on each hit. A ``.md`` and a ``.html``
    file, read as markdown and as HTML, log none, also when the reader keeps a statement without
    a letter and without a digit."""
    plain = _write(tmp_path / "notes.txt", ONE_DASH_DOC)
    markdown = _write(tmp_path / "plan.md", "## Plan\n\nReport 1 opens.\n\n?!\n")
    html = _write(tmp_path / "plan.html", "<h2>Plan</h2>\n<p>Report 1 opens.</p>\n<p>?!</p>\n")
    dd = _memoising()
    for _ in range(3):
        dd.embed_statements(plain, input_format="auto")
        assert dd.embed_statements(markdown, input_format="auto")[0][-1] == "?!"
        assert dd.embed_statements(html, input_format="auto")[0][-1] == "?!"
    assert len(models.encoder.batches) == 3
    assert len(pool_warnings) == 3
    assert all(w.startswith(f"{plain}: 1 of 3 statements hold no letter") for w in pool_warnings)
    # the same line on the encode and on both hits
    assert len(set(pool_warnings)) == 1


def test_one_line_html_text_is_not_read_as_a_mistyped_path(
    models, pool_warnings, tmp_path, monkeypatch
):
    """Every closing tag holds ``/``, so a raw HTML text on one line of under 256 characters met
    the rule for a mistyped path. A text that holds ``<`` is no path: it logs no path warning. A
    path that names no file still logs it."""
    monkeypatch.chdir(tmp_path)  # docs/missing.md names no file here
    html = (
        "<h2>Steps</h2><ol><li>open the valve</li><li>close the valve</li></ol>"
        "<blockquote><p>a quoted line</p></blockquote>"
        "<p>See <a href='http://x'>the guide</a> for <strong>more</strong>.</p>"
    )
    assert len(html) < 256
    dd = pipeline.DocDistance()
    dd.embed_statements(html, input_format="auto")
    assert pool_warnings == []
    dd.embed_statements("docs/missing.md", input_format="auto")
    assert len(pool_warnings) == 1 and "looks like a file path" in pool_warnings[0]


@pytest.mark.parametrize("name", ["missing.md", "missing.markdown", "missing.html", "missing.htm"])
def test_missing_file_with_a_reader_extension_logs_the_path_warning(
    name, models, pool_warnings, tmp_path, monkeypatch
):
    """A name without a slash that ends in one of the four extensions of the extension rule and
    names no file is read as raw text, and the warning for a mistyped path names it."""
    monkeypatch.chdir(tmp_path)  # the name is no file here
    pipeline.DocDistance().embed_statements(name)
    assert len(pool_warnings) == 1 and "looks like a file path" in pool_warnings[0]
