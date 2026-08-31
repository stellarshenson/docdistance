"""Pure-logic tests for the pipeline's statement-map builders.

No models load - ``_build_source_map`` and ``_build_semantic_details`` take statement texts and
pre-computed embeddings, so the map shape and content are exercised on synthetic arrays in the
lightweight uv ``.venv``.
"""

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
    }
    assert set(m["definitions"]) == {
        "changed",
        "no_counterpart",
        "identical",
        "share_below_changed",
    }
    for text in m["definitions"].values():
        assert isinstance(text, str) and text


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
