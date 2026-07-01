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
