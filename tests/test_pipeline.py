"""Pure-logic tests for the pipeline's statement-map builders.

No models load - ``_build_source_map`` and ``_build_transport_map`` take statement texts and
pre-computed embeddings, so the map shape and content are exercised on synthetic arrays in the
lightweight uv ``.venv``.
"""

import numpy as np
import pytest

from docdistance.pipeline import _build_source_map, _build_transport_map


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
    m = _build_transport_map(sa, ea, sb, eb, anisotropy=False)

    assert m["n_statements"] == {"a": 5, "b": 4}
    assert m["smd"] >= 0.0 and m["anisotropy"] is False
    assert len(m["flows"]) == 5  # one group per A statement
    first = m["flows"][0]
    assert first["index"] == 0 and first["text"] == "a0"
    assert first["matches"]  # every statement sends its mass somewhere
    weights = [mt["weight"] for mt in first["matches"]]
    assert weights == sorted(weights, reverse=True)  # descending weight
    assert sum(weights) == pytest.approx(1.0, abs=1e-3)  # row-normalized fractions
    for mt in first["matches"]:
        assert 0 <= mt["target_index"] < 4
        assert mt["target_text"] == sb[mt["target_index"]]
        assert mt["cost"] >= 0.0  # ground distance of the match


def test_transport_map_anisotropy_flag_recorded():
    sa, ea = [f"a{i}" for i in range(4)], _emb(4, seed=13)
    sb, eb = [f"b{i}" for i in range(4)], _emb(4, seed=14)
    m = _build_transport_map(sa, ea, sb, eb, anisotropy=True)
    assert m["anisotropy"] is True
    assert len(m["flows"]) == 4


def test_transport_map_identical_maps_each_to_itself():
    """Identical documents: each statement maps to its twin at full weight and zero cost."""
    s, e = [f"s{i}" for i in range(5)], _emb(5, seed=21)
    m = _build_transport_map(s, e, s, e, anisotropy=False)
    assert m["smd"] == pytest.approx(0.0, abs=1e-5)
    for flow in m["flows"]:
        best = flow["matches"][0]
        assert best["target_index"] == flow["index"]  # statement -> itself
        assert best["weight"] == pytest.approx(1.0, abs=1e-3)
        assert best["cost"] == pytest.approx(0.0, abs=1e-4)
