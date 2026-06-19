"""Pure-logic tests for the pipeline's statement-to-source map builder.

No models load - ``_build_source_map`` takes statement texts and pre-computed embeddings, so the
alignment-map shape and content are exercised on synthetic arrays in the lightweight uv ``.venv``.
"""

import numpy as np

from docdistance.pipeline import _build_source_map


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
