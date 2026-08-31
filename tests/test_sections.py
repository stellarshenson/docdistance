"""Section-awareness tests: heading paths on statement records and per-section aggregates.

No models load - ``_statement_sections`` is pure text, the detail builders take synthetic
embeddings, and the end-to-end path fakes the segmenter + encoder like ``test_e2e.py``. Covers
ACC-MAP-3 (every record in the semantic flows, the structural statements and the source map
carries ``section``, the nearest preceding heading chain) and ACC-MAP-9 (the ``sections``
aggregate block in both detail files).
"""

import hashlib
import json

import numpy as np
import pytest

from docdistance import pipeline, settings
from docdistance.pipeline import (
    DIFF_CHANGED_COST,
    DocDistance,
    _build_semantic_details,
    _build_source_map,
    _build_structural_details,
    _statement_sections,
)


def _emb(n: int, dim: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _basis(dim: int = 8) -> np.ndarray:
    """Orthogonal unit vectors - pairwise ground cost sqrt(2), identical rows cost 0."""
    return np.eye(dim, dtype=np.float32)


# --- _statement_sections -------------------------------------------------------------------------


def test_statement_sections_chain_and_pre_heading_empty():
    body = (
        "Intro line.\n\n"
        "# Delivery options\n"
        "Common text.\n\n"
        "## Option A\n"
        "A text.\n\n"
        "## Option B\n"
        "B text.\n"
    )
    stmts = ["Intro line.", "Common text.", "A text.", "B text."]
    assert _statement_sections(body, stmts) == [
        "",
        "Delivery options",
        "Delivery options > Option A",
        "Delivery options > Option B",
    ]


def test_statement_sections_sibling_heading_replaces_deeper_chain():
    body = "## One\n### Deep\ndeep text.\n## Two\ntwo text.\n"
    assert _statement_sections(body, ["deep text.", "two text."]) == ["One > Deep", "Two"]


def test_statement_sections_heading_statement_carries_its_own_chain():
    """The markdown segmenter reports raw heading lines as statements - they belong to themselves."""
    body = "# Top\n## Sub\nText under sub.\n"
    assert _statement_sections(body, ["# Top", "## Sub", "Text under sub."]) == [
        "Top",
        "Top > Sub",
        "Top > Sub",
    ]


def test_statement_sections_fenced_hash_line_is_not_a_heading():
    body = "## Real\nprose here.\n```bash\n# not a heading\necho hi\n```\nmore prose.\n"
    stmts = ["prose here.", "# not a heading", "echo hi", "more prose."]
    assert _statement_sections(body, stmts) == ["Real"] * 4


def test_statement_sections_duplicate_text_resolved_sequentially():
    body = "## S1\nSame line.\n## S2\nSame line.\n"
    assert _statement_sections(body, ["Same line.", "Same line."]) == ["S1", "S2"]


def test_statement_sections_unlocatable_statement_inherits_cursor_chain():
    body = "## Only\nknown text.\n"
    assert _statement_sections(body, ["known text.", "totally absent"]) == ["Only", "Only"]


def test_statement_sections_no_headings_is_all_empty():
    assert _statement_sections("Just prose. More prose.", ["Just prose.", "More prose."]) == [
        "",
        "",
    ]


def test_statement_sections_statement_spanning_lines_matches_normalised():
    """A SaT statement built from joined prose lines still locates in the whitespace-normalised body."""
    body = "## Sec\nfirst half\nsecond half.\n"
    assert _statement_sections(body, ["first half second half."]) == ["Sec"]


# --- semantic details: section on flows + aggregates ---------------------------------------------


def test_semantic_flows_carry_section_and_aggregates():
    u = _basis()
    sa, ea = ["a0", "a1", "a2"], u[[0, 1, 2]]
    sb, eb = ["b0", "b1", "b2"], u[[0, 1, 7]]  # a2 has no counterpart (cost sqrt(2))
    m = _build_semantic_details(sa, ea, sb, eb, anisotropy=False, sections=["X", "X", "Y"])

    assert [f["section"] for f in m["flows"]] == ["X", "X", "Y"]
    assert m["sections"] == [
        {"section": "X", "n_statements": 2, "mean_cost": 0.0, "share_below_changed": 1.0},
        {
            "section": "Y",
            "n_statements": 1,
            "mean_cost": pytest.approx(np.sqrt(2.0), abs=1e-3),
            "share_below_changed": 0.0,
        },
    ]
    json.dumps(m)  # whole dict round-trips through JSON


def test_semantic_details_default_sections_are_empty_strings():
    sa, ea = [f"a{i}" for i in range(3)], _emb(3, seed=1)
    sb, eb = [f"b{i}" for i in range(3)], _emb(3, seed=2)
    m = _build_semantic_details(sa, ea, sb, eb, anisotropy=False)
    assert all(f["section"] == "" for f in m["flows"])
    assert len(m["sections"]) == 1
    assert m["sections"][0]["section"] == "" and m["sections"][0]["n_statements"] == 3


# --- structural details: section / target_section + aggregates -----------------------------------


def test_structural_records_carry_section_and_target_section():
    """A statement kept in place but filed under another heading in B is a cross-section target."""
    u = _basis()
    s = [f"s{i}" for i in range(4)]
    e = u[[0, 1, 2, 3]]
    diff = _build_structural_details(
        s,
        e,
        s,
        e,  # identical content and order - identity alignment, zero displacement
        anisotropy=False,
        sections_a=["X", "X", "Y", "Y"],
        sections_b=["X", "X", "X", "Y"],  # in B, statement 2 sits under X
    )
    st = {r["index"]: r for r in diff["statements"]}
    assert st[2]["section"] == "Y" and st[2]["target_section"] == "X"
    assert st[3]["section"] == "Y" and st[3]["target_section"] == "Y"

    aggs = {row["section"]: row for row in diff["sections"]}
    assert set(aggs) == {"X", "Y"}
    for row in diff["sections"]:
        assert set(row) == {
            "section",
            "n_statements",
            "mean_cost",
            "share_below_changed",
            "median_displacement_norm",
            "share_target_other_section",
        }
    assert aggs["X"]["n_statements"] == 2
    assert aggs["X"]["share_target_other_section"] == 0.0
    assert aggs["X"]["median_displacement_norm"] == 0.0
    assert aggs["X"]["mean_cost"] == 0.0 and aggs["X"]["share_below_changed"] == 1.0
    assert aggs["Y"]["share_target_other_section"] == 0.5  # statement 2 of 2 crossed sections
    json.dumps(diff)


def test_structural_aggregates_all_no_counterpart_section_is_null():
    """A section whose every statement aligns to an absorber has no displacement or target share."""
    u = _basis()
    diff = _build_structural_details(
        ["a0"],
        u[[0]],
        ["b0"],
        u[[1]],  # orthogonal - cost sqrt(2), above NO_COUNTERPART_COST
        anisotropy=False,
        sections_a=["X"],
        sections_b=["Z"],
    )
    (row,) = diff["sections"]
    assert row["section"] == "X" and row["n_statements"] == 1
    assert row["mean_cost"] == pytest.approx(np.sqrt(2.0), abs=1e-3)
    assert row["share_below_changed"] == 0.0
    assert row["median_displacement_norm"] is None
    assert row["share_target_other_section"] is None


# --- source map: section on a/b records ----------------------------------------------------------


def test_source_map_records_carry_section():
    sa, ea = ["a0"], _emb(1, seed=4)
    sb, eb = ["b0"], _emb(1, seed=5)
    ss, es = ["s0", "s1"], _emb(2, seed=6)
    m = _build_source_map(
        sa, ea, sb, eb, ss, es, anisotropy=False, top_k=1, sections_a=["X"], sections_b=["Y"]
    )
    assert m["a"][0]["section"] == "X"
    assert m["b"][0]["section"] == "Y"
    m_default = _build_source_map(sa, ea, sb, eb, ss, es, anisotropy=False, top_k=1)
    assert m_default["a"][0]["section"] == "" and m_default["b"][0]["section"] == ""


# --- end to end through the pipeline (fake segmenter + encoder, two-section documents) -----------

DOC_A = (
    "## Alpha\nThe cat sat still.\nThe dog ran fast.\n## Beta\nBirds fly high.\nFish swim deep.\n"
)
# same statements, but "Fish swim deep." moved from Beta into Alpha
DOC_B = (
    "## Alpha\nThe cat sat still.\nThe dog ran fast.\nFish swim deep.\n## Beta\nBirds fly high.\n"
)


def _vec(text: str, dim: int = 32) -> np.ndarray:
    seed = int(hashlib.sha1(text.encode()).hexdigest(), 16) % (2**32)
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class _LineSegmenter:
    """One statement per non-empty line - keeps the crafted documents' heading layout intact."""

    def __init__(self, offline: bool = True):
        pass

    def split(self, text: str) -> list[str]:
        return [ln.strip() for ln in text.splitlines() if ln.strip()]


class _FakeEncoder:
    def encode(self, statements: list[str]) -> np.ndarray:
        return np.stack([_vec(s) for s in statements])


@pytest.fixture(autouse=True)
def _wired(monkeypatch):
    """Mark the wmd mode ready and swap in the fake segmenter + encoder for every test here."""
    settings.reset()
    settings.mark_ready("wmd")
    monkeypatch.setattr(pipeline, "Segmenter", _LineSegmenter)
    monkeypatch.setattr(
        pipeline, "load_encoder", lambda backend, offline=True, device=None: _FakeEncoder()
    )
    yield
    settings.reset()


def test_e2e_semantic_details_carry_heading_paths_and_aggregates():
    dd = DocDistance()
    _, details = dd.semantic_distance_with_details(DOC_A, DOC_B)
    by_text = {f["text"]: f for f in details["flows"]}
    assert by_text["## Alpha"]["section"] == "Alpha"
    assert by_text["The cat sat still."]["section"] == "Alpha"
    assert by_text["Fish swim deep."]["section"] == "Beta"
    aggs = {row["section"]: row for row in details["sections"]}
    assert set(aggs) == {"Alpha", "Beta"}
    assert aggs["Alpha"]["n_statements"] == 3  # heading + two prose statements
    assert aggs["Beta"]["n_statements"] == 3
    # every A statement has an identical twin in B - zero cost, all below the changed threshold
    assert aggs["Alpha"]["share_below_changed"] == 1.0
    assert aggs["Beta"]["share_below_changed"] == 1.0
    assert aggs["Alpha"]["mean_cost"] < DIFF_CHANGED_COST


def test_e2e_structural_details_localize_the_cross_section_move():
    dd = DocDistance()
    _, details = dd.structural_distance_with_details(DOC_A, DOC_B)
    by_text = {r["text"]: r for r in details["statements"]}
    fish = by_text["Fish swim deep."]
    assert fish["section"] == "Beta"
    assert fish["target_section"] == "Alpha"  # its identical twin sits under Alpha in B
    assert by_text["Birds fly high."]["section"] == "Beta"
    assert by_text["Birds fly high."]["target_section"] == "Beta"
    aggs = {row["section"]: row for row in details["sections"]}
    assert aggs["Alpha"]["share_target_other_section"] == 0.0
    assert aggs["Beta"]["share_target_other_section"] == pytest.approx(1.0 / 3.0, abs=1e-3)
