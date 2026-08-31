"""Offline tests for markdown-aware segmentation (ACC-SEGM-1) and the fragment filter (ACC-SEGM-2).

The line scanner (``encoders.split_markdown``) and the filter (``pipeline._filter_fragments``) are
pure functions, exercised directly. The pipeline and CLI paths run with the same fake segmenter +
encoder pattern as ``test_e2e.py`` (fake split on the full stop, deterministic unit vectors), so no
model loads and the suite stays offline in the uv ``.venv``.
"""

import hashlib
import json
from typing import ClassVar

import numpy as np
import pytest
from typer.testing import CliRunner

from docdistance import pipeline, settings
from docdistance.cli import app
from docdistance.encoders import split_markdown
from docdistance.pipeline import DocDistance, _filter_fragments, semantic_distance

runner = CliRunner()
_WIDE = {"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"}

MD_DOC = (
    "## Results\n"
    "\n"
    "The model performed well. It exceeded the baseline.\n"
    "\n"
    "- first item\n"
    "- second item\n"
    "\n"
    "| name | value |\n"
    "|------|-------|\n"
    "| alpha | 1 |\n"
    "| beta | 2 |\n"
    "\n"
    "```python\n"
    "x = 1\n"
    "```\n"
)

# under the default (fake SaT) segmentation this yields two table fragments under 4 words
DOC_FRAG = "The cat sat on the mat. | 7 |. Joint |. Birds fly high in the sky."
DOC_CLEAN = "The cat sat on the mat. Birds fly high in the sky."


def _vec(text: str, dim: int = 32) -> np.ndarray:
    seed = int(hashlib.sha1(text.encode()).hexdigest(), 16) % (2**32)
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _prose_split(text: str) -> list[str]:
    return [s.strip() for s in text.replace("\n", ". ").split(".") if s.strip()]


class _FakeSegmenter:
    def __init__(self, offline: bool = True):
        pass

    def split(self, text: str) -> list[str]:
        return _prose_split(text)


class _CapturingEncoder:
    """Records every batch it encodes, so a test can assert what text was actually embedded."""

    seen: ClassVar[list[list[str]]] = []

    def encode(self, statements: list[str]) -> np.ndarray:
        _CapturingEncoder.seen.append(list(statements))
        return np.stack([_vec(s) for s in statements])


def _fake_load_encoder(backend, offline=True, device=None):
    return _CapturingEncoder()


@pytest.fixture(autouse=True)
def _wired(monkeypatch):
    """Mark the wmd mode ready and swap in the fake segmenter + capturing encoder."""
    settings.reset()
    settings.mark_ready("wmd")
    monkeypatch.setattr(pipeline, "Segmenter", _FakeSegmenter)
    monkeypatch.setattr(pipeline, "load_encoder", _fake_load_encoder)
    _CapturingEncoder.seen = []
    yield
    settings.reset()


# ------------------------------------------------------------------ ACC-SEGM-1: markdown scanner


def test_split_markdown_one_statement_per_structural_line():
    """One statement per heading, table row and list item; markers stripped from the embedded text,
    the raw line reported; the table separator row and the fence delimiters carry no content."""
    reported, embedded = split_markdown(MD_DOC, _prose_split)
    assert reported == [
        "## Results",
        "The model performed well",
        "It exceeded the baseline",
        "- first item",
        "- second item",
        "| name | value |",
        "| alpha | 1 |",
        "| beta | 2 |",
        "x = 1",
    ]
    assert embedded == [
        "Results",
        "The model performed well",
        "It exceeded the baseline",
        "first item",
        "second item",
        "name value",
        "alpha 1",
        "beta 2",
        "x = 1",
    ]


def test_split_markdown_only_prose_reaches_the_prose_splitter():
    calls: list[str] = []

    def spy(text: str) -> list[str]:
        calls.append(text)
        return _prose_split(text)

    split_markdown(MD_DOC, spy)
    assert calls == ["The model performed well. It exceeded the baseline."]


def test_split_markdown_numbered_list_and_tilde_fence():
    text = "1. first step\n2) second step\n\n~~~bash\n# a comment\n~~~\n"
    reported, embedded = split_markdown(text, _prose_split)
    assert reported == ["1. first step", "2) second step", "# a comment"]
    assert embedded == ["first step", "second step", "# a comment"]


def test_embed_statements_markdown_reports_raw_and_embeds_stripped():
    """The pipeline boundary: reported texts keep the raw lines, the encoder sees the stripped ones."""
    dd = DocDistance()
    statements, emb = dd.embed_statements(MD_DOC, markdown=True)
    assert statements[0] == "## Results" and statements[3] == "- first item"
    assert "| alpha | 1 |" in statements
    embedded = _CapturingEncoder.seen[0]
    assert embedded[0] == "Results" and embedded[3] == "first item"
    assert "alpha 1" in embedded and "|------|-------|" not in " ".join(embedded)
    assert emb.shape[0] == len(statements) == len(embedded)


def test_embed_statements_default_segmentation_unchanged():
    """Without the markdown option the segmenter gets the flattened body whole - reported text is
    embedded text, exactly the fake SaT split."""
    dd = DocDistance()
    statements, _ = dd.embed_statements(MD_DOC)
    assert statements == _prose_split(MD_DOC.strip())
    assert _CapturingEncoder.seen[0] == statements  # no reported/embedded split by default


def test_cli_markdown_flag_runs_and_details_report_raw_lines(tmp_path):
    out = tmp_path / "content.json"
    res = runner.invoke(
        app,
        ["distance-semantic", MD_DOC, MD_DOC, "--markdown", "--details-json", str(out)],
        env=_WIDE,
    )
    assert res.exit_code == 0, res.output
    details = json.loads(out.read_text())
    texts = [f["text"] for f in details["flows"]]
    assert "## Results" in texts and "| alpha | 1 |" in texts  # raw lines reported
    assert details["smd"] == pytest.approx(0.0, abs=2e-3)  # identical docs


# ------------------------------------------------------------------ ACC-SEGM-2: fragment filter


def test_filter_fragments_noop_at_defaults():
    reported = ["The cat sat on the mat", "| 7 |"]
    kept_r, kept_e, n = _filter_fragments(reported, reported)
    assert kept_r is reported and kept_e is reported and n == 0


def test_filter_fragments_min_words_and_table_rules():
    reported = ["The cat sat on the mat", "| 7 |", "-", "Joint | note", "short one"]
    kept_r, _kept_e, n = _filter_fragments(reported, reported, skip_table_fragments=True)
    assert kept_r == ["The cat sat on the mat", "short one"] and n == 3
    kept_r, _kept_e, n = _filter_fragments(reported, reported, min_words=3)
    assert kept_r == ["The cat sat on the mat", "| 7 |", "Joint | note"] and n == 2
    kept_r, _kept_e, n = _filter_fragments(
        reported, reported, min_words=3, skip_table_fragments=True
    )
    assert kept_r == ["The cat sat on the mat"] and n == 4


def test_cli_min_words_drops_fragments_and_records_them_in_the_header(tmp_path):
    """ACC-SEGM-2 acceptance path: --min-words 4 on a fixture with table fragments - the fragments
    are dropped, n_dropped and the rules land in the header, and the smd is computed over the
    filtered set (here the filtered A equals B, so smd ~ 0)."""
    out = tmp_path / "content.json"
    res = runner.invoke(
        app,
        [
            "distance-semantic",
            DOC_FRAG,
            DOC_CLEAN,
            "--min-words",
            "4",
            "--details-json",
            str(out),
        ],
        env=_WIDE,
    )
    assert res.exit_code == 0, res.output
    details = json.loads(out.read_text())
    assert details["n_dropped"] == {"a": 2, "b": 0}  # the two table fragments
    assert details["drop_rules"] == {"min_words": 4, "skip_table_fragments": False}
    assert details["n_statements"] == {"a": 2, "b": 2}  # the filtered set
    texts = [f["text"] for f in details["flows"]]
    assert "| 7 |" not in texts and "Joint |" not in texts
    assert details["smd"] == pytest.approx(0.0, abs=2e-3)  # smd over the filtered set


def test_cli_json_carries_n_dropped_when_filter_active():
    res = runner.invoke(
        app, ["distance-semantic", DOC_FRAG, DOC_CLEAN, "--min-words", "4", "--json"], env=_WIDE
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["n_dropped"] == {"a": 2, "b": 0}
    assert payload["drop_rules"] == {"min_words": 4, "skip_table_fragments": False}
    assert payload["n_statements_a"] == 2  # scored over the filtered set


def test_cli_json_default_has_no_drop_keys():
    """Without an active drop rule the machine JSON stays exactly the result dict."""
    res = runner.invoke(app, ["distance-semantic", DOC_FRAG, DOC_CLEAN, "--json"], env=_WIDE)
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert "n_dropped" not in payload and "drop_rules" not in payload


def test_min_words_changes_smd_because_marginals_renormalise():
    """Documented behaviour: dropping fragments renormalises the 1/n marginals, so the SMD moves."""
    unfiltered = semantic_distance(DOC_FRAG, DOC_CLEAN)
    filtered = semantic_distance(DOC_FRAG, DOC_CLEAN, min_words=4)
    assert filtered.smd == pytest.approx(0.0, abs=2e-3)  # filtered A equals B
    assert unfiltered.smd > filtered.smd + 0.01  # fragment mass had to move somewhere


def test_all_statements_dropped_raises_loudly():
    dd = DocDistance()
    with pytest.raises(ValueError, match="dropped by the fragment filter"):
        dd.embed_statements("| 1 |. | 2 |", min_words=4)
