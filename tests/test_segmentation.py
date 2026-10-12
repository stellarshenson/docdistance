"""Offline tests for markdown-aware segmentation (ACC-SEGM-1, DEF-SEGM-15), the fragment filter
(ACC-SEGM-2), the marker-statement warning of the plain reader (DEF-SEGM-12), the HTML reader
(ACC-SEGM-85, ACC-SEGM-86, ACC-SEGM-96), the reader chosen by file extension (ACC-SEGM-93) and the
option ``input_format`` of ``embed_statements`` (ACC-SEGM-94, ACC-SEGM-95, ACC-SEGM-91).

The line scanner (``encoders.split_markdown``), the converter (``encoders.html_to_markdown``) and
the filter (``pipeline._filter_fragments``) are pure functions, exercised directly. The pipeline
and CLI paths run with the same fake segmenter + encoder pattern as ``test_e2e.py`` (fake split on
the full stop, deterministic unit vectors), so no model loads and the suite stays offline in the
uv ``.venv``.
"""

import hashlib
import json
import textwrap
from typing import ClassVar

from loguru import logger
import numpy as np
import pytest
from typer.testing import CliRunner

from docdistance import pipeline, settings
from docdistance.cli import app
from docdistance.distance import compute_distance
from docdistance.encoders import html_to_markdown, split_markdown
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

# 7 statements under the fake SaT split, 3 of them without a letter or a digit: the bare "-" twice
# and the bare "###". "42%" holds a digit and "## Key findings" holds letters, so neither counts.
# The markdown reader drops the 3 marker lines and returns the other 4 (DEF-SEGM-15)
DOC_MARKERS = "## Key findings\n-\nAdoption held steady. Barriers remain.\n###\n42%\n-\n"

# prose with one line that starts with a dash. The plain reader returns the bare "-" as a statement
# (1 of 3), the markdown reader drops it
DOC_ONE_DASH = "Adoption held steady.\n-\nBarriers remain.\n"

# one document written twice. Three heading levels, a paragraph that the HTML twin wraps over
# three lines and the markdown twin writes on one, a one-line paragraph, a nested list, an ordered
# list, a table and a code block with two "# " comment lines; the second one starts a line of the
# HTML text as well
TWIN_MD = (
    "# Adoption report\n"
    "\n"
    "## Findings\n"
    "\n"
    "Adoption held steady across the surveyed regions. Barriers remain in two of them.\n"
    "\n"
    "The second paragraph is one line. It names the cost.\n"
    "\n"
    "- first point\n"
    "  - nested point\n"
    "- second point\n"
    "\n"
    "1. first step\n"
    "2. second step\n"
    "\n"
    "### Numbers\n"
    "\n"
    "| region | share |\n"
    "|--------|-------|\n"
    "| north | 42% |\n"
    "| south | 17% |\n"
    "\n"
    "```bash\n"
    "# install the package\n"
    "pip install docdistance\n"
    "# then provision the models\n"
    "docdistance init\n"
    "```\n"
)
TWIN_HTML = (
    "<!DOCTYPE html>\n"
    "<html>\n"
    "<head><title>Adoption report</title><style>p { margin: 0 }</style></head>\n"
    "<body>\n"
    "<h1>Adoption report</h1>\n"
    "<h2>Findings</h2>\n"
    "<p>Adoption held steady across\n"
    "the surveyed regions. Barriers remain\n"
    "in two of them.</p>\n"
    "<p>The second paragraph is one line. It names the cost.</p>\n"
    "<ul>\n"
    "<li>first point\n"
    "<ul>\n"
    "<li>nested point</li>\n"
    "</ul>\n"
    "</li>\n"
    "<li>second point</li>\n"
    "</ul>\n"
    "<ol>\n"
    "<li>first step</li>\n"
    "<li>second step</li>\n"
    "</ol>\n"
    "<h3>Numbers</h3>\n"
    "<table>\n"
    "<thead>\n"
    "<tr><th>region</th><th>share</th></tr>\n"
    "</thead>\n"
    "<tbody>\n"
    "<tr><td>north</td><td>42%</td></tr>\n"
    "<tr><td>south</td><td>17%</td></tr>\n"
    "</tbody>\n"
    "</table>\n"
    '<pre><code class="language-bash"># install the package\n'
    "pip install docdistance\n"
    "# then provision the models\n"
    "docdistance init\n"
    "</code></pre>\n"
    "</body>\n"
    "</html>\n"
)
# the statements of the twin under the fake SaT split, with the title line
TWIN_STATEMENTS = [
    "# Adoption report",
    "## Findings",
    "Adoption held steady across the surveyed regions",
    "Barriers remain in two of them",
    "The second paragraph is one line",
    "It names the cost",
    "- first point",
    "  - nested point",
    "- second point",
    "1. first step",
    "2. second step",
    "### Numbers",
    "| region | share |",
    "| north | 42% |",
    "| south | 17% |",
    "# install the package",
    "pip install docdistance",
    "# then provision the models",
    "docdistance init",
]

# one text that the three readers split into three statement lists: the plain reader returns the
# bare "-" and the tags, the markdown reader drops the "-", the HTML reader drops both
EXTENSION_DOC = "## Findings\n\n<p>Adoption held steady.</p>\n\n-\n"
EXTENSION_STATEMENTS = {
    "plain": ["## Findings", "<p>Adoption held steady", "</p>", "-"],
    "markdown": ["## Findings", "<p>Adoption held steady", "</p>"],
    "html": ["## Findings", "Adoption held steady"],
}

# one document as HTML with every block on one line. Each block is longer than 72 characters, and
# the second paragraph holds a line break element
WRAP_HTML = (
    "<h2>Findings of the adoption survey across the regions that took part in it this year</h2>\n"
    "<p>Adoption held steady across the surveyed regions. Barriers remain in two of them, "
    "and the cost of the licences is the one that the respondents name most often.</p>\n"
    "<p>The second paragraph names the cost.<br>Its second line stands behind a line break "
    "element, which is the only line break that a paragraph of an HTML document holds.</p>\n"
    "<ul>\n"
    "<li>the first point of the list is long enough to be wrapped over two lines of the source "
    "text</li>\n"
    "</ul>\n"
)
WRAP_STATEMENTS = [
    "## Findings of the adoption survey across the regions that took part in it this year",
    (
        "Adoption held steady across the surveyed regions. Barriers remain in two of them, "
        "and the cost of the licences is the one that the respondents name most often."
    ),
    "The second paragraph names the cost.",
    (
        "Its second line stands behind a line break element, which is the only line break that "
        "a paragraph of an HTML document holds."
    ),
    "- the first point of the list is long enough to be wrapped over two lines of the source text",
]


def _vec(text: str, dim: int = 32) -> np.ndarray:
    seed = int(hashlib.sha1(text.encode()).hexdigest(), 16) % (2**32)
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _prose_split(text: str) -> list[str]:
    return [s.strip() for s in text.replace("\n", ". ").split(".") if s.strip()]


def _line_split(text: str) -> list[str]:
    """A segmenter that cuts at line breaks and nowhere else."""
    return [line.strip() for line in text.splitlines() if line.strip()]


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


@pytest.fixture
def logged():
    """The messages of the loguru warnings logged during the test, in order."""
    messages: list[str] = []
    sink_id = logger.add(lambda msg: messages.append(msg.record["message"]), level="WARNING")
    yield messages
    logger.remove(sink_id)


@pytest.fixture
def reader_lines():
    """The DEBUG lines that name the reader of a document, logged during the test, in order."""
    messages: list[str] = []
    sink_id = logger.add(
        lambda msg: messages.append(msg.record["message"]),
        level="DEBUG",
        filter=lambda record: " reader" in record["message"],
    )
    yield messages
    logger.remove(sink_id)


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


def test_split_markdown_drops_marker_only_lines():
    """DEF-SEGM-15: a bare heading marker, a bare bullet and a horizontal rule are no statements.

    The repro of the defect: the four marker lines came back as prose statements. Each marker line
    also ends the prose paragraph before it, so the prose splitter gets the two paragraphs in two
    calls.
    """
    calls: list[str] = []

    def spy(text: str) -> list[str]:
        calls.append(text)
        return _prose_split(text)

    text = "## Key findings\n-\nAdoption held steady.\n###\n---\nBarriers remain.\n* * *\n"
    reported, embedded = split_markdown(text, spy)
    assert reported == ["## Key findings", "Adoption held steady", "Barriers remain"]
    assert embedded == ["Key findings", "Adoption held steady", "Barriers remain"]

    for marker in ("#", "######", "-", "*", "+", "- ", "---", "***", "___", "- - -", "_ _ _ _"):
        calls.clear()
        assert split_markdown(f"First.\n{marker}\nSecond.", spy)[0] == ["First", "Second"], marker
        assert calls == ["First.", "Second."], marker


def test_split_markdown_keeps_lines_that_are_not_marker_only():
    """DEF-SEGM-15: what is dropped is the marker-only line, nothing beside it.

    A bare numbered marker stays prose, because "42." can be the last word of a wrapped paragraph.
    Two characters are no rule and neither are three different ones. A line inside a fenced block
    is code, also when it reads ``---``.
    """
    assert split_markdown("The total was\n42.\nNext.", _prose_split)[0] == [
        "The total was",
        "42",
        "Next",
    ]
    for line in ("--", "-*-"):
        assert split_markdown(f"First.\n{line}\nSecond.", _prose_split)[0] == [
            "First",
            line,
            "Second",
        ]
    assert split_markdown("```yaml\n---\nkey: 1\n```\n", _prose_split)[0] == ["---", "key: 1"]


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


# ------------------------------------------------------------------ DEF-SEGM-12: marker statements


def test_default_reader_warns_about_statements_without_a_letter_or_digit(logged):
    """DEF-SEGM-12: one warning names the count, the statement total and the two options that
    remove such statements. The statements and the embeddings are returned as before."""
    statements, emb = DocDistance().embed_statements(DOC_MARKERS)
    assert statements == _prose_split(DOC_MARKERS)  # the three markers are still statements
    assert _CapturingEncoder.seen == [statements]
    assert np.array_equal(emb, np.stack([_vec(s) for s in statements]))
    assert len(logged) == 1
    assert "3 of 7 statements" in logged[0]
    for option in ("--markdown", "markdown=True", "--min-words", "min_words"):
        assert option in logged[0]


def test_default_reader_does_not_warn_when_every_statement_holds_a_letter_or_digit(logged):
    """DEF-SEGM-12: a statement of digits and punctuation only ("42%") is not a marker."""
    DocDistance().embed_statements("42%. " + DOC_CLEAN)
    assert logged == []


def test_markdown_reader_does_not_warn_about_marker_statements(logged):
    """DEF-SEGM-12 / DEF-SEGM-15: the warning belongs to the default reader. With markdown on the
    three markers are dropped, so the remedy the warning names removes them, and nothing is logged."""
    statements, _ = DocDistance().embed_statements(DOC_MARKERS, markdown=True)
    assert statements == ["## Key findings", "Adoption held steady", "Barriers remain", "42%"]
    assert logged == []


def test_marker_warning_names_the_file_of_a_file_document(logged, tmp_path):
    """DEF-SEGM-12: the line starts with the path as given when the document is a file, a ``str``
    or a ``Path``, so a ``--pairs`` run says which document it concerns. Raw text gets no name.
    The file is read with ``input_format="plain"``: the default reader of a ``.md`` file is the
    markdown reader, which logs no such warning."""
    doc = tmp_path / "findings.md"
    doc.write_text(DOC_MARKERS, encoding="utf-8")
    dd = DocDistance()
    dd.embed_statements(str(doc), input_format="plain")
    dd.embed_statements(doc, input_format="plain")
    dd.embed_statements(DOC_MARKERS)
    assert logged[0].startswith(f"{doc}: 3 of 7 statements hold no letter")
    assert logged[1] == logged[0]
    assert logged[2] == logged[0].removeprefix(f"{doc}: ")


def test_marker_warning_counts_the_statements_left_by_the_fragment_filter(logged):
    """DEF-SEGM-12: the count is taken on the statements returned. ``min_words=2`` removes every
    one-word statement and nothing is logged; ``skip_table_fragments`` removes the two "-" and
    leaves "###", 1 of 5."""
    dd = DocDistance()
    dd.embed_statements(DOC_MARKERS, min_words=2)
    assert logged == []
    dd.embed_statements(DOC_MARKERS, skip_table_fragments=True)
    assert len(logged) == 1 and "1 of 5 statements" in logged[0]


def test_marker_warning_is_logged_once_per_document_and_changes_no_distance(logged):
    """DEF-SEGM-12: a scoring call logs one warning for each document that holds such statements,
    and the distance is the one the returned statements give."""
    longer = DOC_MARKERS + "Snow fell overnight."  # 3 of 8
    result = semantic_distance(DOC_MARKERS, longer)
    assert [w.split(" statements")[0] for w in logged] == ["3 of 7", "3 of 8"]
    emb_a = np.stack([_vec(s) for s in _prose_split(DOC_MARKERS)])
    emb_b = np.stack([_vec(s) for s in _prose_split(longer)])
    assert result == compute_distance(emb_a, emb_b)

    logged.clear()
    semantic_distance(DOC_MARKERS, DOC_CLEAN)
    assert len(logged) == 1 and "3 of 7 statements" in logged[0]


# ------------------------------------------------------------------ ACC-SEGM-85: HTML reader


def test_html_headings_become_atx_lines():
    """``h1`` to ``h6``: the level in hashes, a space and the text on one line with single spaces."""
    html = (
        "".join(f"<h{n}>Level {n}</h{n}>" for n in range(1, 7)) + "<h2>\n  Key\n  findings </h2>"
    )
    assert html_to_markdown(html).splitlines() == [
        "# Level 1",
        "## Level 2",
        "### Level 3",
        "#### Level 4",
        "##### Level 5",
        "###### Level 6",
        "## Key findings",
    ]


def test_html_list_items_become_dash_lines():
    """``li`` of a ``ul``: a dash, a space and the text on one line. ``li`` of an ``ol``: its
    number in place of the dash."""
    html = "<ul><li>first</li><li> second\n item </li></ul><ol><li>third</li></ol>"
    assert html_to_markdown(html) == "- first\n- second item\n1. third"


def test_html_ordered_list_items_are_numbered_from_1_in_each_list():
    """``li`` of an ``ol``: its number, a full stop, a space and the text, as a markdown ordered
    list writes it. Each ``ol`` counts its own items from 1, a nested one as well, and the
    ``start`` attribute is not read. An item without text takes its number and writes no line."""
    html = (
        "<ol><li>one</li><li><b>two</b><ol><li>inner one</li><li>inner two</li></ol></li>"
        "<li>three<ul><li>bullet</li></ul></li><li> </li><li>five</li></ol>"
        '<ol start="7"><li>one again</li></ol>'
    )
    assert html_to_markdown(html).splitlines() == [
        "1. one",
        "2. two",
        "  1. inner one",
        "  2. inner two",
        "3. three",
        "  - bullet",
        "5. five",
        "1. one again",
    ]


def test_html_nested_list_items_are_indented_by_their_depth():
    """A nested ``li`` is its own line, two spaces deeper per list around it, as a markdown
    nested list writes it. The text of the parent item ends where the nested list starts."""
    html = (
        "<ul><li>Parent<ul><li>Child<ol><li>Grandchild</li></ol></li></ul></li>"
        "<li>Sibling</li></ul>"
    )
    assert html_to_markdown(html) == "- Parent\n  - Child\n    1. Grandchild\n- Sibling"


def test_html_table_rows_become_pipe_rows():
    """``tr``: one pipe row from its ``td`` texts. A pipe inside a cell becomes a space and the
    whitespace of a cell collapses."""
    html = (
        "<table><tr><td>alpha</td><td>1</td></tr>"
        "<tr><td>a|b</td><td> two\n words </td></tr></table>"
    )
    assert html_to_markdown(html) == "| alpha | 1 |\n| a b | two words |"


def test_html_table_with_header_cells():
    """``th`` cells are written like ``td`` cells; no separator row is written."""
    html = (
        "<table><thead><tr><th>name</th><th>value</th></tr></thead>"
        "<tbody><tr><td>alpha</td><td>1</td></tr></tbody></table>"
    )
    assert html_to_markdown(html) == "| name | value |\n| alpha | 1 |"


def test_html_pre_becomes_a_fenced_block():
    """``pre``: a fence, the lines of its text with their indentation, a fence."""
    html = (
        '<pre><code class="language-bash"># install\npip install docdistance\n'
        "    --upgrade\n</code></pre>"
    )
    assert html_to_markdown(html) == "```\n# install\npip install docdistance\n    --upgrade\n```"
    assert html_to_markdown("<pre>\nx = 1\n\ny = 2\n\n</pre>") == "```\nx = 1\n\ny = 2\n```"


@pytest.mark.parametrize(
    "tag",
    ["p", "div", "blockquote", "section", "article", "dt", "dd", "figcaption", "caption"]
    + ["header", "footer", "main", "nav", "aside", "address", "form"]
    + ["figure", "details", "summary"],
)
def test_html_paragraph_tags_become_paragraphs(tag):
    """Each such block is a paragraph followed by a blank line, and so is the text before, between
    and after the blocks."""
    html = f"lead<{tag}>one</{tag}><{tag}>two</{tag}>tail"
    assert html_to_markdown(html) == "lead\n\none\n\ntwo\n\ntail"


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        (
            "<header>Site</header><nav>Home</nav><main>Body text</main><footer>Foot</footer>",
            "Site\n\nHome\n\nBody text\n\nFoot",
        ),
        (
            "<details><summary>Sum</summary>Hidden text</details><address>Addr</address>",
            "Sum\n\nHidden text\n\nAddr",
        ),
        (
            (
                "<body><noscript>Enable JS</noscript><svg><title>icon</title>"
                "<text>svgtext</text></svg><p>real</p></body>"
            ),
            "real",
        ),
    ],
    ids=["page regions", "details and address", "noscript and svg"],
)
def test_html_adjacent_page_blocks_do_not_join_into_one_word(html, expected):
    """The blocks of a page stand without whitespace between them here. Each is a paragraph of
    its own, and the text of ``noscript`` and of ``svg`` is no text of the document."""
    assert html_to_markdown(html) == expected


def test_html_text_outside_any_block_is_a_paragraph():
    assert html_to_markdown("Just text.\nSecond line.") == "Just text. Second line."


def test_html_paragraph_keeps_the_line_breaks_of_its_source_text():
    """ACC-SEGM-96 reversed what the name of this test says: a paragraph is written on one line.
    Every run of whitespace of its source text, line breaks included, is one space, so the prose
    segmenter, which cuts at every line break, reads a wrapped paragraph as the same text on one
    line."""
    html = "<p>  The model performed\n\n   well. It exceeded\r\nthe\tbaseline.  </p>"
    assert html_to_markdown(html) == "The model performed well. It exceeded the baseline."


def test_html_br_is_the_only_line_break_inside_a_paragraph():
    """ACC-SEGM-96: a ``br`` breaks the line of a paragraph whatever stands around it in the
    source text. A line that two of them leave empty is removed, and a paragraph that holds
    nothing else writes no line."""
    assert html_to_markdown("<p>one\nmore<br>\n  two\nmore</p>") == "one more\ntwo more"
    assert html_to_markdown("<p>one<br> <br>two</p>") == "one\ntwo"
    assert html_to_markdown("<p>one</p><p> <br>\n</p><p>two</p>") == "one\n\ntwo"
    assert html_to_markdown("one<br>two\nthree") == "one\ntwo three"  # text outside any block


def test_html_pre_keeps_the_line_breaks_of_its_source_text():
    """ACC-SEGM-96 concerns paragraphs: the text of a ``pre`` keeps its lines, and a ``br`` adds
    one."""
    assert html_to_markdown("<pre>x = 1\ny = 2<br>z = 3</pre>") == "```\nx = 1\ny = 2\nz = 3\n```"


def test_html_wrapped_at_72_columns_reads_the_statements_of_one_line_per_paragraph():
    """ACC-SEGM-96: one document as HTML with every block on one line and with its source text
    wrapped at 72 columns. The conversions are equal, so a segmenter that cuts at every line
    break returns equal statements; the review read 37 against 10 statements for such a pair."""
    wrapped = "\n".join(
        textwrap.fill(line, 72, break_long_words=False, break_on_hyphens=False)
        for line in WRAP_HTML.splitlines()
    )
    assert (
        max(map(len, wrapped.splitlines()))
        <= 72
        < min(len(line) for line in WRAP_HTML.splitlines() if " " in line)
    )
    assert len(wrapped.splitlines()) == 12 and len(WRAP_HTML.splitlines()) == 6

    assert html_to_markdown(wrapped) == html_to_markdown(WRAP_HTML)
    reported, _ = split_markdown(html_to_markdown(wrapped), _line_split)
    assert reported == WRAP_STATEMENTS
    assert split_markdown(html_to_markdown(WRAP_HTML), _line_split)[0] == WRAP_STATEMENTS

    dd = DocDistance()
    one_line, _ = dd.embed_statements(WRAP_HTML, input_format="html")
    assert dd.embed_statements(wrapped, input_format="html")[0] == one_line
    assert len(one_line) == 6  # the fake segmenter also cuts at the full stop inside a line


def test_html_br_breaks_the_line_and_hr_breaks_the_paragraph():
    assert html_to_markdown("<p>one<br>two</p>") == "one\ntwo"
    assert html_to_markdown("<h2>Key<br/>findings</h2>") == "## Key findings"
    assert html_to_markdown("one<hr>two") == "one\n\ntwo"


@pytest.mark.parametrize("tag", ["script", "style", "template", "noscript", "svg", "title"])
def test_html_dropped_elements_leave_no_text(tag):
    """``script``, ``style``, ``template``, ``noscript``, ``svg`` and ``title`` are dropped with
    what they hold. The ``title`` here stands outside a ``head``: a document may omit that
    tag."""
    html = f"<p>before</p><{tag}>dropped <b>text</b></{tag}><p>after</p>"
    assert html_to_markdown(html) == "before\n\nafter"


@pytest.mark.parametrize("tag", ["template"])
def test_html_dropped_element_ends_at_its_own_end_tag(tag):
    """A ``script`` or a ``style`` inside a ``template`` does not end the dropped region: the
    text after it is dropped as well."""
    html = f"<{tag}>a<script>var b;</script>b<style>p {{}}</style>c</{tag}><p>after</p>"
    assert html_to_markdown(html) == "after"


def test_html_document_without_the_head_end_tag_and_the_body_tag_keeps_its_text():
    """A document may omit ``</head>`` and ``<body>``. ``head`` is no dropped element: when it
    was one, the whole document stood inside the dropped region and the conversion was the empty
    string. The ``title`` is dropped by itself."""
    html = (
        "<!DOCTYPE html><html><head><title>T</title><meta charset=utf-8><h1>Title</h1><p>Text</p>"
    )
    assert html_to_markdown(html) == "# Title\nText"


def test_html_body_tag_inside_a_dropped_element_ends_no_dropped_region():
    """A ``body`` tag inside a dropped element is like every other tag there: it changes nothing.
    Before, it ended every open dropped region, which a ``head`` without its end tag needed."""
    html = "<p>before</p><template><body><p>dropped</p></body></template><p>after</p>"
    assert html_to_markdown(html) == "before\n\nafter"


def test_html_tags_inside_a_dropped_element_change_nothing():
    """A start or an end tag inside a dropped element neither ends the open block nor opens one:
    the paragraph around the ``template`` stays one paragraph."""
    html = (
        "<p>one <template><h2>x</h2><ul><li>y</li></ul><table><tr><td>z</td></tr></table>"
        "<br></template>two</p>"
    )
    assert html_to_markdown(html) == "one two"


def test_html_empty_heading_and_empty_list_item_write_no_line():
    assert html_to_markdown("<h2> </h2><ul><li></li><li>a</li><li>\n</li></ul>") == "- a"


def test_html_text_that_ends_in_an_open_reference_is_kept():
    """The parser holds back a text that ends in ``&`` and a word until it is closed."""
    assert html_to_markdown("<p>kept</p>AT&T") == "kept\n\nAT&T"


def test_html_comments_and_the_document_frame_leave_no_text():
    html = (
        "<!DOCTYPE html><html><head><title>Title</title></head><body>"
        "<!-- a comment <p>not text</p> --><p>kept</p></body></html>"
    )
    assert html_to_markdown(html) == "kept"


def test_html_inline_tags_leave_their_text_and_img_leaves_nothing():
    html = (
        '<p>The <strong>model</strong> <em>performed</em> <a href="x.html">well</a><sup>1</sup> '
        '<span id="s">here</span>: <code>x = 1</code><img src="i.png" alt="a picture">.</p>'
    )
    assert html_to_markdown(html) == "The model performed well1 here: x = 1."


def test_html_character_references_are_decoded():
    html = "<p>Tom &amp; Jerry &lt;3 &#8211; caf&eacute; &#x41;</p><ul><li>a &gt; b</li></ul>"
    assert html_to_markdown(html) == "Tom & Jerry <3 – café A\n\n- a > b"


def test_html_list_item_and_cell_keep_the_paragraphs_inside_them():
    """A ``p`` inside an ``li`` is text of that item; every block inside a cell is text of that
    cell. A block edge is a space there."""
    loose = "<ul><li><p>loose item</p></li><li><p>one</p><p>two</p></li></ul>"
    assert html_to_markdown(loose) == "- loose item\n- one two"
    cells = (
        "<table><tr><td><p>a</p><p>b</p></td>"
        "<td><ul><li>c</li><li>d</li></ul>e</td><td><h3>f</h3><pre>g</pre></td></tr></table>"
    )
    assert html_to_markdown(cells) == "| a b | c d e | f g |"


def test_html_pre_and_heading_inside_a_list_item_start_their_own_block():
    html = "<ul><li>Run:<pre>pip install x</pre></li><li>Then<h3>Check</h3></li></ul>"
    assert html_to_markdown(html) == "- Run:\n```\npip install x\n```\n- Then\n### Check"


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ("<p>one<p>two<h2>Title<p>three", "one\n\ntwo\n\n## Title\nthree"),
        ("<ul><li>a<li>b</ul><p>c</p>", "- a\n- b\nc"),
        ("<table><tr><td>a<td>b<tr><td>c</table><p>d</p>", "| a | b |\n| c |\nd"),
        ("<td>a</td><td>b</td>", "| a | b |"),
        ("<tr><td>a</td></tr><p>after</p>", "| a |\nafter"),
        ("<li>a</li>", "- a"),
        ("</script></li></p></td></tr></ul><p>a</p>", "a"),
        ("</ol><ul><li>a<ul><li>b</li></ul></li></ul>", "- a\n  - b"),
        ("<html><head><title>T</title><body><p>a</p>", "a"),
    ],
    ids=[
        "unclosed p and heading",
        "unclosed li",
        "unclosed td and tr",
        "cells outside a row",
        "row outside a table",
        "item outside a list",
        "end tags without a start tag",
        "list end tag without a start tag",
        "head that is not closed",
    ],
)
def test_html_malformed_input_is_read_without_an_error(html, expected):
    """An unclosed block ends at the next block or at the end of the text."""
    assert html_to_markdown(html) == expected


@pytest.mark.parametrize(
    "tail",
    ["<p", '<a href="x', "<!-- open comment", "</", "<![CDATA[ x", "<![x y", "&#xZZ; &bogus"],
)
def test_html_truncated_input_is_read_without_an_error(tail):
    """A text that ends inside a tag, a comment, a section or a reference raises nothing; what
    stands before that construct is converted."""
    assert html_to_markdown("<p>kept</p>" + tail).startswith("kept")


def test_html_byte_order_mark_is_removed():
    """U+FEFF is no text of the document: written as the character in front of the document or
    as a character reference, it leaves nothing in the converted text."""
    html = "\ufeff<!DOCTYPE html>\n<html><body>\n<h2>Key&#xFEFF; findings</h2>\n<p>kept</p>\n"
    assert html_to_markdown(html) == "## Key findings\nkept"


# ------------------------------------------------------------------ ACC-SEGM-86: HTML twin


def test_html_twin_splits_into_the_statements_of_the_markdown_twin():
    """The twin property: the converted HTML twin and the markdown twin give the markdown scanner
    the same reported lines, the same embedded texts and the same prose paragraphs."""
    calls: dict[str, list[str]] = {"html": [], "md": []}

    def spy(name: str):
        return lambda text: calls[name].append(text) or _prose_split(text)

    from_html = split_markdown(html_to_markdown(TWIN_HTML), spy("html"))
    from_md = split_markdown(TWIN_MD, spy("md"))
    assert from_html == from_md
    assert from_md[0] == TWIN_STATEMENTS
    assert calls["html"] == calls["md"]
    assert calls["md"] == [
        "Adoption held steady across the surveyed regions. Barriers remain in two of them.",
        "The second paragraph is one line. It names the cost.",
    ]


def test_embed_statements_html_returns_the_statements_of_the_markdown_twin():
    """``input_format="html"`` on the HTML twin against ``markdown=True`` on the markdown twin, both
    as raw text: the statements, the texts that reach the encoder and the vectors are equal."""
    dd = DocDistance()
    from_md, emb_md = dd.embed_statements(TWIN_MD, markdown=True)
    from_html, emb_html = dd.embed_statements(TWIN_HTML, input_format="html")
    assert from_html == from_md == TWIN_STATEMENTS
    assert _CapturingEncoder.seen[0] == _CapturingEncoder.seen[1]
    assert np.array_equal(emb_html, emb_md)


def test_embed_statements_html_file_drops_the_title_like_a_markdown_file(tmp_path):
    """The twins as files. The title is no statement of a file: the ``h1`` is removed like the
    ``# `` line of the markdown file. The two ``# `` lines inside the ``pre`` block are code and
    stay, like the ones inside the fence."""
    md, html = tmp_path / "twin.md", tmp_path / "twin.html"
    md.write_text(TWIN_MD, encoding="utf-8")
    html.write_text(TWIN_HTML, encoding="utf-8")
    dd = DocDistance()
    from_md, emb_md = dd.embed_statements(md, markdown=True)
    from_html, emb_html = dd.embed_statements(str(html), input_format="html")
    assert from_html == from_md == TWIN_STATEMENTS[1:]
    assert {"# install the package", "# then provision the models"} <= set(from_html)
    assert np.array_equal(emb_html, emb_md)


@pytest.mark.parametrize("value", ["html", "auto"])
def test_embed_statements_html_file_with_a_byte_order_mark_gains_no_statement(value, tmp_path):
    """An HTML file that starts with a byte order mark: no statement holds U+FEFF. Before, the
    mark was a paragraph of its own and came back as the first statement."""
    doc = tmp_path / "bom.html"
    doc.write_text(
        "\ufeff<!DOCTYPE html>\n<html><head><title>t</title></head>\n<body>\n<h1>Title</h1>\n"
        "<p>One sentence</p>\n</body></html>\n",
        encoding="utf-8",
    )
    assert DocDistance().embed_statements(doc, input_format=value)[0] == ["One sentence"]


@pytest.mark.parametrize("as_files", [False, True], ids=["raw text", "files"])
def test_details_of_the_html_twin_carry_the_sections_of_the_markdown_twin(as_files, tmp_path):
    """The section scan reads the converted text, so both detail dicts of the HTML twins equal the
    ones of the markdown twins key for key, section fields included. The twins as files are also
    read without ``input_format``: the default reader follows the extension of each file."""
    md_a, md_b = TWIN_MD, TWIN_MD.replace("- second point\n", "")
    html_a, html_b = TWIN_HTML, TWIN_HTML.replace("<li>second point</li>\n", "")
    if as_files:
        docs = {"a.md": md_a, "b.md": md_b, "a.html": html_a, "b.html": html_b}
        for name, text in docs.items():
            (tmp_path / name).write_text(text, encoding="utf-8")
        md_a, md_b, html_a, html_b = (str(tmp_path / name) for name in docs)
    # the title line of a file is removed, so it heads no section there
    sections = ["Findings", "Findings > Numbers"]
    if not as_files:
        sections = ["Adoption report", *(f"Adoption report > {name}" for name in sections)]

    dd = DocDistance()
    for call in (dd.semantic_distance_with_details, dd.structural_distance_with_details):
        from_md = call(md_a, md_b, markdown=True)
        assert call(html_a, html_b, input_format="html") == from_md
        if as_files:  # raw text has no extension and is read by the plain reader
            assert call(html_a, html_b) == from_md
            assert call(md_a, md_b) == from_md

    _, semantic = dd.semantic_distance_with_details(html_a, html_b, input_format="html")
    assert [row["section"] for row in semantic["sections"]] == sections
    assert sorted({target["section"] for target in semantic["targets"]}) == sections
    _, structural = dd.structural_distance_with_details(html_a, html_b, input_format="html")
    assert sorted({record["target_section"] for record in structural["statements"]}) == sections


# ------------------------------------------------- ACC-SEGM-93, ACC-SEGM-94: input_format


def test_embed_statements_markdown_flag_equals_input_format_markdown():
    dd = DocDistance()
    by_flag, emb_flag = dd.embed_statements(MD_DOC, markdown=True)
    by_value, emb_value = dd.embed_statements(MD_DOC, input_format="markdown")
    by_both, _ = dd.embed_statements(MD_DOC, markdown=True, input_format="markdown")
    assert by_value == by_flag == by_both
    assert by_flag != dd.embed_statements(MD_DOC)[0]  # raw text: the plain reader by default
    assert _CapturingEncoder.seen[0] == _CapturingEncoder.seen[1]
    assert np.array_equal(emb_value, emb_flag)


def test_embed_statements_input_format_plain_is_the_default_reader():
    """The plain reader is the default reader of raw text, which has no file extension."""
    dd = DocDistance()
    assert dd.embed_statements(MD_DOC, input_format="plain")[0] == dd.embed_statements(MD_DOC)[0]
    assert _CapturingEncoder.seen[0] == _CapturingEncoder.seen[1] == _prose_split(MD_DOC.strip())


@pytest.mark.parametrize("as_files", [False, True], ids=["raw text", "files"])
def test_embed_statements_auto_chooses_the_reader_per_document(as_files, tmp_path, reader_lines):
    """``auto`` reads a ``.html`` file with the HTML reader, a ``.md`` file with the markdown
    reader and a ``.txt`` file with the plain reader. The same three texts as raw text name no
    file: the plain reader reads each of them. A call without ``input_format`` returns the same
    statements. The reader that the same text gets as the other kind of argument returns other
    statements, and one DEBUG line per document names the reader, for a file also that its
    extension chose it."""
    docs = {"html": TWIN_HTML, "markdown": TWIN_MD, "plain": DOC_ONE_DASH}
    names = {"html": "twin.html", "markdown": "twin.md", "plain": "notes.txt"}
    if as_files:
        for reader, text in docs.items():
            (tmp_path / names[reader]).write_text(text, encoding="utf-8")
        docs = {reader: str(tmp_path / names[reader]) for reader in docs}

    dd = DocDistance()
    for by_extension, doc in docs.items():
        chosen = by_extension if as_files else "plain"
        other = "plain" if as_files else by_extension
        statements = dd.embed_statements(doc, input_format=chosen)[0]
        if other != chosen:  # the third document is plain text as a file and as raw text
            assert dd.embed_statements(doc, input_format=other)[0] != statements
        reader_lines.clear()
        assert dd.embed_statements(doc, input_format="auto")[0] == statements
        assert dd.embed_statements(doc)[0] == statements
        line = f"read by the {chosen} reader"
        if as_files:
            line = f"{doc}: {line} (chosen by the file extension)"
        assert reader_lines == [line, line]


@pytest.mark.parametrize("as_path", [False, True], ids=["str", "Path"])
@pytest.mark.parametrize(
    ("name", "reader"),
    [
        ("doc.md", "markdown"),
        ("doc.markdown", "markdown"),
        ("doc.html", "html"),
        ("doc.htm", "html"),
        ("doc.MD", "markdown"),
        ("doc.MarkDown", "markdown"),
        ("doc.HTML", "html"),
        ("doc.Htm", "html"),
        ("doc.txt", "plain"),
        ("doc.rst", "plain"),
        ("doc", "plain"),
        ("doc.md.txt", "plain"),
        ("doc.txt.md", "markdown"),
        ("md", "plain"),
        (".md", "plain"),
        ("..md", "plain"),
    ],
)
def test_auto_chooses_the_reader_of_a_file_from_its_extension(
    name, reader, as_path, tmp_path, reader_lines
):
    """ACC-SEGM-93: ``.md`` and ``.markdown`` choose the markdown reader, ``.html`` and ``.htm``
    the HTML reader, in any letter case. Every other extension and a file without one choose the
    plain reader; the extension is the part of the name from its last dot, so ``doc.md.txt``
    reads plain, and ``.md`` and ``..md`` are names without extension (``pathlib`` of Python 3.12
    and 3.13 reads the suffix ``.md`` from ``..md``). The content is the same in every file
    and is not read for the choice: each reader returns its own statements of it. The document
    argument is a ``str`` or a ``Path``, with ``input_format="auto"`` and without the option."""
    doc = tmp_path / name
    doc.write_text(EXTENSION_DOC, encoding="utf-8")
    argument = doc if as_path else str(doc)
    dd = DocDistance()
    assert dd.embed_statements(argument)[0] == EXTENSION_STATEMENTS[reader]
    assert dd.embed_statements(argument, input_format="auto")[0] == EXTENSION_STATEMENTS[reader]
    line = f"{argument}: read by the {reader} reader (chosen by the file extension)"
    assert reader_lines == [line, line]


def test_auto_reads_raw_text_with_the_plain_reader(tmp_path, monkeypatch, reader_lines):
    """ACC-SEGM-93: raw text names no file and has no extension, so the plain reader reads it
    whatever it holds and whatever it ends in. The second text is an HTML document. The third
    text ends in ``.md``, and the fourth is one line that ends in ``.md`` and names no file: both
    are raw text. On each text the reader named beside it returns other statements."""
    monkeypatch.chdir(tmp_path)  # notes.md names no file here
    dd = DocDistance()
    texts = {
        EXTENSION_DOC: "markdown",
        TWIN_HTML: "html",
        EXTENSION_DOC + "See notes.md": "markdown",
        "- see notes.md": "markdown",
    }
    for text, other in texts.items():
        plain = dd.embed_statements(text, input_format="plain")[0]
        assert plain != dd.embed_statements(text, input_format=other)[0]
        reader_lines.clear()
        assert dd.embed_statements(text)[0] == plain
        assert dd.embed_statements(text, input_format="auto")[0] == plain
        assert reader_lines == ["read by the plain reader", "read by the plain reader"]


@pytest.mark.parametrize("name", ["doc.md", "doc.html", "doc.txt"])
def test_explicit_input_format_reads_a_file_of_every_extension(name, tmp_path, reader_lines):
    """ACC-SEGM-94: an explicit ``plain``, ``markdown`` or ``html`` names the reader whatever the
    extension of the file is, and the DEBUG line then does not name the extension."""
    doc = tmp_path / name
    doc.write_text(EXTENSION_DOC, encoding="utf-8")
    dd = DocDistance()
    for value, statements in EXTENSION_STATEMENTS.items():
        assert dd.embed_statements(doc, input_format=value)[0] == statements
    assert reader_lines == [f"{doc}: read by the {value} reader" for value in EXTENSION_STATEMENTS]


def test_embed_statements_input_format_plain_on_a_markdown_file_is_the_reader_of_1_1_5(tmp_path):
    """ACC-SEGM-95: ``input_format="plain"`` gives the body of a ``.md`` file, title line removed,
    whole to the prose segmenter, as 1.1.5 read every document: the reported text is the embedded
    text. Without the option the file is read by the markdown reader, which returns other
    statements."""
    doc = tmp_path / "results.md"
    doc.write_text("# Results of the run\n\n" + MD_DOC, encoding="utf-8")
    dd = DocDistance()
    statements, _ = dd.embed_statements(doc, input_format="plain")
    assert statements == _prose_split(MD_DOC.strip())
    assert _CapturingEncoder.seen == [statements]
    by_default, _ = dd.embed_statements(doc)
    assert by_default == dd.embed_statements(doc, markdown=True)[0] != statements


def test_reader_line_names_a_requested_reader_without_detection(reader_lines):
    dd = DocDistance()
    dd.embed_statements(TWIN_HTML, input_format="html")
    dd.embed_statements(TWIN_MD)
    assert len(reader_lines) == 2
    assert "html reader" in reader_lines[0] and "plain reader" in reader_lines[1]
    assert not any("extension" in line for line in reader_lines)


@pytest.mark.parametrize("value", ["html", "plain"])
def test_markdown_flag_beside_another_reader_raises(value):
    """``markdown=True`` beside ``html`` or ``plain`` names two readers."""
    with pytest.raises(ValueError, match=f"markdown=True and input_format='{value}'"):
        DocDistance().embed_statements(MD_DOC, markdown=True, input_format=value)
    with pytest.raises(ValueError, match=f"markdown=True and input_format='{value}'"):
        semantic_distance(MD_DOC, MD_DOC, markdown=True, input_format=value)


def test_unknown_input_format_raises_and_lists_the_four_values():
    with pytest.raises(ValueError, match="unknown input_format 'pdf'") as raised:
        DocDistance().embed_statements(MD_DOC, input_format="pdf")
    assert "plain, markdown, html, auto" in str(raised.value)
    with pytest.raises(ValueError, match="unknown input_format 'pdf'"):
        semantic_distance(MD_DOC, MD_DOC, input_format="pdf")
    assert _CapturingEncoder.seen == []  # nothing was read or encoded


def test_markdown_flag_beside_the_default_auto_is_the_markdown_reader(tmp_path):
    """ACC-SEGM-94: ``auto`` is the default of every call, so ``markdown=True`` beside it is
    ``markdown=True`` alone: the markdown reader on every document, also on a ``.txt`` and on a
    ``.html`` file, where ``auto`` alone chooses another reader."""
    dd = DocDistance()
    for name in ("doc.txt", "doc.html"):
        doc = tmp_path / name
        doc.write_text(EXTENSION_DOC, encoding="utf-8")
        by_flag, _ = dd.embed_statements(doc, markdown=True, input_format="auto")
        assert by_flag == dd.embed_statements(doc, markdown=True)[0]
        assert by_flag == EXTENSION_STATEMENTS["markdown"] != dd.embed_statements(doc)[0]


# ------------------------------------------------------------------ ACC-SEGM-91: marker warning


def test_marker_warning_is_logged_under_auto_when_the_plain_reader_ran(logged):
    """The warning follows the reader that ran: under ``auto`` a document read as plain text logs
    it."""
    dd = DocDistance()
    statements, _ = dd.embed_statements(DOC_ONE_DASH, input_format="auto")
    assert statements == ["Adoption held steady", "-", "Barriers remain"]
    assert len(logged) == 1 and "1 of 3 statements" in logged[0]
    for option in ("--markdown", "markdown=True", "--min-words", "min_words"):
        assert option in logged[0]
    assert "auto" not in logged[0]

    dd.embed_statements(DOC_ONE_DASH)
    assert logged[1] == logged[0]


def test_marker_warning_is_not_logged_when_the_markdown_or_the_html_reader_ran(logged, tmp_path):
    """Under ``auto`` a ``.md`` file is read by the markdown reader, which drops the bare
    markers; the HTML reader of a ``.html`` file ends in the markdown reader as well. Neither logs
    the warning, also when a statement without a letter and without a digit is left: the prose
    line ``?!``."""

    def file_of(name: str, text: str) -> str:
        (tmp_path / name).write_text(text, encoding="utf-8")
        return str(tmp_path / name)

    dd = DocDistance()
    kept = ["## Key findings", "Adoption held steady", "?!"]
    markdown = "## Key findings\n\nAdoption held steady.\n\n?!\n"
    html = "<h2>Key findings</h2>\n<p>Adoption held steady.</p>\n<p>?!</p>\n"
    assert dd.embed_statements(file_of("kept.md", markdown), input_format="auto")[0] == kept
    assert dd.embed_statements(markdown, markdown=True)[0] == kept
    assert dd.embed_statements(file_of("kept.html", html), input_format="auto")[0] == kept
    assert dd.embed_statements(html, input_format="html")[0] == kept
    assert logged == []
    dd.embed_statements(markdown)  # raw text: the plain reader on the same text logs it
    assert len(logged) == 1 and "1 of 3 statements" in logged[0]
    logged.clear()
    assert dd.embed_statements(file_of("markers.md", DOC_MARKERS), input_format="auto")[0] == [
        "## Key findings",
        "Adoption held steady",
        "Barriers remain",
        "42%",
    ]
    html = (
        "<h2>Key findings</h2>\n<p>-</p>\n<p>Adoption held steady.</p>\n<ul>\n<li></li>\n</ul>\n"
    )
    for doc, value in ((html, "html"), (file_of("markers.html", html), "auto")):
        assert dd.embed_statements(doc, input_format=value)[0] == [
            "## Key findings",
            "Adoption held steady",
        ]
    assert dd.embed_statements(DOC_ONE_DASH, input_format="markdown")[0] == [
        "Adoption held steady",
        "Barriers remain",
    ]
    assert logged == []
