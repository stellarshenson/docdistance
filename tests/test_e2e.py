"""Functional end-to-end tests driving the real CLI and API on crafted documents.

The full path - argument parsing, ``_read``, segmentation, encode, ``_build_structural_details`` /
``_build_semantic_details``, JSON output - runs for real; only the model layer (SAT segmenter + mmBERT
encoder) is faked so the suite stays offline in the uv ``.venv``. The fake encoder maps each statement
text to a deterministic unit vector (identical text -> identical vector, different text -> near-orthogonal
vector), so a reorder is content-preserving (SMD ~ 0, positive order-gap, localized via the structural
displacement axis) and a reword shows up as a per-flow ``changed`` spike on the semantic content axis.
Real-model e2e lives in ``notebooks/09-kj-docdistance-api-e2e.ipynb``. The last section is the
exception: three tests start child processes, one imports ``openvino`` and two load the real
models where they are installed.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from rich.console import Console
from typer.testing import CliRunner

from docdistance import pipeline, settings
from docdistance.cli import app
from docdistance.distance import DistanceResult, StructuralResult
from docdistance.pipeline import (
    DIFF_CHANGED_COST,
    DocDistance,
    semantic_distance,
    structural_distance,
)

runner = CliRunner()
_WIDE = {"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"}

# crafted documents - four statements each; the fake segmenter splits on the full stop
DOC = "The cat sat still. The dog ran fast. Birds fly high. Fish swim deep."
DOC_SWAP = "The cat sat still. The dog ran fast. Fish swim deep. Birds fly high."  # statements 2<->3 swapped
DOC_REWORD = "The cat sat still. The dog ran fast. Birds fly high. Volcanoes erupt violently."  # stmt 3 reworded
# six statements, and the same document minus statements 1 and 4 with the order of the rest kept
DOC_LONG = "The cat sat still. The dog ran fast. Birds fly high. Fish swim deep. Snow fell overnight. Rivers run east."
DOC_LONG_MINUS_2 = "The cat sat still. Birds fly high. Fish swim deep. Rivers run east."


def _vec(text: str, dim: int = 32) -> np.ndarray:
    seed = int(hashlib.sha1(text.encode()).hexdigest(), 16) % (2**32)
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class _FakeSegmenter:
    def __init__(self, offline: bool = True):
        pass

    def split(self, text: str) -> list[str]:
        return [s.strip() for s in text.replace("\n", ". ").split(".") if s.strip()]


class _FakeEncoder:
    def encode(self, statements: list[str]) -> np.ndarray:
        return np.stack([_vec(s) for s in statements])


def _fake_load_encoder(backend, offline=True, device=None):
    return _FakeEncoder()


@pytest.fixture(autouse=True)
def _wired(monkeypatch):
    """Mark the wmd mode ready and swap in the fake segmenter + encoder for every test here."""
    settings.reset()
    settings.mark_ready("wmd")
    monkeypatch.setattr(pipeline, "Segmenter", _FakeSegmenter)
    monkeypatch.setattr(pipeline, "load_encoder", _fake_load_encoder)
    yield
    settings.reset()


def _by_index(diff: dict) -> dict[int, dict]:
    """Index the structural details' per-statement records (displacement / moved) by source index."""
    return {s["index"]: s for s in diff["statements"]}


def _by_flow(details: dict) -> dict[int, dict]:
    """Index the semantic details' per-statement transport flows (matches / changed) by source index."""
    return {f["index"]: f for f in details["flows"]}


# --------------------------------------------------------------------------- CLI


def test_cli_distance_plain_reports_a_verdict():
    res = runner.invoke(app, ["distance-semantic", DOC, DOC_SWAP], env=_WIDE)
    assert res.exit_code == 0, res.output
    assert "SMD" in res.output
    assert "closeness" in res.output


def test_cli_distance_json_emits_machine_readable():
    res = runner.invoke(app, ["distance-semantic", DOC, DOC_REWORD, "--json"], env=_WIDE)
    assert res.exit_code == 0, res.output
    assert '"smd"' in res.output and '"verdict"' in res.output


def test_cli_structural_details_localizes_a_reorder(tmp_path):
    """A pure statement swap: content preserved (SMD ~ 0), positive order-gap, only the swapped pair moves."""
    out = tmp_path / "order.json"
    res = runner.invoke(
        app, ["distance-structural", DOC, DOC_SWAP, "--details-json", str(out)], env=_WIDE
    )
    assert res.exit_code == 0, res.output
    diff = json.loads(out.read_text())

    assert diff["smd"] == pytest.approx(
        0.0, abs=2e-3
    )  # same statements, reordered (float floor ~3e-4)
    assert diff["order_gap"] > 0.0  # arrangement changed
    assert diff["structure_closeness"] < 1.0
    st = _by_index(diff)
    assert st[0]["displacement"] == 0 and st[0]["moved"] is False  # statements 0,1 stayed put
    assert st[1]["displacement"] == 0 and st[1]["moved"] is False
    assert st[2]["moved"] is True and st[3]["moved"] is True  # the swapped pair moved


def test_cli_semantic_details_localizes_a_reword(tmp_path):
    """One reworded statement: its best-match transport cost spikes and it flags changed, others do not."""
    out = tmp_path / "content.json"
    res = runner.invoke(
        app, ["distance-semantic", DOC, DOC_REWORD, "--details-json", str(out)], env=_WIDE
    )
    assert res.exit_code == 0, res.output
    details = json.loads(out.read_text())

    fl = _by_flow(details)
    assert fl[3]["changed"] is True  # the reworded statement
    assert fl[3]["matches"][0]["cost"] > DIFF_CHANGED_COST  # its best match is a far statement
    for i in (0, 1, 2):
        assert fl[i]["changed"] is False
        assert (
            fl[i]["matches"][0]["cost"] < 0.01
        )  # untouched statements map to their twin at ~0 cost


def test_cli_semantic_details_writes_flows(tmp_path):
    out = tmp_path / "content.json"
    res = runner.invoke(
        app, ["distance-semantic", DOC, DOC_SWAP, "--details-json", str(out)], env=_WIDE
    )
    assert res.exit_code == 0, res.output
    m = json.loads(out.read_text())
    assert m["n_statements"] == {"a": 4, "b": 4}
    assert len(m["flows"]) == 4
    assert all(f["matches"] for f in m["flows"])


def test_cli_result_only_prints_a_bare_scalar():
    res = runner.invoke(app, ["distance-semantic", DOC, DOC_SWAP, "--result-only"], env=_WIDE)
    assert res.exit_code == 0, res.output
    floats = [
        float(tok) for line in res.output.splitlines() for tok in line.split() if _isfloat(tok)
    ]
    assert floats and floats[-1] == pytest.approx(0.0, abs=1e-3)  # reorder -> SMD ~ 0


def test_cli_structural_machine_output_does_not_route_through_rich(monkeypatch):
    """``--json`` / ``--result-only`` must bypass the Rich console.

    Rich applies its repr highlighter and wraps at the console width, so a coloured or narrow
    terminal turns the documented machine-readable output into unparseable text while the exit
    code stays 0. The suite's own ``_WIDE`` env (TERM=dumb, NO_COLOR) hides it, and TERM=dumb
    also defeats a plain ``force_terminal`` console - hence the explicit ``color_system``.
    """
    import docdistance.cli as cli_mod

    monkeypatch.setattr(
        cli_mod, "_out", Console(force_terminal=True, width=80, no_color=False, color_system="256")
    )

    res = runner.invoke(app, ["distance-structural", DOC, DOC_SWAP, "--result-only"], env=_WIDE)
    assert res.exit_code == 0, res.output
    float(res.output.strip())  # raises if Rich wrapped or colourized it

    res = runner.invoke(app, ["distance-structural", DOC, DOC_SWAP, "--json"], env=_WIDE)
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)  # raises if Rich wrapped or colourized it
    assert payload["verdict"] in {"similar", "not similar"}


def _isfloat(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _plain(text: str) -> str:
    """Strip the ANSI codes Rich writes around highlighted numbers regardless of NO_COLOR."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_cli_structural_shows_the_matched_order_gap_after_a_deletion():
    """ACC-CLI-74: a document against itself minus 2 statements, order kept.

    The panel names the matched order gap and the matched statement count; ``--json`` carries both
    keys; ``--result-only`` still prints the bare ``structure_closeness``.
    """
    res = runner.invoke(
        app, ["distance-structural", DOC_LONG, DOC_LONG_MINUS_2, "--json"], env=_WIDE
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["n_statements_a"] == 6 and payload["n_statements_b"] == 4
    assert payload["n_matched"] == 4
    assert payload["matched_order_gap"] < 1e-6 < payload["order_gap"]  # the deletion is no reorder

    res = runner.invoke(app, ["distance-structural", DOC_LONG, DOC_LONG_MINUS_2], env=_WIDE)
    assert res.exit_code == 0, res.output
    plain = _plain(res.output)
    assert re.search(r"matched order gap\s+0\.0000\b", plain)
    assert re.search(r"matched statements\s+4\b", plain)

    res = runner.invoke(
        app, ["distance-structural", DOC_LONG, DOC_LONG_MINUS_2, "--result-only"], env=_WIDE
    )
    assert res.exit_code == 0, res.output
    assert res.output.strip() == str(payload["structure_closeness"])


class _FakeReranker:
    def score_grid(self, docs: list[str], sources: list[str]) -> np.ndarray:
        return np.full((len(docs), len(sources)), 0.5, dtype=np.float32)


class _FakeNLI:
    def entail(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        return np.full(len(premises), 0.5, dtype=np.float32)


def test_cli_wrt_source_shows_the_unsupported_counts(monkeypatch):
    """ACC-CLI-75: document A holds one statement the source does not, document B none.

    The panel names both counts; ``--json`` carries ``n_unsupported_a`` / ``n_unsupported_b``;
    ``--result-only`` still prints three comma-separated numbers (``d_sel,grd_a,grd_b`` here, the
    shipped output when the grounding models scored).
    """
    settings.mark_ready("wmd-wrt-source")
    monkeypatch.setattr(pipeline, "load_reranker", lambda *args, **kwargs: _FakeReranker())
    monkeypatch.setattr(pipeline, "load_nli", lambda *args, **kwargs: _FakeNLI())
    doc_a = "The cat sat still. Birds fly high. Volcanoes erupt violently."
    doc_b = "The dog ran fast. Fish swim deep. Rivers run east."
    args = ["distance-wrt-source", doc_a, doc_b, "--source", DOC_LONG]

    res = runner.invoke(app, [*args, "--json"], env=_WIDE)
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert (payload["n_unsupported_a"], payload["n_unsupported_b"]) == (1, 0)

    res = runner.invoke(app, args, env=_WIDE)
    assert res.exit_code == 0, res.output
    assert re.search(r"unsupported statements\s+A 1 / B 0\b", _plain(res.output))

    res = runner.invoke(app, [*args, "--result-only"], env=_WIDE)
    assert res.exit_code == 0, res.output
    assert res.output.strip() == f"{payload['d_sel']},{payload['grd_a']},{payload['grd_b']}"


# --------------------------------------------------------------------------- --pairs memos


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _affinity(left: str, right: str) -> float:
    """A value in [0, 1] read from the two texts, so different texts give different values."""
    return float(_vec(left) @ _vec(right) + 1.0) / 2.0


class _CountingEncoder(_FakeEncoder):
    """The fake encoder; records every batch it encodes."""

    def __init__(self):
        self.batches: list[list[str]] = []

    def encode(self, statements: list[str]) -> np.ndarray:
        self.batches.append(list(statements))
        return super().encode(statements)


class _TextReranker:
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
def grounding(monkeypatch):
    """The source-conditioned mode ready, with a reranker and an NLI stub that read the texts."""
    settings.mark_ready("wmd-wrt-source")
    reranker = _TextReranker()
    monkeypatch.setattr(pipeline, "load_reranker", lambda *args, **kwargs: reranker)
    monkeypatch.setattr(pipeline, "load_nli", lambda *args, **kwargs: _TextNLI())
    return reranker


def _count_encodes(patch) -> _CountingEncoder:
    """Swap in a counting encoder while ``patch`` is active and return it."""
    encoder = _CountingEncoder()
    patch.setattr(pipeline, "load_encoder", lambda *args, **kwargs: encoder)
    return encoder


def _switch_memo_off(patch) -> None:
    """Every ``DocDistance`` built while ``patch`` is active has both memos off."""
    init = DocDistance.__init__
    patch.setattr(
        DocDistance,
        "__init__",
        lambda self, *args, **kwargs: init(self, *args, **{**kwargs, "memo_size": 0}),
    )


def _sweep_doc(i: int) -> str:
    """A heading, prose, a list item, a bare list marker and a table row.

    Documents of different ``i`` share no statement, and an odd ``i`` holds one more two-word
    statement, so ``--min-words 4 --skip-table-fragments`` drops 3 statements of an even document
    and 4 of an odd one.
    """
    note = f"Note {i}. " if i % 2 else ""
    return (
        f"## Section {i}\n\nReport {i} opens with a long first statement. {note}\n\n"
        f"- point {i} of the list\n- \n\n| cell {i} | value |\n\n"
        f"Report {i} closes with a long last statement."
    )


def _sweep_files(tmp_path, n: int) -> list[str]:
    files = []
    for i in range(n):
        path = tmp_path / f"report-{i}.md"
        path.write_text(_sweep_doc(i), encoding="utf-8")
        files.append(str(path))
    return files


def _all_pairs(files: list[str]) -> list[tuple[str, str]]:
    return [(a, b) for i, a in enumerate(files) for b in files[i + 1 :]]


def _write_tsv(tmp_path, rows: list[tuple[str, str]]) -> tuple[str, list[str]]:
    """Write ``rows`` as a pairs TSV with the ids ``row-<n>``; returns its path and the ids."""
    ids = [f"row-{n}" for n in range(len(rows))]
    tsv = tmp_path / "pairs.tsv"
    tsv.write_text("".join(f"{a}\t{b}\t{pair}\n" for (a, b), pair in zip(rows, ids)))
    return str(tsv), ids


# the nine sweeps of E12-H67: command, rows over the 22 sweep files, options
_FILTER = ["--min-words", "4", "--skip-table-fragments"]
_ARMS = {
    "all-pairs": ("distance-semantic", lambda f: _all_pairs(f[:11]), []),
    "all-pairs details": ("distance-semantic", lambda f: _all_pairs(f[:11]), ["--details-json"]),
    "source-x2": ("distance-semantic", lambda f: [(f[21], doc) for doc in f[:2]], []),
    "source-x5": ("distance-semantic", lambda f: [(f[21], doc) for doc in f[:5]], []),
    "source-x11": ("distance-semantic", lambda f: [(f[21], doc) for doc in f[:11]], []),
    "disjoint": ("distance-semantic", lambda f: list(zip(f[:22:2], f[1:22:2])), []),
    "structural": ("distance-structural", lambda f: _all_pairs(f[:11]), ["--details-json"]),
    "filtered": ("distance-semantic", lambda f: _all_pairs(f[:11]), _FILTER),
    "markdown": ("distance-semantic", lambda f: _all_pairs(f[:11]), ["--markdown"]),
    "plain": ("distance-semantic", lambda f: _all_pairs(f[:11]), ["--input-format", "plain"]),
}


@pytest.mark.parametrize("arm", _ARMS)
def test_cli_pairs_output_is_byte_identical_with_the_memo_on_and_off(arm, tmp_path, monkeypatch):
    """ACC-CLI-30, ACC-CLI-29: each sweep runs with the memo on and with it off.

    The sha256 of the JSONL and of every per-pair details file is equal under both settings. With
    the memo each distinct file is encoded once (11 encodes for the 110 slots of the all-pairs
    sweep); without it every slot is encoded.
    """
    command, rows_of, options = _ARMS[arm]
    rows = rows_of(_sweep_files(tmp_path, 22))
    tsv, _ = _write_tsv(tmp_path, rows)
    digests, encodes = {}, {}
    for setting in ("on", "off"):
        details = tmp_path / f"details-{setting}"
        args = [command, "--pairs", tsv, *options]
        if "--details-json" in options:
            args.append(str(details))
        with monkeypatch.context() as patch:
            encoder = _count_encodes(patch)
            if setting == "off":
                _switch_memo_off(patch)
            res = runner.invoke(app, args, env=_WIDE)
        assert res.exit_code == 0, res.output
        assert len(res.stdout.splitlines()) == len(rows)
        files = {path.name: _sha256(path.read_bytes()) for path in sorted(details.glob("*.json"))}
        assert len(files) == (len(rows) if "--details-json" in options else 0)
        digests[setting] = (_sha256(res.stdout.encode()), files)
        encodes[setting] = len(encoder.batches)

    assert digests["on"] == digests["off"]
    slots = [doc for row in rows for doc in row]
    assert encodes == {"on": len(set(slots)), "off": len(slots)}


def test_cli_pairs_filtered_lines_carry_the_drop_counts_of_each_document(tmp_path):
    """ACC-CLI-36: in a filtered batch every line reports the drop count of its two documents, 3 for
    an even sweep document and 4 for an odd one, on rows served from the memo as on the first.
    The counts are those of the plain reader, which the command names: the markdown reader, the
    default of the ``.md`` sweep files, returns no statement for the bare list marker."""
    files = _sweep_files(tmp_path, 4)
    rows = _all_pairs(files)
    tsv, _ = _write_tsv(tmp_path, rows)
    res = runner.invoke(
        app,
        ["distance-semantic", "--pairs", tsv, *_FILTER, "--input-format", "plain"],
        env=_WIDE,
    )
    assert res.exit_code == 0, res.output
    dropped = {doc: 3 + i % 2 for i, doc in enumerate(files)}
    assert [json.loads(line)["n_dropped"] for line in res.stdout.splitlines()] == [
        {"a": dropped[a], "b": dropped[b]} for a, b in rows
    ]


def _source_file(tmp_path) -> str:
    source = tmp_path / "source.md"
    source.write_text("\n\n".join(_sweep_doc(i) for i in range(0, 22, 3)), encoding="utf-8")
    return str(source)


def test_cli_wrt_source_pairs_emits_one_line_per_row_in_row_order(
    grounding, tmp_path, monkeypatch
):
    """ACC-CLI-37, ACC-CLI-38: 55 rows against one source.

    One ``DocDistance`` is built, 55 lines come out with the ids in row order, and the keys of a
    line are ``pair`` followed by the keys of the one-shot ``--json`` output. The 110 (document,
    source) slots cost 11 relevance grids.
    """
    rows = _all_pairs(_sweep_files(tmp_path, 11))
    tsv, ids = _write_tsv(tmp_path, rows)
    source = _source_file(tmp_path)
    constructed = []
    init = DocDistance.__init__
    with monkeypatch.context() as patch:
        patch.setattr(
            DocDistance,
            "__init__",
            lambda self, *args, **kwargs: (constructed.append(1), init(self, *args, **kwargs))[1],
        )
        res = runner.invoke(
            app, ["distance-wrt-source", "--pairs", tsv, "--source", source], env=_WIDE
        )
    assert res.exit_code == 0, res.output
    assert len(constructed) == 1
    assert len(grounding.grids) == 11

    lines = [json.loads(line) for line in res.stdout.splitlines()]
    assert len(lines) == 55
    assert [line["pair"] for line in lines] == ids

    a, b = rows[0]
    one_shot = runner.invoke(
        app, ["distance-wrt-source", a, b, "--source", source, "--json"], env=_WIDE
    )
    assert one_shot.exit_code == 0, one_shot.output
    for line in lines:
        assert list(line) == ["pair", *json.loads(one_shot.stdout)]


@pytest.mark.parametrize(
    ("n_files", "options"),
    [(11, []), (4, ["--no-anisotropy"]), (4, _FILTER), (4, ["--markdown"])]
    + [(4, ["--input-format", "plain"])],
    ids=["default-55-rows", "no-anisotropy", "filtered", "markdown", "plain"],
)
def test_cli_wrt_source_batch_line_equals_the_one_shot_json(
    n_files, options, grounding, tmp_path, monkeypatch
):
    """ACC-CLI-39: every batch line against the one-shot ``--json`` output of its pair.

    A batch line is one line and starts with the ``pair`` id; the one-shot output is the same
    object printed with ``indent=2``. The line without ``pair``, printed the one-shot way, is
    byte-identical to the one-shot stdout, with the memo on and with it off, and the two batch
    outputs are byte-identical.
    """
    rows = _all_pairs(_sweep_files(tmp_path, n_files))
    tsv, ids = _write_tsv(tmp_path, rows)
    source = _source_file(tmp_path)
    one_shot = {}
    for (a, b), pair in zip(rows, ids):
        res = runner.invoke(
            app, ["distance-wrt-source", a, b, "--source", source, "--json", *options], env=_WIDE
        )
        assert res.exit_code == 0, res.output
        one_shot[pair] = res.stdout

    batch = {}
    for setting in ("on", "off"):
        grounding.grids.clear()
        with monkeypatch.context() as patch:
            if setting == "off":
                _switch_memo_off(patch)
            res = runner.invoke(
                app,
                ["distance-wrt-source", "--pairs", tsv, "--source", source, *options],
                env=_WIDE,
            )
        assert res.exit_code == 0, res.output
        assert len(grounding.grids) == (n_files if setting == "on" else 2 * len(rows))
        batch[setting] = res.stdout
        printed = {}
        for line in res.stdout.splitlines():
            obj = json.loads(line)
            pair = obj.pop("pair")
            printed[pair] = json.dumps(obj, indent=2) + "\n"
        assert printed == one_shot
    assert batch["on"] == batch["off"]
    assert len(set(one_shot.values())) == len(rows)  # every pair has its own output


@pytest.mark.parametrize(
    "options", [[], ["--no-anisotropy"], _FILTER], ids=["default", "no-anisotropy", "filtered"]
)
def test_cli_wrt_source_pairs_writes_one_source_map_per_pair(options, grounding, tmp_path):
    """ACC-CLI-37: with ``--pairs``, ``--source-map-json`` names a directory or a ``{pair}``
    template. Each pair gets the map file the one-shot command writes for it under the same
    options, and the map calls are served from the grounding memo: 3 grids for 3 documents."""
    rows = _all_pairs(_sweep_files(tmp_path, 3))
    tsv, ids = _write_tsv(tmp_path, rows)
    source = _source_file(tmp_path)
    batch = ["distance-wrt-source", "--pairs", tsv, "--source", source, *options]

    res = runner.invoke(app, [*batch, "--source-map-json", str(tmp_path / "maps")], env=_WIDE)
    assert res.exit_code == 0, res.output
    assert len(grounding.grids) == 3
    template = str(tmp_path / "map-{pair}.json")
    res = runner.invoke(app, [*batch, "--source-map-json", template], env=_WIDE)
    assert res.exit_code == 0, res.output

    maps = set()
    for (a, b), pair in zip(rows, ids):
        single = tmp_path / "single.json"
        one_shot = ["distance-wrt-source", a, b, "--source", source, *options]
        res = runner.invoke(app, [*one_shot, "--source-map-json", str(single)], env=_WIDE)
        assert res.exit_code == 0, res.output
        assert (tmp_path / "maps" / f"{pair}.json").read_bytes() == single.read_bytes()
        assert (tmp_path / f"map-{pair}.json").read_bytes() == single.read_bytes()
        maps.add(single.read_bytes())
    assert len(maps) == len(rows)  # every pair has its own map


# --------------------------------------------------------------------------- --input-format

# one document as an HTML file and as its markdown conversion: 5 statements under the markdown
# reader, a heading, two prose statements and two list items
TWIN_HTML = (
    "<h2>Findings</h2>\n<p>The cat sat still. The dog ran fast.</p>\n"
    "<ul>\n<li>birds fly high</li>\n<li>fish swim deep</li>\n</ul>\n"
)
TWIN_MD = (
    "## Findings\n\nThe cat sat still. The dog ran fast.\n\n- birds fly high\n- fish swim deep\n"
)


@pytest.mark.parametrize(
    "command", ["distance-semantic", "distance-structural", "distance-wrt-source"]
)
def test_cli_input_format_auto_reads_an_html_file_as_its_markdown_twin(
    command, grounding, tmp_path
):
    """ACC-SEGM-94: an HTML file against its markdown conversion with ``--input-format auto``
    and without the option, which is the same reader.

    The command prints what it prints for the markdown file against itself with ``--markdown``:
    5 statements on each side, for a single pair and for the same pair in a ``--pairs`` batch.
    With ``--input-format plain`` the prose segmenter splits the raw HTML text into other
    statements.
    """
    html, md = tmp_path / "twin.html", tmp_path / "twin.md"
    html.write_text(TWIN_HTML, encoding="utf-8")
    md.write_text(TWIN_MD, encoding="utf-8")
    source = ["--source", str(md)] if command == "distance-wrt-source" else []

    def run(*args: str) -> str:
        res = runner.invoke(app, [command, *args, *source], env=_WIDE)
        assert res.exit_code == 0, res.output
        return res.stdout

    expected = json.loads(run(str(md), str(md), "--markdown", "--json"))
    assert (expected["n_statements_a"], expected["n_statements_b"]) == (5, 5)
    assert json.loads(run(str(html), str(md), "--input-format", "auto", "--json")) == expected
    assert json.loads(run(str(html), str(md), "--json")) == expected
    plain = json.loads(run(str(html), str(md), "--input-format", "plain", "--json"))
    assert plain["n_statements_a"] == 8

    tsv = tmp_path / "pairs.tsv"
    tsv.write_text(f"{html}\t{md}\ttwin\n")
    line = json.loads(run("--pairs", str(tsv), "--input-format", "auto"))
    assert line == {"pair": "twin", **expected}
    assert json.loads(run("--pairs", str(tsv))) == line


@pytest.mark.parametrize(
    "command", ["distance-semantic", "distance-structural", "distance-wrt-source"]
)
def test_cli_default_reader_follows_the_file_extension(command, grounding, tmp_path):
    """ACC-SEGM-94, ACC-SEGM-93: without ``--input-format`` each command reads a ``.md`` file with
    the markdown reader, a ``.html`` file with the HTML reader and a ``.txt`` file with the plain
    reader.

    Document B is document A without its last list item. The pair is written three times:
    markdown files, their HTML twins and ``.txt`` copies of the markdown files. The ``.md`` pair
    and the ``.html`` pair print what the ``.md`` pair prints with ``--markdown``. The ``.txt``
    pair prints what the ``.md`` pair prints with ``--input-format plain``, the reader of 1.1.5,
    which is another output. The source of ``distance-wrt-source`` is document A in the kind of
    file of the pair.
    """
    texts = {
        "a.md": TWIN_MD,
        "b.md": TWIN_MD.replace("- fish swim deep\n", ""),
        "a.html": TWIN_HTML,
        "b.html": TWIN_HTML.replace("<li>fish swim deep</li>\n", ""),
    }
    texts |= {"a.txt": texts["a.md"], "b.txt": texts["b.md"]}
    for name, text in texts.items():
        (tmp_path / name).write_text(text, encoding="utf-8")

    def run(extension: str, *options: str) -> dict:
        a, b = (str(tmp_path / f"{stem}.{extension}") for stem in "ab")
        source = ["--source", a] if command == "distance-wrt-source" else []
        res = runner.invoke(app, [command, a, b, *source, "--json", *options], env=_WIDE)
        assert res.exit_code == 0, res.output
        return json.loads(res.stdout)

    by_markdown = run("md", "--markdown")
    by_plain = run("md", "--input-format", "plain")
    assert (by_markdown["n_statements_a"], by_markdown["n_statements_b"]) == (5, 4)
    assert by_plain != by_markdown
    assert run("md") == by_markdown
    assert run("html") == by_markdown
    assert run("txt") == by_plain


# --------------------------------------------------------------------------- API


def test_api_semantic_distance_on_text():
    r = semantic_distance(DOC, DOC_REWORD)
    assert isinstance(r, DistanceResult)
    assert 0.0 <= r.closeness <= 1.0
    assert r.verdict in {"similar", "not similar"}
    assert r.n_statements_a == 4 and r.n_statements_b == 4
    assert r.wcd <= r.smd + 1e-6 and r.rwmd <= r.smd + 1e-6  # both are lower bounds of SMD


def test_api_structural_distance_on_text():
    r = structural_distance(DOC, DOC_SWAP)
    assert isinstance(r, StructuralResult)
    assert r.order_gap >= 0.0  # a reorder opens a positive order-gap
    assert 0.0 <= r.structure_closeness <= 1.0
    assert r.smd == pytest.approx(0.0, abs=2e-3)  # the swap preserves content
    assert r.verdict in {"similar", "not similar"}
    assert r.n_statements_a == 4 and r.n_statements_b == 4


def test_api_structural_with_details_localizes_a_reorder():
    """A reorder is localized on the structural axis: content held, the swapped pair moves."""
    dd = DocDistance()

    result, reorder = dd.structural_distance_with_details(DOC, DOC_SWAP)
    assert isinstance(result, StructuralResult)
    assert reorder["smd"] == pytest.approx(0.0, abs=2e-3)
    assert reorder["order_gap"] > 0.0
    moved = {s["index"] for s in reorder["statements"] if s["moved"]}
    assert moved == {2, 3}

    # a reword leaves the order intact - nothing moves on the structural axis
    _, reword = dd.structural_distance_with_details(DOC, DOC_REWORD)
    assert all(s["displacement"] == 0 for s in reword["statements"])


def test_api_semantic_with_details_localizes_a_reword():
    """A reword is localized on the semantic content axis: its transport flow flags changed, others do not."""
    dd = DocDistance()

    result, reword = dd.semantic_distance_with_details(DOC, DOC_REWORD)
    assert isinstance(result, DistanceResult)
    changed = {f["index"] for f in reword["flows"] if f["changed"]}
    assert changed == {3}
    assert reword["flows"][3]["matches"][0]["cost"] > DIFF_CHANGED_COST  # far best match


def test_api_semantic_with_details_shares_one_encode_pass():
    dd = DocDistance()
    result, m = dd.semantic_distance_with_details(DOC, DOC_SWAP)
    assert m["smd"] == pytest.approx(result.smd, abs=1e-6)  # map and result agree
    assert len(m["flows"]) == 4


def test_api_and_cli_agree_on_smd(tmp_path):
    """The same pair scored through the API and through the CLI details path gives the same SMD."""
    api_smd = semantic_distance(DOC, DOC_REWORD).smd
    out = tmp_path / "content.json"
    res = runner.invoke(
        app, ["distance-semantic", DOC, DOC_REWORD, "--details-json", str(out)], env=_WIDE
    )
    assert res.exit_code == 0, res.output
    cli_smd = json.loads(out.read_text())["smd"]
    assert cli_smd == pytest.approx(api_smd, abs=1e-6)


# --------------------------------------------------------------------------- real encoder load

# no main guard: a child process started through forkserver or spawn re-imports this script
_NO_MAIN_GUARD_SCRIPT = """\
import multiprocessing

print("MARKER", flush=True)

_start = multiprocessing.Process.start


def _report_start(process):
    print("CHILD", flush=True)
    _start(process)


multiprocessing.Process.start = _report_start

from docdistance import encoders

# the import the encoder load starts with; it alone started the child process, so no model is loaded
import openvino
"""


def test_loading_the_openvino_encoder_starts_no_child_process(tmp_path):
    """DEF-DEPS-11: a script without a main guard that imports what the encoder load imports runs once.

    ``import openvino`` sends a usage event through ``multiprocessing.Process``. Under the forkserver
    start method (the Linux default from Python 3.14) that child re-imports the main module, so the
    script ran twice. ``CI`` is removed because OpenVINO switches its telemetry off when it is set.
    """
    script = tmp_path / "no_main_guard.py"
    script.write_text(_NO_MAIN_GUARD_SCRIPT, encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if key != "CI"}
    # a closed local port as proxy: no request the script makes leaves the machine
    env["http_proxy"] = env["https_proxy"] = "http://127.0.0.1:9"
    res = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, text=True, check=False
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.count("MARKER") == 1
    assert "CHILD" not in res.stdout


# the Hub library is imported before docdistance, as in a host process that uses it: it has then
# read the environment, without HF_HUB_OFFLINE
_OFFLINE_LOAD_SCRIPT = """\
import huggingface_hub

from docdistance import DocDistance

if __name__ == "__main__":
    print("SHAPE", DocDistance().embed("The cat sat still. The dog ran fast.").shape, flush=True)
"""


def _run_script(script, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, text=True, check=False
    )


def test_offline_load_of_cached_models_works_with_the_hub_unreachable(tmp_path):
    """DEF-DEPS-14: ``offline=True`` loads the cached models and embeds while the Hub is unreachable.

    The child process has no ``HF_HUB_OFFLINE`` and its proxies name a closed local port, so every
    request to the Hub fails. On 1.1.12 the constructors set the variable after the library had
    read it, the segmenter load sent a request and raised ``ModelsNotInstalled``. Skipped on a
    machine without the models: there the same load also fails with ``HF_HUB_OFFLINE=1`` preset.
    """
    script = tmp_path / "offline_load.py"
    script.write_text(_OFFLINE_LOAD_SCRIPT, encoding="utf-8")
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    env["http_proxy"] = env["https_proxy"] = "http://127.0.0.1:9"
    res = _run_script(script, env)
    if res.returncode != 0 and _run_script(script, {**env, "HF_HUB_OFFLINE": "1"}).returncode != 0:
        pytest.skip("the models are not installed")
    assert res.returncode == 0, res.stderr
    assert "SHAPE (2, 768)" in res.stdout


def test_hub_offline_sets_the_flag_requests_read_and_puts_it_back(monkeypatch):
    """DEF-DEPS-14: the scope sets the flag that the Hub library checks before a request.

    With ``offline=False`` the mode of the process is left as it is. The value the flag had is put
    back afterwards, also when the load raises, so a later ``docdistance init`` reaches the Hub.
    """
    from huggingface_hub import constants

    from docdistance import encoders

    for before in (True, False):
        monkeypatch.setattr(constants, "HF_HUB_OFFLINE", before)
        with encoders._hub_offline(True):
            assert constants.is_offline_mode() is True
        assert constants.is_offline_mode() is before
        with encoders._hub_offline(False):
            assert constants.is_offline_mode() is before
        assert constants.is_offline_mode() is before
    with pytest.raises(encoders.ModelsNotInstalled), encoders._hub_offline(True):
        raise encoders.ModelsNotInstalled("a model is missing")
    assert constants.is_offline_mode() is False


def test_hub_offline_scopes_that_overlap_on_two_threads_keep_the_flag_until_the_last_ends(
    monkeypatch,
):
    """Two loads that overlap on two threads: the flag stays set until the second scope ends.

    The first scope starts first and ends first. On 1.1.13 each scope put back the value it had read
    at its own start: the end of the first scope unset the flag while the second load ran, and the
    end of the second scope set it, because that scope had started inside the first one.
    """
    from huggingface_hub import constants

    from docdistance import encoders

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    first_started, second_started, first_ended = (threading.Event() for _ in range(3))
    overlapped: list[bool] = []
    inside_second: list[bool] = []

    def first_load() -> None:
        with encoders._hub_offline(True):
            first_started.set()
            overlapped.append(second_started.wait(timeout=10))
        first_ended.set()

    def second_load() -> None:
        first_started.wait(timeout=10)
        with encoders._hub_offline(True):
            second_started.set()
            first_ended.wait(timeout=10)
            inside_second.append(constants.is_offline_mode())

    threads = [threading.Thread(target=first_load), threading.Thread(target=second_load)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert overlapped == [True]  # the second scope started while the first was open
    assert inside_second == [True]
    assert constants.is_offline_mode() is False


def test_hub_offline_reads_and_writes_the_flag_with_its_lock_held(monkeypatch):
    """The scope holds its lock at every read and write of the flag, at its start and at its end.

    Without the lock two loads that start or end at the same moment on two threads can both act
    on one count of open scopes. The Hub library is a stub whose flag records the state of the
    lock at each access.
    """
    from docdistance import encoders

    held: list[bool] = []

    class _Constants:
        _flag = False

        @property
        def HF_HUB_OFFLINE(self) -> bool:
            held.append(encoders._hub_lock.locked())
            return self._flag

        @HF_HUB_OFFLINE.setter
        def HF_HUB_OFFLINE(self, value: bool) -> None:
            held.append(encoders._hub_lock.locked())
            self._flag = value

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(constants=_Constants()))
    with encoders._hub_offline(True):
        at_start = len(held)
    assert at_start and len(held) > at_start  # the flag is accessed at the start and at the end
    assert all(held)


@pytest.mark.parametrize(
    "loader",
    [
        "Segmenter",
        "OpenVINOEncoder",
        "TorchEncoder",
        "OpenVINOReranker",
        "TorchReranker",
        "OpenVINONLI",
        "TorchNLI",
    ],
)
def test_every_model_constructor_reads_the_hub_inside_the_offline_scope(
    loader, monkeypatch, tmp_path
):
    """DEF-DEPS-14: each constructor has the offline flag set at every call that can reach the Hub.

    The model libraries are stubs that record the flag at ``SaT``, ``snapshot_download`` and each
    ``from_pretrained``. With ``offline=True`` every such call sees the flag set and the flag is
    unset again after the constructor; with ``offline=False`` no call sees it set.
    """
    from docdistance import config, encoders

    hub = SimpleNamespace(constants=SimpleNamespace(HF_HUB_OFFLINE=False))
    seen: list[bool] = []

    def hub_reader(*args, **kwargs):
        seen.append(hub.constants.HF_HUB_OFFLINE)
        return MagicMock()

    def snapshot_download(repo: str) -> str:
        seen.append(hub.constants.HF_HUB_OFFLINE)
        return str(tmp_path)

    hub.snapshot_download = snapshot_download
    auto = SimpleNamespace(from_pretrained=hub_reader)
    stubs = {
        "huggingface_hub": hub,
        "transformers": SimpleNamespace(
            AutoConfig=auto,
            AutoModel=auto,
            AutoModelForSequenceClassification=auto,
            AutoTokenizer=auto,
        ),
        "wtpsplit": SimpleNamespace(SaT=hub_reader),
        "openvino": MagicMock(),
        "torch": MagicMock(),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(encoders, "_require_models_extra", lambda: None)
    monkeypatch.setattr(encoders, "_set_hf_token", lambda: None)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")  # read by the NLI constructor
    for local in ("MMBERT_OPENVINO_LOCAL", "RERANKER_OPENVINO_LOCAL", "NLI_OPENVINO_LOCAL"):
        # no local model directory, so the OpenVINO constructors resolve the Hub snapshot
        monkeypatch.setattr(config, local, tmp_path / "absent")

    getattr(encoders, loader)(offline=True)
    assert seen and all(seen)
    assert hub.constants.HF_HUB_OFFLINE is False

    seen.clear()
    getattr(encoders, loader)(offline=False)
    assert seen and not any(seen)


@pytest.mark.parametrize("tokenizer_in_folder", [True, False])
def test_segmenter_loads_the_tokenizer_from_the_model_folder_when_it_is_there(
    tokenizer_in_folder, monkeypatch, tmp_path
):
    """DEF-DEPS-17: a model folder that holds ``tokenizer.json`` is also the tokenizer source of SaT.

    wtpsplit loads the tokenizer by its Hub name unless ``tokenizer_name_or_path`` is passed. A
    folder without the file, and the Hub name used when no folder is recorded, get the call of
    1.1.5: the model source alone. ``wtpsplit.SaT`` is a stub that records its arguments.
    """
    from docdistance import config, encoders

    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setitem(
        sys.modules,
        "wtpsplit",
        SimpleNamespace(SaT=lambda *args, **kwargs: calls.append((args, kwargs))),
    )
    monkeypatch.setattr(encoders, "_require_models_extra", lambda: None)
    monkeypatch.setattr(encoders, "_set_hf_token", lambda: None)
    folder = tmp_path / "sat-3l-sm"
    folder.mkdir()
    if tokenizer_in_folder:
        (folder / "tokenizer.json").write_text("{}", encoding="utf-8")

    encoders.Segmenter()  # no folder recorded: the Hub name
    settings.configure(model_paths={"sat": str(folder)})
    encoders.Segmenter()

    tokenizer = {"tokenizer_name_or_path": str(folder)} if tokenizer_in_folder else {}
    assert calls == [((config.SAT_MODEL,), {}), ((str(folder),), tokenizer)]


# the statements of the segmenter loaded by its Hub name from the cache, then `init` from the source
# folder; the text holds accented and non-Latin words, which a wrong tokenizer reads differently
_SOURCE_FOLDER_TEXT = "Zażółć gęślą jaźń. Dzień dobry, świecie! Привет, мир. The cat sat still."
_PROVISION_SCRIPT = f"""\
import json
import os

from docdistance import init
from docdistance.encoders import Segmenter

if __name__ == "__main__":
    statements = Segmenter().split({_SOURCE_FOLDER_TEXT!r})
    init("wmd", source=os.path.join(os.environ["DOCDISTANCE_HOME"], "models"))
    print("STATEMENTS", json.dumps(statements), flush=True)
"""
# the statements of the segmenter loaded from the folder that `init` recorded in docdistance.json
_SOURCE_FOLDER_LOAD_SCRIPT = f"""\
import json

from docdistance import settings
from docdistance.encoders import Segmenter

if __name__ == "__main__":
    settings.load_config_file()
    print("STATEMENTS", json.dumps(Segmenter().split({_SOURCE_FOLDER_TEXT!r})), flush=True)
"""


def _printed_statements(res: subprocess.CompletedProcess) -> list[str]:
    line = next(ln for ln in res.stdout.splitlines() if ln.startswith("STATEMENTS "))
    return json.loads(line.removeprefix("STATEMENTS "))


def _cached_sat_snapshots() -> tuple[Path, Path]:
    """The cached snapshot folders of the SaT weights and of the SaT tokenizer.

    Skips the test on a machine whose cache lacks the weights or the tokenizer.
    """
    from huggingface_hub import try_to_load_from_cache

    from docdistance import config

    weights = try_to_load_from_cache(f"segment-any-text/{config.SAT_MODEL}", "model.safetensors")
    tokenizer = try_to_load_from_cache(config.SAT_TOKENIZER, "tokenizer.json")
    if not (isinstance(weights, str) and isinstance(tokenizer, str)):
        pytest.skip("the models are not installed")
    return Path(weights).parent, Path(tokenizer).parent


def _home_with_a_sat_source_folder(tmp_path: Path, weights: Path) -> tuple[Path, Path]:
    """A home whose model folder of ``sat`` is the source folder ``init`` reads; returns both.

    The folder holds ``config.json`` and the weights only, as symbolic links to the cached
    snapshot ``weights``, so the 816 MB are not copied.
    """
    from docdistance import config

    home = tmp_path / "home"
    sat_dir = home / "models" / config.MODEL_REGISTRY["sat"]["subdir"]
    sat_dir.mkdir(parents=True)
    for name in ("config.json", "model.safetensors"):
        (sat_dir / name).symlink_to((weights / name).resolve())
    # the encoder is never loaded; its folder keeps `init` from asking the Hub for it
    mmbert_dir = home / "models" / config.MODEL_REGISTRY["mmbert"]["subdir"]
    mmbert_dir.mkdir()
    (mmbert_dir / "openvino_model.xml").write_text("<ir/>", encoding="utf-8")
    return home, sat_dir


def _statements_before_and_after_init(tmp_path: Path, home: Path) -> tuple[list[str], list[str]]:
    """Provision ``home`` from its own model folder, then load the segmenter without a cache.

    Step 1 runs with ``HF_HUB_OFFLINE=1`` and the proxies on a closed local port, step 2 in a
    process whose HuggingFace cache is an empty folder and whose proxies name the closed port.
    Returns the statements of the segmenter loaded by its Hub name in step 1 and the statements
    of the segmenter loaded from the folder in step 2.
    """
    provision = tmp_path / "provision.py"
    provision.write_text(_PROVISION_SCRIPT, encoding="utf-8")
    load = tmp_path / "load.py"
    load.write_text(_SOURCE_FOLDER_LOAD_SCRIPT, encoding="utf-8")
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    env["http_proxy"] = env["https_proxy"] = "http://127.0.0.1:9"
    env["DOCDISTANCE_HOME"] = str(home)

    provisioned = _run_script(provision, {**env, "HF_HUB_OFFLINE": "1"})
    assert provisioned.returncode == 0, provisioned.stderr

    empty_cache = tmp_path / "empty-cache"
    loaded = _run_script(
        load, {**env, "HF_HOME": str(empty_cache), "HF_HUB_CACHE": str(empty_cache / "hub")}
    )
    assert loaded.returncode == 0, loaded.stderr
    return _printed_statements(provisioned), _printed_statements(loaded)


def test_segmenter_loads_after_init_from_a_source_folder_without_the_tokenizer(tmp_path):
    """DEF-DEPS-17: after ``init --source`` the segmenter loads with an empty HuggingFace cache.

    The source folder of ``sat`` holds ``config.json`` and the weights only, as symbolic links to
    the cached snapshot, and it is the model folder of the home, so the 816 MB are not copied.
    Step 1 provisions it with ``HF_HUB_OFFLINE=1`` and the proxies on a closed local port: ``init``
    holds the Hub library online whatever the variable says, so the tokenizer has to come from the
    cache without a request. Step 2 loads the segmenter in a process whose HuggingFace cache is an
    empty folder and whose proxies name the closed port. On 1.1.18 step 2 raised
    ``ModelsNotInstalled``: wtpsplit asked for the tokenizer by its Hub name. Skipped on a machine
    whose cache lacks the SaT weights or the tokenizer.
    """
    weights, _ = _cached_sat_snapshots()
    home, _ = _home_with_a_sat_source_folder(tmp_path, weights)
    by_hub_name, loaded = _statements_before_and_after_init(tmp_path, home)
    assert len(by_hub_name) == 4
    assert loaded == by_hub_name


def test_segmenter_loads_after_init_from_a_source_folder_with_the_tokenizer_files_of_the_hub(
    tmp_path,
):
    """DEF-DEPS-17-1: a source folder that holds the tokenizer files as the Hub publishes them.

    ``tokenizer.json`` and ``tokenizer_config.json`` are copied from the cached snapshot of
    ``facebookAI/xlm-roberta-base`` beside ``config.json`` and the weights of ``sat``. Beside that
    ``config.json`` they load without the special token ids. On 1.1.24 ``init`` passed, because
    ``tokenizer.json`` was there, and the first split in step 2 raised ``TypeError``. ``init`` now
    loads the tokenizer of the folder, reads the ids and saves the tokenizer from the cache over
    the two files; the segmenter then splits the text into the statements of the load by Hub
    name. Skipped on a machine whose cache lacks the SaT weights or one of the two tokenizer
    files.
    """
    weights, tokenizer = _cached_sat_snapshots()
    if not (tokenizer / "tokenizer_config.json").exists():
        pytest.skip("the models are not installed")
    home, sat_dir = _home_with_a_sat_source_folder(tmp_path, weights)
    # copies, not links: `init` writes the tokenizer it saves over these two files
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(tokenizer / name, sat_dir / name)
    as_published = (sat_dir / "tokenizer_config.json").read_bytes()

    by_hub_name, loaded = _statements_before_and_after_init(tmp_path, home)
    assert len(by_hub_name) == 4
    assert loaded == by_hub_name
    assert (sat_dir / "tokenizer_config.json").read_bytes() != as_published
