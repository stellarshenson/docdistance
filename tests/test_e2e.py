"""Functional end-to-end tests driving the real CLI and API on crafted documents.

The full path - argument parsing, ``_read``, segmentation, encode, ``_build_structural_details`` /
``_build_semantic_details``, JSON output - runs for real; only the model layer (SAT segmenter + mmBERT
encoder) is faked so the suite stays offline in the uv ``.venv``. The fake encoder maps each statement
text to a deterministic unit vector (identical text -> identical vector, different text -> near-orthogonal
vector), so a reorder is content-preserving (SMD ~ 0, positive order-gap, localized via the structural
displacement axis) and a reword shows up as a per-flow ``changed`` spike on the semantic content axis.
Real-model e2e lives in ``notebooks/09-kj-docdistance-api-e2e.ipynb``.
"""

import hashlib
import json

import numpy as np
import pytest
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

    assert diff["smd"] == pytest.approx(0.0, abs=2e-3)  # same statements, reordered (float floor ~3e-4)
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
        assert fl[i]["matches"][0]["cost"] < 0.01  # untouched statements map to their twin at ~0 cost


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
    floats = [float(tok) for line in res.output.splitlines() for tok in line.split() if _isfloat(tok)]
    assert floats and floats[-1] == pytest.approx(0.0, abs=1e-4)  # reorder -> SMD ~ 0


def _isfloat(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


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
