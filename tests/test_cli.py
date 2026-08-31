"""CLI smoke tests - help, version and command structure render without loading any model.

The real end-to-end behaviour (segment -> embed -> distance) is exercised in
``notebooks/09-kj-docdistance-api-e2e.ipynb`` and by running the CLI against the fixtures.
"""

from typer.testing import CliRunner

from docdistance.cli import app

runner = CliRunner()
# widen so option / command names are not wrapped, and force plain output (no ANSI) so the
# substring assertions hold regardless of the ambient terminal - CI runners set FORCE_COLOR,
# which otherwise makes Rich colorize the help and splits flags like "--json" with escape codes
_WIDE = {"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"}


def test_app_help_lists_subcommands():
    res = runner.invoke(app, ["--help"], env=_WIDE)
    assert res.exit_code == 0
    assert "distance-semantic" in res.output
    assert "distance-structural" in res.output
    assert "distance-wrt-source" in res.output
    assert "init" in res.output


def test_distance_help_has_flags_and_examples():
    res = runner.invoke(app, ["distance-semantic", "--help"], env=_WIDE)
    assert res.exit_code == 0
    assert "--json" in res.output
    assert "--result-only" in res.output
    assert "--backend" in res.output
    assert "--gpu" in res.output
    assert "--details-json" in res.output
    assert "--home" in res.output
    assert "Examples" in res.output


def test_gpu_flag_errors_when_not_secured(monkeypatch):
    """--gpu must fail loudly (exit 1) when no CUDA device is visible, never silently fall back to CPU."""
    import docdistance.encoders as enc

    monkeypatch.setattr(enc, "require_gpu", _raise_gpu_unavailable)
    res = runner.invoke(app, ["distance-semantic", "x", "y", "--gpu"], env=_WIDE)
    assert res.exit_code == 1


def _raise_gpu_unavailable():
    from docdistance.encoders import GpuNotAvailable

    raise GpuNotAvailable("no CUDA device")


def test_wrt_source_help_has_source_option():
    res = runner.invoke(app, ["distance-wrt-source", "--help"], env=_WIDE)
    assert res.exit_code == 0
    assert "--source" in res.output
    assert "--result-only" in res.output
    assert "--source-map-json" in res.output


def test_init_help_has_mode_and_source_flags():
    res = runner.invoke(app, ["init", "--help"], env=_WIDE)
    assert res.exit_code == 0
    assert "--source" in res.output
    assert "--backend" in res.output
    assert "--aws-profile" in res.output


def test_distance_without_init_exits_with_clear_error(tmp_path):
    """A mode that was never init'd must fail loudly with a 'run docdistance init' message, exit 1."""
    from docdistance import settings

    settings.reset()
    env = {**_WIDE, "DOCDISTANCE_HOME": str(tmp_path)}  # empty home -> no docdistance.json
    res = runner.invoke(app, ["distance-semantic", "hello world", "goodbye world"], env=env)
    assert res.exit_code == 1
    assert "not initialized" in res.output
    settings.reset()


def test_distance_without_init_error_names_searched_path(tmp_path):
    """ACC-CLI-13: the un-init'd error must name the config path searched and the env overrides,
    so a run from the wrong folder points straight at the fix."""
    import re

    from docdistance import settings

    settings.reset()
    env = {**_WIDE, "DOCDISTANCE_HOME": str(tmp_path)}  # wrong dir - no docdistance.json here
    res = runner.invoke(app, ["distance-semantic", "hello world", "goodbye world"], env=env)
    assert res.exit_code == 1
    # Rich path-highlights the message with ANSI codes regardless of NO_COLOR - strip them
    plain = re.sub(r"\x1b\[[0-9;]*m", "", res.output)
    assert str(tmp_path / "docdistance.json") in plain  # the searched config path
    assert "DOCDISTANCE_HOME" in plain
    assert "DOCDISTANCE_CONFIG" in plain
    settings.reset()


def test_distance_home_override_passes_readiness_gate(monkeypatch, tmp_path):
    """ACC-CLI-13: --home pointing at an init'd folder satisfies the readiness gate from any cwd,
    mirroring init --home; embedding is monkeypatched so no model loads."""
    import json

    from docdistance import pipeline, settings

    cfg_home = tmp_path / "inited"
    cfg_home.mkdir()
    settings.reset()
    settings.configure(modes=["wmd"], home=str(cfg_home))
    settings.save_config_file(str(cfg_home / "docdistance.json"))
    settings.reset()  # a later process starts cold - only --home can find the config

    docs = {
        "docA": ([f"a{i}" for i in range(3)], _emb(3, seed=1)),
        "docB": ([f"b{i}" for i in range(3)], _emb(3, seed=2)),
    }
    monkeypatch.setattr(pipeline.DocDistance, "_ensure_base", lambda self: None)
    monkeypatch.setattr(pipeline.DocDistance, "embed_statements", lambda self, doc: docs[doc])

    wrong = tmp_path / "elsewhere"
    wrong.mkdir()
    out = tmp_path / "content.json"
    env = {**_WIDE, "DOCDISTANCE_HOME": str(wrong)}  # wrong dir - CliRunner restores it after
    res = runner.invoke(
        app,
        ["distance-semantic", "docA", "docB", "--home", str(cfg_home), "--details-json", str(out)],
        env=env,
    )
    assert res.exit_code == 0, res.output
    assert json.loads(out.read_text())["n_statements"] == {"a": 3, "b": 3}
    settings.reset()


def _emb(n, dim=32, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _wire_embeddings(monkeypatch, docs):
    """Mark wmd ready and swap in monkeypatched statement embeddings - no model load."""
    from docdistance import pipeline, settings

    settings.reset()
    settings.mark_ready("wmd")
    monkeypatch.setattr(pipeline.DocDistance, "_ensure_base", lambda self: None)
    monkeypatch.setattr(pipeline.DocDistance, "embed_statements", lambda self, doc: docs[doc])


def test_distance_semantic_details_json_writes_content_flows(monkeypatch, tmp_path):
    """distance-semantic --details-json writes the content-alignment transport map (flows); embedding is monkeypatched."""
    import json

    from docdistance import settings

    docs = {
        "docA": ([f"a{i}" for i in range(4)], _emb(4, seed=1)),
        "docB": ([f"b{i}" for i in range(3)], _emb(3, seed=2)),
    }
    _wire_embeddings(monkeypatch, docs)

    out = tmp_path / "content.json"
    res = runner.invoke(
        app, ["distance-semantic", "docA", "docB", "--details-json", str(out)], env=_WIDE
    )
    assert res.exit_code == 0, res.output
    assert out.exists()

    details = json.loads(out.read_text())
    assert set(details) == {
        "generator",
        "version",
        "smd",
        "closeness",
        "closeness_definition",
        "thresholds",
        "definitions",
        "summary",
        "anisotropy",
        "n_statements",
        "n_dropped",
        "drop_rules",
        "sections",
        "flows",
        "targets",
    }
    assert details["n_statements"] == {"a": 4, "b": 3}
    assert details["n_dropped"] == {"a": 0, "b": 0}  # no filter active
    assert details["drop_rules"] == {"min_words": 0, "skip_table_fragments": False}
    assert len(details["flows"]) == 4  # one flow per A statement
    for flow in details["flows"]:
        assert flow["matches"]  # every statement sends its mass somewhere
        assert isinstance(flow["changed"], bool)
    settings.reset()


def test_distance_structural_details_json_writes_order_details(monkeypatch, tmp_path):
    """distance-structural --details-json writes per-statement displacement / moved order details; embedding is monkeypatched."""
    import json

    from docdistance import settings

    docs = {
        "docA": ([f"a{i}" for i in range(4)], _emb(4, seed=1)),
        "docB": ([f"b{i}" for i in range(3)], _emb(3, seed=2)),
    }
    _wire_embeddings(monkeypatch, docs)

    out = tmp_path / "order.json"
    res = runner.invoke(
        app, ["distance-structural", "docA", "docB", "--details-json", str(out)], env=_WIDE
    )
    assert res.exit_code == 0, res.output
    assert out.exists()

    details = json.loads(out.read_text())
    assert {"order_gap", "structure_closeness", "statements"} <= set(details)
    assert details["n_statements"] == {"a": 4, "b": 3}
    assert len(details["statements"]) == 4
    for st in details["statements"]:
        assert isinstance(st["displacement"], int)
        assert isinstance(st["moved"], bool)
    settings.reset()


def test_version():
    res = runner.invoke(app, ["--version"], env=_WIDE)
    assert res.exit_code == 0
    assert "docdistance" in res.output


def test_json_mode_emits_error_object_on_missing_config(tmp_path):
    """DEF-CLI-4: --json with no docdistance.json must put a parseable JSON error object on stdout
    (exit 1), so a pipeline discarding stderr fails loudly instead of hitting JSONDecodeError."""
    import json

    from docdistance import settings

    settings.reset()
    env = {**_WIDE, "DOCDISTANCE_HOME": str(tmp_path)}  # empty home -> no docdistance.json
    res = runner.invoke(
        app, ["distance-semantic", "hello world", "goodbye world", "--json"], env=env
    )
    assert res.exit_code == 1
    obj = json.loads(res.stdout)  # stdout alone must parse - the pipeline never sees stderr
    assert obj["error"] == "not_initialized"
    assert "docdistance init" in obj["message"]
    assert str(tmp_path / "docdistance.json") in obj["message"]  # the searched config path
    settings.reset()


def test_result_only_mode_emits_error_object_on_missing_config(tmp_path):
    """DEF-CLI-4: --result-only is machine-readable too - same JSON error object on stdout, exit 1."""
    import json

    from docdistance import settings

    settings.reset()
    env = {**_WIDE, "DOCDISTANCE_HOME": str(tmp_path)}
    res = runner.invoke(
        app, ["distance-structural", "hello world", "goodbye world", "--result-only"], env=env
    )
    assert res.exit_code == 1
    obj = json.loads(res.stdout)
    assert obj["error"] == "not_initialized"
    settings.reset()


def _count_constructions(monkeypatch):
    """Wrap DocDistance.__init__ to record every construction - the batch must build exactly one."""
    from docdistance import pipeline

    constructed = []
    orig_init = pipeline.DocDistance.__init__

    def counting_init(self, *args, **kwargs):
        constructed.append(1)
        orig_init(self, *args, **kwargs)

    monkeypatch.setattr(pipeline.DocDistance, "__init__", counting_init)
    return constructed


def test_pairs_semantic_batch_one_model_load_and_jsonl(monkeypatch, tmp_path):
    """ACC-CLI-12: --pairs with 3 pairs builds one DocDistance and emits one JSON line per pair,
    carrying the TSV id or the row index; embedding is monkeypatched."""
    import json

    from docdistance import settings

    docs = {
        "docA": ([f"a{i}" for i in range(3)], _emb(3, seed=1)),
        "docB": ([f"b{i}" for i in range(3)], _emb(3, seed=2)),
        "docC": ([f"c{i}" for i in range(3)], _emb(3, seed=3)),
    }
    _wire_embeddings(monkeypatch, docs)
    constructed = _count_constructions(monkeypatch)

    tsv = tmp_path / "pairs.tsv"
    tsv.write_text("docA\tdocB\tfirst\ndocB\tdocC\ndocA\tdocC\tthird\n")
    res = runner.invoke(app, ["distance-semantic", "--pairs", str(tsv)], env=_WIDE)
    assert res.exit_code == 0, res.output
    assert len(constructed) == 1  # one model load for the whole batch

    lines = [json.loads(line) for line in res.stdout.splitlines()]
    assert [obj["pair"] for obj in lines] == ["first", "1", "third"]
    for obj in lines:
        assert "smd" in obj and "verdict" in obj
    settings.reset()


def test_pairs_structural_batch_emits_structure_fields(monkeypatch, tmp_path):
    """ACC-CLI-12: distance-structural --pairs emits one structural JSON object per pair."""
    import json

    from docdistance import settings

    docs = {
        "docA": ([f"a{i}" for i in range(3)], _emb(3, seed=1)),
        "docB": ([f"b{i}" for i in range(3)], _emb(3, seed=2)),
    }
    _wire_embeddings(monkeypatch, docs)
    constructed = _count_constructions(monkeypatch)

    tsv = tmp_path / "pairs.tsv"
    tsv.write_text("docA\tdocB\tab\ndocB\tdocA\tba\n")
    res = runner.invoke(app, ["distance-structural", "--pairs", str(tsv)], env=_WIDE)
    assert res.exit_code == 0, res.output
    assert len(constructed) == 1

    lines = [json.loads(line) for line in res.stdout.splitlines()]
    assert [obj["pair"] for obj in lines] == ["ab", "ba"]
    for obj in lines:
        assert "structure_closeness" in obj and "order_gap" in obj
    settings.reset()


def test_pairs_details_json_directory_writes_one_map_per_pair(monkeypatch, tmp_path):
    """ACC-CLI-12: in --pairs mode --details-json takes a directory and writes <pair>.json per pair."""
    import json

    from docdistance import settings

    docs = {
        "docA": ([f"a{i}" for i in range(3)], _emb(3, seed=1)),
        "docB": ([f"b{i}" for i in range(3)], _emb(3, seed=2)),
    }
    _wire_embeddings(monkeypatch, docs)

    tsv = tmp_path / "pairs.tsv"
    tsv.write_text("docA\tdocB\tfirst\ndocB\tdocA\n")
    out_dir = tmp_path / "maps"
    res = runner.invoke(
        app,
        ["distance-semantic", "--pairs", str(tsv), "--details-json", str(out_dir)],
        env=_WIDE,
    )
    assert res.exit_code == 0, res.output
    for name in ("first.json", "1.json"):
        details = json.loads((out_dir / name).read_text())
        assert details["n_statements"] == {"a": 3, "b": 3}
    settings.reset()


def test_pairs_details_json_template_writes_one_map_per_pair(monkeypatch, tmp_path):
    """ACC-CLI-12: a --details-json value containing {pair} is a per-pair filename template."""
    import json

    from docdistance import settings

    docs = {
        "docA": ([f"a{i}" for i in range(3)], _emb(3, seed=1)),
        "docB": ([f"b{i}" for i in range(3)], _emb(3, seed=2)),
    }
    _wire_embeddings(monkeypatch, docs)

    tsv = tmp_path / "pairs.tsv"
    tsv.write_text("docA\tdocB\tfirst\n")
    template = tmp_path / "order-{pair}.json"
    res = runner.invoke(
        app,
        ["distance-structural", "--pairs", str(tsv), "--details-json", str(template)],
        env=_WIDE,
    )
    assert res.exit_code == 0, res.output
    details = json.loads((tmp_path / "order-first.json").read_text())
    assert details["n_statements"] == {"a": 3, "b": 3}
    settings.reset()


def test_pairs_and_positionals_mutually_exclusive(tmp_path):
    """ACC-CLI-12: --pairs with positional documents (and neither) fails with a clear error."""
    tsv = tmp_path / "pairs.tsv"
    tsv.write_text("docA\tdocB\n")
    res = runner.invoke(app, ["distance-semantic", "a", "b", "--pairs", str(tsv)], env=_WIDE)
    assert res.exit_code == 2
    assert "not both" in res.output

    res = runner.invoke(app, ["distance-semantic"], env=_WIDE)
    assert res.exit_code == 2
    assert "required" in res.output


def test_pairs_malformed_line_names_line_number(tmp_path):
    """ACC-CLI-12: a TSV line without the two path fields fails loudly naming the line number."""
    import re

    tsv = tmp_path / "pairs.tsv"
    tsv.write_text("docA\tdocB\ndocC\n")  # line 2 has one field
    res = runner.invoke(app, ["distance-semantic", "--pairs", str(tsv)], env=_WIDE)
    assert res.exit_code == 1
    # Rich highlights paths and numbers with ANSI codes regardless of NO_COLOR - strip them
    plain = re.sub(r"\x1b\[[0-9;]*m", "", res.output)
    assert "line 2" in plain


def test_human_mode_keeps_stdout_clean_on_missing_config(tmp_path):
    """Without --json / --result-only the error stays on stderr only - stdout carries nothing."""
    from docdistance import settings

    settings.reset()
    env = {**_WIDE, "DOCDISTANCE_HOME": str(tmp_path)}
    res = runner.invoke(app, ["distance-semantic", "hello world", "goodbye world"], env=env)
    assert res.exit_code == 1
    assert res.stdout == ""
    settings.reset()


def test_pairs_missing_file_emits_machine_error_object(tmp_path):
    """A --pairs path that cannot be read funnels through _run: one-line io_error JSON on stdout, exit 1."""
    import json

    res = runner.invoke(
        app, ["distance-semantic", "--pairs", str(tmp_path / "absent.tsv")], env=_WIDE
    )
    assert res.exit_code == 1
    obj = json.loads(res.stdout)  # batch mode is machine output - stdout alone must parse
    assert obj["error"] == "io_error"
    assert "absent.tsv" in obj["message"]


def test_pairs_malformed_line_emits_machine_error_object(tmp_path):
    """A malformed TSV line funnels through _run like a missing file: io_error JSON on stdout, exit 1."""
    import json

    tsv = tmp_path / "pairs.tsv"
    tsv.write_text("docA\tdocB\ndocC\n")  # line 2 has one field
    res = runner.invoke(app, ["distance-semantic", "--pairs", str(tsv)], env=_WIDE)
    assert res.exit_code == 1
    obj = json.loads(res.stdout)
    assert obj["error"] == "io_error"
    assert "line 2" in obj["message"]
