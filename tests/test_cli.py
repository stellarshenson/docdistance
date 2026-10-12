"""CLI smoke tests - help, version and command structure render without loading any model.

The real end-to-end behaviour (segment -> embed -> distance) is exercised in
``notebooks/09-kj-docdistance-api-e2e.ipynb`` and by running the CLI against the fixtures.
"""

import pytest
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
    assert "--pairs" in res.output


def test_init_help_has_mode_and_source_flags():
    res = runner.invoke(app, ["init", "--help"], env=_WIDE)
    assert res.exit_code == 0
    assert "--source" in res.output
    assert "--backend" in res.output
    assert "--aws-profile" in res.output


def test_init_prints_the_message_of_a_missing_model_and_exits_1(monkeypatch):
    """DEF-DEPS-17: ``init`` turns ``ModelsNotInstalled`` into its message and exit code 1, the
    code the distance commands use for that error. The stub raises it as ``bootstrap.init`` does
    when the tokenizer of the segmenter is in no place it reads."""
    import re

    from docdistance import bootstrap
    from docdistance.encoders import ModelsNotInstalled

    def init(mode, **kwargs):
        raise ModelsNotInstalled(
            "the tokenizer of the segmenter, facebookAI/xlm-roberta-base, is not in the source"
        )

    monkeypatch.setattr(bootstrap, "init", init)
    res = runner.invoke(app, ["init", "wmd", "--source", "models"], env=_WIDE)
    assert res.exit_code == 1
    assert isinstance(res.exception, SystemExit)  # the command handled the error
    # Rich highlights the name with ANSI codes regardless of NO_COLOR - strip them
    plain = re.sub(r"\x1b\[[0-9;]*m", "", res.output)
    assert "error: the tokenizer of the segmenter, facebookAI/xlm-roberta-base" in plain


# the help of an option is built when the command line module is imported, so the constant is
# changed before that import, in a process of its own
_ANISOTROPY_HELP_SCRIPT = """\
import typer

from docdistance import distance

distance.ANISOTROPY_MIN_POOL = 123

from docdistance.cli import app

commands = typer.main.get_command(app).commands
for name in ("distance-semantic", "distance-structural"):
    (option,) = [param for param in commands[name].params if param.name == "anisotropy"]
    print(option.help)
"""


def test_anisotropy_help_reads_the_minimum_pool_from_the_constant(tmp_path):
    """The ``--anisotropy`` help of both symmetric commands names ``ANISOTROPY_MIN_POOL``.

    The two help texts carried the literal 71, which a change of the constant would have left
    behind.
    """
    import subprocess
    import sys

    script = tmp_path / "anisotropy_help.py"
    script.write_text(_ANISOTROPY_HELP_SCRIPT, encoding="utf-8")
    res = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=False
    )
    assert res.returncode == 0, res.stderr
    lines = res.stdout.splitlines()
    assert len(lines) == 2
    assert all("needs about 123 pooled statements" in line for line in lines)


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


def test_distance_structural_output_shows_the_matched_order_gap(monkeypatch):
    """ACC-CLI-74: the panel names the matched order gap and the matched statement count, --json
    carries both keys and --result-only still prints the bare structure_closeness; embedding is
    monkeypatched. docB is docA minus statements 3 and 6, order kept, plus one unrelated statement
    at the end: 6 matched statements against 8 and 7 statements."""
    import json
    import re

    import numpy as np

    from docdistance import settings

    E = _emb(8, seed=5)
    docs = {
        "docA": ([f"a{i}" for i in range(8)], E),
        "docB": (
            [f"a{i}" for i in (0, 1, 2, 4, 5, 7)] + ["b0"],
            np.concatenate([E[[0, 1, 2, 4, 5, 7]], _emb(1, seed=6)]),
        ),
    }
    _wire_embeddings(monkeypatch, docs)

    res = runner.invoke(app, ["distance-structural", "docA", "docB", "--json"], env=_WIDE)
    assert res.exit_code == 0, res.output
    obj = json.loads(res.stdout)
    assert obj["n_matched"] == 6
    assert obj["matched_order_gap"] < 1e-6 < obj["order_gap"]

    res = runner.invoke(app, ["distance-structural", "docA", "docB"], env=_WIDE)
    assert res.exit_code == 0, res.output
    # Rich highlights numbers with ANSI codes regardless of NO_COLOR - strip them
    plain = re.sub(r"\x1b\[[0-9;]*m", "", res.output)
    assert re.search(r"matched order gap\s+0\.0000\b", plain)
    assert re.search(r"matched statements\s+6\b", plain)

    res = runner.invoke(app, ["distance-structural", "docA", "docB", "--result-only"], env=_WIDE)
    assert res.exit_code == 0, res.output
    assert res.stdout.strip() == str(obj["structure_closeness"])
    settings.reset()


def test_distance_structural_panel_when_no_statement_is_matched(monkeypatch):
    """ACC-CLI-74 / ACC-TRANS-52: two unrelated documents - the panel states that no statement is
    matched and --json carries a null matched_order_gap; embedding is monkeypatched."""
    import json

    from docdistance import settings

    docs = {
        "docA": ([f"a{i}" for i in range(4)], _emb(4, seed=1)),
        "docB": ([f"b{i}" for i in range(3)], _emb(3, seed=2)),
    }
    _wire_embeddings(monkeypatch, docs)

    res = runner.invoke(app, ["distance-structural", "docA", "docB"], env=_WIDE)
    assert res.exit_code == 0, res.output
    assert "no statement matched" in res.output

    res = runner.invoke(app, ["distance-structural", "docA", "docB", "--json"], env=_WIDE)
    assert res.exit_code == 0, res.output
    obj = json.loads(res.stdout)
    assert obj["n_matched"] == 0 and obj["matched_order_gap"] is None
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


def test_pairs_with_anisotropy_warns_once_per_pool_below_the_minimum(monkeypatch, tmp_path):
    """ACC-GEOM-23: distance-semantic --pairs with anisotropy on logs one warning line per pre-pass
    pool below the minimum on stderr, naming the pooled statement count and the minimum of 71;
    stdout still carries the JSON line of each pair. Without --anisotropy no such line is logged.
    Embedding is monkeypatched."""
    import json
    import re

    from docdistance import settings

    docs = {
        "docA": ([f"a{i}" for i in range(4)], _emb(4, seed=1)),
        "docB": ([f"b{i}" for i in range(5)], _emb(5, seed=2)),
        "docC": ([f"c{i}" for i in range(6)], _emb(6, seed=3)),
    }
    _wire_embeddings(monkeypatch, docs)

    def pool_lines(res):
        # loguru colours its lines with ANSI codes regardless of NO_COLOR - strip them
        plain = re.sub(r"\x1b\[[0-9;]*m", "", res.stderr)
        return [line for line in plain.splitlines() if "pre-pass pool" in line]

    one_pair = tmp_path / "one-pair.tsv"
    one_pair.write_text("docA\tdocB\n")  # the 2-document TSV: one pool of 4 + 5 statements
    res = runner.invoke(
        app, ["distance-semantic", "--pairs", str(one_pair), "--anisotropy"], env=_WIDE
    )
    assert res.exit_code == 0, res.output
    lines = pool_lines(res)
    assert len(lines) == 1
    assert "holds 9 statements" in lines[0] and "minimum of 71" in lines[0]
    assert json.loads(res.stdout)["anisotropy"] is True  # the pair is scored

    two_pairs = tmp_path / "two-pairs.tsv"
    two_pairs.write_text("docA\tdocB\ndocB\tdocC\n")  # two pools: 9 and 11 statements
    res = runner.invoke(
        app, ["distance-semantic", "--pairs", str(two_pairs), "--anisotropy"], env=_WIDE
    )
    assert res.exit_code == 0, res.output
    lines = pool_lines(res)
    assert len(lines) == 2
    assert "holds 9 statements" in lines[0] and "holds 11 statements" in lines[1]
    assert len(res.stdout.splitlines()) == 2

    res = runner.invoke(app, ["distance-semantic", "--pairs", str(one_pair)], env=_WIDE)
    assert res.exit_code == 0, res.output
    assert pool_lines(res) == []
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
    template = tmp_path / "maps" / "order-{pair}.json"
    res = runner.invoke(
        app,
        ["distance-structural", "--pairs", str(tsv), "--details-json", str(template)],
        env=_WIDE,
    )
    assert res.exit_code == 0, res.output
    details = json.loads((tmp_path / "maps" / "order-first.json").read_text())
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


def test_wrt_source_pairs_and_positionals_mutually_exclusive(tmp_path):
    """ACC-CLI-37: distance-wrt-source takes two documents or --pairs FILE, as the other two
    distance commands do - both, or neither, exits 2 with the error of those commands."""
    import re

    def plain(res):
        # Rich highlights brackets with ANSI codes regardless of NO_COLOR - strip them
        return re.sub(r"\x1b\[[0-9;]*m", "", res.output)

    tsv = tmp_path / "pairs.tsv"
    tsv.write_text("docA\tdocB\n")
    res = runner.invoke(
        app, ["distance-wrt-source", "a", "b", "--pairs", str(tsv), "--source", "s"], env=_WIDE
    )
    assert res.exit_code == 2
    assert "pass two documents or --pairs FILE, not both" in plain(res)

    res = runner.invoke(app, ["distance-wrt-source", "--source", "s"], env=_WIDE)
    assert res.exit_code == 2
    assert "two documents (or --pairs FILE) required" in plain(res)


def test_wrt_source_pairs_missing_file_emits_machine_error_object(tmp_path):
    """ACC-CLI-37: batch mode is machine output on distance-wrt-source too - a --pairs path that
    cannot be read gives the one-line io_error JSON on stdout, exit 1."""
    import json

    res = runner.invoke(
        app,
        ["distance-wrt-source", "--pairs", str(tmp_path / "absent.tsv"), "--source", "s"],
        env=_WIDE,
    )
    assert res.exit_code == 1
    obj = json.loads(res.stdout)
    assert obj["error"] == "io_error"
    assert "absent.tsv" in obj["message"]


# ------------------------------------------------------------------ ACC-CLI-88: --input-format

# per distance command: the option that writes its per-pair file and the documents of one call
_FORMAT_COMMANDS = {
    "distance-semantic": ("--details-json", 2),
    "distance-structural": ("--details-json", 2),
    "distance-wrt-source": ("--source-map-json", 3),
}
_FORMAT_MODES = ["one-shot", "details", "filtered", "pairs", "pairs-details"]


def _record_embed_kwargs(monkeypatch):
    """Both modes ready and stub embeddings; returns the keyword arguments of every
    ``embed_statements`` call, one dict per document, in call order."""
    import numpy as np

    from docdistance import pipeline, settings

    docs = {
        name: ([f"{name}{i}" for i in range(3)], _emb(3, seed=seed))
        for seed, name in enumerate(("docA", "docB", "docS"), start=1)
    }
    calls = []

    def embed_statements(self, doc, **kwargs):
        calls.append(kwargs)
        return docs[doc]

    def grounding(self, doc_sents, src_sents):
        return (
            np.full((len(doc_sents), len(src_sents)), 0.5, dtype=np.float32),
            np.full(len(doc_sents), 0.5, dtype=np.float32),
        )

    settings.reset()
    settings.mark_ready("wmd")
    settings.mark_ready("wmd-wrt-source")
    monkeypatch.setattr(pipeline.DocDistance, "_ensure_base", lambda self: None)
    monkeypatch.setattr(pipeline.DocDistance, "_ensure_grounding", lambda self: None)
    monkeypatch.setattr(pipeline.DocDistance, "embed_statements", embed_statements)
    monkeypatch.setattr(pipeline.DocDistance, "_grounding", grounding)
    return calls


def _format_args(command, mode, tmp_path):
    """The arguments of ``command`` in ``mode``, without the options under test."""
    file_option, _ = _FORMAT_COMMANDS[command]
    args = [command]
    if mode.startswith("pairs"):
        tsv = tmp_path / "pairs.tsv"
        tsv.write_text("docA\tdocB\tfirst\n")
        args += ["--pairs", str(tsv)]
    else:
        args += ["docA", "docB"]
    if command == "distance-wrt-source":
        args += ["--source", "docS"]
    if mode == "details":
        args += [file_option, str(tmp_path / "details.json")]
    if mode == "pairs-details":
        args += [file_option, str(tmp_path / "details")]
    if mode == "filtered":
        args += ["--min-words", "1"]
    return args


@pytest.mark.parametrize("mode", _FORMAT_MODES)
@pytest.mark.parametrize("command", _FORMAT_COMMANDS)
def test_input_format_auto_reaches_the_pipeline(command, mode, monkeypatch, tmp_path):
    """ACC-SEGM-94: ``auto`` is the default of the pipeline, so ``--input-format auto`` reaches
    the read of every document in the call shape of a command without the option: no
    ``input_format`` argument. On the three distance commands, for a single pair on its three
    paths and for ``--pairs`` with and without per-pair files."""
    from docdistance import settings

    calls = _record_embed_kwargs(monkeypatch)
    args = _format_args(command, mode, tmp_path)
    res = runner.invoke(app, [*args, "--input-format", "auto"], env=_WIDE)
    assert res.exit_code == 0, res.output
    expected = {"min_words": 1} if mode == "filtered" else {}
    assert calls == [expected] * _FORMAT_COMMANDS[command][1]
    settings.reset()


@pytest.mark.parametrize("mode", _FORMAT_MODES)
@pytest.mark.parametrize("command", _FORMAT_COMMANDS)
def test_explicit_input_format_reaches_the_pipeline_on_every_path(
    command, mode, monkeypatch, tmp_path
):
    """ACC-SEGM-94: ``--input-format plain``, the value that returns the 1.1.5 reading, is passed
    to the read of every document of the call, on the three distance commands, for a single pair
    on its three paths and for ``--pairs`` with and without per-pair files."""
    from docdistance import settings

    calls = _record_embed_kwargs(monkeypatch)
    args = _format_args(command, mode, tmp_path)
    res = runner.invoke(app, [*args, "--input-format", "plain"], env=_WIDE)
    assert res.exit_code == 0, res.output
    expected = {"input_format": "plain"} | ({"min_words": 1} if mode == "filtered" else {})
    assert calls == [expected] * _FORMAT_COMMANDS[command][1]
    settings.reset()


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ([], {}),
        (["--input-format", "auto"], {}),
        (["--input-format", "plain"], {"input_format": "plain"}),
        (["--input-format", "markdown"], {"input_format": "markdown"}),
        (["--input-format", "html"], {"input_format": "html"}),
        (["--markdown"], {"input_format": "markdown"}),
        (["--markdown", "--input-format", "markdown"], {"input_format": "markdown"}),
    ],
    ids=[
        "default",
        "auto",
        "plain",
        "markdown",
        "html",
        "markdown flag",
        "markdown flag and value",
    ],
)
@pytest.mark.parametrize("command", _FORMAT_COMMANDS)
def test_input_format_values_and_the_markdown_flag(
    command, options, expected, monkeypatch, tmp_path
):
    """ACC-SEGM-94: every value reaches the pipeline as given; without the option, and with
    ``auto``, the read keeps the call shape of the default reader. ``--markdown`` alone equals
    ``--input-format markdown``, and the two together are no conflict."""
    from docdistance import settings

    calls = _record_embed_kwargs(monkeypatch)
    for mode in ("one-shot", "pairs"):
        calls.clear()
        res = runner.invoke(app, [*_format_args(command, mode, tmp_path), *options], env=_WIDE)
        assert res.exit_code == 0, res.output
        assert calls == [expected] * _FORMAT_COMMANDS[command][1]
    settings.reset()


@pytest.mark.parametrize("value", ["html", "auto", "plain"])
@pytest.mark.parametrize("command", _FORMAT_COMMANDS)
def test_markdown_flag_beside_another_input_format_exits_2(command, value, monkeypatch, tmp_path):
    """ACC-SEGM-94: ``--markdown`` beside ``--input-format html``, ``plain`` or an explicit
    ``auto`` is a usage error: exit code 2, both options named, no document read. The command
    line tells a given ``auto`` from the default, which ``--markdown`` alone leaves in place."""
    import re

    from docdistance import settings

    calls = _record_embed_kwargs(monkeypatch)
    for mode in ("one-shot", "pairs"):
        args = _format_args(command, mode, tmp_path)
        res = runner.invoke(app, [*args, "--markdown", "--input-format", value], env=_WIDE)
        assert res.exit_code == 2, res.output
        plain = re.sub(r"\x1b\[[0-9;]*m", "", res.output)
        assert "--markdown" in plain and f"--input-format {value}" in plain
        assert res.stdout == ""
    assert calls == []
    settings.reset()


@pytest.mark.parametrize("command", _FORMAT_COMMANDS)
def test_unknown_input_format_exits_2(command, monkeypatch, tmp_path):
    """ACC-CLI-88: a value outside ``plain``, ``markdown``, ``html``, ``auto`` is a usage error
    that names the option and the value."""
    import re

    from docdistance import settings

    calls = _record_embed_kwargs(monkeypatch)
    for mode in ("one-shot", "pairs"):
        args = _format_args(command, mode, tmp_path)
        res = runner.invoke(app, [*args, "--input-format", "pdf"], env=_WIDE)
        assert res.exit_code == 2, res.output
        plain = re.sub(r"\x1b\[[0-9;]*m", "", res.output)
        assert "Invalid value for '--input-format'" in plain and "'pdf'" in plain
    assert calls == []
    settings.reset()


@pytest.mark.parametrize("command", _FORMAT_COMMANDS)
def test_distance_help_lists_the_input_format_option(command):
    """ACC-CLI-88: the help of each distance command lists the option; its choices are the four
    values of the pipeline and its help text names the default."""
    import typer

    from docdistance import pipeline

    res = runner.invoke(app, [command, "--help"], env=_WIDE)
    assert res.exit_code == 0
    assert "--input-format" in res.output

    params = typer.main.get_command(app).commands[command].params
    (option,) = [param for param in params if param.name == "input_format"]
    assert tuple(option.type.choices) == pipeline.INPUT_FORMATS
    assert "auto (the default" in option.help
    for extension in (".md", ".markdown", ".html", ".htm"):
        assert extension in option.help
    (flag,) = [param for param in params if param.name == "markdown"]
    assert "--input-format markdown" in flag.help


@pytest.mark.parametrize(
    "options",
    [["--markdown", "--input-format", value] for value in ("plain", "html", "auto")]
    + [["--input-format", "pdf"]],
    ids=["markdown flag and plain", "markdown flag and html", "markdown flag and auto"]
    + ["unknown value"],
)
@pytest.mark.parametrize("command", _FORMAT_COMMANDS)
def test_reader_error_of_the_command_line_loads_no_model(command, options, monkeypatch, tmp_path):
    """ACC-SEGM-94: a reader error of the command line ends the command with exit code 2 before
    a ``DocDistance`` is built and before the segmenter or a model loads, for a single pair and
    for ``--pairs``."""
    from docdistance import pipeline, settings

    settings.reset()
    settings.mark_ready("wmd")
    settings.mark_ready("wmd-wrt-source")
    loads: list[str] = []
    for name in ("DocDistance", "Segmenter", "load_encoder", "load_reranker", "load_nli"):
        monkeypatch.setattr(
            pipeline, name, lambda *args, _name=name, **kwargs: loads.append(_name)
        )
    for mode in ("one-shot", "pairs"):
        res = runner.invoke(app, [*_format_args(command, mode, tmp_path), *options], env=_WIDE)
        assert res.exit_code == 2, res.output
        assert res.stdout == ""
    assert loads == []
    settings.reset()


# ------------------------------------------------------------------ ACC-CLI-104: --relevance


class _RampReranker:
    """A stub reranker whose score of query ``j`` over passage ``i`` is ``(j + 1) / (i + n)``."""

    def score_grid(self, docs, sources):
        import numpy as np

        n = len(docs) + len(sources)
        return np.array(
            [[(j + 1) / (i + n) for i in range(len(sources))] for j in range(len(docs))],
            dtype=np.float32,
        )


def _wire_relevance(monkeypatch):
    """Stub embeddings for ``docA`` (4 statements) and ``docB`` (3) plus a stub reranker; returns
    the list that records every model load attempt."""
    from docdistance import pipeline

    docs = {
        "docA": ([f"a{i}" for i in range(4)], _emb(4, seed=1)),
        "docB": ([f"b{i}" for i in range(3)], _emb(3, seed=2)),
    }
    _wire_embeddings(monkeypatch, docs)
    loads: list[str] = []

    def ensure_reranker(self):
        loads.append("reranker")
        self._reranker = _RampReranker()

    monkeypatch.setattr(pipeline.DocDistance, "_ensure_reranker", ensure_reranker)
    return loads


_RELEVANCE_KEYS = {"relevance", "relevance_target_index"}


@pytest.mark.parametrize("extra", [[], ["--json"]], ids=["plain", "json"])
@pytest.mark.parametrize("mode", ["one-shot", "pairs"])
def test_relevance_without_details_json_exits_2_before_a_model_loads(
    extra, mode, monkeypatch, tmp_path
):
    """ACC-CLI-104: ``--relevance`` without ``--details-json`` is a usage error: exit code 2, the
    message names both options, no model load attempt and no document read."""
    import re

    from docdistance import pipeline, settings

    loads = _wire_relevance(monkeypatch)
    monkeypatch.setattr(pipeline.DocDistance, "_ensure_base", lambda self: loads.append("base"))
    monkeypatch.setattr(
        pipeline.DocDistance, "embed_statements", lambda self, doc, **kw: loads.append("embed")
    )
    if mode == "pairs":
        tsv = tmp_path / "pairs.tsv"
        tsv.write_text("docA\tdocB\n")
        args = ["--pairs", str(tsv)]
    else:
        args = ["docA", "docB"]
    res = runner.invoke(app, ["distance-semantic", *args, "--relevance", *extra], env=_WIDE)
    assert res.exit_code == 2, res.output
    plain = re.sub(r"\x1b\[[0-9;]*m", "", res.output)
    assert "--relevance" in plain and "--details-json" in plain
    assert res.stdout == ""
    assert loads == []
    settings.reset()


def test_relevance_single_pair_writes_the_fields_into_the_map(monkeypatch, tmp_path):
    """ACC-CLI-104: one pair with ``--details-json`` and ``--relevance`` writes a map whose flows,
    targets and sections carry the relevance fields."""
    import json

    from docdistance import settings

    loads = _wire_relevance(monkeypatch)
    out = tmp_path / "content.json"
    res = runner.invoke(
        app,
        ["distance-semantic", "docA", "docB", "--details-json", str(out), "--relevance"],
        env=_WIDE,
    )
    assert res.exit_code == 0, res.output
    details = json.loads(out.read_text())
    assert loads == ["reranker"]
    assert all(_RELEVANCE_KEYS <= set(flow) for flow in details["flows"])
    assert all("relevance_source_index" in target for target in details["targets"])
    assert all("relevance_rank" in sec for sec in details["sections"])
    assert details["relevance"]["queries"] == "b" and "top_relevant" in details["summary"]
    settings.reset()


def test_relevance_pairs_writes_the_fields_into_every_map(monkeypatch, tmp_path):
    """ACC-CLI-104: with ``--pairs`` every row gets a map with the relevance fields."""
    import json

    from docdistance import settings

    _wire_relevance(monkeypatch)
    tsv = tmp_path / "pairs.tsv"
    tsv.write_text("docA\tdocB\tfirst\ndocA\tdocB\tsecond\n")
    out_dir = tmp_path / "maps"
    res = runner.invoke(
        app,
        ["distance-semantic", "--pairs", str(tsv), "--details-json", str(out_dir), "--relevance"],
        env=_WIDE,
    )
    assert res.exit_code == 0, res.output
    for name in ("first.json", "second.json"):
        details = json.loads((out_dir / name).read_text())
        assert all(_RELEVANCE_KEYS <= set(flow) for flow in details["flows"])
    settings.reset()


def test_run_without_relevance_writes_no_relevance_field(monkeypatch, tmp_path):
    """ACC-CLI-104: without the flag, in both modes, no map holds a relevance key and the
    reranker is not loaded."""

    from docdistance import settings

    loads = _wire_relevance(monkeypatch)
    single = tmp_path / "single.json"
    res = runner.invoke(
        app, ["distance-semantic", "docA", "docB", "--details-json", str(single)], env=_WIDE
    )
    assert res.exit_code == 0, res.output
    tsv = tmp_path / "pairs.tsv"
    tsv.write_text("docA\tdocB\tfirst\n")
    res = runner.invoke(
        app,
        ["distance-semantic", "--pairs", str(tsv), "--details-json", str(tmp_path / "maps")],
        env=_WIDE,
    )
    assert res.exit_code == 0, res.output
    for path in (single, tmp_path / "maps" / "first.json"):
        assert "relevance" not in path.read_text()
    assert loads == []
    settings.reset()


def test_distance_help_describes_the_relevance_option():
    """ACC-CLI-104: the help names the document order and the cost."""
    import re

    res = runner.invoke(app, ["distance-semantic", "--help"], env=_WIDE)
    plain = re.sub(r"\s+", " ", res.output)
    assert "--relevance" in plain
    assert "long document first" in plain and "brief second" in plain
    assert "one reranker pass per pair of statements" in plain
