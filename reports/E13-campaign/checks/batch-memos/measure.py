"""Readings for the batch-memos package of the E13 build stage (ACC-CLI-29..43) on the real models.

Phases, run in the order given on the command line:

- ``semantic``: the shipped batch loop ``docdistance.cli._run_batch`` on the E12-H67 all-pairs TSV
  (three interleaved off/on repeats) and on the disjoint TSV (five interleaved off/on repeats),
  model load excluded, with an encoder call counter
- ``wrt``: the same loop with the source-conditioned call on the E12-H68 all-pairs TSV, one
  interleaved off/on pass, then three interleaved off/on repeats of the no-fanout TSV, with a
  grounding call counter; the 55 lines of both settings are kept
- ``arms``: the nine E12-H67 sweeps through the CLI commands with the memo on and off, sha256 of
  the JSONL and of every per-pair details file
- ``oneshot``: 55 ``distance-wrt-source --json`` subprocesses, each compared with its batch line

Every call of the build names the plain reader (``input_format="plain"``, ``--input-format
plain``), the reader of 1.1.5 and of the readings this script was written for; the markdown arm
names ``--markdown``. Memo off is ``DocDistance(memo_size=0)``. Before every timed run the 1-minute load average and the
``some avg10`` value of ``/proc/pressure/cpu`` are recorded. Every record is appended to
``readings.jsonl`` beside the outputs as it lands. The outputs go to ``out/`` beside this script.

Run from the repository root, naming the phases:

    CI=true .venv/bin/python reports/E13-campaign/checks/batch-memos/measure.py --wait semantic wrt arms oneshot
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent / "out"
H67 = ROOT / "data/interim/E12-H67"
H68 = ROOT / "data/interim/E12-H68"
IBM_SOURCE = ROOT / "data/interim/exec-summaries/ibm-ai-adoption/source/source-article.md"
LOAD_MAX = 8.0
LOAD_WAIT_S = 600  # the one wait before the first timed run
FILTER = {"markdown": False, "min_words": 18, "skip_table_fragments": True}  # the E12-H67 filter arm
NOSEG = {"markdown": False, "input_format": "plain", "min_words": 0, "skip_table_fragments": False}
PLAIN = ["--input-format", "plain"]  # the same reader on the command line


def record(kind: str, **fields) -> dict:
    row = {"kind": kind, "at": time.strftime("%H:%M:%S"), **fields}
    with (OUT / "readings.jsonl").open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    return row


def load_and_pressure() -> tuple[float, float]:
    """(1-minute load average, share of time in % a runnable task waited for a CPU over 10 s)."""
    some = Path("/proc/pressure/cpu").read_text().splitlines()[0]
    return os.getloadavg()[0], float(some.split("avg10=")[1].split()[0])


def wait_for_load() -> None:
    start = time.time()
    while os.getloadavg()[0] >= LOAD_MAX and time.time() - start < LOAD_WAIT_S:
        time.sleep(30)
    load, pressure = load_and_pressure()
    record("load-wait", waited_s=round(time.time() - start), load_1m=load, cpu_some_avg10=pressure,
           below_ceiling=load < LOAD_MAX)


class Counter:
    """Counts the calls of a bound method by replacing it on its object."""

    def __init__(self, obj, name: str):
        self.n = 0
        original = getattr(obj, name)

        def counted(*args, **kwargs):
            self.n += 1
            return original(*args, **kwargs)

        setattr(obj, name, counted)


def new_instance(models, memo_on: bool):
    """A DocDistance with an empty memo, as one CLI invocation builds, sharing the loaded models."""
    from docdistance.pipeline import MEMO_SIZE, DocDistance

    dd = DocDistance(memo_size=MEMO_SIZE if memo_on else 0)
    dd._segmenter, dd._encoder = models._segmenter, models._encoder
    dd._reranker, dd._nli = models._reranker, models._nli
    return dd


def timed_batch(rows, dist, dist_details=None, details_dir=None, extra=None, stdout=None):
    """One run of the shipped batch loop; returns (stdout text, wall seconds, load, pressure)."""
    from docdistance.cli import _run_batch

    buf = stdout or io.StringIO()
    load, pressure = load_and_pressure()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        _run_batch(rows, details_dir, dist, dist_details, extra=extra)
    wall = time.perf_counter() - t0
    return (buf.getvalue() if stdout is None else None), wall, load, pressure


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def semantic_models():
    """Loaded segmenter and encoder, an encode counter and one warm-up pair."""
    from docdistance.cli import _read_pairs
    from docdistance.pipeline import DocDistance

    models = DocDistance()
    models._ensure_base()
    encodes = Counter(models._encoder, "encode")
    warm = new_instance(models, False)
    warm.semantic_distance(*_read_pairs(str(H67 / "allpairs-55.tsv"))[0][:2], input_format="plain")  # first-call overhead
    return models, encodes


def semantic_sweep(models, encodes, name: str, tsv: str, repeats: int, order=("off", "on")) -> None:
    """Interleaved repeats of the shipped batch loop over one TSV, in the given order per repeat."""
    from docdistance.cli import _read_pairs

    rows = _read_pairs(str(H67 / tsv))[: int(os.environ.get("MEASURE_SMOKE_ROWS", 55))]
    walls, outs, counts = {"off": [], "on": []}, {"off": [], "on": []}, {}
    for repeat in range(repeats):
        for setting in order:
            dd = new_instance(models, setting == "on")
            encodes.n = 0
            out, wall, load, pressure = timed_batch(rows, lambda x, y, dd=dd: dd.semantic_distance(x, y, input_format="plain"))
            walls[setting].append(wall)
            outs[setting].append(out)
            counts[setting] = encodes.n
            record("timed-run", sweep=name, setting=setting, repeat=repeat, wall_s=round(wall, 4),
                   load_1m=load, cpu_some_avg10=pressure, encodes=encodes.n, lines=len(out.splitlines()))
    med_off, med_on = statistics.median(walls["off"]), statistics.median(walls["on"])
    record(
        "semantic-summary", sweep=name, order=list(order), rows=len(rows), slots=2 * len(rows),
        median_off_s=round(med_off, 4), median_on_s=round(med_on, 4),
        ratio_off_over_on=round(med_off / med_on, 4), delta_on_vs_off_pct=round(100 * (med_on - med_off) / med_off, 3),
        off_s=[round(w, 4) for w in walls["off"]], on_s=[round(w, 4) for w in walls["on"]],
        pairwise_delta_pct=[round(100 * (on - off) / off, 3) for off, on in zip(walls["off"], walls["on"])],
        encodes_off=counts["off"], encodes_on=counts["on"],
        jsonl_identical=len(set(outs["off"]) | set(outs["on"])) == 1, sha256=sha(outs["on"][0].encode()),
    )


def phase_semantic() -> None:
    models, encodes = semantic_models()
    semantic_sweep(models, encodes, "all-pairs-55", "allpairs-55.tsv", 3)
    semantic_sweep(models, encodes, "disjoint-55", "disjoint-55.tsv", 5)


def phase_disjoint_reversed() -> None:
    """The disjoint control again with the memo-on run first in every repeat."""
    models, encodes = semantic_models()
    semantic_sweep(models, encodes, "disjoint-55-on-first", "disjoint-55.tsv", 5, order=("on", "off"))


def phase_overhead() -> None:
    """The bookkeeping of the memo alone on the 110 slots of the disjoint TSV: key, lookup, copies, eviction."""
    import numpy as np

    from docdistance.cli import _read_pairs
    from docdistance.pipeline import MEMO_SIZE, _Memo, _memo_key

    slots = [doc for a, b, _ in _read_pairs(str(H67 / "disjoint-55.tsv")) for doc in (a, b)]
    statements, emb = ["a statement of a document"] * 13, np.ones((13, 768), dtype=np.float32)
    walls = []
    for _ in range(7):
        memo = _Memo(MEMO_SIZE)
        t0 = time.perf_counter()
        for doc in slots:
            key = _memo_key(doc, "plain", 0, False)
            if memo.get(key) is None:
                memo.put(key, (list(statements), emb.copy(), 0))
        walls.append((time.perf_counter() - t0) * 1e3)
    record("overhead", slots=len(slots), bound=MEMO_SIZE, entries_kept=len(memo._entries),
           median_ms=round(statistics.median(walls), 3), runs_ms=[round(w, 3) for w in walls])


def phase_wrt() -> None:
    from docdistance.cli import _read_pairs
    from docdistance.pipeline import DocDistance

    models = DocDistance()
    models._ensure_grounding()
    encodes = Counter(models._encoder, "encode")
    grids = Counter(models._reranker, "score_grid")
    source = str(IBM_SOURCE)

    def run(name: str, rows, setting: str, repeat: int, keep: Path | None = None) -> float:
        dd = new_instance(models, setting == "on")
        encodes.n = grids.n = 0
        stdout = keep.open("w", buffering=1) if keep else None
        out, wall, load, pressure = timed_batch(
            rows, lambda x, y: dd.distance_wrt_source(x, y, source, anisotropy=True, **NOSEG), stdout=stdout
        )
        if stdout:
            stdout.close()
        record("timed-run", sweep=name, setting=setting, repeat=repeat, wall_s=round(wall, 3), load_1m=load,
               cpu_some_avg10=pressure, encodes=encodes.n, grids=grids.n, slots=2 * len(rows),
               grid_hits=2 * len(rows) - grids.n, embed_hits=3 * len(rows) - encodes.n)
        return wall

    rows = _read_pairs(str(H68 / "allpairs-55.tsv"))
    walls = {s: run("wrt-all-pairs-55", rows, s, 0, OUT / f"wrt-batch-{s}.jsonl") for s in ("off", "on")}
    on, off = (OUT / "wrt-batch-on.jsonl").read_bytes(), (OUT / "wrt-batch-off.jsonl").read_bytes()
    record("wrt-summary", sweep="wrt-all-pairs-55", off_s=round(walls["off"], 3), on_s=round(walls["on"], 3),
           ratio_off_over_on=round(walls["off"] / walls["on"], 4), lines_on=len(on.splitlines()),
           lines_off=len(off.splitlines()), on_equals_off=on == off, sha256_on=sha(on), sha256_off=sha(off))

    rows = _read_pairs(str(H68 / "nofanout-2.tsv"))
    walls = {"off": [], "on": []}
    for repeat in range(3):
        for setting in ("off", "on"):
            walls[setting].append(run("wrt-nofanout-2", rows, setting, repeat))
    med_off, med_on = statistics.median(walls["off"]), statistics.median(walls["on"])
    record("wrt-summary", sweep="wrt-nofanout-2", median_off_s=round(med_off, 3), median_on_s=round(med_on, 3),
           delta_on_vs_off_pct=round(100 * (med_on - med_off) / med_off, 3),
           off_s=[round(w, 3) for w in walls["off"]], on_s=[round(w, 3) for w in walls["on"]],
           pairwise_delta_pct=[round(100 * (on - off) / off, 3) for off, on in zip(walls["off"], walls["on"])])


def phase_arms() -> None:
    """The nine E12-H67 sweeps through the CLI commands, memo on and off."""
    from typer.testing import CliRunner

    from docdistance.cli import app
    from docdistance.pipeline import DocDistance

    runner = CliRunner()
    init = DocDistance.__init__

    def memo_off_init(self, *args, **kwargs):
        init(self, *args, **{**kwargs, "memo_size": 0})

    filt = ["--min-words", "18", "--skip-table-fragments"]
    arms = {
        "all-pairs 55": ("distance-semantic", "allpairs-55.tsv", PLAIN, False),
        "all-pairs 55 +details": ("distance-semantic", "allpairs-55.tsv", PLAIN, True),
        "1-source-x2": ("distance-semantic", "source-x2.tsv", PLAIN, False),
        "1-source-x5": ("distance-semantic", "source-x5.tsv", PLAIN, False),
        "1-source-x11": ("distance-semantic", "source-x11.tsv", PLAIN, False),
        "disjoint 55": ("distance-semantic", "disjoint-55.tsv", PLAIN, False),
        "all-pairs 55 structural +details": ("distance-structural", "allpairs-55.tsv", PLAIN, True),
        "all-pairs 55 filtered": ("distance-semantic", "allpairs-55.tsv", [*filt, *PLAIN], False),
        "all-pairs 55 markdown": ("distance-semantic", "allpairs-55.tsv", ["--markdown"], False),
    }
    identical = 0
    for name, (command, tsv, options, details) in arms.items():
        digest = {}
        for setting in ("on", "off"):
            ddir = OUT / "details" / f"{name.replace(' ', '_')}-{setting}"
            args = [command, "--pairs", str(H67 / tsv), *options, *(["--details-json", str(ddir)] if details else [])]
            DocDistance.__init__ = init if setting == "on" else memo_off_init
            try:
                res = runner.invoke(app, args)
            finally:
                DocDistance.__init__ = init
            assert res.exit_code == 0, res.output
            files = {p.name: sha(p.read_bytes()) for p in sorted(ddir.glob("*.json"))} if details else {}
            digest[setting] = {"jsonl": sha(res.stdout.encode()), "lines": len(res.stdout.splitlines()),
                               "details_files": len(files), "details": sha(json.dumps(files, sort_keys=True).encode()),
                               "files": files}
        same = digest["on"]["jsonl"] == digest["off"]["jsonl"] and digest["on"]["files"] == digest["off"]["files"]
        identical += same
        record("arm", arm=name, command=command, lines=digest["on"]["lines"], jsonl_sha256_on=digest["on"]["jsonl"],
               jsonl_sha256_off=digest["off"]["jsonl"], details_files=digest["on"]["details_files"],
               details_sha256_on=digest["on"]["details"], details_sha256_off=digest["off"]["details"], identical=same)
    record("arms-summary", arms=len(arms), identical=identical)


def phase_oneshot() -> None:
    """55 one-shot CLI subprocesses against the batch lines of both settings."""
    from docdistance.cli import _read_pairs

    rows = _read_pairs(str(H68 / "allpairs-55.tsv"))
    out = OUT / "oneshot"
    out.mkdir(exist_ok=True)

    def one(n: int) -> None:
        target = out / f"{n}.json"
        if target.exists() and target.stat().st_size:
            return
        a, b, _ = rows[n]
        res = subprocess.run(
            [str(ROOT / ".venv/bin/docdistance"), "distance-wrt-source", a, b, "--source", str(IBM_SOURCE), "--json", *PLAIN],
            cwd=ROOT, env={**os.environ, "CI": "true"}, capture_output=True, check=False,
        )
        if res.returncode != 0:
            raise RuntimeError(f"one-shot {n} exit {res.returncode}: {res.stderr[-400:]!r}")
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(res.stdout)
        tmp.rename(target)
        print(f"one-shot {n} done", flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, range(len(rows))))

    for setting in ("on", "off"):
        equal, mismatched = 0, []
        for n, line in enumerate((OUT / f"wrt-batch-{setting}.jsonl").read_text().splitlines()):
            obj = json.loads(line)
            pair = obj.pop("pair")
            assert pair == rows[n][2]
            printed = (json.dumps(obj, indent=2) + "\n").encode()
            reference = (out / f"{n}.json").read_bytes()
            if sha(printed) == sha(reference):
                equal += 1
            else:
                mismatched.append(pair)
        record("oneshot-summary", setting=setting, equal=equal, of=len(rows), mismatched=mismatched)


def phase_cli_counts() -> None:
    """The two all-pairs sweeps through the CLI commands themselves, with model call counters."""
    from typer.testing import CliRunner

    from docdistance import pipeline
    from docdistance.cli import app

    counters = {}
    loaders = {name: getattr(pipeline, name) for name in ("load_encoder", "load_reranker")}

    def counting(name: str, method: str):
        def load(*args, **kwargs):
            model = loaders[name](*args, **kwargs)
            counters[name] = Counter(model, method)
            return model

        return load

    pipeline.load_encoder = counting("load_encoder", "encode")
    pipeline.load_reranker = counting("load_reranker", "score_grid")
    runner = CliRunner()
    try:
        res = runner.invoke(app, ["distance-semantic", "--pairs", str(H67 / "allpairs-55.tsv"), *PLAIN])
        assert res.exit_code == 0, res.output
        record("cli-count", command="distance-semantic --pairs", tsv="E12-H67 allpairs-55.tsv",
               lines=len(res.stdout.splitlines()), slots=110, encodes=counters["load_encoder"].n,
               jsonl_sha256=sha(res.stdout.encode()))

        load, pressure = load_and_pressure()
        t0 = time.perf_counter()
        res = runner.invoke(
            app, ["distance-wrt-source", "--pairs", str(H68 / "allpairs-55.tsv"), "--source", str(IBM_SOURCE), *PLAIN]
        )
        wall = time.perf_counter() - t0
        assert res.exit_code == 0, res.output
        (OUT / "wrt-cli.jsonl").write_bytes(res.stdout.encode())
        reference = (OUT / "wrt-batch-on.jsonl").read_bytes()
        record("cli-count", command="distance-wrt-source --pairs", tsv="E12-H68 allpairs-55.tsv",
               lines=len(res.stdout.splitlines()), slots=110, grids=counters["load_reranker"].n,
               encodes=counters["load_encoder"].n, wall_with_model_load_s=round(wall, 2), load_1m=load,
               cpu_some_avg10=pressure, jsonl_sha256=sha(res.stdout.encode()),
               equals_batch_loop_output=res.stdout.encode() == reference,
               ids_in_row_order=[json.loads(line)["pair"] for line in res.stdout.splitlines()]
               == [json.loads(line)["pair"] for line in reference.decode().splitlines()])
    finally:
        pipeline.load_encoder, pipeline.load_reranker = loaders["load_encoder"], loaders["load_reranker"]


PHASES = {"semantic": phase_semantic, "wrt": phase_wrt, "arms": phase_arms, "oneshot": phase_oneshot,
          "disjoint-reversed": phase_disjoint_reversed, "overhead": phase_overhead,
          "cli-counts": phase_cli_counts}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phases", nargs="+", choices=list(PHASES))
    parser.add_argument("--wait", action="store_true", help="wait at most 10 minutes for a load below 8 first")
    cli_args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    os.chdir(ROOT)
    os.environ["DOCDISTANCE_HOME"] = str(ROOT)
    from importlib.metadata import version

    record("start", phases=cli_args.phases, docdistance=version("docdistance"), uptime=subprocess.run(
        ["uptime"], capture_output=True, text=True, check=False).stdout.strip())
    if cli_args.wait:
        wait_for_load()
    for phase in cli_args.phases:
        PHASES[phase]()
        record("phase-done", phase=phase)
    sys.exit(0)
