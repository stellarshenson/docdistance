"""The default reader against the plain reader on the ibm fixture (ACC-SEGM-95).

``input_format`` defaults to ``auto``, which reads a ``.md`` file with the markdown reader; 1.1.5
read every document with the plain reader. On the 55 pairs of
``data/interim/E12-H67/allpairs-55.tsv`` and on the 3 grounded triples of ``baseline.py`` this
script calls each scoring method without ``input_format`` and with ``input_format="plain"``. Per
output field of ``distance-semantic``, ``distance-structural`` and ``distance-wrt-source`` (the keys
of their ``--json`` output) it prints the number of rows whose two values differ and the largest
absolute difference; a field whose differing values are not numbers has no such difference.

It then compares the 55 default semantic distances with the ``auto`` arm that E16-H81 stored on
1.1.21, where the content detector chose the markdown reader for the same 11 files: in the pair
orientation of that arm, and in the row orientation of the pairs file. The stored arm is read from
the copy of the metrics file as recorded on 1.1.21, which no notebook run rewrites. The script
exits with code 1 when one of the 55 distances differs from that arm in its pair orientation.

One ``DocDistance`` with both memos on scores every row, as a ``--pairs`` batch does. The readings
go to stdout; the log lines of the library (the marker warning of each plain read of the source
article) go to stderr.

Run from the repository root:

    CI=true HF_HUB_OFFLINE=1 .venv/bin/python reports/E13-campaign/checks/default-reader/diff.py 2>&1 | tee logs/extension-default-diff.log
"""

import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[4]
PAIRS_TSV = ROOT / "data/interim/E12-H67/allpairs-55.tsv"
BASELINE_SCRIPT = ROOT / "reports/E13-campaign/baseline.py"
H81_METRICS = ROOT / "reports/E13-campaign/results/E16-H81-metrics-recorded-1.1.21.json"
PLAIN = {"input_format": "plain"}


def gap(default, plain) -> float | None:
    """The absolute difference of two field values; ``None`` when one of them is not a number."""
    numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (default, plain))
    return abs(default - plain) if numbers else None


def report(command: str, default_rows: list[dict], plain_rows: list[dict]) -> None:
    """Print, per output field, the rows whose two values differ and the largest absolute difference."""
    print(f"{command}: {len(default_rows)} rows, default call against input_format=\"plain\"")
    for field in default_rows[0]:
        differing = [(d[field], p[field]) for d, p in zip(default_rows, plain_rows) if d[field] != p[field]]
        gaps = [g for g in (gap(d, p) for d, p in differing) if g is not None]
        largest = f"{max(gaps):.6e}" if gaps else ("0" if not differing else "no numeric difference")
        print(f"  {field:22s} rows that differ {len(differing):2d} of {len(default_rows)} | largest absolute difference {largest}")


def equal_floats(now: list[float], stored: list[float]) -> str:
    """How many values of ``now`` equal their stored value as floats, and the largest difference."""
    equal = sum(a == b for a, b in zip(now, stored))
    largest = max(abs(a - b) for a, b in zip(now, stored))
    return f"{equal} of {len(stored)} equal as floats | largest absolute difference {largest:.6e}"


def main() -> int:
    import docdistance
    from docdistance.cli import _read_pairs
    from docdistance.pipeline import MEMO_SIZE, DocDistance

    print("docdistance", docdistance.__version__)
    dd = DocDistance(memo_size=MEMO_SIZE)
    rows = _read_pairs(str(PAIRS_TSV))
    for command, call in (("distance-semantic", dd.semantic_distance), ("distance-structural", dd.structural_distance)):
        default_rows = [call(a, b).to_dict() for a, b, _ in rows]
        plain_rows = [call(a, b, **PLAIN).to_dict() for a, b, _ in rows]
        report(command, default_rows, plain_rows)
        if command == "distance-semantic":
            default_smd = {frozenset((a, b)): row["smd"] for (a, b, _), row in zip(rows, default_rows)}

    baseline = runpy.run_path(str(BASELINE_SCRIPT))
    fixture = baseline["FIXTURE"]
    files = {p.stem.replace("exec-summary-", ""): p for p in sorted((fixture / "summaries").glob("*.md"))}
    source = fixture / "source" / "source-article.md"
    triples = [(files[a], files[b]) for a, b in baseline["GROUNDED"]]
    default_rows = [dd.distance_wrt_source(a, b, source).to_dict() for a, b in triples]
    plain_rows = [dd.distance_wrt_source(a, b, source, **PLAIN).to_dict() for a, b in triples]
    report("distance-wrt-source", default_rows, plain_rows)

    stored = json.loads(H81_METRICS.read_text(encoding="utf-8"))
    documents = stored["fixture"]["documents"]
    pairs = [tuple(ROOT / documents[label] for label in pair.split("|")) for pair in stored["fixture"]["pairs"]]
    auto = stored["arms"]["auto"]["pair_distances_55"]
    in_stored_orientation = [dd.semantic_distance(a, b).smd for a, b in pairs]
    in_row_orientation = [default_smd[frozenset((str(a), str(b)))] for a, b in pairs]
    print(f"default semantic distance against the auto arm of E16-H81 (docdistance {stored['environment']['docdistance']}):")
    print(f"  pair orientation of the stored arm: {equal_floats(in_stored_orientation, auto)}")
    print(f"  row orientation of allpairs-55.tsv: {equal_floats(in_row_orientation, auto)}")
    return 0 if in_stored_orientation == auto else 1


if __name__ == "__main__":
    sys.exit(main())
