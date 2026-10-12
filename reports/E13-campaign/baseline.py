"""Default-path guardrail for the E13 build stage.

``capture`` records what the tree produces on the ibm fixture: the semantic and structural results and
detail files for all 55 summary pairs, and (with ``--grounded``) three source-conditioned results.
``verify`` recomputes them and fails when any recorded key is missing or differs. Keys the build adds
are ignored, so additive fields pass; ``--allow`` names keys whose value a registered change may alter.

Every call passes ``input_format="plain"``: the stored outputs are those of 1.1.5, which read every
document with the plain reader, and the default reader of the build reads a ``.md`` file as markdown.
The stored file was written by ``capture`` on the 1.1.5 source (commit ``bd0002e``), before this
argument existed: 1.1.5 does not accept it, and ``capture`` on a later tree replaces the 1.1.5 outputs
with the outputs of that tree.

Run from the repo root (``CI=true`` switches the OpenVINO usage telemetry off):

    CI=true .venv/bin/python reports/E13-campaign/baseline.py verify --grounded --allow moved,n_moved
"""

import argparse
import gzip
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "data/interim/exec-summaries/ibm-ai-adoption"
OUT = Path(__file__).resolve().parent / "baseline" / "default-path.json.gz"
GROUNDED = [("gold-opus-4-6", "1-opus-4-6"), ("gold-opus-4-6", "adv1-a-sonnet-4-6"), ("gold-opus-4-6", "adv2-b-haiku-4-5")]
SKIP = {"version"}  # the release stamp changes with every version, never a regression
PLAIN = {"input_format": "plain"}  # the reader of 1.1.5, which wrote the stored outputs


def compute(grounded: bool) -> dict:
    from docdistance.pipeline import DocDistance

    dd = DocDistance()
    files = {p.stem.replace("exec-summary-", ""): p for p in sorted((FIXTURE / "summaries").glob("*.md"))}
    out: dict = {"pairs": {}, "grounded": {}}
    for a, b in itertools.combinations(sorted(files), 2):
        sem, sem_details = dd.semantic_distance_with_details(files[a], files[b], **PLAIN)
        struct, struct_details = dd.structural_distance_with_details(files[a], files[b], **PLAIN)
        out["pairs"][f"{a}|{b}"] = {
            "semantic": sem.to_dict(),
            "semantic_details": sem_details,
            "structural": struct.to_dict(),
            "structural_details": struct_details,
        }
    if grounded:
        source = FIXTURE / "source" / "source-article.md"
        for a, b in GROUNDED:
            out["grounded"][f"{a}|{b}"] = dd.distance_wrt_source(files[a], files[b], source, **PLAIN).to_dict()
    return json.loads(json.dumps(out))  # normalise tuples and numpy scalars the way the file stores them


def diff(base, new, path: str, allow: set[str], found: list[str]) -> None:
    """Every key recorded in ``base`` must exist in ``new`` with an equal value; extra keys in ``new`` pass."""
    if isinstance(base, dict):
        if not isinstance(new, dict):
            found.append(f"{path}: type changed")
            return
        for key, value in base.items():
            if key in SKIP or key in allow:
                continue
            if key not in new:
                found.append(f"{path}.{key}: missing")
            else:
                diff(value, new[key], f"{path}.{key}", allow, found)
    elif isinstance(base, list):
        if not isinstance(new, list) or len(base) != len(new):
            found.append(f"{path}: list length {len(base)} -> {len(new) if isinstance(new, list) else 'n/a'}")
            return
        for i, (x, y) in enumerate(zip(base, new)):
            diff(x, y, f"{path}[{i}]", allow, found)
    elif base != new:
        found.append(f"{path}: {base!r} -> {new!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["capture", "verify"])
    parser.add_argument("--grounded", action="store_true", help="include the three source-conditioned results")
    parser.add_argument("--allow", default="", help="comma-separated key names a registered change may alter")
    args = parser.parse_args()
    if args.mode == "capture":
        data = compute(args.grounded)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(OUT, "wt", encoding="utf-8") as fh:
            json.dump(data, fh, sort_keys=True, ensure_ascii=False)
        print(f"captured {len(data['pairs'])} pairs, {len(data['grounded'])} grounded -> {OUT.relative_to(ROOT)}")
        return 0
    with gzip.open(OUT, "rt", encoding="utf-8") as fh:
        base = json.load(fh)
    if not args.grounded:
        base["grounded"] = {}
    new = compute(args.grounded)
    found: list[str] = []
    diff(base, new, "", {k for k in args.allow.split(",") if k}, found)
    for line in found[:40]:
        print("DIFF", line)
    print(f"verify: {len(base['pairs'])} pairs, {len(base['grounded'])} grounded, {len(found)} difference(s)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
