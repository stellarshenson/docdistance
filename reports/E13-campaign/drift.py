"""Metric drift of the re-executed experiment notebooks against the committed readings.

For every ``reports/*.json`` that ``git`` reports as modified, compare the working-tree file with the
``HEAD`` version leaf by leaf: numeric leaves by relative difference, other leaves by equality.

    .venv/bin/python reports/E13-campaign/drift.py [--tol 0.01]

With ``--against FOLDER`` the working-tree file is compared with the copy of the same name in that
folder, for every ``*.json`` the folder holds, in place of the ``HEAD`` version.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def leaves(node, path=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from leaves(value, f"{path}[{i}]")
    else:
        yield path, node


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tol", type=float, default=0.01, help="relative difference above which a number is listed")
    parser.add_argument("--show", type=int, default=6, help="listed leaves per file")
    parser.add_argument("--against", type=Path, help="folder of earlier copies, compared in place of HEAD")
    args = parser.parse_args()
    if args.against:
        changed = [f"reports/{path.name}" for path in args.against.glob("*.json")]
    else:
        changed = subprocess.run(
            ["git", "diff", "--name-only", "HEAD", "--", "reports"], cwd=ROOT, capture_output=True, text=True
        ).stdout.split()
    for name in sorted(n for n in changed if n.endswith(".json")):
        if args.against:
            old = json.loads((args.against / Path(name).name).read_text())
        else:
            old = json.loads(
                subprocess.run(["git", "show", f"HEAD:{name}"], cwd=ROOT, capture_output=True, text=True).stdout
            )
        new = json.loads((ROOT / name).read_text())
        a, b = dict(leaves(old)), dict(leaves(new))
        numeric, text, over = 0, [], []
        for key in a.keys() & b.keys():
            x, y = a[key], b[key]
            if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                if x != y:
                    text.append((key, x, y))
                continue
            numeric += 1
            rel = abs(x - y) / max(abs(x), abs(y), 1e-12)
            if rel > args.tol:
                over.append((rel, key, x, y))
        over.sort(reverse=True)
        print(f"{name}: {numeric} shared numbers, {len(over)} above {args.tol:.0%}, {len(text)} non-numeric changes, "
              f"{len(a.keys() - b.keys())} keys gone, {len(b.keys() - a.keys())} keys new")
        for rel, key, x, y in over[: args.show]:
            print(f"    {key}: {x} -> {y} ({rel:.1%})")
        for key, x, y in text[: args.show]:
            print(f"    {key}: {str(x)[:60]!r} -> {str(y)[:60]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
