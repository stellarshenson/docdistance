"""ACC-MAP-105: the relevance the shipped command writes equals the E18-H87 reading.

Runs ``docdistance distance-semantic --pairs ... --details-json ... --relevance`` once on the
report of ``data/processed/relevance-fixture`` against the 12 briefs of its ``e18`` set, reads
``relevance`` and ``section`` of every flow from the map files, applies the labels of
``e18/labels.json`` and prints the mean AUC beside ``summary_new_briefs_headline`` of
``reports/E18-reranker-relevance-metrics.json``.

Labels: the leaf heading is the last element of the section path split at ``" > "``; statements
of the neutral sections are left out; a statement is positive when its leaf is a target section
of the brief, negative otherwise. The AUC is the Mann-Whitney statistic, ties count one half.
The map rounds scores to 4 decimals, which creates ties, so a difference of the two means up to
0.005 is accepted.

Run from the repository root:

    CI=true .venv/bin/python reports/E18-relevance/checks/shipped_relevance.py \
        2>&1 | tee logs/E18-shipped-relevance.log

Exit code 0 when the means differ by at most 0.005, 1 otherwise.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "data/processed/relevance-fixture"
METRICS = ROOT / "reports/E18-reranker-relevance-metrics.json"
TOLERANCE = 0.005
SECTION_SEPARATOR = " > "


def auc(score: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUC of ``score`` for the positives against the rest; ties count one half."""
    ranks = rankdata(score)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def run_command(labels: dict, maps: Path) -> None:
    """One ``--pairs`` batch: the report against each brief, one map file per brief."""
    report = (FIXTURE / "e18" / labels["report"]).resolve()
    rows = [
        f"{report}\t{(FIXTURE / 'e18' / spec['file']).resolve()}\t{brief}"
        for brief, spec in labels["briefs"].items()
    ]
    pairs = maps / "pairs.tsv"
    pairs.write_text("\n".join(rows) + "\n", encoding="utf-8")
    cmd = [sys.executable, "-m", "docdistance", "distance-semantic", "--pairs", str(pairs)]
    subprocess.run([*cmd, "--details-json", str(maps), "--relevance"], check=True, cwd=ROOT)


def brief_auc(flows: list[dict], spec: dict, neutral: list[str]) -> float:
    """AUC of the written relevance of one brief's map, neutral sections left out."""
    leaf = np.array([f["section"].split(SECTION_SEPARATOR)[-1] for f in flows])
    keep = ~np.isin(leaf, neutral)
    score = np.array([f["relevance"] for f in flows])
    return auc(score[keep], np.isin(leaf, spec["target_sections"])[keep])


def main() -> int:
    labels = json.loads((FIXTURE / "e18/labels.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        maps = Path(tmp)
        run_command(labels, maps)
        aucs = {}
        for brief, spec in labels["briefs"].items():
            flows = json.loads((maps / f"{brief}.json").read_text(encoding="utf-8"))["flows"]
            aucs[brief] = brief_auc(flows, spec, labels["neutral_sections"])
    shipped = float(np.mean(list(aucs.values())))
    measured = json.loads(METRICS.read_text(encoding="utf-8"))["summary_new_briefs_headline"][
        "reranker"
    ]["mean_auc"]
    for brief, value in aucs.items():
        print(f"{brief}  AUC {value:.4f}")
    print(f"shipped mean AUC over {len(aucs)} briefs: {shipped:.4f}")
    print(f"E18-H87 notebook mean AUC:               {measured:.4f}")
    print(f"difference: {shipped - measured:+.4f} (tolerance {TOLERANCE})")
    return 0 if abs(shipped - measured) <= TOLERANCE else 1


if __name__ == "__main__":
    sys.exit(main())
