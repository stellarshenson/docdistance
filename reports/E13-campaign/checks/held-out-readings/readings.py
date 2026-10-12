"""Three held-out readings of the experiments ledger that no notebook computes (E15-H78, E14-H76).

The ledger quotes them in the verdict lines of two records:

- E15-H78, held-out balanced accuracy of the nearest-source flag "with each scored foreign statement
  left out of the calibration": 0.9052 (cutoff from half A, scored on half B) and 0.9143 (the other
  direction)
- E14-H76, ``DR(D_sel)`` of arm a (centering plus renormalisation) over the shipped pre-pass "on the
  7 documents not read before registration": 0.5390

This script recomputes them with the fixtures, the encoder and the functions of the two notebooks
(``notebooks/experiments/E15-H78-kj-raw-nearest-source-count.ipynb`` cells 8, 10, 14 and 26,
``notebooks/experiments/E14-H76-kj-conditioned-centering-probe.ipynb`` cells 8, 10, 18 and 19). No
per-statement cost is stored, so the statements are encoded again with the plain reader
(``input_format="plain"``, the reader of both notebooks and the default reader up to 1.1.24) and the
OpenVINO INT8 encoder. Before each reading it compares a number the notebook did store
with its own value: the cutoffs and held-out balanced accuracies of the registered reading, and the
``D_sel`` of every document under both arms.

The first pair depends on a rule the ledger does not state: at which cutoff the own statements of
the held-out half are scored when the foreign statement is left out. Both rules are printed:

- rule 1 - at each leave-one-out cutoff: the share of own statements not flagged is the mean over
  the 38 cutoffs
- rule 2 - at the cutoff of the full calibration

Run from the repository root:

    HF_HUB_OFFLINE=1 CI=true .venv/bin/python reports/E13-campaign/checks/held-out-readings/readings.py
"""

import contextlib
import io
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
IBM_DIR = ROOT / "data/interim/exec-summaries/ibm-ai-adoption"
WERG_FILE = ROOT / "data/interim/ai-society-wergeland/source/source-article.md"
H78_METRICS = ROOT / "reports/E15-H78-raw-nearest-source-count-metrics.json"
H76_METRICS = ROOT / "reports/E14-H76-conditioned-centering-probe-metrics.json"

# label -> file of the 11 ibm summaries, in the order of both notebooks; the first one is the anchor
DOCS = {
    "gold": "exec-summary-gold-opus-4-6.md",
    "gold-2": "exec-summary-gold-2-opus-4-6.md",
    "v1": "exec-summary-1-opus-4-6.md",
    "v2": "exec-summary-2-opus-4-6.md",
    "opus": "exec-summary-opus-4-6.md",
    "sonnet": "exec-summary-sonnet-4-6.md",
    "haiku": "exec-summary-haiku-4-5.md",
    "adv1-a": "exec-summary-adv1-a-sonnet-4-6.md",
    "adv1-b": "exec-summary-adv1-b-sonnet-4-6.md",
    "adv2-a": "exec-summary-adv2-a-haiku-4-5.md",
    "adv2-b": "exec-summary-adv2-b-haiku-4-5.md",
}
ANCHOR = "gold"

# E15-H78: the registered split of the 7 gold-tier summaries by document, and the two directions
HALF = {"A": ("gold", "v2", "sonnet", "haiku"), "B": ("gold-2", "v1", "opus")}
DIRECTIONS = {"A→B": ("A", "B"), "B→A": ("B", "A")}  # name -> (calibration half, held-out half)
MIN_WORDS = 5  # the foreign set holds the wergeland statements of 5 words or more
GRID = np.round(np.arange(401) * 0.005, 3)  # cutoff grid 0 to 2, step 0.005
LEDGER_LEAVE_ONE_OUT = {"A→B": 0.9052, "B→A": 0.9143}
RULES = {
    "rule_1": "rule 1, own side at each leave-one-out cutoff",
    "rule_2": "rule 2, own side at the full-calibration cutoff",
}

# E14-H76: the three documents whose triples E14-H75 read before the registration
READ_BEFORE = ("v1", "adv1-a", "adv2-b")
ARM_A = "a center+renorm"  # key of the arm in the stored metrics
LEDGER_ARM_A_OVER_SHIPPED = 0.5390

HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def load_embeddings() -> tuple[dict, np.ndarray, np.ndarray]:
    """Embeddings of the 11 summaries, of the ibm source and of the foreign set (E15-H78 cell 10)."""
    from docdistance.pipeline import DocDistance

    with contextlib.redirect_stderr(io.StringIO()):
        dd = DocDistance(backend="openvino", offline=True)
        _, source = dd.embed_statements(IBM_DIR / "source/source-article.md", input_format="plain")
        emb = {
            label: dd.embed_statements(IBM_DIR / "summaries" / fname, input_format="plain")[1]
            for label, fname in DOCS.items()
        }
        # the E13-H73 foreign fixture: provenance comment and title lines removed before the split
        text = HTML_COMMENT_RE.sub("", WERG_FILE.read_text(encoding="utf-8"))
        body = "\n".join(ln for ln in text.splitlines() if not ln.startswith("# ")).strip()
        werg_texts = dd.segmenter.split(body)
        werg = dd.encoder.encode(werg_texts)
    prose = np.array([len(t.split()) >= MIN_WORDS for t in werg_texts])
    return emb, source, werg[prose]


def calibrate(own: np.ndarray, foreign: np.ndarray) -> float:
    """The cutoff rule of the record: the midpoint of the longest run of grid points at the highest
    calibration balanced accuracy, the lowest run among runs of equal length (E15-H78 cell 14)."""
    flagged_foreign = (foreign[None, :] > GRID[:, None]).sum(1)
    kept_own = (own[None, :] <= GRID[:, None]).sum(1)
    score = flagged_foreign * own.size + kept_own * foreign.size
    at_max = np.flatnonzero(score == score.max())
    runs = np.split(at_max, np.flatnonzero(np.diff(at_max) > 1) + 1)
    longest = max(runs, key=len)
    return round((float(GRID[longest[0]]) + float(GRID[longest[-1]])) / 2, 4)


def share_kept(own: np.ndarray, cutoff: float) -> float:
    """Share of own statements not flagged: cost at or below the cutoff."""
    return float((own <= cutoff).mean())


def leave_one_out(cal_own: np.ndarray, held_own: np.ndarray, foreign: np.ndarray) -> dict:
    """Held-out balanced accuracy with each scored foreign statement left out of the calibration.

    Foreign statement ``j`` is flagged at the cutoff calibrated on ``cal_own`` and the other
    foreign statements. Rule 1 scores the own statements at each of those cutoffs, rule 2 at the
    cutoff of the full calibration.
    """
    full = calibrate(cal_own, foreign)
    cutoffs = np.array([calibrate(cal_own, np.delete(foreign, j)) for j in range(foreign.size)])
    flagged = float((foreign > cutoffs).mean())
    kept_at_each = float(np.mean([share_kept(held_own, c) for c in cutoffs]))
    return {
        "full_cutoff": full,
        "full_ba": (float((foreign > full).mean()) + share_kept(held_own, full)) / 2,
        "cutoffs": sorted(set(cutoffs.tolist())),
        "foreign_flagged": int((foreign > cutoffs).sum()),
        "rule_1": (flagged + kept_at_each) / 2,
        "rule_2": (flagged + share_kept(held_own, full)) / 2,
    }


def center_renorm(pool: dict) -> dict:
    """Arm a: subtract the pooled mean and re-L2-normalise every row, no SVD (E14-H76 cell 18)."""
    keys = list(pool)
    stacked = np.concatenate([pool[k] for k in keys], 0)
    centered = stacked - stacked.mean(0)
    fixed = centered / (np.linalg.norm(centered, axis=1, keepdims=True) + 1e-9)
    bounds = np.cumsum([0] + [len(pool[k]) for k in keys])
    return {k: fixed[lo:hi].astype(np.float32) for k, lo, hi in zip(keys, bounds, bounds[1:])}


def d_sel_panel(emb: dict, source: np.ndarray) -> dict:
    """``D_sel`` of every non-anchor document against the anchor, per-triple pool, under both arms."""
    from docdistance.distance import compute_source_conditioned

    panel = {"shipped": {}, ARM_A: {}}
    for label in DOCS:
        if label == ANCHOR:
            continue
        triple = {"a": emb[label], "b": emb[ANCHOR], "s": source}
        fixed = center_renorm(triple)
        panel["shipped"][label] = compute_source_conditioned(
            triple["a"], triple["b"], triple["s"], anisotropy=True
        ).d_sel
        panel[ARM_A][label] = compute_source_conditioned(
            fixed["a"], fixed["b"], fixed["s"], anisotropy=False
        ).d_sel
    return panel


def dynamic_range(score: dict, labels: list) -> float:
    """The E04-H15 dynamic range, ``np.std`` of the ``D_sel`` values, over ``labels``."""
    return float(np.std([score[label] for label in labels]))


def report_h78(emb: dict, source: np.ndarray, foreign_emb: np.ndarray) -> None:
    """Print the registered reading beside its stored value, then the leave-one-out readings."""
    from docdistance.distance import cost_matrix

    stored = json.loads(H78_METRICS.read_text(encoding="utf-8"))
    stored = stored["arms"]["raw_kept"]["clause_1"]["whole"]["directions"]
    own = {
        half: np.concatenate([cost_matrix(emb[label], source).min(1) for label in labels])
        for half, labels in HALF.items()
    }
    foreign = cost_matrix(foreign_emb, source).min(1)
    print(f"E15-H78 | ibm source {len(source)} statements | foreign set {foreign.size} statements")
    for name, (cal, held) in DIRECTIONS.items():
        r = leave_one_out(own[cal], own[held], foreign)
        ledger = LEDGER_LEAVE_ONE_OUT[name]
        print(
            f"  {name} registered reading: cutoff {r['full_cutoff']:.4f} "
            f"(stored {stored[name]['cutoff']:.4f}), held-out balanced accuracy {r['full_ba']:.6f} "
            f"(stored {stored[name]['held_ba']:.6f})"
        )
        print(
            f"  {name} leave-one-out: cutoffs {r['cutoffs']}, "
            f"foreign statements flagged {r['foreign_flagged']} of {foreign.size}"
        )
        for rule, text in RULES.items():
            print(
                f"  {name} {text}: {r[rule]:.6f} ({r[rule]:.4f}) | ledger {ledger:.4f} | "
                f"difference {r[rule] - ledger:+.6f}"
            )


def report_h76(emb: dict, source: np.ndarray) -> None:
    """Print the dynamic-range ratio of arm a over the shipped pre-pass on the held-out documents."""
    stored = json.loads(H76_METRICS.read_text(encoding="utf-8"))["ibm_panel"]
    panel = d_sel_panel(emb, source)
    largest = max(
        abs(panel[arm][label] - stored[arm]["d_sel"][label])
        for arm in panel
        for label in panel[arm]
    )
    held_out = [label for label in DOCS if label != ANCHOR and label not in READ_BEFORE]
    ratio = dynamic_range(panel[ARM_A], held_out) / dynamic_range(panel["shipped"], held_out)
    stored_ratio = dynamic_range(stored[ARM_A]["d_sel"], held_out) / dynamic_range(
        stored["shipped"]["d_sel"], held_out
    )
    print(
        f"E14-H76 | D_sel of {len(panel['shipped'])} documents under both arms, largest "
        f"difference from the stored values {largest:.2e}"
    )
    print(f"  documents not read before registration ({len(held_out)}): {', '.join(held_out)}")
    print(
        f"  DR(D_sel) arm a / shipped on them: {ratio:.6f} ({ratio:.4f}) | ledger "
        f"{LEDGER_ARM_A_OVER_SHIPPED:.4f} | difference {ratio - LEDGER_ARM_A_OVER_SHIPPED:+.6f} | "
        f"from the stored D_sel values {stored_ratio:.6f}"
    )


if __name__ == "__main__":
    import docdistance

    print("docdistance", docdistance.__version__)
    embeddings, source_emb, foreign = load_embeddings()
    report_h78(embeddings, source_emb, foreign)
    report_h76(embeddings, source_emb)
