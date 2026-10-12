"""Real-model check of the detail-map build against the 1.1.5 source (commit bd0002e).

Both modes encode the E12-H61 fixtures, the reorder-pool base documents and the ibm panel with the
encoder of the package they run on. ``reference`` (run with the 1.1.5 package first on PYTHONPATH)
saves the 1.1.5 structural details and flags to ``ref.json.gz`` beside this script. ``build`` (run on
the installed build) compares the build with that file and prints the readings of ACC-TRANS-15 to 19
and 77 to 79. The embeddings are not saved: a field that is not a flag differs from the 1.1.5 value
when the build encodes a document differently. ``build`` names the plain reader in every call
(``input_format="plain"``), the reader of 1.1.5; the 1.1.5 package of ``reference`` has no such
argument.

Run from the repository root:

    CI=true .venv/bin/python reports/E13-campaign/checks/detail-map/verify_build.py build
"""
import copy
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
FIX = ROOT / "data/processed/structure-fixture"
SUMMARY_DIR = ROOT / "data/interim/exec-summaries/ibm-ai-adoption/summaries"

DOC_A = """The API now supports pagination on all list endpoints.
Authentication tokens expire after 24 hours.
The dashboard loads in under two seconds on average.
Users can export reports as CSV or PDF.
The mobile app supports offline mode for cached data.
We fixed a memory leak in the background sync worker."""
DOC_B = """The API now supports pagination on all list endpoints.
Dark mode is now available across the entire interface.
Users can export reports as CSV or PDF.
The dashboard loads in under two seconds on average.
Push notifications can now be scheduled by the user.
We fixed a memory leak in the background sync worker."""
NOVEL = "The billing service migrated to the new invoicing backend."
DOC_B_INS = NOVEL + "\n" + DOC_A
SWAP = (2, 3)
SHRINK_N, SHRINK_REMOVED = 12, {2, 5, 8, 11}
SHRINK_KEPT = [i for i in range(SHRINK_N) if i not in SHRINK_REMOVED]
SHRINK_TEXT = [
    "The API now supports pagination on all list endpoints.",
    "Authentication tokens expire after 24 hours.",
    "The dashboard loads in under two seconds on average.",
    "Users can export reports as CSV or PDF.",
    "The mobile app supports offline mode for cached data.",
    "We fixed a memory leak in the background sync worker.",
    "Search results are ranked by recency as well as relevance.",
    "Audit logs retain twelve months of account activity.",
    "Webhooks retry failed deliveries with exponential backoff.",
    "The onboarding wizard was reduced from nine steps to four.",
    "Scheduled reports can be delivered to a shared mailbox.",
    "Container images are rebuilt nightly against the latest base image.",
]
DOCS = [
    ("exec-summary-gold-opus-4-6.md", "gold", "gold"),
    ("exec-summary-gold-2-opus-4-6.md", "gold-2", "gold"),
    ("exec-summary-1-opus-4-6.md", "v1", "gold"),
    ("exec-summary-2-opus-4-6.md", "v2", "gold"),
    ("exec-summary-opus-4-6.md", "opus", "gold"),
    ("exec-summary-sonnet-4-6.md", "sonnet", "gold"),
    ("exec-summary-haiku-4-5.md", "haiku", "gold"),
    ("exec-summary-adv1-a-sonnet-4-6.md", "adv1-a", "adv1"),
    ("exec-summary-adv1-b-sonnet-4-6.md", "adv1-b", "adv1"),
    ("exec-summary-adv2-a-haiku-4-5.md", "adv2-a", "adv2"),
    ("exec-summary-adv2-b-haiku-4-5.md", "adv2-b", "adv2"),
]
ADDED_RECORD = {"matched_displacement"}
ADDED_DEFINITIONS = {"moved", "matched_displacement", "kept", "removed", "added"}


def fixtures(emb: dict, texts: dict) -> dict:
    orth = np.eye(SHRINK_N, dtype=np.float32)
    s_orth = [f"a{i}" for i in range(SHRINK_N)]
    e = emb["shrink"]
    return {
        "insertion": (texts["a"], emb["a"], texts["b_ins"], emb["b_ins"]),
        "worked_example": (texts["a"], emb["a"], texts["b"], emb["b"]),
        "shrink_orth": (s_orth, orth, [f"b{i}" for i in SHRINK_KEPT], orth[SHRINK_KEPT]),
        "shrink_text": (SHRINK_TEXT, e, [SHRINK_TEXT[i] for i in SHRINK_KEPT], e[SHRINK_KEPT]),
        "reversal_orth": (s_orth, orth, s_orth[::-1], orth[::-1]),
        "reversal_text": (SHRINK_TEXT, e, SHRINK_TEXT[::-1], e[::-1]),
    }


def stripped(details: dict) -> bytes:
    """The details dict without the flag fields and without the keys the build adds."""
    d = copy.deepcopy(details)
    d.pop("version")
    for key in ADDED_DEFINITIONS:
        d["definitions"].pop(key, None)
    d["thresholds"].pop("match_cutoff", None)
    d["summary"].pop("n_moved")
    for record in d["statements"]:
        record.pop("moved")
        for key in ADDED_RECORD:
            record.pop(key, None)
    return json.dumps(d, sort_keys=True).encode()


def encode(dd, stmts: dict, bases: list[str], reader: dict) -> tuple[dict, dict]:
    """The statement texts of the three example documents and every embedding the checks read.

    ``reader`` is the reader argument of every read: the plain reader on the build, none on 1.1.5.
    """
    texts, emb = {}, {}
    for key, doc in (("a", DOC_A), ("b", DOC_B), ("b_ins", DOC_B_INS)):
        texts[key], emb[key] = dd.embed_statements(doc, **reader)
    emb["shrink"] = dd.encoder.encode(SHRINK_TEXT)
    for base in bases:
        emb[f"pool__{base}"] = dd.encoder.encode(stmts[base]["statements"])
    for fname, label, _ in DOCS:
        emb[f"ibm__{label}"] = dd.embed(SUMMARY_DIR / fname, **reader)
    return texts, emb


def pool_cases(stmts: dict, pool: dict, bases: list[str]):
    for base in bases:
        s = stmts[base]["statements"]
        for entry in pool[base]:
            if entry["k"] != 0:  # k = 0 is the document in its own order; k = -1 is the full scramble
                yield base, s, np.asarray(entry["perm"])


def main(mode: str) -> None:
    from docdistance import distance as core
    from docdistance import pipeline
    from docdistance.pipeline import MOVED_TOLERANCE, DocDistance, _build_structural_details

    print("mode", mode, "| package", Path(pipeline.__file__).parent, "| version stamp", pipeline._VERSION)
    stmts = json.loads((FIX / "statements.json").read_text())
    pool = json.loads((FIX / "reorder_pool.json").read_text())
    bases = json.loads((FIX / "meta.json").read_text())["base_docs"]

    dd = DocDistance(backend="openvino", offline=True)
    reader = {} if mode == "reference" else {"input_format": "plain"}
    texts, emb = encode(dd, stmts, bases, reader)
    if mode == "reference":
        ref = {"texts": texts, "fixtures": {}, "pool_moved": [], "pool_displacement": []}
        for name, args in fixtures(emb, texts).items():
            ref["fixtures"][name] = _build_structural_details(*args)
        for base, s, perm in pool_cases(stmts, pool, bases):
            e = emb[f"pool__{base}"]
            det = _build_structural_details(s, e, [s[j] for j in perm], e[perm])
            ref["pool_moved"].append([r["moved"] for r in det["statements"]])
            ref["pool_displacement"].append([r["displacement"] for r in det["statements"]])
        ref["ibm_smd"] = {label: core.smd(emb[f"ibm__{label}"], emb["ibm__gold"]) for _, label, _ in DOCS if label != "gold"}
        with gzip.open(OUT / "ref.json.gz", "wt", encoding="utf-8") as fh:
            json.dump(ref, fh)
        print("reference written:", {k: v["summary"]["n_moved"] for k, v in ref["fixtures"].items()}, "| pool cases", len(ref["pool_moved"]))
        return

    with gzip.open(OUT / "ref.json.gz", "rt", encoding="utf-8") as fh:
        ref = json.load(fh)
    print("build segmenter returns the statements of the 1.1.5 run:", texts == ref["texts"])

    # --- six E12-H61 fixtures: flag-stripped byte diff, fixture readings
    print("\nfixture            n_moved 1.1.5 -> build | moved build            | smd equal | order_gap equal | flag-stripped bytes equal")
    build = {}
    all_equal = True
    for name, args in fixtures(emb, texts).items():
        det = _build_structural_details(*args)
        build[name] = det
        old = ref["fixtures"][name]
        eq = stripped(det) == stripped(old)
        all_equal &= eq and det["smd"] == old["smd"] and det["order_gap"] == old["order_gap"]
        moved = [r["index"] for r in det["statements"] if r["moved"]]
        print(f"{name:18s} {old['summary']['n_moved']:>5} -> {det['summary']['n_moved']:<5}            | {str(moved):22s} | {det['smd'] == old['smd']!s:9} | {det['order_gap'] == old['order_gap']!s:15} | {eq}")
    print("ACC-TRANS-19 six fixtures, every non-flag field byte-identical:", all_equal)
    we = build["worked_example"]["statements"]
    print("worked example matched_displacement:", [r["matched_displacement"] for r in we], "| swap pair moved:", [we[i]["moved"] for i in SWAP])
    st = build["shrink_text"]["statements"]
    print("shrink_text: kept moved", sum(st[i]["moved"] for i in SHRINK_KEPT), "of 8 | removed moved", sum(st[i]["moved"] for i in SHRINK_REMOVED), "of 4",
          "| removed no_counterpart", [st[i]["no_counterpart"] for i in sorted(SHRINK_REMOVED)],
          "| 1.1.5 removed moved", [ref["fixtures"]["shrink_text"]["statements"][i]["moved"] for i in sorted(SHRINK_REMOVED)])
    print("insertion matched_displacement:", [r["matched_displacement"] for r in build["insertion"]["statements"]])

    # --- the same fixtures through the public entry point (real segmenter and encoder)
    _, entry_ins = dd.structural_distance_with_details(DOC_A, DOC_B_INS, **reader)
    _, entry_we = dd.structural_distance_with_details(DOC_A, DOC_B, **reader)
    _, entry_shrink = dd.structural_distance_with_details("\n".join(SHRINK_TEXT), "\n".join(SHRINK_TEXT[i] for i in SHRINK_KEPT), **reader)
    _, entry_rev = dd.structural_distance_with_details("\n".join(SHRINK_TEXT), "\n".join(SHRINK_TEXT[::-1]), **reader)
    print("\nentry point structural_distance_with_details:")
    print("  insertion n_moved", entry_ins["summary"]["n_moved"], "of", len(entry_ins["statements"]))
    print("  worked example moved", [r["index"] for r in entry_we["statements"] if r["moved"]], "n_moved", entry_we["summary"]["n_moved"])
    print("  12 to 8 deletion: statements", len(entry_shrink["statements"]), "targets", entry_shrink["n_statements"]["b"], "n_moved", entry_shrink["summary"]["n_moved"],
          "moved", [r["index"] for r in entry_shrink["statements"] if r["moved"]])
    print("  reversal: n_moved", entry_rev["summary"]["n_moved"], "of", len(entry_rev["statements"]))

    # --- reorder pool sweep (E12-H61) against the build
    cases = relocated = shipped_true = build_true = lost = false_fires = equal_conj = fully_matched = 0
    for case, (base, s, perm) in enumerate(pool_cases(stmts, pool, bases)):
        e = emb[f"pool__{base}"]
        n = len(s)
        det = _build_structural_details(s, e, [s[j] for j in perm], e[perm])
        moved = np.array([r["moved"] for r in det["statements"]])
        shipped = np.array(ref["pool_moved"][case])                       # the 1.1.5 flag, from the 1.1.5 source
        conjunction = shipped & (np.array(ref["pool_displacement"][case]) != 0)
        truth = np.argsort(perm) != np.arange(n)
        cases += 1
        relocated += int(truth.sum())
        shipped_true += int((shipped & truth).sum())
        build_true += int((moved & truth).sum())
        lost += int((shipped & truth & ~moved).sum())
        false_fires += int((moved & ~truth).sum())
        equal_conj += int(np.array_equal(moved, conjunction))
        fully_matched += int(all(r["matched_displacement"] is not None for r in det["statements"]))
    print(f"\nreorder pool: {cases} permutations | relocated {relocated} | flagged by 1.1.5 and relocated {shipped_true} | flagged by the build and relocated {build_true}"
          f" | lost {lost} | false fires of the build {false_fires} | flags equal to the E12-H61 conjunction {equal_conj}/{cases} | fully matched {fully_matched}/{cases}")

    # --- V on the ibm panel
    labels = [label for _, label, _ in DOCS]
    tier = {label: t for _, label, t in DOCS}
    smd_now = {label: core.smd(emb[f"ibm__{label}"], emb["ibm__gold"]) for label in labels if label != "gold"}
    gold = [smd_now[l] for l in labels if tier[l] == "gold" and l != "gold"]
    adv = [smd_now[l] for l in labels if tier[l] in ("adv1", "adv2")]
    v = sum(1 for g in gold for a in adv if g >= a)
    print(f"ibm panel: V {v}/{len(gold) * len(adv)} | smd of the 10 documents against the anchor equal to the 1.1.5 values: {smd_now == ref['ibm_smd']}")
    print("moved_tolerance", MOVED_TOLERANCE)


if __name__ == "__main__":
    main(sys.argv[1])
