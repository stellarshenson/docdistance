"""Real-model check of the source map of the build against the 1.1.5 source (commit bd0002e).

``reference`` (1.1.5 package first on PYTHONPATH) saves the map of one ibm triple; ``build`` (the
installed build) computes the map of the same triple, removes the keys the build adds and compares.
``ref_source_map.json`` beside this script is the saved reference. ``build`` names the plain reader
(``input_format="plain"``), the reader of 1.1.5; the 1.1.5 package of ``reference`` has no such
argument.

Run from the repository root:

    CI=true .venv/bin/python reports/E13-campaign/checks/detail-map/verify_source_map.py build
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
FIXTURE = ROOT / "data/interim/exec-summaries/ibm-ai-adoption"
A = FIXTURE / "summaries/exec-summary-gold-opus-4-6.md"
B = FIXTURE / "summaries/exec-summary-adv1-a-sonnet-4-6.md"
S = FIXTURE / "source/source-article.md"
ADDED_HEADER = ("source_support_cutoff", "n_unsupported")
ADDED_RECORD = ("nearest_source_index", "nearest_source_cost", "unsupported")


def main(mode: str) -> None:
    from docdistance import pipeline
    from docdistance.pipeline import DocDistance

    print("mode", mode, "| package", Path(pipeline.__file__).parent)
    dd = DocDistance()
    reader = {} if mode == "reference" else {"input_format": "plain"}
    result, smap = dd.distance_wrt_source_with_map(A, B, S, **reader)
    smap = json.loads(json.dumps(smap))
    if mode == "reference":
        (OUT / "ref_source_map.json").write_text(json.dumps({"map": smap, "result": result.to_dict()}))
        print("reference written: header keys", [k for k in smap if k not in ("a", "b")], "| record keys", sorted(smap["a"][0]))
        return
    ref = json.loads((OUT / "ref_source_map.json").read_text())
    print("header keys", [k for k in smap if k not in ("a", "b")])
    print("record keys", sorted(smap["a"][0]))
    print("source_support_cutoff", smap["source_support_cutoff"], "| n_unsupported", smap["n_unsupported"],
          "| result n_unsupported_a / b", result.n_unsupported_a, result.n_unsupported_b,
          "| n_statements", smap["n_statements"])
    print("header counts equal the result of the same call:",
          smap["n_unsupported"] == {"a": result.n_unsupported_a, "b": result.n_unsupported_b})
    print("unsupported a:", [(r["index"], r["nearest_source_index"], r["nearest_source_cost"]) for r in smap["a"] if r["unsupported"]])
    print("unsupported b:", [(r["index"], r["nearest_source_index"], r["nearest_source_cost"]) for r in smap["b"] if r["unsupported"]])
    old = copy.deepcopy(smap)
    for key in ADDED_HEADER:
        old.pop(key)
    for side in ("a", "b"):
        for record in old[side]:
            for key in ADDED_RECORD:
                record.pop(key)
    print("map without the added keys byte-identical to the 1.1.5 map:",
          json.dumps(old, sort_keys=True) == json.dumps(ref["map"], sort_keys=True))
    shared = {k: v for k, v in result.to_dict().items() if k in ref["result"]}
    print("result fields of 1.1.5 equal:", shared == ref["result"], "| fields", len(ref["result"]))


if __name__ == "__main__":
    main(sys.argv[1])
