---
name: docdistance
description: Install and run the docdistance library (PyPI `docdistance`, CLI `docdistance`) to measure how far two documents are - semantic content distance and structural/order distance - with interpretable per-statement diffs. Use when an agent must check whether a converted, extracted, summarized, or rewritten document preserved the original, find what changed between two documents (meaning vs order), or get an embedding-grounded document distance without model logits/KL. Triggers - "how different are these two documents", "did the conversion/extraction drift", "compare doc A to doc B", "semantic distance between documents", "score many document pairs in batch", "install/use docdistance". Ships a preflight that checks install and upgrades.
---

# docdistance

Distance between two documents, grounded in word/statement embeddings (Word Mover's Distance / Optimal Transport). Two axes: content (meaning) and structure (order). Per-statement diff localizes what changed.

Package `docdistance` on PyPI. Import `docdistance`. CLI `docdistance`. Python 3.13 only.

## Preflight - check install, upgrade if stale

Run first, every session, before use:

```bash
# 1. installed? which version?
python3 -c "import docdistance; print('docdistance', docdistance.__version__)" 2>/dev/null || echo "not installed"
# 2. install-or-upgrade to latest (idempotent - installs if missing, upgrades if behind)
pip install -U docdistance          # inside a uv project: uv pip install -U docdistance
```

- `-U` covers both cases - fresh install and upgrade - so one command is enough after the check
- pulling models from S3 later → `pip install -U 'docdistance[s3]'`
- python 3.13 required (`requires-python ~=3.13`); on other pythons install fails - use a 3.13 venv

## Provision models once - required before any distance

docdistance needs models on disk; every distance call fails until provisioned.

```bash
export DOCDISTANCE_HOME="$PWD"          # where docdistance.json + model mirror live - keep it stable
docdistance init                        # 'wmd' mode: semantic + structural, from HuggingFace, openvino INT8 on CPU
docdistance init wmd-wrt-source         # adds reranker + NLI - only if using distance-wrt-source
```

- default source HuggingFace (no flag); S3 mirror → `docdistance init --source s3://bucket/prefix --aws-profile NAME` (needs `docdistance[s3]`)
- `--backend torch` fetches GPU weights; default `openvino` runs on any CPU, no GPU
- re-run in the same `DOCDISTANCE_HOME` is a no-op once `docdistance.json` exists

## Use it - CLI (default agentic path)

`A` and `B` are file paths or raw text.

```bash
docdistance distance-semantic   A B              # content closeness - Statement Mover's Distance
docdistance distance-structural A B              # order closeness - OPW order-gap
docdistance distance-wrt-source A B --source S   # source-conditioned d(A,B|S) - needs wmd-wrt-source mode
```

Machine-readable, for an agent to parse:

- `--json` → full result dict (`smd`/`closeness`/`verdict`, or `order_gap`/`structure_closeness`/`verdict`)
- `--result-only` → one bare number, pipe it straight
- errors in `--json`/`--result-only` mode still write a one-line JSON `{error, message}` object to stdout (exit 1) - parse stdout, never treat empty as success
- `--details-json FILE` → per-statement interpretable diff; the header is self-describing: `generator`, `version`, `closeness` + `closeness_definition`, every threshold in force (`thresholds`), a `summary` aggregate block (cost bands, `n_changed`, `n_no_counterpart`, top changed) and a per-`sections` table
  - on `distance-semantic`: `flows` per source statement with matched targets and `cost`, plus `changed` (cheapest-match cost above `changed_cost`), `identical` (normalised-text equality), `no_counterpart` (removed content - cheapest match above `no_counterpart_cost`), `section` heading path; a `targets` array gives the B-side view (`best_source_index`, `min_incoming_cost`, `total_incoming_weight`) → added content reads directly
  - on `distance-structural`: per statement `displacement_norm` (length-normalised, null without counterpart), `moved` (|displacement_norm| above `moved_tolerance`), ground `cost`, `semantic_target_index` + `targets_differ` (true → moved-and-reworded), `section`/`target_section`
- read both details to tell a reword (`changed`) from a pure rearrangement (`moved`); removed content is `no_counterpart` on the source side, added content is a target with a high `min_incoming_cost`

Batch and input options:

- `--pairs FILE` (`distance-semantic` and `distance-structural` only) → TSV `a<TAB>b[<TAB>id]`: one model load for the whole sweep, one JSONL line per pair to stdout; `--details-json` takes a directory or a `{pair}` template
- `--home PATH` → name the init home explicitly instead of relying on cwd / `DOCDISTANCE_HOME` discovery
- `--markdown` → structure-aware segmentation: each table row, heading, list item and fenced-code line becomes one statement (markup stripped for embedding, raw line reported); default feeds the whole body to the prose splitter and fragments tables
- `--min-words N` / `--skip-table-fragments` → drop fragments before encoding; `n_dropped` + `drop_rules` appear in detail headers and machine JSON; filtering renormalises the transport marginals, so the SMD legitimately changes

## Use it - Python

```python
from docdistance import semantic_distance, structural_distance
r = semantic_distance("a.md", "b.md")      # DistanceResult: r.smd, r.closeness, r.verdict
s = structural_distance("a.md", "b.md")    # StructuralResult: s.order_gap, s.structure_closeness, s.verdict

from docdistance import DocDistance         # reuse one loaded pipeline for details / batches
dd = DocDistance(backend="openvino")
res, content = dd.semantic_distance_with_details("a.md", "b.md")     # (DistanceResult, content flows dict)
res, order   = dd.structural_distance_with_details("a.md", "b.md")   # (StructuralResult, order details dict)
```

- set `DOCDISTANCE_HOME` (env) to the init home before import, else calls raise "not initialized"
- args accept a path or a raw string
- both `*_with_details` take `markdown=`, `min_words=`, `skip_table_fragments=` - same semantics as the CLI flags; `semantic_distance_with_details` additionally takes `changed_cost=` (recompute the `changed` flag against your own band); details dicts match the `--details-json` schema above

## Why - the agentic use

- frontier-model document conversion / extraction / summarization exposes no logits, so no KL divergence to score drift
- docdistance gives an embedding-grounded distance + per-statement diff instead - an agent scores whether the output preserved the source and localizes what changed, meaning (`changed`) apart from order (`moved`)

## Gotchas

- preflight then `init` before first use; distance needs `docdistance.json` reachable via `DOCDISTANCE_HOME`, cwd or `--home`; on a miss the error names the searched path
- a mistyped file path is scored as its literal text (exit 0, plausible number) - a stderr warning flags it, check stderr when a path argument is involved
- detail-file fields and batch/segmentation flags above need the release after 1.1.4; the preflight `-U` upgrade covers it
- python 3.13 only
- `openvino` backend is CPU, portable, no GPU; `--gpu` / `--backend torch` are opt-in
- `distance-wrt-source` needs the `wmd-wrt-source` mode provisioned and is slower (reranker × NLI)
