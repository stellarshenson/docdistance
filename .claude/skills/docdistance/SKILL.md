---
name: docdistance
description: Install and run the docdistance library (PyPI `docdistance`, CLI `docdistance`) to measure how far two documents are - semantic content distance and structural/order distance - with interpretable per-statement diffs. Use when an agent must check whether a converted, extracted, summarized, or rewritten document preserved the original, find what changed between two documents (meaning vs order), or get an embedding-grounded document distance without model logits/KL. Triggers - "how different are these two documents", "did the conversion/extraction drift", "compare doc A to doc B", "semantic distance between documents", "install/use docdistance". Ships a preflight that checks install and upgrades.
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
- `--details-json FILE` → per-statement interpretable diff:
  - on `distance-semantic`: content flows, each with its matched target, `cost` (the semantic gap) and a `changed` bool → what drifted in meaning
  - on `distance-structural`: per statement `displacement` and `moved` → what moved in order
- read both details to tell a reword (`changed`) from a pure rearrangement (`moved`)

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

## Why - the agentic use

- frontier-model document conversion / extraction / summarization exposes no logits, so no KL divergence to score drift
- docdistance gives an embedding-grounded distance + per-statement diff instead - an agent scores whether the output preserved the source and localizes what changed, meaning (`changed`) apart from order (`moved`)

## Gotchas

- preflight then `init` before first use; distance needs `docdistance.json` reachable via `DOCDISTANCE_HOME` (or cwd)
- python 3.13 only
- `openvino` backend is CPU, portable, no GPU; `--gpu` / `--backend torch` are opt-in
- `distance-wrt-source` needs the `wmd-wrt-source` mode provisioned and is slower (reranker × NLI)
