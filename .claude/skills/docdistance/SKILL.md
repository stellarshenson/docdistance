---
name: docdistance
description: Install and run the docdistance library (PyPI `docdistance`, CLI `docdistance`) to measure how far two documents are - semantic content distance and structural/order distance - with interpretable per-statement diffs. Use when an agent must check whether a converted, extracted, summarized, or rewritten document preserved the original, find what changed between two documents (meaning vs order), or get an embedding-grounded document distance without model logits/KL. Triggers - "how different are these two documents", "did the conversion/extraction drift", "compare doc A to doc B", "semantic distance between documents", "score many document pairs in batch", "install/use docdistance". Ships a preflight that checks install and upgrades.
---

# docdistance

Distance between two documents, grounded in word/statement embeddings (Word Mover's Distance / Optimal Transport). Two axes: content (meaning) and structure (order). Per-statement diff localizes what changed.

Package `docdistance` on PyPI. Import `docdistance`. CLI `docdistance`. Python 3.12 to 3.14; releases up to 1.1.5 install on 3.13 only.

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
- python 3.12, 3.13 or 3.14 required (`requires-python >=3.12,<3.15`); releases up to 1.1.5 require 3.13 (`~=3.13`) and fail to install on other pythons

## Provision models once - required before any distance

docdistance needs models on disk; every distance call fails until provisioned.

```bash
export DOCDISTANCE_HOME="$PWD"          # where docdistance.json + model mirror live - keep it stable
docdistance init                        # 'wmd' mode: semantic + structural, from HuggingFace, openvino INT8 on CPU
docdistance init wmd-wrt-source         # adds reranker + NLI - only if using distance-wrt-source or distance-semantic --relevance
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

- `--json` → full result dict (`smd`/`closeness`/`verdict`, or `order_gap`/`structure_closeness`/`verdict` + `matched_order_gap`/`n_matched`)
- `--result-only` → one bare number, pipe it straight
- errors in `--json`/`--result-only` mode still write a one-line JSON `{error, message}` object to stdout (exit 1) - parse stdout, never treat empty as success
- `--details-json FILE` → per-statement interpretable diff; the header is self-describing: `generator`, `version`, `closeness` + `closeness_definition`, every threshold in force (`thresholds`), a `summary` aggregate block (cost bands, `n_changed`, `n_no_counterpart`, top changed) and a per-`sections` table
  - on `distance-semantic`: `flows` per source statement with matched targets and `cost`, plus `changed` (cheapest-match cost above `changed_cost`), `identical` (normalised-text equality), `no_counterpart` (removed content - cheapest match above `no_counterpart_cost`), `section` heading path; every flow also carries `min_cost` (the cheapest ground cost `changed` is read on) and `best_target_index`; a `targets` array gives the B-side view (`best_source_index`, `min_incoming_cost`, `total_incoming_weight`) → added content reads directly; one-to-one matching (ground cost below `match_cutoff` 0.325): `match_index` + `match_status` `kept`/`removed` per flow, `match_source_index` + `match_status` `kept`/`added` per target, `n_kept`/`n_removed`/`n_added` in `summary`
  - `distance-semantic --details-json FILE --relevance` (long document first, brief second; needs `--details-json`, else exit 2; one reranker pass per pair of statements; no NLI model): `relevance` + `relevance_target_index` per flow, `relevance` + `relevance_source_index` per target, `mean_relevance` + `relevance_rank` per section, `summary.top_relevant`, a `relevance` header block; a score (sigmoid of the reranker logit), not a distance or a probability, not on one scale across pairs; no match / no-match rule for one brief is confirmed - read the ranks; measured on two fixtures only: mean AUC of target-section statements 0.8611 vs 0.6183 for `min_cost` (constructed report, 12 briefs) and 0.8321 vs 0.6374 (public EU report of 883 statements, 13 notes), about 27 ms per pair on a shared CPU; `min_cost` under `--anisotropy` reads 0.7250 on the public report without a reranker and did not meet its bar on the constructed report (E17), no default changes
  - on `distance-structural`: per statement `displacement_norm` (length-normalised, null without counterpart), `matched_displacement` (rank shift among matched statements, null without a match), `moved` (matched AND `matched_displacement` != 0 AND |matched_displacement / n_matched| above `moved_tolerance` AND NOT `no_counterpart`), ground `cost`, `semantic_target_index` + `targets_differ` (true → moved-and-reworded), `section`/`target_section`
- read both details to tell a reword (`changed`) from a pure rearrangement (`moved`); removed content is `match_status` `removed` on a flow (`no_counterpart` only above cost 0.6), added content is `match_status` `added` on a target
- structural verdict fires on deletions that keep the order (`order_gap` rises with a statement-count difference) → read `matched_order_gap` (`--json`), the order gap over the `n_matched` statements both documents hold: at or below 0.005 on such a deletion for documents of 11 to 14 statements; the verdict reads `order_gap` only
- `distance-wrt-source --json`: `n_unsupported_a`/`n_unsupported_b` → statements whose nearest source statement costs more than 0.34625 (embedding distance, not entailment); `residual_a`/`closeness_a` are the transport cost to the whole source, not the unsupported share; `--source-map-json` records carry `nearest_source_index`, `nearest_source_cost`, `unsupported`

Batch and input options:

- `--pairs FILE` (all three distance commands; on `distance-wrt-source` every pair against the one `--source`) → TSV `a<TAB>b[<TAB>id]`: one model load for the whole sweep, one JSONL line per pair to stdout; `--details-json` (`--source-map-json` on `distance-wrt-source`) takes a directory or a `{pair}` template; a file that recurs across rows is encoded once (memo of 64 files, raw text not kept)
- `--home PATH` → name the init home explicitly instead of relying on cwd / `DOCDISTANCE_HOME` discovery
- `--markdown` → structure-aware segmentation of every document: each table row, heading, list item and fenced-code line becomes one statement (markup stripped for embedding, raw line reported); equals `--input-format markdown`
- `--input-format auto|plain|markdown|html` → reader of every document; default `auto` → per document from the file extension, any letter case, content not read: `.md`/`.markdown` → `markdown`, `.html`/`.htm` → `html`, every other file, a file without extension and raw text → `plain`
  - explicit `plain`, `markdown` or `html` → that reader for every document; `--markdown` beside `plain`, `html` or a given `auto` exits 2
  - numbers for `.md`/`.markdown`/`.html`/`.htm` files can differ from 1.1.5, which read every document as `plain` → pass `--input-format plain` / `input_format="plain"` for the 1.1.5 numbers
  - `html` → converted to markdown lines, then the markdown reader: `h1`-`h6` → `#` lines, `li` → `- ` line in a `ul`, numbered line in an `ol` (`1. `, `2. `, from 1 in each `ol`; nested: two spaces per level), `tr` → `| cell | cell |` row of its `td`/`th` texts, `pre` → fenced block, `p`/`div`/`blockquote`/`section`/`article`/`dt`/`dd`/`figcaption`/`caption`/`header`/`footer`/`main`/`nav`/`aside`/`address`/`form`/`figure`/`details`/`summary` and loose text → paragraph on one line, whitespace collapsed, `br` → the only line break inside a paragraph, `hr` → paragraph end, `script`/`style`/`template`/`noscript`/`svg`/`title`/comments/byte order mark dropped, every other tag (`head` among them) → its text only; the `h1` of an HTML file is dropped like a markdown `# ` title
  - HTML document vs its markdown conversion → `docdistance distance-semantic page.html page.md`; the statements of an HTML file do not depend on how its source text is wrapped
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
- both `*_with_details` take `markdown=`, `input_format=`, `min_words=`, `skip_table_fragments=` - same semantics as the CLI flags, and so do the one-shot functions and the other `DocDistance` scoring methods; `semantic_distance_with_details` additionally takes `changed_cost=` (recompute the `changed` flag against your own band) and `relevance=` (default `False`; `True` adds the reranker fields above, `a` the long document, `b` the brief); details dicts match the `--details-json` schema above
- sweeps over many pairs → `DocDistance(memo_size=64)`: a document file that recurs across calls is segmented and encoded once; default `0` keeps nothing

## Why - the agentic use

- frontier-model document conversion / extraction / summarization exposes no logits, so no KL divergence to score drift
- docdistance gives an embedding-grounded distance + per-statement diff instead - an agent scores whether the output preserved the source and localizes what changed, meaning (`changed`) apart from order (`moved`)

## Gotchas

- preflight then `init` before first use; distance needs `docdistance.json` reachable via `DOCDISTANCE_HOME`, cwd or `--home`; on a miss the error names the searched path
- a mistyped file path is scored as its literal text (exit 0, plausible number) - a stderr warning flags it, check stderr when a path argument is involved
- detail-file fields and batch/segmentation flags above need the release after 1.1.4; the matching fields (`match_*`, `matched_*`, `n_unsupported_*`), `--pairs` on `distance-wrt-source`, `memo_size=` and `--input-format` / `input_format=` need the release after 1.1.5; the preflight `-U` upgrade covers it
- `plain` reader keeps bare markdown markers (`-`, `###`) as statements and logs a stderr warning with their count → pass `--markdown` or `--min-words`; under the default `auto` that concerns raw text and files that are not `.md`/`.markdown`/`.html`/`.htm`
- python 3.12 to 3.14; releases up to 1.1.5 are 3.13 only
- `openvino` backend is CPU, portable, no GPU; `--gpu` / `--backend torch` are opt-in
- `distance-wrt-source` needs the `wmd-wrt-source` mode provisioned and is slower (reranker × NLI)
