# CLI reference

`docdistance <command>`: four commands - `init` (provision a mode's models), `distance-semantic` (symmetric content SMD),
`distance-structural` (symmetric order distance), and `distance-wrt-source` (source-conditioned `d(A,B|S)`). Human output is rich and coloured; `--json` is
machine-readable, `--result-only` is the bare scalar. Logs go to stderr, so stdout carries only the result.

- **Init** - `pip install docdistance`, then `docdistance init <mode>` once to provision that mode's models
- **Arguments** - `A`, `B`, `S` are each a file path or raw text (auto-detected); every markdown `# ` title line in a file is stripped so the title is not a statement, and so is the `h1` of a file read as HTML ([Input format](#input-format))
- **Global** - `--version` prints the version; `--help` (and `<command> --help`) prints usage; exit code 1 on a missing model, an un-init'd mode, or an unsecured `--gpu`
- **Backends** - `--backend openvino` (CPU INT8, default) or `torch`; `--gpu` forces torch on CUDA and errors if GPU support is not installed
- **Offline** - after `docdistance init <mode>`, every distance call for that mode runs fully offline from the local cache and makes no request to the HuggingFace Hub. `init --source` copies each model folder it finds in the source; for `sat-3l-sm` it also saves the tokenizer of the SaT segmenter beside the weights when the source folder holds none that loads, from the HuggingFace cache or else the Hub, and a source folder that carries the tokenizer files that `init` saved needs no Hub. `init` loads the tokenizer a source folder carries and reads its special token ids: the tokenizer files as the Hub publishes them load without those ids and a truncated `tokenizer.json` does not load, and `init` replaces both by the tokenizer from the HuggingFace cache or else the Hub (the `sat-3l-sm` bullet of the [AWS deployment reference](aws-deployment-reference.md#models-and-footprint))

## init

Provision a mode's models from local / S3 / HuggingFace and write `docdistance.json` (the readiness record). `MODE` is `wmd` (symmetric: mmBERT + SaT) or `wmd-wrt-source` (adds the bge-reranker-v2-m3 + mDeBERTa NLI grounding models). The only command that downloads.

| Flag | Default | Effect |
| --- | --- | --- |
| `MODE` (argument) | `wmd` | which distance mode to provision (`wmd` or `wmd-wrt-source`) |
| `--source URI` | HuggingFace | model source base: an `s3://bucket/prefix`, a local dir, or omit for the Hub |
| `--backend openvino\|torch` | `openvino` | which weights to fetch (OpenVINO INT8 or torch) |
| `--aws-profile NAME` | none | AWS named profile for an `s3://` source (omit in Lambda for the execution-role chain) |
| `--aws-endpoint-url URL` | none | custom S3 endpoint for an S3-compatible store |
| `--region NAME` | none | AWS region for an `s3://` source |
| `--home DIR` | `$DOCDISTANCE_HOME` or cwd | where to write `docdistance.json` + the model mirror |
| `--verbose`, `-v` | off | DEBUG logging to stderr |

- **Resolution** - per model: the source named by `--source` (an `s3://` prefix or a local dir, not both), then the HuggingFace Hub (the always-available fallback); the source served is recorded per model in `docdistance.json`
- **Readiness** - a distance run whose mode is not init'd exits 1 with `mode '<mode>' is not initialized - run:  docdistance init <mode>`

## distance-semantic

Symmetric Statement Mover's Distance between two documents - the exact content metric, one scalar.

| Flag | Default | Effect |
| --- | --- | --- |
| `--backend openvino\|torch` | `openvino` | statement encoder backend |
| `--gpu` | off | force the torch backend on CUDA; errors if GPU support is not secured |
| `--anisotropy / --no-anisotropy` | `--no-anisotropy` | all-but-the-top anisotropy removal, estimated from the pooled statements of the two documents; the minimum estimation pool is about 71 pooled statements (E12-H65: the statement count is the rule, 8 of the ibm summaries is that fixture's example); a smaller pool logs one warning that names its count and is scored all the same; off for a bare pair |
| `--threshold FLOAT` | `0.77` | closeness cutoff for the similar / not-similar verdict |
| `--markdown` | off | markdown-aware segmentation of every document: one statement per heading, table row, list item and fenced-code line (markup stripped from the embedded text), only prose goes through SaT; equals `--input-format markdown` |
| `--input-format auto\|plain\|markdown\|html` | `auto` | reader of every document: `auto` chooses it per document from the file extension (`.md`, `.markdown`: `markdown`; `.html`, `.htm`: `html`; every other file and raw text: `plain`), `plain` gives the whole text to the prose segmenter as 1.1.5 read every document, `markdown` reads as `--markdown` does, `html` converts the document to markdown lines and reads those ([Input format](#input-format)) |
| `--min-words N` | `0` | drop statements under `N` words after segmentation, before encoding; the transport marginals renormalise over the survivors, so the SMD changes |
| `--skip-table-fragments` | off | drop table-fragment statements (pipes or separator punctuation) after segmentation, before encoding; the transport marginals renormalise over the survivors |
| `--details-json FILE` | off | also write the content alignment (which B statements each A statement's mass flows to, with weights, match cost, and a per-flow changed flag) to `FILE` |
| `--relevance` | off | add the reranker relevance of each statement of the first document over the second to the map; needs `--details-json` (without it the command exits with code 2 before a model loads); pass the long document first and the brief second; costs one reranker pass per pair of statements; with `--pairs` it applies to every row |
| `--pairs FILE` | off | batch mode: TSV of `a-path<TAB>b-path[<TAB>id]`; one model load, one JSON object per pair per line to stdout (see [Batch mode](#batch-mode)) |
| `--json` | off | machine-readable JSON to stdout |
| `--result-only` | off | bare SMD scalar to stdout, no clutter |
| `--home DIR` | `$DOCDISTANCE_HOME` or cwd | folder that holds `docdistance.json` + the model mirror |
| `--verbose`, `-v` | off | DEBUG logging to stderr |

- **Default output** - a rich panel: SMD, closeness, verdict + threshold, the `WCD ≤ RWMD ≤ SMD` bounds, statement counts, anisotropy on/off
- **`--json`** - the result dict: `smd`, `wcd`, `rwmd`, `closeness`, `threshold`, `verdict`, `anisotropy`, `n_statements_a`, `n_statements_b`
- **`--details-json`** - writes a separate JSON file (the result still prints as usual): `{smd, anisotropy, n_statements, flows}`, where `flows` is a per-A-statement list of `{index, text, matches, changed}` and each match is `{target_index, target_text, weight, cost}` - the B statements that statement's transport mass flows to, `weight` the fraction of its mass (sums to 1 per statement), `cost` the ground distance of the match (the semantic gap of the aligned pair), `changed` = the statement's best-match `cost` past the changed-cost cutoff; the exact OT coupling behind the SMD, so a human or a machine can read which statement aligns to which and what drifted in MEANING
- **Matching fields of the details file (from the release after 1.1.5, written "after 1.1.5" below)** - each flow also carries `match_index` (the B statement its A statement is paired with one to one, `null` without a pair) and `match_status` (`kept` or `removed`); each `targets` record carries `match_source_index` and `match_status` (`kept` or `added`); `summary` carries `n_kept` / `n_removed` / `n_added` and `thresholds` carries `match_cutoff`. A pair needs a ground cost below `MATCH_CUTOFF` 0.325. On 900 edited document pairs the three labels read a held-out macro-F1 of 0.9772 at the cutoff 0.325, and 0.9457 in the other direction of the split with the cutoff 0.3100 chosen there, against 0.3130 and 0.3266 for the 1.1.5 flags (E13-H71). The labels annotate the map; the plan and the SMD are unchanged. Calibration and limits: [API reference](api-reference.md#matching-and-the-matched-order-gap)
- **Nearest cost fields (from the release after 1.1.32)** - each flow also carries `min_cost`, the cheapest ground cost of the statement over the other document (4 decimals; `changed` and `no_counterpart` read the unrounded value), and `best_target_index`, the statement that gives it
- **`--relevance` fields (from the release after 1.1.32)** - the statements of the second document are the queries and the statements of the first the passages, so the long document goes first and the brief second. Each flow carries `relevance` (the highest reranker score of the statement over the second document) and `relevance_target_index`; each `targets` record carries `relevance` and `relevance_source_index`; each `sections` record carries `mean_relevance` (the mean of the unrounded relevance of its statements) and `relevance_rank` (1 is the highest mean, ties in document order); `summary.top_relevant` lists the flows with the highest relevance, as many as `top_changed`; the header block `relevance` holds `backend`, `model`, `queries: "b"` and `passages: "a"`. The score is the sigmoid of the reranker logit for the pair: a score, not a distance and not a probability of relevance, and not on one scale across pairs of documents. The command loads the reranker and not the NLI model; a missing reranker model ends it with exit code 1 and a message that names the call `docdistance init wmd-wrt-source`; with `--json`, `--result-only` or `--pairs` the output carries the error code `models_not_installed`. The result and every other key of the map are unchanged. Measured cost: 26.8 ms per pair of statements with the OpenVINO reranker on a CPU shared with other jobs (E19). Readings of batches E17 to E19 and their fixture limits: [README](../README.md#reading-a-long-report-against-a-brief)
- **`--result-only`** - the bare SMD float, for scripts
- **Reading it** - closeness `1.0` identical, `0.0` unrelated; near 1 with verdict `similar` is a faithful match, falling toward 0 with `not similar` means the meaning changed; the content alignment names *which* statement of B each statement of A maps to and how good the match is (low `cost`)

## distance-structural

Symmetric statement-order distance between two documents - the E11-H55 OPW order-gap, translation-invariant, with a `structure_closeness` readout on the SMD-closeness scale.

| Flag | Default | Effect |
| --- | --- | --- |
| `--backend openvino\|torch` | `openvino` | statement encoder backend |
| `--gpu` | off | force the torch backend on CUDA; errors if GPU support is not secured |
| `--anisotropy / --no-anisotropy` | `--no-anisotropy` | all-but-the-top anisotropy removal, estimated from the pooled statements of the two documents; the minimum estimation pool is about 71 pooled statements (E12-H65: the statement count is the rule, 8 of the ibm summaries is that fixture's example); a smaller pool logs one warning that names its count and is scored all the same; off for a bare pair |
| `--threshold FLOAT` | `0.90` | structure_closeness cutoff for the similar / not-similar verdict |
| `--markdown` | off | markdown-aware segmentation of every document: one statement per heading, table row, list item and fenced-code line (markup stripped from the embedded text), only prose goes through SaT; equals `--input-format markdown` |
| `--input-format auto\|plain\|markdown\|html` | `auto` | reader of every document: `auto` chooses it per document from the file extension (`.md`, `.markdown`: `markdown`; `.html`, `.htm`: `html`; every other file and raw text: `plain`), `plain` gives the whole text to the prose segmenter as 1.1.5 read every document, `markdown` reads as `--markdown` does, `html` converts the document to markdown lines and reads those ([Input format](#input-format)) |
| `--min-words N` | `0` | drop statements under `N` words after segmentation, before encoding; the transport marginals renormalise over the survivors, so the SMD changes |
| `--skip-table-fragments` | off | drop table-fragment statements (pipes or separator punctuation) after segmentation, before encoding; the transport marginals renormalise over the survivors |
| `--details-json FILE` | off | also write the order details (per statement: its aligned target and rank displacement) to `FILE` |
| `--pairs FILE` | off | batch mode: TSV of `a-path<TAB>b-path[<TAB>id]`; one model load, one JSON object per pair per line to stdout (see [Batch mode](#batch-mode)) |
| `--json` | off | machine-readable JSON to stdout |
| `--result-only` | off | bare structure_closeness scalar to stdout, no clutter |
| `--home DIR` | `$DOCDISTANCE_HOME` or cwd | folder that holds `docdistance.json` + the model mirror |
| `--verbose`, `-v` | off | DEBUG logging to stderr |

- **Default output** - a rich panel: structure_closeness, order_gap, matched order gap, matched statements (with the cutoff), SMD, verdict + threshold, statement counts, anisotropy on/off
- **`--json`** - the result dict: `smd`, `order_gap`, `structure_closeness`, `threshold`, `verdict`, `anisotropy`, `n_statements_a`, `n_statements_b`, and after 1.1.5 `matched_order_gap`, `n_matched`
- **`matched_order_gap` / `n_matched` (after 1.1.5)** - the order gap over the statements both documents hold: the `n_matched` one-to-one pairs with a ground cost below `MATCH_CUTOFF` 0.325; `null` (panel: `no statement matched`) when no pair is matched. `order_gap` rises when statements are deleted and the order is kept, and the verdict can then read `not similar`. `matched_order_gap` reads such a deletion at the identity floor of the shorter document: at or below 0.005 in 280 of 280 order-preserving deletions on documents of 11 to 14 statements, where `order_gap` reads above 0.005 in 280 of 280 (E13-H70). `order_gap`, `structure_closeness` and the verdict keep their 1.1.5 values, and the verdict is read from `order_gap` only. Readings and limits: [structure document](solution/wmd-structure-distance-sota.md#calibration-of-the-structural-verdict)
- **`--details-json`** - writes a separate JSON file (the result still prints as usual): `{smd, order_gap, structure_closeness, anisotropy, n_statements, statements}`, where `order_gap` is the E11-H55 OPW structural distance (translation-invariant, `>= 0`) and `structure_closeness` = `1 - order_gap / sqrt(2)` the shipped SOTA 0..1 structure readout on the SMD-closeness scale (1 = same order); `statements` is a per-A-statement list of `{index, text, section, target_index, target_text, target_section, cost, semantic_target_index, semantic_target_cost, targets_differ, displacement, matched_displacement, displacement_norm, moved, no_counterpart}` - `displacement` the rank shift of that statement in the exact-plan alignment (0 = in place), `matched_displacement` its rank shift among the matched statements (`null` without a match), and after 1.1.5 `moved` = matched AND `matched_displacement != 0` AND `abs(matched_displacement / n_matched) > moved_tolerance` (0.05) AND NOT `no_counterpart`; this pins down what MOVED in order, statement by statement. `thresholds` carries `match_cutoff` beside `moved_tolerance`. The details file does not carry `matched_order_gap`; it is in the `--json` result
- **`--result-only`** - the bare structure_closeness float, for scripts
- **Reading it** - structure_closeness `1.0` same order, `0.0` fully reordered; the order details name *which* statement of B each statement of A maps to and how far it MOVED (`displacement`, `matched_displacement`). The threshold `0.90` is calibrated on one corpus: [per-corpus procedure](solution/wmd-structure-distance-sota.md#calibration-of-the-structural-verdict)

## distance-wrt-source

Source-conditioned distance `d(A, B | S)` - selection divergence plus each document's distance to the shared source `S`.

| Flag | Default | Effect |
| --- | --- | --- |
| `--source`, `-s` | required | the common source document |
| `--backend openvino\|torch` | `openvino` | statement encoder backend |
| `--gpu` | off | force the torch backend on CUDA; errors if GPU support is not secured |
| `--anisotropy / --no-anisotropy` | `--anisotropy` | anisotropy removal on the conditioned selection axis, on by default (E04-H15), estimated from the pooled statements of A, B and the source; the minimum estimation pool is about 71 pooled statements (E12-H65, measured on pools of summaries without a source: the statement count is the rule, 8 of the ibm summaries is that fixture's example); this command logs no warning for a smaller pool |
| `--markdown` | off | markdown-aware segmentation of every document: one statement per heading, table row, list item and fenced-code line (markup stripped from the embedded text), only prose goes through SaT; equals `--input-format markdown` |
| `--input-format auto\|plain\|markdown\|html` | `auto` | reader of every document: `auto` chooses it per document from the file extension (`.md`, `.markdown`: `markdown`; `.html`, `.htm`: `html`; every other file and raw text: `plain`), `plain` gives the whole text to the prose segmenter as 1.1.5 read every document, `markdown` reads as `--markdown` does, `html` converts the document to markdown lines and reads those ([Input format](#input-format)) |
| `--min-words N` | `0` | drop statements under `N` words after segmentation, before encoding; the transport marginals renormalise over the survivors, so the SMD changes |
| `--skip-table-fragments` | off | drop table-fragment statements (pipes or separator punctuation) after segmentation, before encoding; the transport marginals renormalise over the survivors |
| `--source-map-json FILE` | off | also write a statement → source alignment map (top-3 source statements each A/B statement covers, with weights) to `FILE`; with `--pairs`, a directory or a template that contains `{pair}` - one map per pair |
| `--pairs FILE` | off | added after 1.1.5 - batch mode: TSV of `a-path<TAB>b-path[<TAB>id]`, every pair against the one `--source`; one model load, one JSON object per pair per line to stdout (see [Batch mode](#batch-mode)) |
| `--json` | off | machine-readable JSON to stdout |
| `--result-only` | off | bare `D_sel,grd_a,grd_b` to stdout |
| `--home DIR` | `$DOCDISTANCE_HOME` or cwd | folder that holds `docdistance.json` + the model mirror |
| `--verbose`, `-v` | off | DEBUG logging to stderr |

- **Requires** - `docdistance init wmd-wrt-source` (pulls the reranker + NLI grounding models); un-init'd exits 1
- **Default output** - a rich panel: `D_sel` (selection divergence), each document's grounding residual `grd_a` / `grd_b` (E03-H11) and the `D_grd` separation, the geometric residual to source + closeness, statement counts (A / B / S), and after 1.1.5 the unsupported statements of A and B with the cutoff
- **`--json`** - the result dict: `d_sel`, `residual_a`, `residual_b`, `closeness_a`, `closeness_b`, `n_statements_a`, `n_statements_b`, `n_statements_source`, `n_unsupported_a`, `n_unsupported_b` (after 1.1.5), `coverage_a`, `coverage_b`, `grd_a`, `grd_b`, `d_grd`
- **`n_unsupported_a` / `n_unsupported_b` (after 1.1.5)** - the count of statements whose nearest source statement costs more than `SOURCE_SUPPORT_CUTOFF` 0.34625, read on the embeddings before the anisotropy pre-pass, so `--anisotropy` and `--no-anisotropy` print the same counts. A score that depends on that calibrated cutoff, not a metric. It reads embedding distance to the source, not entailment: a faithful summary can read above 0. Readings and limits: [source-conditioned document](solution/wmd-source-conditioned-docdistance-solution-sota.md#distance-to-the-source-and-unsupported-statements)
- **`--source-map-json`** - writes a separate JSON file (the result still prints as usual): `{top_k, anisotropy, n_statements, a, b}`, where `a` and `b` are per-statement lists of `{index, text, matches}` and each match is `{source_index, source_text, weight}` - the top-3 source statements that statement covers, weights summing toward 1. After 1.1.5 each `a` and `b` record also carries `nearest_source_index`, `nearest_source_cost` and `unsupported`, and the header carries `source_support_cutoff` and `n_unsupported` with the keys `a` and `b`; these are read before the pre-pass, the `matches` after it
- **`--result-only`** - `D_sel,grd_a,grd_b`, comma-separated
- **Reading it** - lower is closer on every axis; `d_sel` near 0 means A and B drew on the same source content, a high grounding residual `grd_a` / `grd_b` flags a document that drifted from `S` (dropped or unsupported content); the source map names *which* source statements each statement covers. The geometric residual (`residual_a` / `residual_b`, panel rows `A → source` / `B → source`) is the balanced transport cost to the whole source: it grows when a document covers only part of the source and is not the share of the document the source does not support (10 statements copied verbatim from a 70-statement source read 0.958, DEF-TRANS-8)

## Input format

`--input-format` on `distance-semantic`, `distance-structural` and `distance-wrt-source` names the reader of every document of the call, the source included. Added after 1.1.5. The default is `auto`, which chooses the reader of each document from its file extension.

| Value | Reader |
| --- | --- |
| `auto` | the default: one of the three below, chosen per document from the file extension |
| `plain` | the whole text goes to the prose segmenter (SaT); the reader of every document up to 1.1.5 |
| `markdown` | the markdown reader of `--markdown`: one statement per heading, table row, list item and fenced-code line, prose paragraphs through SaT |
| `html` | the document is converted to markdown lines (table below), then read by the markdown reader |

Extension rule of `auto`:

| Document | Reader |
| --- | --- |
| a file with the extension `.md` or `.markdown` | `markdown` |
| a file with the extension `.html` or `.htm` | `html` |
| a file with another extension, a file without extension | `plain` |
| raw text | `plain` |

- **Extension** - the part of the file name from its last dot, compared without letter case: `REPORT.MD` reads as `markdown`, `notes.md.txt` as `plain`. A name whose only dots stand at its start (`.md`) has no extension. The content of the file is not read for the choice: a `.txt` file that holds HTML reads as `plain`, and raw text reads as `plain` whatever it holds, also when it ends in `.md`
- **Explicit value** - `plain`, `markdown` or `html` is the reader of every document of the call, whatever the extension of each file is
- **`--markdown`** - equals `--input-format markdown`. `--markdown` beside `--input-format plain`, `html` or a given `auto` names two readers: the command exits with code 2. A value outside the four also exits with code 2. Both exit before a model loads
- **Numbers against 1.1.5** - releases up to 1.1.5 read every document with the plain reader. With the default `auto` the statements and every number for `.md`, `.markdown`, `.html` and `.htm` files can differ from 1.1.5; `--input-format plain` restores the 1.1.5 reading. Files with another extension and raw text read as before
- **Title** - the `h1` of an HTML file is removed like the `# ` title line of a markdown file; raw text keeps both
- **Which reader ran** - `--verbose` logs one DEBUG line per document read that names the reader and, for a file under `auto`, that its extension chose it; a document served from the memo of a batch logs none
- **Filters and sections** - `--min-words`, `--skip-table-fragments` and the `section` fields of the detail files read the converted lines, so they work on an HTML document as on a markdown one

Conversion of the `html` reader (`html_to_markdown`):

| HTML | Written as |
| --- | --- |
| `h1` to `h6` | `#` repeated by the level, a space, the text on one line |
| `li` | a marker and the text on one line: `- ` in a `ul`; in an `ol` the number of the item and a full stop (`1. `, `2. `, ...), counted from 1 in each `ol`, a nested one as well. The `start` attribute is not read, and an item without text takes its number and writes no line. A nested item is indented by two spaces per level, and the text of an item ends where a nested list starts |
| `tr` | `\| cell \| cell \|` from its `td` and `th` texts; a `\|` inside a cell becomes a space; no separator row |
| `pre` | a fenced block with the lines of its text |
| `p`, `div`, `blockquote`, `section`, `article`, `dt`, `dd`, `figcaption`, `caption`, `header`, `footer`, `main`, `nav`, `aside`, `address`, `form`, `figure`, `details`, `summary`, and text outside any block | a paragraph on one line, its whitespace collapsed to single spaces, followed by a blank line |
| `br`, `hr` | a line break inside the block, the only one a paragraph holds; the end of the paragraph |
| `script`, `style`, `template`, `noscript`, `svg`, `title` (also outside a `head`), comments | dropped with their content |
| every other tag (`a`, `em`, `strong`, `span`, `code`, `sup`, `img`, `head`, ...) | the text inside it only; an `img` leaves nothing. `head` is not dropped, so a document that omits `</head>` and `<body>` keeps its text |

Character references are decoded. A byte order mark (U+FEFF) is removed. Whitespace inside a heading, a list item and a cell collapses to single spaces. A `p` inside an `li` is text of that item and every block inside a cell is text of that cell. Malformed HTML raises no error: an unclosed block ends at the next block or at the end of the text. Inline markup that a markdown file keeps as characters (`**bold**`) has no counterpart in the converted text, so such statements of an HTML document and of its markdown conversion differ by those characters. A line break inside a paragraph of the HTML source is whitespace: the prose segmenter cuts at every line break, and a paragraph written on one line keeps the statements of an HTML file independent of how its source text is wrapped. Text outside any block is one paragraph up to the next block, so a text without tags that is read by the `html` reader is one line. A `pre` block whose text holds a line that starts with three backticks or three tildes is not read as one fenced block, because the markdown reader opens or closes a fence at every such line. With two such lines the lines between them are read as markdown structure (a `# ` line as a heading); with one such line the rest of the document is read as code lines.

## Batch mode

`--pairs FILE` on `distance-semantic`, `distance-structural` and, after 1.1.5, `distance-wrt-source` scores many pairs with one model load.

- **Input** - a TSV with one pair per line, `a-path<TAB>b-path[<TAB>id]`; blank lines are skipped. The positional `A` and `B` are not given. On `distance-wrt-source` every pair is scored against the one `--source`
- **Output** - one JSON object per pair per line (JSONL) on stdout: the key `pair` (the id of the line, or its 0-based row index) and the keys of the one-shot `--json` output
- **Per-pair files** - `--details-json` (`--source-map-json` on `distance-wrt-source`) takes a directory, which receives `<pair>.json`, or a template that contains `{pair}`
- **Memo (after 1.1.5)** - a batch keeps the statements and the embeddings of up to 64 document files (`MEMO_SIZE`) and evicts the least recently used one beyond that, so a file that recurs across rows is segmented and encoded once. On `distance-wrt-source` it also keeps the relevance grid and the entailment vector of up to 64 (document file, source file) pairs. A file is identified by its resolved path, modification time, size and the segmentation options, with `--input-format` as given: one `.md` file read under `auto` and under `markdown` is two entries. Two names of one file whose extensions choose two readers (a symbolic link `notes.txt` to `a.md`) are two entries, each read by the reader of its own name. A document given as raw text is not kept, and a one-shot call keeps nothing
- **Same output with and without the memo** - on the notebook prototypes the JSONL and the per-pair files were byte-identical in 9 of 9 arms (E12-H67) and on 73 of 73 pair payloads (E12-H68). The limit: a file rewritten with equal byte size inside one timestamp step of the file system keeps its memo entry
- **Measured on the notebook prototypes** - all 55 pairs of the 11 ibm summaries: `distance-semantic` 11.032 s → 1.110 s, 9.94x (10.12x on the re-run of 2026-10-09), with the model load excluded and 1.98x with it included, document encodes 110 → 11 (E12-H67); `distance-wrt-source` against one source 2671.72 s → 266.32 s, 10.03x (9.66x on the re-run of 2026-10-09), grounding grids 110 → 11 (E12-H68). On a batch without a repeated file the memo changed the wall time by −0.63% and +0.22% on the two controls
- **Warnings** - the marker-statement warning (statements of the plain reader that hold no letter and no digit, such as `-` or `###`; `--markdown` or `--min-words` removes them) is logged once per document of each row that the plain reader read, with and without a memo hit. Under the default `auto` that is every raw-text document and every file whose extension is not `.md`, `.markdown`, `.html` or `.htm`. A document read by the markdown or the HTML reader logs none. The pool warning of `--anisotropy` is logged once per pair on the two symmetric commands

## Examples

```bash
# one-time setup
pip install docdistance                        # from PyPI ('docdistance[s3]' for S3 sources)
docdistance init wmd                           # provision the symmetric-distance models
docdistance init wmd-wrt-source                # + the reranker + NLI grounding models
docdistance init wmd-wrt-source --source s3://your-bucket --aws-profile NAME
docdistance init wmd --source /path/to/models  # from a local mirror

# method 1 - semantic (content) distance
docdistance distance-semantic report_v1.md report_v2.md
docdistance distance-semantic "first text" "second text"          # raw text, not files
docdistance distance-semantic a.md b.md --json                    # machine-readable
docdistance distance-semantic a.md b.md --details-json content.json   # + statement → statement content map
docdistance distance-semantic report.md brief.md --details-json map.json --relevance   # + reranker relevance, long document first
docdistance distance-semantic a.md b.md --result-only             # bare SMD scalar
docdistance distance-semantic a.md b.md --threshold 0.8           # stricter verdict
docdistance distance-semantic a.md b.md --gpu                     # torch on CUDA
docdistance distance-semantic page.html page.md                   # HTML against its markdown conversion, readers by extension
docdistance distance-semantic a.md b.md --input-format plain      # the reader and the numbers of 1.1.5

# method 2 - structural (order) distance
docdistance distance-structural report_v1.md report_v2.md
docdistance distance-structural a.md b.md --json                  # machine-readable
docdistance distance-structural a.md b.md --details-json order.json   # + per-statement order displacement
docdistance distance-structural a.md b.md --result-only           # bare structure_closeness scalar

# method 3 - source-conditioned d(A,B|S)
docdistance distance-wrt-source sum_a.md sum_b.md --source article.md
docdistance distance-wrt-source a.md b.md -s s.md --json
docdistance distance-wrt-source a.md b.md -s s.md --source-map-json map.json   # + statement → source map
docdistance distance-wrt-source a.md b.md -s s.md --result-only   # D_sel,grd_a,grd_b

# batch - many pairs, one model load, one JSON line per pair
docdistance distance-semantic --pairs pairs.tsv --details-json 'details/{pair}.json'
docdistance distance-structural --pairs pairs.tsv
docdistance distance-wrt-source --pairs pairs.tsv -s s.md --source-map-json maps/   # every pair against one source
```
