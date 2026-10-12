# Changelog

All notable changes to docdistance are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow semantic versioning.

## [1.1.34] - 2026-10-12

The additive policy holds: without `relevance=True` / `--relevance` every key the detail file held
keeps its value.

### Added

- Semantic detail file: `min_cost` (the cheapest ground cost of the statement over the other
  document, the number `changed` is read on, 4 decimals) and `best_target_index` on each flow, with
  both keys in `definitions`
- `DocDistance.semantic_distance_with_details(..., relevance=True)` and `distance-semantic
  --relevance` (needs `--details-json`, exit code 2 without it, applies to every row of `--pairs`):
  the reranker score of each statement of the first document over the statements of the second,
  as `relevance` and `relevance_target_index` on each flow, `relevance` and `relevance_source_index`
  on each `targets` record, `mean_relevance` and `relevance_rank` on each section,
  `summary.top_relevant` and a `relevance` header block. The long document goes first and the brief
  second; the cost is one reranker pass per pair of statements; the NLI model is not loaded. The
  score is a score, not a distance and not on one scale across pairs. On one report and 12 briefs
  written by the experimenter, the briefs not seen when it was chosen, it ranks the target sections
  at a mean AUC of 0.8611 against 0.6183 for the nearest cost (E17, E18). On one public European
  Commission report of 883 statements and 13 briefing notes of the European Parliamentary Research
  Service, target sections assigned by the experimenter before a note was scored, it reads 0.8321
  against 0.6374, above the nearest cost in 13 of 13 notes, at 26.8 ms per pair of statements on a
  shared CPU (E19). No rule for match and no match of one brief is confirmed

### Changed

- A missing reranker model raises `ModelsNotInstalled` with a message that names the reranker and
  the call that installs it: `docdistance init wmd-wrt-source`, with `--backend torch` on the torch
  backend. This holds for `distance-wrt-source` and for `--relevance`; 1.1.32 named `docdistance
  init`, which does not install the reranker

## [1.1.32] - 2026-10-12

Every number and every detail-file key that 1.1.5 produces keeps its value, with three registered
exceptions: `moved` and `n_moved` of the structural detail file, the statements that
`--markdown` returns for a document with marker-only lines, and the default reader of a file with
the extension `.md`, `.markdown`, `.html` or `.htm` (first entry under Changed); with
`--input-format plain` / `input_format="plain"` the third exception does not apply. Other new
behaviour ships as new functions, new fields and opt-in parameters whose default keeps the earlier
behaviour.

### Added

- `partial_matching` and `MATCH_CUTOFF` (0.325): a one-to-one matching of the statements of two
  documents, where a pair needs a ground cost below the cutoff; an annotation, not a distance
- Semantic detail file: `match_index` and `match_status` (`kept` / `removed`) on each flow,
  `match_source_index` and `match_status` (`kept` / `added`) on each `targets` record,
  `n_kept` / `n_removed` / `n_added` in `summary` and `match_cutoff` under `thresholds`
- `matched_order_gap` with `n_matched`, as a function and as fields of `StructuralResult`: the order
  gap over the statements both documents hold; `distance-structural` shows both in its panel and in
  `--json`
- Structural detail file: `matched_displacement` on each statement record and `match_cutoff` under
  `thresholds`
- `unsupported_count` and `SOURCE_SUPPORT_CUTOFF` (0.34625): `n_unsupported_a` / `n_unsupported_b`
  on `SourceConditionedResult`, shown by `distance-wrt-source`; the source map gains
  `nearest_source_index`, `nearest_source_cost` and `unsupported` on each record and
  `source_support_cutoff` and `n_unsupported` in its header
- `--pairs FILE` on `distance-wrt-source`: every pair against the one `--source`, one JSON line per
  pair; `--source-map-json` accepts a directory or a `{pair}` template in that mode
- `DocDistance(memo_size=...)`, default 0: bounded memos of the statements and embeddings of a
  document file and of the grounding arrays of a (document file, source file) pair. The `--pairs`
  batches use 64 entries, so a file that recurs across rows is encoded once
- `ANISOTROPY_MIN_POOL` (71): a symmetric call with the anisotropy pre-pass on and a smaller pool of
  statements logs one warning that names the count
- One warning per document when the plain reader returns statements without a letter and without
  a digit (bare markdown markers such as `-` or `###`); it names their count and the options
  `--markdown` / `--min-words`
- `--input-format auto|plain|markdown|html` on the three distance commands, single pair and
  `--pairs`, and `input_format=` on the three one-shot functions and on every `DocDistance` method
  that takes `markdown`. `plain` is the prose segmenter on the whole text, the reader of 1.1.5;
  `markdown` and `html` read the block structure; `auto`, the default, chooses one of the three
  per document from its file extension (first entry under Changed). An explicit `plain`,
  `markdown` or `html` is the reader of every document of the call. `--markdown` /
  `markdown=True` equals `markdown`; beside `plain`, `html` or, on the command line, a given
  `auto` it is an error (exit code 2, `ValueError` in Python), and so is an unknown value; each
  is raised before a model loads
- HTML reader, `html` and `html_to_markdown` (in `docdistance.encoders`): the document is written
  as markdown lines, which the markdown reader then segments. `h1` to `h6` become `#` lines, `li`
  a `- ` line in a `ul` and a numbered line in an `ol` (`1. `, `2. `, counted from 1 in each `ol`;
  a nested item indented by two spaces per level), `tr` one `| cell | cell |` row of its `td` and
  `th` texts, `pre` a fenced block, `p` / `div` / `blockquote` / `section` / `article` / `dt` /
  `dd` / `figcaption` / `caption` / `header` / `footer` / `main` / `nav` / `aside` / `address` /
  `form` / `figure` / `details` / `summary` and text outside any block a paragraph on one line
  with its whitespace collapsed to single spaces; `br` is the only line break inside a paragraph
  and `hr` ends a paragraph, so the statements of an HTML document do not depend on how its source
  text is wrapped; `script`, `style`, `template`, `noscript`, `svg`, `title`, comments and a byte
  order mark are dropped; every other tag leaves its text only. `head` is one of those tags, so a
  document that omits `</head>` and `<body>` keeps its text. The `h1` of an HTML file is removed
  like the `# ` title line of a markdown file, and the section fields of the detail files read the
  converted lines. Parser of the standard library, no new dependency
- `SolverNotOptimal`, exported from the package root, raised when the exact solver stops before an
  optimal coupling
- Python 3.12 and 3.14 beside 3.13 (`requires-python >=3.12,<3.15`)
- Documentation: the per-corpus calibration procedure of the structural threshold, the fragility
  card of the semantic threshold, the measured limits of the order gap and of the matching, and the
  reading of the kernel pre-pass probe (no kernel pre-pass ships)

### Changed

- The default reader of a document follows its file extension (`input_format` `auto`), in any
  letter case: a `.md` or `.markdown` file is read by the markdown reader, a `.html` or `.htm` file
  by the HTML reader, every other file, a file without extension and raw text by the plain reader.
  The content is not read for the choice. 1.1.5 read every document with the plain reader, so the
  statements and every number for `.md`, `.markdown`, `.html` and `.htm` files can differ from
  1.1.5.
  `--input-format plain` / `input_format="plain"` restores the 1.1.5 reading. Files with another
  extension and raw text read as before
- `moved` in the structural detail file is read on the one-to-one matched statements: the statement
  is matched, `matched_displacement` is not 0, `abs(matched_displacement / n_matched)` is above
  `moved_tolerance` and the record is not `no_counterpart`. 1.1.5 read
  `abs(displacement_norm) > moved_tolerance`. A statement inserted or removed without a match flags
  no statement. On a document that repeats a statement this does not always hold: `moved` flagged a
  kept statement in 4 of 120 pure deletions of a 62-statement document with 9 repeated statements
  (E15-H80; bullet "Limit: repeated statements" of `docs/api-reference.md`). `n_moved` changes with
  the flag
- The iteration cap of the exact solver grows with the problem size (`max(100000, n·m)`); the
  result of a problem that the former cap solved is unchanged
- Docstrings and documents state that `residual_a` / `closeness_a` are the transport cost to the
  whole source and not the unsupported share of a document; the comments of `DEFAULT_THRESHOLD` and
  `STRUCTURE_THRESHOLD` state the corpus and the band each value was placed in
- `scipy` is a declared dependency
- `huggingface_hub` is a declared dependency

### Fixed

- OpenVINO usage telemetry is blocked before `import openvino`: on Python 3.14 a script without a
  main guard no longer runs twice, and no usage event is sent
- A solve that reached the iteration cap returned a non-optimal value with only a warning from POT;
  it now raises `SolverNotOptimal`
- The `moved` flag of a removed statement was set by an exact tie of the transport plan and
  differed between POT versions
- `offline=True` had no effect on the HuggingFace Hub library: the model constructors set
  `HF_HUB_OFFLINE` after the library had read it, so a load of cached models sent requests to the
  Hub and raised `ModelsNotInstalled` without network. Each constructor now holds the library in
  offline mode while its model loads and puts the earlier mode back, so `docdistance init` in the
  same process still reaches the Hub. Loads that overlap on two threads stay offline until the last
  of them ends, and the earlier mode is put back then
- `docdistance init` stayed offline when `HF_HUB_OFFLINE=1` was set before the HuggingFace Hub
  library was imported: `init` removed the variable from the environment after the library had read
  it, so it could not download. `init` now holds the library online while it fetches the models and
  puts the earlier mode back; the environment variable is left as it is (`DEF-DEPS-16`)
- `--markdown` / `markdown=True` returned a line that holds only a marker as a statement: a bare
  heading marker (`###`), a bare bullet (`-`, `*`, `+`) or a horizontal rule (`---`, `***`, `___`,
  with or without spaces). These lines are now dropped, so the statements and the distance of a
  document with such lines change in that mode. The plain reader is unchanged
- After `docdistance init --source` from a source folder `sat-3l-sm` without the tokenizer files,
  the first load of the SaT segmenter requested its tokenizer `facebookAI/xlm-roberta-base` from the
  Hub, because wtpsplit loads the tokenizer by its Hub name. `init` now saves the tokenizer beside
  the weights when the source folder lacks it, from the HuggingFace cache or else the Hub, and the
  segmenter loads it from that folder, so the load makes no Hub request (`DEF-DEPS-17`). `init`
  also loads a tokenizer that the source folder carries and reads its three special token ids:
  `tokenizer.json` and `tokenizer_config.json` as the Hub publishes them load without those ids,
  and a truncated `tokenizer.json` does not load. `init` saves the tokenizer over such files
  (`DEF-DEPS-17-1`)
- The warning for a mistyped file path now also names an argument without a slash that ends in
  `.markdown` or `.htm`; it covered `.md`, `.txt`, `.rst` and `.html` before
- Documentation: the structure solution document quoted the calibration of the structural
  threshold `0.90` as valid for the ibm corpus. It now states that the calibration was read on
  stored fixture statements that keep the title line, encoded with the torch backend, and that
  `distance-structural` on the 55 pairs of the ibm summary files reads 3 pairs below `0.90` with
  `--input-format plain` and 4 with the default reader (`DEF-DOCS-18`). The threshold is unchanged

## [1.1.5] - 2026-08-31

Field-use improvement wave: seven defects fixed and thirteen enhancements implemented from
production insights, hardened by a four-lens adversarial review (architect, data scientist,
bug hunter, slop hunter - five rounds to a clean SHIP). The transport plan and the default-path
`smd` / `order_gap` numbers are bit-identical to 1.1.4 - only reporting, segmentation options
and the CLI surface grew.

### Added

- `--pairs FILE` batch mode on `distance-semantic` and `distance-structural`: TSV `a<TAB>b[<TAB>id]`,
  one model load for the whole sweep, one JSONL line per pair on stdout; `--details-json` accepts a
  directory or a `{pair}` template
- `--markdown` structure-aware segmentation: each table row, heading, list item and fenced-code line
  becomes one statement (markup stripped for embedding, raw line reported); the default prose splitter
  is unchanged
- `--min-words N` and `--skip-table-fragments` fragment filters; `n_dropped` and `drop_rules` are
  reported in detail headers and machine JSON, and the transport marginals renormalise accordingly
- `--home PATH` to name the init home explicitly; a miss now reports the searched path
- Self-describing detail files: header with `generator`, `version`, `closeness` plus
  `closeness_definition`, every threshold in force under `thresholds`, a `summary` aggregate block
  (cost bands, `n_changed`, `n_no_counterpart`, top changed statements) and a per-`sections` table
- Semantic flows: `no_counterpart` (removed content - cheapest match above `no_counterpart_cost` 0.6),
  `identical` (normalised-text equality), `section` heading path, and a `targets` array giving the
  B-side view (`best_source_index`, `min_incoming_cost`, `total_incoming_weight`) so added content
  reads directly
- Structural records: length-normalised `displacement_norm` with `moved` (tolerance 0.05), ground
  `cost`, `semantic_target_index` with `targets_differ` (true names moved-and-reworded), and
  `section` / `target_section`
- `changed_cost=` parameter on `semantic_distance_with_details` to recompute the `changed` flag
  against a caller-chosen band
- Python: both `*_with_details` accept `markdown=`, `min_words=` and `skip_table_fragments=` with the
  same semantics as the CLI flags

### Changed

- `changed` on a semantic flow now reads the cheapest match in the statement's cost row, not the
  first listed flow, so a statement split across targets is judged by its best counterpart
- Statement ordering in detail outputs uses a stable argsort, making detail files reproducible
  across runs

### Fixed

- Machine-readable error contract: with `--json` or `--result-only`, every failure path writes a
  one-line `{"error": code, "message": ...}` object to stdout and exits 1 - including detail-file
  and source-map writes, `--pairs` file reads and malformed TSV lines (codes: `not_initialized`,
  `models_not_installed`, `io_error`, `empty_document`)
- A mistyped file path scored as literal text now emits a stderr warning naming the suspect argument
- H1 heading strip is fence-aware and no longer removes a `# ` line inside a fenced code block

## [1.1.4] - 2026-08-12

### Fixed

- Machine-readable output, calibration and provisioning defects found by an adversarial review
- Ruff pinned and its widened default rules absorbed

## [1.1.3] and earlier

Released before this changelog existed; see the git history and the
[Medium article](https://medium.com/towards-artificial-intelligence/docdistance-a-way-to-measure-distance-between-documents-229d4f762eaf)
for the feature arc: SMD semantic distance, the OPW order-gap structural read, the
source-conditioned `d(A,B|S)` variant and the OpenVINO INT8 CPU pipeline.
