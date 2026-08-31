# Changelog

All notable changes to docdistance are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow semantic versioning.

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
