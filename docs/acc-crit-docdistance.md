# Acceptance Criteria - docdistance

Criteria for the docdistance semantic and structural document-distance library, tracked with pm-tools; ids are permanent, status lives in the checkbox.

## Authors

- `@kj` Konrad Jelen

## Segmentation and input handling `SEGM`

How document bodies are read, stripped and split into statements before embedding

- [x] `ACC-SEGM-1` **markdown-aware segmentation option** - HIGH; a segmentation option treats a table row, heading, list item and fenced-code line as one statement each, embedding marker-stripped text while reporting the raw text, with only prose paragraphs going through SaT; today the segmenter gets the flattened body whole, `encoders.py:141-143`, `pipeline.py:259`, producing the fragment population (40% of a field document); default behaviour unchanged
  - evidence: split_markdown: one statement per table row/heading/list item/fenced line, markers stripped for embedding, raw line reported, prose only to SaT; default path unchanged; make test 141 passed 1 skipped
  - related: ACC-SEGM-2 - the post-hoc filter for the same fragment noise
  - test: segment a doc with a table, list and fence under the markdown option; assert one statement per table row, heading and list item, markers stripped from embedded text, raw line reported
  - test-tags: UNIT
  - log: 2026-08-31T12:30:33Z @kj added
  - log: 2026-08-31T13:42:59Z @kj closed
- [x] `ACC-SEGM-2` **fragment filter with dropped statements counted** - HIGH; a --min-words N and --skip-table-fragments filter applies before encoding, with n_dropped and the drop rules recorded in both detail-file headers and the CLI JSON so nothing is silently lost; today everything is embedded and every fragment carries exactly 1/n transport mass, `pipeline.py:256-262`, `distance.py:48-53`; documentation states that filtering changes the SMD because the marginals renormalise
  - evidence: _filter_fragments no-op at defaults, min_words/table rules drop before encoding, n_dropped+drop_rules in all detail headers and machine JSON when active, all-dropped raises; SMD renormalisation documented; make test 141 passed
  - related: ACC-SEGM-1 - the source-level fix for the same fragment noise
  - test: run with --min-words 4 on a fixture with table fragments; assert fragments dropped, n_dropped and rules in the header, smd computed over the filtered set
  - test-tags: UNIT
  - log: 2026-08-31T12:30:33Z @kj added
  - log: 2026-08-31T13:42:59Z @kj closed

## Detail-file exposure `MAP`

Fields, views and metadata the semantic and structural detail files carry for downstream readers

- [x] `ACC-MAP-3` **heading path on every statement record** - HIGH; every record in the semantic flows, the structural statements and the source map carries a section field holding the nearest preceding heading chain; today records hold only index and text, `pipeline.py:116-128`, `pipeline.py:194-203`, so every per-section figure needs an external heading-split and text-search join
  - evidence: section field on semantic flows, targets, structural records (+target_section) and source map; chains exact on two-section fixture, pre-heading empty; 15 section tests green
  - test: details of a two-section doc; assert every record carries section equal to its nearest heading chain
  - test-tags: UNIT
  - log: 2026-08-31T12:30:33Z @kj added
  - log: 2026-08-31T13:42:59Z @kj closed
- [x] `ACC-MAP-4` **target-side view in the semantic map** - HIGH; the semantic detail file emits a targets array beside flows: per B statement its index, text, best incoming source, min incoming cost and total incoming weight, so added-content queries read directly; today consumers invert the per-source matches lists themselves (three independent reimplementations in the forensics archive), `pipeline.py:132-163`
  - evidence: targets array one record per B statement; best_source_index/min_incoming_cost equal cost-column minima, total_incoming_weight equals summed plan; verified against hand inversion of flows
  - test: assert the targets array lists every B statement with best incoming source and min incoming cost matching an inversion of flows
  - test-tags: UNIT
  - log: 2026-08-31T12:30:33Z @kj added
  - log: 2026-08-31T13:42:59Z @kj closed
- [x] `ACC-MAP-5` **ground cost on structural statement records** - MEDIUM; each structural statement record carries the ground cost of its aligned pair, cost[i, align[i]], so moved-intact separates from vanished-with-absorption without joining the semantic file; the cost matrix is already computed in the same function, `pipeline.py:185` vs the record at `pipeline.py:192-203`; pure exposure, nothing about the transport changes
  - evidence: each structural record cost equals round(cost_matrix[i, align[i]], 4), hand-checked on synthetic embeddings against order_alignment; pure exposure, order_gap untouched
  - test: assert each structural record carries cost equal to cost_matrix[i, align[i]] rounded to 4 places
  - test-tags: UNIT
  - log: 2026-08-31T12:30:50Z @kj added
  - log: 2026-08-31T13:43:07Z @kj closed
- [x] `ACC-MAP-6` **both targets on one structural record** - MEDIUM; each structural record carries semantic_target_index with its cost plus a targets_differ boolean beside the structural target; the two couplings disagreed on 233 of 1,086 field statements and that disagreement locates moved-and-reworded content; the transport plan is already computed and discarded in _build_structural_details, `pipeline.py:185-189`; neither distance changes
  - evidence: semantic_target_index = plan argmax with cost and targets_differ on every record, hand-verified from the already-computed plan; no extra solve, distances untouched
  - test: record where the plan argmax differs from the structural target has targets_differ true and both indices present
  - test-tags: UNIT
  - log: 2026-08-31T12:30:50Z @kj added
  - log: 2026-08-31T13:43:07Z @kj closed
- [x] `ACC-MAP-7` **self-describing detail headers** - MEDIUM; every details file is stamped with generator name and version, closeness value with its definition string (1 - smd/sqrt(2)), and the changed/moved definitions and thresholds in force; today headers carry only smd/anisotropy/n_statements plus the structural scalars, `pipeline.py:158-163`, `pipeline.py:204-211` - a field agent computed 1 - SMD and reported 0.846 where the CLI said 0.891, and hand-rolled substitute files were schema-indistinguishable from genuine ones
  - evidence: headers carry generator, version, closeness with definition 1 - smd/sqrt(2) (structural mirror), and all changed/moved/no_counterpart thresholds; closeness recomputed from unrounded smd matches
  - test: assert the header carries generator, version, closeness, closeness_definition and the changed/moved thresholds
  - test-tags: UNIT
  - log: 2026-08-31T12:30:50Z @kj added
  - log: 2026-08-31T13:43:07Z @kj closed
- [x] `ACC-MAP-8` **aggregate summary block in detail headers** - MEDIUM; the semantic header carries counts of best-match cost per band (0-0.1, 0.1-0.35, 0.35-0.6, above 0.6), n_changed and the top-N statements by cheapest-match cost; the structural header mirrors a displacement and moved summary; today every consumer of a 560-630 KB file recomputes the same aggregates, `pipeline.py:158-163`
  - evidence: summary cost_bands/n_changed/n_no_counterpart/top_changed match hand aggregation over cheapest-row cost; structural n_moved/median_displacement_norm match, median null on all-no-counterpart fixture
  - test: assert header cost-band counts equal bands recomputed from the flows array
  - test-tags: UNIT
  - log: 2026-08-31T12:30:50Z @kj added
  - log: 2026-08-31T13:43:07Z @kj closed
- [x] `ACC-MAP-9` **per-section aggregates in the detail file** - MEDIUM; given heading paths, the detail files emit per-section statement counts, mean cost, share below a cost band, and for the structural map the median displacement and the share of statements whose target lies in a different section - the section drift table computed once by the tool instead of once per consumer (insights doc, What could be done)
  - evidence: per-section n_statements/mean_cost/share_below_changed plus structural median_displacement_norm and share_target_other_section match hand recomputation on two-section fixture; nulls correct
  - blocked-by: ACC-MAP-3
  - test: assert the per-section table matches a reference computed from the raw records for a two-version fixture
  - test-tags: UNIT
  - log: 2026-08-31T12:30:51Z @kj added
  - log: 2026-08-31T13:43:07Z @kj closed

## Transport and flag semantics `TRANS`

The OT plan, its marginals, and the changed/moved flags derived from it

- [x] `ACC-TRANS-10` **changed threshold exposed and identical flag added** - HIGH; a changed_cost parameter on semantic_distance_with_details defaults to DIFF_CHANGED_COST and is emitted in the detail header, and each flow carries an identical field (normalised-text equality), so identical / edited-below-threshold / changed are three queryable states; today the threshold is a frozen module constant written nowhere, `pipeline.py:32-35`, `pipeline.py:154-157`, and the copy-editing band (0 of 41 article removals flagged) is unreachable without reverse-engineering it
  - evidence: changed_cost parameter flags per threshold (1e-9 all, 10 none), value emitted in header thresholds; per-flow identical equals normalised-text equality with cheapest target; smd bit-identical across values
  - related: DEF-TRANS-3 - the flag this makes queryable reads the wrong match
  - test: pass changed_cost=0.1; assert flags recomputed against it and changed_cost plus per-flow identical present in the output
  - test-tags: UNIT
  - log: 2026-08-31T12:31:07Z @kj added
  - log: 2026-08-31T13:43:16Z @kj closed
- [x] `ACC-TRANS-11` **stable argsort tie-break in the rank displacement** - LOW; both argsort calls in the displacement rank use kind=stable so that when n_A > n_B forces shared crisp targets, tied ranks order by source position by contract and an order-preserving shrinkage yields displacement 0 on every numpy build, `distance.py:249-253`; hardening only - the failure never reproduced on numpy 2.4.6 across 300+ adversarial monotone alignments; plan, distance and metric guarantee untouched
  - evidence: both argsort calls in structure_displacement carry kind=stable; 300 monotone shrinkages with shared crisp targets all yield zero displacement; regression test shipped
  - related: DEF-TRANS-2 - hardens the same displacement rank
  - test: monotone shrinkage alignment with shared targets yields all-zero displacement
  - test-tags: UNIT
  - log: 2026-08-31T12:31:07Z @kj added
  - log: 2026-08-31T13:43:16Z @kj closed
- [ ] `ACC-TRANS-14` **moved is the conjunction of rank displacement and norm displacement** - HIGH; moved fires only when displacement != 0 AND abs(displacement_norm) > MOVED_TOLERANCE (E12-H61); one statement inserted at position 0 of the 6-statement worked example flags 0 of 6 untouched statements (shipped norm-only flag: 4 of 6)
  - test: insertion fixture from E12-H61 (doc-a plus B with one novel statement at position 0); assert n_moved == 0
  - test-tags: UNIT
  - log: 2026-09-02T05:35:03Z @kj added
- [ ] `ACC-TRANS-15` **Edge: genuine swap still flagged** - HIGH; on the unmodified worked example (docs/example-map-interpretations.md doc-a / doc-b) both swapped statements read moved under the conjunction
  - test: run structural_distance_with_details on doc-a / doc-b; assert both swapped statements carry moved
  - test-tags: UNIT
  - log: 2026-09-02T05:35:03Z @kj added
- [ ] `ACC-TRANS-16` **Edge: reversal control 12/12** - HIGH; a fully reversed 12-statement document flags all 12 statements moved under the conjunction
  - test: reverse a 12-statement document; assert n_moved == 12
  - test-tags: UNIT
  - log: 2026-09-02T05:35:03Z @kj added
- [ ] `ACC-TRANS-17` **Edge: shrinkage stays quiet** - MEDIUM; a 12 to 8 shrinkage (four statements removed, order kept) flags 0 of the 8 kept statements moved
  - test: build the 12->8 pair from E12-H61; assert no kept statement carries moved
  - test-tags: UNIT
  - log: 2026-09-02T05:35:03Z @kj added
- [ ] `ACC-TRANS-18` **Edge: reorder pool retains every true relocation** - HIGH; across the structure-fixture reorder pool (data/processed/structure-fixture, 1680 permutations) no statement flagged moved by the shipped definition and relocated in ground truth loses its flag (E12-H61 read 0 of 13675)
  - test: re-run the E12-H61 pool sweep against the shipped build; assert lost == 0
  - test-tags: FUNCTIONAL
  - log: 2026-09-02T05:35:03Z @kj added
- [ ] `ACC-TRANS-19` **Conjunction leaves the distance layer bit-identical** - CRITICAL; default-path smd, order_gap and every non-flag field of the details dict are byte-identical to the 1.1.5 output on the six E12-H61 fixtures and the ibm 11-summary panel (V stays 0/24)
  - test: np.array_equal on smd / order_gap and a byte diff of the flag-stripped details dict before and after; count V on the ibm panel
  - test-tags: UNIT, FUNCTIONAL
  - log: 2026-09-02T05:35:03Z @kj added
- [ ] `ACC-TRANS-20` **Definitions follow the conjunction** - MEDIUM; the pipeline header definitions block, the thresholds/definitions field of the structural detail JSON and the moved table in docs/example-map-interpretations.md all state the conjunction definition
  - test: read the definitions field of a structural --details-json output and the docs table; assert both name displacement AND norm
  - test-tags: UNIT, MANUAL
  - log: 2026-09-02T05:35:03Z @kj added

## CLI and config operation `CLI`

Command-line behaviour, config discovery, error output and batch execution

- [x] `ACC-CLI-12` **batch mode with one model load** - HIGH; distance-semantic and distance-structural accept --pairs FILE (TSV of a-path, b-path, optional id), build one DocDistance, loop the pairs and emit one JSON array or JSONL to stdout, --details-json taking a directory or template in that mode; today each invocation loads models fresh, `cli.py:239-263`, `cli.py:309-337`, while the library already caches per instance, `pipeline.py:214-234` - a 120-pair field sweep cost 45 minutes through the Python API
  - evidence: --pairs TSV on both distance commands: one DocDistance construction for 3 pairs, JSONL out with ids, --details-json dir and {pair} template per pair; positional/pairs exclusion exit 2, malformed line exit 1 naming the line
  - test: distance-semantic --pairs with 3 pairs runs in one process with one model load and emits one result per pair
  - test-tags: FUNCTIONAL
  - log: 2026-08-31T12:31:07Z @kj added
  - log: 2026-08-31T13:43:16Z @kj closed
- [x] `ACC-CLI-13` **config error names the searched path, distance commands take an override** - HIGH; NotInitializedError names the config path searched and the DOCDISTANCE_HOME/DOCDISTANCE_CONFIG variables, and the three distance commands accept a --home/--config override mirroring init; today the message says only run docdistance init, `settings.py:74-79`, and --home exists on init alone, `cli.py:432-437` - the upstream cause of the field job losing map pairs
  - evidence: un-initd run exits 1 naming the searched docdistance.json and DOCDISTANCE_HOME/DOCDISTANCE_CONFIG in human and JSON output; same command with --home at an initd folder passes on all three distance commands
  - related: DEF-CLI-4 - the silent stdout half of the same config miss
  - test: run a distance command from a folder without docdistance.json; the error names the searched path; the same command with --home pointing at the config succeeds
  - test-tags: FUNCTIONAL
  - log: 2026-08-31T12:31:07Z @kj added
  - log: 2026-08-31T13:43:16Z @kj closed
- [ ] `ACC-CLI-29` **Embed memo on --pairs batches** - HIGH; DocDistance memoises segment-and-encode per (resolved path, mtime, size, segmentation kwargs including fragment filters) so the E12-H67 55-row all-pairs sweep performs 11 real encodes for 110 slots
  - test: run the E12-H67 all-pairs TSV with an encoder call counter; assert real encodes == 11
  - test-tags: UNIT
  - log: 2026-09-02T05:35:03Z @kj added
- [ ] `ACC-CLI-30` **Memo output byte-identical** - CRITICAL; distance-semantic --pairs JSONL, distance-structural --pairs JSONL and every per-pair --details-json file are byte-identical with the memo on and off across the nine E12-H67 arms
  - test: sha256 each arm's JSONL and details files under both settings; assert equal
  - test-tags: FUNCTIONAL
  - log: 2026-09-02T05:35:04Z @kj added
- [ ] `ACC-CLI-31` **Batch speed-up at least 3x** - HIGH; median wall time of the 55-row all-pairs sweep with the memo on is at most one third of the memo-off time on a quiet CPU (E12-H67 measured 9.94x)
  - test: three interleaved off/on repeats at 1-minute load below 8; assert median ratio >= 3
  - test-tags: FUNCTIONAL
  - log: 2026-09-02T05:35:04Z @kj added
- [ ] `ACC-CLI-32` **Edge: zero-hit batch within 2%** - MEDIUM; on a disjoint-pairs sweep of 110 unique files the memo-on wall time is within 2% of memo-off (E12-H67 control -0.63%)
  - test: run the E12-H67 disjoint-55 TSV both ways; assert abs(delta) <= 2%
  - test-tags: FUNCTIONAL
  - log: 2026-09-02T05:35:04Z @kj added
- [ ] `ACC-CLI-33` **Edge: rewritten file re-encodes** - HIGH; a file rewritten between two rows (new mtime or size) is segmented and encoded again, never served from the memo
  - test: score a pair, rewrite one file, score again; assert one new real encode and the new distance
  - test-tags: UNIT
  - log: 2026-09-02T05:35:04Z @kj added
- [ ] `ACC-CLI-34` **Edge: memo is bounded** - HIGH; the embed memo carries a size bound (entries or bytes) and evicts beyond it, so a long sweep cannot grow memory without limit
  - test: set the bound to 3 and encode 4 distinct documents; assert the first is re-encoded on its next use
  - test-tags: UNIT
  - log: 2026-09-02T05:35:04Z @kj added
- [ ] `ACC-CLI-35` **Edge: raw text bypasses the memo** - MEDIUM; inputs given as text rather than a path are never memoised and never served from the memo
  - test: score the same raw text twice with a call counter; assert two real encodes
  - test-tags: UNIT
  - log: 2026-09-02T05:35:04Z @kj added
- [ ] `ACC-CLI-36` **Edge: dropped-fragment count restored on a hit** - MEDIUM; the segmentation drop count reported for a document is identical whether the document was encoded or served from the memo
  - test: encode a document that drops fragments, read n_dropped, score it again from the memo; assert equal
  - test-tags: UNIT
  - log: 2026-09-02T05:35:04Z @kj added
- [ ] `ACC-CLI-37` **--pairs on distance-wrt-source** - HIGH; distance-wrt-source accepts --pairs <tsv> with a fixed --source and emits one JSON line per row in row order, the same shape as the one-shot --json output
  - test: 55-row TSV against one source; assert 55 lines, ids in order, keys equal to the one-shot output
  - test-tags: FUNCTIONAL
  - log: 2026-09-02T05:35:04Z @kj added
- [ ] `ACC-CLI-38` **Grounding-grid memo per (doc, source)** - HIGH; within one DocDistance instance the reranker relevance grid and the NLI entailment vector are computed once per (doc, source) and reused, so the E12-H68 55-pair sweep evaluates 11 grids for 110 slots
  - test: run the sweep with a grid call counter; assert grids == 11
  - test-tags: UNIT
  - log: 2026-09-02T05:35:04Z @kj added
- [ ] `ACC-CLI-39` **Wrt-source batch lines byte-identical** - CRITICAL; each of the 55 batch lines is byte-identical to the one-shot distance-wrt-source --json output for that pair, with the memo on and off
  - test: sha256 each batch line against its one-shot run; assert 55/55 equal
  - test-tags: FUNCTIONAL
  - log: 2026-09-02T05:35:04Z @kj added
- [ ] `ACC-CLI-40` **Wrt-source sweep speed-up at least 5x** - HIGH; wall time of the 55-pair sweep with the memo on is at most one fifth of the warm sequential loop (E12-H68 measured 10.03x)
  - test: one interleaved off/on pass at 1-minute load below 8; assert ratio >= 5
  - test-tags: FUNCTIONAL
  - log: 2026-09-02T05:35:04Z @kj added
- [ ] `ACC-CLI-41` **Edge: memo hits return copies** - HIGH; a grid or entailment vector returned from the memo is a copy, so a caller mutating it does not change a later hit
  - test: take a hit, mutate it in place, take a second hit; assert the second equals the original
  - test-tags: UNIT
  - log: 2026-09-02T05:35:04Z @kj added
- [ ] `ACC-CLI-42` **Edge: grid memo is bounded** - HIGH; the grounding memo carries a size bound and evicts beyond it
  - test: set the bound to 2 and score 3 distinct documents against one source; assert the first grid is recomputed on its next use
  - test-tags: UNIT
  - log: 2026-09-02T05:35:04Z @kj added
- [ ] `ACC-CLI-43` **Edge: no-fanout sweep within 2%** - MEDIUM; a wrt-source sweep where every document appears once records 0 memo hits and a wall time within 2% of the memo-off loop (E12-H68 control +0.22%)
  - test: run the E12-H68 nofanout TSV both ways; assert hits == 0 and abs(delta) <= 2%
  - test-tags: FUNCTIONAL
  - log: 2026-09-02T05:35:04Z @kj added

## Embedding geometry pre-pass `GEOM`

The anisotropy (common-mode rejection) pre-pass on the conditioned selection axis: what it computes and how much corpus it needs

- [ ] `ACC-GEOM-21` **Anisotropy pre-pass is centering plus renormalisation** - MEDIUM; the conditioned-path anisotropy pre-pass centers the pooled statement matrix and re-L2-normalises without the SVD projection (E12-H64: centering carries 103.7% of the gain, the projection isolate -4.5%); ships only after a second fixture reproduces the attribution, and the ibm panel reads V 0/24 with margin within 1 pt of the E12-H64 arm a (+19.95 pts INT8)
  - test: re-run the E01 metric panel on the shipped path after the change; assert V == 0 and margin within 1 pt of +19.95 (INT8)
  - test-tags: FUNCTIONAL
  - log: 2026-09-02T05:35:03Z @kj added
- [ ] `ACC-GEOM-22` **Minimum estimation pool stated in statements** - MEDIUM; the anisotropy docstring (src/docdistance/distance.py, anisotropy parameter) and the --anisotropy rows of docs/cli-reference.md state the minimum estimation pool as about 71 pooled statements (the E12-H65 reading: 8 of the ibm summaries, direction cosine >= 0.98 to the full-pool direction, 93% of the full-pool d'), with the statement count as the rule and the document count only as that fixture's example
  - test: grep the docstring and the CLI reference for the pooled-statement minimum; assert the rule is stated in statements
  - test-tags: MANUAL
  - log: 2026-09-02T05:35:03Z @kj added
- [ ] `ACC-GEOM-23` **Edge: pool below the minimum warns** - LOW; a --pairs or corpus run whose pooled statement count is below the documented minimum logs one warning naming the count and the minimum and proceeds
  - test: run distance-semantic --pairs on a 2-document TSV with anisotropy on; assert one warning line with both numbers
  - test-tags: UNIT
  - log: 2026-09-02T05:35:03Z @kj added

## Verdict calibration guidance `CALIB`

How the shipped verdict thresholds were placed, how fragile they are, and how a user recalibrates them on their own corpus

- [ ] `ACC-CALIB-24` **Structural threshold comment names its corpus** - MEDIUM; the STRUCTURE_THRESHOLD comment in src/docdistance/distance.py states 0.90 is placed on the ibm ladder with 0.006144 headroom, quotes the ibm band (0.892817, 0.906144] and the wergeland band (0.908048, 0.973676], and says no cross-article band exists (E12-H59)
  - test: read the comment; assert the two bands and the empty-band sentence are present
  - test-tags: MANUAL
  - log: 2026-09-02T05:35:03Z @kj added
- [ ] `ACC-CALIB-25` **Per-corpus structural calibration procedure** - HIGH; docs/solution/wmd-structure-distance-sota.md carries a per-corpus procedure for STRUCTURE_THRESHOLD: build the corpus's own k-swap ladder and un-reordered pairs, sweep 0.85-0.99, take the band that keeps every rung on its side, with the E12-H59 two-article sweep as the worked example and the empty cross-article band stated
  - test: read the section; assert the four steps, the worked example and the empty-band statement are present
  - test-tags: MANUAL
  - log: 2026-09-02T05:35:03Z @kj added
- [ ] `ACC-CALIB-26` **Semantic threshold fragility card published** - HIGH; docs/solution/wmd-docdistance-solution-sota.md carries the E12-H63 card for DEFAULT_THRESHOLD 0.77: shipped-path band [0.7474, 0.7944] on INT8 and [0.7566, 0.8027] on fp, safe on both over 0.760-0.790, leave-one-out edge moves (haiku +2.30 pts, adv2-b -1.64 pts) and INT8-to-fp drift up to 40.6% of a margin
  - test: read the card; assert both bands, the safe range and the two edge documents are present
  - test-tags: MANUAL
  - log: 2026-09-02T05:35:03Z @kj added
- [ ] `ACC-CALIB-27` **Provenance comment quotes the shipped-path band** - MEDIUM; the DEFAULT_THRESHOLD comment in src/docdistance/distance.py quotes the shipped semantic_distance band (79.44 / 74.74 on INT8) and names notebook 09 cell 10's dd.embed path as the origin of the former 79.7 / 75.0
  - test: read the comment; assert the shipped-path numbers and the origin sentence are present
  - test-tags: MANUAL
  - log: 2026-09-02T05:35:03Z @kj added
- [ ] `ACC-CALIB-28` **Structural verdict caveat under deletions** - MEDIUM; the structural verdict documentation states that order-preserving deletions lower structure_closeness below 0.90 (E12-H62: 0.8902 pooled at k=4, 0.8219 for tail deletions at k=3), so a content drop can read as a rearrangement until a length correction ships
  - test: read docs/solution/wmd-structure-distance-sota.md; assert the deletion caveat with both numbers is present
  - test-tags: MANUAL
  - log: 2026-09-02T05:35:03Z @kj added

