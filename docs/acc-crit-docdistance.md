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

