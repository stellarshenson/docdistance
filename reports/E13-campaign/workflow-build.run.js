export const meta = {
  name: 'docdistance-e13-build',
  description: 'Stage B of the E13 campaign: implement the work packages in order (opus implementer, adversarial verify, one repair round each), then one final gate. The three-lens review is stage C (workflow-review.js)',
  phases: [
    { title: 'Build', detail: 'one implementer per work package, in order, one working tree', model: 'opus' },
    { title: 'Verify', detail: 'adversarial check of each package against its criteria and defects' },
    { title: 'Gate', detail: 'lint, tests, baseline and Python 3.12 syntax on the final tree' },
  ],
}

// args: { packages: [{ key, title, items: [ids], files: [paths], brief, grounded }], allow: 'moved,n_moved' }
// The package list lives in reports/E13-campaign/build-args.json; pass its content as `args`.
const ROOT = '/home/lab/workspace/learning/projects/docdistance'
const PRIMER = `${ROOT}/reports/E13-campaign/PRIMER.md`
const BUILD_MODEL = 'opus'
const BUILD_AGENT = 'senior-backend-engineer'
const REVIEW_AGENT = 'devils-advocate:adversarial-reviewer'
// Generated from workflow-build.js and build-args.json (2026-10-09). Do not edit: change the two inputs and regenerate.
const BUILD_ARGS = {
 "allow": "moved,n_moved",
 "packages": [
  {
   "key": "runtime-guards",
   "title": "OpenVINO telemetry block and exact-solver guard",
   "items": [
    "DEF-DEPS-11",
    "DEF-TRANS-9"
   ],
   "files": [
    "src/docdistance/encoders.py",
    "src/docdistance/distance.py",
    "tests/test_distance.py",
    "tests/test_e2e.py"
   ],
   "grounded": false,
   "brief": "DEF-DEPS-11: stop OpenVINO's usage telemetry before the first `import openvino` in encoders.py, so no child process starts and no event leaves the machine. Candidate mechanism, verify it before relying on it: `sys.modules.setdefault('openvino_telemetry', None)` makes `import openvino_telemetry` raise ImportError and OpenVINO falls back to its bundled `openvino.tools.ovc.telemetry_stub`. Test in a subprocess with `CI` removed from the environment: a script without a main guard that prints a marker and loads the encoder prints the marker once. DEF-TRANS-9: route the `ot.emd` / `ot.emd2` call sites in distance.py through one private helper that sets `numItermax` from the problem size, reads the solver's result with `log=True` and raises a clear error when the solver reports it stopped before optimality. Values on problems the default cap already solves must stay bit-identical. Test: a cap forced low on a small problem raises; 3000 x 3000 random unit vectors solve without the warning."
  },
  {
   "key": "transport-core",
   "title": "partial matching, matched order gap and the unsupported-statement count",
   "items": [
    "DEF-TRANS-10",
    "ACC-TRANS-47",
    "ACC-TRANS-48",
    "ACC-TRANS-49",
    "ACC-TRANS-50",
    "ACC-TRANS-51",
    "ACC-TRANS-52",
    "ACC-TRANS-53",
    "ACC-TRANS-54",
    "ACC-TRANS-71",
    "ACC-TRANS-76",
    "DEF-TRANS-8",
    "ACC-TRANS-66",
    "ACC-TRANS-67",
    "ACC-TRANS-68",
    "ACC-TRANS-69",
    "ACC-TRANS-70",
    "ACC-TRANS-72",
    "ACC-CLI-74",
    "ACC-CLI-75"
   ],
   "files": [
    "src/docdistance/distance.py",
    "src/docdistance/__init__.py",
    "src/docdistance/cli.py",
    "pyproject.toml",
    "tests/test_distance.py",
    "tests/test_structure.py",
    "tests/test_cli.py",
    "tests/test_e2e.py"
   ],
   "grounded": true,
   "brief": "Two additions to distance.py, both additive: every number 1.1.5 returns stays bit-identical. PART 1 - partial matching and the matched order gap (E13-H70 Promoted, E13-H71 Promoted, E15-H79 for the constants). `MATCH_CUTOFF = 0.325` beside the existing thresholds with the provenance ACC-TRANS-48 asks for; note in the comment that it is a measured value and not the derived `DIFF_CHANGED_COST` of pipeline.py (0.3253). `partial_matching(X, Y)` returns the one-to-one index pairs: `scipy.optimize.linear_sum_assignment` on `np.minimum(cost, c) + eps * abs(i / n - j / m)` with `cost = cost_matrix(X, Y)`, `c = MATCH_CUTOFF`, `eps = 1e-6`, i and j counted from 0, n and m the statement counts; a pair whose ground cost WITHOUT the position term is >= c stays unmatched. These are the constants the experiment persisted (reports/E15-H79-matching-tie-break-metrics.json, key `build_constants`); copy the form exactly, including the float32 ground cost. `scipy` becomes a direct import: declare it under `dependencies` in pyproject.toml before `make install`. `matched_order_gap(X, Y)` is the shipped `opw_gap` on the matched sub-sequences, each in its own document order; `None` when no pair is matched. When both documents have the same count and every statement is matched it must call `opw_gap(X, Y)` on the unmodified arrays so the value is exactly the shipped one (ACC-TRANS-50, np.array_equal). `StructuralResult` gains `matched_order_gap` and `n_matched`; `order_gap`, `structure_closeness`, `threshold` and `verdict` keep their values (ACC-TRANS-54). The human output of `distance-structural` shows both new values, `--result-only` is unchanged (ACC-CLI-74). The position term is the rule among equal-cost matchings (ACC-TRANS-71): E15-H79 was Refuted on its bar (matched sub-sequence textually equal in 111/120 pure deletions on a document with 9 repeated statements, bar 114), so the docstring states the term as a determinism rule with that measured limit and does not claim that repeated statements are solved. Geometry rule (ACC-TRANS-76): `MATCH_CUTOFF` was calibrated on raw embeddings, so `compute_structural` computes the matched pairs on the embeddings as passed in, before `all_but_the_top`; with `anisotropy=True` the order gap over those pairs is `opw_gap` of the matched sub-sequences in the pre-pass geometry, the geometry `order_gap` uses, so a fully matched equal-length pair still reads `matched_order_gap == order_gap`. DEF-TRANS-10 is answered by the new field plus corrected docstrings: `opw_gap` and `StructuralResult` state that on unequal statement counts the order gap includes the cost of the surplus statements, that `opw_gap(P, P)` is not zero and grows with length (5.182e-03 at 30 statements, 6.786e-03 at 53), and that `matched_order_gap` is the reading over statements both documents hold. Its repro (ibm gold summary minus 2 statements, rng seed 0: `opw_gap` 0.227) becomes a test on synthetic unit vectors: a pure deletion reads `matched_order_gap == opw_gap(B, B)` while `opw_gap(A, B)` is larger. PART 2 - unsupported-statement count (E15-H78 Promoted; fix of DEF-TRANS-8). `SOURCE_SUPPORT_CUTOFF = 0.34625` beside `MATCH_CUTOFF` with the provenance ACC-TRANS-66 asks for (reports/E15-H78-raw-nearest-source-count-metrics.json). The count of statements whose cost to the nearest source statement, `cost_matrix(X, S).min(1)`, is above the cutoff. `compute_source_conditioned` computes it on the embeddings as passed in, BEFORE `all_but_the_top` runs: after the pre-pass the reading fails its bar (held-out balanced accuracy 0.8168 / 0.7277 against 0.9412 / 0.9272 on raw embeddings), so `anisotropy=True` and `anisotropy=False` return the same counts. `SourceConditionedResult` gains `n_unsupported_a` and `n_unsupported_b`; every existing field keeps its value (ACC-TRANS-70, the `--grounded` guardrail). The human output of `distance-wrt-source` shows both counts, `--result-only` is unchanged (ACC-CLI-75). The docstrings of `ungrounded_residual` and `SourceConditionedResult` are corrected as ACC-TRANS-72 says: `residual_a` is the balanced transport cost to the whole source and reads 0.958 for 10 statements copied verbatim from a 70-statement source; it is not the unsupported share of a document. The count is a score, not a metric - say so. Tests are unit tests on synthetic unit vectors (seeded `np.random.default_rng`), no model load; the CLI tests follow the patterns already in tests/test_cli.py and tests/test_e2e.py. Export the new public names in `__init__.py` the way the existing ones are exported."
  },
  {
   "key": "detail-map",
   "title": "moved flag on matched pairs and the matching fields of the detail files",
   "items": [
    "DEF-TRANS-13",
    "ACC-TRANS-77",
    "ACC-TRANS-78",
    "ACC-TRANS-79",
    "ACC-TRANS-15",
    "ACC-TRANS-16",
    "ACC-TRANS-17",
    "ACC-TRANS-18",
    "ACC-TRANS-19",
    "ACC-TRANS-20",
    "ACC-MAP-55",
    "ACC-MAP-56",
    "ACC-MAP-57",
    "ACC-MAP-58",
    "ACC-MAP-59",
    "ACC-MAP-60",
    "ACC-MAP-61",
    "ACC-MAP-62",
    "ACC-MAP-65",
    "ACC-MAP-73"
   ],
   "files": [
    "src/docdistance/pipeline.py",
    "tests/test_pipeline.py",
    "tests/test_structure.py",
    "tests/test_sections.py",
    "docs/example-map-interpretations.md"
   ],
   "grounded": true,
   "brief": "All in pipeline.py, built on `partial_matching` and `MATCH_CUTOFF` of the transport-core package (already in the tree). Additive policy: every key the 1.1.5 detail files carry keeps its value on the 55 ibm pairs, except `moved` and `n_moved` (ACC-MAP-59, the guardrail runs with `--allow moved,n_moved`; the key name `moved` inside `definitions` is covered by the same allowance). Existing tests assert exact key sets of the detail dicts and records - extend those sets, do not loosen them. Geometry rule (ACC-TRANS-76, criterion of the previous package): the matching is computed on the embeddings as passed in, before `all_but_the_top`, in all three builders, so the match fields do not change with `anisotropy`. PART 1 - structural detail file (`_build_structural_details`). Each statement record gains `matched_displacement`: the rank of its matched target among the matched targets minus its own rank among the matched statements, `null` when the statement has no match (ACC-MAP-60). `moved` becomes the definition E15-H80 measured (ACC-TRANS-77; ACC-TRANS-14 was rejected as overridden): `matched AND matched_displacement != 0 AND abs(matched_displacement / n_matched) > MOVED_TOLERANCE AND NOT no_counterpart`. Use exactly that float form, `matched_displacement / n_matched`: it is the measured one, and it differs from `j / n_b - i / n_a` at a rank step of exactly 5% of the length. The shipped rule that a `no_counterpart` record reads `moved: false` stays. `displacement`, `displacement_norm`, `target_index`, `no_counterpart` and every other shipped field keep their values. Readings to reproduce as tests: one statement inserted at position 0 of the 6-statement worked example flags 0 of 6; the worked-example swap flags both swapped statements (ACC-TRANS-15); a reversed 12-statement document flags 12 of 12 (ACC-TRANS-16); a 12 to 8 order-preserving deletion flags 0 of 12, kept and removed alike, with `n_moved == 0` (ACC-TRANS-17, ACC-TRANS-78 - this is the repro of DEF-TRANS-13, which read 1 or 2 removed statements as moved depending on the build); on fully matched equal-length pairs the flag equals the 1.1.5 conjunction `displacement != 0 and abs(displacement_norm) > MOVED_TOLERANCE` statement for statement (ACC-TRANS-79, ACC-TRANS-18: re-run the reorder-pool sweep of notebooks/experiments/E12-H61-kj-moved-conjunction.ipynb against the build and assert no true relocation loses its flag; the fixtures are in data/processed/structure-fixture and the fixture texts of the insertion, swap, reversal and shrinkage cases are in that notebook). Reference readings: reports/E15-H80-moved-on-matched-pairs-metrics.json. The `definitions` block of the structural file, the module comment above `MOVED_TOLERANCE`, the docstring of `_build_structural_details` and the moved table in docs/example-map-interpretations.md state the new definition and name both the displacement and its norm (ACC-TRANS-20); `thresholds` gains `match_cutoff`. ACC-TRANS-19: `smd`, `order_gap` and every non-flag field stay byte-identical. PART 2 - semantic detail file (`_build_semantic_details`). Each flow gains `match_index` (statement index in B, or `null`) and `match_status` `kept` or `removed` (ACC-MAP-55); each `targets` entry gains `match_status` `kept` or `added` and `match_source_index` (ACC-MAP-56); `summary` gains `n_kept`, `n_removed`, `n_added` with `n_kept + n_removed == n_a` and `n_kept + n_added == n_b` (ACC-MAP-57); `thresholds` gains `match_cutoff` and `definitions` one line each for kept, removed and added (ACC-MAP-58). Edges: a document against itself reads every flow kept (ACC-MAP-61); two documents with no pair under the cutoff read `n_kept == 0`, every flow removed, every target added (ACC-MAP-62). Reference readings: reports/E13-H71-partial-matching-map-metrics.json (held-out macro-F1 0.9772 / 0.9457 over kept / removed / added). PART 3 - source map (`_build_source_map`, ACC-MAP-73). Each `a` and `b` record gains `nearest_source_index`, `nearest_source_cost` (rounded like the other costs) and `unsupported` (`nearest_source_cost > SOURCE_SUPPORT_CUTOFF`), read on raw embeddings, before the pre-pass the map applies for its `matches`; the map header gains `source_support_cutoff` and `n_unsupported: {a, b}`. The counts must equal `n_unsupported_a` / `n_unsupported_b` of the result returned by the same call. `matches` and every existing key keep their values. PART 4 - ACC-MAP-65: one unit test with a segmenter stub that returns no statement asserts `ValueError` matching 'no statements' on `semantic_distance_with_details`, `structural_distance_with_details` and `distance_wrt_source_with_map`. The behaviour exists (`embed_statements`); only the test is new - confirm it fails if the raise is removed. Tests are unit tests on synthetic or stubbed embeddings wherever the criterion allows; a test that needs the model follows the existing pattern in tests/test_e2e.py."
  },
  {
   "key": "reader-and-pool-warnings",
   "title": "marker-statement warning, minimum pool warning and the pre-pass docstring",
   "items": [
    "DEF-SEGM-12",
    "ACC-GEOM-22",
    "ACC-GEOM-23",
    "ACC-GEOM-64"
   ],
   "files": [
    "src/docdistance/pipeline.py",
    "src/docdistance/distance.py",
    "src/docdistance/cli.py",
    "tests/test_segmentation.py",
    "tests/test_pipeline.py",
    "tests/test_cli.py",
    "docs/cli-reference.md",
    "docs/solution/wmd-docdistance-solution-sota.md"
   ],
   "grounded": false,
   "brief": "Three warnings and docstrings, no number changes. DEF-SEGM-12: when the default reader (markdown off) returns statements that hold no letter and no digit (bare list or heading markers such as '-', '###'), `DocDistance.embed_statements` logs ONE loguru warning per document naming the count, the statement total and the two options that remove them (`--markdown` / `markdown=True`, `--min-words` / `min_words`); the statements, the embeddings and every number stay as they are, and no warning is logged when markdown is on or when no such statement exists. Test with the ibm source `data/interim/exec-summaries/ibm-ai-adoption/source/source-article.md`: 21 of 70 with markdown off, 0 of 48 with markdown on - if the data file is absent in the test environment, build the fixture text inline (a heading line, a bare '-' line, two prose sentences). ACC-GEOM-22: state the minimum estimation pool as about 71 pooled statements in the `anisotropy` docstrings of distance.py and in the `--anisotropy` rows of docs/cli-reference.md, statement count as the rule and the 8 ibm summaries as that fixture's example (reading: reports/E12-H65-cmr-pool-crossover-metrics.json). ACC-GEOM-23: one named constant for that minimum beside the existing thresholds; a `--pairs` or corpus run with anisotropy on whose pooled statement count is below it logs one warning naming the count and the minimum and proceeds. Find where the pooled pre-pass is applied in pipeline.py / cli.py and warn there, once per pool. ACC-GEOM-64: the `all_but_the_top` docstring and the anisotropy section of docs/solution/wmd-docdistance-solution-sota.md state both measured readings: on symmetric panels centering carries the gain (E12-H64 103.7% on the ibm panel, E14-H75 102.8% on the Wergeland pool, projection isolate -4.5% and -3.6%); on the source-conditioned path with the ibm fixture centering alone reads 0.5471x the shipped DR(D_sel) and 1/28 selection-axis violations (E14-H76), a difference traced to 22 one-word markup statements that the default reader produces from that source (with them removed centering alone reads 0.9165x at 0/28). Do not write that the projection removes source content, and do not state the conditioned-path reading as general: it is one source and one reader setting. `all_but_the_top` itself does not change. Readings: reports/E14-H75-centering-attribution-transfer-metrics.json, reports/E14-H76-conditioned-centering-probe-metrics.json."
  },
  {
   "key": "batch-memos",
   "title": "embed memo, grounding-grid memo and --pairs on distance-wrt-source",
   "items": [
    "ACC-CLI-29",
    "ACC-CLI-30",
    "ACC-CLI-31",
    "ACC-CLI-32",
    "ACC-CLI-33",
    "ACC-CLI-34",
    "ACC-CLI-35",
    "ACC-CLI-36",
    "ACC-CLI-37",
    "ACC-CLI-38",
    "ACC-CLI-39",
    "ACC-CLI-40",
    "ACC-CLI-41",
    "ACC-CLI-42",
    "ACC-CLI-43"
   ],
   "files": [
    "src/docdistance/cli.py",
    "src/docdistance/pipeline.py",
    "tests/test_cli.py",
    "tests/test_pipeline.py",
    "tests/test_e2e.py"
   ],
   "grounded": true,
   "brief": "Build what E12-H67 and E12-H68 prototyped in their notebooks (notebooks/experiments/E12-H67-kj-pairs-embed-memo.ipynb, notebooks/experiments/E12-H68-kj-grounding-grid-memo.ipynb; readings in reports/E12-H67-*-metrics.json and reports/E12-H68-*-metrics.json). The byte-identity criteria (ACC-CLI-30, ACC-CLI-39) are CRITICAL: memo on and memo off produce the same bytes. The earlier packages added warnings (marker statements, pool below the minimum): a memo hit must not change which warnings a batch logs per document in a way a criterion forbids - read ACC-CLI-36 for the dropped-fragment count rule. The speed criteria (ACC-CLI-31, ACC-CLI-40) name a 1-minute load below 8 as the condition of the reading. On 2026-10-09 the load figure of this host stayed between 8 and 38 while the CPU was about 86% idle: file scans of another session keep threads in uninterruptible wait, which the load average counts. Take both readings in any case, with interleaved off/on repeats, and record before every timed run `os.getloadavg()[0]` and the `some avg10` value of `/proc/pressure/cpu` (the share of time a runnable task waited for a CPU); before the first timed run wait at most 10 minutes for the load to fall below 8. Report the ratio, the encode and grid counts (110 -> 11) and the load of every timed run in the evidence. When any timed run started at a load of 8 or above, return the item with met=false and the words 'load condition not met' in its evidence - the orchestrator repeats the reading; do not wait longer and do not edit the criterion. The wrt-source memo-off sweep takes about 45 minutes and ACC-CLI-39 needs its 55 lines anyway: run it detached (`setsid nohup ... > logs/<name>.log 2>&1 &`) and poll."
  },
  {
   "key": "calibration-notes",
   "title": "threshold provenance comments, calibration guidance and the documentation of what the build added",
   "items": [
    "ACC-CALIB-24",
    "ACC-CALIB-25",
    "ACC-CALIB-26",
    "ACC-CALIB-27",
    "ACC-CALIB-28",
    "ACC-MAP-63",
    "ACC-GEOM-46"
   ],
   "files": [
    "src/docdistance/distance.py",
    "docs/solution/wmd-structure-distance-sota.md",
    "docs/solution/wmd-docdistance-solution-sota.md",
    "docs/solution/wmd-source-conditioned-docdistance-solution-sota.md",
    "docs/cli-reference.md",
    "docs/api-reference.md",
    "docs/example-map-interpretations.md",
    ".claude/skills/docdistance/SKILL.md",
    "README.md",
    "CHANGELOG.md"
   ],
   "grounded": false,
   "brief": "Comments and documents only, no behaviour change. Read `git diff HEAD --stat -- src tests` and the earlier packages' results first: this package documents what they built. (1) ACC-CALIB-24 and ACC-CALIB-27: the two threshold comments in distance.py, with the numbers the criteria quote. (2) ACC-CALIB-25, ACC-CALIB-26, ACC-CALIB-28: the calibration sections of docs/solution/wmd-structure-distance-sota.md and docs/solution/wmd-docdistance-solution-sota.md. The deletion caveat of ACC-CALIB-28 now names the remedy that shipped: `matched_order_gap` reads an order-preserving deletion at the identity floor `opw_gap(B, B)` (E13-H70: 280/280 at or below 0.005 on documents of 11 to 14 statements), while `order_gap`, `structure_closeness` and the structural verdict keep their 1.1.5 values. (3) ACC-MAP-63: docs/api-reference.md, docs/cli-reference.md and docs/example-map-interpretations.md describe every field, function, constant and option the earlier packages added, with the measured readings from the ledger docs/experiments/wmd-docdistance-experiments.md (sections E13, E14, E15) - copy numbers from the ledger or the metrics files under reports/, never from memory. (4) ACC-GEOM-46: docs/solution/wmd-docdistance-solution-sota.md gains the E14-H77 reading per bandwidth (m = 0.25, 1, 4, 16, 64 times the inverse median squared chordal distance): not a copy of the linear pre-pass at any bandwidth (lowest cost Spearman over the 11 gate pools 0.9816, 0.9091, 0.5087, 0.0712, -0.0664), component stability per pool group, the three panel readings and the verdict Refuted - no kernel pre-pass ships. Source of every number: reports/E14-H77-kernel-pca-prepass-metrics.json. (5) Limits measured in E13 and E15 go into the Limitations sections of the two solution documents, one bullet each: the identity floor `opw_gap(P, P)` is not zero and grows with length (5.182e-03 at 30 statements, 6.786e-03 at 53, about 0.007 to 0.010 at 300); the order gap grows with the count of displaced statements and does not level off beyond the kernel reach (E13-H74); a capped ground cost was not built (E13-H72 killed at its gate); the nearest-source cost separates own from foreign statements on raw embeddings and not after the pre-pass (E13-H73: AUC 0.9670 against 0.7889). (6) docs/solution/wmd-source-conditioned-docdistance-solution-sota.md: document what transport-core added to the source-conditioned result and correct any sentence that reads `residual_a` / `closeness_a` as the unsupported share of a document (DEF-TRANS-8: a verbatim extract of 10 of 70 source statements reads residual 0.958). (7) README.md and .claude/skills/docdistance/SKILL.md: the new fields, the new options and the Python 3.12 to 3.14 range, in the terse form those files already use. (8) CHANGELOG.md: one `## [Unreleased]` section above `## [1.1.5]` with Added / Changed / Fixed bullets for what this build added; no version number and no date - the release is the user's step. Professional tone, literal language, no figures of speech, no em-dashes, unicode arrows. Do not restate a number in two documents where a link to the other document serves. (9) docs/solution/wmd-source-conditioned-docdistance-solution-sota.md quotes E06 readings that the metrics file does not hold. Correct them from the ledger (sections E06-H25, E06-H26, E06-H27 and `### Results (E06)`): trained ColBERT recall@3 0.4456 on the run of 2026-07-01 and 0.5408 on the re-run of 2026-10-09 with later model code, both below the 0.60 kill-gate; trained bge-m3 sparse recall@15 0.867 and `D_grd` Spearman +0.770 against the 0.95 bar - the ranking is not inverted; length-bucketing 42.5% (42.1% on the re-run) at score Spearman 0.99996 with a largest score difference of 0.0389, which is not 'numerically identical'."
  }
 ]
}
const packages = (args && args.packages) || BUILD_ARGS.packages
const allow = (args && args.allow) || BUILD_ARGS.allow
const baselineCmd = pkg => `CI=true .venv/bin/python reports/E13-campaign/baseline.py verify --allow ${allow}${pkg && pkg.grounded ? ' --grounded' : ''}`

const BUILD_SCHEMA = {
  type: 'object',
  required: ['package', 'status', 'items', 'files_changed', 'make_test', 'make_lint', 'baseline_verify'],
  properties: {
    package: { type: 'string' }, status: { enum: ['done', 'partial', 'blocked'] },
    items: { type: 'array', items: { type: 'object', required: ['id', 'met', 'evidence'], properties: { id: { type: 'string' }, met: { type: 'boolean' }, evidence: { type: 'string' } } } },
    files_changed: { type: 'array', items: { type: 'string' } },
    tests_added: { type: 'array', items: { type: 'string' } },
    make_test: { type: 'string' }, make_lint: { type: 'string' }, baseline_verify: { type: 'string' },
    out_of_scope_needs: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
}

const FINDING = { type: 'object', required: ['severity', 'claim', 'evidence'], properties: { severity: { enum: ['blocking', 'major', 'minor'] }, file: { type: 'string' }, line: { type: 'integer' }, claim: { type: 'string' }, evidence: { type: 'string' } } }

const VERIFY_SCHEMA = {
  type: 'object', required: ['package', 'verified', 'findings', 'gates'],
  properties: {
    package: { type: 'string' }, verified: { type: 'boolean' },
    findings: { type: 'array', items: FINDING },
    gates: { type: 'object', required: ['make_test', 'make_lint', 'baseline_verify'], properties: { make_test: { type: 'string' }, make_lint: { type: 'string' }, baseline_verify: { type: 'string' } } },
    items_not_met: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
}

const implPrompt = (pkg, done) => `You are the IMPLEMENTER for work package "${pkg.key}" (${pkg.title}) of the docdistance E13 campaign, stage B (build). Repo root: ${ROOT}. Work from the repo root, in the main working tree. Execution model: ${BUILD_MODEL}.

FIRST read ${PRIMER} in full - environment, hard rules, the stage B contract and return schema. THEN read every item of this package in docs/acc-crit-docdistance.md and docs/defects-docdistance.md (read only): ${pkg.items.join(', ')}. Each item's body is the assertion, its test: or repro: line is the check, its mechanism: or root-cause: line is the intended design.

Files this package owns: ${pkg.files.join(', ')}. An edit outside them is reported in out_of_scope_needs, not made.

Package brief: ${pkg.brief}

Packages already built in this tree, in order (their changes are present; build on them, do not redo them):
${done.length ? done.map(d => `- ${d.key}: ${d.result ? d.result.status : 'no result'} - files ${d.result ? (d.result.files_changed || []).join(', ') : 'n/a'}`).join('\n') : '- none'}

Procedure: read the code you will change and its callers before the first edit; write the failing test for each item, then the change; run \`make install\` once (new build to test - it increments the patch version, which is wanted); run \`make format\`, \`make lint\`, \`make test\`; run the guardrail \`${baselineCmd(pkg)}\` and read its last line. A difference the guardrail reports is a regression of a shipped number: fix the change, never the baseline. Return the stage B result JSON only, one entry in items per id listed above.`

const verifyPrompt = (pkg, res) => `Lens: qa-engineer. Target: work package "${pkg.key}" (${pkg.title}) of the docdistance E13 build, as it stands in the working tree of ${ROOT}. Find where the implementer's claim does not hold. You may run commands; you never edit, stage or revert a file.

Read ${PRIMER} (hard rules, stage B contract). Read the package items in docs/acc-crit-docdistance.md and docs/defects-docdistance.md: ${pkg.items.join(', ')}. Read the change: \`git diff HEAD -- ${pkg.files.join(' ')}\` and \`git status --short\`.

Implementer result:
${JSON.stringify(res, null, 2)}

Checks: (1) each item: the assertion holds in the code and a test exercises it - run the item's test: or repro: line where it is a command; an item claimed met without a test that would fail on the unchanged tree is MAJOR; (2) run \`make test\`, \`make lint\` and \`${baselineCmd(pkg)}\` yourself and report their last lines in gates - a red gate is BLOCKING; (3) additive policy: no shipped number, default or detail-file key changed except the registered moved-flag exception; (4) files outside the package list were not edited by this package; (5) a new quantity's docstring says whether it is a metric, a lower bound or a score, and nothing approximate is called exact; (6) syntax valid on Python 3.12 (\`uvx ruff@0.16.2 check --target-version py312\`); (7) no dead code, no speculative parameter, no fallback for an impossible state; (8) no commit, tag, hand edit of the version, journal or tracker write happened - the patch increments that \`make install\` makes are expected and are not a finding; you never run \`make install\` yourself. verified=true only with no blocking or major finding. Return the JSON only.`

const repairPrompt = (pkg, res, ver, done) => `${implPrompt(pkg, done)}

REPAIR ROUND. An adversarial verifier rejected the package. Your previous result:
${JSON.stringify(res, null, 2)}
Verifier findings (fix every blocking and major one; the items stay as written):
${JSON.stringify(ver, null, 2)}
Apply the fixes in place, run \`make install\` once (new build to test), re-run the three gates and return the updated result JSON only.`

const built = []
phase('Build')
if (!packages.length) { log('no packages in args - nothing to build'); return { packages: [] } }
log(`stage B: ${packages.length} packages in order on ${BUILD_MODEL}: ${packages.map(p => p.key).join(' -> ')}`)
for (const pkg of packages) {
  let res = await agent(implPrompt(pkg, built), { label: `build:${pkg.key}`, phase: 'Build', schema: BUILD_SCHEMA, model: BUILD_MODEL, agentType: BUILD_AGENT })
  let ver = res ? await agent(verifyPrompt(pkg, res), { label: `verify:${pkg.key}`, phase: 'Verify', schema: VERIFY_SCHEMA, effort: 'high', agentType: REVIEW_AGENT }) : null
  let repaired = false
  if (res && ver && !ver.verified) {
    log(`${pkg.key}: verifier rejected (${ver.findings.length} findings) - repair round`)
    const res2 = await agent(repairPrompt(pkg, res, ver, built), { label: `repair:${pkg.key}`, phase: 'Build', schema: BUILD_SCHEMA, model: BUILD_MODEL, agentType: BUILD_AGENT })
    if (res2) {
      res = res2; repaired = true
      ver = await agent(verifyPrompt(pkg, res), { label: `reverify:${pkg.key}`, phase: 'Verify', schema: VERIFY_SCHEMA, effort: 'high', agentType: REVIEW_AGENT })
    }
  }
  log(`${pkg.key}: ${res ? res.status : 'no result'} - verified=${ver ? ver.verified : 'n/a'}`)
  built.push({ key: pkg.key, result: res, verify: ver, repaired })
  if (!res || res.status === 'blocked') { log(`${pkg.key}: blocked - later packages still run, the orchestrator decides`) }
}

phase('Gate')
const final = await agent(`Lens: qa-engineer. Final gate of the docdistance E13 build in ${ROOT}. You run commands and read; you never edit. Run and report the last line of each: \`make lint\`; \`make test\`; \`${baselineCmd({ grounded: true })}\`; \`uvx ruff@0.16.2 check --target-version py312\`; \`git status --short\`. Then confirm against ${PRIMER} that no commit or tag happened (\`git log -1 --oneline\`, \`git tag --points-at HEAD\`) and report the version line (\`grep -n '^version' pyproject.toml\`) in notes - patch increments made by \`make install\` are expected. verified=true only when every gate is green. Return the JSON only, package "final-gate".`, { label: 'verify:final-gate', phase: 'Gate', schema: VERIFY_SCHEMA, agentType: REVIEW_AGENT })
return { packages: built, final }
