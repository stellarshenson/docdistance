# Defects - docdistance

Defects observed in the docdistance library and CLI, tracked with pm-tools; every event is logged, nothing is rewritten.

## Authors

- `@kj` Konrad Jelen

## Transport and flag semantics `TRANS`

The OT plan, its marginals, and the changed/moved flags derived from it

- [x] `DEF-TRANS-1` **removed content absorbed by unrelated targets, nothing marks it** - MAJOR; semantic map narrates false merges: a source statement with no counterpart still ships its full 1/n mass to the cheapest unclaimed target at high weight (field job: weight 0.95, cost 0.76 onto an unrelated legend line); cause balanced uniform-marginal OT with no representation for removal, `distance.py:48-53`, `pipeline.py:151-157`; fix stage 1 annotate no_counterpart from cheapest-match cost above a stated threshold, stage 2 opt-in sink for the detail map only, reported smd stays the exact balanced distance and any sinked plan is labelled non-metric
  - evidence: synthetic removed statement flagged no_counterpart at min cost 1.414>0.6, no_counterpart_cost=0.6 in header, smd unchanged vs hand-computed transport; make test 94 passed 1 skipped
  - repro: semantic map of a pair where A carries content B removed; the removed statement matches an unrelated target at weight near 1, cost above 0.6, no marker
  - test-tags: UNIT
  - log: 2026-08-31T12:30:00Z @kj added
  - log: 2026-08-31T13:02:56Z @kj closed: fixed: no_counterpart flag from row-min cost, threshold in header; sink stage deferred
- [x] `DEF-TRANS-2` **moved flag is zero-tolerance and saturates on length change** - MAJOR; moved fired on 1,077 of 1,086 statements when the target document is a third shorter; cause moved = bool(displacement != 0) over raw statement-index rank shift whose scale is the document length, `pipeline.py:200-201`, definition stated nowhere in the emitted file; fix report length-normalised displacement, define moved against a tolerance band, null displacement for no-counterpart statements, emit displacement_definition in the header
  - evidence: 12->8 shrinkage: moved quiet on all 8 kept (|norm|<=1/24), 4 removed get null/false/no_counterpart true, reversal control fires 12/12; make test 94 passed 1 skipped
  - related: DEF-TRANS-1 - scattered absorber targets are what perturb the displacement globally
  - repro: distance-structural on a 1,086-statement doc vs its 730-statement revision; moved true on 1,077 of 1,086 (archived str-v1-v4.json)
  - test-tags: UNIT
  - log: 2026-08-31T12:30:00Z @kj added
  - log: 2026-08-31T13:02:56Z @kj closed: fixed: displacement_norm target-relative, moved tolerance 0.05, absorber records nulled, definitions in header
- [x] `DEF-TRANS-3` **changed flag tests the heaviest match, not the cheapest** - MEDIUM; statements whose true counterpart carries minority transport weight are flagged changed although a near-zero-cost match exists (19 flows in the archived sem-v1-v4.json, one with min cost 0.0003 flagged at 0.868); cause changed reads matches[0] which is weight-descending, `pipeline.py:125` and `pipeline.py:154-157`, and weight follows marginal constraints not cost; every archived consumer bypasses the flag and reads the cheapest match; fix define changed against the row minimum cost or emit both best_weight_cost and min_cost
  - evidence: 3x2 synthetic: heaviest match cost 1.194 (old defn flagged) with row min 0.29 -> changed false; details smd matches hand LP exactly; make test 94 passed 1 skipped
  - related: ACC-TRANS-10 - the exposure that makes the fixed flag queryable
  - repro: build a flow whose mass splits between a cost 0.0003 target (minority weight) and a cost 0.868 target (majority); changed comes back true
  - test-tags: UNIT
  - log: 2026-08-31T12:30:00Z @kj added
  - log: 2026-08-31T13:02:56Z @kj closed: fixed: changed reads the full cost-row minimum, not matches[0]
- [x] `DEF-TRANS-8` **source residual counts transport to unrelated source statements** - MAJOR; 10 statements copied verbatim from a 70-statement source score residual_a 0.958, closeness_a 0.322; cause balanced transport forces 1 - n/m of each statement's mass off its nearest source statement; fix one-sided transport, nearest source statement only; `src/docdistance/distance.py:304`
  - evidence: tests/test_distance.py::test_verbatim_extract_reads_no_unsupported_statement: 10 statements copied from a 70-statement source read n_unsupported_a 0 while residual_a stays above 0.5; docstrings no longer call the residual the unsupported share (1.1.13: 269 passed)
  - related: ACC-TRANS-67, ACC-TRANS-68, ACC-TRANS-72 - the unsupported-statement count and the corrected docstrings (E15-H78 Promoted)
  - repro: embed source-article.md (ibm fixture), X = source[0::7][:10], compute_source_conditioned(X, X, source): residual_a 0.958, nearest-source mean cost 0.0001
  - test-tags: UNIT
  - root-cause: 2026-10-07T10:53:01Z @kj ungrounded_residual calls smd(X, S) with uniform marginals on both sides: each source statement accepts 1/m, so a statement keeps at most n/m of its mass on its own copy
  - log: 2026-10-07T10:53:01Z @kj added
  - log: 2026-10-10T00:34:27Z @kj closed
- [x] `DEF-TRANS-9` **exact solver stops at its iteration cap and the result is used** - MEDIUM; smd on 2500 or more statements per side returns a non-optimal value with only the POT warning `numItermax reached before optimality` (excess 0.03% at 3000); cause default 100000-iteration cap, result code unread at 5 call sites; fix scale the cap with size, raise on the warning; `src/docdistance/distance.py:58`
  - evidence: tests/test_distance.py::test_exact_solver_cap_scales_with_the_problem_size (3000 x 3000 solves without the warning, 0.661777 equal to POT at numItermax 20000000), ::test_exact_solver_raises_when_it_stops_at_the_iteration_cap, ::test_exact_solver_matches_the_bare_pot_call (1.1.13: 269 passed)
  - root-cause: 2026-10-09T23:17:36Z @kj POT stops at its fixed numItermax of 100000; 3000 x 3000 problems with non-negative uniform entries, shared-direction vectors or float64 normal vectors need about 138000 iterations and reach it, while isotropic float32 normal unit vectors need about 98000 and do not, so the repro depends on the generator
  - repro: two sets of 3000 random unit vectors, dim 768: d.smd(A, B) = 0.882362 with the warning; ot.emd2 with numItermax=20000000 = 0.882068
  - test-tags: UNIT
  - log: 2026-10-07T10:53:01Z @kj added
  - log: 2026-10-10T00:34:26Z @kj closed
- [x] `DEF-TRANS-10` **order gap reports a reorder after a pure deletion** - MAJOR; deleting 2 of 10 statements with the order kept gives order_gap 0.227, structure_closeness 0.839, structural verdict not similar; likely cause: on unequal lengths the order-preserving plan must spread surplus mass off the diagonal; fix compute the order gap over matched statement pairs only, pending E13; `src/docdistance/distance.py:206`
  - evidence: tests/test_distance.py::test_compute_structural_reads_a_pure_deletion_on_the_matched_gap and tests/test_structure.py::test_matched_order_gap_ignores_a_pure_deletion: matched_order_gap reads the identity floor where order_gap reads the surplus; opw_gap docstring states both limits (1.1.13: 269 passed)
  - related: ACC-CALIB-28 - the documented caveat for this behaviour
  - related: DEF-TRANS-8 - same cause, unequal-length marginals
  - related: ACC-TRANS-49, ACC-TRANS-54 - the matched order gap
  - repro: ibm gold summary A (10 statements), B = A minus 2 statements (rng seed 0): opw_gap(A, B) = 0.227; E12-H62 read 280 of 280 deletions above its bar
  - test-tags: UNIT
  - log: 2026-10-07T10:53:01Z @kj added
  - log: 2026-10-10T00:34:26Z @kj closed
- [x] `DEF-TRANS-13` **moved flag on a removed statement is set by a float tie** - MEDIUM; 12 to 8 order-preserving deletion: each removed statement's exact-plan row splits into two equal parts of 1/24; argmax picks one by float rounding; moved fires on 1 of 4 removed statements with POT 0.9.7.post1, 2 of 4 with 0.9.6.post1; fix: read moved on matched pairs only, pending E15-H80; src/docdistance/pipeline.py:553
  - evidence: tests/test_structure.py::test_moved_quiet_on_an_order_preserving_deletion: a 12 to 8 deletion flags 0 of 12, kept and removed alike, n_moved 0; the flag is read on matched pairs, no argmax of a float tie (1.1.13: 269 passed)
  - related: DEF-TRANS-10, ACC-MAP-60 - same cause; matched_displacement is the reading on matched pairs
  - related: ACC-TRANS-77, ACC-TRANS-78 - the fix E15-H80 measured
  - repro: E12-H61 shrink_text fixture: d._aligned_plan(X, X[kept]) rows 2, 5, 8, 11 each hold two entries of 0.04167; reports/E12-H61 metrics removed_moved_conjunction 2, at commit bd0002e 1
  - test-tags: UNIT
  - root-cause: 2026-10-09T10:42:46Z @kj uniform marginals on unequal lengths leave 1/24 spare capacity per target, so a removed statement splits evenly over two targets and order_alignment takes the argmax of an exact tie
  - log: 2026-10-09T10:42:46Z @kj added
  - log: 2026-10-10T00:34:27Z @kj closed

## CLI and config operation `CLI`

Command-line behaviour, config discovery, error output and batch execution

- [x] `DEF-CLI-4` **config miss emits empty stdout in --json mode** - MEDIUM; a --json distance command with no docdistance.json prints nothing to stdout, only a rich stderr message before exit 1, so a pipeline using 2>/dev/null reads empty stdout and fails downstream with JSONDecodeError (field job lost ten runs before diagnosis); cause _run prints to the stderr console and raises typer.Exit(1) before any stdout writer runs, `cli.py:64-75`; fix emit a machine-readable error object to stdout when --json or --result-only is active, stderr and exit code unchanged
  - evidence: empty dir, distance-semantic --json 2>/dev/null: exit 1, stdout parses as {error: not_initialized, config searched: path}; human mode stdout stays empty; 3 CliRunner tests green
  - related: ACC-CLI-13 - the diagnosability half of the same config miss
  - repro: cd to a folder without docdistance.json; docdistance distance-semantic a.md b.md --json 2>/dev/null; stdout empty, exit 1
  - test-tags: FUNCTIONAL
  - log: 2026-08-31T12:30:17Z @kj added
  - log: 2026-08-31T13:02:56Z @kj closed: fixed: _run(machine=) emits one-line JSON error object to stdout in --json/--result-only
  - log: 2026-08-31T16:17:49Z @kj adversarial review (4 lenses, 5 rounds, SHIP) extended the machine-error funnel this fix built: OSError->io_error arm in _run, detail/source-map writes and both --pairs read paths incl. malformed lines now emit the stdout JSON object; 3 pinning tests, suite 143 passed
- [x] `DEF-CLI-5` **nonexistent file path silently scored as raw text** - MEDIUM; a mistyped document path is treated as literal text and scored, exit 0, producing a plausible but meaningless distance between the filename string and the other document; cause _read falls back to doc.strip() when the path does not resolve, with no diagnostic, `pipeline.py:45-57`; fix warn to stderr when an argument that looks like a path does not resolve, or add explicit --text/--file flags
  - evidence: _read('v1-bdoy.md') warns once, MISSING.MD warns, prose and real files silent, return value unchanged; regression test green in make test 94 passed
  - repro: docdistance distance-semantic v1-bdoy.md v2-body.md (typo in the first path); exit 0 with a distance for the filename string
  - test-tags: UNIT
  - log: 2026-08-31T12:30:17Z @kj added
  - log: 2026-08-31T13:02:56Z @kj closed: fixed: loguru warning when a non-resolving path-like str is scored as raw text

## Segmentation and input handling `SEGM`

How document bodies are read, stripped and split into statements before embedding

- [x] `DEF-SEGM-6` **H1 stripping deletes fenced-code comment lines** - MEDIUM; every line starting with hash-space is dropped from file inputs with no fence awareness, so a bash or python comment inside a fenced code sample, or any second H1, silently vanishes from the compared body; two documents differing only in such lines compare as identical; cause the startswith filter over all lines in _load_body, `pipeline.py:41-42`, documented intent is title stripping only; fix track fence state and strip hash-space lines only outside fences, or strip only the first H1
  - evidence: fenced '# comment' lines survive verbatim incl. indented ~~~ fences, title H1 still stripped; regression test green in make test 94 passed
  - repro: file with a fenced bash block holding a hash-space comment line; _load_body output lacks the line
  - test-tags: UNIT
  - log: 2026-08-31T12:30:17Z @kj added
  - log: 2026-08-31T13:02:56Z @kj closed: fixed: _load_body tracks fence state, hash-space strip applies only outside fences
- [x] `DEF-SEGM-12` **default reader keeps markdown markers as statements, no warning** - MAJOR; ibm source read with markdown off: 70 statements, 21 are list or heading markers ('-' 13, '###' 6, '##' 2); they repeat, break one-to-one matching and carry 68% of the centered pool variance; fix: one warning naming the count and --markdown / --min-words, defaults unchanged
  - evidence: tests/test_segmentation.py::test_default_reader_warns_about_statements_without_a_letter_or_digit; ibm source with markdown off: one warning '21 of 70 statements hold no letter and no digit', statements and embeddings unchanged; markdown on: 48 statements, no warning (1.1.13: 269 passed)
  - related: ACC-SEGM-1, ACC-SEGM-2 - the shipped options that remove the markers
  - repro: DocDistance().embed_statements(ibm source-article.md): 21 of 70 statements hold no letter or digit; with markdown=True: 0 of 48
  - test-tags: UNIT
  - root-cause: 2026-10-09T10:27:02Z @kj the default splitter is the prose segmenter; it has no markdown rule, so a bare list or heading marker comes back as its own statement
  - log: 2026-10-09T10:27:02Z @kj added
  - log: 2026-10-10T00:34:29Z @kj closed
- [x] `DEF-SEGM-15` **markdown reader keeps bare marker lines as statements** - MEDIUM; split_markdown returns a line that holds only a marker ('-', '###', '---', '* * *') as a prose statement, although its docstring says marker-only lines are dropped; the DEF-SEGM-12 warning names --markdown as the remedy and is silent in that mode
  - evidence: tests/test_segmentation.py::test_split_markdown_drops_marker_only_lines and ::test_split_markdown_keeps_lines_that_are_not_marker_only: '-', '###', '---', '* * *' are dropped, a bare numbered marker such as '42.' stays prose by design; ibm source markdown on still 48 statements; --markdown sweep sha256 unchanged (62d98b6b3f) (1.1.13: 269 passed)
  - related: DEF-SEGM-12 - the warning that names this reader as the remedy
  - repro: split_markdown on the lines '## Key findings', '-', a prose paragraph, '###', '---', '* * *': the four marker lines come back as statements
  - test-tags: UNIT
  - root-cause: 2026-10-09T23:17:36Z @kj the heading and list patterns require whitespace after the marker and the line is stripped first, so a bare marker matches neither and falls through to prose; horizontal rules have no rule
  - log: 2026-10-09T23:17:36Z @kj added
  - log: 2026-10-10T00:34:31Z @kj closed

## Field documentation accuracy `DOCS`

Correctness of the external field-use insights report against its archived forensics data

- [x] `DEF-DOCS-7` **insights small-edit counts differ from the forensics archive** - MINOR; the small-edit table in the external docdistance-insights.md understates its own archive: corrections.json holds 48 article removals not 41, 21 classes not 18, 8 second-person not 9, 6 fee-to-cost not 8; the load-bearing conclusions replicate (changed false on 0 of 48 article removals, cost median 0.096 in the stated band, moved on exactly 1,077 of 1,086), so a count inaccuracy only; regenerate any quoted count from the archived corrections.json before using it in calibration work
  - evidence: recount: 21 classes/510 pairs, 48 article removals 0/48 changed, 8 second-person 8/8, fee->cost 2/6, team->crew 19/96 re-derived from sem maps; patched doc matches archive
  - repro: recount scripts-and-logs/corrections.json in docdistance-insights_forensics.zip against the insights small-edit table
  - test-tags: MANUAL
  - log: 2026-08-31T12:30:17Z @kj added
  - log: 2026-08-31T13:02:56Z @kj closed: fixed: insights small-edit counts regenerated from corrections.json; team->crew changed cell 0/96 -> 19/96 patched by orchestrator after verifier flagged it
- [x] `DEF-DOCS-18` **Structural calibration quoted for statements the command does not read** - MAJOR; calibration of STRUCTURE_THRESHOLD 0.90 (lowest un-reordered ibm pair 0.906144, 0 of 72 flagged) is read on fixture statements that keep the title line as statement 0; distance-structural removes title lines and reads 3 of 55 un-reordered ibm pairs below 0.90 on 1.1.5, 4 of 55 with the default reader
  - evidence: docs/solution/wmd-structure-distance-sota.md bullet 'Readings of this document and the reader of a file' names the fixture statements and the command readings (3 of 55 plain, 4 default, lowest 0.897057); the two other solution documents carry the same heading and name the fixture statements as the output of no reader; 0.90 unchanged
  - repro: distance-structural on exec-summary gold-2 against gold, haiku and 2 reads structure_closeness 0.8971 to 0.8974, not similar (same in reports/E13-campaign/baseline/default-path.json.gz), while docs/solution/wmd-structure-distance-sota.md does not name the fixture statements and these readings beside the calibration
  - test-tags: MANUAL, FUNCTIONAL
  - root-cause: 2026-10-12T01:50:51Z @kj two differences between calibration and command: fixture statements keep the title line that pipeline._load_body removes, and E12-H59 encoded with the torch backend where the command uses OpenVINO INT8; on stored statements without the title the backend alone moves gold|gold-2 from 0.900150 to 0.897407
  - root-cause: 2026-10-12T01:28:24Z @kj notebook 11 segments the fixture per blank-line paragraph and keeps the title line; pipeline._load_body removes title lines; E12-H59 calibrated on the fixture statements and the solution document did not name the difference
  - log: 2026-10-12T01:28:24Z @kj added
  - log: 2026-10-12T01:42:00Z @kj three solution documents now name the fixture statements (title line kept, no reader) against the command readings: 3 of 55 ibm pairs below 0.90 plain, 4 default, lowest 0.897057; recalibration on read statements listed in ledger next steps, user decision
  - log: 2026-10-12T01:42:00Z @kj closed
  - log: 2026-10-12T01:50:56Z @kj edited repro "distance-structural on exec-summary gold-2 against gold, haiku and 2 (ibm summaries): structure_closeness 0.8971 to 0.8974, verdict not similar; same values in reports/E13-campaign/baseline/default-path.json.gz" -> "distance-structural on exec-summary gold-2 against gold, haiku and 2 reads structure_closeness 0.8971 to 0.8974, not similar (same in reports/E13-campaign/baseline/default-path.json.gz), while docs/solution/wmd-structure-distance-sota.md does not name the fixture statements and these readings beside the calibration"
  - log: 2026-10-12T01:52:51Z @kj edited evidence "docs/solution/wmd-structure-distance-sota.md bullet 'Readings of this document and the reader of a file' names the fixture statements and the command readings (3 of 55 plain, 4 default, lowest 0.897057); same bullet in the two other solution documents; 0.90 unchanged" -> "docs/solution/wmd-structure-distance-sota.md bullet 'Readings of this document and the reader of a file' names the fixture statements and the command readings (3 of 55 plain, 4 default, lowest 0.897057); the two other solution documents carry the same heading and name the fixture statements as the output of no reader; 0.90 unchanged"
  - log: 2026-10-12T02:02:09Z @kj calibration section of docs/solution/wmd-structure-distance-sota.md now names both differences (title line, torch against OpenVINO INT8) and the command readings; confirming review wf_17294010-49e round 1, build 1.1.31

## Dependency behaviour at runtime `DEPS`

What third-party packages do at import or run time inside a docdistance call

- [x] `DEF-DEPS-11` **script without a main guard runs twice on Python 3.14** - MAJOR; on Python 3.14 a script calling DocDistance().embed() runs twice, the second time in a child process, and each model load sends a usage event to Google Analytics; cause import openvino starts telemetry through multiprocessing.Process, which re-imports the main module under forkserver; fix block the telemetry before the import; `src/docdistance/encoders.py:232`
  - evidence: tests/test_e2e.py::test_loading_the_openvino_encoder_starts_no_child_process: a script without a main guard and with CI unset prints its marker once and starts no child process; it failed on the 1.1.5 tree (1.1.13: 269 passed)
  - repro: Python 3.14, Linux: script with a top-level print, then DocDistance().embed(file), no main guard: the print appears twice (pids differ, second __name__ is __mp_main__)
  - test-tags: INTEGRATION
  - root-cause: 2026-10-07T10:53:01Z @kj openvino/__init__.py imports openvino.tools.ovc, whose init_ovc_telemetry sends a GA4 event through openvino_telemetry/backend/backend_ga4.py:86 multiprocessing.Process; consent defaults to accepted when no opt-out file exists
  - log: 2026-10-07T10:53:01Z @kj added
  - log: 2026-10-10T00:34:26Z @kj closed
- [x] `DEF-DEPS-14` **offline flag is set after the hub library is imported** - MAJOR; the model constructors set HF_HUB_OFFLINE=1 after _require_models_extra() has imported transformers and huggingface_hub, which read the flag at import; without network a load of cached models raises ModelsNotInstalled, with network every load contacts the Hub
  - evidence: tests/test_e2e.py::test_offline_load_of_cached_models_works_with_the_hub_unreachable (no flag in the environment, proxies on a closed port: embed returns (2, 768); raised ModelsNotInstalled on 1.1.12), ::test_every_model_constructor_reads_the_hub_inside_the_offline_scope, ::test_hub_offline_scopes_that_overlap_on_two_threads_keep_the_flag_until_the_last_ends pass (1.1.15: 272 passed)
  - repro: env without HF_HUB_OFFLINE, http_proxy and https_proxy on a closed local port, models cached: DocDistance().embed(text) raises ModelsNotInstalled; with HF_HUB_OFFLINE=1 preset it returns shape (2, 768)
  - test-tags: E2E
  - root-cause: 2026-10-09T23:17:36Z @kj huggingface_hub.constants.HF_HUB_OFFLINE is evaluated once at import; os.environ.setdefault after the import does not change it, and transformers 5.0.0 calls model_info during the tokenizer load
  - log: 2026-10-09T23:17:36Z @kj added
  - log: 2026-10-10T01:01:20Z @kj closed
- [x] `DEF-DEPS-16` **init stays offline when HF_HUB_OFFLINE is preset** - MEDIUM; docdistance init removes HF_HUB_OFFLINE from the environment after _require_models_extra() has imported the Hub library, so with HF_HUB_OFFLINE=1 preset the library stays offline and init cannot download
  - evidence: tests/test_bootstrap.py::test_init_holds_the_hub_library_online_and_puts_the_earlier_mode_back: with the flag preset to offline the Hub library reads online during the init fetch and its earlier mode afterwards; it read offline during the fetch on 1.1.13 (1.1.15: 272 passed)
  - related: DEF-DEPS-14 - same cause, opposite direction
  - repro: HF_HUB_OFFLINE=1 preset: after the pop in bootstrap.py, huggingface_hub.constants.is_offline_mode() is still True
  - test-tags: UNIT
  - root-cause: 2026-10-10T00:34:26Z @kj the Hub library reads the variable once at import; removing it from os.environ afterwards changes nothing - the cause of DEF-DEPS-14 in the opposite direction
  - log: 2026-10-10T00:34:26Z @kj added
  - log: 2026-10-10T01:01:20Z @kj closed
- [x] `DEF-DEPS-17` **Segmenter fails after init --source** - MAJOR; regression against 1.1.5; host provisioned by init --source with no facebookAI/xlm-roberta-base in the HuggingFace cache raises ModelsNotInstalled at the first Segmenter load; 1.1.5 fetched the tokenizer from the Hub at that load
  - evidence: tests/test_e2e.py::test_segmenter_loads_after_init_from_a_source_folder_without_the_tokenizer passes with the real models and fails without the two source edits; 4 init tests in tests/test_bootstrap.py; tests/test_cli.py::test_init_prints_the_message_of_a_missing_model_and_exits_1 (1.1.21: 496 passed)
  - related: DEF-DEPS-14
  - repro: sat source folder with config.json and weights only, empty HF_HOME, proxies on a closed port: init --source, then Segmenter().split(text)
  - test-tags: UNIT, E2E
  - root-cause: 2026-10-10T02:51:43Z @kj wtpsplit loads the SaT tokenizer by Hub name; _mirror_one returns on a source hit before _warm_hf; the DEF-DEPS-14 offline scope then ends the load
  - log: 2026-10-10T02:51:43Z @kj added
  - log: 2026-10-10T06:01:28Z @kj closed
  - log: 2026-10-10T08:04:38Z @kj regressed as DEF-DEPS-17-1
- [x] `DEF-DEPS-17-1` **init accepts tokenizer files that do not load** - MAJOR; a sat-3l-sm source folder with tokenizer.json and tokenizer_config.json as the Hub publishes them passes init --source; the first Segmenter.split raises TypeError and distance-semantic --json exits 1 with a stack trace and an empty stdout
  - evidence: 1.1.26: test_segmenter_loads_after_init_from_a_source_folder_with_the_tokenizer_files_of_the_hub passes on the real models and fails on the 1.1.24 package with the TypeError; 5 stub cases in test_bootstrap.py pass; verifier repeated it with a truncated tokenizer.json and with no Hub
  - test-tags: UNIT, E2E
  - repro: source folder sat-3l-sm with config.json, weights, and tokenizer.json plus tokenizer_config.json copied from the cache snapshot of facebookAI/xlm-roberta-base; empty HF_HOME: init --source, then distance-semantic on two texts
  - root-cause: 2026-10-10T08:04:42Z @kj the tokenizer_config.json of the Hub repository holds no tokenizer_class and the config.json in the folder is the SaT model's, so AutoTokenizer returns a tokenizer whose cls, sep and pad ids are None; _add_sat_tokenizer and Segmenter test only that tokenizer.json exists
  - log: 2026-10-10T08:04:38Z @kj regression of DEF-DEPS-17: regression found by the final review (wf_73bb512c-5af round 3): a sat-3l-sm source folder that holds tokenizer.json and tokenizer_config.json as the Hub publishes them passes init, then the first Segmenter.split raises TypeError (special token ids None); init checked only that tokenizer.json exists
  - log: 2026-10-10T08:04:50Z @kj amended title "Segmenter fails after init --source" -> "init accepts tokenizer files that do not load"; text "regression against 1.1.5; host provisioned by init --source with no facebookAI/xlm-roberta-base in the HuggingFace cache raises ModelsNotInstalled at the first Segmenter load; 1.1.5 fetched the tokenizer from the Hub at that load" -> "a sat-3l-sm source folder with tokenizer.json and tokenizer_config.json as the Hub publishes them passes init --source; the first Segmenter.split raises TypeError and distance-semantic --json exits 1 with a stack trace and an empty stdout"
  - log: 2026-10-10T08:04:50Z @kj edited repro added "source folder sat-3l-sm with config.json, weights, and tokenizer.json plus tokenizer_config.json copied from the cache snapshot of facebookAI/xlm-roberta-base; empty HF_HOME: init --source, then distance-semantic on two texts"; test-tags added "UNIT, E2E"
  - log: 2026-10-11T23:53:21Z @kj closed
  - log: 2026-10-12T01:15:15Z @kj limit, review wf_2fef834e-695: the check runs in init only; a home that the unreleased builds 1.1.20 to 1.1.24 provisioned from Hub-published tokenizer files raises TypeError at the first split until init --source runs again; no released version writes such a home
  - log: 2026-10-12T01:15:15Z @kj limit, review wf_2fef834e-695: init does not load the tokenizer it saved; a HuggingFace cache with the tokenizer files of facebookAI/xlm-roberta-base and no config.json saves ids None (read on 1.1.28); no such cache is known; a check after the save waits for the user's word

