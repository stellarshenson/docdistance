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

## Field documentation accuracy `DOCS`

Correctness of the external field-use insights report against its archived forensics data

- [x] `DEF-DOCS-7` **insights small-edit counts differ from the forensics archive** - MINOR; the small-edit table in the external docdistance-insights.md understates its own archive: corrections.json holds 48 article removals not 41, 21 classes not 18, 8 second-person not 9, 6 fee-to-cost not 8; the load-bearing conclusions replicate (changed false on 0 of 48 article removals, cost median 0.096 in the stated band, moved on exactly 1,077 of 1,086), so a count inaccuracy only; regenerate any quoted count from the archived corrections.json before using it in calibration work
  - evidence: recount: 21 classes/510 pairs, 48 article removals 0/48 changed, 8 second-person 8/8, fee->cost 2/6, team->crew 19/96 re-derived from sem maps; patched doc matches archive
  - repro: recount scripts-and-logs/corrections.json in docdistance-insights_forensics.zip against the insights small-edit table
  - test-tags: MANUAL
  - log: 2026-08-31T12:30:17Z @kj added
  - log: 2026-08-31T13:02:56Z @kj closed: fixed: insights small-edit counts regenerated from corrections.json; team->crew changed cell 0/96 -> 19/96 patched by orchestrator after verifier flagged it

