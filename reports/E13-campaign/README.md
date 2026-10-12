# E13 campaign - transport-overflow restrictions

Prepared 2026-10-07, launched 2026-10-09. The goal for the campaign is the project done-state: every acceptance criterion closed, no open defect, no unrun hypothesis, experiment notebooks re-executed with 0 errors, adversarial review survived.

## State

- Stage A (measure) is complete: twelve hypotheses with Result and Verdict in the ledger (E13-H70 to E15-H80 and E16-H81); full returns in `results/stage-a-run-<run id>.json`. The ledger holds a verdict for 50 of 50 hypotheses
- Stage B (build) is complete: seven packages and one follow-up (runs `wf_e1b16a4b-b94`, `wf_099e69d3-480`), the packages `review-followup` and `input-format` (run `wf_1d07273f-fc6`) with ten fixes from their verifiers (`input-format-fixes-brief.md`), and the package `extension-default` (run `wf_7940fe08-cb5`, brief `extension-default-brief.md`), verified on 1.1.26; returns in `results/stage-b-run-<run id>.json`. The orchestrator applied the three minor findings of its last verifier as build 1.1.27: `make test` 570 passed and 1 skipped, `make lint` clean, baseline 0 differences on 55 pairs and 3 grounded results
- Trackers: 0 criteria open (96 closed, 6 rejected as overridden), 0 defects open (19 closed). The override pairs and their reasons are in the tracker (`pm-tools list docs/acc-crit-docdistance.md --columns id,title,overrides`). Defects found during the build and the reviews and fixed: `DEF-DEPS-14`, `DEF-SEGM-15`, `DEF-DEPS-16`, `DEF-DEPS-17`, `DEF-DEPS-17-1`, `DEF-DOCS-18`
- Stage C (review): run `wf_366ba93d-451` from `workflow-review.run.js` reviewed the whole change in three rounds and was stopped by the orchestrator during round 4, because two packages changed the tree. Run `wf_73bb512c-5af` from `workflow-review.run2.js` reviewed the work since then and ended at its limit of three rounds with one major finding, now `DEF-DEPS-17-1`. Run `wf_2fef834e-695` from `workflow-review.run3.js` reviewed the `extension-default` build and ended at its limit of three rounds (fix builds 1.1.28, 1.1.29 and 1.1.30); its round 3 major, the reader claim of the three solution documents, is `DEF-DOCS-18`, closed by the text change. The confirming review of the delta since that round, run `wf_17294010-49e` from `workflow-review.run4.js` and `review-args-final3.json`, ended clean: round 1 planned one text change (the first sentence of the calibration section of the structure document; build 1.1.31), rounds 2 and 3 ruled clean with minors declined as immaterial (three lenses SHIP in both). Records: `results/stage-c-run-<run id>.json`
- Input format (the user's request and decisions of 2026-10-10): the option `input_format` with the values `auto`, `plain`, `markdown`, `html`. `auto` is the default and chooses the reader from the file extension (`.md`, `.markdown` → markdown; `.html`, `.htm` → HTML; every other file and raw text → plain); the content detector is removed. E16-H81 (reader chosen by content as a candidate default) was killed at its gate; its registration record is `results/E16-H81-registration.json`
- Notebooks: the 33 notebooks E01 to E15 are records on docdistance 1.1.5, the source of commit `bd0002e`, with 0 error outputs each; they were not executed on the campaign build, and the note 'Build of record' at the top of both ledgers gives the reasons. `results/reexec-drift-audit.md` compares the committed readings of E01 to E12 with the readings of their re-execution on 2026-10-09. E16-H81 carries the removed detector in its notebook and was re-executed on 1.1.26 with 0 error outputs; `results/E16-H81-metrics-recorded-1.1.21.json` is its metrics file as first recorded
- The load average of this host was not a measure of CPU contention on 2026-10-09 and 2026-10-10: hung file scans of another session kept threads in uninterruptible wait (load 31 to 55 at 97% idle CPU). Timing readings record `/proc/pressure/cpu` beside the load, and the two speed criteria read the quiet condition from it
- Decisions taken: executors and implementers on `opus` up to run `wf_2fef834e-695`, on the session model after it (Star Colonel, 2026-10-12); additive policy - with `input_format="plain"` every number 1.1.5 reports stays bit-identical except `moved` / `n_moved` and the statements of a `--markdown` read of a document with marker-only lines; the third registered exception is the default reader of a `.md`, `.markdown`, `.html` or `.htm` file
- Decisions left to the user: hard-wrapped markdown and plain text are cut at every line break; inline markup kept by the markdown reader; a byte order mark under the `plain` and `markdown` readers; the three verdict labels of the structure log that contradict their stored artefacts (E08-H44, E08-H47, E09-H53); the load condition of `ACC-CLI-81` and `ACC-CLI-82`; whether `STRUCTURE_THRESHOLD` 0.90 is calibrated again on statements as the command reads them (title line removed, OpenVINO INT8 encoder; `DEF-DOCS-18`, ledger next steps); whether the showcase notebook `notebooks/09-kj-docdistance-api-e2e.ipynb`, which passes `.md` files, is re-executed under the new default
- Environment: `.venv` on Python 3.14.8, package range 3.12 to 3.14. `make install` runs whenever a dependency is added or a new build is to be tested, and it increments the patch number each time (Star Colonel, 2026-10-09 and 2026-10-12)
- Released as docdistance 1.1.32 on 2026-10-12 (`make publish`, PyPI); the change since commit `bd0002e` is the release commit

## Files

- `PRIMER.md` - the agent contract for both stages
- `workflow-measure.js` - stage A: one executor per experiment notebook, verify, one repair round
- `workflow-build.js` - stage B: packages in order, verify, one repair round each, one final gate
- `workflow-build.run.js` - `workflow-build.js` with the content of `build-args.json` embedded, generated at launch; this is the script the run executes
- `workflow-review.js` - stage C: architect, bug-hunter and data-scientist lenses, adjudicated change plan and fix round per review round, repeated until a round is clean (at most 4 rounds)
- `reexec.sh`, `reexec-status.tsv` - re-execution runner for experiment notebooks and its status lines (name, exit code, error outputs, seconds): the E01 to E12 notebooks on 2026-10-09, E16-H81 on 2026-10-12
- `reexec-after-lane.sh` - waits for a running lane's `LANE-DONE` line, then runs `reexec.sh` on the notebooks given; used to re-run a notebook that must execute alone
- `build-args.json` - the stage B package list, six packages; `build-args-repairs.json` and `workflow-build.run2.js` - the seventh package
- `repairs-brief.md` - the confirmed verifier findings the seventh package and the follow-up applied, and the ones declined with the reason
- `checks/` - scripts that repeat the real-model readings quoted as evidence (see its `README.md`)
- `workflow-review.run.js` - `workflow-review.js` with the content of `review-args.json` embedded; this is the script the review run executes
- `review-args.json` - the stage C arguments: allowed keys, locked decisions, and what stage B already repaired
- `review-followup-brief.md`, `input-format-brief.md`, `input-format-fixes-brief.md` - the briefs of the two packages built after the first review run and of the fixes from their verifiers
- `build-args-followup.json`, `workflow-build.run3.js` - the package list of those two packages and the script their build run executed
- `workflow-measure.run-e16.js` - `workflow-measure.js` with the one entry for E16-H81; the script that run executed
- `review-args-final.json`, `workflow-review.run2.js` - the arguments and the script of the final review (the work since round 3 of the first review run, at most 3 rounds)
- `results/reexec-drift-audit.md` - committed readings against re-executed readings of the E04 to E12 notebooks, per hypothesis, with the corrected notebook sentences
- `results/E16-H81-registration.json` - block hash and time of the E16-H81 registration, taken before its lever was built
- `extension-default-brief.md`, `build-args-extension.json`, `workflow-build.run4.js` - the brief of the package that makes the extension rule the default reader, its package list and the script its build run executed
- `review-args-final2.json`, `workflow-review.run3.js` - the arguments and the script of the review of that build (at most 3 rounds)
- `review-args-final3.json`, `workflow-review.run4.js` - the arguments and the script of the confirming review of the delta since round 3 of run `wf_2fef834e-695` (run `wf_17294010-49e`, clean); two consecutive clean rounds end it (at most 3 rounds); the fix agent runs on the session model
- `notebook-reader-pin-brief.md` - the brief for naming the plain reader in the experiment notebooks; withdrawn except its section on E16-H81 (status line at its top)
- `results/E16-H81-metrics-recorded-1.1.21.json` - the E16-H81 metrics file as recorded on build 1.1.21, before the re-execution on 1.1.26; `checks/default-reader/diff.py` reads its `auto` arm
- `drift.py` - lists every number of a modified `reports/*.json` that differs from its committed value by more than a tolerance; with `--against FOLDER` it compares with the copies in that folder
- `baseline.py`, `baseline/default-path.json.gz` - the default-path guardrail captured at commit `bd0002e` source (55 ibm pairs, 3 grounded results)
- `results/H<n>.json` - written by stage A

## Launch procedure

1. Toolchain gate, then confirm the environment: `.venv/bin/python -c "import ot, docdistance"`, `make test` (143 passed, 1 skipped), `CI=true .venv/bin/python reports/E13-campaign/baseline.py verify --grounded` (0 differences). If `.venv` is missing: `make install` (it increments the version), then `.venv/bin/docdistance init wmd-wrt-source`
2. Stage A: `Workflow({scriptPath: "reports/E13-campaign/workflow-measure.js"})`. Record the run id in the journal at launch
3. After stage A: read each agent's return from the run's `journal.jsonl`; record Result, Verdict and a log line per hypothesis with `hypothesis-tools result / verdict / log-event ... --author @kj`; add the results table to the `## E13` section; show the finding table
4. File acceptance criteria for the Promoted levers with `pm-tools add docs/acc-crit-docdistance.md ... --author @kj`; replace the `FILL` entries in `build-args.json` with the new ids and the measured cutoff
5. Stage B: regenerate `workflow-build.run.js` from `workflow-build.js` and `build-args.json`, then `Workflow({scriptPath: "reports/E13-campaign/workflow-build.run.js"})`
6. After stage B: run the gates yourself (`make test`, `make lint`, baseline verify), close met criteria and fixed defects with evidence through `pm-tools close`, run `pm-tools check docs`, `hypothesis-tools check`
7. Stage C: `Workflow({scriptPath: "reports/E13-campaign/workflow-review.js", args: {allow, locked, prior}})`; done on a clean round. Then update the solution documents, append the journal entry, report, ask for commit approval

## Resume

- Same session: `Workflow({scriptPath, resumeFromRunId: "<run id>"})` - completed agents return cached results
- After a restart: re-run the persisted script. Stage A executors are idempotent on `results/H<n>.json`. Stage B implementers build on the working tree as it stands; check `git status` and the returned package results first, then remove finished packages from `build-args.json` before re-running
