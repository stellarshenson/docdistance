# E19 primer - relevance reading on real documents

Read by every executor and verifier of the workflow `reports/E19-relevance/workflow-measure.js`. Batch E19 of `docs/experiments/wmd-docdistance-experiments.md` holds five pre-registered hypotheses in two notebooks. The registered blocks and the batch opening (shared Setup) win over this primer wherever they differ; report any difference.

## Environment and hard rules

- Repository root `/home/lab/workspace/learning/projects/docdistance`, package `docdistance` 1.1.32 installed editable in `.venv`
- Execute a notebook: `CI=true HF_HUB_OFFLINE=1 .venv/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 <notebook> > logs/<name>.log 2>&1`, from the repository root. The first code cell sets `os.environ["DOCDISTANCE_HOME"]` to the repository root and `CUDA_VISIBLE_DEVICES=""` before any torch import
- A run that can exceed 9 minutes starts detached (`nohup setsid ... &`) and you poll its log and its output files; never wait on your own child process. The reranker notebook takes about 2 hours
- Another session can change files under `src/` and `tests/` and can run `make install` while you work (a build of new map fields). Do not edit there, and record the `docdistance.__version__` your notebook ran on. If an import fails during such an install, wait one minute and execute again
- No edit under `src/`, `tests/`, `docs/`, `data/processed/`, `data/external/`, `.claude/`, `reports/E17-*`, `reports/E18-*`, and no edit of `reports/E19-relevance/E19-relevance-registration.json`, this primer or the workflow script. No `make install`, no package install, no network request, no git commit. Never look into an `@archive` or `.ipynb_checkpoints` folder
- An executor writes only its own notebook, metrics file, log and result file (names below), and for the reranker notebook the grid file. Scratch files go to `/tmp/claude-1000/-home-lab-workspace-learning-projects-docdistance/48fccd81-2e03-4952-88be-100e909f2767/scratchpad/e19-exec/`
- Nobody but the orchestrator writes the ledger, the journal or a tracker file
- The experiment lives in the notebook: all computation in notebook cells. Figures render inline; no figure files
- Nothing is tuned to the outcome: no parameter, threshold, label, exclusion or metric beyond the registered text. If a registered step cannot be computed as written, stop and report it
- Plain literal English in notebook text: no figures of speech, no em-dashes (use ` - `), unicode arrows. No client or customer name anywhere

## Fixture and shared definitions

- `data/processed/relevance-fixture-eu/`: `report.md` (883 statements with the default reader), `briefs/<stem>.md` (14 notes), `labels.json`, `sections.json`. Assert that the sha256 of every file listed under `fixture_sha256` in `reports/E19-relevance/E19-relevance-registration.json` equals the stored value
- Template: `notebooks/experiments/E18-kj-contrast-replication.ipynb` and `notebooks/experiments/E18-kj-reranker-relevance.ipynb`. Copy their cells for the reading (one `DocDistance()`, default reader, `embed_statements`, the `section` field of the `flows` of `semantic_distance_with_details` with the report as the first document), the AUC (Mann-Whitney, ties one half), the bootstrap (10000 resamples, seed 0, reported), the balanced accuracy and the exact Otsu split, without changing their logic. Say in the notebook which cells were copied. The labels and the section reading differ from E18 and follow this primer
- Labels (`labels.json`, field `rule`): split the `section` path of a statement at ` > `. The statement is neutral for every note when one element of the path equals a heading in `neutral_sections`; neutral for one note when an element equals one of that note's `related_sections`; positive when an element equals one of the note's `target_sections`; negative otherwise. Neutral statements are left out of every metric. Assert for each note that its targets hold at least 5 statements in total, and print the positive, neutral and negative counts per note. Print the distinct path elements once so the reader sees that the heading texts match
- The note under `no_target_notes` has no positive statements: it is read without a bar (see below) and is not part of any mean
- Baseline: `b_i = min_j c_ij` with `c = cost_matrix(E_report, E_note)`; match rule `b_i <= DIFF_CHANGED_COST`
- AUC per note of a score for positives against negatives; headline = mean over the 13 notes; per-note difference to the baseline, count of notes above 0, bootstrap interval of the mean difference
- Reported AUC with hard negatives: negatives restricted to the statements whose first path element (the chapter) equals the chapter of the note's first target
- Section reading: the units are the headings of `sections.json` with `role` in (`chapter`, `numbered`, `subheading`). A statement belongs to the deepest element of its path that is such a unit. Leave out units that hold fewer than 5 statements and units that are neutral for the note (a unit is neutral when one of the note's neutral headings is the unit or one of its ancestors). A unit's score is the mean score of its statements; rank the units from most to least relevant. A note has a top-3 hit when one of the 3 best-ranked units is a target heading or lies below one. Report per note the rank of its best target unit and the 3 best-ranked units
- Gates, in this order, printed with their numbers before any bar: baseline mean AUC over the 13 notes at or above 0.95 kills E19-H94, E19-H95 and E19-H96 at the gate (their readings are still computed and stored, labelled as beyond the gate); fixed-cutoff mean balanced accuracy at or above 0.65 kills E19-H97

## Notebook B - `notebooks/experiments/E19-kj-real-document-map.ipynb` (E19-H93, E19-H94, E19-H96, E19-H97)

- E19-H93: bar: baseline mean AUC over the 13 notes at least 0.80 AND top-3 hit for at least 10 of 13 notes. No gate
- E19-H94: `all_but_the_top({"a": E_report, "b": E_note}, k=1)` before the cost matrix, score `min_j c'_ij`; bar: mean AUC difference at least 0.03 and above 0 in at least 10 of 13 notes
- E19-H96: `s_i(n) = b_i(n) - mean over the other 12 notes of b_i`; the note without a target is not part of any background; same bar
- E19-H97: exact Otsu split of the E19-H96 score of every report statement of the pair, match iff the score is at or below the threshold; balanced accuracy per note on the non-neutral statements; bar: mean at least 0.70 and at least 0.10 above the fixed cutoff's
- Reported without a bar: the section reading for every score; the hard-negative AUC; for the note without a target, per score its 3 best-ranked units and the quantiles of the score, and the share of report statements that the fixed cutoff and the Otsu split of its contrast score (background: the 13 notes) mark as a match
- Metrics: `reports/E19-real-document-map-metrics.json`; log: `logs/E19-real-document-map.log`; result: `reports/E19-relevance/results/B.json`. Store the baseline nearest costs of all 14 notes in `data/interim/E19/baseline-costs.npz` (git-ignored) for the checks

## Notebook A - `notebooks/experiments/E19-kj-real-document-reranker.ipynb` (E19-H95)

- `load_reranker()` from `docdistance.encoders` with its defaults; `R = reranker.score_grid(note_statements, report_statements)`; `r_i = max_j R[j, i]`. About 444,000 pairs over the 14 notes, about 2 hours on CPU. Compute one note at a time, store each grid in `data/interim/E19/reranker-grids.npz` as soon as it is computed (rewrite the file after every note), and reuse a stored grid on a later run, so that an interrupted run continues. State in the notebook how many grids were computed in the last run and how many were read from the file
- E19-H95: AUC of `r_i` per note against the baseline AUC; bar: mean difference at least 0.03 and above 0 in at least 10 of 13 notes; gate as above (the notebook computes the baseline itself)
- Reported without a bar: the section reading and the top-3 hit count; the hard-negative AUC; the contrast of the reranker score (`r_i` minus its mean over the other 12 notes) with its AUC; the Otsu split of `r_i` (match on the high side) with its balanced accuracy; the reading of the note without a target; the reranker time per note and per 1000 pairs
- Metrics: `reports/E19-real-document-reranker-metrics.json`; log: `logs/E19-real-document-reranker.log`; result: `reports/E19-relevance/results/A.json`

## What a notebook holds

Title cell with the purpose in two sentences; environment cell; imports; one configuration cell (paths, constants, bars) with a table; fixture hash check and label counts; gates printed with their numbers before any bar; one results table per hypothesis; one per-note table that makes losing notes visible; one inline figure of the per-note AUC of the baseline and of each score; a closing markdown cell with the conclusions per hypothesis, grounded in the printed numbers and no stronger than they are, including what the fixture cannot show (one report, 13 notes, one annotator, section-level labels). The metrics file holds: environment, fixture hash equality, statement counts, label counts, per note every score's AUC, the section reading, every gate value and every bar clause as value and boolean, and `verdict_by_bar` per hypothesis out of Confirmed, Refuted, Killed-at-gate.

## Executor result (the JSON you return and also write to your result file)

`notebook`, `metrics_json`, `log`, `notebook_errors` (must be 0), `harness_check` (here: the fixture hash check and the label-count assertion; `largest_difference` 0 when both pass), per hypothesis: `gate` (reading, passed), `clauses` (each with value and met), `verdict_by_bar`, `key_numbers`; `per_brief` (the per-note table as a list); `deviations`; `observations`; `not_verified`.
