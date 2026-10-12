# E18 primer - relevance reading for a single brief

Read by every executor and verifier of the workflow `reports/E18-relevance/workflow-measure.js`. Batch E18 of `docs/experiments/wmd-docdistance-experiments.md` holds six pre-registered hypotheses in two notebooks. The registered blocks and the batch opening (shared Setup) win over this primer wherever they differ; report any difference.

## Environment and hard rules

- Repository root `/home/lab/workspace/learning/projects/docdistance`, package `docdistance` 1.1.32 installed editable in `.venv`
- Execute a notebook: `CI=true HF_HUB_OFFLINE=1 .venv/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 <notebook> 2>&1 | tee logs/<name>.log`, from the repository root. The first code cell sets `os.environ["DOCDISTANCE_HOME"]` to the repository root and `CUDA_VISIBLE_DEVICES=""` before any torch import
- No edit under `src/`, `tests/`, `docs/`, `data/processed/relevance-fixture/`, `.claude/`, `reports/E17-*`, and no edit of `reports/E18-relevance/E18-relevance-registration.json`, this primer or the workflow script. No `make install`, no package install, no network request, no git commit. Never look into an `@archive` or `.ipynb_checkpoints` folder
- An executor writes only its own notebook, metrics file, log and result file (names below). Scratch files go to `/tmp/claude-1000/-home-lab-workspace-learning-projects-docdistance/48fccd81-2e03-4952-88be-100e909f2767/scratchpad/e18-exec/`
- Nobody but the orchestrator writes the ledger, the journal or a tracker file
- The experiment lives in the notebook: all computation in notebook cells. Figures render inline; no figure files
- Nothing is tuned to the outcome: no parameter, threshold, label, exclusion or metric beyond the registered text. If a registered step cannot be computed as written, stop and report it
- Plain literal English in notebook text: no figures of speech, no em-dashes (use ` - `), unicode arrows

## Fixture and shared definitions

- Report: `data/processed/relevance-fixture/report.md`, 93 statements with the default reader. New briefs: `data/processed/relevance-fixture/e18/briefs/brief-13.md` to `brief-24.md` with `e18/labels.json`. Briefs of E17: `data/processed/relevance-fixture/briefs/brief-01.md` to `brief-12.md` with `labels.json`. Controls: the 7 summaries named in both label files
- Assert that the sha256 of every file listed under `fixture_sha256` in `reports/E18-relevance/E18-relevance-registration.json` equals the stored value
- Template: `notebooks/experiments/E17-kj-relevance-map.ipynb`. Copy its cells for the reading (one `DocDistance()`, default reader, `embed_statements`, the `section` field of the `flows` of `semantic_distance_with_details` with the report as the first document), the labels (leaf = last element of the section path split at ` > `, part = first element, neutral sections left out for briefs, part-level labels for controls), the AUC (Mann-Whitney, ties one half), the bootstrap (10000 resamples, seed 0, reported), the section reading (top 2), the balanced accuracy and the exact Otsu split, without changing their logic. Say in the notebook which cells were copied
- Harness check, before any new brief is scored: reproduce from the copied cells the per-brief baseline AUC of briefs 01 to 12 stored in `reports/E17-relevance-map-metrics.json` (`auc_per_brief[<brief>]["baseline"]`), and for the contrast notebook also `["H85"]`, with an absolute difference below 1e-9. A failed harness check stops the run
- Baseline: `b_i = min_j c_ij` with `c = cost_matrix(E_report, E_brief)`; match rule `b_i <= DIFF_CHANGED_COST`
- Headline set: briefs 13 to 24. Every bar is read on them. Readings on briefs 01 to 12 are reported and labelled as seen data
- Batch gate: baseline mean AUC over briefs 13 to 24 at or above 0.95 kills E18-H87, E18-H88, E18-H89 and E18-H92 at the gate; their readings are still computed and stored, labelled as beyond the gate
- Gate of E18-H90 and E18-H91: mean balanced accuracy of the fixed cutoff over briefs 13 to 24 at or above 0.65 kills them at the gate

## Notebook A - `notebooks/experiments/E18-kj-reranker-relevance.ipynb` (E18-H87, E18-H91)

- `load_reranker()` from `docdistance.encoders` with its defaults; `R = reranker.score_grid(brief_statements, report_statements)`, shape brief x report; `r_i = max_j R[j, i]`, higher = more relevant. Read the function before calling it and state in the notebook what its score is (sigmoid of the pair logit) and how the pair is ordered
- E18-H87: AUC of `r_i` per brief against the baseline AUC; bar: mean difference over briefs 13 to 24 at least 0.03, difference above 0 in at least 10 of 12, control mean AUC (reranker score on the 7 controls) not more than 0.01 below the baseline's control mean
- E18-H91: exact Otsu split of the `r_i` of every report statement of the pair, match iff `r_i` is above the threshold; balanced accuracy per brief on the non-neutral statements; bar: mean over briefs 13 to 24 at least 0.70 and at least 0.10 above the fixed cutoff's on the nearest cost
- Reported without a bar: E18-H87 and E18-H91 on briefs 01 to 12; the AUC against the negatives of the brief's own part; the top-2 section share; the fixed rule `r_i >= 0.5`; the contrast of the reranker score (`r_i` minus the mean of `r_i` over the other 11 new briefs) with its AUC; the wall time of the reranker per pair of documents
- Store the reranker grids of all 24 briefs and 7 controls in `data/interim/E18/reranker-grids.npz` (git-ignored) so that a check can read them without a second model run
- Metrics: `reports/E18-reranker-relevance-metrics.json`; log: `logs/E18-reranker-relevance.log`; result: `reports/E18-relevance/results/A.json`

## Notebook B - `notebooks/experiments/E18-kj-contrast-replication.ipynb` (E18-H88, E18-H89, E18-H90, E18-H92)

- E18-H88: `s_i(b) = b_i(b) - mean over the other 11 new briefs of b_i`; for a control: its `b_i` minus the mean over the 12 new briefs; bar as E18-H87. Reported: the background of the 12 briefs of E17, and of all 23 other briefs
- E18-H89: for each new brief and k in (1, 2, 3, 5): 200 draws of k briefs without replacement from the 23 other briefs of both sets, one `numpy.random.default_rng(0)` used in the order k, brief, draw; the AUC of the contrast per draw; per brief the mean over its draws. Bar at k = 3: mean over briefs 13 to 24 of (mean AUC over draws - baseline AUC) at least 0.03 and above 0 in at least 10 of 12. Reported: the other k, the share of single draws with a gain above 0 per k, the reading over all 24 briefs
- E18-H90: exact Otsu split of the E18-H88 score of every report statement of the pair (background: the other 11 new briefs), match iff the score is at or below the threshold (lower = more relevant); bar: mean balanced accuracy over briefs 13 to 24 at least 0.70 and at least 0.10 above the fixed cutoff's
- E18-H92: `g_i` = mean over the section groups S that do not hold statement i of `min over k in S of C[i, k]`, with `C = cost_matrix(E_report, E_report)` and the section groups = the distinct values of the `section` field (neutral sections and part headings included); score `b_i - g_i`, lower = more relevant; bar as E18-H87 (control: the same score on the 7 controls). Reported: the same score on briefs 01 to 12
- Metrics: `reports/E18-contrast-replication-metrics.json`; log: `logs/E18-contrast-replication.log`; result: `reports/E18-relevance/results/B.json`

## What a notebook holds

Title cell with the purpose in two sentences; environment cell; imports; one configuration cell (paths, constants, bars) with a table; harness check; gates printed with their numbers before any bar; one results table per hypothesis; one per-brief table that makes losing briefs visible; one inline figure of the per-brief AUC of the baseline and of each score; a closing markdown cell with the conclusions per hypothesis, grounded in the printed numbers, including what the fixture cannot show. The metrics file holds: environment, fixture hash equality, statement counts, per brief and per control every score's AUC, every gate value and every bar clause as value and boolean, and `verdict_by_bar` per hypothesis out of Confirmed, Refuted, Killed-at-gate.

## Executor result (the JSON you return and also write to your result file)

`notebook`, `metrics_json`, `log`, `notebook_errors` (must be 0), `harness_check` (passed and largest difference), per hypothesis: `gate` (value, passed), `clauses` (each with value and met), `verdict_by_bar`, `key_numbers`; `per_brief` (the per-brief table as a list); `deviations` (every place where you could not follow the registered text or this primer); `observations` (anything that can make a reading misleading); `not_verified`.
