# Build brief - relevance reading of the semantic map

Written 2026-10-12. Contract of the work package `relevance-map`: acceptance criteria `ACC-MAP-101`, `ACC-MAP-102`, `ACC-MAP-103`, `ACC-CLI-104`, `ACC-MAP-105`, `ACC-MAP-106` of `docs/acc-crit-docdistance.md`. Where this brief and a criterion differ, the criterion wins; report the difference.

## Why

The user reads the semantic detail map of a long report against a short client brief to find the parts of the report that matter for the client; one report against a single brief is the frequent case. Measured in batches E17 and E18 of `docs/experiments/wmd-docdistance-experiments.md`:

- The nearest ground cost of a report statement over the brief (the number behind `changed`) ranks the target sections at a mean AUC of 0.5865 and 0.6183 on two sets of 12 constructed briefs (E17, E18)
- The highest score of the shipped reranker per report statement ranks them at 0.8611 on 12 briefs it had not seen, above the baseline in 12 of 12, with one brief and no background (E18-H87, Confirmed). The share of target sections among the 2 best-ranked sections rises from 0.4167 to 0.9167
- No rule marks match and no match for one brief: the Otsu split of the reranker scores reads a balanced accuracy of 0.6400 against a bar of 0.70 (E18-H91, Refuted). Do not ship or document a match rule
- The map writes `changed` and `no_counterpart` for a flow but not the number they are read on

## Entries

- **R1 (`ACC-MAP-101`)** - every `flows` record of the semantic detail map gains `min_cost` (the row minimum of the ground cost, rounded as `min_incoming_cost` is) and `best_target_index` (its argmin, the statement `identical` is already judged against). `changed` and `no_counterpart` keep reading the unrounded value. Add both keys to the `definitions` block. Nothing else in the map changes
- **R2 (`ACC-MAP-102`)** - `DocDistance.semantic_distance_with_details(a, b, ..., relevance: bool = False)`. With `relevance=True`:
  - `R = reranker.score_grid(sb, sa)`: the statements of `b` are the queries, the statements of `a` the passages; shape `[n_b, n_a]`. The statements are the ones the map lists, after `min_words` and `skip_table_fragments`
  - flow `i`: `relevance` = `max_j R[j, i]`, `relevance_target_index` = `argmax_j R[j, i]`
  - target `j`: `relevance` = `max_i R[j, i]`, `relevance_source_index` = `argmax_i R[j, i]`
  - section `k` of `a` (the existing `sections` list): `mean_relevance` = mean of the unrounded flow relevance of its statements; `relevance_rank` = position when the sections are sorted by `mean_relevance` from high to low, ties in document order, 1 is the most relevant
  - `summary.top_relevant`: the indices of the flows with the highest relevance, as many as `top_changed` holds, ties by index
  - a header block `relevance` with the reranker model identity as the package knows it (`backend` and the model name of that backend), `queries: "b"`, `passages: "a"`; `definitions` entries for `relevance` and `mean_relevance` that say: the score is the sigmoid of the reranker logit for the pair, it is a score and not a distance or a probability of relevance, and it is not on one scale across pairs of documents
  - rounding as the other scores of the map (4 decimals) for written values; ranks and maxima are taken on unrounded values
  - the reading does not depend on `anisotropy`, `threshold` or `changed_cost`
- **R3 (`ACC-MAP-103`)** - `relevance=True` loads the reranker and not the NLI model: take the reranker load out of `_ensure_grounding` into a method of its own that `_ensure_grounding` calls. A missing reranker model raises `ModelsNotInstalled` before any scoring, naming the reranker and the `docdistance init` call that installs it; do not require the NLI model. The distance result, and every key the map holds without `relevance`, are equal to a call with `relevance=False`
- **R4 (`ACC-CLI-104`)** - `distance-semantic` gains `--relevance`. It needs `--details-json`: without it the command exits with code 2 and the project's error form (plain and `--json`), before a model loads. It applies to a single pair and to every row of `--pairs`. Without the flag the command's output and its map files are unchanged. The help text says which document goes first (the long document first, the brief second) and that the cost is one reranker pass per pair of statements. No new command, no panel table
- **R5 (`ACC-MAP-105`)** - `reports/E18-relevance/checks/shipped_relevance.py`: runs the command with `--relevance` on `data/processed/relevance-fixture/report.md` against the 12 briefs of `data/processed/relevance-fixture/e18/` (one `--pairs` batch), reads `relevance` and `section` of every flow from the map files, applies the labels of `e18/labels.json` (leaf heading = last element of the section path split at ` > `; statements of `neutral_sections` are left out; positive = leaf in the brief's `target_sections`), computes the AUC per brief (Mann-Whitney, ties one half) and prints the mean beside `summary_new_briefs_headline` of `reports/E18-reranker-relevance-metrics.json`. Exit code 1 when the two means differ by more than 0.005 (the map rounds scores to 4 decimals, which creates ties). Log to `logs/E18-shipped-relevance.log`; add no line to `logs/README.md` (the orchestrator does). Run it and report its output
- **R6 (`ACC-MAP-106`)** - documents: `README.md` (a short section on reading a long report against a brief: which document goes first, `--relevance`, the fields, the cost, the evidence in two sentences with the fixture limits, and the statement that no match rule for one brief is confirmed), `docs/cli-reference.md`, `docs/api-reference.md` (the map fields of R1 and R2), `.claude/skills/docdistance/SKILL.md` where it lists options or map fields, and `CHANGELOG.md` under a new `## [Unreleased]` section above `## [1.1.32]`. Quote only numbers from this brief or from the ledger batches E17 and E18. Batch E19 is running: write nothing about it. Plain literal English, no figures of speech, no em-dashes, unicode arrows, no client or customer name

## Tests

Follow the conventions of `tests/test_pipeline.py` and `tests/test_cli.py` (how they replace models, how they read maps). Every new behaviour gets a test that fails on the unchanged tree:

- R1: `min_cost` and `best_target_index` equal the row minimum and its argmin; `changed` agrees with `min_cost` against `changed_cost`
- R2: with a stub reranker that returns a known grid, each field equals its definition, including ties, `relevance_rank` and `top_relevant`; with `relevance=False` no relevance key exists anywhere in the map and the reranker is not loaded
- R3: after a `relevance=True` call the NLI model is still unloaded; result and common keys equal the plain call; a missing reranker raises `ModelsNotInstalled`
- R4: `--relevance` without `--details-json` exits 2 without a model load (plain and `--json`); single pair and `--pairs` write maps with the fields; a run without the flag writes none
- One end-to-end test with the real models, in the style of the existing end-to-end tests: the map of two short documents with `relevance=True` holds scores between 0 and 1 and ranks the statement that answers the query above an unrelated one

## Gates and limits of this run

- Do NOT run `make install` in this run. A long experiment notebook is executing and imports the package from `src`; the editable install already serves your change, and the orchestrator runs `make install` after the notebook ends. This overrides the line "New build to test" of `reports/E13-campaign/PRIMER.md`. No dependency changes
- Gates: `make format`, `make lint`, `make test`, `CI=true HF_HUB_OFFLINE=1 .venv/bin/python reports/E13-campaign/baseline.py verify --allow moved,n_moved` (must report 0 differences; new keys pass), `uvx ruff@0.16.2 check --target-version py312`
- The host runs another CPU-heavy job, so quote no timing
- Do not edit the ledger, the journal, a tracker file, `data/`, `notebooks/`, or the reports of E17 to E19 other than `reports/E18-relevance/checks/`
