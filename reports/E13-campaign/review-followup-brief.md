# Review follow-up - brief for the implementer

Written 2026-10-10 after review run `wf_366ba93d-451` (architect, bug-hunter, data-scientist; stopped by the orchestrator after the fix of round 3, tree at version 1.1.18). The adjudicators of rounds 1 to 3 handed these entries to the orchestrator; I confirmed each against the tree. Round 3 added the entry G6. The implementer applies every entry and reports per entry. An entry that turns out to be wrong against the code is not applied - report it with the reason.

Rules that hold for every entry: the stage B contract of `PRIMER.md` (additive policy with its two registered exceptions, `make install` once at the end and never `SKIP_VERSION_INCREMENT`, the three gates), no existing test is renamed, no edit to `docs/acc-crit-docdistance.md`, `docs/defects-docdistance.md`, `docs/experiments/wmd-docdistance-experiments.md` or `.claude/JOURNAL.md` (the orchestrator writes those through their tools), no commit, no request to the HuggingFace Hub or any other network service (run every real-model step with `HF_HUB_OFFLINE=1`; the models and the tokenizer are in the local HuggingFace cache).

## G1 - defect `DEF-DEPS-17`: the segmenter does not load after `init --source`

Reproduced by the review (round 1, bug-hunter and architect): a host provisioned with `docdistance init <mode> --source <s3 prefix or local dir>` whose HuggingFace cache lacks `facebookAI/xlm-roberta-base` raises `ModelsNotInstalled` at the first `Segmenter` load. Version 1.1.5 requested that tokenizer from the Hub at the first load, so it worked on a host with Hub access. The cause has three parts:

- wtpsplit loads the tokenizer of SaT by a Hub name: `SaT.__init__(model_name_or_model, tokenizer_name_or_path="facebookAI/xlm-roberta-base", ...)` (`.venv/lib/python3.14/site-packages/wtpsplit/__init__.py`, class `SaT`)
- the source folder `sat-3l-sm` holds `config.json` and `model.safetensors` only, and `_mirror_one` (`src/docdistance/bootstrap.py`) returns on a source hit before `_warm_hf`, the only place where `init` constructs `SaT`
- `Segmenter.__init__` (`src/docdistance/encoders.py`) now holds the Hub library offline during the load (the `DEF-DEPS-14` fix), so the missing tokenizer ends the load

Fix, inside the repository:

- `src/docdistance/config.py`: one constant for the tokenizer name, `SAT_TOKENIZER = "facebookAI/xlm-roberta-base"`
- `src/docdistance/bootstrap.py`: after a source hit (S3 or local) for the key `sat`, when the copied folder holds no `tokenizer.json`, `init` saves the tokenizer beside the weights (`AutoTokenizer.from_pretrained(config.SAT_TOKENIZER).save_pretrained(dest)`). `init` is the online step and already holds the Hub library online. When the tokenizer is in the source folder, `init` makes no call for it. When it is neither in the source folder, nor in the HuggingFace cache, nor reachable, `init` raises an error that names the tokenizer and the two remedies (copy the tokenizer files into the source folder `sat-3l-sm`, or run `init` once on a host with Hub access). Raise `ModelsNotInstalled`, the type the first load raises today and the command line already maps to an error exit; do not add a new class
- `src/docdistance/encoders.py`, `Segmenter.__init__`: when the SaT source is a folder that holds `tokenizer.json`, pass the folder as `tokenizer_name_or_path`; otherwise call `SaT(sat_src)` as today (the HuggingFace source of `init` keeps working from the cache)
- Keep it to these three places. No new command line option, no change to the S3 bucket, no change to `make sync_models_up`

Tests (`tests/test_bootstrap.py`, `tests/test_e2e.py` or `tests/test_segmentation.py`; match the surrounding style):

- a source folder for `sat` without `tokenizer.json`: `init` calls the tokenizer save once, with `config.SAT_TOKENIZER` and the destination folder (stub the save; no network)
- a source folder for `sat` with `tokenizer.json`: `init` makes no tokenizer call. `test_init_local_source_records_local_and_never_touches_hf` keeps its name and its meaning: give `_fake_local_mirror` a `tokenizer.json` in the `sat` folder and make the tokenizer save raise in that test
- the tokenizer save raises: `init` raises the named error and the message names `facebookAI/xlm-roberta-base`
- `Segmenter` passes the folder as `tokenizer_name_or_path` when the folder holds `tokenizer.json`, and passes none when it does not (stub `wtpsplit.SaT`, record the arguments)
- one real-model test in a child process, skipped when the models are not installed (pattern: `test_offline_load_of_cached_models...` in `tests/test_e2e.py`): step 1 builds a source folder from the cached snapshot of `segment-any-text/sat-3l-sm` (`config.json` and the weights only) and provisions it with `HF_HUB_OFFLINE=1`, so the tokenizer comes from the cache; step 2 loads `Segmenter` from the provisioned folder in a child process whose `HF_HOME` is an empty folder and whose proxies name a closed local port, and splits a text. Assert that the statements equal the statements of a `Segmenter` loaded by the Hub name from the cache. The test must fail on the tree before the fix (step 2 raises `ModelsNotInstalled`); confirm that on a copy of the tree or by reverting the two source edits for one run, and say which you did

Documents: rounds 1 and 2 of the review added sentences that describe the defect as a limit. Rewrite them to the fixed behaviour, from the code: `docs/aws-deployment-reference.md` (line 3 and the `sat-3l-sm` bullet near line 12), `README.md` near line 136, `docs/cli-reference.md` near line 11, the module docstring of `src/docdistance/bootstrap.py`. State: what `init --source` copies, that it saves the tokenizer beside the SaT weights when the source folder lacks it (from the HuggingFace cache or the Hub), that a source folder which carries the tokenizer needs no Hub, that loads after `init` make no Hub request, and how to make a source folder complete (copy the tokenizer files that `init` saved in `<home>/models/sat-3l-sm/` into the source folder). Do not state anything about the content of the S3 bucket as tested. `CHANGELOG.md`: one line under the unreleased section.

## G2 - wording of the unsupported count

The docstrings name `n_unsupported_a` / `n_unsupported_b` as 'the reading of content the source does not support'. E15-H78 calibrated the count on statements of another article. On the 98 statements of the ibm summaries it flags 17, of which 10 are not entailed by the model-graded label, and it leaves 33 not-entailed statements unflagged (`reports/E15-H78-raw-nearest-source-count-metrics.json`, `arms.raw_kept.entailment.all_summaries`). Reword three sites to what the count is: the count of statements with no near source statement (nearest-source cost above `SOURCE_SUPPORT_CUTOFF`), not a reading of entailment; the model-graded reading is `grounding_residual` / `grd_a`, `grd_b`.

- `src/docdistance/distance.py`, docstring of `ungrounded_residual` (the sentence that starts ':func:`unsupported_count`')
- `src/docdistance/distance.py`, docstring of `SourceConditionedResult` (the sentence '``n_unsupported_a`` / ``n_unsupported_b`` are the reading of content the source does not support')
- `docs/api-reference.md`, the row `n_unsupported_a`, `n_unsupported_b`

The partial-coverage sentence and the pointer to `n_unsupported_a` / `n_unsupported_b` stay (criterion `ACC-TRANS-72` asserts both). `docs/solution/wmd-source-conditioned-docdistance-solution-sota.md` already states the limit; add the two counts (17 of 98 flagged, 10 of the 43 not-entailed statements among them) to its bullet 'Limits of the count' if they are not there.

## G3 - the solver call named in the documents

`_exact_ot` calls `ot.emd` with `log=True`, not `ot.emd2`. Correct the name at every site that describes the shipped solver call: `docs/solution/wmd-docdistance-solution-sota.md` (near lines 32, 63, 201, 223), `docs/solution/wmd-structure-distance-sota.md` (near lines 34, 120, 187, 208). Leave `src/docdistance/distance.py:229` (an inequality that names `emd2` as the optimum) and the reproduction line of `DEF-TRANS-9` in the defects register.

## G4 - the literal 71 in the command line help

`src/docdistance/cli.py`, the `--anisotropy` help of `distance-semantic` and `distance-structural` (near lines 354 and 490) carries the literal 71. Read it from `ANISOTROPY_MIN_POOL` so the help cannot disagree with the constant. The printed help text stays the same; if a test compares the help text, it stays green.

## G5 - three ledger readings without a stored computation

Three readings in recorded ledger lines are computed by no notebook and stored in no metrics file: 0.9052 / 0.9143 (E15-H78, held-out balanced accuracy 'with each scored foreign statement left out of the calibration') and 0.5390 (E14-H76, arm a against shipped DR(D_sel) 'on the 7 documents not read before registration'). The data-scientist reviewer reports that the first pair depends on a rule: own side scored at each leave-one-out cutoff reads 0.9052 / 0.9143, own side scored at the full-calibration cutoff reads 0.9017 / 0.9140.

Write `reports/E13-campaign/checks/held-out-readings/readings.py` (run from the repository root with `CI=true .venv/bin/python reports/E13-campaign/checks/held-out-readings/readings.py`, `HF_HUB_OFFLINE=1`): it recomputes the three readings from the same inputs and code paths as `notebooks/experiments/E15-H78-*.ipynb` and `notebooks/experiments/E14-H76-*.ipynb` (read the notebooks for the fixtures, the encoder and the cutoff rule; reuse stored per-statement values where the notebooks or `data/interim/` store them, otherwise re-encode), prints both rule variants for the first pair, and prints the difference of each reading from the ledger value. Add its line to `reports/E13-campaign/checks/README.md`. Do not edit a notebook. If a reading does not reproduce to 4 decimals, do not tune: report the value you get and the rule you used.

## G6 - two readings of the pre-pass gain with the wrong setting

Round 3 of the review (architect, confirmed by the adjudicator):

- `docs/solution/wmd-source-conditioned-docdistance-solution-sota.md` quotes 3.39x for the widening of the `D_sel` dynamic range near lines 10, 65, 78, 121 and 137, and near lines 78 and 137 it names `k = 1`. The record of E04-H15 reads 3.39x at best `k = 2` on the E04 pool (ledger `docs/experiments/wmd-docdistance-experiments.md`, the E04-H15 block). The shipped setting `k = 1` on the pool of one triple {document, anchor, source} reads 6.3433x at 0 of 28 selection-axis violations (E14-H76, `reports/E14-H76-conditioned-centering-probe-metrics.json`, `kill_gate.dr_ratio_shipped_over_raw`). Correct the sentences that attribute 3.39x to `k = 1`: state each reading with its setting and its pool. Read the two ledger blocks first; do not edit the ledger
- `notebooks/05-kj-source-conditioned-distance.ipynb`, markdown cell 15, says `~7.4x at 0 violations`; the stored output of code cell 16 reads 5.93x at 1 violation. Edit the markdown cell to the stored output. Do not execute the notebook; change the source of that one cell only (through `nbformat`, so the file stays valid) and report the sentence before and after

## Gates and report

After all entries: `make install` once, then `make lint`, `make test`, `CI=true .venv/bin/python reports/E13-campaign/baseline.py verify --allow moved,n_moved --grounded` (expected result line: `verify: 55 pairs, 3 grounded, 0 difference(s)`). Report: per entry what you changed (file and function), the test names you added, for the real-model test how you confirmed that it fails before the fix, the three recomputed readings with the rule, the version after `make install`, the three gate result lines verbatim, and anything you did not do with the reason.
