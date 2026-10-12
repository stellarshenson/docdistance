# Stage B repairs - brief for the seventh package

Written 2026-10-10 after the six build packages of run `wf_e1b16a4b-b94` returned. Each entry below is a verifier finding or an implementer report that I confirmed against the tree at version 1.1.12. The implementer applies every entry; the verifier checks every entry. Full text of the findings: `results/stage-b-run-wf_e1b16a4b-b94.json`.

Rules that hold for every entry: the stage B contract of `PRIMER.md` (additive policy, `make install` once, the three gates), no existing test is renamed, and an entry that turns out to be wrong against the code is not applied - report it with the reason.

## Defects

- **D1 - `DEF-DEPS-14`** (`src/docdistance/encoders.py`): the constructors set `HF_HUB_OFFLINE` after `_require_models_extra()` has imported `transformers` and `huggingface_hub`, which read the flag at import. Reproduction that I ran: environment without `HF_HUB_OFFLINE`, `http_proxy` and `https_proxy` on `http://127.0.0.1:9`, models cached → `DocDistance().embed(text)` raises `ModelsNotInstalled`; with `HF_HUB_OFFLINE=1` preset it returns shape `(2, 768)`. Fix it so that `offline=True` loads cached models without a request to the Hub, whether or not the host process imported `huggingface_hub` earlier. `docdistance init` stays the online step (`bootstrap.py` removes the flag). Test: a subprocess with the flag removed and the proxies on the closed port loads the encoder and embeds; it skips when the models are not installed; it fails on the tree before the fix
- **D2 - `DEF-SEGM-15`** (`src/docdistance/encoders.py`, `split_markdown`): a line that holds only a marker comes back as a prose statement, against the docstring. Confirmed: the lines `-`, `###`, `---` and `* * *` are returned as statements. Drop marker-only lines: a bare list marker, a bare heading marker, a horizontal rule (`---`, `***`, `___`, with or without spaces). A table separator row is already dropped. `tests/test_segmentation.py::test_markdown_reader_does_not_warn_about_marker_statements` asserts the old return and changes with the fix. Checks: the ibm source `data/interim/exec-summaries/ibm-ai-adoption/source/source-article.md` with `markdown=True` still returns 48 statements; the sha256 of `docdistance distance-semantic --pairs data/interim/E12-H67/allpairs-55.tsv --markdown` stays `62d98b6b3f...` (the `markdown` arm in `logs/E13-batch-memos-measure.log`). The default reader does not change

## Statements that are false

- **R1** (`src/docdistance/distance.py`, `_exact_ot` docstring, and the docstring of the raise test): the coupling POT returns at the iteration cap is not always feasible and its cost is not always an upper bound of the distance. State what holds
- **R2** (`_emd_iter_cap` docstring): an `n x 1` problem needs exactly `n * m` iterations, so the cap has no slack on that shape; the sentence '96000 to 138000, more than POT's fixed default allows' includes values below the default of 100000. Correct both
- **R3** (`partial_matching` docstring, criterion `ACC-TRANS-83`, which overrides `ACC-TRANS-71`): the general statement that copies of a repeated statement pair in document order is false. When every tied copy shifts the same way by more than the spacing of the copies, the in-order and the crossed pairing have the same position sum and the solver decides (transport-core verifier: 16 one-hot statements, copies at 8 and 9, statements 11, 13, 14 deleted → pairs `[8, 9]`, `[9, 8]`). State the rule as `ACC-TRANS-83` words it: deterministic, copies ranked by position, the tie condition, and the measured limit of E15-H79 (111 of 120). The position term itself does not change: `abs` with `1e-6` is the registered form
- **R4** (`src/docdistance/distance.py`, comment on `ANISOTROPY_MIN_POOL`): 71 is the smallest of the pool sizes measured in E12-H65 at which the reading holds, not 'the smallest pool'; name the encoder of the measurement. Numbers from `reports/E12-H65-cmr-pool-crossover-metrics.json`
- **R5** (`src/docdistance/cli.py`, `--anisotropy` help of `distance-semantic` and `distance-structural`): 'needs a corpus, off by default for a pair' contradicts `docs/cli-reference.md`. Every shipped call estimates the pre-pass from the statements of the two documents of one pair; the help states that and the minimum of about 71 pooled statements
- **R6** (`docs/solution/wmd-docdistance-solution-sota.md`): the sentence 'removing them is pure signal gain' stands beside the bullet that reads the projection step alone at -4.5% and -3.6%; the sentence 'estimated from the pooled statements of all documents compared, so it sharpens a batch or corpus' describes a pool that no command builds. Correct both from the code and the ledger
- **R7** (`docs/solution/wmd-source-conditioned-docdistance-solution-sota.md`, the bullet '`D_grd` not yet in the library'): the library returns `grd_a`, `grd_b` and `d_grd`. Correct the bullet

## Behaviour without a test that can fail

- **R8** (`tests/test_distance.py`): no test pins the floor `EMD_MIN_ITER`; with `max(EMD_MIN_ITER, n * m)` changed to `n * m` the suite stays green. Add one
- **R9** (`tests/test_distance.py` or `tests/test_structure.py`): the pair filter `<` at `MATCH_CUTOFF` and the count `>` at `SOURCE_SUPPORT_CUTOFF` have no boundary test. Add one each: a pair exactly at the cutoff stays unmatched, a statement exactly at the cutoff is not counted
- **R10** (`tests/test_structure.py`): `summary.n_moved` is asserted only where it equals the count of nonzero `matched_displacement`. Add a case where the two differ
- **R11** (`tests/test_structure.py`, the assertion `assert rounding == 7`): it checks arithmetic written inside the test. Remove it or assert an output of the build
- **R12** (`tests/test_pipeline.py`): the docstring of `_warn_small_pool` says the source-conditioned calls log no pool warning. Assert it for `distance_wrt_source` and `distance_wrt_source_with_map`
- **R13** (`tests/test_pipeline.py`): the rule 'the memo key is taken before the file is read' has no test. Add one that fails when the key is taken after the read
- **R14** (`tests/test_e2e.py::test_loading_the_openvino_encoder_starts_no_child_process`): the verifier reports that `import openvino` alone starts the child process on a tree without the block, and that the model load is 8 of the test's 10 seconds. Check it on a copy of the tree without the block. If it reproduces, the test imports `openvino` and loads no model; if not, leave the test and report

## Additions

- **R15** (`_warn_marker_statements`): the warning names no document, so a `--pairs` run does not say which file it concerns. Name the path when the input is a file. One line per call stays: a memo hit logs it again, which keeps memo on and memo off equal
- **R16** (`src/docdistance/__init__.py`): export `SolverNotOptimal` like the other public names; `docs/api-reference.md` names the root import
- **R17** (`reports/E13-campaign/checks/`): the real-model readings that the build quoted came from scripts in the session scratchpad `/tmp/claude-1000/-home-lab-workspace-learning-projects-docdistance/48fccd81-2e03-4952-88be-100e909f2767/scratchpad/`. Copy the ones a criterion's evidence rests on and make each run from the repository root with `CI=true .venv/bin/python reports/E13-campaign/checks/<name>.py`: `batch-memos/measure.py` (timed runs, nine-arm digests, one-shot comparison), `batch-memos/guardrail_memo_on.py` (the baseline with the memo on), and from `detail-map/` the script or scripts that read the six E12-H61 fixtures and the source map against the build. A script that needs reference data kept only in the scratchpad is copied with that data when the data is under 200 KB, otherwise it is left out and reported. Remove absolute scratchpad paths. One `README.md` in the folder, one line per script: what it reads and the command
- **R18** (`logs/README.md`): one line for the family `E13-batch-memos-*.log`

## Declined, with the reason

- The capture fixture written twice in `tests/test_pipeline.py` and the stub harness defined twice in `tests/test_e2e.py`: duplication without an effect on behaviour
- The tolerance `1e-5` in `test_matched_order_gap_ignores_a_pure_deletion`: it is the tolerance of `ACC-TRANS-80`, set from the measured 7.813e-06
- The marker warning repeated for every row a document appears in: memo on and memo off logging the same lines is a tested property; R15 adds the name instead
- `ACC-CLI-32` read on a median of five interleaved repeats: the criterion names no statistic, E12-H67 used medians, and the bookkeeping timed alone is 4.39 ms of about 17 s

## Follow-up after the repairs run (2026-10-10)

Run `wf_099e69d3-480` applied D1, D2 and R1 to R18 and was verified with 7 minor findings. Confirmed and still to do:

- **F1** (`src/docdistance/encoders.py`, `_hub_offline`): the scope changes a process-wide flag without a lock. Two loads that overlap on two threads leave the second load online after the first ends, and the flag stays `True` after both end, against the docstring. Make the scope safe for overlapping loads: the flag is set when the first offline scope starts and the earlier value is put back when the last one ends. Test: two overlapping scopes on two threads; the flag is `True` until the second ends and has its earlier value afterwards; the test fails on the current code
- **F2** (`pyproject.toml`): `huggingface_hub` is imported directly by `encoders.py` and `bootstrap.py` and is not declared. Declare it under `dependencies`, then `make install`
- **F3** (`docs/solution/wmd-source-conditioned-docdistance-solution-sota.md`, Implementation section): the bullets 'Grounding axis (notebook) ... not yet folded into the library' and 'Next - wire the H11 relevance-gated `D_grd` ... into the library' are false. The library returns `grd_a`, `grd_b` and `d_grd`. Correct both from the code
- **F4** (`reports/E13-campaign/checks/`): the numbers in the docstrings of `_exact_ot` and `_emd_iter_cap` come from `scratchpad/repairs/r1_r2.py`. Copy that script into `checks/solver-cap/`, make it run from the repository root, add its line to `checks/README.md`
- **F5** (`tests/test_distance.py::test_exact_solver_cap_is_never_below_the_pot_default`): it asserts that `EMD_MIN_ITER` equals POT's default `numItermax`. The stated property is 'never below'. Assert `>=`
- **F6 - `DEF-DEPS-16`** (`src/docdistance/bootstrap.py`): `init` removes `HF_HUB_OFFLINE` from the environment after the Hub library is imported, so with `HF_HUB_OFFLINE=1` preset the library stays offline and `init` cannot download. Make `init` hold the library online for its run, with the same care for the earlier value as F1. Test without network: with the flag preset to offline, the Hub library reads online inside the init step and reads its earlier value afterwards; the test fails on the current code
- **F7** (`docs/aws-deployment-reference.md`, the 'Offline' bullet): 'the loaders set `HF_HUB_OFFLINE=1` after init' is false from 1.1.13. State what the loaders do
- **F8** (`src/docdistance/distance.py`, docstring of `compute_distance`): it still says 'the smallest pool'. Word it as the comment on `ANISOTROPY_MIN_POOL` does
