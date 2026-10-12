# E13 campaign - agent primer

Two stages. Stage A (measure): one executor per hypothesis builds and executes one notebook, no source edit. Stage B (build): one implementer per work package changes `src/` and `tests/`. The registered record is the contract: bars and kill-gates are applied literally, never adjusted to a result.

## Environment (verified 2026-10-07, rebuilt and re-verified 2026-10-09)

- Repo root `/home/lab/workspace/learning/projects/docdistance`; run everything from it
- Python `.venv/bin/python` 3.14.8 - docdistance (editable; 1.1.5 during stage A, incremented by every `make install` of stage B), POT 0.9.6.post1, OpenVINO 2026.2.1, torch 2.11.0. `/opt/conda/bin/python` has no docdistance - never use it
- Up to 1.1.5 every command that imports docdistance needed the prefix `CI=true` (it switches the OpenVINO usage telemetry off) and a standalone script needed `if __name__ == "__main__":` (DEF-DEPS-11). From 1.1.6 the library blocks the telemetry itself; the prefix is harmless and the commands below keep it
- Execute a notebook: `CI=true .venv/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 <notebook> 2>&1 | tee logs/<name>.log`. `jupyter nbconvert` resolves to `/opt/conda` and fails on `import docdistance`
- Models resolve offline through `docdistance.json` (mmBERT, reranker, NLI: OpenVINO INT8). `DocDistance()` load plus one embed about 14 s; one grounded source-conditioned pair about 49 s. Stay on CPU
- This environment reproduces the 55 saved 1.1.5 ibm pair distances at difference 0.0
- A Bash call is capped at 10 minutes: run longer executions detached (`setsid nohup <command> > logs/<name>.log 2>&1 &`) and poll the log
- Never read `**/@archive` or `**/.ipynb_checkpoints`
- Other notebooks execute on this machine at the same time. A wall-clock reading is not a measurement unless the record asks for one; report `uptime` beside any timing you quote

## Hard rules, both stages

- No `git commit`, `git push`, `git tag`, `git checkout --`, `git stash`, no hand edit of the version in `pyproject.toml`, no `pip install`, `uv pip install` or `uv add`, no `make publish`
- `make install` is run by a stage B implementer only, as the stage B contract says. Stage A executors, verifiers and reviewers never run it
- Never write `.claude/JOURNAL.md`, `docs/acc-crit-docdistance.md`, `docs/defects-docdistance.md` or the ledger `docs/experiments/wmd-docdistance-experiments.md` - the orchestrator records through `journal-tools`, `pm-tools` and `hypothesis-tools`
- No web access, no MCP tools, no further subagents
- Language in every file: literal, no figures of speech, no em-dashes (use ` - `), unicode arrows `→`

## Stage A - measure

- Contract: `hypothesis-tools show docs/experiments/wmd-docdistance-experiments.md <id>` (fallback: the `### <id>` block). The ids are `E13-H70` to `E13-H74`, `E14-H75` to `E14-H77`, `E15-H78` to `E15-H80`, and `E16-H81`, which runs after stage B on the build that holds the option `input_format` (not on 1.1.5)
- Standing guardrails: `V` stays `0/24`, no edit under `src/`, `docs/`, `tests/`, metric property untouched. Every variant is prototyped inside the notebook: import the shipped functions and wrap or re-implement them there
- Deliverables, named by id:
  - notebook `notebooks/experiments/<id>-kj-<slug>.ipynb`, executed in place, 0 errors
  - metrics JSON `reports/<id>-<slug>-metrics.json`, written by the notebook: every measured number, bar readings, guardrail readings
  - result JSON `reports/E13-campaign/results/H<n>.json` - the return schema, identical to what the agent returns
  - log `logs/<id>-<slug>.log`
- Idempotency: if the result JSON exists with `notebook_errors: 0` and its notebook exists, return that JSON unchanged
- Order inside the notebook: kill-gate section first (a fired gate ends the run with `Killed-at-gate`, the notebook still executes so the reading is persisted) → harness sanity check where the record names one (a failed sanity check returns `Blocked` with the discrepancy) → registered experiment → guardrails measured (`np.array_equal` or a byte diff, never `allclose`; `V` counted) → conclusions cell grounded in the outputs
- Notebook conventions: `/home/lab/.claude/plugins/cache/stellarshenson-marketplace/datascience/1.8.18/skills/notebook-standards/SKILL.md` - header, environment cell, imports in one cell, seed, configuration cell with a rendered summary, a 1-2 sentence overview under every `##`, plots inline only (no `savefig`)
- Numbers in the result JSON appear verbatim in notebook outputs
- Verdict vocabulary (closed): `Ships` / `Kept` / `Promoted` / `Confirmed` / `Blocked` / `Dropped` / `Refuted` / `Refuted (null)` / `Killed-at-gate`. A probe whose prediction held and bar was met is `Confirmed`; a toggle or build that met its bar is `Promoted`; bar not met with the prediction falsified is `Refuted`; an effect indistinguishable from baseline is `Refuted (null)`; a failed precondition is `Killed-at-gate`

## Fixture map

- ibm panel: `data/interim/exec-summaries/ibm-ai-adoption/summaries/*.md` (11 files, 1 to 13 statements), source `.../ibm-ai-adoption/source/source-article.md` (70 statements), anchor `exec-summary-gold-opus-4-6.md`. Adversarial tier: `adv1-a`, `adv1-b`, `adv2-a`, `adv2-b`; gold tier: the other six non-anchor files. Panel code: `notebooks/experiments/E12-H66-kj-norm-mass-marginals.ipynb`. Naive row on OpenVINO INT8: margin +4.71, d' 4.68, R 1.670, DR 0.0873, V 0/24
- wergeland source: `data/interim/ai-society-wergeland/source/source-article.md`
- structure fixture: `data/processed/structure-fixture/{meta,pairs,reorder_pool,statements}.json` - 10 base docs (`gold, gold-2, v1, v2, opus, sonnet, haiku, werg-a, werg-b, werg-c`), `swap_ks` 0-40, 14 seeds per k. `statements.json` maps a document key to `{statements, block_id, ...}`; `<base>~bt` is the back-translation paraphrase of a base (werg-a, werg-b, werg-c: 11, 13, 14 statements)
- paraphrases: `data/interim/structure-paraphrase/<base>.bt.json` - a list of strings, the first element is the H1 title line; verify the statement alignment against the base doc before using it as ground truth
- order-preserving deletion set (280 cases): built in `notebooks/experiments/E12-H62-kj-order-gap-length-residual.ipynb`; paraphrase reference p95 0.005
- pooled statements: `data/interim/01-statements.parquet` (163 rows, column `text`)
- model-graded grounding signals: `data/interim/E12-H69-grounding-signals.npz` - `R__<doc>` `[n, 70]` relevance, `ent__<doc>` `[n]` entailment
- constants: `src/docdistance/distance.py:30,40` (0.77, 0.90), `src/docdistance/pipeline.py:40,47,53` (changed cost 0.325, no-counterpart cost 0.6, moved tolerance 0.05)
- orientation readings from the sizing probe of 2026-10-07 (not results): no arc above 0.592 in the 55 ibm plans; 1.00 targets per statement on the 4 equal-length pairs; summary-vs-source plans carry 44-51% of mass on arcs above 0.6; gold minus 2 statements reads order gap 0.227

## Stage A return schema

```json
{
  "id": "<id>", "slug": "...", "model": "<execution model>",
  "verdict": "<closed set>", "verdict_number": "<the number that justifies it>",
  "kill_gate": {"ran": true, "passed": true, "reading": "..."},
  "sanity_check": {"required": false, "passed": true, "reading": "..."},
  "prediction_check": {"predicted": "...", "observed": "...", "direction_matched": true},
  "acceptance": {"bar": "...", "met": true, "readings": {}},
  "guardrails": {"V": "0/24", "src_untouched": true, "metric_property": "...", "notes": "..."},
  "key_numbers": {}, "result_summary": "...",
  "artifacts": {"notebook": "...", "metrics_json": "...", "log": "..."},
  "notebook_errors": 0, "caveats": "..."
}
```

## Stage B - build

- Contract: the acceptance criteria and defects named in the work package. Read them in `docs/acc-crit-docdistance.md` and `docs/defects-docdistance.md` (read only); every `test:` and `repro:` line is a test to write or a check to run
- Additive policy: every number and every detail-file key that 1.1.5 produces stays bit-identical. New behaviour ships as a new function, a new field or an opt-in parameter with the default off. The registered exceptions are the `moved` flag (ACC-TRANS-77, ACC-TRANS-15..20), the statements of a `--markdown` read of a document with marker-only lines (DEF-SEGM-15), and, by the user's decision of 2026-10-10, the default reader of a file with the extension `.md`, `.markdown`, `.html` or `.htm` (ACC-SEGM-93, ACC-SEGM-94): `input_format` defaults to `auto`, which chooses the reader from the extension. With `input_format="plain"` every 1.1.5 number stays bit-identical, and `baseline.py verify` pins that reader (ACC-SEGM-95)
- Guardrail command, must report `0 difference(s)`: `CI=true .venv/bin/python reports/E13-campaign/baseline.py verify --allow moved,n_moved` (add `--grounded` when the package touches the source-conditioned path; about 3 minutes)
- New build to test: after your change and before the gates run `make install 2>&1 | tee logs/env-install.log | tail -3` once. It syncs the environment, reinstalls the package and increments the patch version in `pyproject.toml`. That increment is wanted (Star Colonel, 2026-10-09): never revert it, never pass `SKIP_VERSION_INCREMENT`. A new direct import needs its package declared under `dependencies` in `pyproject.toml` first; `make install` then installs it. Report the version after the install in `notes`
- Gates before returning, on the installed build: `make test` green, `make lint` clean (`uvx ruff@0.16.2 format --check` and `check`; run `make format` to fix formatting)
- Code rules: match the surrounding style, comment density and naming; the minimum that satisfies the items; no speculative parameter, no fallback branch for an impossible state; remove what your change orphans; syntax valid on Python 3.12
- Optimal-transport correctness: SMD is a metric only because the ground cost is a metric. State in the docstring whether a new quantity is a metric, a lower bound or a score, and never label an approximation as exact
- Every new behaviour gets a test that fails on the unchanged tree. A test that cannot fail is a defect
- Stay inside the files the package lists; a needed edit outside them is reported, not made
- Portability: nothing host-specific; OpenVINO INT8 on any CPU stays the default path

## Stage B return schema

```json
{
  "package": "...", "status": "done | partial | blocked",
  "items": [{"id": "ACC-... | DEF-...", "met": true, "evidence": "<test name or command and what it showed>"}],
  "files_changed": ["..."], "tests_added": ["..."],
  "make_test": "<N passed, M skipped>", "make_lint": "clean | <errors>",
  "baseline_verify": "<N difference(s)>",
  "out_of_scope_needs": ["..."], "notes": "..."
}
```
