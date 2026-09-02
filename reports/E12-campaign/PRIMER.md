# E12 campaign - executor primer

One executor agent per hypothesis. The agent builds and executes one notebook, writes one metrics JSON and one result JSON, and returns the result. The registered record in the ledger is the contract: bars and kill-gates are applied literally, never adjusted to a result.

## Where the contract lives

- Ledger: `docs/experiments/wmd-docdistance-experiments.md`, section `## E12` - one `### E12-H<n>` block per hypothesis with Hypothesis / Lever / Mechanism / Prediction / Acceptance bar / Kill-gate / Evidence / Experiment
- Print one record: `hypothesis-tools show docs/experiments/wmd-docdistance-experiments.md E12-H<n>` (or read the block)
- Standing guardrails on every hypothesis: `V` stays `0/24`, default-path `smd` / `order_gap` bit-identical unless the hypothesis is a registered recalibration, metric property untouched

## Environment (verified 2026-09-01)

- Python `/opt/conda/bin/python` - docdistance 1.1.5, POT 0.9.7, openvino, torch; Jupyter kernel `python3`; run everything from the repo root `/home/lab/workspace/learning/projects/docdistance`
- Models resolve offline through `docdistance.json` at the repo root (mmBERT OpenVINO INT8, bge-reranker INT8, mDeBERTa NLI INT8, HF cache). `DocDistance()` load + one embed 17 s, grounding models 14 s
- CPU 64 threads, shared by every phase-1 executor at once - wall-clock inside phase 1 is not a measurement. H67 and H68 run alone in the timing phase
- GPUs are occupied by foreign jobs. Stay on CPU. H63's fp backbone arm: `TorchEncoder(device="cpu")` from `docdistance.encoders` is acceptable
- Never read `**/@archive` or `**/.ipynb_checkpoints`
- Long executions run detached: `setsid nohup jupyter nbconvert ... > logs/<name>.log 2>&1 &`, then poll the log (`sleep 60; tail -5 logs/<name>.log`) until `Writing ... bytes` appears; a Bash call is capped at 10 minutes, the notebook is not

## Deliverables (named by id)

- Notebook `notebooks/experiments/E12-H<n>-kj-<slug>.ipynb`, executed in place, 0 errors:
  `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/experiments/E12-H<n>-kj-<slug>.ipynb 2>&1 | tee logs/E12-H<n>-<slug>.log`
- Metrics JSON `reports/E12-H<n>-<slug>-metrics.json`, written by the notebook: every measured number, bar readings, guardrail readings
- Result JSON `reports/E12-campaign/results/H<n>.json` - the return schema below, identical to what the agent returns
- Log `logs/E12-H<n>-<slug>.log` (the orchestrator adds the `logs/README.md` line)
- Idempotency: if `reports/E12-campaign/results/H<n>.json` already exists with `notebook_errors: 0` and its notebook exists, return that JSON unchanged - do not re-run

## Rules

- No edits under `src/`, `docs/`, `tests/`, the ledger, or another hypothesis's files. Every variant is prototyped inside the notebook: import the shipped functions from `docdistance.distance` / `docdistance.pipeline` / `docdistance.encoders` and wrap, monkeypatch or re-implement them there
- Notebook conventions - `datascience:notebook-standards` (`/home/lab/.claude/plugins/cache/stellarshenson-marketplace/datascience/1.7.14/skills/notebook-standards/SKILL.md`): header (title, approach steps), environment cell first (CPU pin note when no GPU), all imports in one cell, seed, configuration cell ending in a rich summary, a 1-2 sentence overview under every `##`, rich output, plots inline only (no savefig), conclusions cell grounded in the outputs above. Mirror `notebooks/experiments/E10-kj-structure-sensitive-distance.ipynb` for shape and `E01-kj-wmd-contrast-hypotheses.ipynb` (cells 13-14: `smd`, `evaluate`, `bench`) for the metric panel
- Kill-gate first, as its own section. A killed gate ends the run with `Killed-at-gate` and the gate reading; the notebook still executes so the reading is persisted
- Harness sanity check before any arm is scored, where the record names one: H64 / H65 / H66 reproduce the naive baseline row on the ibm 11-summary fixture (margin +4.58, R 1.68, DR 0.0854, d' 4.60, V 0/24); H65's full-pool arm reproduces E01-H3 (margin ~+18.5, V 0/24); H61's reversal control reads 12/12. A failed sanity check returns `Blocked` with the discrepancy, never a verdict
- Guardrails are measured, not asserted: bit-identity by `np.array_equal` or a byte diff, never `allclose`; `V` counted; the metric property checked where the record names it
- Numbers in the result JSON must appear verbatim in notebook outputs

## Fixture map (paths verified)

- ibm 11-summary fixture: `data/interim/exec-summaries/ibm-ai-adoption/summaries/*.md` (11 files, live opus-4-6 generation; anchor `exec-summary-gold-opus-4-6.md`); source `data/interim/exec-summaries/ibm-ai-adoption/source/source-article.md`; tier definitions and panel code in the E01 notebook
- wergeland source: `data/interim/ai-society-wergeland/source/source-article.md`
- structure fixture: `data/processed/structure-fixture/{meta,pairs,reorder_pool,statements}.json` (swap_ks 0-40, 14 seeds per k, 10 base docs); built by `notebooks/11-kj-structure-fixture.ipynb`
- paraphrase back-translations: `data/interim/structure-paraphrase/*.bt.json` (10 docs)
- E11 decision report: `reports/E11-structure-sota-decision.json` - keys `scramble.{ibm,wergeland}.{H44,H55}`, `diffuse.{ibm,wergeland}` (medians only, ibm n=55), `paraphrase.by_kind`, `shift`, `triangle.{H44,H55}`, `config`; produced by `notebooks/experiments/E11-kj-structure-sota-decision.ipynb`
- worked example (H61): `docs/example-map-interpretations.md` doc-a / doc-b, 6 statements
- quantization fidelity (H63): `reports/02-mmbert-quantization-metrics.json`
- CLI: `docs/cli-reference.md`, `.claude/skills/docdistance/SKILL.md`; `--pairs` TSV is `a-path<TAB>b-path[<TAB>id]` per line (`src/docdistance/cli.py:162`); API `docs/api-reference.md`
- Calibration constants: `src/docdistance/distance.py:30,40` (0.77, 0.90), `src/docdistance/pipeline.py:40,47,53` (changed cost, 0.6, 0.05)

## Verdict vocabulary (closed set - `hypothesis-tools check` rejects anything else)

`Ships` / `Kept` / `Promoted` / `Confirmed` / `Blocked` / `Dropped` / `Refuted` / `Refuted (null)` / `Killed-at-gate`. For E12: a probe whose prediction held and bar was met is `Confirmed`; a toggle or build that met its bar is `Promoted` (shipping is a follow-up decision); bar not met with the prediction falsified is `Refuted`; an effect indistinguishable from baseline is `Refuted (null)`; a precondition or harness failure is `Blocked`. A pre-registered dual verdict (H62) reports whichever branch the numbers land in as `Confirmed` and names the branch

## Return schema

```json
{
  "id": "E12-H<n>", "slug": "...", "model": "<execution model>",
  "verdict": "<one of the closed set>", "verdict_number": "<the number that justifies it>",
  "kill_gate": {"ran": true, "passed": true, "reading": "..."},
  "sanity_check": {"required": true, "passed": true, "reading": "..."},
  "prediction_check": {"predicted": "...", "observed": "...", "direction_matched": true},
  "acceptance": {"bar": "...", "met": true, "readings": {"...": 0.0}},
  "guardrails": {"V": "0/24", "smd_bit_identical": true, "order_gap_bit_identical": true, "metric_property": "...", "notes": "..."},
  "key_numbers": {"...": 0.0},
  "result_summary": "2-4 sentences with the numbers, ledger-ready, terse technical style, no em-dashes",
  "artifacts": {"notebook": "notebooks/experiments/...", "metrics_json": "reports/...", "log": "logs/..."},
  "notebook_errors": 0,
  "caveats": "..."
}
```
