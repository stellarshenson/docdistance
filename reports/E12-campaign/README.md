# E12 campaign - run record

Execution of the pre-registered E12 batch (H59-H69 in `docs/experiments/wmd-docdistance-experiments.md`, section `## E12`) as a dynamic Workflow.

- **Launched** - 2026-09-01 21:50 local, run id `wf_08ea9581-df7`, executors on opus, verify and synthesis on the session model
- **Script** - `reports/E12-campaign/workflow.js` (phases Execute → Verify → Timing → Synthesize)
- **Primer** - `reports/E12-campaign/PRIMER.md` - the executor contract every agent reads first
- **Phase outputs** - `reports/E12-campaign/results/H<n>.json` (one per hypothesis, the return schema), notebooks `notebooks/experiments/E12-H<n>-kj-<slug>.ipynb`, metrics `reports/E12-H<n>-<slug>-metrics.json`, logs `logs/E12-H<n>-<slug>.log`
- **Transcripts** - `~/.claude/projects/-home-lab-workspace-learning-projects-docdistance/48fccd81-2e03-4952-88be-100e909f2767/subagents/workflows/wf_08ea9581-df7/` (`journal.jsonl` holds every agent's return value)

## Resume

- **Same session** - `Workflow({scriptPath: "reports/E12-campaign/workflow.js", resumeFromRunId: "wf_08ea9581-df7"})`; completed agents return cached results
- **After a container restart** - re-run `Workflow({scriptPath: "reports/E12-campaign/workflow.js"})`; every executor is idempotent against `results/H<n>.json` (a result with `notebook_errors: 0` and an existing notebook is returned unchanged), so only unfinished hypotheses re-run
- **Single hypothesis by hand** - spawn one agent with the executor prompt from `workflow.js` (`execPrompt`) for that id; the primer and the ledger record are the whole priming

## After completion

1. Read the synthesis (`journal.jsonl`, last entry) - the finding table and the ledger-ready Result / Verdict text per hypothesis
2. Record Result and Verdict under each `### E12-H<n>` block in the ledger, add the `model: opus` experiment line, fill the at-a-glance and per-batch results tables; `hypothesis-tools check` must pass with no new errors
3. Add the eleven `logs/README.md` lines, journal the outcome, show the finding table
