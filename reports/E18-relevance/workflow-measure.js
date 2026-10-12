export const meta = {
  name: 'docdistance-e18-measure',
  description: 'Batch E18 of docdistance: execute the six pre-registered hypotheses on the relevance reading for a single brief in two notebooks, one executor and one independent verifier per notebook, one repair round',
  phases: [
    { title: 'Execute', detail: 'two executors in parallel, one notebook each' },
    { title: 'Verify', detail: 'methodologist verifier per notebook, one repair round' },
  ],
}

// args: { only: ['A'] } runs a subset of the notebooks; the default is both
const ROOT = '/home/lab/workspace/learning/projects/docdistance'
const LEDGER = `${ROOT}/docs/experiments/wmd-docdistance-experiments.md`
const PRIMER = `${ROOT}/reports/E18-relevance/PRIMER.md`
const EXEC_AGENT = 'senior-backend-engineer'
const VERIFY_AGENT = 'devils-advocate:adversarial-reviewer'
const VERDICTS = ['Confirmed', 'Refuted', 'Killed-at-gate']

const NOTEBOOKS = [
  { key: 'A', ids: ['E18-H87', 'E18-H91'], nb: 'notebooks/experiments/E18-kj-reranker-relevance.ipynb', metrics: 'reports/E18-reranker-relevance-metrics.json', log: 'logs/E18-reranker-relevance.log', result: 'reports/E18-relevance/results/A.json' },
  { key: 'B', ids: ['E18-H88', 'E18-H89', 'E18-H90', 'E18-H92'], nb: 'notebooks/experiments/E18-kj-contrast-replication.ipynb', metrics: 'reports/E18-contrast-replication-metrics.json', log: 'logs/E18-contrast-replication.log', result: 'reports/E18-relevance/results/B.json' },
]

const RESULT_SCHEMA = {
  type: 'object',
  required: ['key', 'notebook', 'metrics_json', 'log', 'notebook_errors', 'harness_check', 'hypotheses', 'deviations', 'observations', 'not_verified'],
  properties: {
    key: { type: 'string' }, notebook: { type: 'string' }, metrics_json: { type: 'string' }, log: { type: 'string' },
    notebook_errors: { type: 'integer' },
    harness_check: { type: 'object', required: ['passed', 'largest_difference'], properties: { passed: { type: 'boolean' }, largest_difference: { type: 'number' }, reading: { type: 'string' } } },
    hypotheses: { type: 'array', items: { type: 'object', required: ['id', 'gate', 'clauses', 'verdict_by_bar', 'key_numbers'], properties: {
      id: { type: 'string' },
      gate: { type: 'object', required: ['reading', 'passed'], properties: { reading: { type: 'string' }, passed: { type: 'boolean' } } },
      clauses: { type: 'array', items: { type: 'object', required: ['name', 'value', 'met'], properties: { name: { type: 'string' }, value: { type: 'number' }, met: { type: 'boolean' } } } },
      verdict_by_bar: { enum: VERDICTS }, key_numbers: { type: 'string' },
    } } },
    per_brief: { type: 'array', items: { type: 'object' } },
    deviations: { type: 'array', items: { type: 'string' } },
    observations: { type: 'array', items: { type: 'string' } },
    not_verified: { type: 'array', items: { type: 'string' } },
  },
}

const VERIFY_SCHEMA = {
  type: 'object', required: ['key', 'verified', 'findings'],
  properties: {
    key: { type: 'string' }, verified: { type: 'boolean' },
    corrected: { type: 'array', items: { type: 'object', required: ['id', 'verdict', 'reason'], properties: { id: { type: 'string' }, verdict: { enum: VERDICTS }, reason: { type: 'string' } } } },
    recomputed: { type: 'string' },
    findings: { type: 'array', items: { type: 'object', required: ['severity', 'claim', 'evidence'], properties: { severity: { enum: ['blocking', 'major', 'minor'] }, claim: { type: 'string' }, evidence: { type: 'string' } } } },
    notes: { type: 'string' },
  },
}

const execPrompt = n => `You are the EXECUTOR of notebook ${n.key} of batch E18 of the docdistance project (hypotheses ${n.ids.join(', ')}). Repository root: ${ROOT}. Work from the repository root.

FIRST read ${PRIMER} in full: environment, hard rules, fixture, shared definitions, the section for notebook ${n.key}, what a notebook holds, and the result you return. THEN read the registered records: \`hypothesis-tools show ${LEDGER} <id>\` for ${n.ids.join(', ')}, and the batch opening under the heading \`## E18 - relevance-single-brief\` of ${LEDGER}. Apply every acceptance bar and kill-gate literally. Never move a bar, a label or a fixture file to fit a result.

Idempotency: if ${ROOT}/${n.result} exists with notebook_errors 0 and ${ROOT}/${n.nb} exists, return that JSON unchanged and stop.

Deliverables (exact names):
- notebook ${n.nb}, executed in place with 0 error outputs by the nbconvert command of the primer
- metrics file ${n.metrics}, written by the notebook
- log ${n.log}
- result file ${n.result}, written by you after the notebook executed without errors

Procedure: fixture hash check, harness check on briefs 01 to 12 against reports/E17-relevance-map-metrics.json, gates printed with their numbers, then the registered readings, then the conclusions cell grounded in the outputs. Build the notebook with nbformat from the cells of the template notebook named in the primer, execute it, fix errors by editing the notebook and executing it again, then write the result file. Before you report, recompute two headline numbers of your notebook with a separate scratch script that shares no function with the notebook and state the differences. Your final answer is the result JSON only.`

const verifyPrompt = (n, res) => `Lens: methodologist. Target: the executed notebook ${n.key} of batch E18 of the docdistance project (hypotheses ${n.ids.join(', ')}). Your job is to find where the executor's claim does not hold. Repository root: ${ROOT}. READ-ONLY inside the repository: never edit a file there and never execute the notebook. You may run python that reads files, and you may write scratch scripts under /tmp/claude-1000/-home-lab-workspace-learning-projects-docdistance/48fccd81-2e03-4952-88be-100e909f2767/scratchpad/e18-verify/ only. No network request (HF_HUB_OFFLINE=1, CI=true), no package install.

Read: ${PRIMER}; the registered records (\`hypothesis-tools show ${LEDGER} <id>\` for ${n.ids.join(', ')}) and the batch opening under \`## E18 - relevance-single-brief\` of ${LEDGER}; the executed notebook ${ROOT}/${n.nb} (parse the ipynb JSON and inspect the cell OUTPUTS, not only the sources); ${ROOT}/${n.metrics}; ${ROOT}/${n.log}; and the executor's result:
${JSON.stringify(res, null, 2)}

Checks: (1) the fixture hash check and the harness check ran before any new brief was scored and passed; (2) each gate was read as registered, before its bar; (3) each acceptance bar was applied literally - a moved bar, a reinterpreted clause, a changed label rule, a dropped control clause or a parameter chosen on the scored data is BLOCKING; (4) every number of the result appears in the notebook outputs and in the metrics file; (5) recompute with your own code, sharing nothing with the notebook, the baseline AUC of briefs 13 to 24 and the headline score of at least two hypotheses (read the fixture through the package with \`${ROOT}/.venv/bin/python\`, or use the stored arrays where the primer names them) and compare with the notebook; (6) the headline set is briefs 13 to 24 and readings on briefs 01 to 12 are labelled as seen data; (7) the random draws use the registered generator and order; (8) 0 error outputs and the log shows that the execution wrote the notebook; (9) each verdict follows from its clauses; (10) the conclusions cell claims nothing the outputs do not show; (11) \`git status --short\` shows no change by this notebook outside its own files. A verdict the evidence does not support goes into \`corrected\`. verified=true only with no blocking and no major finding. Return the JSON only.`

const repairPrompt = (n, res, ver) => `${execPrompt(n)}

REPAIR ROUND. A verifier rejected the previous result. Previous result:
${JSON.stringify(res, null, 2)}
Verifier findings (answer every blocking and major one; the bars, the gates, the labels and the fixture stay as registered - correct the notebook and the numbers, never the contract):
${JSON.stringify(ver, null, 2)}
Ignore the idempotency rule in this round: edit the notebook, execute it again to 0 error outputs, rewrite ${n.metrics} and ${n.result}, and return the updated result JSON only.`

const runOne = async n => {
  let res = await agent(execPrompt(n), { label: `run:${n.key}`, phase: 'Execute', schema: RESULT_SCHEMA, agentType: EXEC_AGENT })
  if (!res) { log(`${n.key}: executor returned nothing`); return { key: n.key, result: null, verify: null, repaired: false } }
  let ver = await agent(verifyPrompt(n, res), { label: `verify:${n.key}`, phase: 'Verify', schema: VERIFY_SCHEMA, effort: 'high', agentType: VERIFY_AGENT })
  let repaired = false
  if (ver && !ver.verified) {
    log(`${n.key}: verifier rejected (${ver.findings.length} findings) - repair round`)
    const res2 = await agent(repairPrompt(n, res, ver), { label: `repair:${n.key}`, phase: 'Execute', schema: RESULT_SCHEMA, agentType: EXEC_AGENT })
    if (res2) {
      res = res2; repaired = true
      ver = await agent(verifyPrompt(n, res), { label: `reverify:${n.key}`, phase: 'Verify', schema: VERIFY_SCHEMA, effort: 'high', agentType: VERIFY_AGENT })
    }
  }
  log(`${n.key}: ${res.hypotheses.map(h => `${h.id} ${h.verdict_by_bar}`).join(', ')} - verified=${ver ? ver.verified : 'n/a'}`)
  return { key: n.key, result: res, verify: ver, repaired }
}

phase('Execute')
const selected = args && args.only ? NOTEBOOKS.filter(n => args.only.includes(n.key)) : NOTEBOOKS
log(`E18: ${selected.length} notebook(s), executor then verifier each`)
const results = (await pipeline(selected, n => runOne(n))).filter(Boolean)
return { results }
