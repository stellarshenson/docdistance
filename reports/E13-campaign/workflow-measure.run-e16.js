export const meta = {
  name: 'docdistance-e13-measure',
  description: 'Stage A contract of the E13 campaign applied to E16-H81 (reader chosen by content as a candidate default): one opus executor builds and executes the notebook, methodologist verify plus one repair round',
  phases: [
    { title: 'Execute', detail: 'one executor, one notebook', model: 'opus' },
    { title: 'Verify', detail: 'methodologist verify per hypothesis, one repair round' },
  ],
}

const ROOT = '/home/lab/workspace/learning/projects/docdistance'
const LEDGER = `${ROOT}/docs/experiments/wmd-docdistance-experiments.md`
const PRIMER = `${ROOT}/reports/E13-campaign/PRIMER.md`
const EXEC_MODEL = 'opus'
const EXEC_AGENT = 'senior-backend-engineer'
const VERIFY_AGENT = 'devils-advocate:adversarial-reviewer'

const VERDICTS = ['Ships', 'Kept', 'Promoted', 'Confirmed', 'Blocked', 'Dropped', 'Refuted', 'Refuted (null)', 'Killed-at-gate']

// Generated from workflow-measure.js for E16-H81 (2026-10-10). Do not edit: change workflow-measure.js or the entry below and regenerate.
const HYPS = [
  {"b": "E16", "n": "H81", "slug": "autodetected-reader-default", "notes": "Run on the installed build: the lever is the option input_format of the package input-format (check first that `DocDistance.embed_statements` takes `input_format` and that `docdistance.encoders` holds `detect_format` and `html_to_markdown`; if not, stop and report - never run `make install`). CI=true and HF_HUB_OFFLINE=1 for every command; no network request. Order: harness check, kill-gate, arms. Harness check: the plain arm (no input_format) reproduces the 55 stored 1.1.5 distances of reports/E13-campaign/baseline/ and the band edges 0.7474 and 0.7944 of E12-H63 within 0.001; reuse the fixture code of notebooks/experiments/E12-H63-kj-semantic-threshold-band.ipynb for the tiers, the reference document and the closeness. HTML renders: markdown_it (`MarkdownIt('commonmark').enable('table')`) on each of the 12 ibm documents, wrapped in `<!doctype html><html><body>...</body></html>`, written to data/interim/E16-H81/html/<name>.html. Plain controls: the embedded statements of the markdown reader joined by blank lines, written to data/interim/E16-H81/plain/<name>.txt. One DocDistance instance on the openvino backend with the memo on. For clause (4) compare the embedded statement lists with `docdistance.encoders.html_to_markdown`, `split_markdown` and the instance's segmenter, and read the SMD through the public call. A wrong detection, an unequal twin or an empty band is a reading: do not change `detect_format`, a constant or a threshold. Copy `block_sha256` and `registered_at_utc` from reports/E13-campaign/results/E16-H81-registration.json into the metrics JSON. The readings without a bar are tables in the notebook and keys in the metrics JSON; state where DEFAULT_THRESHOLD 0.77 lies against the band under auto, and the midpoint of that band."},
]

const RESULT_SCHEMA = {
  type: 'object',
  required: ['id', 'slug', 'model', 'verdict', 'verdict_number', 'kill_gate', 'acceptance', 'guardrails', 'result_summary', 'artifacts', 'notebook_errors'],
  properties: {
    id: { type: 'string' }, slug: { type: 'string' }, model: { type: 'string' },
    verdict: { enum: VERDICTS }, verdict_number: { type: 'string' },
    kill_gate: { type: 'object', required: ['ran', 'passed', 'reading'], properties: { ran: { type: 'boolean' }, passed: { type: 'boolean' }, reading: { type: 'string' } } },
    sanity_check: { type: 'object', properties: { required: { type: 'boolean' }, passed: { type: 'boolean' }, reading: { type: 'string' } } },
    prediction_check: { type: 'object', properties: { predicted: { type: 'string' }, observed: { type: 'string' }, direction_matched: { type: 'boolean' } } },
    acceptance: { type: 'object', required: ['bar', 'met', 'readings'], properties: { bar: { type: 'string' }, met: { type: 'boolean' }, readings: { type: 'object' } } },
    guardrails: { type: 'object', properties: { V: { type: 'string' }, src_untouched: { type: ['boolean', 'string'] }, metric_property: { type: 'string' }, notes: { type: 'string' } } },
    key_numbers: { type: 'object' },
    result_summary: { type: 'string' },
    artifacts: { type: 'object', required: ['notebook', 'metrics_json', 'log'], properties: { notebook: { type: 'string' }, metrics_json: { type: 'string' }, log: { type: 'string' } } },
    notebook_errors: { type: 'integer' },
    caveats: { type: 'string' },
  },
}

const VERIFY_SCHEMA = {
  type: 'object', required: ['id', 'verified', 'findings'],
  properties: {
    id: { type: 'string' }, verified: { type: 'boolean' },
    corrected_verdict: { type: 'string' },
    findings: { type: 'array', items: { type: 'object', required: ['severity', 'claim', 'evidence'], properties: { severity: { enum: ['blocking', 'major', 'minor'] }, claim: { type: 'string' }, evidence: { type: 'string' } } } },
    notes: { type: 'string' },
  },
}

const files = h => ({
  id: `${h.b || 'E13'}-${h.n}`,
  nb: `notebooks/experiments/${h.b || 'E13'}-${h.n}-kj-${h.slug}.ipynb`,
  metrics: `reports/${h.b || 'E13'}-${h.n}-${h.slug}-metrics.json`,
  log: `logs/${h.b || 'E13'}-${h.n}-${h.slug}.log`,
  result: `reports/E13-campaign/results/${h.n}.json`,
})

const execPrompt = h => {
  const f = files(h)
  return `You are the EXECUTOR for hypothesis ${f.id} (${h.slug}) of the docdistance E13 experiment campaign, stage A (measure). Repo root: ${ROOT}. Work from the repo root. Execution model: ${EXEC_MODEL} (write it into the result's "model" field).

FIRST read ${PRIMER} in full - environment, hard rules, stage A contract, fixture map, verdict vocabulary, return schema. THEN read your registered record: run \`hypothesis-tools show ${LEDGER} ${f.id}\` (fallback: the \`### ${f.id}\` block in ${LEDGER}). Apply the Acceptance bar and the Kill-gate literally - never move a bar to fit a result.

Idempotency: if ${ROOT}/${f.result} exists with notebook_errors 0 and ${ROOT}/${f.nb} exists, return that JSON unchanged and stop.

Deliverables (exact names):
- notebook ${f.nb}, executed in place with 0 errors (the nbconvert command is in the primer; detached plus polling when it may exceed 10 minutes)
- metrics JSON ${f.metrics}, written by the notebook
- result JSON ${f.result} (the stage A return schema), written by you after the notebook executed clean
- log ${f.log}

Hypothesis-specific notes: ${h.notes}

Procedure: kill-gate section first; harness sanity check where the record or the notes name one; then the registered experiment with guardrails measured; conclusions cell grounded in the outputs. Build the notebook with nbformat, execute it, fix errors by editing the notebook and re-executing, then write the result JSON. No edits under src/, docs/, tests/, the ledger, the trackers, or another hypothesis's files. Your final answer is the result JSON only, no prose.`
}

const verifyPrompt = (h, res) => {
  const f = files(h)
  return `Lens: methodologist. Target: the executed experiment ${f.id} (${h.slug}) of the docdistance E13 campaign. Your job is to find where the executor's claim does not hold. Repo root: ${ROOT}. READ-ONLY: never edit, execute a notebook or write a file; python for parsing an .ipynb is fine.

Read: ${PRIMER}; the registered record (\`hypothesis-tools show ${LEDGER} ${f.id}\`, fallback the \`### ${f.id}\` block); the executed notebook ${ROOT}/${f.nb} (parse the ipynb JSON and inspect cell OUTPUTS, not only sources); ${ROOT}/${f.metrics}; ${ROOT}/${f.log}; and the executor's result:
${JSON.stringify(res, null, 2)}

Checks: (1) the kill-gate ran first and was read as registered; (2) the acceptance bar was applied literally - a moved goalpost, a reinterpreted bar or a silently dropped guardrail is BLOCKING; (3) every number in the result appears in notebook outputs; (4) guardrails were measured, not asserted; (5) the sanity check reproduced where one is required; (6) 0 notebook errors and the log confirms the execution wrote the notebook; (7) the verdict label follows the closed set and its semantics in the primer; (8) the prediction check is honest on direction and magnitude; (9) a cutoff or threshold was never tuned on the data it is scored on; (10) nothing under src/, docs/, tests/, the ledger or the trackers was modified by this hypothesis (git status, git diff --stat). A verdict the evidence does not support gets corrected_verdict. verified=true only with no blocking or major finding. Return the JSON only.`
}

const repairPrompt = (h, res, ver) => {
  const f = files(h)
  return `${execPrompt(h)}

REPAIR ROUND. A methodologist verifier rejected the previous result. Previous result:
${JSON.stringify(res, null, 2)}
Verifier findings (fix every blocking and major one; the bar and the kill-gate stay as registered - fix the notebook and the numbers, never the contract):
${JSON.stringify(ver, null, 2)}
Ignore the idempotency rule for this round: edit the notebook, re-execute it to 0 errors, rewrite ${f.metrics} and ${f.result}, and return the updated result JSON only.`
}

const runOne = async h => {
  const f = files(h)
  let res = await agent(execPrompt(h), { label: `run:${f.id}`, phase: 'Execute', schema: RESULT_SCHEMA, model: EXEC_MODEL, agentType: EXEC_AGENT })
  if (!res) { log(`${f.id}: executor returned nothing`); return { id: f.id, slug: h.slug, result: null, verify: null, repaired: false } }
  let ver = await agent(verifyPrompt(h, res), { label: `verify:${f.id}`, phase: 'Verify', schema: VERIFY_SCHEMA, effort: 'high', agentType: VERIFY_AGENT })
  let repaired = false
  if (ver && !ver.verified) {
    log(`${f.id}: verifier rejected (${ver.findings.length} findings) - repair round`)
    const res2 = await agent(repairPrompt(h, res, ver), { label: `repair:${f.id}`, phase: 'Execute', schema: RESULT_SCHEMA, model: EXEC_MODEL, agentType: EXEC_AGENT })
    if (res2) {
      res = res2; repaired = true
      ver = await agent(verifyPrompt(h, res), { label: `reverify:${f.id}`, phase: 'Verify', schema: VERIFY_SCHEMA, effort: 'high', agentType: VERIFY_AGENT })
    }
  }
  log(`${f.id}: ${res.verdict} (${res.verdict_number}) - verified=${ver ? ver.verified : 'n/a'}`)
  return { id: f.id, slug: h.slug, result: res, verify: ver, repaired }
}

phase('Execute')
log(`stage A: ${(args && args.only ? args.only.length : HYPS.length)} executors in parallel on ${EXEC_MODEL}`)
// args.only: optional list of hypothesis numbers (['H76']) to run a subset; the default is every entry of HYPS
const selected = args && args.only ? HYPS.filter(h => args.only.includes(h.n)) : HYPS
const results = (await pipeline(selected, h => runOne(h))).filter(Boolean)
return { results }
