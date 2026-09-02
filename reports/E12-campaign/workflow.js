export const meta = {
  name: 'docdistance-e12-campaign',
  description: 'Execute the pre-registered E12 batch (H59-H69): one opus executor per hypothesis, methodologist verify + one repair round, timing hypotheses alone, synthesis',
  phases: [
    { title: 'Execute', detail: 'nine executors in parallel (H59-H66, H69), each one notebook', model: 'opus' },
    { title: 'Verify', detail: 'methodologist verify per hypothesis, one repair round' },
    { title: 'Timing', detail: 'H67 then H68 alone on a quiet CPU', model: 'opus' },
    { title: 'Synthesize', detail: 'finding table + ledger-ready Result/Verdict text' },
  ],
}

const ROOT = '/home/lab/workspace/learning/projects/docdistance'
const LEDGER = `${ROOT}/docs/experiments/wmd-docdistance-experiments.md`
const PRIMER = `${ROOT}/reports/E12-campaign/PRIMER.md`
const EXEC_MODEL = 'opus'

const VERDICTS = ['Ships', 'Kept', 'Promoted', 'Confirmed', 'Blocked', 'Dropped', 'Refuted', 'Refuted (null)', 'Killed-at-gate']

const HYPS = [
  { n: 'H59', slug: 'structure-threshold-transfer', timing: false, notes: 'Read-only probe. Per-pair diffuse values are not in the E11 report (medians only): recompute them via the E11 notebook path (re-encode the fixture docs, opw_gap per pair) and state which pairs you rebuilt.' },
  { n: 'H60', slug: 'no-counterpart-valley', timing: false, notes: 'Kill-gate is a read of summary.cost_bands[">0.6"] on the fully-matched pairs only (gold vs gold-2, v1 vs v2, werg-a/b/c) via semantic_distance_with_details or the CLI --details-json. Deletion variants (drop 20% of B statements) are built inside the notebook from the fixture texts; never write into data/.' },
  { n: 'H61', slug: 'moved-conjunction', timing: false, notes: 'The moved definition changes ONLY inside the notebook (re-implement or monkeypatch the flag logic of _build_structural_details); src stays untouched. Build the 12->8 shrinkage synthetic locally in the notebook. Kill-gate: order_alignment on the insertion fixture must be monotone.' },
  { n: 'H62', slug: 'order-gap-length-residual', timing: false, notes: 'Entry points opw_gap and smd from docdistance.distance; re-encode the fixture docs once with the shipped encoder; the paraphrase p95 0.005 is the registered reference; k in {1,2,3,4} at head, tail and random interior positions.' },
  { n: 'H63', slug: 'semantic-threshold-band', timing: false, notes: 'INT8 = the shipped OpenVINO path (DocDistance() default); fp = TorchEncoder(device="cpu") from docdistance.encoders. Corpus-generation finding: infer only from notebook 09 recorded cell text; never open any @archive directory.' },
  { n: 'H64', slug: 'centering-vs-projection', timing: false, notes: 'Kill-gate: norm of the pooled mean of the raw statement embeddings across the fixture >= 0.15. Reproduce the naive baseline row AND the E01-H3 full-composition row before scoring the arms; report cos(uncentered PC1, mean direction).' },
  { n: 'H65', slug: 'cmr-pool-crossover', timing: false, notes: 'Every pool subset contains the two scored documents; >= 5 seeded random subsets per pool size in {2,3,4,6,8,11}; report mean and spread per pool size; the full-pool arm must reproduce E01-H3 (margin ~+18.5, V 0/24) or return Blocked.' },
  { n: 'H66', slug: 'norm-mass-marginals', timing: false, notes: 'Pre-L2 pooled norms: run the OpenVINO encoder and mean-pool the token outputs before the L2 division (encoders.py:246-267, division at 260) inside the notebook. Kill-gate first: CV of the norms < 0.03 kills; corr(norm, token count) > 0.8 kills. Masses feed ot.emd2 only at the semantic solve.' },
  { n: 'H67', slug: 'pairs-embed-memo', timing: true, notes: 'TIMING PHASE - you run alone, wall-clock is a valid measurement; check `uptime` load average is below 8 before every timed run. Memo prototype: subclass or monkeypatch DocDistance.embed_statements inside the notebook, keyed as registered (resolved path, mtime, segmentation kwargs including the fragment-filter kwargs), restoring _last_n_dropped on hits; drive the shipped CLI batch path in-process (typer CliRunner on docdistance.cli.app, or docdistance.cli._run_batch) with memo off vs on; byte-diff the JSONL streams and the details files; 3 repeats, report medians; include the disjoint-pairs control and the distance-structural rerun. No src edits - the memo ships in a follow-up if Promoted.' },
  { n: 'H68', slug: 'grounding-grid-memo', timing: true, notes: 'TIMING PHASE - you run alone; check `uptime` load average is below 8 before timing. The warm sequential baseline of 55 distance_wrt_source calls takes roughly 1-2 hours: run the notebook detached and poll the log. Memo prototype inside the notebook over (doc, source) for the embed_statements and _grounding outputs; the CLI --pairs addition to distance-wrt-source is a follow-up build, not part of this experiment. Byte-diff every memoised pair JSON (to_dict) against the one-shot distance_wrt_source output; report the cold-CLI load-amortisation line separately.' },
  { n: 'H69', slug: 'relevance-gate-form', timing: false, notes: 'Registered as a lean. Executing the registered single-phase experiment (re-derive the reranker chain on the current embeddings over the ibm fixture, persist per-statement max-r by tier in the metrics JSON, then score the three arms: no gate, shipped soft gate, hard gate swept over tau) turns it into a modelled test: report that test\'s verdict per its bar and state explicitly that the lean was executed. Report the gold-vs-Set2 max-r overlap.' },
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
    guardrails: { type: 'object', properties: { V: { type: 'string' }, smd_bit_identical: { type: ['boolean', 'string'] }, order_gap_bit_identical: { type: ['boolean', 'string'] }, metric_property: { type: 'string' }, notes: { type: 'string' } } },
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

const SYNTH_SCHEMA = {
  type: 'object', required: ['rows', 'implications', 'follow_ups'],
  properties: {
    rows: { type: 'array', items: { type: 'object', required: ['id', 'slug', 'lever', 'predicted', 'result', 'verdict', 'ledger_result', 'ledger_verdict'], properties: {
      id: { type: 'string' }, slug: { type: 'string' }, lever: { type: 'string' }, predicted: { type: 'string' }, result: { type: 'string' },
      verdict: { enum: VERDICTS }, ledger_result: { type: 'string' }, ledger_verdict: { type: 'string' }, experiment_line: { type: 'string' },
    } } },
    implications: { type: 'string' },
    follow_ups: { type: 'array', items: { type: 'string' } },
    harness_notes: { type: 'string' },
    kpi_updates: { type: 'string' },
  },
}

const files = h => ({
  id: `E12-${h.n}`,
  nb: `notebooks/experiments/E12-${h.n}-kj-${h.slug}.ipynb`,
  metrics: `reports/E12-${h.n}-${h.slug}-metrics.json`,
  log: `logs/E12-${h.n}-${h.slug}.log`,
  result: `reports/E12-campaign/results/${h.n}.json`,
})

const execPrompt = h => {
  const f = files(h)
  return `You are the EXECUTOR for hypothesis ${f.id} (${h.slug}) of the docdistance E12 experiment campaign. Repo root: ${ROOT}. Work from the repo root. Execution model: ${EXEC_MODEL} (write it into the result's "model" field).

FIRST read ${PRIMER} in full - it is the contract: environment, deliverables, rules, fixture map, verdict vocabulary, return schema. THEN read your registered record: run \`hypothesis-tools show ${LEDGER} ${f.id}\` (fallback: the \`### ${f.id}\` block in ${LEDGER}). Apply the Acceptance bar and the Kill-gate literally - never move a bar to fit a result.

Idempotency: if ${ROOT}/${f.result} exists with notebook_errors 0 and ${ROOT}/${f.nb} exists, return that JSON unchanged and stop.

Deliverables (exact names):
- notebook ${f.nb}, executed in place with 0 errors (nbconvert command in the primer; detached + poll when it may exceed 10 minutes)
- metrics JSON ${f.metrics}, written by the notebook
- result JSON ${f.result} (the return schema), written by you after the notebook executed clean
- log ${f.log}

Hypothesis-specific notes: ${h.notes}

Procedure: kill-gate section first; harness sanity check where the record names one; then the registered experiment with guardrails measured (array_equal / byte diff, V counted); conclusions cell grounded in the outputs. Build the notebook with nbformat (or write the ipynb JSON), execute with nbconvert, fix errors by editing the notebook and re-executing, then write the result JSON. No edits under src/, docs/, tests/, the ledger, or another hypothesis's files. Your final answer is the result JSON only (StructuredOutput), no prose.`
}

const verifyPrompt = (h, res) => {
  const f = files(h)
  return `You are a METHODOLOGIST verifier for ${f.id} (${h.slug}) of the docdistance E12 campaign - adversarial: your job is to find where the executor's claim does not hold. Repo root: ${ROOT}. READ-ONLY: never edit, execute or write anything; python for parsing an .ipynb is fine.

Read: ${PRIMER}; the registered record (\`hypothesis-tools show ${LEDGER} ${f.id}\`, fallback the \`### ${f.id}\` block); the executed notebook ${ROOT}/${f.nb} (parse the ipynb JSON and inspect cell OUTPUTS, not only sources); ${ROOT}/${f.metrics}; ${ROOT}/${f.log}; and the executor's result:
${JSON.stringify(res, null, 2)}

Checks: (1) the kill-gate ran first and was read as registered; (2) the acceptance bar was applied literally - a moved goalpost, a reinterpreted bar or a silently dropped guardrail is BLOCKING; (3) every number in the result appears in notebook outputs; (4) guardrails were measured, not asserted (bit-identity via array_equal or byte diff, V counted); (5) the sanity check reproduced where the record requires one; (6) 0 notebook errors and the log confirms the execution wrote the notebook; (7) the verdict label follows the closed set and its semantics in the primer; (8) the prediction check is honest on direction and magnitude; (9) nothing under src/, docs/ or the ledger was modified (git status). A verdict the evidence does not support gets corrected_verdict. verified=true only with no blocking or major finding. Return the JSON only.`
}

const repairPrompt = (h, res, ver) => {
  const f = files(h)
  return `${execPrompt(h)}

REPAIR ROUND. A methodologist verifier rejected the previous result. Previous result:
${JSON.stringify(res, null, 2)}
Verifier findings (fix every blocking and major one; the bar and kill-gate stay as registered - fix the notebook and the numbers, never the contract):
${JSON.stringify(ver, null, 2)}
Ignore the idempotency rule for this round: edit the notebook, re-execute it to 0 errors, rewrite ${f.metrics} and ${f.result}, and return the updated result JSON only.`
}

const runOne = async (h, phaseName) => {
  const f = files(h)
  let res = await agent(execPrompt(h), { label: `run:${f.id}`, phase: phaseName, schema: RESULT_SCHEMA, model: EXEC_MODEL })
  if (!res) { log(`${f.id}: executor returned nothing`); return { id: f.id, slug: h.slug, result: null, verify: null, repaired: false } }
  let ver = await agent(verifyPrompt(h, res), { label: `verify:${f.id}`, phase: 'Verify', schema: VERIFY_SCHEMA, effort: 'high' })
  let repaired = false
  if (ver && !ver.verified) {
    log(`${f.id}: verifier rejected (${ver.findings.length} findings) - repair round`)
    const res2 = await agent(repairPrompt(h, res, ver), { label: `repair:${f.id}`, phase: phaseName, schema: RESULT_SCHEMA, model: EXEC_MODEL })
    if (res2) {
      res = res2; repaired = true
      ver = await agent(verifyPrompt(h, res), { label: `reverify:${f.id}`, phase: 'Verify', schema: VERIFY_SCHEMA, effort: 'high' })
    }
  }
  log(`${f.id}: ${res.verdict} (${res.verdict_number}) - verified=${ver ? ver.verified : 'n/a'}`)
  return { id: f.id, slug: h.slug, result: res, verify: ver, repaired }
}

phase('Execute')
const PHASE1 = HYPS.filter(h => !h.timing)
const TIMING = HYPS.filter(h => h.timing)
log(`phase 1: ${PHASE1.length} executors in parallel on ${EXEC_MODEL}; timing phase: ${TIMING.map(h => h.n).join(', ')} afterwards, alone`)
const phase1 = (await parallel(PHASE1.map(h => () => runOne(h, 'Execute')))).filter(Boolean)

phase('Timing')
const timing = []
for (const h of TIMING) timing.push(await runOne(h, 'Timing'))

const all = [...phase1, ...timing]

phase('Synthesize')
const synth = await agent(
  `You are the synthesis judge of the docdistance E12 experiment campaign. Repo root: ${ROOT}. READ-ONLY (reading notebooks, metrics and result JSONs is expected; never edit anything).

Read ${PRIMER} and the \`## E12\` section of ${LEDGER} (all eleven records and the pre-registration table). Then the campaign outputs - executor results with their verifier verdicts:
${JSON.stringify(all, null, 2)}

Produce, for every one of the eleven hypotheses (a missing or unverified one gets verdict Blocked with the reason):
(a) rows for the finding table: id, slug, lever (short), predicted (the registered prediction, short), result (the measured numbers, short), verdict (closed set), and ledger-ready text - ledger_result: a Result bullet body in terse technical style (numbers inline, prediction vs observed, bar readings, guardrail readings, artifacts named by path; no em-dashes, use " - "; no full stop at the end), ledger_verdict: "<Label>; <the number that justifies it>", experiment_line: one line naming the notebook, the metrics JSON and "model: ${EXEC_MODEL}";
(b) implications: cross-hypothesis reading - anisotropy mechanism (H64, H65), threshold calibration (H59, H63), diff annotation (H60, H61), order axis (H62), transport weights (H66), runtime (H67, H68), grounding cascade (H69); where a verifier corrected a verdict, use the corrected one and say so;
(c) follow_ups: the concrete next actions each verdict implies (a recalibration that may ship, a build to promote, a new round to register);
(d) harness_notes: sanity-check failures, blocked runs, anything the executors could not reproduce;
(e) kpi_updates: which KPI rows of the ledger the results touch (reporting fidelity, batch throughput, calibration). Return the JSON only.`,
  { label: 'synthesize:E12', phase: 'Synthesize', schema: SYNTH_SCHEMA, effort: 'high' }
)

return { results: all, synthesis: synth }
