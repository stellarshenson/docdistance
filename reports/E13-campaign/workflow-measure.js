export const meta = {
  name: 'docdistance-e13-measure',
  description: 'Stage A of the E13 campaign: execute the six pre-registered hypotheses (E13-H70 to H74 transport overflow, E14-H75 centering attribution), one opus executor per notebook, methodologist verify plus one repair round',
  phases: [
    { title: 'Execute', detail: 'six executors in parallel, one notebook each', model: 'opus' },
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

const HYPS = [
  { n: 'H70', slug: 'matched-order-gap', notes: 'Kill-gate: opw_gap(B, B) on the 10 base docs. Rebuild the 280 order-preserving deletion cases exactly as the E12-H62 notebook does and reproduce its shipped reading (median 0.134 at k=2) before scoring the lever. Matched sub-sequences: scipy.optimize.linear_sum_assignment on np.minimum(cost, c); pairs at cost >= c are unmatched; keep each side in its own document order, then shipped opw_gap on the two sub-sequences. Bit-identity arm: every equal-length fully matched pair of the structure fixture, np.array_equal against shipped opw_gap.' },
  { n: 'H71', slug: 'partial-matching-map', notes: 'Verify the .bt.json statement alignment against each base doc first (the first list element is the H1 title line); if a base doc does not align one-to-one, exclude it and say so. Kill-gate on the calibration split only. Choose c on the calibration split, never on the scored split; report both directions of the 5 / 5 split. Shipped baseline labels come from DocDistance.semantic_distance_with_details on the same edited pairs. Persist the two cost populations (true paraphrase pairs, nearest non-counterpart) in the metrics JSON with their percentiles - the build stage reads the cutoff from there.' },
  { n: 'H72', slug: 'capped-ground-cost', notes: 'Sanity check first: reproduce the naive row on OpenVINO INT8 (margin +4.71, d-prime 4.68, V 0/24) within 5%. Kill-gate: share of cross-article inserted statements whose cheapest cost to the host document exceeds 0.6. The cap is ot.emd2 on np.minimum(cost, c) with the shipped uniform marginals. Triangle check over all ordered triples of the 11 summaries (495 inequalities).' },
  { n: 'H73', slug: 'nearest-source-residual', notes: 'Kill-gate: statement-level AUC of nearest-source cost, own summary statements against wergeland statements, measured against the ibm source. Reproduce the DEF-TRANS-8 reading first (10 verbatim source statements, source[0::7][:10], compute_source_conditioned: residual_a 0.958, closeness_a 0.322 with the default anisotropy pre-pass). Score the one-sided residual with and without the pre-pass and state which one each number belongs to.' },
  { n: 'H74', slug: 'order-kernel-reach', notes: 'Kill-gate: rebuild the kernel K exactly as _opw_plan does for N = M = 300 with the shipped constants and read the entries at |i - j| = 60. Embed the pooled statements once and cache the array in memory; opw_gap on 300 x 300 takes about a second. Report the displacement curve as a table and one inline plot, per seed and as medians.' },
  { b: 'E14', n: 'H75', slug: 'centering-attribution-transfer', notes: 'Kill-gate: pooled mean norm, and the full-composition margin gain over naive, on the 76 Wergeland statements. Reuse the three-arm code of notebooks/experiments/E12-H64-kj-centering-vs-projection.ipynb unchanged (arm b = center, remove PC1, restore the mean, renorm). Sanity check: the E12-H64 ibm rows on OpenVINO INT8 within 5% (naive margin +4.71, arm a +19.95, full composition +19.41). The Wergeland documents are the keys werg-a, werg-b, werg-c and werg-a~bt, werg-b~bt, werg-c~bt of data/processed/structure-fixture/statements.json; no ibm statement enters the estimation pool or the scored pairs of the second-fixture panel. V is counted per anchor over the 12 (positive, negative) orderings; the standing V 0/24 guardrail applies to the ibm sanity rows. Conditioned-path reading, no bar: the three grounded triples of reports/E13-campaign/baseline.py (gold against 1-opus, adv1-a and adv2-b with the ibm source) with the pre-pass replaced by arm a inside the notebook - the largest absolute closeness difference against the shipped pre-pass; if that needs more than wrapping all_but_the_top, report the selection-axis reading only and say so.' },
  { b: 'E14', n: 'H76', slug: 'conditioned-centering-probe', notes: 'Kill-gate: on the shipped INT8 path with the per-triple pool {document, anchor, source}, DR(D_sel) of the shipped pre-pass against the raw arm and the selection-axis violation count, both defined exactly as E04-H15 defines them in notebooks/experiments/E04-kj-source-conditioned-performance.ipynb - quote those two definitions in a markdown cell before any number is computed. D_sel needs embeddings only: call shipped compute_source_conditioned without the reranker and entailment arrays. Arm a = center the pooled triple, re-L2-normalise, then call compute_source_conditioned with anisotropy=False (the way notebooks/experiments/E14-H75-kj-centering-attribution-transfer.ipynb builds its conditioned-path reading). The standing V 0/24 guardrail is read on the ibm symmetric panel as in the primer. Record the decision-rule outcome for ACC-GEOM-21 in key_numbers as geom21_changes_conditioned_prepass (true only on Refuted).' },  { b: 'E14', n: 'H77', slug: 'kernel-pca-prepass', notes: 'The bandwidth is never optimised: gamma = m / (median squared chordal distance of the pool) for the five registered values m in {0.25, 1, 4, 16, 64}, computed per pool before any panel number. Build the empirical kernel map (Gram matrix under exp(-gamma * (2 - 2cos)), double-centered, eigendecomposed with numpy.linalg.eigh, coordinates = eigenvectors times the square root of the non-negative eigenvalues, top-1 coordinate dropped, rows re-L2-normalised). Harness check first: the same map with the linear kernel reproduces the shipped all_but_the_top(k=1) pairwise costs to 1e-5. Kill-gate per bandwidth: Spearman of the pairwise statement costs against the shipped pre-pass on the ibm symmetric pool and on each of the 10 ibm conditioned triples; a bandwidth at or above 0.995 on all of them is a monotone copy and is not scored. Every bandwidth that passes is scored on all three panels, and the table shows all five rows with cost Spearman, top-eigenvalue share and bootstrap stability over 50 resamples (fixed seed) - never only the best row. Reuse the conditioned panel code of notebooks/experiments/E14-H76-kj-conditioned-centering-probe.ipynb and the Wergeland panel code of notebooks/experiments/E14-H75-kj-centering-attribution-transfer.ipynb when those notebooks exist; otherwise build the panels from the E04-H15 definitions in notebooks/experiments/E04-kj-source-conditioned-performance.ipynb.' },
  { b: 'E15', n: 'H78', slug: 'raw-nearest-source-count', notes: 'Kill-gate: AUC of nearest-source cost, own gold-tier statements against the foreign set, on each calibration half, raw embeddings. Reuse the fixture code of notebooks/experiments/E13-H73-kj-nearest-source-residual.ipynb (foreign set = Wergeland statements of 5+ words with the HTML provenance comment removed, 38 statements; DEF-TRANS-8 extract source[0::7][:10]). Gold tier = the 7 ibm summaries that are not adv1-a, adv1-b, adv2-a, adv2-b; split by document 4 / 3 with the anchor exec-summary-gold-opus-4-6.md in half A, the other six in sorted file-name order alternating B, A, B, A, B, A - write the split in the configuration cell before any number. The cutoff is chosen on the calibration half only and never on a scored half; the second-article clause uses the mean of the two cutoffs with no refit. State for every number whether source statements under 5 words were kept (registered reading: kept, the shipped default reader) or removed (reported reading).' },
  { b: 'E15', n: 'H79', slug: 'matching-tie-break', notes: 'Kill-gate first: reproduce the E13-H70 third-pass readings without the tie-break at c = 0.6 (survivors recovered 88/120, matched gap above shipped in 21/120), then show one exact bitwise cost tie inside a group of repeated statements. Import or copy the case builders of notebooks/experiments/E13-H70-kj-matched-order-gap.ipynb and notebooks/experiments/E13-H71-kj-partial-matching-map.ipynb unchanged, including seeds; say so in the notebook and verify by reproducing their recorded readings (280/280, AUC 0.999903, 1847/1847 at c = 0.6; macro-F1 0.9674 half A and 0.9772 half B at c = 0.325). The tie-break is one line: assignment cost = np.minimum(cost, c) + eps * abs(i / n - j / m) with i, j the statement positions; pairs whose untie-broken cost is >= c stay unmatched. Clause (3) is np.array_equal, never allclose. Persist in the metrics JSON the constants a build needs: c, eps, and the count of fully matched equal-length pairs at c = 0.325.' },
  { b: 'E15', n: 'H80', slug: 'moved-on-matched-pairs', notes: 'Harness check first: reproduce the E12-H61 pool reading (13675 flagged true relocations over 1680 permutations, 0 lost under the conjunction) with the builders of notebooks/experiments/E12-H61-kj-moved-conjunction.ipynb copied unchanged, including the ground-truth relocation rule. Then the kill-gate on the 280 order-preserving deletion cases built by the code of notebooks/experiments/E13-H70-kj-matched-order-gap.ipynb (copied unchanged, seeds included; verify by reproducing 280/280 at or below 0.005 under the matched order gap). The four arms are computed in the notebook from the shipped per-statement records (src/docdistance/pipeline.py, structural details: displacement, displacement_norm, no_counterpart) plus the matching; the shipped rule that a statement with no_counterpart reads moved false stays in every arm. Matching: scipy linear_sum_assignment on np.minimum(cost, c) + eps * abs(i / n - j / m) with c = 0.325, eps = 1e-6; a pair whose cost without the tie-break is >= c is unmatched. Lever: matched AND matched_displacement != 0 AND abs(matched_displacement / n_matched) > MOVED_TOLERANCE, ranks taken inside the matched sub-sequences in document order. Clause (2) is np.array_equal over the per-statement flag arrays, never a count comparison. Reworded reorders: A = the base statements, B = the back-translated statements (<base>~bt in data/processed/structure-fixture/statements.json) in the permuted order; check the 1:1 statement alignment of every ~bt key before use and report any base left out. Ground truth for a relocation is the E12-H61 rule applied to the permutation. Persist in the metrics JSON, per arm: flagged kept and flagged removed statements on the 280 deletion cases, cases with at least one flag, the pool equality count, the reworded retention numerator and denominator.' },
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
