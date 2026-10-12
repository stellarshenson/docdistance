export const meta = {
  name: 'docdistance-relevance-review',
  description: 'Adversarial review of the relevance reading of the semantic map and of the experiment batches E17 to E19 behind it, by the architect, bug-hunter and data-scientist lenses; two consecutive clean rounds end it (at most 3 rounds)',
  phases: [
    { title: 'Review', detail: 'three lenses in parallel per round' },
    { title: 'Adjudicate', detail: 'one change plan per round, findings confirmed against the code first' },
    { title: 'Fix', detail: 'one implementer applies the plan, gates re-run' },
  ],
}

// args (all optional): { install: true|false, notebookEnded: true|false, target: '<what is reviewed>', locked: [...], prior: '<state before the run>', focus: { architect, 'bug-hunter', 'data-scientist' }, maxRounds: n }
// The defaults below describe the review of the build (run 1). A confirming review of a later delta passes its own target, prior and focus.
// Derived from reports/E13-campaign/workflow-review.run4.js: same loop, same schemas, same agent types.
const ROOT = '/home/lab/workspace/learning/projects/docdistance'
const PRIMER = `${ROOT}/reports/E13-campaign/PRIMER.md`
const BRIEF = `${ROOT}/reports/E18-relevance/relevance-build-brief.md`
const LEDGER = 'docs/experiments/wmd-docdistance-experiments.md'
const BUILD_AGENT = 'senior-backend-engineer'
const REVIEW_AGENT = 'devils-advocate:adversarial-reviewer'
const ADJUDICATOR = 'devils-advocate:adjudicator'
const MAX_ROUNDS = (args && args.maxRounds) || 3
const install = !!(args && args.install)
// false while the E19 reranker notebook executes (run 1); true for a review after it ended and after `make install` ran
const notebookEnded = !!(args && args.notebookEnded)
const baselineCmd = 'CI=true HF_HUB_OFFLINE=1 .venv/bin/python reports/E13-campaign/baseline.py verify --allow moved,n_moved --grounded'

const LOCKED = [
  'Additive policy (the user\'s decision): without `relevance=True` / `--relevance` the semantic detail map gains only `min_cost` and `best_target_index` per flow and their two `definitions` entries; every other key and every number of the command stays equal to commit 2ff35bf (release 1.1.32). `baseline.py verify` checks it',
  'The score is the registered lever of E18-H87: `r_i = max_j sigmoid(logit(b_j, a_i))` from `score_grid` of the shipped reranker, the statements of the second document as queries. Another aggregation (mean, top-k), a calibration of the score across pairs, a normalisation or a match / no-match rule is a new hypothesis and not a finding against this change. No match rule for one brief ships: E17-H86 and E18-H91 are Refuted',
  'The written relevance values are rounded to 4 decimals as the other scores of the map; maxima, section means, `relevance_rank` and `top_relevant` are taken on unrounded values. Known reading: on the 12 pairs of E18 the unrounded mean AUC is 0.8611 and the mean AUC of the written values is 0.8601; 180 of 1116 flow scores are below 5e-5 and are written as 0.0. Report this only with a concrete harm for a reader of the map file that the unrounded rank fields do not cover',
  'Heading lines of a document are statements under the default reader and are scored as such; this is the existing behaviour of the map and of every batch since E01',
  'Recorded Result and Verdict lines in the ledger are immutable. A reviewer who disagrees with a verdict reports it as a finding for the orchestrator; no agent edits the ledger, the two tracker files (docs/acc-crit-docdistance.md, docs/defects-docdistance.md) or .claude/JOURNAL.md',
  'No agent commits, tags, publishes or hand-edits the version',
  'The fixtures of E17 and E18 (data/processed/relevance-fixture) were written by the experimenter before scoring and cover one broad topic; the labels of E19 (data/processed/relevance-fixture-eu/labels.json) were assigned by one annotator before scoring. These limits are stated in the ledger. Report a limit only where a sentence of a document or of the ledger claims more than the fixture supports',
  notebookEnded
    ? 'The contrast across several briefs (Confirmed in E17-H85, E18-H88 and E19-H96) is not built in this change. The documents state the two measured readings of `min_cost` under `--anisotropy` (E19-H94 Confirmed on the public report, E17-H82 Refuted on the constructed report) and change no default; whether the contrast is built and whether a document recommends `--anisotropy` for this use are put to the user'
    : 'The contrast across several briefs (Confirmed in E17-H85, E18-H88 and E19-H96) is not built in this change, and the documents do not recommend `--anisotropy` for this use yet: both are put to the user after batch E19 ends',
  'The name of the user\'s commercial report and of its client stay out of the repository; the ledger says "private dataset"',
  'The message of a missing reranker on the source-conditioned path now names the reranker and `docdistance init wmd-wrt-source` (entry R3 of the brief directs it)',
  'The locked decisions of the earlier campaign still hold: the list REVIEW_ARGS.locked in reports/E13-campaign/workflow-review.run4.js',
]

const PRIOR = 'Commit 2ff35bf is release 1.1.32. After it: batches E17, E18 and E19 of the ledger (relevance reading of the semantic map: a long report against a short brief), each pre-registered with the fixture hashes stored before scoring (reports/E17-relevance-map-registration.json, reports/E18-relevance/E18-relevance-registration.json, reports/E19-relevance/E19-relevance-registration.json), executed by an executor agent and checked by an independent verifier agent and by a script of the orchestrator (records: reports/E18-relevance/results/, reports/E19-relevance/results/). Then build run wf_107b2527-8e4 (reports/E18-relevance/workflow-build.js, contract reports/E18-relevance/relevance-build-brief.md, entries R1 to R6): implementer, verifier (rejected on one major finding: the real-model end-to-end test skipped on any failure), repair round, second verifier (SHIP, 5 minor), final gate. The orchestrator then applied the minor findings: one row of the stub grid holds its maximum twice, so the tie rule of `relevance_source_index` is tested; the test that pins the 4-decimal rounding is named for what it asserts; the docstring and the `definitions` entry of `min_cost` say that it is the ground cost of one pair of statements and not a document distance; docs/cli-reference.md names exit code 1 for a missing reranker. STATE OF BATCH E19: E19-H93, E19-H94, E19-H96 and E19-H97 are recorded; E19-H95 (the reranker on the real documents) is executing in a detached notebook (about 3 hours) and its block has no Result yet. For that reason README.md, docs/cli-reference.md, docs/api-reference.md, .claude/skills/docdistance/SKILL.md and CHANGELOG.md quote E17 and E18 only, the ledger has no "E19 results" section and no at-a-glance rows for E19, the sections Lessons learned, Conclusions and Next steps of the ledger do not carry E17 to E19, and criterion ACC-MAP-106 stays open. None of that is a finding in this run: the E19 text and its tables are added after E19-H95 has a verdict and get a confirming review of their own. No `make install` ran since 1.1.32, so the version is unchanged and the editable install serves the working tree.'

const TARGET = (args && args.target) || `the uncommitted change of the docdistance project in ${ROOT} since commit 2ff35bf (\`git status --short\`, \`git diff 2ff35bf\`, plus the untracked files). Its parts: (1) code - src/docdistance/pipeline.py (\`min_cost\` and \`best_target_index\` per flow, \`_add_relevance\`, \`DocDistance._ensure_reranker\`, the \`relevance\` parameter of \`semantic_distance_with_details\`) and src/docdistance/cli.py (\`--relevance\` on \`distance-semantic\`); (2) tests - tests/test_pipeline.py, tests/test_cli.py, tests/test_e2e.py; (3) documents - README.md (section "Reading a long report against a brief"), docs/cli-reference.md, docs/api-reference.md, .claude/skills/docdistance/SKILL.md, CHANGELOG.md (section Unreleased); (4) the check reports/E18-relevance/checks/shipped_relevance.py with logs/E18-shipped-relevance.log; (5) experiments - the sections "E17 - relevance-reading", "E18 - relevance-single-brief" and "E19 - relevance-real-documents" of ${LEDGER} with their notebooks under notebooks/experiments/ (E17-kj-relevance-map, E18-kj-reranker-relevance, E18-kj-contrast-replication, E19-kj-real-document-map), their metrics files under reports/, the fixtures data/processed/relevance-fixture/ (with e18/) and data/processed/relevance-fixture-eu/ (git-ignored paths, present on disk), notebooks/14-kj-eu-relevance-fixture.ipynb and the entries added to data/external/download-fixtures.py; (6) the six open criteria ACC-MAP-101, ACC-MAP-102, ACC-MAP-103, ACC-CLI-104, ACC-MAP-105 and ACC-MAP-106 of docs/acc-crit-docdistance.md. The contract of the code is ${BRIEF}`

const FOCUS = {
  architect: 'whether the relevance reading is the smallest sound design for its purpose (one opt-in flag, one function that adds fields after the builder, the reranker load split out of `_ensure_grounding`) and holds no dead code, no speculative parameter and no second code path for one job; whether the new field names and the `definitions` entries fit the existing fields of the map (`min_incoming_cost`, `best_source_index`, `top_changed`); whether the load split leaves the source-conditioned path equal in behaviour; whether the Python signature, the CLI option, the five documents and the six criteria describe one and the same contract; whether a document promises more than the code does or than the ledger measured; whether the additive policy holds',
  'bug-hunter': 'wrong results and broken guarantees, each confirmed by a run: the direction of the grid (statements of the second document are the queries), ties, rounding, `relevance_rank` and `top_relevant`; documents of one statement, documents that share every statement, documents where the fragment filters (`--min-words`, `--skip-table-fragments`) drop statements, a statement count above the batch size of the reranker; `--relevance` with `--pairs` and its memos (two rows that share a document, the same pair twice), with `--json`, with `--result-only`, with `--anisotropy`, with the torch backend; `--relevance` without `--details-json`; a missing reranker model in every output mode; whether the source-conditioned commands still load and score as before the load split (`baseline.py verify --grounded`); whether reports/E18-relevance/checks/shipped_relevance.py can pass on a wrong map (labels, the AUC, the tolerance); tests that cannot fail',
  'data-scientist': `whether each recorded verdict of E17-H82 to E19-H97 follows from its registered bar and its stored artefact (recompute at least the headline number of E18-H87, E18-H88, E19-H94 and E19-H96 from the fixtures or the stored arrays with your own code); whether a label, a parameter, a bar or an exclusion was chosen after scoring (compare the registration files and their hashes with the fixtures); whether the AUC, the bootstrap, the balanced accuracy and the Otsu split are computed as the ledger defines them; whether the baseline of each batch is the plain map as shipped; whether a sentence of the ledger, of README.md, of docs/cli-reference.md, of docs/api-reference.md, of the skill or of CHANGELOG.md claims more than the batches measured (generalisation beyond the fixtures, a probability reading of the score, a match rule, a scale that holds across pairs of documents); whether the shipped check (mean AUC 0.8601 of the written values against 0.8611 of the notebook) supports criterion ACC-MAP-105; whether the differences between the three batches (one-topic report against real report, constructed briefs against real notes) are stated where a reader of the documents needs them`,
}
const LENSES = ['architect', 'bug-hunter', 'data-scientist'].map(lens => ({ lens, focus: (args && args.focus && args.focus[lens]) || FOCUS[lens] }))
const locked = (args && args.locked) || LOCKED
const prior = (args && args.prior) || PRIOR

const FINDING = { type: 'object', required: ['severity', 'claim', 'evidence'], properties: { severity: { enum: ['blocking', 'major', 'minor'] }, file: { type: 'string' }, line: { type: 'integer' }, claim: { type: 'string' }, evidence: { type: 'string' } } }
const REVIEW_SCHEMA = { type: 'object', required: ['lens', 'verdict', 'findings'], properties: { lens: { type: 'string' }, verdict: { type: 'string' }, findings: { type: 'array', items: FINDING } } }
const PLAN_SCHEMA = {
  type: 'object', required: ['changes', 'declined'],
  properties: {
    changes: { type: 'array', items: { type: 'object', required: ['root_cause', 'change', 'files'], properties: { root_cause: { type: 'string' }, change: { type: 'string' }, files: { type: 'array', items: { type: 'string' } }, could_break: { type: 'string' } } } },
    declined: { type: 'array', items: { type: 'object', required: ['finding', 'reason'], properties: { finding: { type: 'string' }, reason: { type: 'string' } } } },
    for_orchestrator: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
}
const FIX_SCHEMA = {
  type: 'object', required: ['applied', 'files_changed', 'make_test', 'make_lint', 'baseline_verify'],
  properties: {
    applied: { type: 'array', items: { type: 'object', required: ['change', 'done', 'evidence'], properties: { change: { type: 'string' }, done: { type: 'boolean' }, evidence: { type: 'string' } } } },
    files_changed: { type: 'array', items: { type: 'string' } },
    make_test: { type: 'string' }, make_lint: { type: 'string' }, baseline_verify: { type: 'string' },
    for_orchestrator: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
}

const RULES = `Hard rules of this run: never read \`**/@archive\` or \`**/.ipynb_checkpoints\`; no network request (\`HF_HUB_OFFLINE=1\`, \`CI=true\`), no package install, no commit, no tag; ${install ? '' : notebookEnded ? 'do NOT run `make install` (it increments the version; the editable install serves the working tree); ' : 'do NOT run `make install` (a long experiment notebook is executing and imports the package from src; the editable install serves the working tree); '}${notebookEnded ? 'no experiment notebook is executed or edited by an agent, and data/interim/E19/ is read only (the stored reranker grids took 3.3 hours to compute).' : 'a notebook named notebooks/experiments/E19-kj-real-document-reranker.ipynb is executing: do not execute, edit or wait for it, and do not write data/interim/E19/.'} Scratch files go under /tmp/claude-1000/-home-lab-workspace-learning-projects-docdistance/48fccd81-2e03-4952-88be-100e909f2767/scratchpad/relevance-review/ only.`

const reviewPrompt = (l, round, history) => `Lens: ${l.lens}. Adversarial review round ${round}. Target: ${TARGET}. Whole-repo mode: read the surrounding code, you may run commands (tests, small probes with \`CI=true HF_HUB_OFFLINE=1 .venv/bin/python\`, the command \`.venv/bin/docdistance\`), you never edit, stage or revert a file of the repository. ${RULES}

Focus: ${l.focus}.

Context you must respect: ${PRIMER} (hard rules of a build, additive policy, metric-property rule: a score is never called a distance, a lower bound or exact). The acceptance criteria and defects are in docs/acc-crit-docdistance.md and docs/defects-docdistance.md (\`pm-tools list docs/acc-crit-docdistance.md --status open\`); the ledger blocks are read with \`hypothesis-tools show ${LEDGER} <id>\`.

Locked decisions - not findings, do not reopen them:
${locked.map(d => `- ${d}`).join('\n')}

State before this review started: ${prior}
${history.length ? `Earlier rounds of this review and what was changed after them (a finding that only restates one of these without new evidence is not material):\n${JSON.stringify(history.map(h => ({ round: h.round, changes: h.plan ? h.plan.changes.map(c => c.change) : [], declined: h.plan ? h.plan.declined : [] })), null, 2)}` : ''}

Report only what is material: a wrong result, a broken guarantee, an untestable or false claim, a verdict its artefact does not support, a criterion that cannot be met as built, dead or duplicated code. Style, naming taste and word count are not findings. Give file and line for every finding and say how you confirmed it (the command you ran or the lines you read). An empty findings list is a valid answer. Return the JSON only, lens "${l.lens}".`

const adjudicatePrompt = (findings, round, history) => `Adjudicate the findings of adversarial review round ${round} of the docdistance change in ${ROOT} into ONE change plan. Confirm each finding against the code, the notebooks or the documents before planning a change for it; a finding that does not reproduce is declined with what you checked. Triage materiality first; group by root cause; propose the smallest changes that answer them; decline with a reason what is duplication, taste, or a locked decision. A change that needs a write to .claude/JOURNAL.md, docs/acc-crit-docdistance.md, docs/defects-docdistance.md or the ledger ${LEDGER} is not planned as a change: list it in for_orchestrator, because only the orchestrator writes those files. A change to an experiment notebook is planned only as an edit of a markdown cell; a finding that needs a notebook to be executed again goes to for_orchestrator. ${RULES}

Authoritative context: ${PRIMER} and the contract ${BRIEF}. Locked decisions:
${locked.map(d => `- ${d}`).join('\n')}
${history.length ? `Changes already made in earlier rounds (revert machinery an earlier round introduced when it is the cause of a new finding, do not stack a second fix on it):\n${JSON.stringify(history.map(h => ({ round: h.round, changes: h.plan ? h.plan.changes : [] })), null, 2)}` : ''}

Findings:
${JSON.stringify(findings, null, 2)}
Return the JSON only.`

const fixPrompt = (plan, round) => `You are the IMPLEMENTER of the fix round after adversarial review round ${round} of the docdistance change. Repo root: ${ROOT}; work in the main working tree. FIRST read ${PRIMER} in full (hard rules of a build) and ${BRIEF} (the contract of the change). ${RULES} Apply exactly this change plan, nothing beyond it:
${JSON.stringify(plan.changes, null, 2)}
A planned change that turns out to be wrong against the code is not applied: report it with done=false and the reason. Never write .claude/JOURNAL.md, the two tracker files or the ledger, and never execute a notebook - put what they need in for_orchestrator. Then run ${install ? '`make install` once (new build to test - it increments the patch version, which is wanted), ' : ''}\`make format\`, \`make lint\`, \`make test\`, \`uvx ruff@0.16.2 check --target-version py312\` and \`${baselineCmd}\` and report the last line of each. Return the JSON only.`

const history = []
let clean = false
let cleanRounds = 0
for (let round = 1; round <= MAX_ROUNDS && cleanRounds < 2; round++) {
  phase('Review')
  const reviews = (await parallel(LENSES.map(l => () => agent(reviewPrompt(l, round, history), { label: `review:r${round}:${l.lens}`, phase: 'Review', schema: REVIEW_SCHEMA, effort: 'high', agentType: REVIEW_AGENT })))).filter(Boolean)
  const findings = reviews.flatMap(r => r.findings.map(f => ({ ...f, lens: r.lens })))
  log(`round ${round}: ${findings.length} findings across ${reviews.length} of ${LENSES.length} lenses`)
  if (reviews.length < LENSES.length) { log(`round ${round}: a lens returned nothing - the round does not count as clean`) }
  if (!findings.length) {
    history.push({ round, reviews, plan: null, fix: null })
    clean = reviews.length === LENSES.length
    cleanRounds = clean ? cleanRounds + 1 : 0
    log(`round ${round}: clean (${cleanRounds} consecutive)`)
    continue
  }
  phase('Adjudicate')
  const plan = await agent(adjudicatePrompt(findings, round, history), { label: `adjudicate:r${round}`, phase: 'Adjudicate', schema: PLAN_SCHEMA, effort: 'high', agentType: ADJUDICATOR })
  if (!plan) { history.push({ round, reviews, plan: null, fix: null }); log(`round ${round}: adjudicator returned nothing - stopping for the orchestrator`); break }
  if (!plan.changes.length) {
    history.push({ round, reviews, plan, fix: null })
    log(`round ${round}: every finding declined after confirmation - no change made`)
    clean = reviews.length === LENSES.length
    cleanRounds = clean ? cleanRounds + 1 : 0
    continue
  }
  phase('Fix')
  const fix = await agent(fixPrompt(plan, round), { label: `fix:r${round}`, phase: 'Fix', schema: FIX_SCHEMA, agentType: BUILD_AGENT })
  history.push({ round, reviews, plan, fix })
  cleanRounds = 0
  log(`round ${round}: ${plan.changes.length} changes planned, fix ${fix ? 'returned' : 'returned nothing'} - a confirming round follows`)
}
clean = cleanRounds >= 2
if (!clean) log(`fewer than two consecutive clean rounds within ${history.length} rounds - the orchestrator decides`)
return { clean, cleanRounds, rounds: history }
