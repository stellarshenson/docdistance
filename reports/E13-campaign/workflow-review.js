export const meta = {
  name: 'docdistance-e13-review',
  description: 'Stage C of the E13 campaign: adversarial review of the whole uncommitted change by the architect, bug-hunter and data-scientist lenses, one adjudicated change plan and one fix round per review round, repeated until a round is clean (at most 4 rounds)',
  phases: [
    { title: 'Review', detail: 'three lenses in parallel per round' },
    { title: 'Adjudicate', detail: 'one change plan per round, findings confirmed against the code first' },
    { title: 'Fix', detail: 'one implementer applies the plan, gates re-run', model: 'opus' },
  ],
}

// args: { allow: 'moved,n_moved', locked: ['<decision the reviewers must not reopen>', ...], prior: '<what earlier rounds already fixed>' }
const ROOT = '/home/lab/workspace/learning/projects/docdistance'
const PRIMER = `${ROOT}/reports/E13-campaign/PRIMER.md`
const LEDGER = 'docs/experiments/wmd-docdistance-experiments.md'
const BUILD_MODEL = 'opus'
const BUILD_AGENT = 'senior-backend-engineer'
const REVIEW_AGENT = 'devils-advocate:adversarial-reviewer'
const ADJUDICATOR = 'devils-advocate:adjudicator'
const MAX_ROUNDS = 4
const allow = (args && args.allow) || 'moved,n_moved'
const locked = (args && args.locked) || []
const prior = (args && args.prior) || 'none'
const baselineCmd = `CI=true .venv/bin/python reports/E13-campaign/baseline.py verify --allow ${allow} --grounded`

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

const TARGET = `the whole uncommitted change of the docdistance project in ${ROOT}: \`git diff HEAD -- src tests docs README.md CHANGELOG.md Makefile pyproject.toml .github .claude/skills\` plus the untracked files \`git status --short\` lists under src/, tests/, notebooks/experiments/ (E13-*, E14-*, E15-*) and reports/ (E13-*, E14-*, E15-*, E13-campaign/). The E01 to E12 notebooks and their metrics files under reports/ were only re-executed; their diffs are stored outputs, not part of the change under review`

const LENSES = [
  { lens: 'architect', focus: 'consistency between the new functions, fields and options and the shipped ones; logic duplicated across distance.py, pipeline.py and cli.py; constants that belong beside the existing thresholds; over-engineering and speculative parameters; whether the additive policy held (a shipped number or detail-file key changed without a registered criterion); whether the documents (docs/solution/*.md, docs/api-reference.md, docs/cli-reference.md, README.md, the docdistance skill) describe what the code does' },
  { lens: 'bug-hunter', focus: 'edge cases - empty documents, one statement, n much larger than m, every statement unmatched, duplicate statements, ties in the assignment, memo keys and invalidation, bounded memo eviction, float comparisons, the exact-solver guard on large inputs, the telemetry block on import, Python 3.12 to 3.14 differences including the multiprocessing start method' },
  { lens: 'data-scientist', focus: `the experiments behind the change - the E13, E14 and E15 notebooks under notebooks/experiments/, their metrics files under reports/, and the Result and Verdict lines of the E13, E14 and E15 sections of ${LEDGER}: a cutoff or threshold tuned on the data it is scored on, leakage between calibration and scored splits, a verdict the notebook outputs do not support, a metric that does not measure the claim, a number in the ledger or in docs/solution/*.md that differs from the metrics file, an unseeded random draw, a conclusion that rests on one fixture stated as general, and whether each constant the code ships (cutoff, cap) is the value the experiment measured` },
]

const reviewPrompt = (l, round, history) => `Lens: ${l.lens}. Adversarial review round ${round}. Target: ${TARGET}. Whole-repo mode: read the surrounding code, you may run commands (tests, small probes with \`CI=true .venv/bin/python\`), you never edit, stage or revert a file. Never read \`**/@archive\` or \`**/.ipynb_checkpoints\`.

Focus: ${l.focus}.

Context you must respect: ${PRIMER} (hard rules, stage B contract - additive policy, registered exceptions, metric-property rule). The acceptance criteria and defects are in docs/acc-crit-docdistance.md and docs/defects-docdistance.md (\`pm-tools report docs\` summarises them).

Locked decisions - not findings, do not reopen them:
${locked.length ? locked.map(d => `- ${d}`).join('\n') : '- none'}

Fixed before this review started: ${prior}
${history.length ? `Earlier rounds of this review and what was changed after them (a finding that only restates one of these without new evidence is not material):\n${JSON.stringify(history.map(h => ({ round: h.round, changes: h.plan ? h.plan.changes.map(c => c.change) : [], declined: h.plan ? h.plan.declined : [] })), null, 2)}` : ''}

Report only what is material: a wrong result, a broken guarantee, an untestable or false claim, a criterion marked met that is not met, dead or duplicated code. Style, naming taste and word count are not findings. Give file and line for every finding and say how you confirmed it (the command you ran or the lines you read). An empty findings list is a valid answer. Return the JSON only, lens "${l.lens}".`

const adjudicatePrompt = (findings, round, history) => `Adjudicate the findings of adversarial review round ${round} of the docdistance change in ${ROOT} into ONE change plan. Confirm each finding against the code, the notebooks or the documents before planning a change for it; a finding that does not reproduce is declined with what you checked. Triage materiality first; group by root cause; propose the smallest changes that answer them; decline with a reason what is duplication, taste, or a locked decision. A change that needs a write to .claude/JOURNAL.md, docs/acc-crit-docdistance.md, docs/defects-docdistance.md or the ledger ${LEDGER} is not planned as a change: list it in for_orchestrator, because only the orchestrator writes those files.

Authoritative context: ${PRIMER} (stage B contract). Locked decisions:
${locked.length ? locked.map(d => `- ${d}`).join('\n') : '- none'}
${history.length ? `Changes already made in earlier rounds (revert machinery an earlier round introduced when it is the cause of a new finding, do not stack a second fix on it):\n${JSON.stringify(history.map(h => ({ round: h.round, changes: h.plan ? h.plan.changes : [] })), null, 2)}` : ''}

Findings:
${JSON.stringify(findings, null, 2)}
Return the JSON only.`

const fixPrompt = (plan, round) => `You are the IMPLEMENTER of the fix round after adversarial review round ${round} of the docdistance change. Repo root: ${ROOT}; work in the main working tree. FIRST read ${PRIMER} in full (hard rules, stage B contract). Apply exactly this change plan, nothing beyond it:
${JSON.stringify(plan.changes, null, 2)}
A planned change that turns out to be wrong against the code is not applied: report it with done=false and the reason. A change to a notebook means editing the notebook and re-executing it with the primer's nbconvert command to 0 errors. Never write .claude/JOURNAL.md, the two tracker files or the ledger - put what they need in for_orchestrator. Then run \`make install\` once (new build to test - it increments the patch version, which is wanted), \`make format\`, \`make lint\`, \`make test\` and \`${baselineCmd}\` and report the last line of each. Return the JSON only.`

const history = []
let clean = false
for (let round = 1; round <= MAX_ROUNDS && !clean; round++) {
  phase('Review')
  const reviews = (await parallel(LENSES.map(l => () => agent(reviewPrompt(l, round, history), { label: `review:r${round}:${l.lens}`, phase: 'Review', schema: REVIEW_SCHEMA, effort: 'high', agentType: REVIEW_AGENT })))).filter(Boolean)
  const findings = reviews.flatMap(r => r.findings.map(f => ({ ...f, lens: r.lens })))
  log(`round ${round}: ${findings.length} findings across ${reviews.length} of ${LENSES.length} lenses`)
  if (reviews.length < LENSES.length) { log(`round ${round}: a lens returned nothing - the round does not count as clean`) }
  if (!findings.length) {
    history.push({ round, reviews, plan: null, fix: null })
    clean = reviews.length === LENSES.length
    if (!clean) continue
    break
  }
  phase('Adjudicate')
  const plan = await agent(adjudicatePrompt(findings, round, history), { label: `adjudicate:r${round}`, phase: 'Adjudicate', schema: PLAN_SCHEMA, effort: 'high', agentType: ADJUDICATOR })
  if (!plan) { history.push({ round, reviews, plan: null, fix: null }); log(`round ${round}: adjudicator returned nothing - stopping for the orchestrator`); break }
  if (!plan.changes.length) {
    history.push({ round, reviews, plan, fix: null })
    log(`round ${round}: every finding declined after confirmation - no change made`)
    clean = reviews.length === LENSES.length
    continue
  }
  phase('Fix')
  const fix = await agent(fixPrompt(plan, round), { label: `fix:r${round}`, phase: 'Fix', schema: FIX_SCHEMA, model: BUILD_MODEL, agentType: BUILD_AGENT })
  history.push({ round, reviews, plan, fix })
  log(`round ${round}: ${plan.changes.length} changes planned, fix ${fix ? 'returned' : 'returned nothing'} - a confirming round follows`)
}
if (!clean) log(`no clean round within ${history.length} rounds - the orchestrator decides`)
return { clean, rounds: history }
