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
// Generated from workflow-review.js and review-args.json (2026-10-10). Do not edit: change the two inputs and regenerate.
const REVIEW_ARGS = {
 "allow": "moved,n_moved",
 "locked": [
  "Additive policy (the user's decision): every number and every detail-file key that 1.1.5 produces stays bit-identical, with two registered exceptions - `moved` and `n_moved` (ACC-TRANS-77), and the statements of a document read with `--markdown` that holds marker-only lines, which the markdown reader now drops as its docstring always said (DEF-SEGM-15). New behaviour ships as new fields, functions or opt-in options",
  "The shipped anisotropy pre-pass `all_but_the_top(k=1)` stays on both paths: E14-H75 and E14-H76 are Confirmed, criterion ACC-GEOM-21 (centering alone on the conditioned path) is rejected by its registered decision rule",
  "No kernel pre-pass ships: E14-H77 is Refuted",
  "The default reader stays the prose segmenter for markdown input; DEF-SEGM-12 is fixed by a warning, because a markdown-aware default would change shipped numbers. The alternative is put to the user separately",
  "`MATCH_CUTOFF = 0.325` (E13-H71, E15-H79) and `SOURCE_SUPPORT_CUTOFF = 0.34625` (E15-H78) are the measured values and stay two constants: they were calibrated on two different tasks",
  "`DIFF_CHANGED_COST` in pipeline.py (derived from the verdict threshold, 0.3253) and the shipped `displacement`, `displacement_norm`, `changed` and `no_counterpart` fields keep their 1.1.5 definitions beside the new matching fields: the additive policy forbids replacing them, so two readings of one statement exist on purpose and the documents say which one to read",
  "The position term `1e-6 * |i/n - j/m|` in the partial matching is a determinism rule: it ranks tied copies by position and does not always pair them in document order (ACC-TRANS-83 states the tie condition; ACC-TRANS-71 is rejected as overridden). E15-H79 is recorded as Refuted and the limit on repeated statements (111 of 120) is documented, not claimed as fixed. A different form of the term (for example a squared one) is a new hypothesis and needs its own measurement; proposing it is not a finding against this change",
  "The matching fields and the unsupported-statement count are read on the embeddings as passed in, whatever the `anisotropy` flag says (ACC-TRANS-76, E15-H78)",
  "Recorded Result and Verdict lines in the ledger are immutable. A reviewer who disagrees with a verdict reports it as a finding for the orchestrator; no agent edits the ledger, the two tracker files or the journal",
  "Patch-version increments made by `make install` are wanted; no agent commits, tags, publishes or hand-edits the version",
  "The E01 to E12 notebooks and their metrics files were re-executed only; their diffs are stored outputs and not part of the change under review",
  "E12-H67 notebook: `LOAD_MAX_TRIES` was changed from 20 to 2 on 2026-10-09 so that the re-execution finishes; the load ceiling and the breach record are unchanged. Not a finding",
  "E06 verdicts were recorded on 2026-10-09 from two runs (2026-07-01 and the re-execution); E06 is outside this review except where a shipped document quotes it",
  "Four criteria were overridden on 2026-10-10 because they could not hold as written, each with the reason in the tracker: ACC-TRANS-49 → ACC-TRANS-80, ACC-TRANS-71 → ACC-TRANS-83, ACC-CLI-31 → ACC-CLI-81, ACC-CLI-40 → ACC-CLI-82. The last two define a quiet CPU by `/proc/pressure/cpu` (`some avg10` below 1), because hung file scans of another session held the load average at 31 to 55 with the CPU 97% idle; the speed bars (3x, 5x) are unchanged",
  "The batch memos are opt-in in Python (`DocDistance(memo_size=0)` by default) and on in the three `--pairs` branches of the CLI with a bound of 64 entries: the additive policy keeps the one-shot paths on the 1.1.5 code",
  "The marker warning (DEF-SEGM-12) is logged once per `embed_statements` call, also on a memo hit, so memo on and memo off log the same lines; it names the file"
 ],
 "prior": "Stage B built seven packages (runs wf_e1b16a4b-b94 and wf_099e69d3-480; full returns in reports/E13-campaign/results/stage-b-run-*.json). Each was checked by an adversarial verifier; no blocking or major finding remained. The verifiers' 30 minor findings were triaged in reports/E13-campaign/repairs-brief.md: 18 repairs (R1 to R18) and two new defects (DEF-DEPS-14 offline flag set after the Hub import, DEF-SEGM-15 markdown reader keeps bare markers) were applied and verified, eight follow-up entries (F1 to F8, including DEF-DEPS-16) were applied after that, and four were declined with the reason written in the brief (two duplicated test fixtures, the 1e-5 tolerance of ACC-TRANS-80, the marker warning repeated per row, the median statistic of ACC-CLI-32). Known and accepted, do not report again without new evidence: a bare numbered marker such as '42.' stays prose in the markdown reader by design; the D1 closed-port test skips where the models are not installed; `checks/detail-map/verify_build.py` was rewritten to encode with the installed package instead of shipping 783 KB of embeddings. The real-model readings quoted as evidence can be repeated with the scripts under reports/E13-campaign/checks/ (see its README.md). State at the start of this review: version 1.1.15, `make test` 272 passed and 1 skipped, `make lint` clean, baseline 0 differences on 55 pairs and 3 grounded results; 0 criteria and 0 defects open (77 criteria closed, 6 rejected as overridden, 16 defects closed). Also accepted: `docdistance init` no longer removes `HF_HUB_OFFLINE` from the environment (the line had no effect on the Hub library, DEF-DEPS-16), and a model load and an `init` that overlap on two threads share one mode of the Hub library, which the docstring of `_hub_flag` states."
}
const allow = (args && args.allow) || REVIEW_ARGS.allow
const locked = (args && args.locked) || REVIEW_ARGS.locked
const prior = (args && args.prior) || REVIEW_ARGS.prior
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
