export const meta = {
  name: 'docdistance-e13-review',
  description: 'Final review of the E13 campaign: adversarial review of the work made after round 3 of the first review run (input format, DEF-DEPS-17, E16-H81, drift audit) by the architect, bug-hunter and data-scientist lenses, one adjudicated change plan and one fix round per review round, until a round is clean (at most 3 rounds)',
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
const MAX_ROUNDS = 3
const allow = (args && args.allow) || 'moved,n_moved'
// Generated from workflow-review.js and review-args-final.json (2026-10-10). Do not edit: change the generator inputs and regenerate.
const REVIEW_ARGS = {
 "allow": "moved,n_moved",
 "locked": [
  "Additive policy (the user's decision): every number and every detail-file key that 1.1.5 produces stays bit-identical, with two registered exceptions - `moved` and `n_moved` (ACC-TRANS-77), and the statements of a document read with `--markdown` that holds marker-only lines, which the markdown reader now drops as its docstring always said (DEF-SEGM-15). New behaviour ships as new fields, functions or opt-in options",
  "The shipped anisotropy pre-pass `all_but_the_top(k=1)` stays on both paths: E14-H75 and E14-H76 are Confirmed, criterion ACC-GEOM-21 (centering alone on the conditioned path) is rejected by its registered decision rule",
  "No kernel pre-pass ships: E14-H77 is Refuted",
  "`MATCH_CUTOFF = 0.325` (E13-H71, E15-H79) and `SOURCE_SUPPORT_CUTOFF = 0.34625` (E15-H78) are the measured values and stay two constants: they were calibrated on two different tasks",
  "`DIFF_CHANGED_COST` in pipeline.py (derived from the verdict threshold, 0.3253) and the shipped `displacement`, `displacement_norm`, `changed` and `no_counterpart` fields keep their 1.1.5 definitions beside the new matching fields: the additive policy forbids replacing them, so two readings of one statement exist on purpose and the documents say which one to read",
  "The position term `1e-6 * |i/n - j/m|` in the partial matching is a determinism rule: it ranks tied copies by position and does not always pair them in document order (ACC-TRANS-83 states the tie condition; ACC-TRANS-71 is rejected as overridden). E15-H79 is recorded as Refuted and the limit on repeated statements (111 of 120) is documented, not claimed as fixed. A different form of the term (for example a squared one) is a new hypothesis and needs its own measurement; proposing it is not a finding against this change",
  "The matching fields and the unsupported-statement count are read on the embeddings as passed in, whatever the `anisotropy` flag says (ACC-TRANS-76, E15-H78)",
  "Recorded Result and Verdict lines in the ledger are immutable. A reviewer who disagrees with a verdict reports it as a finding for the orchestrator; no agent edits the ledger, the two tracker files or the journal",
  "Patch-version increments made by `make install` are wanted; no agent commits, tags, publishes or hand-edits the version",
  "E12-H67 notebook: `LOAD_MAX_TRIES` was changed from 20 to 2 on 2026-10-09 so that the re-execution finishes; the load ceiling and the breach record are unchanged. Not a finding",
  "E06 verdicts were recorded on 2026-10-09 from two runs (2026-07-01 and the re-execution); E06 is outside this review except where a shipped document quotes it",
  "Four criteria were overridden on 2026-10-10 because they could not hold as written, each with the reason in the tracker: ACC-TRANS-49 → ACC-TRANS-80, ACC-TRANS-71 → ACC-TRANS-83, ACC-CLI-31 → ACC-CLI-81, ACC-CLI-40 → ACC-CLI-82. The last two define a quiet CPU by `/proc/pressure/cpu` (`some avg10` below 1), because hung file scans of another session held the load average at 31 to 55 with the CPU 97% idle; the speed bars (3x, 5x) are unchanged",
  "The batch memos are opt-in in Python (`DocDistance(memo_size=0)` by default) and on in the three `--pairs` branches of the CLI with a bound of 64 entries: the additive policy keeps the one-shot paths on the 1.1.5 code",
  "The marker warning (DEF-SEGM-12) is logged once per `embed_statements` call, also on a memo hit, so memo on and memo off log the same lines; it names the file",
  "Input format (the user's request of 2026-10-10: autodetection of HTML, markdown and other input): one option `input_format` with the values plain, markdown, html, auto; the default stays `plain`, so every 1.1.5 number of a call without the option is unchanged; whether `auto` becomes the default is the user's decision after E16-H81 and would be a registered exception. `--markdown` / `markdown=True` stay and equal `markdown`",
  "The detector reads the content only and never the file extension: raw text has no extension, and with a content rule `auto` differs from `plain` only for a document that holds markup",
  "The HTML reader is a conversion to markdown lines (`html_to_markdown`, standard library parser, no new dependency) in front of the existing markdown reader, so sections, fragment filters and the twin property need no second code path",
  "DEF-DEPS-17 is fixed inside the repository: `init` saves the tokenizer `facebookAI/xlm-roberta-base` beside the SaT weights when the source folder lacks it, and `Segmenter` reads it from that folder; no change to the S3 bucket or to `make sync_models_up`",
  "The `moved` flag on an insertion into a reworded document, where the inserted statement takes the match of a kept one (scratch reading 25 of 150 cases, 123 of 150 under the 1.1.5 rule), is a documented limit and a clause for the next fixture, not a defect: the flag meets ACC-TRANS-77, and a rule for the case is a new hypothesis",
  "The drift audit of 2026-10-10 (reports/E13-campaign/results/reexec-drift-audit.md) is recorded as log lines in both ledgers; recorded Result and Verdict lines are immutable. Three verdict labels of the structure log (E08-H44, E08-H47, E09-H53) contradict their stored artefacts in the committed files as well; whether they get a confirming round is the user's decision",
  "The E04 notebook stays as re-executed on the GPU on 2026-10-09; the audit's CPU run on build 1.1.19 does not reproduce the record of docdistance 1.1.2 (best k 2 at 3.39x) either, so an in-place CPU run would not restore agreement with the record",
  "A byte order mark in a document read by the `plain` or the `markdown` reader stays in the first statement: that is 1.1.5 behaviour, and removing it changes shipped numbers; it is put to the user separately. The HTML reader and the detector ignore it",
  "Declined verifier findings of the two packages, with reasons, are listed in reports/E13-campaign/input-format-fixes-brief.md under 'Declined'"
 ],
 "prior": "Review run wf_366ba93d-451 reviewed the whole uncommitted change in three rounds (record: reports/E13-campaign/results/stage-c-run-wf_366ba93d-451.json). Round 1: the documents no longer say that copies of a repeated statement pair in document order; the offline claim of the deployment documents was corrected. Round 2: the same offline claim at three more sites; `_details_path` creates the folder of a `{pair}` template. Round 3: four documents no longer infer that an order-keeping insertion flags nothing (new bullet 'Limit: inserted statement that takes a match' in docs/api-reference.md); the docstring of `compute_source_conditioned` reads 6.34x (E14-H76). Verdicts of round 3 before its fix: architect SHIP, bug-hunter SHIP, data-scientist DO-NOT-SHIP on that one sentence. The run was stopped by the orchestrator during round 4, because the work below changes the tree. After it, build run wf_1d07273f-fc6 (record: reports/E13-campaign/results/stage-b-run-wf_1d07273f-fc6.json) built two packages, each checked by an adversarial verifier with minor findings only: `review-followup` (DEF-DEPS-17; wording of the unsupported count, ACC-TRANS-84 overriding ACC-TRANS-72; `ot.emd` named in the solution documents; the help text reads ANISOTROPY_MIN_POOL; reports/E13-campaign/checks/held-out-readings/readings.py reproduces 0.9052, 0.9143 and 0.5390; the pre-pass gain is stated per setting) and `input-format` (ACC-SEGM-85 to ACC-SEGM-92, ACC-CLI-88). A fix agent then applied ten verifier findings (X1 to X10 of reports/E13-campaign/input-format-fixes-brief.md: ten more block tags and three dropped tags in the HTML conversion, numbered ordered-list items, fenced lines excluded from the HTML rule of the detector, a document marker behind a byte order mark or an XML declaration, the advice of the marker warning under auto, no path warning for a text that holds '<', a separator-row rule without backtracking, three tests for behaviours that had none, and a command line test of the DEF-DEPS-17 error). The orchestrator's gates on build 1.1.21: lint clean, 496 passed and 1 skipped, baseline 'verify: 55 pairs, 3 grounded, 0 difference(s)', Python 3.12 syntax check clean; trackers 0 open (86 criteria closed, 6 rejected as overridden, 17 defects closed). E16-H81 was executed by run wf_86296670-5d0 and verified with 6 minor findings: Killed-at-gate, because the default reader returns a marker statement on 0 of 11 scored summaries; its side readings (band unchanged, Spearman 0.9816, HTML twins equal for 11 of 12) are recorded as such in the ledger and in docs/solution/wmd-docdistance-solution-sota.md. The drift audit added 26 log lines to the two ledgers and corrected 55 markdown sentences in 9 notebooks through nbformat; no notebook was executed for it."
}
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

const TARGET = `the part of the uncommitted change of the docdistance project in ${ROOT} that was made after round 3 of review run wf_366ba93d-451. A copy of the files before it (taken 2026-10-10 04:55) is in /tmp/claude-1000/-home-lab-workspace-learning-projects-docdistance/48fccd81-2e03-4952-88be-100e909f2767/scratchpad/pre-followup-snapshot: \`diff -ru /tmp/claude-1000/-home-lab-workspace-learning-projects-docdistance/48fccd81-2e03-4952-88be-100e909f2767/scratchpad/pre-followup-snapshot/src src\`, and the same for tests, docs, README.md, CHANGELOG.md and .claude/skills/docdistance, shows the delta (the two tracker files and the two ledgers under docs/ also differ; they are written by the orchestrator through their tools). New since then: (1) the option input_format with the HTML reader and the detector - briefs reports/E13-campaign/input-format-brief.md and reports/E13-campaign/input-format-fixes-brief.md; (2) the fix of DEF-DEPS-17 and five further entries - brief reports/E13-campaign/review-followup-brief.md; (3) hypothesis E16-H81 - notebooks/experiments/E16-H81-kj-autodetected-reader-default.ipynb, reports/E16-H81-autodetected-reader-default-metrics.json, the E16 section of ${LEDGER}, the registration record reports/E13-campaign/results/E16-H81-registration.json; (4) the drift audit of the re-executed E01 to E12 notebooks - reports/E13-campaign/results/reexec-drift-audit.md, its log lines in ${LEDGER} and docs/experiments/wmd-structure-distance-experiments.md, and the corrected markdown cells of 9 notebooks under notebooks/experiments/. The rest of the uncommitted change was reviewed in three rounds: read it for context, and report a finding on it only when the new work broke it`

const LENSES = [
  { lens: 'architect', focus: 'the option input_format: one parameter beside the `markdown` flag on 11 calls and 3 commands, the conversion of HTML to markdown lines in front of the existing markdown reader, the content-only detector with its two constants, the reader stored in the memo entry, `_read_as`; whether any of it is more than the request needs (autodetection of HTML, markdown and other input) or duplicates logic that exists; the fix of DEF-DEPS-17 (`init` saves the tokenizer beside the SaT weights, `Segmenter` reads it from the folder); whether the default path is unchanged (additive policy: a shipped number or detail-file key changed without a registered criterion); whether README.md, docs/cli-reference.md, docs/api-reference.md, docs/aws-deployment-reference.md, the docdistance skill, CHANGELOG.md and docs/solution/*.md describe what the code does' },
  { lens: 'bug-hunter', focus: '`html_to_markdown` and `detect_format` on odd and hostile input - unclosed and nested tags, a table inside a list, a pipe in a cell, character references, a byte order mark, an XML declaration, HTML inside a markdown fence, markdown inside a `pre`, very long lines, the empty string; the twin property on nested and ordered lists; `auto` when a file changes between two calls with the memo on; the memo key and the reader stored in the entry; the marker warning under each reader; `--markdown` beside `--input-format`; `_looks_like_path`; `init --source` when the tokenizer is in the source folder, in the cache or nowhere, and a second `init` into the same home; differences of `html.parser` between Python 3.12 and 3.14' },
  { lens: 'data-scientist', focus: `E16-H81: the registration (block hash and time in reports/E13-campaign/results/E16-H81-registration.json against the block in ${LEDGER}), whether the kill-gate and the acceptance bar were applied as registered, whether every number of the Result and Verdict lines is in the metrics file and in the notebook outputs, whether the twin check measures what it claims, and whether any statement about a default of \`auto\` follows from the readings; the drift audit: whether its table rows equal the two versions of each metrics file (\`git show HEAD:<file>\` against the working tree), whether each log line in the two ledgers states the readings of the audit, whether a corrected notebook sentence contradicts the stored output of its notebook, and whether a solution document quotes a reading that the audit shows as different from the stored file; the script reports/E13-campaign/checks/held-out-readings/readings.py against the three ledger readings it recomputes` },
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
