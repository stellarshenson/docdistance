export const meta = {
  name: 'docdistance-e13-review',
  description: 'Confirming review of the delta after round 3 of run wf_2fef834e-695 (reader claims of the three solution documents, DEF-DOCS-18) by the architect, bug-hunter and data-scientist lenses; two consecutive clean rounds end it (at most 3 rounds)',
  phases: [
    { title: 'Review', detail: 'three lenses in parallel per round' },
    { title: 'Adjudicate', detail: 'one change plan per round, findings confirmed against the code first' },
    { title: 'Fix', detail: 'one implementer applies the plan, gates re-run' },
  ],
}

// args: { allow: 'moved,n_moved', locked: ['<decision the reviewers must not reopen>', ...], prior: '<what earlier rounds already fixed>' }
const ROOT = '/home/lab/workspace/learning/projects/docdistance'
const PRIMER = `${ROOT}/reports/E13-campaign/PRIMER.md`
const LEDGER = 'docs/experiments/wmd-docdistance-experiments.md'
const BUILD_AGENT = 'senior-backend-engineer'
const REVIEW_AGENT = 'devils-advocate:adversarial-reviewer'
const ADJUDICATOR = 'devils-advocate:adjudicator'
const MAX_ROUNDS = 3
const allow = (args && args.allow) || 'moved,n_moved'
// Generated from workflow-review.run3.js and review-args-final3.json (2026-10-12). Do not edit: change the generator inputs and regenerate.
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
  "The HTML reader is a conversion to markdown lines (`html_to_markdown`, standard library parser, no new dependency) in front of the existing markdown reader, so sections, fragment filters and the twin property need no second code path",
  "DEF-DEPS-17 is fixed inside the repository: `init` saves the tokenizer `facebookAI/xlm-roberta-base` beside the SaT weights when the source folder lacks it, and `Segmenter` reads it from that folder; no change to the S3 bucket or to `make sync_models_up`",
  "The `moved` flag on an insertion into a reworded document, where the inserted statement takes the match of a kept one (scratch reading 25 of 150 cases, 123 of 150 under the 1.1.5 rule), is a documented limit and a clause for the next fixture, not a defect: the flag meets ACC-TRANS-77, and a rule for the case is a new hypothesis",
  "The drift audit of 2026-10-10 (reports/E13-campaign/results/reexec-drift-audit.md) is recorded as log lines in both ledgers; recorded Result and Verdict lines are immutable. Three verdict labels of the structure log (E08-H44, E08-H47, E09-H53) contradict their stored artefacts in the committed files as well; whether they get a confirming round is the user's decision",
  "The E04 notebook stays as re-executed on the GPU on 2026-10-09; the audit's CPU run on build 1.1.19 does not reproduce the record of docdistance 1.1.2 (best k 2 at 3.39x) either, so an in-place CPU run would not restore agreement with the record",
  "The user's decisions of 2026-10-10: the reader of a document is chosen from the file extension (.md and .markdown → markdown, .html and .htm → html, every other extension, a file without extension and raw text → plain), never from the content; an explicit `--input-format` / `input_format` or `--markdown` applies to every document; `auto` (the extension rule) is the default. The content detector is removed on purpose",
  "Third registered exception to the additive policy (the user's decision, PRIMER.md, ACC-SEGM-94): with the default reader the numbers for .md, .markdown, .html and .htm files differ from 1.1.5. With `input_format=\"plain\"` every 1.1.5 number is unchanged; `baseline.py verify` and the scripts under reports/E13-campaign/checks pin the plain reader (ACC-SEGM-95)",
  "An HTML paragraph is written on one line with its whitespace collapsed (ACC-SEGM-96); the twin of a hard-wrapped markdown paragraph is given up for it. The `plain` and the `markdown` reader keep the 1.1.5 rule that the segmenter cuts at every line break, and they keep a byte order mark in the first statement: changing either changes shipped numbers and is put to the user",
  "The notebooks E01 to E15 are records on docdistance 1.1.5 (source of commit bd0002e) and are not edited or executed on the campaign build: the note 'Build of record' at the top of both ledgers gives the reasons (the default reader, two notebooks that replace source lines the build changed, one that times a prototype against a command that now holds the memo, 16 that store a source-tree reading). Naming the plain reader in their calls was tried and withdrawn on 2026-10-12, because the argument does not exist on 1.1.5. The showcase notebook 09 is not an experiment record; whether it is re-executed under the new default is put to the user",
  "E16-H81 stays Killed-at-gate; the user's decision for the default is a product decision recorded as a log line under it, not a verdict. Its notebook was recorded on build 1.1.21, which was never committed, so it carries the removed detector as code of its own and was re-executed on 1.1.25 (ACC-SEGM-98); readings that follow the changed HTML converter are logged under the record, the recorded Result and Verdict lines are immutable",
  "Differences between the records of E01 to E11 in the two ledgers and their stored artefacts, beyond the ones the drift audit and the final review logged, are outside this review: the notes at the top of both ledgers say that the artefact governs and that the long tail is not reconciled",
  "Declined findings of earlier rounds, with reasons, are in reports/E13-campaign/results/stage-c-run-wf_73bb512c-5af.json and reports/E13-campaign/input-format-fixes-brief.md",
  "The three solution documents state under Limitations which of their readings come from a fixture file read by the plain reader and which come from the stored statements of data/processed/structure-fixture/statements.json or from statements an experiment segmented itself (no reader of the package), and how the default call differs on the ibm corpus; the readings of the showcase and experiment notebooks are not recomputed under the markdown reader",
  "STRUCTURE_THRESHOLD 0.90 is not moved: a calibration on statements as the command reads them is a hypothesis of its own, named in the next steps of the ledger and in DEF-DOCS-18, and whether it is registered is the user's decision",
  "Declined in round 3 of run wf_2fef834e-695 with reasons (reports/E13-campaign/results/stage-c-run-wf_2fef834e-695.json): the name of the test test_every_call_that_takes_markdown_takes_input_format_with_the_default_plain stays; the documented command of checks/default-reader/diff.py keeps `| tee` and the printed count is the reading, not the exit status"
 ],
 "prior": "Two review runs came before this one. Run wf_366ba93d-451 reviewed the whole uncommitted change in three rounds. Run wf_73bb512c-5af reviewed the input-format work, DEF-DEPS-17, E16-H81 and the drift audit in three rounds and ended at its limit: round 3 read architect SHIP with 0 findings, data-scientist SHIP with 4 minor, bug-hunter DO-NOT-SHIP with 1 major, the tokenizer files of the Hub that pass `init` (now DEF-DEPS-17-1). Its round 1 major, an HTML paragraph that kept the line breaks of its source, was documented as a limit and is now changed in code. Records: reports/E13-campaign/results/stage-c-run-<run id>.json. After it the user decided on 2026-10-10 that the reader follows the file extension and that this rule is the default. Build run wf_7940fe08-cb5 applied reports/E13-campaign/extension-default-brief.md (Z1 to Z7). The build run ended verified on build 1.1.26 (record: reports/E13-campaign/results/stage-b-run-wf_7940fe08-cb5.json). Its verifier found one major finding, a test that could not fail when markdown=True was ignored by 5 of the 8 methods; the repair round added the assertion on .txt files and raw text. It found three minor ones: the memo key for two names of one file (the key now also holds the reader that the extension of the file name chooses, ACC-SEGM-99), the file name ..md (a test case was added) and a search that matches record files (no action). The second verifier read SHIP with 3 minor findings, which the orchestrator applied on 2026-10-12 as build 1.1.27: the section-scan test of the details calls stores its HTML under .txt names; the bullet Memo and log of docs/api-reference.md names the reader of the file name; reports/E13-campaign/checks/default-reader/diff.py compares with the fixed copy reports/E13-campaign/results/E16-H81-metrics-recorded-1.1.21.json and exits with code 1 on a difference; the warning for a mistyped path covers .markdown and .htm (ACC-SEGM-100). Gates on 1.1.27, run by the orchestrator: make test 570 passed and 1 skipped, make lint clean, baseline verify 0 differences on 55 pairs and 3 grounded results, diff.py exit 0 with 55 of 55 distances equal. The E16-H81 notebook was re-executed on 1.1.26 with exit code 0 and 0 error outputs: 1091 of the 1104 readings recorded on 1.1.21 are equal and the 13 others name the environment (log line under the record). Every make install increments the patch number; never pass SKIP_VERSION_INCREMENT. Run wf_2fef834e-695 (reports/E13-campaign/workflow-review.run3.js) reviewed the extension-default build in three rounds and ended at its limit of 3 rounds: round 1 read 3 planned changes (build 1.1.28), round 2 read 1 (build 1.1.29), round 3 read architect SHIP (2 minor), bug-hunter SHIP (1 minor) and data-scientist DO-NOT-SHIP with 1 major: the bullet 'Readings of this document are from the plain reader' of the three solution documents claimed that every number was measured with the plain reader, while the structural calibration of 0.90 (E12-H59) was read on the stored fixture statements, which keep the title line as statement 0, and the command reads 3 of the 55 ibm summary pairs below 0.90 with the plain reader (stored 1.1.5 outputs) and 4 with the default reader. The orchestrator filed and then closed DEF-DOCS-18 on it; the round 3 fix rewrote the bullet in the three documents (build 1.1.30, no source change after 1.1.29); the orchestrator added the clause 'except the readings of the default call that this bullet quotes' to the two shorter bullets, the next-step bullet on a new calibration to the ledger, and corrected three ledger sentences to 'the plain reader, the default up to build 1.1.24'. STRUCTURE_THRESHOLD 0.90 is unchanged; whether a new calibration is registered is the user's decision. The record of the run: reports/E13-campaign/results/stage-c-run-wf_2fef834e-695.json. This run is the confirming review of that delta: two consecutive clean rounds end it."
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

const TARGET = `the part of the uncommitted change of the docdistance project in ${ROOT} that was made after round 3 of review run wf_2fef834e-695 (its record, with the findings, the change plan and the fix report of every round: reports/E13-campaign/results/stage-c-run-wf_2fef834e-695.json). The delta: (1) the Limitations bullet 'Readings of this document and the reader of a file' of docs/solution/wmd-structure-distance-sota.md, docs/solution/wmd-docdistance-solution-sota.md and docs/solution/wmd-source-conditioned-docdistance-solution-sota.md, rewritten per the round 3 change plan; (2) in ${LEDGER}: the note 'Build of record' at the top, the bullet 'Structural calibration on statements as the command reads them (DEF-DOCS-18)' under Next steps, and the sentences under the E14 results table and under Lessons learned and Next steps that now say 'the plain reader, the default up to build 1.1.24'; (3) docs/defects-docdistance.md: DEF-DOCS-18 closed with its evidence line; docs/acc-crit-docdistance.md: the text of ACC-SEGM-98; (4) build 1.1.30 from the round 3 fix (\`make install\` only; no source line changed after 1.1.29 - confirm with \`git diff --stat\` that src/ and tests/ hold nothing the round 3 record does not name). Everything else in the uncommitted change was reviewed in nine rounds over three runs: read it for context, and report a finding on it only when the delta broke it or when a statement of the delta contradicts it`

const LENSES = [
  { lens: 'architect', focus: 'whether the rewritten bullet of each solution document says what the code and the artefacts do and nothing beyond it, is consistent across the three documents and with the ledger note and the two tracker items, and adds no mechanism or obligation the user did not ask for; whether the next-step bullet in the ledger states a hypothesis without pre-deciding it; whether the defect closure names evidence that exists' },
  { lens: 'bug-hunter', focus: 'every number and every claim of the three rewritten bullets and of the ledger next-step bullet against its source: data/processed/structure-fixture/statements.json (title line as statement 0), notebooks/11-kj-structure-fixture.ipynb, reports/E12-H59-structure-threshold-transfer-metrics.json (0.906144, 0 of 72), reports/E13-campaign/baseline/default-path.json.gz (3 of 55 pairs below 0.90 with the plain reader, lowest 0.897057), and a run of distance-structural or DocDistance.structural_distance on the 55 pairs of data/interim/E12-H67/allpairs-55.tsv on the current build with input_format="plain" and with the default (3 and 4 pairs below 0.90); whether `pipeline._load_body` removes the title line for every reader as the bullets state; whether git diff shows a change under src/ or tests/ after build 1.1.29' },
  { lens: 'data-scientist', focus: `whether each reading quoted in the rewritten bullets is attributed to the statements it was measured on (stored fixture statements, a file read by the plain reader, a file read by the default reader) and whether any sentence of the three documents, the README, the CHANGELOG or ${LEDGER} still claims that the plain reader returns the calibration readings or that every reading of a document comes from one reader; whether the calibration section of docs/solution/wmd-structure-distance-sota.md and the bullet contradict each other; whether DEF-DOCS-18 is closed on evidence that answers its repro line; whether the next-step bullet states the right design for a new calibration (the pairs that set the threshold are not the pairs that test it)` },
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
