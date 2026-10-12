export const meta = {
  name: 'docdistance-e13-build',
  description: 'Stage B of the E13 campaign: implement the work packages in order (opus implementer, adversarial verify, one repair round each), then one final gate. The three-lens review is stage C (workflow-review.js)',
  phases: [
    { title: 'Build', detail: 'one implementer per work package, in order, one working tree', model: 'opus' },
    { title: 'Verify', detail: 'adversarial check of each package against its criteria and defects' },
    { title: 'Gate', detail: 'lint, tests, baseline and Python 3.12 syntax on the final tree' },
  ],
}

// args: { packages: [{ key, title, items: [ids], files: [paths], brief, grounded, verify_note }], allow: 'moved,n_moved' }
// The package list lives in reports/E13-campaign/build-args.json; pass its content as `args`.
const ROOT = '/home/lab/workspace/learning/projects/docdistance'
const PRIMER = `${ROOT}/reports/E13-campaign/PRIMER.md`
const BUILD_MODEL = 'opus'
const BUILD_AGENT = 'senior-backend-engineer'
const REVIEW_AGENT = 'devils-advocate:adversarial-reviewer'
// Generated from workflow-build.js and build-args-followup.json (2026-10-10). Do not edit: change the two inputs and regenerate.
const BUILD_ARGS = {
 "allow": "moved,n_moved",
 "packages": [
  {
   "key": "review-followup",
   "title": "the defect and the confirmed items that review run wf_366ba93d-451 handed to the orchestrator",
   "items": [
    "DEF-DEPS-17",
    "ACC-TRANS-84"
   ],
   "files": [
    "src/docdistance/config.py",
    "src/docdistance/bootstrap.py",
    "src/docdistance/encoders.py",
    "src/docdistance/distance.py",
    "src/docdistance/cli.py",
    "tests/test_bootstrap.py",
    "tests/test_e2e.py",
    "tests/test_segmentation.py",
    "tests/test_cli.py",
    "docs/aws-deployment-reference.md",
    "docs/cli-reference.md",
    "docs/api-reference.md",
    "README.md",
    "CHANGELOG.md",
    "docs/solution/wmd-docdistance-solution-sota.md",
    "docs/solution/wmd-structure-distance-sota.md",
    "docs/solution/wmd-source-conditioned-docdistance-solution-sota.md",
    "notebooks/05-kj-source-conditioned-distance.ipynb",
    "reports/E13-campaign/checks/"
   ],
   "brief": "Read reports/E13-campaign/review-followup-brief.md in full. It lists one defect (G1, DEF-DEPS-17) and five entries (G2 to G6); each was confirmed against the tree at version 1.1.18. Apply every entry. Read `git diff HEAD --stat -- src tests` first: the tree holds the uncommitted build of the campaign. In `notes` report one line per entry G1 to G6: done or not done, and the test or the lines that changed. The item ACC-TRANS-84 is met by G2. No network request at any step: run every real-model step with HF_HUB_OFFLINE=1.",
   "verify_note": "the brief of this package is reports/E13-campaign/review-followup-brief.md. Check every entry G1 to G6 against the tree: applied as written, or reported as not applied with a reason that holds. For G1 run the real-model test yourself and confirm by your own means that it fails without the two source edits (work on a copy of the tree under the session scratchpad, never revert files in the repository). For G5 run the script and compare its three readings with the ledger values. An entry applied beyond what it says is a finding. No network request: HF_HUB_OFFLINE=1 for every real-model step.",
   "grounded": true
  },
  {
   "key": "input-format",
   "title": "HTML reader, format detector and the option input_format (user request of 2026-10-10)",
   "items": [
    "ACC-SEGM-85",
    "ACC-SEGM-86",
    "ACC-SEGM-87",
    "ACC-CLI-88",
    "ACC-SEGM-89",
    "ACC-SEGM-90",
    "ACC-SEGM-91",
    "ACC-SEGM-92"
   ],
   "files": [
    "src/docdistance/encoders.py",
    "src/docdistance/pipeline.py",
    "src/docdistance/cli.py",
    "src/docdistance/__init__.py",
    "tests/test_segmentation.py",
    "tests/test_pipeline.py",
    "tests/test_cli.py",
    "tests/test_e2e.py",
    "tests/test_sections.py",
    "README.md",
    "docs/cli-reference.md",
    "docs/api-reference.md",
    ".claude/skills/docdistance/SKILL.md",
    "CHANGELOG.md"
   ],
   "brief": "Read reports/E13-campaign/input-format-brief.md in full: design, tests, documents, gates. The default reader does not change in this package; the default of the new option is plain. The package review-followup was built before this one in the same tree. In `notes` report the detector constants you ended with and any boundary case that forced a change.",
   "verify_note": "the brief of this package is reports/E13-campaign/input-format-brief.md. Check each criterion against the code and by running the tests. Try to break `detect_format` and `html_to_markdown` with inputs of your own: malformed and nested HTML, markdown with inline HTML and with HTML blocks, an HTML fragment without html and body tags, a table cell that holds a pipe, entities, a document of 1 MB. Check that a call without input_format and without markdown is unchanged (the baseline line) and that the memo key separates the values. A design element the brief does not ask for is a finding.",
   "grounded": true
  }
 ]
}
const packages = (args && args.packages) || BUILD_ARGS.packages
const allow = (args && args.allow) || BUILD_ARGS.allow
const baselineCmd = pkg => `CI=true .venv/bin/python reports/E13-campaign/baseline.py verify --allow ${allow}${pkg && pkg.grounded ? ' --grounded' : ''}`

const BUILD_SCHEMA = {
  type: 'object',
  required: ['package', 'status', 'items', 'files_changed', 'make_test', 'make_lint', 'baseline_verify'],
  properties: {
    package: { type: 'string' }, status: { enum: ['done', 'partial', 'blocked'] },
    items: { type: 'array', items: { type: 'object', required: ['id', 'met', 'evidence'], properties: { id: { type: 'string' }, met: { type: 'boolean' }, evidence: { type: 'string' } } } },
    files_changed: { type: 'array', items: { type: 'string' } },
    tests_added: { type: 'array', items: { type: 'string' } },
    make_test: { type: 'string' }, make_lint: { type: 'string' }, baseline_verify: { type: 'string' },
    out_of_scope_needs: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
}

const FINDING = { type: 'object', required: ['severity', 'claim', 'evidence'], properties: { severity: { enum: ['blocking', 'major', 'minor'] }, file: { type: 'string' }, line: { type: 'integer' }, claim: { type: 'string' }, evidence: { type: 'string' } } }

const VERIFY_SCHEMA = {
  type: 'object', required: ['package', 'verified', 'findings', 'gates'],
  properties: {
    package: { type: 'string' }, verified: { type: 'boolean' },
    findings: { type: 'array', items: FINDING },
    gates: { type: 'object', required: ['make_test', 'make_lint', 'baseline_verify'], properties: { make_test: { type: 'string' }, make_lint: { type: 'string' }, baseline_verify: { type: 'string' } } },
    items_not_met: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
}

const implPrompt = (pkg, done) => `You are the IMPLEMENTER for work package "${pkg.key}" (${pkg.title}) of the docdistance E13 campaign, stage B (build). Repo root: ${ROOT}. Work from the repo root, in the main working tree. Execution model: ${BUILD_MODEL}.

FIRST read ${PRIMER} in full - environment, hard rules, the stage B contract and return schema. THEN read every item of this package in docs/acc-crit-docdistance.md and docs/defects-docdistance.md (read only): ${pkg.items.join(', ')}. Each item's body is the assertion, its test: or repro: line is the check, its mechanism: or root-cause: line is the intended design.

Files this package owns: ${pkg.files.join(', ')}. An edit outside them is reported in out_of_scope_needs, not made.

Package brief: ${pkg.brief}

Packages already built in this tree, in order (their changes are present; build on them, do not redo them):
${done.length ? done.map(d => `- ${d.key}: ${d.result ? d.result.status : 'no result'} - files ${d.result ? (d.result.files_changed || []).join(', ') : 'n/a'}`).join('\n') : '- none'}

Procedure: read the code you will change and its callers before the first edit; write the failing test for each item, then the change; run \`make install\` once (new build to test - it increments the patch version, which is wanted); run \`make format\`, \`make lint\`, \`make test\`; run the guardrail \`${baselineCmd(pkg)}\` and read its last line. A difference the guardrail reports is a regression of a shipped number: fix the change, never the baseline. Return the stage B result JSON only, one entry in items per id listed above.`

const verifyPrompt = (pkg, res) => `Lens: qa-engineer. Target: work package "${pkg.key}" (${pkg.title}) of the docdistance E13 build, as it stands in the working tree of ${ROOT}. Find where the implementer's claim does not hold. You may run commands; you never edit, stage or revert a file.

Read ${PRIMER} (hard rules, stage B contract). Read the package items in docs/acc-crit-docdistance.md and docs/defects-docdistance.md: ${pkg.items.join(', ')}. Read the change: \`git diff HEAD -- ${pkg.files.join(' ')}\` and \`git status --short\`.

Implementer result:
${JSON.stringify(res, null, 2)}
${pkg.verify_note ? `\nAlso check: ${pkg.verify_note}\n` : ''}
Checks: (1) each item: the assertion holds in the code and a test exercises it - run the item's test: or repro: line where it is a command; an item claimed met without a test that would fail on the unchanged tree is MAJOR; (2) run \`make test\`, \`make lint\` and \`${baselineCmd(pkg)}\` yourself and report their last lines in gates - a red gate is BLOCKING; (3) additive policy: no shipped number, default or detail-file key changed except the registered moved-flag exception; (4) files outside the package list were not edited by this package; (5) a new quantity's docstring says whether it is a metric, a lower bound or a score, and nothing approximate is called exact; (6) syntax valid on Python 3.12 (\`uvx ruff@0.16.2 check --target-version py312\`); (7) no dead code, no speculative parameter, no fallback for an impossible state; (8) no commit, tag, hand edit of the version, journal or tracker write happened - the patch increments that \`make install\` makes are expected and are not a finding; you never run \`make install\` yourself. verified=true only with no blocking or major finding. Return the JSON only.`

const repairPrompt = (pkg, res, ver, done) => `${implPrompt(pkg, done)}

REPAIR ROUND. An adversarial verifier rejected the package. Your previous result:
${JSON.stringify(res, null, 2)}
Verifier findings (fix every blocking and major one; the items stay as written):
${JSON.stringify(ver, null, 2)}
Apply the fixes in place, run \`make install\` once (new build to test), re-run the three gates and return the updated result JSON only.`

const built = []
phase('Build')
if (!packages.length) { log('no packages in args - nothing to build'); return { packages: [] } }
log(`stage B: ${packages.length} packages in order on ${BUILD_MODEL}: ${packages.map(p => p.key).join(' -> ')}`)
for (const pkg of packages) {
  let res = await agent(implPrompt(pkg, built), { label: `build:${pkg.key}`, phase: 'Build', schema: BUILD_SCHEMA, model: BUILD_MODEL, agentType: BUILD_AGENT })
  let ver = res ? await agent(verifyPrompt(pkg, res), { label: `verify:${pkg.key}`, phase: 'Verify', schema: VERIFY_SCHEMA, effort: 'high', agentType: REVIEW_AGENT }) : null
  let repaired = false
  if (res && ver && !ver.verified) {
    log(`${pkg.key}: verifier rejected (${ver.findings.length} findings) - repair round`)
    const res2 = await agent(repairPrompt(pkg, res, ver, built), { label: `repair:${pkg.key}`, phase: 'Build', schema: BUILD_SCHEMA, model: BUILD_MODEL, agentType: BUILD_AGENT })
    if (res2) {
      res = res2; repaired = true
      ver = await agent(verifyPrompt(pkg, res), { label: `reverify:${pkg.key}`, phase: 'Verify', schema: VERIFY_SCHEMA, effort: 'high', agentType: REVIEW_AGENT })
    }
  }
  log(`${pkg.key}: ${res ? res.status : 'no result'} - verified=${ver ? ver.verified : 'n/a'}`)
  built.push({ key: pkg.key, result: res, verify: ver, repaired })
  if (!res || res.status === 'blocked') { log(`${pkg.key}: blocked - later packages still run, the orchestrator decides`) }
}

phase('Gate')
const final = await agent(`Lens: qa-engineer. Final gate of the docdistance E13 build in ${ROOT}. You run commands and read; you never edit. Run and report the last line of each: \`make lint\`; \`make test\`; \`${baselineCmd({ grounded: true })}\`; \`uvx ruff@0.16.2 check --target-version py312\`; \`git status --short\`. Then confirm against ${PRIMER} that no commit or tag happened (\`git log -1 --oneline\`, \`git tag --points-at HEAD\`) and report the version line (\`grep -n '^version' pyproject.toml\`) in notes - patch increments made by \`make install\` are expected. verified=true only when every gate is green. Return the JSON only, package "final-gate".`, { label: 'verify:final-gate', phase: 'Gate', schema: VERIFY_SCHEMA, agentType: REVIEW_AGENT })
return { packages: built, final }
