# Reader pin in the experiment notebooks - brief for the editing agents

> Status, 2026-10-12: only the section 'E16-H81 only' was applied. The pin was withdrawn for the notebooks E12 to E15 and their files were restored byte for byte: they are records on docdistance 1.1.5, where the argument `input_format` does not exist, and four of them cannot run on the campaign build for other reasons. The note 'Build of record' at the top of `docs/experiments/wmd-docdistance-experiments.md` holds the reasons.

Written 2026-10-12. Build 1.1.25 changed the default reader of every pipeline call: `input_format` now defaults to `"auto"`, which chooses the reader from the file extension (`.md`, `.markdown` → markdown reader; `.html`, `.htm` → HTML reader; every other file and raw text → plain reader). This is the user's decision of 2026-10-10. Up to 1.1.24 a call without `markdown=True` read every document with the plain reader.

Every experiment in `notebooks/experiments/` was registered and recorded with the plain reader. A notebook that passes a `.md` file to a pipeline call without naming a reader would read it with the markdown reader on a re-execution and would no longer reproduce its stored readings. The task: name the plain reader in those calls, so that a re-execution on 1.1.25 returns the stored readings. With `input_format="plain"` build 1.1.25 returns the numbers of 1.1.5 bit for bit (`reports/E13-campaign/baseline.py verify` reads 0 differences on 55 pairs and 3 grounded results).

## Rules

- Work in `/home/lab/workspace/learning/projects/docdistance`. Edit only the notebooks assigned to you. No edit under `src/`, `tests/`, `docs/`, `reports/`, no `make install`, no package install, no network request, no commit. Never read `**/@archive` or `**/.ipynb_checkpoints`
- Edit the notebook file with a script that changes cell sources only (`nbformat` version 4, or `json` with `indent=1` and `ensure_ascii=False`). Keep every output, execution count and metadata field. Keep scripts in your own scratch folder, not in the repository
- **Do not execute a notebook.** The orchestrator re-executes them after the edits, one after another, and compares every stored reading. A notebook run writes its metrics file under `reports/` and would overwrite the record
- Change nothing but the reader: no constant, no logic, no reading, no conclusion. Do not rewrite a line that needs no pin
- Read the signatures in `src/docdistance/pipeline.py` and `src/docdistance/cli.py` first: `embed_statements`, `semantic_distance`, `semantic_distance_with_details`, `structural_distance`, `structural_distance_with_details`, `distance_wrt_source`, `distance_wrt_source_with_map` of `DocDistance`, the one-shot functions `semantic_distance`, `structural_distance`, `source_conditioned_distance`, and the option `--input-format` of the commands `distance-semantic`, `distance-structural`, `distance-wrt-source`

## The pin

- A pipeline call in which a document argument is, or can be, a file whose name ends in `.md`, `.markdown`, `.html` or `.htm` gets `input_format="plain"`
- A call whose documents are raw text in every execution stays as it is: raw text is read by the plain reader under `auto`. When you cannot tell from the code whether an argument is text or a file, pin it: the pin does not change the reading of raw text
- A call with `markdown=True` stays as it is: it names the markdown reader, and `markdown=True` beside `input_format="plain"` raises `ValueError`. Where `markdown` is a variable, the call must name `plain` only when the variable is false
- A notebook that replaces `DocDistance.embed_statements` by its own function (a memo prototype, a counter): the pipeline methods pass `input_format` on to `self.embed_statements` only when it differs from `"auto"` (`_seg_kwargs`). The replacement must therefore either take and forward `input_format`, or name the plain reader in its own call of the original function. Choose the form with the fewest changed lines and say which one you chose
- A command line (in process through `CliRunner` or `cli._run_batch`, or in a child process): `--input-format plain`, never beside `--markdown`. For `cli._run_batch` and similar internal entry points read their signature in `src/docdistance/cli.py` at 1.1.25
- A check inside the notebook that compares two paths of the same computation (memo on against memo off, batch against one-shot) must read both sides with the same reader after your edit
- At the first pinned line of a notebook add the comment `# plain reader named: from docdistance 1.1.25 the default reader follows the file extension`. In the markdown cell that describes the configuration or the data, add one sentence: `Every call on a file names the plain reader (input_format="plain"): the experiment was recorded with it, and from docdistance 1.1.25 the default reader follows the file extension.` Add nothing else to a markdown cell

## Check before you report

- Every edited code cell parses: `ast.parse` of the cell source (cells that hold IPython magics are checked without the magic lines)
- `git diff --stat` (or a comparison with your own copy for an untracked notebook) shows changes in your notebooks only, and the changed lines are the pins, the comment and the sentence
- A notebook in which no call needs a pin is left byte-identical: report it with the reason

## Report

Per notebook: every changed line as `cell <index>: <old> → <new>`; the calls left without a pin, each with the reason; for a replaced `embed_statements`, the form you chose; anything you could not decide. State plainly that no notebook was executed.

## E16-H81 only

`notebooks/experiments/E16-H81-kj-autodetected-reader-default.ipynb` was recorded on docdistance 1.1.21, when `input_format="auto"` chose the reader from the content with `docdistance.encoders.detect_format`. Build 1.1.25 removed that function, and `auto` now means the extension rule. The notebook must run on 1.1.25 and still measure what its hypothesis registered (the content detector as the default reader).

- Carry the detector in the notebook: one code cell that holds `detect_format` and everything only it needs (constants, regular expressions, helpers), copied without change from the last build that shipped it. The orchestrator gives you the path of that copy of `encoders.py`. Name the origin in a comment at the top of the cell. Remove the import of `detect_format` from the package
- The arm that the notebook calls `auto` is the content detector. In 1.1.21 the reader of a document under `auto` was `detect_format` of the file text as read from disk (before the title lines are removed), and the document was then read exactly as with that explicit reader. Keep that meaning: add one cell that, for every document the notebook reads under `auto`, compares the detector's reader with the reader of the extension rule (`docdistance.pipeline._reader_by_extension`) and stores the result in the metrics. Where the two agree for every document of a call, `input_format="auto"` on 1.1.25 reads the same statements and the call can stay. Where they differ for a document, the call must name the detector's reader for that document; when one call would need two different explicit readers, report it and do not guess
- The arm `plain` must name `input_format="plain"` in every call (check what the notebook's helper returns for it)
- The HTML converter changed after 1.1.21 (a paragraph is one line, `head` is no dropped element). Readings of the HTML twins can therefore differ on re-execution. Do not tune anything to the stored readings; the orchestrator records the difference
- Add one markdown cell directly under the title cell: `Re-execution note (2026-10-12): the notebook was recorded on docdistance 1.1.21. Build 1.1.25 removed the content detector and made the extension rule the default reader. The detector is carried in this notebook as it stood in the last build that shipped it, so that the registered measurement can be repeated.` Do not edit the other markdown cells
