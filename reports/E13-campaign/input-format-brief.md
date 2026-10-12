# Input format: HTML reader and autodetection - brief for the implementer

Written 2026-10-10. The user asked for autodetection of HTML, markdown and other inputs, so that the reader no longer depends on the flag `--markdown` alone. The intended use is a conversion pipeline that compares an HTML document with its markdown conversion. This package adds an HTML reader, a detector and one option. The default reader does not change in this package: a default of `auto` changes 1.1.5 numbers for markdown inputs and is the user's decision after a registered experiment.

Rules that hold for every entry: the stage B contract of `PRIMER.md` (additive policy with its two registered exceptions, `make install` once at the end and never `SKIP_VERSION_INCREMENT`, the three gates), no existing test is renamed, no new dependency (the HTML parser is `html.parser` of the standard library), no edit to the trackers, the ledger or the journal, no commit, no network request.

## Design

Three readers exist after this package: `plain` (the prose segmenter on the whole text, the default, unchanged), `markdown` (`split_markdown`, unchanged) and `html`. A fourth value, `auto`, chooses one of the three per document from its content.

**HTML reader = conversion to markdown lines, then the markdown reader.** `html_to_markdown(text) -> str` in `src/docdistance/encoders.py` writes the block structure of an HTML document as the markdown lines `split_markdown` already reads. The text that reaches the section scan, the fragment filters and `split_markdown` is the converted text, so the section fields, `--min-words` and `--skip-table-fragments` work for an HTML input as for a markdown input, without a second code path.

| HTML | written as |
|---|---|
| `h1` to `h6` | `#` repeated by the level, a space, the text on one line |
| `li` (in `ul` or `ol`, nested or not) | `- ` and the text on one line |
| `tr` | `\| cell \| cell \|` from its `td` and `th` texts; a `\|` inside a cell becomes a space |
| `pre` | a fenced block with the lines of its text |
| `p`, `div`, `blockquote`, `section`, `article`, `dt`, `dd`, `figcaption`, `caption`, and text outside any block | a paragraph followed by a blank line |
| `br`, `hr` | a line break inside the paragraph, a paragraph break |
| `script`, `style`, `head`, `template`, comments | dropped with their content |
| inline tags (`a`, `em`, `strong`, `span`, `code`, `sup`, ...) | the text only; `img` is dropped |

Character references are decoded. Whitespace inside a heading, a list item and a cell collapses to single spaces. A paragraph keeps the line breaks of its source text (each line stripped, empty lines removed), so a rendered hard-wrapped markdown paragraph gives the prose segmenter the same lines as the markdown original. Malformed HTML does not raise: the parser of the standard library is lenient, and unclosed blocks end at the next block or at the end of the text.

**Detector.** `detect_format(text) -> str` in `src/docdistance/encoders.py` returns `html`, `markdown` or `plain` from the content only. The file extension is not read: raw text has none, and with a content rule `auto` differs from `plain` only for a document that holds markup. Rule, in this order:

1. `html` when the text holds `<!doctype html`, `<html` or `<body` (case-insensitive), or when at least half of its non-blank lines start with a tag. A markdown document with inline tags (`<br>`, `<sup>`, `<span id=...>`, which this repository's own documents hold) must not read as `html`
2. `markdown` when the text holds at least one of: an ATX heading line (`#` to `######`, a space, text), a table separator row, a fence line, or two list-item lines
3. `plain` otherwise

Keep the numbers of the rule as named module constants and state the rule in the docstring. If a boundary case shows that a number must differ, change the constant, add the case as a test and report it.

**Option.** One parameter, `input_format`, with the values `plain` (default), `markdown`, `html`, `auto`:

- command line: `--input-format` on `distance-semantic`, `distance-structural` and `distance-wrt-source`. `--markdown` stays and equals `--input-format markdown`. `--markdown` together with `--input-format html`, `auto` or `plain` given explicitly is a usage error (exit code 2, the code the commands already use for conflicting arguments)
- Python: `input_format: str = "plain"` on every method and one-shot function of `src/docdistance/pipeline.py` that takes `markdown` today. `markdown=True` stays and equals `input_format="markdown"`; `markdown=True` with another non-default `input_format` raises `ValueError`; an unknown value raises `ValueError` that lists the four values
- resolution happens once per document in `embed_statements`: `auto` → `detect_format(text)`; `html` → `text = html_to_markdown(text)`, then the markdown path. One DEBUG log line per document names the reader that ran and whether it was detected
- `_seg_kwargs` passes `input_format` only when it is not the default, like the other options
- `_memo_key` carries the requested value (`plain`, `markdown`, `html`, `auto`) in place of the `markdown` flag, and is still taken before the file is read
- the marker warning of the default reader (`_warn_marker_statements`) is logged when the reader that ran is `plain`, also under `auto`; its advice names `--input-format auto` beside `--markdown`
- a call without `input_format` and without `markdown` keeps the call shape and every number of today: the baseline reads 0 differences

Keep the change inside `src/docdistance/encoders.py`, `src/docdistance/pipeline.py`, `src/docdistance/cli.py` and `src/docdistance/__init__.py` (export `detect_format` and `html_to_markdown` like `split_markdown` if that one is exported; otherwise export neither).

## Tests

No model is needed for any of them; use the stub segmenter and encoder patterns of `tests/test_segmentation.py` and `tests/test_pipeline.py`.

- `html_to_markdown`: one test per row of the table above, plus nested lists, a table with `th`, character references, a `script` and a `style` block, an unclosed `p`, text outside any block
- twin property: one fixture document written twice in the test file, as markdown and as HTML (two heading levels, a nested list, a table, two paragraphs of which one is hard-wrapped, a code block). `split_markdown(html_to_markdown(html), split)` and `split_markdown(md, split)` return equal `embedded` lists, with a deterministic stub for `split`
- `detect_format`: a full HTML document, an HTML fragment without `html` and `body`, a markdown document with inline tags, a markdown document with one heading only, prose with one line that starts with `- ` (plain), prose without markup (plain), the empty string (plain). Corpus check as one parametrised test: every `*.md` file under `data/interim/exec-summaries/` and `data/external/` reads as `markdown` or `plain`, none as `html`
- `embed_statements` with stubs: `input_format="html"` on the HTML twin returns the statements of `markdown=True` on the markdown twin; `input_format="auto"` chooses `html`, `markdown` and `plain` for the three kinds of document; the section fields of a details call are equal for the two twins; the memo holds separate entries for one file read under two values; the marker warning is logged under `auto` for a plain document and not for a markdown one
- command line: `--input-format auto` reaches the pipeline for the three commands (single pair and `--pairs`); `--markdown --input-format html` exits with code 2; an unknown value exits with code 2; `--markdown` alone still equals `--input-format markdown`
- each new test fails when its behaviour is removed; say for which tests you confirmed that and how

## Documents

From the code after the change: `README.md` (the feature bullet 'Markdown-aware' near line 25 and the example near line 93), `docs/cli-reference.md` (the option table of the three commands and the warnings bullet), `docs/api-reference.md` (the methods, `detect_format`, `html_to_markdown`, the conversion table), `.claude/skills/docdistance/SKILL.md` (where it names `--markdown`), `CHANGELOG.md` (unreleased section). In `docs/cli-reference.md` complete the option tables of the three distance commands from the help strings of `src/docdistance/cli.py`: `--home`, `--markdown`, `--min-words`, `--skip-table-fragments`, `--pairs` and `--input-format` have a row in each table (round 3 of the review found the first five missing in two or three tables). State the detector rule with its numbers, the conversion table, that the default is `plain`, and that `auto` on a markdown file returns other numbers than the default reader.

## Gates and report

After all entries: `make install` once, then `make lint`, `make test`, `CI=true .venv/bin/python reports/E13-campaign/baseline.py verify --allow moved,n_moved --grounded` (expected result line: `verify: 55 pairs, 3 grounded, 0 difference(s)`). Report per criterion of this package: met or not, with the test names. Report the detector constants you ended with, any boundary case that forced a change, the files changed, the version after `make install`, the three gate result lines verbatim, and anything you did not do with the reason.
