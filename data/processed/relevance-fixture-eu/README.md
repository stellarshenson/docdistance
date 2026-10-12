# relevance-fixture-eu - one long EU report and 14 short briefing notes

Text fixture for relevance work on documents of different length: one long report and 14 short notes on
related digital-policy subjects, converted from public PDFs to clean markdown. Nothing here is scored; no
note is labelled against the report. Labels are assigned elsewhere.

## Contents

- `report.md` - State of the Digital Decade 2024, Annex 1 (European Commission): 883 statements as read by the package reader, 113 headings
- `briefs/<PE>-<slug>.md` - 14 EPRS "At a glance" notes (European Parliamentary Research Service), 22 to 56 statements each
- `sections.json` - heading tree of the report: level, number, title, role, parent, and the count of statements below each heading (`statements_own` directly below, `statements_total` with the deeper headings)

Provenance, titles, source URLs and reuse notices of the 15 documents: `data/external/eu-digital-decade-relevance_origin.md`.

## Format

- first line `# <title>`, which the markdown reader removes; one paragraph per line, never a wrapped paragraph
- headings follow the reader's rules: one statement per heading, list item (`- `) and paragraph sentence
- report heading level follows the numbering depth (`##` for `1.`, `###` for `1.1.`, `####` for `2.1.3.`); unnumbered
  sub-headings, recommendation box titles and box labels sit below their numbered parent (never past six hashes)
- notes: `##` section headings (14 pt in the PDF), `###` bold sub-headings and box titles

## How it was built

Notebook `notebooks/14-kj-eu-relevance-fixture.ipynb` reads the PDFs in `data/external/`, keeps text by font and size
with `pdfplumber`, joins wrapped lines into paragraphs, repairs words split at a line break, and writes these files.
Statements are counted with the markdown segmentation of the package (`docdistance` 1.1.32), without an encoder.

## Removed

- report: cover, legal notice, blank page and imprint pages, page numbers, footnote text and markers, chart labels, figure captions
- notes: series labels, running header, EPRS footer with author, reference and date, disclaimer and copyright line, chart,
  table and infographic text with captions, figure source lines, footnote markers (`*`)

## Limits

- the text is not edited; typing errors of the sources stay (for example `alҐso`, `four in10`, `74°%` and `1025` for 10 to the power 25)
- paragraph, heading and box detection rely on font size and line spacing; see the notebook for the pages where it can fail
- the annex PDF has no table of contents (pages 3 and 75 are text-free decoration); its recommendation boxes become a box title and box labels, so 46 of 64 leaf sections hold fewer than 5 statements; all of them have the role `boxtitle` or `boxlabel` in `sections.json`, none is a numbered section or a sub-heading
- box titles and labels under a five-level numbered heading share level six, so the box title is an empty leaf there
- the notes keep the closing box of procedure references (first-reading report, rapporteur) where the PDF has one
- the PDFs are public documents kept for calibration and fixtures only; they are not redistributed beyond the licence and not used for training
