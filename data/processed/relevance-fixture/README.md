# Relevance fixture (E17)

One long report and 12 short client briefs with the relevant sections of the report fixed before any scoring. It serves the hypotheses E17-H82 to E17-H86 of `docs/experiments/wmd-docdistance-experiments.md`.

- `report.md` - the two source articles of the repository as one document under one title line; `build_report.py` writes it and moves every heading of an article one level down
- `briefs/brief-01.md` to `brief-12.md` - constructed client briefs, written by the experimenter on 2026-10-12 with the report in view: a title line, one paragraph of four sentences on the situation of a fictitious client that bears on the target sections, one paragraph of two sentences on client facts the report does not cover. No sentence and no figure of the report is copied
- `labels.json` - per brief the part of the report and the leaf headings of its target sections; `neutral_sections` are the two sections of part 1 that restate the detailed sections and are left out of the scoring for the 12 briefs; `controls` are the 7 gold summaries of the ibm article, for which every statement of part 1 is relevant

Limits: two articles on one broad topic, 12 briefs by one author, section-level labels. A report statement outside the target sections can still bear on a brief (the barriers and the workforce sections both speak of skills).
