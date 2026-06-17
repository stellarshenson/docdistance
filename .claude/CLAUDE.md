<!-- @import /home/lab/workspace/.claude/CLAUDE.md -->

# Project-Specific Configuration

This file imports workspace-level configuration from `/home/lab/workspace/.claude/CLAUDE.md`.
All workspace rules apply. Project-specific rules below strengthen or extend them.

The workspace `/home/lab/workspace/.claude/` directory contains additional instruction files
(MERMAID.md, NOTEBOOK.md, DATASCIENCE.md, GIT.md, and others) referenced by CLAUDE.md.
Consult workspace CLAUDE.md and the .claude directory to discover all applicable standards.

## Mandatory Bans (Reinforced)

The following workspace rules are STRICTLY ENFORCED for this project:

- **No automatic git tags** - only create tags when user explicitly requests
- **No automatic version changes** - only modify version in package.json/pyproject.toml/etc. when user explicitly requests
- **No automatic publishing** - never run `make publish`, `npm publish`, `twine upload`, or similar without explicit user request
- **No manual package installs if Makefile exists** - use `make install` or equivalent Makefile targets, not direct `pip install`/`uv install`/`npm install`
- **No automatic git commits or pushes** - only when user explicitly requests

## Project Context

`docdistance-estimator` computes meaningful semantic distance between two documents using the
Word Mover's Distance (WMD) / Optimal Transport theory from Kusner et al. 2015 (`From Word
Embeddings To Document Distances`). The intended use is agentic document conversion and extraction
pipelines that operate through frontier models, where token-level logits are unavailable and KL
divergence cannot be computed directly - WMD provides an embedding-grounded distance instead.

**Technology Stack**:
- Python 3.13, `uv` environment manager, environment name `docdistance-estimator`
- Package layout under `src/docdistance_estimator/` (config, dataset, features, modeling, plots)
- `typer` CLI, `loguru` logging, `tqdm` progress, `python-dotenv` for secrets
- `ruff` for linting and formatting, `pytest` for tests
- Scaffolded from copier-data-science template v1.3.9; secrets via encrypted `.env`

**Conventions**:
- Notebooks: number-initials-description prefix (e.g. `01-jqp-data-exploration.ipynb`)
- Data: `raw/` immutable, `interim/` for transforms, `processed/` for final datasets
- Reference papers and their markdown digests live under `references/papers/`
- Use Makefile targets (`make install`, `make test`, `make lint`, `make format`, `make build`)

## Journal Rules (Project-Specific)

- **APPEND ONLY**: New journal entries MUST be appended at the end of the file, never inserted between existing entries
- Entries maintain strict chronological order by position - the last entry in the file is always the most recent work
- Never reorder, move, or insert entries out of sequence
- The Stellars **journal plugin** is the canonical tool for this file: create via `/journal:create`, append via `/journal:update`, archive via `/journal:archive`. The `journal:journal` skill auto-triggers on any mention of "journal" and runs `journal-tools check` after every write
- Direct edits to `JOURNAL.md` are a last resort - prefer the plugin so modus secundis format, continuous numbering and append-only order are enforced automatically

## Strengthened Rules

- **Data science standards apply**: follow the `notebook-standards`, `datascience`, and `rich-output` skills for all notebooks and scripts - GPU selection before torch/tensorflow imports, structured section order, centralized configuration cell
- **Background jobs log to `logs/`**: any long-running training or embedding job must `tee` to `logs/<name>.log` with a `logs/README.md` describing each log
- **Optimal-transport correctness**: WMD is a metric only because the underlying word cost is a metric - preserve this property; document any approximation (WCD, RWMD) as a lower bound, never conflate it with exact WMD
