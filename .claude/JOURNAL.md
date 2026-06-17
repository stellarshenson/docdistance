# Claude Code Journal

This journal tracks substantive work on documents, diagrams, and documentation content.

---

1. **Task - Project initialization** (v0.1.0): Initialized Claude Code configuration for the `docdistance-estimator` project scaffolded from copier-data-science v1.3.9<br>
   **Result**: Created `.claude/CLAUDE.md` importing workspace-level configuration from `/home/lab/workspace/.claude/CLAUDE.md` with the mandatory reinforced bans (no auto tags/version/publish/commits, Makefile-first installs) and the append-only journal rules. Added project context describing the Word Mover's Distance / Optimal Transport purpose - meaningful document distance for agentic conversion/extraction pipelines that run through frontier models without access to logits for KL divergence. Documented the stack (Python 3.13, `uv`, `typer`, `loguru`, `ruff`, `pytest`) and strengthened rules for notebook standards, background-job logging, and optimal-transport metric correctness. Project already had a local git repo from copier (no commits yet), so git init was skipped.

2. **Task - WMD paper digest** (v0.1.0): Processed the source PDF `[paper] From Word Embeddings To Document Distances.pdf` into a markdown digest<br>
   **Result**: Extracted all 10 pages of the Kusner et al. 2015 ICML paper with `pdfplumber` (poppler was unavailable for direct PDF rendering) and wrote `references/papers/from-word-embeddings-to-document-distances.md` following modus primaris. The digest covers the BOW/TF-IDF distance problem, the nBOW representation, the word travel cost `c(i,j) = ||x_i - x_j||₂`, the EMD/transportation linear program, the WCD and RWMD lower bounds with prefetch-and-prune, the eight-dataset kNN results (0.42 of BOW error, best on six of eight), limitations around out-of-vocabulary terms, and a closing note on relevance to this project's logit-free document-distance use case.
