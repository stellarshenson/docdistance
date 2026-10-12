# data/processed - final fixture dataset

The canonical structure-distance fixture the experiments load. Written by notebook 11 (Stage 3); experiments E07-E11 and the E2E showcase read it directly.

- **Stage role** - pipeline output, the final dataset of `external → interim → processed`
- **Self-contained** - statements, permutations, pairs and metadata; consumers compute embeddings and distances from it
- **Deterministic** - regenerating from the tracked interim inputs reproduces these files byte-for-byte

## Artefacts (`structure-fixture/`)

- `statements.json` - every document segmented into statements, each carrying its block id/name, tier, kind, and article
- `reorder_pool.json` - byte-identical `k`-swap permutations per base across a displacement grid; the order-isolation upper bound
- `pairs.json` - the evaluation pair regimes: cross-summary (diffuse), tier-contrast, paraphrase (content-invariance), section-swap
- `meta.json` - seeds, swap grid, displacement bins, article map, and the paraphrase provenance

## Relevance fixtures

- `relevance-fixture/` - constructed report (the ibm and the Wergeland article under one title) with 24 briefs written for batches E17 and E18, labels and controls; `build_report.py` rebuilds `report.md`
- `relevance-fixture-eu/` - the fixture of batch E19: one European Commission report and 14 notes of the European Parliamentary Research Service. Tracked: `build_fixture.py`, `labels.json` (the target sections of each note, assigned before scoring) and `README.md`
- **Texts not in the repository** - `relevance-fixture-eu/report.md`, `sections.json` and the 14 files of `briefs/` are converted from third-party documents and are re-created on demand: `.venv/bin/python data/processed/relevance-fixture-eu/build_fixture.py` downloads the 15 PDFs into `data/external/`, converts them with `notebooks/14-kj-eu-relevance-fixture.ipynb` and compares the sha256 of the 18 fixture files with `reports/E19-relevance/E19-relevance-registration.json` (exit code 0 when all are equal). It needs `make install` and `docdistance init` in the repository root
- **Provenance** - titles, source URLs and reuse notices of the 15 documents: `data/external/eu-digital-decade-relevance_origin.md`
