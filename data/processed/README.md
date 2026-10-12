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
- `relevance-fixture-eu/` - `report.md` (European Commission, *State of the Digital Decade 2024*, Annex 1, reused under CC BY 4.0, converted to markdown), `labels.json`, `sections.json` and `README.md`; batch E19
- **Note texts not in the repository** - the 14 files of `relevance-fixture-eu/briefs/` are EPRS "At a glance" notes, and their reuse notice asks for prior notice to the European Parliament for a reproduction. `notebooks/14-kj-eu-relevance-fixture.ipynb` rebuilds them from the PDFs that `data/external/download-fixtures.py` fetches; the sha256 of each file is stored in `reports/E19-relevance/E19-relevance-registration.json`
- **Provenance** - `data/external/eu-digital-decade-relevance_origin.md`
