# data/external - third-party source documents

Raw source documents from third parties, with their provenance records and a fetch script. This is the entry point of the data-prep pipeline (notebook 11); nothing here is produced by the project.

- **Stage role** - pipeline input of `external → interim → processed`; the PDFs are converted to text in `data/interim/` by notebook 11
- **PDFs are gitignored** - binaries are not committed; re-fetch with `python data/external/download-fixtures.py`
- **Provenance is tracked** - each source has a `<name>_origin.md` sidecar with title, publisher, source URL, and the calibration-only usage statement
- **Sources retained for calibration and fixtures only** - not redistributed, not used for training, not part of any published artifact

## Artefacts

- `impact-of-ai-on-society.pdf` (ignored) + `impact-of-ai-on-society_origin.md` - Wergeland *Impact of AI on Society* curriculum, a two-column InDesign PDF; the E11 second article
- `ibm-enterprise-ai-adoption.pdf` (ignored) + `ibm-enterprise-ai-adoption_origin.md` - IBM Global AI Adoption Index 2023 press release; basis of the exec-summary fixture
- `eu-digital-decade-2024-annex1.pdf` and 14 `eprs-ata-<PE>-<slug>.pdf` (ignored) + `eu-digital-decade-relevance_origin.md` - Commission *State of the Digital Decade 2024* Annex 1 and 14 EPRS "At a glance" notes; one sidecar for all 15; converted to markdown by notebook 14 into `data/processed/relevance-fixture-eu/`; the converted texts are not tracked, `data/processed/relevance-fixture-eu/build_fixture.py` downloads and converts them in one call
- `download-fixtures.py` - fetches the PDFs from their recorded source URLs into this directory (`--only PREFIX` limits it to the files whose name starts with the prefix); the IBM URL serves HTML, so its PDF is a print-to-PDF render of the page
