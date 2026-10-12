"""Re-create the EU relevance fixture (batch E19) from its public source documents.

The texts of this fixture are not in the repository. ``report.md``, ``sections.json`` and the 14
files of ``briefs/`` are converted from 15 PDF documents: Annex 1 of the European Commission's
*State of the Digital Decade 2024* and 14 "At a glance" notes of the European Parliamentary
Research Service. Titles, source URLs and reuse notices of the 15 documents are recorded in
``data/external/eu-digital-decade-relevance_origin.md``. ``labels.json`` (the target sections of
each note, assigned before any note was scored) and ``README.md`` are tracked.

Steps:

1. download the PDFs that are missing into ``data/external/`` (``data/external/download-fixtures.py``)
2. convert them with ``notebooks/14-kj-eu-relevance-fixture.ipynb``; the executed copy of the
   notebook goes to a temporary folder, the tracked notebook stays as committed
3. compare the sha256 of the 18 fixture files with the registration of batch E19
   (``reports/E19-relevance/E19-relevance-registration.json``)

Needs the project environment (``make install``) and the models of ``docdistance init`` with the
repository root as ``DOCDISTANCE_HOME``. Run from the repository root; the exit code is 0 when
every file equals its registered sha256:

    .venv/bin/python data/processed/relevance-fixture-eu/build_fixture.py
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
DOWNLOAD = ROOT / "data/external/download-fixtures.py"
NOTEBOOK = ROOT / "notebooks/14-kj-eu-relevance-fixture.ipynb"
REGISTRATION = ROOT / "reports/E19-relevance/E19-relevance-registration.json"
# file name prefixes of the 15 documents in download-fixtures.py
PDF_PREFIXES = ("eu-digital-decade", "eprs-ata-")


def main() -> int:
    only = [arg for prefix in PDF_PREFIXES for arg in ("--only", prefix)]
    subprocess.run([sys.executable, str(DOWNLOAD), *only], check=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute"]
            + ["--ExecutePreprocessor.timeout=-1", "--output-dir", tmp, str(NOTEBOOK)],
            check=True,
            env={**os.environ, "DOCDISTANCE_HOME": str(ROOT)},
        )
    registered = json.loads(REGISTRATION.read_text(encoding="utf-8"))["fixture_sha256"]
    differ = [
        name
        for name, digest in registered.items()
        if not (ROOT / name).is_file()
        or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
    ]
    equal = len(registered) - len(differ)
    print(f"{equal} of {len(registered)} fixture files equal their registered sha256")
    for name in differ:
        print(f"differs or is missing: {name}")
    return 1 if differ else 0


if __name__ == "__main__":
    raise SystemExit(main())
