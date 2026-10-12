#!/usr/bin/env python3
"""Download the external fixture PDFs for docdistance.

The PDFs under ``data/external/`` are gitignored (binaries); only their ``*_origin.md``
provenance companions and this script are tracked. Run this to fetch the PDFs into
``data/external/`` from their recorded source URLs. Source, licence and purpose for each
file live in the matching ``*_origin.md`` beside it - the PDFs are retained solely for
calibration and test-fixture construction, not redistributed and not used for training.

Usage:
    python data/external/download-fixtures.py           # download any missing file
    python data/external/download-fixtures.py --force    # re-download everything
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Each fixture: target filename, source URL, and a provenance note. Keep the filename
# byte-identical to what the rest of the pipeline expects.
FIXTURES = [
    {
        "filename": "impact-of-ai-on-society.pdf",
        "url": "https://wergelandcentre.org/content/uploads/2023/01/FINAL-Impact-of-AI-on-society.pdf",
        "note": "Direct PDF, hosted by The European Wergeland Centre (wergelandcentre.org).",
    },
    {
        "filename": "ibm-enterprise-ai-adoption.pdf",
        "url": "https://newsroom.ibm.com/2024-01-10-Data-Suggests-Growth-in-Enterprise-Adoption-of-AI-is-Due-to-Widespread-Deployment-by-Early-Adopters",
        "note": ("IBM Newsroom article page (HTML), not a direct PDF link. The local fixture PDF is a "
                 "print-to-PDF render of this page; this URL will return HTML, so save the page as PDF "
                 "from a browser to reproduce the exact fixture."),
    },
    {
        "filename": "eu-digital-decade-2024-annex1.pdf",
        "url": "https://ec.europa.eu/newsroom/dae/redirection/document/106688",
        "note": "European Commission, State of the Digital Decade 2024, Annex 1 (CC BY 4.0). Library page: https://digital-strategy.ec.europa.eu/en/library/report-state-digital-decade-2024",
    },
    {
        "filename": "eprs-ata-760421-gigabit-infrastructure-act.pdf",
        "url": "https://www.europarl.europa.eu/RegData/etudes/ATAG/2024/760421/EPRS_ATA(2024)760421_EN.pdf",
        "note": "EPRS At a glance, Gigabit infrastructure act.",
    },
    {
        "filename": "eprs-ata-751386-eu-chips-act.pdf",
        "url": "https://www.europarl.europa.eu/RegData/etudes/ATAG/2023/751386/EPRS_ATA(2023)751386_EN.pdf",
        "note": "EPRS At a glance, EU chips act.",
    },
    {
        "filename": "eprs-ata-759620-european-media-freedom-act.pdf",
        "url": "https://www.europarl.europa.eu/RegData/etudes/ATAG/2024/759620/EPRS_ATA(2024)759620_EN.pdf",
        "note": "EPRS At a glance, European media freedom act.",
    },
    {
        "filename": "eprs-ata-747926-ai-act-parliament-position.pdf",
        "url": "https://www.europarl.europa.eu/RegData/etudes/ATAG/2023/747926/EPRS_ATA(2023)747926_EN.pdf",
        "note": "EPRS At a glance, Parliament's negotiating position on the artificial intelligence act.",
    },
    {
        "filename": "eprs-ata-772906-ai-act-implementation-timeline.pdf",
        "url": "https://www.europarl.europa.eu/RegData/etudes/ATAG/2025/772906/EPRS_ATA(2025)772906_EN.pdf",
        "note": "EPRS At a glance, AI Act implementation timeline.",
    },
    {
        "filename": "eprs-ata-779252-cybersecurity-act-review.pdf",
        "url": "https://www.europarl.europa.eu/RegData/etudes/ATAG/2025/779252/EPRS_ATA(2025)779252_EN.pdf",
        "note": "EPRS At a glance, Cybersecurity Act review: What to expect.",
    },
    {
        "filename": "eprs-ata-760355-foreign-interference-preparedness.pdf",
        "url": "https://www.europarl.europa.eu/RegData/etudes/ATAG/2024/760355/EPRS_ATA(2024)760355_EN.pdf",
        "note": "EPRS At a glance, Foreign interference and EU preparedness.",
    },
    {
        "filename": "eprs-ata-779193-simplifying-digital-laws.pdf",
        "url": "https://www.europarl.europa.eu/RegData/etudes/ATAG/2025/779193/EPRS_ATA(2025)779193_EN.pdf",
        "note": "EPRS At a glance, Simplifying EU digital laws for competitiveness.",
    },
    {
        "filename": "eprs-ata-739206-european-year-of-skills.pdf",
        "url": "https://www.europarl.europa.eu/RegData/etudes/ATAG/2022/739206/EPRS_ATA(2022)739206_EN.pdf",
        "note": "EPRS At a glance, Proposal to make 2023 European Year of Skills.",
    },
    {
        "filename": "eprs-ata-739350-online-age-verification.pdf",
        "url": "https://www.europarl.europa.eu/RegData/etudes/ATAG/2023/739350/EPRS_ATA(2023)739350_EN.pdf",
        "note": "EPRS At a glance, Online age verification methods for children.",
    },
    {
        "filename": "eprs-ata-754642-european-health-data-space.pdf",
        "url": "https://www.europarl.europa.eu/RegData/etudes/ATAG/2023/754642/EPRS_ATA(2023)754642_EN.pdf",
        "note": "EPRS At a glance, European health data space.",
    },
    {
        "filename": "eprs-ata-760392-ai-investment-indicators.pdf",
        "url": "https://www.europarl.europa.eu/RegData/etudes/ATAG/2024/760392/EPRS_ATA(2024)760392_EN.pdf",
        "note": "EPRS At a glance, AI investment: EU and global indicators.",
    },
    {
        "filename": "eprs-ata-759583-strategic-technologies-platform.pdf",
        "url": "https://www.europarl.europa.eu/RegData/etudes/ATAG/2024/759583/EPRS_ATA(2024)759583_EN.pdf",
        "note": "EPRS At a glance, Strategic technologies for Europe platform.",
    },
    {
        "filename": "eprs-ata-760391-gdpr-cross-border-procedural-rules.pdf",
        "url": "https://www.europarl.europa.eu/RegData/etudes/ATAG/2024/760391/EPRS_ATA(2024)760391_EN.pdf",
        "note": "EPRS At a glance, New procedural rules to strengthen GDPR enforcement in cross-border cases.",
    },
]

# Some hosts (newsroom CDNs) reject the default urllib User-Agent with a 403.
UA = "Mozilla/5.0 (X11; Linux x86_64) docdistance-fixture-downloader"


def download(url: str, dest: Path) -> int:
    """Stream a URL to dest in chunks; return the byte count written."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    total = 0
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
        while chunk := resp.read(1 << 16):
            f.write(chunk)
            total += len(chunk)
    return total


def is_pdf(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(5) == b"%PDF-"


def main() -> int:
    ap = argparse.ArgumentParser(description="Download docdistance external fixture PDFs.")
    ap.add_argument("--force", action="store_true", help="re-download even if the file already exists")
    args = ap.parse_args()

    rc = 0
    for fx in FIXTURES:
        dest = HERE / fx["filename"]
        if dest.exists() and not args.force:
            print(f"[skip] {fx['filename']} already present ({dest.stat().st_size:,} bytes)")
            continue
        print(f"[get ] {fx['filename']}")
        print(f"       {fx['url']}")
        try:
            n = download(fx["url"], dest)
        except Exception as e:  # network / 403 / 404 - report and move on
            print(f"[FAIL] {fx['filename']}: {e}")
            rc = 1
            continue
        if is_pdf(dest):
            print(f"[ok  ] {fx['filename']} ({n:,} bytes)")
        else:
            print(f"[warn] {fx['filename']}: downloaded {n:,} bytes but it is not a PDF")
            print(f"       {fx['note']}")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
