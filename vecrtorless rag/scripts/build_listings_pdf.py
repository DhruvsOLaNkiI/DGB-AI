#!/usr/bin/env python3
"""Build a small structured PDF from Noida listings for PageIndex local demo."""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
CSV_PATH = REPO / "data" / "clean_dataset.csv"
OUT_PDF = ROOT / "data" / "noida_listings_sample.pdf"
SECTOR_RE = re.compile(r"sector\s*([0-9]+[a-z]?)", re.I)
MAX_PER_SECTOR = 8
MAX_SECTORS = 25


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing CSV: {CSV_PATH}")

    by_sector: dict[str, list[dict[str, str]]] = defaultdict(list)
    with CSV_PATH.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            address = (row.get("address") or "").strip()
            match = SECTOR_RE.search(address)
            sector = f"Sector {match.group(1)}" if match else "Other"
            if len(by_sector[sector]) < MAX_PER_SECTOR:
                by_sector[sector].append(row)

    sectors = sorted(by_sector.keys(), key=lambda s: (s == "Other", s))[:MAX_SECTORS]
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(OUT_PDF), pagesize=letter)
    story = [
        Paragraph("Noida Real Estate Listings Sample", styles["Title"]),
        Paragraph(
            "Hierarchical sample for PageIndex vectorless RAG demo.",
            styles["Normal"],
        ),
        Spacer(1, 12),
    ]

    for sector in sectors:
        story.append(Paragraph(sector, styles["Heading1"]))
        for row in by_sector[sector]:
            bedroom = (row.get("bedroom") or "").strip() or "?"
            price = (row.get("price_in_lakh") or "").strip() or "?"
            status = (row.get("status") or "").strip()
            type2 = (row.get("type2") or "").strip()
            address = (row.get("address") or "").strip()
            story.append(
                Paragraph(
                    f"<b>{bedroom} BHK {type2}</b> — {address}. "
                    f"Price: {price} lakh. Status: {status}.",
                    styles["Normal"],
                )
            )
            story.append(Spacer(1, 6))

    doc.build(story)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
