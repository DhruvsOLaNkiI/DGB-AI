#!/usr/bin/env python3
"""Convert a CSV into a simple text PDF for PageIndex (local mode is PDF-only)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

DEFAULT_MAX_ROWS = 80


def csv_to_pdf(csv_path: Path, pdf_path: Path, max_rows: int = DEFAULT_MAX_ROWS) -> Path:
    csv_path = csv_path.expanduser().resolve()
    pdf_path = pdf_path.expanduser().resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    story = [
        Paragraph(csv_path.name.replace("&", "&amp;"), styles["Title"]),
        Paragraph(
            "Converted from CSV for PageIndex vectorless indexing. "
            f"Showing up to {max_rows} data rows.",
            styles["Normal"],
        ),
        Spacer(1, 12),
    ]

    with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if headers:
            story.append(
                Paragraph(
                    "Columns: " + ", ".join(h for h in headers if h),
                    styles["Heading2"],
                )
            )
            story.append(Spacer(1, 8))

        for i, row in enumerate(reader, start=1):
            if i > max_rows:
                story.append(
                    Paragraph(
                        f"…truncated after {max_rows} rows.",
                        styles["Italic"],
                    )
                )
                break
            bits = []
            for key in headers:
                val = (row.get(key) or "").strip()
                if not val:
                    continue
                bits.append(f"<b>{key}</b>: {val}")
            if not bits:
                continue
            story.append(Paragraph(f"Row {i}", styles["Heading3"]))
            story.append(Paragraph(" | ".join(bits[:12]), styles["Normal"]))
            story.append(Spacer(1, 6))

    SimpleDocTemplate(str(pdf_path), pagesize=letter).build(story)
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    args = parser.parse_args()
    out = csv_to_pdf(args.csv, args.out, max_rows=args.max_rows)
    print(out)


if __name__ == "__main__":
    main()
