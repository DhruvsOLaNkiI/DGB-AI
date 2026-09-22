#!/usr/bin/env python3
"""
Index a PDF (or CSV → PDF) into PageIndex (vectorless).

Usage:
  python scripts/index_document.py /path/to/report.pdf
  python scripts/index_document.py /path/to/data.csv
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.csv_to_pdf import csv_to_pdf  # noqa: E402
from service.pageindex_client import (  # noqa: E402
    UPLOADS,
    get_client,
    load_dotenv_files,
    upsert_registry,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index a PDF/CSV with PageIndex")
    parser.add_argument("file", type=Path, help="Path to a .pdf or .csv file")
    parser.add_argument(
        "--mode",
        choices=("flash", "standard"),
        default="flash",
        help="PageIndex indexing mode (default: flash)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=80,
        help="Max CSV rows to include when converting to PDF",
    )
    args = parser.parse_args()

    load_dotenv_files()
    src = args.file.expanduser().resolve()
    if not src.is_file():
        raise SystemExit(f"File not found: {src}")

    suffix = src.suffix.lower()
    UPLOADS.mkdir(parents=True, exist_ok=True)

    if suffix == ".csv":
        print(f"Converting CSV → PDF (max {args.max_rows} rows)…")
        pdf_path = UPLOADS / f"{src.stem}.pdf"
        csv_to_pdf(src, pdf_path, max_rows=args.max_rows)
        display_name = src.name
    elif suffix == ".pdf":
        pdf_path = UPLOADS / src.name
        if src != pdf_path:
            shutil.copy2(src, pdf_path)
        display_name = src.name
    else:
        raise SystemExit(
            "Unsupported file type. Upload a .pdf or .csv "
            "(CSV is converted to PDF because PageIndex local mode is PDF-only)."
        )

    client = get_client()
    print(f"Indexing {pdf_path.name} ({args.mode})…")
    result = client.submit_document(str(pdf_path), mode=args.mode)
    doc_id = result["doc_id"]
    entry = {
        "doc_id": doc_id,
        "name": result.get("name") or display_name,
        "path": str(pdf_path),
        "source_name": display_name,
        "mode": args.mode,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    upsert_registry(entry)
    print("Done.")
    print(f"  doc_id: {doc_id}")
    print(f"  saved:  {pdf_path}")
    print("Ask about it in DBG with Vectorless mode, or:")
    print(f'  python scripts/chat_document.py --doc-id {doc_id} "Your question"')


if __name__ == "__main__":
    main()
