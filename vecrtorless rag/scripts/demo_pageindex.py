#!/usr/bin/env python3
"""
Smoke-test PageIndex local mode on the sample listings PDF.

Prefers GROQ_API_KEY, then OPENAI_API_KEY, then GEMINI_API_KEY.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from service.pageindex_client import get_client, load_dotenv_files, resolve_models  # noqa: E402

PDF = ROOT / "data" / "noida_listings_sample.pdf"


def main() -> None:
    if not PDF.exists():
        raise SystemExit(
            f"Missing {PDF}. Run: python scripts/build_listings_pdf.py"
        )

    load_dotenv_files()
    index_model, chat_model = resolve_models()
    client = get_client()

    print(f"Indexing {PDF.name} with {index_model} (flash)…")
    result = client.submit_document(str(PDF), mode="flash")
    doc_id = result["doc_id"]
    print("Indexed doc_id:", doc_id)

    structure = client.get_document_structure(doc_id)
    nodes = structure.get("structure") if isinstance(structure, dict) else structure
    print("Tree nodes (top-level):", len(nodes or []))
    print("Chat model:", chat_model)

    answer = client.chat(
        "Summarize 2 BHK options mentioned in the document.",
        doc_id=doc_id,
    )
    print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()
