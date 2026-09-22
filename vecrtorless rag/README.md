# Vectorless RAG (PageIndex) — DBG-AI

This folder hosts [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) so DBG-AI can answer from **listings** or your **uploaded PDFs** in Vectorless mode.

## Setup (already done if `.venv` exists)

```bash
cd "vecrtorless rag"
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e "./PageIndex" -r requirements.txt
```

## Upload a PDF (easiest: in the app)

1. Run `npm run dev`
2. Switch header to **Vectorless**
3. Use **Upload PDF** under the chat box
4. Wait for indexing (uses **Gemini** by default via `PAGEINDEX_LLM=gemini`; set `PAGEINDEX_LLM=ollama` for local models)
5. Select the PDF in the dropdown and ask questions

## Upload via CLI

```bash
# from repo root
npm run vectorless:index -- /absolute/path/to/your.pdf

# or
cd "vecrtorless rag"
source .venv/bin/activate
python scripts/index_document.py ~/Documents/report.pdf
python scripts/chat_document.py --doc-id <doc_id> "What is the summary?"
```

# nor 
When the 
PDFs are stored in `data/uploads/`. Index registry: `data/documents.json`.

PageIndex local mode supports **PDF only**.

## Optional Python service

```bash
npm run vectorless:service
# POST http://127.0.0.1:8765/documents  (multipart file)
# POST http://127.0.0.1:8765/chat       {"question":"...","doc_id":"pi-..."}
```

## Modes in DBG

| Mode | Source |
| --- | --- |
| Vector RAG | Pinecone listings |
| Vectorless + no PDF | Listings CSV tree |
| Vectorless + PDF selected | PageIndex tree over your document |
