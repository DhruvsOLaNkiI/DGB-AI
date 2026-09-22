# DBG-AI

Gemini-style real estate chat assistant with **two retrieval modes**:

| Mode | How it works |
| --- | --- |
| **Vector RAG** | Local MiniLM embeddings + Pinecone similarity |
| **Vectorless** | PageIndex-style hierarchical tree + Gemini reasoning (no vectors) |

Switch modes anytime in the chat header.

## Run

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Keys in `.env.local`

```env
GEMINI_API_KEY=your_gemini_key          # chat + vectorless tree reasoning
PINECONE_API_KEY=your_pinecone_key      # required only for Vector RAG
PINECONE_INDEX=minilm                   # Dense, cosine, dimension 384
LOCAL_EMBEDDING_MODEL=Xenova/all-MiniLM-L6-v2
EMBEDDING_DIMENSIONS=384
```

Never commit `.env.local`. Use `.env.example` as a template.

## Index listings (Vector RAG — local MiniLM)

1. Put CSV at `data/clean_dataset.csv`
2. Pinecone index `minilm` must be **384** dimensions
3. Run:

```bash
npm run index-listings
```

First run downloads MiniLM weights. Then upserts ~9k listings into Pinecone.

## Vectorless / PageIndex folder

Cloned + installed under [`vecrtorless rag/`](./vecrtorless%20rag/README.md):

```bash
npm run vectorless:demo-pdf   # sample PDF for PageIndex
npm run vectorless:service    # optional FastAPI on :8765
```

In-app **Vectorless** mode uses Gemini over a sector/BHK tree of the CSV (no Pinecone). The PageIndex repo is available for PDF demos and further experiments.

## What’s built

- Gemini-style UI: sidebar, ask, photo, voice, **RAG ↔ Vectorless toggle**
- Local **all-MiniLM-L6-v2** embeddings → Pinecone `minilm`
- `/api/chat` accepts `mode: "rag" | "vectorless"`
- Token budget meter in the header

Full RAG explanation: [`docs/RAG.md`](docs/RAG.md)
