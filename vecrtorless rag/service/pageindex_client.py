"""Shared PageIndex local client helpers."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
STORAGE = ROOT / "data" / ".pageindex"
UPLOADS = ROOT / "data" / "uploads"
REGISTRY = ROOT / "data" / "documents.json"


def load_dotenv_files() -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env.local")
    load_dotenv(ROOT / ".env")


def _ensure_prefix(model: str, prefix: str) -> str:
    model = (model or "").strip()
    if not model:
        return model
    if model.startswith(
        ("ollama/", "groq/", "openai/", "gemini/", "litellm/", "huggingface/")
    ):
        return model
    return f"{prefix}{model}"


def _gemini_model() -> str | None:
    gemini = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not gemini:
        return None
    os.environ.setdefault("GOOGLE_API_KEY", gemini)
    os.environ.setdefault("GEMINI_API_KEY", gemini)
    model = (
        os.environ.get("PAGEINDEX_GEMINI_MODEL")
        or os.environ.get("GEMINI_MODEL")
        or "gemini-3.6-flash"
    ).strip()
    return _ensure_prefix(model, "gemini/")


def resolve_models() -> tuple[str, str]:
    """Prefer Gemini for PageIndex index + chat; Ollama/Groq remain optional."""
    llm = (os.environ.get("PAGEINDEX_LLM") or "gemini").strip().lower()
    chat_provider = (
        os.environ.get("PAGEINDEX_CHAT_PROVIDER") or llm
    ).strip().lower()
    index_override = (os.environ.get("PAGEINDEX_INDEX_MODEL") or "").strip()
    chat_override = (os.environ.get("PAGEINDEX_CHAT_MODEL") or "").strip()
    ollama_model = (
        os.environ.get("PAGEINDEX_OLLAMA_MODEL")
        or os.environ.get("OLLAMA_MODEL")
        or "ollama/llama3.2:3b"
    ).strip()
    gemini = _gemini_model()

    # --- index model ---
    if index_override:
        index = index_override
        if not any(
            index.startswith(p)
            for p in ("ollama/", "groq/", "openai/", "gemini/", "litellm/")
        ):
            # bare name: Gemini when that is the configured LLM, else Ollama
            prefix = "gemini/" if llm == "gemini" and gemini else "ollama/"
            index = _ensure_prefix(index, prefix)
    elif llm == "gemini" and gemini:
        index = gemini
    elif llm == "ollama":
        index = _ensure_prefix(ollama_model, "ollama/")
        os.environ.setdefault(
            "OLLAMA_API_BASE",
            os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
        )
    elif llm == "hybrid" and gemini:
        # Index locally if Ollama is intended later; default hybrid → Gemini
        index = gemini
    elif os.environ.get("GROQ_API_KEY"):
        index = _ensure_prefix(
            os.environ.get("PAGEINDEX_GROQ_MODEL", "groq/openai/gpt-oss-20b"),
            "groq/",
        )
    elif gemini:
        index = gemini
    else:
        raise RuntimeError(
            "No index model. Set GEMINI_API_KEY (PAGEINDEX_LLM=gemini) "
            "or PAGEINDEX_LLM=ollama with Ollama running."
        )

    # --- chat model ---
    if chat_override:
        chat = chat_override
        if not any(
            chat.startswith(p)
            for p in ("ollama/", "groq/", "openai/", "gemini/", "litellm/")
        ):
            prefix = "gemini/" if gemini and chat_provider != "ollama" else "ollama/"
            chat = _ensure_prefix(chat, prefix)
    elif chat_provider == "ollama":
        chat = _ensure_prefix(ollama_model, "ollama/")
    elif gemini:
        chat = gemini
    else:
        chat = index

    if index.startswith("ollama/") or chat.startswith("ollama/"):
        os.environ.setdefault(
            "OLLAMA_API_BASE",
            os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
        )

    return index, chat


def get_client():
    from pageindex import PageIndexClient

    index_model, chat_model = resolve_models()
    STORAGE.mkdir(parents=True, exist_ok=True)
    return PageIndexClient(
        index={"model": index_model, "storage_path": str(STORAGE)},
        chat=chat_model,
    )


def read_registry() -> list[dict]:
    if not REGISTRY.exists():
        return []
    try:
        data = json.loads(REGISTRY.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def write_registry(docs: list[dict]) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(docs, indent=2), encoding="utf-8")


def upsert_registry(entry: dict) -> list[dict]:
    docs = [d for d in read_registry() if d.get("doc_id") != entry.get("doc_id")]
    docs.insert(0, entry)
    write_registry(docs)
    return docs


def looks_like_tree_dump(text: str) -> bool:
    raw = text.strip()
    if not (raw.startswith("{") and raw.endswith("}")):
        return False
    try:
        data = json.loads(raw)
    except ValueError:
        return False
    if not isinstance(data, dict):
        return False
    keys = set(data)
    return bool(keys & {"node_id", "start_index", "end_index", "key_items"})


def gemini_answer_from_context(question: str, context: str) -> str:
    """Turn retrieved tree/context into a plain-language answer with Gemini."""
    model = _gemini_model()
    if not model:
        return context
    import litellm

    prompt = (
        "You are DBG-AI, a Noida real-estate assistant.\n"
        "Answer the user question using ONLY the CONTEXT below.\n"
        "Reply in simple plain text — never return JSON.\n"
        "If context is a tree node with a summary, use that summary and key facts.\n"
        "If you cannot answer from context, say you don't have that information "
        "in the provided listings.\n"
        "CRITICAL: Never confuse rate (₹/sqft) with price_in_lakh (total price). "
        "Do not invent missing numbers. Apply every filter condition. "
        "Finish complete human-readable sentences; never truncate or use raw row indexes alone.\n\n"
        f"QUESTION:\n{question}\n\nCONTEXT:\n{context[:12000]}"
    )
    result = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1200,
    )
    text = (result.choices[0].message.content or "").strip()
    return text or context
