#!/usr/bin/env python3
"""
Answer questions from an already-indexed PageIndex tree.

Upload once → tree is saved under data/.pageindex + documents.json.
Select that doc in the UI → every question uses THIS tree (not the CSV listings tree).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from service.pageindex_client import (  # noqa: E402
    STORAGE,
    load_dotenv_files,
    read_registry,
    resolve_models,
)


def _walk_nodes(nodes: list[Any], out: list[dict[str, Any]], depth: int = 0) -> None:
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        title = str(node.get("title") or node.get("node_title") or "")
        summary = str(node.get("summary") or node.get("prefix_summary") or "")
        text = str(node.get("text") or "")
        start = node.get("start_index") or node.get("page_index")
        end = node.get("end_index") or start
        out.append(
            {
                "title": title,
                "summary": summary,
                "text": text,
                "start": start,
                "end": end,
                "depth": depth,
            }
        )
        kids = node.get("nodes") or node.get("children") or []
        if kids:
            _walk_nodes(kids, out, depth + 1)


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1}


def _score_node(question: str, node: dict[str, Any]) -> float:
    q = question.lower()
    q_tokens = _tokens(question)
    blob = f"{node.get('title','')} {node.get('summary','')} {node.get('text','')}".lower()
    score = 0.0
    for tok in q_tokens:
        if tok in blob:
            score += 1.0
    # Extra weight for sector / bhk phrases
    sector = re.search(r"sector\s*([0-9]+[a-z]?)", q)
    if sector and f"sector {sector.group(1)}" in blob:
        score += 5.0
    bhk = re.search(r"(\d+)\s*bhk", q)
    if bhk and f"{bhk.group(1)} bhk" in blob:
        score += 3.0
    if any(w in q for w in ("price", "lakh", "highest", "lowest", "sqft", "rate")):
        if any(w in blob for w in ("price", "lakh", "rate", "sqft")):
            score += 2.0
    return score


def _load_tree_nodes(doc_id: str) -> list[dict[str, Any]]:
    from pageindex import PageIndexClient

    index_model, chat_model = resolve_models()
    STORAGE.mkdir(parents=True, exist_ok=True)
    client = PageIndexClient(
        index={"model": index_model, "storage_path": str(STORAGE)},
        chat=chat_model,
    )

    # Prefer tree with text attached.
    try:
        tree = client.get_document_structure(doc_id)
    except Exception:
        tree = None

    nodes: list[dict[str, Any]] = []
    if isinstance(tree, dict):
        structure = tree.get("structure") or tree.get("result") or tree.get("data") or []
        if isinstance(structure, list):
            _walk_nodes(structure, nodes)
        elif isinstance(structure, dict):
            _walk_nodes([structure], nodes)
    elif isinstance(tree, list):
        _walk_nodes(tree, nodes)

    # Also pull page OCR text so we can answer from full page content.
    try:
        from pageindex.client import PageIndexClient as _C  # noqa: F401
        api = client._api  # local API
        pages = api._require_pages(doc_id, "chat")
        for p in pages:
            idx = p.get("page_index")
            md = (p.get("markdown") or "").strip()
            if md:
                nodes.append(
                    {
                        "title": f"Page {idx}",
                        "summary": "",
                        "text": md,
                        "start": idx,
                        "end": idx,
                        "depth": 0,
                    }
                )
    except Exception as exc:  # noqa: BLE001
        print(f"[pageindex] page load warning: {exc}", file=sys.stderr)

    # If structure nodes lack text, try raw_tree with text
    if nodes and not any(n.get("text") for n in nodes if n.get("title") != f"Page {n.get('start')}"):
        try:
            raw = client._api.raw_tree(doc_id) or []
            filled: list[dict[str, Any]] = []
            _walk_nodes(raw if isinstance(raw, list) else [], filled)
            if filled:
                nodes.extend(filled)
        except Exception:
            pass

    return nodes


def _required_filters(question: str) -> tuple[str | None, str | None]:
    q = question.lower()
    sector_m = re.search(r"sector\s*([0-9]+[a-z]?)", q)
    bhk_m = re.search(r"(\d+)\s*bhk", q)
    sector = f"sector {sector_m.group(1)}" if sector_m else None
    bhk = f"{bhk_m.group(1)} bhk" if bhk_m else None
    return sector, bhk


def _node_matches_required(question: str, node: dict[str, Any]) -> bool:
    sector, bhk = _required_filters(question)
    blob = f"{node.get('title','')} {node.get('summary','')} {node.get('text','')}".lower()
    if sector and sector not in blob:
        return False
    if bhk and bhk not in blob:
        return False
    return True


def _build_context(question: str, nodes: list[dict[str, Any]], limit: int = 6) -> str:
    ranked = sorted(nodes, key=lambda n: _score_node(question, n), reverse=True)
    picked = [
        n
        for n in ranked
        if _score_node(question, n) > 0 and _node_matches_required(question, n)
    ][:limit]
    # Do not fall back to unrelated top nodes — that causes sector/BHK hallucinations.

    blocks: list[str] = []
    for i, node in enumerate(picked, start=1):
        body = (node.get("text") or node.get("summary") or "").strip()
        title = node.get("title") or f"Section {i}"
        if not body:
            continue
        # Cap each block to keep prompt small for local models
        if len(body) > 2500:
            body = body[:2500] + "…"
        blocks.append(f"[{i}] {title}\n{body}")
    return "\n\n".join(blocks)


DATA_RULES = """CRITICAL DATA INTERPRETATION RULES:
1. Raw rows may use pipe delimiters (e.g. size | price_in_lakh | rate). Never confuse per-square-foot rate with total price_in_lakh. rate is already ₹/sqft; price_in_lakh is the full property price in lakh.
2. For numerical conditions (age_of_property, price_in_lakh, bedroom, size, carpet_area, rate), use only the named column. If a value is missing/null/NaN/empty, say it is not listed — never invent numbers.
3. Multi-condition filters (location + bedrooms + amenities + budget/age, etc.) require EVERY condition to match. Exclude any property that fails even one condition.
4. Always finish complete answers. Do not truncate mid-sentence. List properties in clear human-readable sentences. Never reply with raw row indexes alone.
5. BOOLEAN / AMENITY FILTERS: For swimming pool = True, power backup = True, amenities_* columns, etc., include ONLY rows where those exact columns are True. Exclude False/missing/empty. Never assume every property in a location has the amenity. Never report a count equal to the full location size unless every property truly matches.
"""


def _answer_with_llm(question: str, context: str, model: str) -> str:
    import litellm

    if not context.strip():
        return (
            "I don't have that information in the indexed PageIndex document. "
            "Re-upload a fuller PDF/CSV, or switch the dropdown to Listings (CSV tree) "
            "for the full Noida dataset."
        )

    prompt = (
        "You are DBG-AI answering ONLY from the PageIndex document CONTEXT below.\n"
        "Give a clear plain-text answer. Do not return JSON.\n"
        "Never invent sector, BHK, amenities, size, rate, or price.\n"
        "Every required condition in the QUESTION must match the same row "
        "(sector AND BHK AND amenities AND price when asked).\n"
        "If CONTEXT has zero rows that satisfy ALL conditions, reply exactly: "
        "I don't have that information in the provided Noida listings.\n"
        "Do not substitute a different sector (e.g. 150 for 130) or bedroom count.\n"
        f"{DATA_RULES}\n"
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT FROM PAGEINDEX TREE:\n{context[:14000]}"
    )
    result = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1400,
    )
    text = (result.choices[0].message.content or "").strip()
    return text or "I couldn't form an answer from the indexed document."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--doc-id", help="PageIndex doc_id (default: latest)")
    parser.add_argument(
        "--model",
        help="Override chat model (Gemini id or litellm id, e.g. gemini-2.0-flash)",
    )
    args = parser.parse_args()

    load_dotenv_files()
    docs = read_registry()
    doc_id = args.doc_id
    if not doc_id:
        if not docs:
            raise SystemExit(
                "No indexed documents. Upload a PDF/CSV once in the UI, then select it."
            )
        doc_id = docs[0]["doc_id"]
        print(f"Using latest doc: {docs[0].get('name')} ({doc_id})", file=sys.stderr)

    question = args.question
    if not question:
        question = input("Question: ").strip()
    if not question:
        raise SystemExit("Empty question.")

    _index_model, chat_model = resolve_models()
    model = chat_model
    if args.model:
        raw = args.model.strip()
        if raw.startswith(("ollama/", "groq/", "openai/", "gemini/", "litellm/")):
            model = raw
        else:
            model = f"gemini/{raw}"

    print(f"[pageindex] doc={doc_id} model={model}", file=sys.stderr)

    nodes = _load_tree_nodes(doc_id)
    if not nodes:
        raise SystemExit(
            f"No PageIndex tree found for {doc_id}. Re-upload the document once."
        )

    context = _build_context(question, nodes)
    answer = _answer_with_llm(question, context, model)
    print(answer)


if __name__ == "__main__":
    main()
