"""
Vectorless listings + document chat service (PageIndex-style).

Run from `vecrtorless rag`:
  source .venv/bin/activate
  uvicorn service.app:app --reload --port 8765
"""
from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .tree import Listing, build_outline, load_listings

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
load_dotenv(REPO / ".env.local")
load_dotenv(ROOT / ".env")

CSV_PATH = Path(
    os.environ.get(
        "LISTINGS_CSV_PATH",
        str(REPO / "data" / "clean_dataset.csv"),
    )
)
if not CSV_PATH.is_absolute():
    CSV_PATH = (REPO / CSV_PATH).resolve()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash").strip()

app = FastAPI(title="DBG Vectorless RAG", version="0.1.0")

_LISTINGS: list[Listing] | None = None
_OUTLINE: str | None = None


def get_listings() -> list[Listing]:
    global _LISTINGS, _OUTLINE
    if _LISTINGS is None:
        if not CSV_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Listings CSV not found: {CSV_PATH}",
            )
        _LISTINGS = load_listings(CSV_PATH)
        _OUTLINE = build_outline(_LISTINGS)
    return _LISTINGS


def get_outline() -> str:
    get_listings()
    assert _OUTLINE is not None
    return _OUTLINE


def parse_filters_regex(question: str) -> dict[str, Any]:
    q = question.lower()
    filters: dict[str, Any] = {}
    bhk = re.search(r"(\d+)\s*(?:bhk|rk)\b", q)
    if bhk:
        filters["bedroom"] = int(bhk.group(1))
    sector = re.search(r"sector\s*([0-9]+[a-z]?)", q, re.I)
    if sector:
        filters["sector"] = f"sector {sector.group(1).lower()}"
        filters["sectors"] = [filters["sector"]]
    lakh = re.search(r"under\s+(\d+(?:\.\d+)?)\s*lakh", q)
    cr = re.search(r"under\s+(\d+(?:\.\d+)?)\s*(?:cr|crore)", q)
    if lakh:
        filters["max_price_lakh"] = float(lakh.group(1))
    elif cr:
        filters["max_price_lakh"] = float(cr.group(1)) * 100
    return filters


def reason_filters(question: str) -> dict[str, Any]:
    fallback = parse_filters_regex(question)
    if not GEMINI_API_KEY:
        return fallback

    try:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        prompt = f"""You navigate a hierarchical tree index of Noida listings (vectorless).
Return ONLY JSON: {{"sectors":["sector 62"],"bedroom":2,"maxPriceLakh":80}}
Use null/[] when unknown. Sectors must match the outline.

QUESTION:
{question}

TREE OUTLINE:
{get_outline()[:120000]}
"""
        result = model.generate_content(
            prompt,
            generation_config={"temperature": 0, "max_output_tokens": 256},
        )
        text = (result.text or "").strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return fallback
        data = json.loads(match.group(0))
        out = dict(fallback)
        sectors = data.get("sectors") or []
        if isinstance(sectors, list) and sectors:
            cleaned = []
            for s in sectors:
                label = str(s).strip().lower()
                if re.match(r"^sector\s+\d+[a-z]?$", label):
                    cleaned.append(label)
            if cleaned:
                out["sectors"] = cleaned
                out["sector"] = cleaned[0]
        if data.get("bedroom") is not None:
            out["bedroom"] = int(data["bedroom"])
        if data.get("maxPriceLakh") is not None:
            out["max_price_lakh"] = float(data["maxPriceLakh"])
        return out
    except Exception as exc:  # noqa: BLE001
        print("[vectorless] reason_filters fallback:", exc)
        return fallback


def filter_listings(listings: list[Listing], filters: dict[str, Any]) -> list[Listing]:
    bedroom = filters.get("bedroom")
    max_price = filters.get("max_price_lakh")
    sectors = filters.get("sectors") or (
        [filters["sector"]] if filters.get("sector") else []
    )

    def ok(item: Listing) -> bool:
        if bedroom is not None and item.bedroom != bedroom:
            return False
        if max_price is not None and (
            item.price_in_lakh is None or item.price_in_lakh > max_price
        ):
            return False
        if sectors:
            return any(
                item.sector == s or s in item.address.lower() for s in sectors
            )
        return True

    has = bedroom is not None or max_price is not None or bool(sectors)
    if not has:
        return listings
    return [item for item in listings if ok(item)]


def format_answer(matched: list[Listing], filters: dict[str, Any], top_k: int) -> str:
    label_bits = []
    if filters.get("bedroom") is not None:
        label_bits.append(f"{filters['bedroom']} BHK")
    if filters.get("sector"):
        label_bits.append(str(filters["sector"]))
    if filters.get("max_price_lakh") is not None:
        label_bits.append(f"under {filters['max_price_lakh']} lakh")
    label = ", ".join(label_bits)

    if not matched:
        if label:
            return f"I don't have any {label} listings in the provided Noida data."
        return "I don't have that information in the provided Noida listings."

    total = len(matched)
    examples = matched[:top_k]
    lines = []
    for i, item in enumerate(examples, start=1):
        bhk = f"{item.bedroom} BHK" if item.bedroom is not None else "Property"
        price = (
            f"{item.price_in_lakh:.2f} lakh"
            if item.price_in_lakh is not None
            else "price not listed"
        )
        place = item.address or item.sector
        bits = [f"{bhk} in {place}", price, item.status]
        lines.append(f"{i}. {' · '.join(b for b in bits if b)}")

    if label:
        return (
            f"There are {total} properties for {label} in the provided data.\n"
            f"Here are {len(examples)} examples:\n" + "\n".join(lines)
        )
    return (
        "There are matching properties in the provided data.\n"
        f"Here are {len(examples)} examples:\n" + "\n".join(lines)
    )


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=40)
    doc_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    mode: str = "vectorless"
    retrieved_count: int = 0
    filters: dict[str, Any] = Field(default_factory=dict)
    doc_id: str | None = None
    source: str = "listings"


@app.get("/health")
def health() -> dict[str, Any]:
    from .pageindex_client import read_registry

    listings_count = 0
    try:
        listings_count = len(get_listings())
    except HTTPException:
        pass
    docs = read_registry()
    return {
        "ok": True,
        "mode": "vectorless",
        "listings": listings_count,
        "documents": len(docs),
        "csv": str(CSV_PATH),
        "gemini": bool(GEMINI_API_KEY),
    }


@app.get("/tree")
def tree() -> dict[str, str]:
    return {"outline": get_outline()}


@app.get("/documents")
def list_documents() -> dict[str, Any]:
    from .pageindex_client import read_registry

    return {"documents": read_registry()}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    from .pageindex_client import UPLOADS, get_client, upsert_registry

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    UPLOADS.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name
    dest = UPLOADS / safe_name
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        client = get_client()
        result = client.submit_document(str(dest), mode="flash")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    entry = {
        "doc_id": result["doc_id"],
        "name": result.get("name") or safe_name,
        "path": str(dest),
        "mode": "flash",
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    upsert_registry(entry)
    return entry


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    question = body.question.strip()

    if body.doc_id:
        try:
            from .pageindex_client import get_client

            client = get_client()
            answer = client.chat(question, doc_id=body.doc_id)
            reply = answer if isinstance(answer, str) else str(answer)
            return ChatResponse(
                reply=reply,
                retrieved_count=1,
                doc_id=body.doc_id,
                source="document",
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    listings = get_listings()
    filters = reason_filters(question)
    matched = filter_listings(listings, filters)
    reply = format_answer(matched, filters, body.top_k)
    return ChatResponse(
        reply=reply,
        retrieved_count=min(len(matched), body.top_k),
        filters=filters,
        source="listings",
    )
