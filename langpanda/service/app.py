from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
load_dotenv(REPO / ".env.local")
load_dotenv(ROOT / ".env")

# Make langpanda/ importable for agent + dataframe
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import build_ask_dgb_state, run_real_estate_pipeline  # noqa: E402

app = FastAPI(
    title="DBG-AI LangPanda",
    description="Pandas code-execution agent with Firecrawl web fallback",
    version="0.2.0",
)


class HistoryTurn(BaseModel):
    role: str
    content: str = ""


class AskBody(BaseModel):
    question: str = Field(..., min_length=1)
    use_llm: bool = False
    model: str | None = None
    force_gemini: bool = False
    force_firecrawl: bool = False
    word_limit: int | None = None
    history: list[HistoryTurn] | None = None
    older_summary: str | None = None
    profile: dict[str, Any] | None = None


class AskResponse(BaseModel):
    reply: str
    source: str = "pandas-agent"
    ask_state: dict[str, Any] | None = None


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "langpanda",
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "gemini": bool(
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        ),
        "ollama": True,
        "web_fallback": (os.environ.get("LANGPANDA_WEB_FALLBACK") or "1")
        .strip()
        .lower()
        not in {"0", "false", "no", "off"},
        "firecrawl": bool((os.environ.get("FIRECRAWL_API_KEY") or "").strip()),
    }


@app.post("/pandas-chat", response_model=AskResponse)
def pandas_chat(body: AskBody):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    history = None
    if body.force_gemini and body.history:
        history = [
            {"role": t.role, "content": t.content}
            for t in body.history
            if (t.content or "").strip()
        ]
    older_summary = (
        (body.older_summary or "").strip() or None if body.force_gemini else None
    )
    profile = body.profile if body.force_gemini and isinstance(body.profile, dict) else None
    try:
        reply, source = run_real_estate_pipeline(
            question,
            use_llm=body.use_llm or body.force_gemini or body.force_firecrawl,
            model=body.model,
            force_gemini=body.force_gemini,
            force_firecrawl=body.force_firecrawl,
            word_limit=body.word_limit,
            history=history,
            older_summary=older_summary,
            profile=profile,
        )
    except Exception as exc:  # noqa: BLE001
        # Never surface raw stack traces / Errno to the chat UI.
        print(f"[langpanda] /pandas-chat failed: {exc}", file=sys.stderr)
        return AskResponse(
            reply=(
                "I couldn’t finish that property question right now. "
                "Please try again with a sector, BHK, or budget — or restart "
                "`npm run langpanda:service` if this keeps happening."
            ),
            source="error",
        )
    reply = str(reply or "").strip()
    if not reply or reply.startswith("Execution error:"):
        return AskResponse(
            reply=(
                reply
                if reply and not reply.startswith("Execution error:")
                else (
                    "I couldn’t find a usable answer for that. Try a clearer "
                    "property question (sector / BHK / price / amenities)."
                )
            ),
            source="error",
        )
    ask_state = None
    if body.force_gemini:
        try:
            ask_state = build_ask_dgb_state(
                history, question, profile=profile
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[langpanda] ask_state build failed: {exc}", file=sys.stderr)
            ask_state = None
    return AskResponse(
        reply=reply,
        source=source or "pandas-agent",
        ask_state=ask_state,
    )
