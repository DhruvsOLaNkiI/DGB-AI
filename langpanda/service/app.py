from __future__ import annotations

import os
from pathlib import Path

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

from agent import run_real_estate_pipeline  # noqa: E402

app = FastAPI(
    title="DBG-AI LangPanda",
    description="Pandas code-execution agent with free DuckDuckGo web fallback",
    version="0.2.0",
)


class AskBody(BaseModel):
    question: str = Field(..., min_length=1)
    use_llm: bool = False
    model: str | None = None
    force_gemini: bool = False
    word_limit: int | None = None


class AskResponse(BaseModel):
    reply: str
    source: str = "pandas-agent"


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
    }


@app.post("/pandas-chat", response_model=AskResponse)
def pandas_chat(body: AskBody):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    try:
        reply, source = run_real_estate_pipeline(
            question,
            use_llm=body.use_llm or body.force_gemini,
            model=body.model,
            force_gemini=body.force_gemini,
            word_limit=body.word_limit,
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
    return AskResponse(reply=reply, source=source or "pandas-agent")
