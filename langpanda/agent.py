"""
Pandas Code-Execution Agent for DBG-AI listings.

Uses local Ollama for free-form questions, plus a fast direct-Pandas path
for common sector / BHK / amenity / furnish filters (no LLM wait).
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load repo .env.local then langpanda/.env
_ROOT = Path(__file__).resolve().parent
_REPO = _ROOT.parent
load_dotenv(_REPO / ".env.local")
load_dotenv(_ROOT / ".env")

import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent

# langchain_community removed ChatOllama — use langchain_ollama.
try:
    from langchain_ollama import ChatOllama
except ImportError:  # pragma: no cover
    from langchain_community.chat_models import ChatOllama  # type: ignore

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover
    ChatGoogleGenerativeAI = None  # type: ignore

from dataframe import load_listings_df

# 1. Load the clean structured dataset
df: pd.DataFrame = load_listings_df()

# Pandas + LLM agent instructions (exact schema + execution constraints).
system_prefix = """
You are an expert real estate data analyst working with `clean_dataset.csv`
loaded as DataFrame `df` (Noida listings).
Only use the `python_repl_ast` tool. Never invent shell commands or external tools.

Follow these rules strictly:

1. Column Grounding:
   - Always run `df.columns.tolist()` or inspect relevant column names before
     assuming a column doesn't exist.
   - For amenities, look for exact names like `amenities_sports_facility`,
     `amenities_gymnasium`, `amenities_swimming_pool`, `amenities_club_house`,
     `amenities_full_power_backup`, `amenities_lift`,
     `amenities_24_x_7_security` (NOT `amenities_24x7_security`).
   - Total price is `price_in_lakh` (INR Lakhs). Per-sqft is `rate`.
     Never use `rate` for property cost/budget. Do not swap `size` and `carpet_area`.
   - Construction status is `status` (NOT `status.1`). `status.1` is furnishing only.
     Typical status values: 'Ready to move', 'Under Construction' (match case-insensitively).
   - Location is `address` via `str.contains(..., case=False, na=False)`.
   - Bedrooms: `bedroom`. Age (years): `age_of_property`.

2. Compound & Negative Logic:
   - When a user asks for "A but NOT B", you MUST explicitly filter:
     `df[(df['amenity_a'] == True) & (df['amenity_b'] == False)]`
     (using the exact `amenities_*` column names).
   - Positive amenity filters: `== True`. Negatives ("without", "no", "not"): `== False`.
   - Never substitute `NaN` for `False` unless explicitly instructed.
   - Multi-condition queries must AND every condition.

3. No Unintended dropna():
   - Never run `df.dropna()` on the entire dataset.
   - Only drop NaN values on the specific numerical column you are aggregating
     (e.g., `df['price_in_lakh'].dropna()`).
   - For price averages/aggregations, use rows where `price_in_lakh` > 0.
   - Price intervals: "between X and Y" / "X to Y" / "X - Y" must apply BOTH
     `price_in_lakh >= min` AND `price_in_lakh <= max` in one mask (never drop
     the lower bound). Convert Crore to Lakh (* 100). Apply BHK + sector + price
     + amenities together before counting or sampling.

4. Cheapest / Highest extremes:
   - Sort filtered rows by `price_in_lakh` (ascending for cheapest, descending for highest)
     and return the exact `.iloc[0]` row details — never a random head(3) sample alone.
   - Include address, price_in_lakh, size, and amenity booleans from that row.

5. Output Format:
   - State the final numeric metrics, counts, and currency rounded to 2 decimal places
     (e.g., ₹116.79 Lakh).
   - Never print raw tracebacks or terminal scratchpad thoughts.
   - Never dump long raw DataFrames. No AgentExecutor logs, Action/Action Input, or ANSI codes.
   - If a filter returns 0 rows, say "0 matching properties found for this combination."
   - If a requested column does not exist after inspecting `df.columns`, say it is unavailable.
   - For listing matches, output:
     1) Total matched records count
     2) Min / Max / Average of `price_in_lakh`
     3) 2–3 sample properties: Address · BHK · Price · Size · confirmed amenities
   - Finish in as few tool calls as possible.
"""

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\[\d+(?:;\d+)*m")
_CHAIN_NOISE_RE = re.compile(
    r"(?is)"
    r"(?:>\s*)?Entering new AgentExecutor chain\.\.\.|"
    r"(?:>\s*)?Finished chain\.|"
    r"Action\s*Input\s*:|"
    r"^Action\s*:|"
    r"^Thought\s*:|"
    r"^Observation\s*:"
)

def _ollama_base_url() -> str:
    return (
        os.environ.get("OLLAMA_API_BASE")
        or os.environ.get("OLLAMA_HOST")
        or "http://127.0.0.1:11434"
    ).strip().rstrip("/")


def _ollama_model_name() -> str:
    """Local model tag (no ollama/ prefix). Default matches DBG-AI's pulled model."""
    raw = (
        os.environ.get("LANGPANDA_OLLAMA_MODEL")
        or os.environ.get("PAGEINDEX_OLLAMA_MODEL")
        or os.environ.get("OLLAMA_MODEL")
        or "llama3.2:3b"
    ).strip()
    return raw.removeprefix("ollama/").strip() or "llama3.2:3b"


def _resolve_llm_choice(model: str | None) -> tuple[str, str]:
    """
    Map UI / API model id → (provider, model_name).
    provider is 'gemini' or 'ollama'.
    """
    raw = (model or "").strip()
    # Retired / blocked-for-new-users ids → current free-tier Flash.
    _GEMINI_ALIASES = {
        "gemini-1.5-flash": "gemini-3.6-flash",
        "gemini-1.5-flash-latest": "gemini-3.6-flash",
        "gemini-1.5-pro": "gemini-3.1-pro-preview",
        "gemini-2.0-flash": "gemini-3.6-flash",
        "gemini-2.0-flash-lite": "gemini-3.5-flash-lite",
        # New API keys often cannot call 2.5 — map to 3.6.
        "gemini-2.5-flash": "gemini-3.6-flash",
        "gemini-2.5-pro": "gemini-3.6-flash",
    }
    if not raw:
        # Prefer Gemini when a key exists so Pandas+LLM matches the UI default.
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            gem = (
                os.environ.get("LANGPANDA_GEMINI_MODEL")
                or os.environ.get("GEMINI_MODEL")
                or "gemini-3.6-flash"
            ).strip()
            name = gem.removeprefix("gemini/").strip() or "gemini-3.6-flash"
            name = _GEMINI_ALIASES.get(name, name)
            return "gemini", name
        return "ollama", _ollama_model_name()

    lower = raw.lower()
    if lower.startswith("ollama/") or lower.startswith("llama"):
        return "ollama", raw.removeprefix("ollama/").removeprefix("OLLAMA/").strip()
    if "gemini" in lower:
        name = raw.removeprefix("gemini/").removeprefix("GEMINI/").strip()
        name = _GEMINI_ALIASES.get(name, name)
        return "gemini", name
    # Unknown id — treat bare tags as Ollama, otherwise Gemini-style ids.
    if "/" not in raw and ":" in raw:
        return "ollama", raw
    name = _GEMINI_ALIASES.get(raw, raw)
    return "gemini", name


def _build_llm(model: str | None = None, temperature: float = 0):
    """Build Gemini or Ollama chat model from the UI selection."""
    provider, name = _resolve_llm_choice(model)
    if provider == "gemini":
        return _build_gemini_llm(name=name, temperature=temperature)

    return ChatOllama(
        model=name,
        base_url=_ollama_base_url(),
        temperature=temperature,
    )


def _build_gemini_llm(
    name: str | None = None,
    temperature: float = 0.2,
):
    """Always build Gemini when GEMINI_API_KEY / GOOGLE_API_KEY is set."""
    api_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or ""
    ).strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to .env.local for Gemini fallback."
        )
    if ChatGoogleGenerativeAI is None:
        raise RuntimeError(
            "langchain-google-genai is not installed in langpanda/.venv"
        )
    model_name = (name or "").strip() or (
        os.environ.get("LANGPANDA_GEMINI_MODEL")
        or os.environ.get("GEMINI_MODEL")
        or "gemini-3.6-flash"
    )
    model_name = model_name.removeprefix("gemini/").strip() or "gemini-3.6-flash"
    # Remap retired / new-user-blocked ids. Keep 3.6 vs 3.5-lite distinct for quota failover.
    aliases = {
        "gemini-1.5-flash": "gemini-3.6-flash",
        "gemini-1.5-flash-latest": "gemini-3.6-flash",
        "gemini-1.5-pro": "gemini-3.1-pro-preview",
        "gemini-2.0-flash": "gemini-3.6-flash",
        "gemini-2.0-flash-lite": "gemini-3.5-flash-lite",
        "gemini-2.5-flash": "gemini-3.6-flash",
        "gemini-2.5-pro": "gemini-3.6-flash",
    }
    model_name = aliases.get(model_name, model_name)
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
    )


def _gemini_available() -> bool:
    return bool(
        (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    )


def _invoke_llm_text(llm: object, prompt: str) -> str:
    with contextlib.redirect_stdout(sys.stderr):
        if hasattr(llm, "invoke"):
            out = llm.invoke(prompt)
            return _llm_content_to_text(getattr(out, "content", out))
        return _llm_content_to_text(llm.predict(prompt))  # type: ignore[attr-defined]


def _invoke_llm_with_gemini_backup(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.2,
) -> tuple[str, str]:
    """
    Try the UI-selected LLM first; on failure (e.g. Ollama down) retry Gemini.
    Returns (text, which) where which is 'ui' or 'gemini-backup'.
    """
    last_err: Exception | None = None
    try:
        llm = _build_llm(model, temperature=temperature)
        text = _clean_agent_reply(_invoke_llm_text(llm, prompt))
        if text:
            return text, "ui"
    except Exception as exc:  # noqa: BLE001
        last_err = exc
        print(f"[langpanda] primary LLM failed: {exc}", file=sys.stderr)

    if _gemini_available():
        try:
            llm = _build_gemini_llm(temperature=temperature)
            text = _clean_agent_reply(_invoke_llm_text(llm, prompt))
            if text:
                return text, "gemini-backup"
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"[langpanda] Gemini backup failed: {exc}", file=sys.stderr)

    if last_err:
        raise last_err
    return "", "ui"


_GEMINI_ONLY_PROMPT = """You are DBG-AI answering in ASK DGB-SUP mode.
Answer ONLY from your own built-in Noida / NCR real-estate knowledge.

HARD BANS (never do these):
- Do NOT use web search, DuckDuckGo, browsing, live listings APIs, or any external tool.
- Do NOT use any CSV / inventory / database.
- Do NOT invent citations, news links, or “according to a recent article” claims.
- Do NOT say you searched the web or looked up live data.

User asked: "{question}"

WORD LIMIT: {word_limit_rule}

CRITICAL RULES:
1. Never say you “found N matching properties in the dataset” or quote CSV rows.
2. Do NOT demand BHK/budget if they asked a general question — but if they gave
   BHK / budget / near transport, use those constraints in your advice.
3. For “best under X crore / BHK / near transport” style asks, give a shortlist of
   3–5 realistic sector / micro-market picks with approximate price bands in **bold**,
   amenities to expect, and why (metro/expressway, rental, lifestyle).
4. Cover plots / villas / flats / commercial when relevant.
5. Mark uncertain figures as approximate. Never invent exact RERA IDs.
6. Do NOT mention CSV, inventory, Gemini, ASK DGB-SUP, web search, or “market knowledge”
   disclaimers in the reply. Start directly with the useful answer.
7. Clean bullets. End with one short practical tip (e.g. verify live listings / share budget).
8. Headings: use ## or ### only (never #### or raw # hashes left unreadable).

COMPARISON FORMAT (when the user compares 2+ options — e.g. sector A vs B,
flats vs plots, fixed vs floating loan, buy vs rent, Project X vs Y):
1. Open with 1–2 short sentences framing the comparison.
2. Then a Markdown table with:
   - First column = Feature / Aspect
   - One column per option being compared
   - Rows for the important differences (price band, appreciation, liquidity,
     risk, rental yield, best for, timeline, etc. — pick what fits the ask)
3. After the table, add:
   ### Which is Best For You?
   - **Choose [Option A] if:** bullet points
   - **Choose [Option B] if:** bullet points
4. End with **Recommendation:** one clear paragraph.
5. Use real Markdown pipe tables, e.g.:
   | Feature | Option A | Option B |
   |---|---|---|
   | Price | **₹X–Y Cr** | **₹A–B Cr** |
Non-comparison questions: do NOT force a table — use bullets / short sections.
Keep the WHOLE reply (including table) inside the WORD LIMIT above.
"""

_GEMINI_KNOWLEDGE_PROMPT = _GEMINI_ONLY_PROMPT


def _word_limit_rule(word_limit: int | None) -> str:
    n = int(word_limit or 0)
    if n <= 0:
        return "No hard word cap — stay concise and useful."
    return (
        f"HARD CAP: write at most {n} words in the entire reply "
        f"(count every word). Prefer fewer if possible. Do not exceed {n}."
    )


def _enforce_word_limit(text: str, word_limit: int | None) -> str:
    """Soft trim if the model overshoots the selected word cap."""
    n = int(word_limit or 0)
    if n <= 0 or not text:
        return text
    words = re.findall(r"\S+|\s+", text)
    tokens = [w for w in words if w.strip()]
    if len(tokens) <= n:
        return text
    # Rebuild from original tokens (words + whitespace) up to n words.
    out: list[str] = []
    count = 0
    for part in words:
        if part.strip():
            if count >= n:
                break
            count += 1
        out.append(part)
    trimmed = "".join(out).rstrip()
    # Prefer ending on a sentence if we cut mid-way.
    for end in (".", "!", "?", "\n"):
        idx = trimmed.rfind(end)
        if idx >= max(20, len(trimmed) // 2):
            trimmed = trimmed[: idx + 1]
            break
    return trimmed.strip()


def answer_with_gemini_only(
    question: str,
    *,
    model: str | None = None,
    word_limit: int | None = None,
) -> tuple[str, str]:
    """
    ASK DGB-SUP: pure Gemini knowledge only.
    Never CSV, never DuckDuckGo / web_search, never external tools.
    """
    if not _gemini_available():
        return (
            "ASK DGB-SUP needs GEMINI_API_KEY in .env.local. "
            "Add the key, or switch to Pandas / Pandas + LLM for CSV answers.",
            "error",
        )
    preferred = (model or "").strip()
    if not preferred or "ollama" in preferred.lower() or preferred.lower().startswith(
        "llama"
    ):
        preferred = (
            os.environ.get("LANGPANDA_GEMINI_MODEL")
            or os.environ.get("GEMINI_MODEL")
            or "gemini-3.6-flash"
        )
    preferred = preferred.removeprefix("gemini/").strip() or "gemini-3.6-flash"
    # New Google AI keys: 2.5/2.0 often 404; free tier is 3.6 + 3.5-flash-lite.
    candidates: list[str] = []
    for mid in (
        preferred,
        "gemini-3.6-flash",
        "gemini-flash-latest",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
    ):
        if mid and mid not in candidates:
            candidates.append(mid)

    limit = int(word_limit or 0)
    prompt = _GEMINI_ONLY_PROMPT.format(
        question=(question or "").strip(),
        word_limit_rule=_word_limit_rule(limit),
    )
    last_err: Exception | None = None
    for mid in candidates:
        try:
            # Plain chat model — no tools / grounding / web search.
            llm = _build_gemini_llm(name=mid, temperature=0.35)
            text = _clean_agent_reply(_invoke_llm_text(llm, prompt))
            if text:
                text = _strip_gemini_disclaimer(text)
                text = _enforce_word_limit(text, limit)
                return text, "gemini-only"
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"[langpanda] ASK DGB-SUP model {mid} failed: {exc}", file=sys.stderr)

    err = str(last_err or "")
    if "API_KEY_INVALID" in err or "API key not valid" in err:
        msg = (
            "ASK DGB-SUP: GEMINI_API_KEY in .env.local is invalid or revoked. "
            "Create a new key at https://aistudio.google.com/apikey, paste it, "
            "restart `npm run langpanda:service`, then retry."
        )
    elif "429" in err or "RESOURCE_EXHAUSTED" in err:
        msg = (
            "Gemini free-tier quota is exhausted for now. "
            "Wait ~1 minute, try Gemini 3.5 Flash-Lite in the model dropdown, "
            "or use Pandas / Pandas + LLM for CSV."
        )
    else:
        msg = (
            "ASK DGB-SUP couldn’t answer right now (API quota or network). "
            "Wait a minute and retry, or switch to Pandas / Pandas + LLM for CSV inventory."
        )
    return msg, "error"


def _strip_gemini_disclaimer(text: str) -> str:
    """Remove visible ‘market knowledge / not CSV’ preamble from ASK DGB-SUP replies."""
    if not text:
        return text
    lines = text.splitlines()
    kept: list[str] = []
    skip_pat = re.compile(
        r"(?i)^\s*("
        r".*gemini\s+market\s+knowledge|"
        r".*market\s+knowledge\s*\(.*csv|"
        r".*not\s+(our\s+)?csv\s+inventory|"
        r".*ask\s+dgb-?sup\s*\(.*|"
        r".*this\s+is\s+(ask\s+dgb|gemini)|"
        r".*(web\s+search|duckduckgo|ddgs)"
        r").*$"
    )
    for i, line in enumerate(lines):
        if i < 3 and skip_pat.search(line):
            continue
        kept.append(line)
    cleaned = "\n".join(kept).strip()
    # Also drop a leading italic/note sentence containing the phrase mid-line.
    cleaned = re.sub(
        r"(?is)^\s*\*?\s*(?:note\s*:?\s*)?[^.!\n]*market knowledge[^.!\n]*\.?\s*",
        "",
        cleaned,
        count=1,
    ).strip()
    return cleaned or text.strip()


def synthesize_external_answer(
    question: str,
    *,
    model: str | None = None,
    missing_attribute: str | None = None,
) -> tuple[str, str]:
    """
    Web search first; if unusable, Gemini knowledge fallback.
    Returns (reply, source) where source is web-fallback | gemini-knowledge.
    """
    from web_search import free_web_search, is_web_search_usable

    web_data = free_web_search(question, max_results=3)
    if is_web_search_usable(web_data):
        reply = format_web_fallback_with_llm(
            question,
            web_data,
            model=model,
            missing_attribute=missing_attribute,
        )
        return reply, "web-fallback"

    # Web dead → Gemini knowledge (requires API key)
    print(
        "[langpanda] web search unusable; trying Gemini knowledge fallback",
        file=sys.stderr,
    )
    if not _gemini_available():
        return (
            "Web search didn’t return usable results, and GEMINI_API_KEY is not set. "
            "Add GEMINI_API_KEY to .env.local, or ask with sector / BHK / budget for "
            "CSV inventory matches.",
            "error",
        )
    try:
        prompt = _GEMINI_KNOWLEDGE_PROMPT.format(
            question=question.strip(),
            word_limit_rule=_word_limit_rule(0),
        )
        text, _which = _invoke_llm_with_gemini_backup(
            prompt, model="gemini-3.6-flash", temperature=0.3
        )
        if text:
            return _strip_gemini_disclaimer(text), "gemini-knowledge"
    except Exception as exc:  # noqa: BLE001
        print(f"[langpanda] Gemini knowledge fallback failed: {exc}", file=sys.stderr)

    return (
        "Web search and Gemini both failed to answer that right now. "
        "Try again shortly, or ask with a sector / BHK / budget for CSV matches.",
        "error",
    )


_llm = None
_agent = None
_agent_key: str | None = None

_DECORATE_PROMPT = """You are DBG-AI, a friendly Noida real-estate assistant.
Rewrite the FACTS into a clear, conversational reply to the USER QUESTION.

Hard rules:
- Keep EVERY number, sector, BHK, amenity, age, and sample property exactly as given.
- Do not invent listings, prices, rates, or amenities.
- If FACTS contain an exact cheapest/most-expensive target_details row, report THAT
  property's address, price_in_lakh, size, and requested amenity booleans directly.
  Do not replace it with a generic average or head(3) sample list.
- Do not drop the price summary or sample properties when they are present.
- Use short paragraphs and bullets. Sound human and helpful.
- Plain text only (no ** markdown). No tool talk, no "as an AI".

USER QUESTION:
{question}

FACTS (ground truth — do not change the numbers):
{facts}
"""

_WEB_FALLBACK_PROMPT = """You are an expert real estate data assistant.
User asked: "{question}"

Live Web Results:
{web_data}

{attribute_note}
{opening}

CRITICAL RULES:
1. Real estate is NOT just BHKs/apartments. Do NOT demand BHK, bedroom count, or budget from the user if they simply asked for general property cost or property in a sector.
2. Synthesize whatever real estate asset exists in the search results for that sector:
   - If it has Residential Plots / Land: Quote the price per sq.ft. / sq.yd. and entry plot pricing.
   - If it has Independent Houses / Villas / Kothis: Quote the house price ranges.
   - If it has High-Rise Society Flats: Quote the typical apartment prices.
   - If it has Commercial Shops / Land: Mention the commercial rate ranges.
3. If the internal CSV only has apartments and had 0 results, explain clearly what the web search reveals about the sector's real estate profile.
4. Output clean, direct bullet points. Put prices in **bold** like **₹8,500/sq.ft.** or **₹1.2–1.8 Cr**.
5. NEVER return an error message or ask the user to re-phrase.
6. NEVER dump raw portal snippets, aggregator counts (e.g. "24,000+ properties"), or paste Source:/Details:/Link: blocks as the answer body.
7. Prefer figures present in the web results; mark uncertain numbers as approximate. Do not invent exact circle rates or RERA IDs.
8. End with one short line that this is public web market context, not rows from our CSV inventory.
9. When a source URL is available and useful, you may add one chip-style line per asset class:
   Link: <url>
"""

_INVESTMENT_BRIEFING_PROMPT = """You are DBG-AI, a Noida real-estate investment advisor.

The user asked a broad / subjective investment question. Our CSV inventory cannot answer
“best for investment” with a single listing. You are given WEB SEARCH SNIPPETS for context.

CRITICAL RULES:
- NEVER output raw search snippets, portal URLs, or aggregator counts
  (e.g. "Find 24,000+ properties", "133+ verified listings", magicbricks/99acres listing dumps).
- Do NOT paste "Source:" / "Details:" / "Link:" blocks from the snippets.
- Synthesize a clear investment briefing. Prefer facts that appear in the snippets;
  where snippets are thin, you may use well-known Noida corridor framing but mark
  uncertain figures as approximate and never invent exact circle rates or RERA IDs.
- Do NOT demand BHK or budget from the user — optionally invite it at the end only.
- Put key prices / ranges in **bold**.
- Plain text with **bold** for prices only (no other markdown).

REQUIRED STRUCTURE (use these exact section headings):

{opening}

1) High Rental Yield Corridors (corporate / metro belt)
   - Name 2–3 sectors / micro-markets
   - Why rentals hold (IT/office demand, metro, tenant pool)
   - Realistic entry budget range for typical stock (flats and/or plots if relevant)

2) High Capital Appreciation Sectors (low-density / lifestyle)
   - Name 2–3 sectors / micro-markets
   - Why prices tend to appreciate (scarcity, gated/lifestyle, end-user demand)
   - Realistic entry budget range

3) Long-Term Infrastructure Corridors (Jewar / Expressway)
   - Name corridors (e.g. Yamuna Expressway, Jewar airport influence, Noida–Greater Noida)
   - Investment rationale and time horizon (5–10+ years)
   - Realistic entry budget range and key risks (delivery, liquidity)

4) Quick pick by goal
   - Rental income → …
   - Appreciation → …
   - Long-term infra play → …

Close with one short disclaimer: this is market context from public web sources, not a
guarantee and not rows from our CSV inventory. Optionally invite budget/BHK only if they
want exact inventory matches next.

USER QUESTION:
{question}

WEB SEARCH SNIPPETS (context only — do not dump):
{web_data}
"""


def _llm_content_to_text(content: object) -> str:
    """Normalize LangChain / Gemini message content to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(p.strip() for p in parts if p and str(p).strip()).strip()
    text = getattr(content, "content", None)
    if text is not None and text is not content:
        return _llm_content_to_text(text)
    return str(content).strip()


def _decorate_facts_with_llm(
    question: str,
    facts: str,
    model: str | None = None,
) -> str:
    """Turn exact Pandas facts into a conversational final reply via the UI LLM."""
    facts = (facts or "").strip()
    if not facts:
        return facts
    try:
        prompt = _DECORATE_PROMPT.format(question=question.strip(), facts=facts)
        text, _which = _invoke_llm_with_gemini_backup(
            prompt, model=model, temperature=0.3
        )
        return text or facts
    except Exception as exc:  # noqa: BLE001
        # Prefer exact facts over a decoration failure.
        print(f"[langpanda] decorate failed: {exc}", file=sys.stderr)
        return facts


def format_web_fallback_with_llm(
    question: str,
    web_data: str,
    model: str | None = None,
    missing_attribute: str | None = None,
) -> str:
    """Synthesize a disclosed web-fallback answer from free search snippets."""
    from web_search import is_investment_advice_query

    web_data = (web_data or "").strip() or "No external web results found."
    investment = is_investment_advice_query(question)

    if missing_attribute:
        opening = (
            f"Our internal inventory does not track “{missing_attribute}”, so we "
            "cannot filter for it directly. However, based on current market "
            "context:"
        )
        attribute_note = (
            f"\nThe user is asking about “{missing_attribute}”, which is NOT a field "
            "in our CSV, so it must be answered from the web snippets."
        )
    elif investment:
        opening = (
            "Our CSV inventory can’t pick a single “best investment” listing — that "
            "depends on goal (rental yield vs appreciation vs long-term infra). "
            "Here is a structured Noida investment briefing based on public market context:"
        )
        attribute_note = ""
    else:
        opening = (
            "Our internal CSV inventory mainly covers apartment-style listings and "
            "had no direct match. Here is what public web sources show about this "
            "area’s broader real-estate profile (plots, houses, flats, commercial):"
        )
        attribute_note = (
            "\nNote: Internal CSV had 0 usable apartment matches for this ask. "
            "Synthesize the sector’s full real-estate mix from the web results."
        )

    # If LLM is unavailable, still return a usable plain-text fallback.
    try:
        if investment:
            prompt = _INVESTMENT_BRIEFING_PROMPT.format(
                question=question.strip(),
                web_data=web_data,
                opening=opening,
            )
        else:
            prompt = _WEB_FALLBACK_PROMPT.format(
                question=question.strip(),
                web_data=web_data,
                opening=opening,
                attribute_note=attribute_note,
            )
        cleaned, _which = _invoke_llm_with_gemini_backup(
            prompt,
            model=model,
            temperature=0.25 if investment else 0.2,
        )
        if cleaned and not _looks_like_raw_portal_dump(cleaned):
            low = cleaned.lower()
            already_opened = (
                "no direct matching properties" in low
                or "does not track" in low
                or "investment briefing" in low
                or "based on current market" in low
                or "csv inventory" in low
                or "broader real-estate profile" in low
            )
            if not already_opened:
                return f"{opening}\n\n{cleaned}"
            return cleaned
    except Exception as exc:  # noqa: BLE001
        print(f"[langpanda] web fallback LLM failed: {exc}", file=sys.stderr)

    if investment:
        return _investment_briefing_fallback(opening)
    return (
        f"{opening}\n\n"
        "• Public listings for this area often mix residential plots / land, "
        "independent houses or villas, society flats, and sometimes commercial stock.\n"
        "• Check recent portal rate cards for **price per sq.ft. / sq.yd.** on plots "
        "and typical ticket sizes for houses vs high-rise flats.\n"
        "• Ask about a specific asset type (plot / villa / flat / shop) if you want "
        "a tighter range next.\n\n"
        "Note: This is public web market context, not rows from our CSV inventory."
    )


def _looks_like_raw_portal_dump(text: str) -> bool:
    """Detect LLM replies that just parrot portal listing aggregators."""
    low = (text or "").lower()
    # Clear aggregator dump: many portal Link lines, or huge listing counts.
    if low.count("link:") >= 3 and (
        "99acres" in low or "magicbricks" in low or "housing.com" in low
    ):
        return True
    if re.search(r"\b\d{1,3},\d{3}\+?\s+(properties|flats|listings)\b", low):
        return True
    if re.search(r"\bfind\s+\d{2,}\+?\s+(properties|flats|listings)\b", low):
        return True
    # Mostly raw snippet paste (Source: + Details: repeated).
    if low.count("source:") >= 2 and low.count("details:") >= 2:
        return True
    return False


def _investment_briefing_fallback(opening: str) -> str:
    """Structured offline briefing when the LLM is down — never dump portal URLs."""
    return (
        f"{opening}\n\n"
        "1) High Rental Yield Corridors (corporate / metro belt)\n"
        "   Sectors along the Noida metro / office belt (e.g. around Sec 62–63 IT hubs "
        "and well-connected mid-Noida micro-markets) typically attract salaried tenants. "
        "Entry for a rentable 2/3 BHK often sits in the mid–high lakh to low-crore band "
        "depending on size and finish.\n\n"
        "2) High Capital Appreciation Sectors (low-density / lifestyle)\n"
        "   Scarcer, end-user-heavy pockets (premium gated / lifestyle belts in "
        "established Noida sectors) usually trade at higher ticket sizes but hold "
        "resale demand better when supply is tight.\n\n"
        "3) Long-Term Infrastructure Corridors (Jewar / Expressway)\n"
        "   Yamuna Expressway / Jewar airport influence zones can offer lower entry "
        "and longer-horizon appreciation, with higher delivery and liquidity risk.\n\n"
        "4) Quick pick by goal\n"
        "   Rental income → metro / office-adjacent ready stock\n"
        "   Appreciation → scarce lifestyle / established sectors\n"
        "   Long-term infra → Expressway / Jewar belt (patient capital)\n\n"
        "Share your budget and preferred BHK and I can match exact listings from our "
        "CSV inventory next. This is general market context, not a guarantee."
    )


def execute_pandas_query(
    user_prompt: str,
    *,
    use_llm: bool = False,
    model: str | None = None,
) -> tuple[str | None, bool]:
    """
    Run deterministic / agent Pandas paths.
    Returns (reply_or_facts, has_inventory_hit).
    """
    extreme = _try_deterministic_extreme_answer(user_prompt)
    if extreme is not None:
        return extreme, not _pandas_reply_is_empty(extreme)

    compare = _try_older_vs_newer_answer(user_prompt)
    if compare is not None:
        return compare, not _pandas_reply_is_empty(compare)

    fast = _try_fast_pandas_answer(user_prompt)
    if fast is not None:
        return fast, not _pandas_reply_is_empty(fast)

    if use_llm:
        agent_reply = _invoke_agent(user_prompt, model=model)
        return agent_reply, not _pandas_reply_is_empty(agent_reply)

    return (
        "Only Pandas mode could not parse that into exact filters. "
        "Switch the Pandas toggle to “Pandas + LLM”, or ask with clear "
        "sector / BHK / amenity / price conditions.",
        False,
    )


def _pandas_reply_is_empty(reply: str | None) -> bool:
    from web_search import is_empty_pandas_reply

    return is_empty_pandas_reply(reply)


def _web_fallback_enabled() -> bool:
    raw = (os.environ.get("LANGPANDA_WEB_FALLBACK") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def run_real_estate_pipeline(
    user_query: str,
    *,
    use_llm: bool = False,
    model: str | None = None,
    force_gemini: bool = False,
    word_limit: int | None = None,
) -> tuple[str, str]:
    """
    Unified pipeline:
      1) Pandas against clean_dataset.csv  (Only Pandas / Pandas + LLM)
      2) If hits → format (optional LLM polish)
      3) Else / pure external market query → web + LLM (or Gemini knowledge)
    Returns (reply, source).
    force_gemini=True → ASK DGB-SUP: Gemini own knowledge ONLY
      (no CSV, no DuckDuckGo / web_search, no external tools).
    """
    from web_search import (
        is_external_market_query,
        is_real_estate_scope,
        is_soft_unsupported_attribute,
        out_of_scope_reply,
        references_unsupported_attribute,
    )

    q = (user_query or "").strip()
    if not q:
        return "Please ask a property question.", "error"

    if force_gemini:
        # ASK DGB-SUP — exit immediately; never reach Pandas or DDGS below.
        if not model or "ollama" in model.lower() or model.lower().startswith("llama"):
            model = (
                os.environ.get("LANGPANDA_GEMINI_MODEL")
                or os.environ.get("GEMINI_MODEL")
                or "gemini-3.6-flash"
            )
        if not is_real_estate_scope(q):
            return out_of_scope_reply(), "out-of-scope"
        return answer_with_gemini_only(q, model=model, word_limit=word_limit)

    # Off-topic (cars, phones, food, …) → polite refusal, never Pandas/web.
    if not is_real_estate_scope(q):
        return out_of_scope_reply(), "out-of-scope"

    intent = parse_listing_intent(q)
    has_inventory_shape = bool(
        intent
        and (
            intent.get("sector")
            or intent.get("bhk") is not None
            or intent.get("min_price") is not None
            or intent.get("max_price") is not None
            or intent.get("required_amenities")
            or intent.get("forbidden_amenities")
            or intent.get("extreme")
            or intent.get("status")
        )
    )

    # Attribute not present as a CSV column (e.g. pet-friendly, gated, loan).
    missing_attr = references_unsupported_attribute(q)
    soft_missing = is_soft_unsupported_attribute(missing_attr)

    # Force web only when:
    #  - hard missing attribute AND no inventory filters, OR
    #  - pure market/investment question with no inventory filters
    # Soft attrs like "near transport" never skip Pandas when BHK/price exist.
    force_web = bool(
        use_llm
        and _web_fallback_enabled()
        and not has_inventory_shape
        and (
            (missing_attr is not None and not soft_missing)
            or is_external_market_query(q)
        )
    )

    def _tag(source: str) -> str:
        return source

    if not force_web:
        facts, has_hit = execute_pandas_query(q, use_llm=use_llm, model=model)
        facts = _clean_agent_reply(facts or "")
        if has_hit and facts:
            if use_llm:
                reply = _decorate_facts_with_llm(q, facts, model=model)
            else:
                reply = facts
            # Soft note: we matched inventory filters but can't filter transport distance.
            if missing_attr and soft_missing:
                reply = (
                    f"{reply.rstrip()}\n\n"
                    f"Note: We can’t filter exact “{missing_attr}” from the CSV — "
                    "confirm metro/transport access for shortlisted listings on a map."
                )
            return reply, _tag("pandas-llm" if use_llm else "pandas-only")

        # Only Pandas: do not auto-web; return the empty/error message.
        if not use_llm or not _web_fallback_enabled():
            return (
                facts or "0 matching properties found for this combination.",
                "pandas-empty",
            )

        # Pandas + LLM with 0 rows / failed agent → web, else Gemini knowledge
        reply, source = synthesize_external_answer(
            q,
            model=model,
            missing_attribute=None if has_inventory_shape else missing_attr,
        )
        return reply, _tag(source)

    # Forced external path: missing hard CSV attribute or pure market question.
    reply, source = synthesize_external_answer(
        q, model=model, missing_attribute=missing_attr
    )
    return reply, _tag(source)


def query_real_estate_agent(
    user_prompt: str,
    use_llm: bool = False,
    model: str | None = None,
    force_gemini: bool = False,
    word_limit: int | None = None,
) -> str:
    """
    use_llm=False → Only Pandas (exact facts; no web).
    use_llm=True  → Pandas first, then free web search fallback when needed.
    force_gemini  → Manual Gemini Only mode (never Ollama).
    model         → UI answer-model id (gemini-… or ollama/…).
    word_limit    → ASK DGB-SUP max words (0 = unlimited).
    Always returns cleaned final text for the API/UI (no chain logs).
    """
    try:
        reply, _source = run_real_estate_pipeline(
            user_prompt,
            use_llm=use_llm,
            model=model,
            force_gemini=force_gemini,
            word_limit=word_limit,
        )
        return _clean_agent_reply(reply)
    except Exception as e:
        return f"Execution error: {str(e)}"


def _format_match_summary(hits: pd.DataFrame, scope: str) -> str:
    """Decorated multi-row reply: natural headline, price summary, 2–3 samples."""
    ranked = rank_listing_hits(hits, prefer_transport=False, extreme="best")
    n = len(ranked)
    n_raw = len(hits)
    if n_raw == 0:
        return (
            f"I checked the listings for {scope}, but none match all of those "
            "conditions in the provided data."
        )
    if n == 0:
        return (
            f"I found {n_raw} raw matches for {scope}, but none look realistic "
            "after removing outlier prices/sizes."
        )

    priced = (
        ranked["price_in_lakh"].dropna()
        if "price_in_lakh" in ranked.columns
        else pd.Series(dtype=float)
    )
    lines = [
        f"Yes — I found {n} realistic matching properties for {scope}"
        + (f" (cleaned from {n_raw} raw rows)" if n_raw != n else "")
        + ".",
        "",
    ]
    if len(priced) > 0:
        lines.extend(
            [
                "Price summary (total price_in_lakh):",
                f"• Minimum: ₹{float(priced.min()):,.2f} lakh",
                f"• Maximum: ₹{float(priced.max()):,.2f} lakh",
                f"• Average: ₹{float(priced.mean()):,.2f} lakh",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Price summary: prices are not listed for these matching rows.",
                "",
            ]
        )

    lines.append("Top options (quality-ranked):")
    for i, (_, row) in enumerate(ranked.head(3).iterrows(), start=1):
        beds = row.get("bedroom")
        beds_s = f"{int(beds)} BHK" if pd.notna(beds) else "BHK n/a"
        addr = str(row.get("address") or "Noida")
        size = row.get("size")
        size_s = f"{int(size):,} sqft" if pd.notna(size) else "size not listed"
        price = row.get("price_in_lakh")
        price_s = (
            f"₹{float(price):,.2f} lakh"
            if pd.notna(price)
            else "price not listed"
        )
        age = row.get("age_of_property") if "age_of_property" in ranked.columns else None
        age_s = f"{float(age):g} years old" if pd.notna(age) else None
        bits = [f"{beds_s} in {addr}", size_s, price_s]
        if age_s:
            bits.append(age_s)
        lines.append(f"{i}. " + " · ".join(bits))

    if n > 3:
        lines.append("")
        lines.append(f"…and {n - 3} more matches in the cleaned dataset.")
    return "\n".join(lines)


def _clean_agent_reply(text: str) -> str:
    """Strip ANSI / AgentExecutor chain noise; keep final answer only."""
    if not text:
        return ""
    cleaned = _ANSI_RE.sub("", str(text)).replace("\r", "")
    # Prefer explicit Final Answer block when chain text leaked in.
    final_m = re.search(
        r"(?is)Final\s*Answer\s*:\s*(.+?)(?:\n\s*(?:Thought|Action|>\s*Finished)|$)",
        cleaned,
    )
    if final_m:
        cleaned = final_m.group(1).strip()
    # Drop common chain lines if still present.
    kept: list[str] = []
    for line in cleaned.splitlines():
        if _CHAIN_NOISE_RE.search(line):
            continue
        if re.match(r"^\s*(Thought|Action|Observation)\s*:", line, re.I):
            continue
        kept.append(line)
    cleaned = "\n".join(kept).strip()
    return cleaned or str(text).strip()


def _invoke_agent(user_prompt: str, model: str | None = None) -> str:
    """
    Run the LangChain agent and return only the final parsed answer.
    Verbose chain logs go to the server terminal (stderr), never the API payload.
    """
    agent = get_agent(model)
    # Keep AgentExecutor verbose prints on the server terminal only.
    with contextlib.redirect_stdout(sys.stderr):
        if hasattr(agent, "invoke"):
            result = agent.invoke({"input": user_prompt})
        else:
            result = agent.run(user_prompt)

    if isinstance(result, dict):
        output = result.get("output")
        if output is None:
            intermediate = result.get("intermediate_steps")
            output = result.get("result") or (
                "" if intermediate is not None else result
            )
        text = str(output if output is not None else "")
    else:
        text = str(result)
    return _clean_agent_reply(text)


def get_agent(model: str | None = None):
    """Lazy-create the Pandas dataframe agent (rebuilds when model changes)."""
    global _llm, _agent, _agent_key
    provider, name = _resolve_llm_choice(model)
    key = f"{provider}:{name}"
    if _agent is not None and _agent_key == key:
        return _agent

    _llm = _build_llm(model)
    # verbose=True logs to stdout; _invoke_agent redirects that to stderr
    # so HTTP/Python API replies never include the chain dump.
    _agent = create_pandas_dataframe_agent(
        _llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        prefix=system_prefix,
        max_iterations=20,  # Local Ollama models need more room to finish
        agent_executor_kwargs={"handle_parsing_errors": True},
        return_intermediate_steps=False,
    )
    _agent_key = key
    return _agent


# Longer / more specific NL phrases first so "parking" wins over "park".
AMENITY_NL_MAP: list[tuple[str, str]] = [
    (r"\b(swimming\s*pool)\b", "amenities_swimming_pool"),
    (r"\bpool\b", "amenities_swimming_pool"),
    (r"\b(gymnasium|gym)\b", "amenities_gymnasium"),
    (r"\b(car\s*parking|parking)\b", "amenities_car_parking"),
    (r"\b(lift|elevator)\b", "amenities_lift"),
    (r"\b(power\s*backup|full_power_backup|backup)\b", "amenities_full_power_backup"),
    (r"\b(24\s*[x×]\s*7\s*security|24/7\s*security|cctv|security)\b", "amenities_24_x_7_security"),
    (r"\b(club\s*house|clubhouse|club)\b", "amenities_club_house"),
    (r"\b(sports\s*facility|sports)\b", "amenities_sports_facility"),
    (r"\b(landscaped(\s*gardens?)?|garden)\b", "amenities_landscaped_gardens"),
    (r"\bpark\b", "amenities_landscaped_gardens"),
    (r"\b(play\s*area|kids\s*area|children'?s?\s*play\s*area)\b", "amenities_childrens_play_area"),
    (r"\b(jogging\s*track)\b", "amenities_jogging_track"),
    (r"\b(golf\s*course|golf)\b", "amenities_golf_course"),
    (r"\bintercom\b", "amenities_intercom"),
    (r"\b(rain\s*water\s*harvesting|rainwater)\b", "amenities_rain_water_harvesting"),
    (r"\bmaintenance\b", "amenities_maintenance_staff"),
    (r"\bhospital\b", "amenities_hospital"),
    (r"\bschool\b", "amenities_school"),
    (r"\b(shopping\s*mall|shopping|mall)\b", "amenities_shopping_mall"),
    (r"\b(indoor\s*games)\b", "amenities_indoor_games"),
    (r"\batm\b", "amenities_atm"),
    (r"\b(cafeteria|cafe)\b", "amenities_cafeteria"),
    (r"\b(vaastu|vastu)\b", "amenities_vaastu_compliant"),
    (r"\bstaff\s*quarter\b", "amenities_staff_quarter"),
    (r"\bmultipurpose\s*room\b", "amenities_multipurpose_room"),
]


def filter_properties(
    frame: pd.DataFrame,
    sector: str | None = None,
    bhk: int | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    required_amenities: list[str] | None = None,
    forbidden_amenities: list[str] | None = None,
    *,
    status: str | None = None,
    min_age: float | None = None,
    max_age: float | None = None,
    older_than: float | None = None,
    require_semi: bool = False,
    require_ac: bool | None = None,
    require_gas: bool | None = None,
) -> pd.DataFrame:
    """
    Unified filter pipeline: apply ALL constraints in one combined boolean mask
    BEFORE counts, statistics, extremes, or samples.
    """
    mask = pd.Series(True, index=frame.index)
    price = pd.to_numeric(frame["price_in_lakh"], errors="coerce")

    if bhk is not None:
        mask &= frame["bedroom"] == bhk

    if sector:
        mask &= frame["address"].astype(str).str.contains(
            sector, case=False, na=False
        )

    if min_price is not None:
        mask &= price >= min_price
    if max_price is not None:
        mask &= price <= max_price

    for col in required_amenities or []:
        if col in frame.columns:
            mask &= frame[col] == True  # noqa: E712

    for col in forbidden_amenities or []:
        if col in frame.columns:
            mask &= frame[col] == False  # noqa: E712

    if status and "status" in frame.columns:
        mask &= (
            frame["status"]
            .astype(str)
            .str.contains(status, case=False, na=False)
        )

    if "age_of_property" in frame.columns:
        age = pd.to_numeric(frame["age_of_property"], errors="coerce")
        if older_than is not None:
            mask &= age > older_than
        if min_age is not None:
            mask &= age >= min_age
        if max_age is not None:
            mask &= age <= max_age

    if require_semi and "status.1" in frame.columns:
        mask &= (
            frame["status.1"]
            .astype(str)
            .str.contains("semi", case=False, na=False)
        )

    if require_ac is not None and "furnish_detail_ac" in frame.columns:
        mask &= frame["furnish_detail_ac"] == require_ac
    if (
        require_gas is not None
        and "furnish_detail_gas_connection" in frame.columns
    ):
        mask &= frame["furnish_detail_gas_connection"] == require_gas

    return frame.loc[mask]


# Noida / GN sectors with relatively strong metro / expressway access.
_TRANSPORT_SECTORS = {
    "15", "16", "18", "22", "32", "34", "50", "51", "52", "53",
    "61", "62", "71", "72", "76", "77", "78", "81", "82", "93",
    "93a", "93b", "100", "101", "104", "116", "119", "120", "121",
    "122", "128", "135", "137", "142", "143", "144", "150", "168",
}


def _sector_token_from_address(addr: str) -> str | None:
    m = re.search(r"sector\s*([0-9]+[a-z]?)", str(addr).lower())
    return m.group(1) if m else None


def _is_transport_address(addr: str) -> bool:
    a = str(addr).lower()
    tok = _sector_token_from_address(a)
    if tok and tok in _TRANSPORT_SECTORS:
        return True
    return bool(
        re.search(
            r"\b(metro|expressway|yamuna|botanical|noida\s+city\s+centre|"
            r"sector\s*18|golf\s+course)\b",
            a,
        )
    )


def sanitize_listing_hits(hits: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate / unrealistic CSV rows so “best under budget” samples
    aren’t ₹19L for 2500 sqft junk or 18,000 sqft typos.
    """
    if hits is None or hits.empty:
        return hits
    out = hits.copy()
    price = pd.to_numeric(out["price_in_lakh"], errors="coerce")
    size = pd.to_numeric(out["size"], errors="coerce") if "size" in out.columns else None
    out = out[price.fillna(0) > 0]
    if size is not None:
        # Typical apartment band; keep plots-ish only if clearly priced.
        ok_size = (size >= 450) & (size <= 4500)
        pps = (price * 100_000) / size.clip(lower=1)
        # Noida flats rarely trade below ~₹2,500/sqft in real market data.
        ok_pps = (pps >= 2500) & (pps <= 45000)
        out = out[ok_size & ok_pps]
    # Deduplicate near-identical listings
    out = out.assign(
        _addr=out["address"].astype(str).str.lower().str.strip(),
        _price=pd.to_numeric(out["price_in_lakh"], errors="coerce").round(2),
        _size=(
            pd.to_numeric(out["size"], errors="coerce").round(0)
            if "size" in out.columns
            else 0
        ),
        _beds=(
            pd.to_numeric(out["bedroom"], errors="coerce")
            if "bedroom" in out.columns
            else 0
        ),
    )
    out = out.drop_duplicates(subset=["_addr", "_price", "_size", "_beds"], keep="first")
    return out.drop(columns=[c for c in out.columns if c.startswith("_")], errors="ignore")


def score_listing_row(row: pd.Series, *, prefer_transport: bool = False) -> float:
    """Higher = better shortlist candidate for ‘best under budget’."""
    price = float(row["price_in_lakh"]) if pd.notna(row.get("price_in_lakh")) else 0.0
    size = float(row["size"]) if pd.notna(row.get("size")) else 0.0
    amen_cols = [c for c in row.index if str(c).startswith("amenities_")]
    amen_n = sum(1 for c in amen_cols if bool(row.get(c)) is True)
    score = amen_n * 12.0 + min(size, 2800) / 40.0 + price * 0.15
    if prefer_transport and _is_transport_address(str(row.get("address") or "")):
        score += 35.0
    return score


def rank_listing_hits(
    hits: pd.DataFrame,
    *,
    prefer_transport: bool = False,
    extreme: str | None = None,
) -> pd.DataFrame:
    clean = sanitize_listing_hits(hits)
    if clean.empty:
        return clean
    if extreme == "min":
        return clean.sort_values(by="price_in_lakh", ascending=True)
    if extreme == "max":
        return clean.sort_values(by="price_in_lakh", ascending=False)
    # best / default shortlist: quality score (not cheapest junk)
    scored = clean.copy()
    scored["_score"] = scored.apply(
        lambda r: score_listing_row(r, prefer_transport=prefer_transport),
        axis=1,
    )
    return scored.sort_values(by=["_score", "price_in_lakh"], ascending=[False, False]).drop(
        columns=["_score"], errors="ignore"
    )


def query_dataset(
    frame: pd.DataFrame,
    bhk: int | None = None,
    sector: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    req_amenities: list[str] | None = None,
    forbidden_amenities: list[str] | None = None,
) -> pd.DataFrame:
    """Public alias matching the unified compound-filter template."""
    return filter_properties(
        frame,
        sector=sector,
        bhk=bhk,
        min_price=min_price,
        max_price=max_price,
        required_amenities=req_amenities or [],
        forbidden_amenities=forbidden_amenities or [],
    )


def _to_lakh(num: str | float, unit: str | None = None) -> float:
    """Currency standardizer: Crore → Lakh (* 100)."""
    value = float(num)
    if unit:
        u = unit.lower().strip()
        if u.startswith("cr"):
            return value * 100.0
    return value


def _parse_price_bounds(q: str) -> tuple[float | None, float | None]:
    """
    Extract min/max price_in_lakh from NL.
    Supports between / X to Y / X - Y / under / above, with Cr→Lakh conversion.
    Never drops the lower bound when an interval is present.
    """
    unit = r"(cr|crore|lakh|lakhs?)?"
    num = r"(\d+(?:\.\d+)?)"

    # 1) Between X and Y
    between_m = re.search(
        rf"\bbetween\s+{num}\s*{unit}\s*(?:and|to|-|–|—)\s*{num}\s*{unit}\b",
        q,
    )
    if between_m:
        lo = _to_lakh(between_m.group(1), between_m.group(2))
        hi = _to_lakh(between_m.group(3), between_m.group(4))
        return (min(lo, hi), max(lo, hi))

    # 2) X to Y / X - Y / from X to Y  (e.g. "40 lakhs to 60 lakhs", "1.2-1.5 cr")
    range_m = re.search(
        rf"\b(?:from\s+)?{num}\s*{unit}\s*(?:to|-|–|—)\s*{num}\s*{unit}\b",
        q,
    )
    if range_m:
        # Avoid age ranges like "2 to 5 years"
        span = range_m.group(0)
        after = q[range_m.end() : range_m.end() + 12]
        if re.search(r"\b(year|yr|years|yrs)\b", span) or re.match(
            r"\s*(year|yr|years|yrs)\b", after
        ):
            pass
        else:
            # Prefer when currency words appear, or price/budget context in query
            has_currency = bool(
                range_m.group(2)
                or range_m.group(4)
                or re.search(r"\b(price|budget|cost|lakh|crore|cr)\b", q)
            )
            if has_currency:
                lo = _to_lakh(range_m.group(1), range_m.group(2))
                hi = _to_lakh(range_m.group(3), range_m.group(4))
                # If only second unit given ("40 to 60 lakh"), apply to both
                if range_m.group(4) and not range_m.group(2):
                    lo = _to_lakh(range_m.group(1), range_m.group(4))
                if range_m.group(2) and not range_m.group(4):
                    hi = _to_lakh(range_m.group(3), range_m.group(2))
                return (min(lo, hi), max(lo, hi))

    min_price: float | None = None
    max_price: float | None = None

    # 3) Under / Below / Less than / up to X
    under_m = re.search(
        rf"\b(?:under|below|less\s+than|at\s+most|max(?:imum)?|upto|up\s+to)"
        rf"\s+{num}\s*{unit}\b",
        q,
    )
    if under_m:
        # Skip age: "under 5 years"
        after = q[under_m.end() : under_m.end() + 12]
        if not re.match(r"\s*(year|yr|years|yrs)\b", after):
            if under_m.group(2) or re.search(
                r"\b(price|budget|cost|lakh|crore|cr)\b", q
            ):
                max_price = _to_lakh(under_m.group(1), under_m.group(2))

    # 4) Above / Over / More than / at least X
    over_m = re.search(
        rf"\b(?:over|above|more\s+than|at\s+least|min(?:imum)?)"
        rf"\s+{num}\s*{unit}\b",
        q,
    )
    if over_m:
        after = q[over_m.end() : over_m.end() + 12]
        # Skip age: "more than 5 years" / "older than…" handled elsewhere
        if not re.match(r"\s*(year|yr|years|yrs)\b", after) and not re.search(
            r"\bolder\s+than\b", q
        ):
            if over_m.group(2) or re.search(
                r"\b(price|budget|cost|lakh|crore|cr)\b", q
            ):
                min_price = _to_lakh(over_m.group(1), over_m.group(2))

    return min_price, max_price


def parse_listing_intent(user_prompt: str) -> dict | None:
    """Extract sector / BHK / price / amenity / extreme intent from NL."""
    q = user_prompt.lower().strip()
    if not q:
        return None

    sector_m = re.search(r"sector\s*([0-9]+[a-z]?)", q)
    bhk_m = re.search(r"(\d+)\s*(?:bhk|rk)\b", q)

    # Price bounds: keep BOTH lower and upper when an interval is present.
    min_price, max_price = _parse_price_bounds(q)

    # Age filters
    older_than: float | None = None
    min_age: float | None = None
    max_age: float | None = None
    older_m = re.search(
        r"\b(?:older\s+than|more\s+than|over|above)\s+(\d+)\s*(?:year|yr)s?\b",
        q,
    )
    newer_m = re.search(
        r"\b(?:under|less\s+than|below|younger\s+than|at\s+most)\s+(\d+)\s*(?:year|yr)s?\b",
        q,
    )
    at_least_age_m = re.search(
        r"\b(?:at\s+least|minimum|min)\s+(\d+)\s*(?:year|yr)s?\b", q
    )
    if older_m:
        older_than = float(older_m.group(1))
    elif at_least_age_m:
        min_age = float(at_least_age_m.group(1))
    if newer_m and "year" in newer_m.group(0):
        max_age = float(newer_m.group(1))

    required: list[str] = []
    forbidden: list[str] = []
    for pattern, col in AMENITY_NL_MAP:
        for match in re.finditer(pattern, q):
            start, end = match.span()
            before = q[max(0, start - 48) : start]
            # "does it have a lift?" is a detail question, not a required filter.
            if re.search(
                r"\b(does\s+it\s+have|do\s+they\s+have|has\s+it\s+got|has\s+it|"
                r"is\s+there(\s+a)?|what\s+about|and\s+does\s+it)\b",
                before,
            ):
                continue
            # Negation must appear BEFORE the amenity term ("without pool").
            negative = bool(
                re.search(
                    r"\b(without|no|not|excluding|exclude|missing|absent)\b"
                    r"|does\s+not\s+have|doesn't\s+have|dont\s+have|don't\s+have"
                    r"|=\s*false|\bfalse\b",
                    before,
                )
            )
            if negative:
                if col not in forbidden:
                    forbidden.append(col)
            else:
                if col not in required:
                    required.append(col)

    # "all amenities" / typo "all aminiter" → require a practical core set
    # (not every amenities_* column, which would almost always return 0).
    if re.search(r"\ball\s+amenit\w*\b|\ball\s+amin\w*\b", q):
        core = [
            "amenities_gymnasium",
            "amenities_swimming_pool",
            "amenities_lift",
            "amenities_full_power_backup",
            "amenities_24_x_7_security",
            "amenities_car_parking",
        ]
        for col in core:
            if col not in required:
                required.append(col)

    # Furnish extras (not amenities_*)
    require_ac: bool | None = None
    require_gas: bool | None = None
    ac_m = re.search(r"\b(ac|a/?c|air\s*condition)", q)
    if ac_m:
        window = q[max(0, ac_m.start() - 40) : ac_m.end() + 20]
        require_ac = not bool(
            re.search(r"\b(without|no|not)\b|=\s*false|\bfalse\b", window)
        )
    gas_m = re.search(r"\bgas(\s*connection)?\b", q)
    if gas_m:
        window = q[max(0, gas_m.start() - 40) : gas_m.end() + 20]
        require_gas = not bool(
            re.search(r"\b(without|no|not)\b|=\s*false|\bfalse\b", window)
        )
    require_semi = bool(re.search(r"\bsemi[-\s]?furnished\b", q))

    status: str | None = None
    if re.search(r"\bready\s*to\s*move\b", q):
        status = "ready to move"
    elif re.search(r"\bunder\s*construction\b", q):
        status = "under construction"

    extreme: str | None = None
    if re.search(
        r"\b(cheapest|least\s+expensive|lowest[-\s]+priced?|lowest\s+price"
        r"|minimum[-\s]+price|min(?:imum)?\s+price)\b",
        q,
    ) or (
        re.search(r"\b(lowest|cheapest)\b", q)
        and re.search(r"\b(propert|listing|flat|apartment|bhk|price|lakh)\b", q)
    ):
        extreme = "min"
    elif re.search(
        r"\b(most\s+expensive|highest[-\s]+priced?|highest\s+price"
        r"|maximum[-\s]+price|max(?:imum)?\s+price|costliest)\b",
        q,
    ) or (
        re.search(r"\b(highest|priciest)\b", q)
        and re.search(r"\b(propert|listing|flat|apartment|bhk|price|lakh)\b", q)
    ):
        extreme = "max"
    elif re.search(r"\bbest\b", q) and re.search(
        r"\b(propert|flat|apartment|bhk|under|below|budget)\b", q
    ):
        # "best 3BHK under 1 Cr" → quality rank, NOT cheapest dump / investment brief
        extreme = "best"

    prefer_transport = bool(
        re.search(
            r"\b("
            r"near(?:\s+by)?\s+(?:to\s+)?(?:the\s+)?"
            r"(?:metro|trans\w*|tras\w*port|trasport|station|transport)|"
            r"near\b.{0,24}\b(?:metro|trans\w*|tras\w*port|station)\b|"
            r"nearby|near\s+by|"
            r"metro\s+(?:access|connectivity)|"
            r"close\s+to\s+(?:metro|transport|station)"
            r")\b",
            q,
        )
    )

    wants_count = bool(
        re.search(r"\b(how\s+many|count|number\s+of|total)\b", q)
    )

    sector = f"sector {sector_m.group(1).lower()}" if sector_m else None
    bhk = int(bhk_m.group(1)) if bhk_m else None

    has_structure = any(
        [
            sector,
            bhk is not None,
            min_price is not None,
            max_price is not None,
            required,
            forbidden,
            status,
            older_than is not None,
            min_age is not None,
            max_age is not None,
            require_semi,
            require_ac is not None,
            require_gas is not None,
            extreme is not None,
            prefer_transport,
        ]
    )
    if not has_structure:
        return None

    labels: list[str] = []
    if sector:
        labels.append(sector)
    if bhk is not None:
        labels.append(f"{bhk} BHK")
    if min_price is not None:
        labels.append(f"price_in_lakh ≥ {min_price:g}")
    if max_price is not None:
        labels.append(f"price_in_lakh ≤ {max_price:g}")
    if status:
        labels.append(f"status≈{status}")
    if older_than is not None:
        labels.append(f"age_of_property > {older_than:g}")
    if min_age is not None:
        labels.append(f"age_of_property ≥ {min_age:g}")
    if max_age is not None:
        labels.append(f"age_of_property ≤ {max_age:g}")
    if require_semi:
        labels.append("Semi-Furnished")
    if require_ac is not None:
        labels.append(f"AC={require_ac}")
    if require_gas is not None:
        labels.append(f"gas={require_gas}")
    for col in required:
        labels.append(f"{col}=True")
    for col in forbidden:
        labels.append(f"{col}=False")
    if prefer_transport:
        labels.append("prefer_transport_sectors")
    if extreme:
        labels.append(f"extreme={extreme}")

    return {
        "sector": sector,
        "bhk": bhk,
        "min_price": min_price,
        "max_price": max_price,
        "required_amenities": required,
        "forbidden_amenities": forbidden,
        "status": status,
        "older_than": older_than,
        "min_age": min_age,
        "max_age": max_age,
        "require_semi": require_semi,
        "require_ac": require_ac,
        "require_gas": require_gas,
        "extreme": extreme,
        "prefer_transport": prefer_transport,
        "wants_count": wants_count,
        "scope": ", ".join(labels) if labels else "filters",
    }


def _row_confirmed_amenities(row: pd.Series) -> str:
    confirmed: list[str] = []
    seen: set[str] = set()
    for _, col in AMENITY_NL_MAP:
        if col in seen:
            continue
        seen.add(col)
        if col in row.index and bool(row.get(col)) is True:
            confirmed.append(col.replace("amenities_", "").replace("_", " "))
    return ", ".join(confirmed[:8]) if confirmed else "none listed True"


def _format_extreme_row(row: pd.Series, label: str, scope: str) -> str:
    beds = row.get("bedroom")
    beds_s = f"{int(beds)} BHK" if pd.notna(beds) else "BHK n/a"
    addr = str(row.get("address") or "Noida")
    price = float(row["price_in_lakh"])
    size = row.get("size")
    size_s = f"{float(size):,.0f} sqft" if pd.notna(size) else "not listed"
    lift = row.get("amenities_lift") if "amenities_lift" in row.index else None
    if lift is None or (isinstance(lift, float) and pd.isna(lift)) or pd.isna(lift):
        lift_s = "not listed"
    elif bool(lift):
        lift_s = "Yes (amenities_lift=True)"
    else:
        lift_s = "No (amenities_lift=False)"

    detail_cols = [
        c
        for c in [
            "address",
            "bedroom",
            "price_in_lakh",
            "size",
            "rate",
            "age_of_property",
            "status",
            "status.1",
            *[col for _, col in AMENITY_NL_MAP],
        ]
        if c in df.columns
    ]

    def _native(val: object, *, as_bool: bool = False):
        if val is None or pd.isna(val):
            return None
        if as_bool:
            return bool(val)
        if hasattr(val, "item"):
            try:
                return val.item()
            except Exception:
                pass
        return val

    target_details = {
        c: _native(row.get(c), as_bool=str(c).startswith("amenities_"))
        for c in detail_cols
    }

    return "\n".join(
        [
            f"Exact {label} property for {scope} "
            f"(sorted by price_in_lakh). This is ONE row — not head(3).",
            "",
            f"Property: {beds_s} in {addr}",
            f"price_in_lakh: ₹{price:,.2f} lakh",
            f"size: {size_s}",
            f"lift: {lift_s}",
            f"Confirmed amenities: {_row_confirmed_amenities(row)}",
            "",
            f"target_details={target_details}",
            f"Filters applied: {scope}",
        ]
    )


def format_filtered_answer(intent: dict, filtered: pd.DataFrame) -> str:
    """Return count + price range + quality samples, or exact extreme row."""
    scope = intent.get("scope") or "filters"
    prefer_transport = bool(intent.get("prefer_transport"))
    extreme = intent.get("extreme")

    ranked = rank_listing_hits(
        filtered,
        prefer_transport=prefer_transport,
        extreme=extreme if extreme in {"min", "max"} else "best",
    )
    n_raw = len(filtered)
    n = len(ranked)
    if n_raw == 0:
        return "0 matching properties found for this combination."
    if n == 0:
        return (
            f"Found {n_raw} rows for {scope}, but none look like realistic "
            "market prices/sizes after cleaning outlier CSV values. "
            "Try a different budget or sector."
        )

    if intent.get("wants_count") and extreme is None:
        return f"There are {n} realistic matching properties for {scope} (from {n_raw} raw rows)."

    priced = ranked[
        pd.to_numeric(ranked["price_in_lakh"], errors="coerce").fillna(0) > 0
    ].copy()

    if extreme in {"min", "max"}:
        if priced.empty:
            return (
                f"Found {n} matching properties for {scope}, but none list "
                "price_in_lakh > 0, so an extreme price row cannot be ranked."
            )
        row = priced.iloc[0]
        label = "cheapest" if extreme == "min" else "most expensive"
        return _format_extreme_row(row, label, scope)

    if extreme == "best":
        top = ranked.iloc[0]
        beds = top.get("bedroom")
        beds_s = f"{int(beds)} BHK" if pd.notna(beds) else "BHK n/a"
        addr = str(top.get("address") or "Noida")
        size = top.get("size")
        size_s = f"{int(size):,} sqft" if pd.notna(size) else "size n/a"
        price = float(top["price_in_lakh"])
        pps = (price * 100_000 / float(size)) if pd.notna(size) and float(size) > 0 else None
        pps_s = f" · ~₹{pps:,.0f}/sqft" if pps else ""
        transport = " · transport-connected belt" if _is_transport_address(addr) else ""
        lines = [
            f"Best match for {scope} "
            f"(ranked by amenities + size + realistic price"
            f"{' + transport-connected sectors' if prefer_transport else ''}; "
            f"junk/outlier CSV rows removed):",
            "",
            f"• {addr} · {beds_s} · ₹{price:,.2f} lakh · {size_s}{pps_s}{transport}",
            f"  Amenities: {_row_confirmed_amenities(top)}",
        ]
        if n > 1:
            lines.append("")
            lines.append("Other strong options:")
            for i, (_, row) in enumerate(ranked.iloc[1:4].iterrows(), start=1):
                beds = row.get("bedroom")
                beds_s = f"{int(beds)} BHK" if pd.notna(beds) else "BHK n/a"
                addr = str(row.get("address") or "Noida")
                size = row.get("size")
                size_s = f"{int(size):,} sqft" if pd.notna(size) else "size n/a"
                price = row.get("price_in_lakh")
                price_s = (
                    f"₹{float(price):,.2f} lakh" if pd.notna(price) else "price n/a"
                )
                transport = (
                    " · transport-connected belt"
                    if _is_transport_address(addr)
                    else ""
                )
                lines.append(
                    f"{i}. {addr} · {beds_s} · {price_s} · {size_s} · "
                    f"amenities: {_row_confirmed_amenities(row)}{transport}"
                )
        if prefer_transport:
            lines.append("")
            lines.append(
                "Note: Exact walking distance to metro/transport isn’t in the CSV — "
                "shortlist favors better-connected sectors; confirm on a map."
            )
        lines.append("")
        lines.append(
            f"Cleaned pool: {n} realistic listings (from {n_raw} raw CSV matches)."
        )
        return "\n".join(lines)

    lines = [
        f"Yes — I found {n} realistic matching properties for {scope}"
        + (f" (cleaned from {n_raw} raw rows)" if n_raw != n else "")
        + ".",
        "",
    ]
    if len(priced) > 0:
        lines.extend(
            [
                "Price range (price_in_lakh):",
                f"• Minimum: ₹{float(priced['price_in_lakh'].min()):,.2f} lakh",
                f"• Maximum: ₹{float(priced['price_in_lakh'].max()):,.2f} lakh",
                f"• Average: ₹{float(priced['price_in_lakh'].mean()):,.2f} lakh",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Price range: prices are not listed for these matching rows.",
                "",
            ]
        )

    lines.append("Top options (quality-ranked, not cheapest junk):")
    for i, (_, row) in enumerate(ranked.head(3).iterrows(), start=1):
        beds = row.get("bedroom")
        beds_s = f"{int(beds)} BHK" if pd.notna(beds) else "BHK n/a"
        addr = str(row.get("address") or "Noida")
        size = row.get("size")
        size_s = f"{int(size):,} sqft" if pd.notna(size) else "size n/a"
        price = row.get("price_in_lakh")
        price_s = (
            f"₹{float(price):,.2f} lakh" if pd.notna(price) else "price n/a"
        )
        transport = " · near transport belt" if _is_transport_address(addr) else ""
        amens = _row_confirmed_amenities(row)
        lines.append(
            f"{i}. {addr} · {beds_s} · {price_s} · {size_s} · amenities: {amens}{transport}"
        )
    if n > 3:
        lines.append("")
        lines.append(f"…and {n - 3} more matches in the cleaned dataset.")
    if prefer_transport:
        lines.append("")
        lines.append(
            "Note: Exact metro/transport distance isn’t in the CSV — "
            "results prefer better-connected sectors; confirm on a map."
        )
    return "\n".join(lines)


def find_most_expensive_property(
    sector: str,
    bedrooms: int,
    *,
    require_lift: bool = False,
    require_security: bool = False,
) -> str:
    """Backward-compatible wrapper around dynamic extreme sorting."""
    required = []
    if require_lift:
        required.append("amenities_lift")
    if require_security:
        required.append("amenities_24_x_7_security")
    intent = {
        "sector": sector,
        "bhk": bedrooms,
        "min_price": None,
        "max_price": None,
        "required_amenities": required,
        "forbidden_amenities": [],
        "status": None,
        "older_than": None,
        "min_age": None,
        "max_age": None,
        "require_semi": False,
        "require_ac": None,
        "require_gas": None,
        "extreme": "max",
        "wants_count": False,
        "scope": ", ".join(
            [sector, f"{bedrooms} BHK", *[f"{c}=True" for c in required]]
        ),
    }
    filtered = filter_properties(
        df,
        sector=sector,
        bhk=bedrooms,
        required_amenities=required,
    )
    return format_filtered_answer(intent, filtered)


def _try_deterministic_extreme_answer(user_prompt: str) -> str | None:
    """Route cheapest/highest questions through dynamic filter + sort."""
    intent = parse_listing_intent(user_prompt)
    if not intent or intent.get("extreme") is None:
        return None
    filtered = filter_properties(
        df,
        sector=intent["sector"],
        bhk=intent["bhk"],
        min_price=intent["min_price"],
        max_price=intent["max_price"],
        required_amenities=intent["required_amenities"],
        forbidden_amenities=intent["forbidden_amenities"],
        status=intent["status"],
        min_age=intent["min_age"],
        max_age=intent["max_age"],
        older_than=intent["older_than"],
        require_semi=intent["require_semi"],
        require_ac=intent["require_ac"],
        require_gas=intent["require_gas"],
    )
    return format_filtered_answer(intent, filtered)


def _is_analytical_question(q: str) -> bool:
    """True when the user wants analysis/comparison, not a plain filter dump."""
    return bool(
        re.search(
            r"\b(compare|comparison|versus|\bvs\.?\b|invest(?:ment|ing)?|"
            r"worth|better|which\s+is|older\s+vs|newer\s+vs|old\s+vs|"
            r"difference\s+between|analyse|analyze|analysis)\b",
            q,
        )
    )


def compare_older_vs_newer(
    sector: str,
    age_cutoff_years: float = 5.0,
) -> str:
    """Deterministic older-vs-newer investment-style summary for a sector."""
    if "age_of_property" not in df.columns:
        return "age_of_property is not available in this dataset."

    base = df[
        df["address"].astype(str).str.contains(sector, case=False, na=False)
    ].copy()
    if base.empty:
        return f"No properties found for {sector} in the provided data."

    base["age_num"] = pd.to_numeric(base["age_of_property"], errors="coerce")
    base["price_num"] = pd.to_numeric(base["price_in_lakh"], errors="coerce")
    base["rate_num"] = pd.to_numeric(base["rate"], errors="coerce")
    aged = base.dropna(subset=["age_num"])
    if aged.empty:
        return (
            f"Found {len(base)} listings in {sector}, but none have "
            "age_of_property listed, so older vs newer cannot be compared."
        )

    older = aged[aged["age_num"] > age_cutoff_years]
    newer = aged[aged["age_num"] <= age_cutoff_years]

    def bucket_stats(label: str, part: pd.DataFrame) -> list[str]:
        priced = part.dropna(subset=["price_num"])
        rated = part.dropna(subset=["rate_num"])
        lines = [f"{label} (age {'>' if 'Older' in label else '≤'} {age_cutoff_years:g} years):"]
        lines.append(f"• Listings with age listed: {len(part)}")
        if len(priced) == 0:
            lines.append("• Price: not listed")
        else:
            lines.append(
                f"• Avg price: ₹{float(priced['price_num'].mean()):,.2f} lakh "
                f"(min ₹{float(priced['price_num'].min()):,.2f} · "
                f"max ₹{float(priced['price_num'].max()):,.2f})"
            )
        if len(rated) > 0:
            lines.append(
                f"• Avg rate: ₹{float(rated['rate_num'].mean()):,.0f}/sqft"
            )
        else:
            lines.append("• Avg rate: not listed")
        return lines

    older_avg = (
        float(older["price_num"].dropna().mean())
        if len(older.dropna(subset=["price_num"]))
        else None
    )
    newer_avg = (
        float(newer["price_num"].dropna().mean())
        if len(newer.dropna(subset=["price_num"]))
        else None
    )

    lines = [
        f"Older vs newer comparison for {sector} "
        f"(cutoff: {age_cutoff_years:g} years).",
        "",
        *bucket_stats("Older flats", older),
        "",
        *bucket_stats("Newer flats", newer),
        "",
    ]

    if older_avg is not None and newer_avg is not None:
        diff = newer_avg - older_avg
        if abs(diff) < 1e-6:
            takeaway = (
                "Takeaway: average total prices are about the same for older "
                "and newer flats in this sector (from price_in_lakh)."
            )
        elif diff > 0:
            takeaway = (
                f"Takeaway: newer flats average about ₹{diff:,.2f} lakh more "
                "in total price than older flats (price_in_lakh). That can mean "
                "a higher entry cost; compare avg ₹/sqft above for value density."
            )
        else:
            takeaway = (
                f"Takeaway: older flats average about ₹{abs(diff):,.2f} lakh more "
                "in total price than newer flats (price_in_lakh). Check avg ₹/sqft "
                "above before judging investment value."
            )
        lines.append(takeaway)
    else:
        lines.append(
            "Takeaway: not enough priced rows in both age groups to compare averages."
        )

    lines.append(
        "Note: this is a data summary from listings, not financial advice."
    )
    return "\n".join(lines)


def _try_older_vs_newer_answer(user_prompt: str) -> str | None:
    q = user_prompt.lower().strip()
    sector_m = re.search(r"sector\s*([0-9]+[a-z]?)", q)
    if not sector_m:
        return None
    wants_compare = bool(
        re.search(
            r"\b(compare|versus|\bvs\.?\b|older\s+vs|newer\s+vs|old\s+vs|"
            r"older.*newer|newer.*older|invest(?:ment|ing)?)\b",
            q,
        )
    ) and bool(re.search(r"\b(old|older|new|newer|age)\b", q))
    if not wants_compare:
        return None

    cutoff_m = re.search(
        r"\b(?:older\s+than|over|above|more\s+than)\s+(\d+)\s*(?:year|yr)s?\b",
        q,
    )
    cutoff = float(cutoff_m.group(1)) if cutoff_m else 5.0
    return compare_older_vs_newer(
        sector=f"sector {sector_m.group(1)}",
        age_cutoff_years=cutoff,
    )


def _try_fast_pandas_answer(user_prompt: str) -> str | None:
    """
    Dynamic intent → filter_properties → exact Pandas answer.
    Handles arbitrary amenities, price filters, and listing summaries.
    """
    q = user_prompt.lower().strip()

    if re.search(r"\bnpm\b|\buvicorn\b|langpanda:service|ollama serve", q):
        return (
            "That looks like a terminal command, not a property question. "
            "Ask something like: “2 BHK in sector 75 with AC and gas, semi-furnished, price?” "
            "Keep LangPanda running with: npm run langpanda:service"
        )

    # Leave compare / investment questions to dedicated helpers or the LLM.
    if _is_analytical_question(q):
        return None

    intent = parse_listing_intent(user_prompt)
    if not intent:
        return None
    # Extremes are handled by _try_deterministic_extreme_answer first.
    if intent.get("extreme") is not None:
        return None

    filtered = filter_properties(
        df,
        sector=intent["sector"],
        bhk=intent["bhk"],
        min_price=intent["min_price"],
        max_price=intent["max_price"],
        required_amenities=intent["required_amenities"],
        forbidden_amenities=intent["forbidden_amenities"],
        status=intent["status"],
        min_age=intent["min_age"],
        max_age=intent["max_age"],
        older_than=intent["older_than"],
        require_semi=intent["require_semi"],
        require_ac=intent["require_ac"],
        require_gas=intent["require_gas"],
    )
    return format_filtered_answer(intent, filtered)


if __name__ == "__main__":
    test_query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else (
            "Find a 2 BHK apartment in Sector 75 that has an AC, a gas connection, "
            "and is semi-furnished, and state its price."
        )
    )
    print(f"Loaded df shape={df.shape} columns={list(df.columns)[:12]}...")
    print(f"Ollama model={_ollama_model_name()} base={_ollama_base_url()}")
    print(f"Query: {test_query}\n")
    print(query_real_estate_agent(test_query))
