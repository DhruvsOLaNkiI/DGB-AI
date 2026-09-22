"""Free web search fallback via DuckDuckGo (ddgs) — no API key required."""

from __future__ import annotations

import re


def free_web_search(query: str, max_results: int = 3) -> str:
    """
    Search the public web for Noida property context.
    Uses the ddgs package (DuckDuckGo). 100% free — no API key.
    Rewrites broad investment / “best sector” asks into market-analysis queries.
    """
    q = (query or "").strip()
    if not q:
        return "No external web results found."

    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # older package name
        except ImportError:
            return (
                "Web search fallback unavailable: install `ddgs` in langpanda/.venv "
                "(`pip install ddgs`)."
            )

    search_query = rewrite_web_search_query(q)
    try:
        results: list[dict] = []
        # ddgs API variants: context manager and/or .text()
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(search_query, max_results=max_results))
        except TypeError:
            ddgs = DDGS()
            results = list(ddgs.text(search_query, max_results=max_results))

        if not results:
            return "No external web results found."

        snippets: list[str] = []
        for i, row in enumerate(results[:max_results], start=1):
            title = (row.get("title") or row.get("source") or f"Result {i}").strip()
            body = (row.get("body") or row.get("snippet") or "").strip()
            link = (row.get("href") or row.get("link") or row.get("url") or "").strip()
            snippets.append(
                f"[{i}] Source: {title}\nDetails: {body}\nLink: {link}"
            )
        return "\n\n".join(snippets)
    except Exception as exc:  # noqa: BLE001
        return f"Web search fallback failed: {str(exc)}"


def is_investment_advice_query(question: str) -> bool:
    """True for broad / subjective investment & “best sector to buy” questions.

    Concrete listing filters (BHK, under X Cr, amenities) are NOT investment advice —
    those must hit Pandas inventory, not the corridor briefing template.
    """
    q = (question or "").lower().strip()
    if not q:
        return False
    # Inventory-shaped asks like "best 3BHK under 1 crore with amenities"
    if re.search(r"\b\d+\s*(?:bhk|rk)\b", q):
        return False
    if re.search(
        r"\b(?:under|below|less\s+than|upto|up\s+to|between|above|over)\s+"
        r"\d+(?:\.\d+)?\s*(?:cr|crore|lakh|lakhs?)?\b",
        q,
    ) and re.search(r"\b(cr|crore|lakh|price|budget|cost)\b", q):
        return False
    if re.search(r"\b(cheapest|most\s+expensive|lowest|highest)\b", q):
        return False
    return bool(
        re.search(
            r"\b("
            r"invest(ment|ing)?|"
            r"roi|rental\s+yield|capital\s+appreciation|"
            r"best\s+(sector|area|place|localit\w*|corridor)s?\s+to\s+(buy|invest)|"
            r"top\s+(sector|area|place|localit\w*)s?\s+to\s+(buy|invest)|"
            r"where\s+(should|to)\s+(i\s+)?(buy|invest)|"
            r"which\s+(sector|area)\s+(is\s+)?best|"
            r"best\s+(sector|area|place|localit\w*|investment)\b|"
            r"propert\w*\s+for\s+invest|"
            r"best\s+propert\w*\s+for\s+invest"
            r")\b",
            q,
        )
    )


def rewrite_web_search_query(question: str) -> str:
    """
    Rewrite the DuckDuckGo query so investment / sector-profile asks hit
    market-analysis pages instead of generic “X properties for sale” portals.
    """
    q = (question or "").strip()
    if is_investment_advice_query(q):
        return (
            "top sectors best ROI rental yield capital appreciation "
            "property investment Noida Jewar airport Yamuna Expressway "
            "metro corridor market analysis 2024 2025"
        )
    # General sector / property cost → cover plots, villas, flats, commercial.
    if re.search(
        r"\b("
        r"sector\s*\d+[a-z]?|"
        r"propert(y|ies)\s+(cost|price|rate)|"
        r"(cost|price|rate)\s+of\s+propert|"
        r"property\s+in\s+|"
        r"real\s+estate\s+(in|of)|"
        r"plot|villa|kothi|land|commercial"
        r")\b",
        q.lower(),
    ):
        return (
            f"{q} Noida sector property rates plots land villas houses "
            "apartments flats commercial shops price per sq ft sq yard "
            "market overview"
        )
    return f"{q} Noida real estate property market"


def is_external_market_query(question: str) -> bool:
    """True when the question needs web/market context beyond clean_dataset.csv."""
    q = question.lower().strip()
    if is_investment_advice_query(q):
        return True
    return bool(
        re.search(
            r"\b("
            r"circle\s*rates?|stamp\s*duty|rera(\s+verif\w*)?|"
            r"metro(\s+station)?|near\s+(the\s+)?metro|"
            r"builder\s+reviews?|market\s+trends?|news|"
            r"completion\s+date|possession\s+timeline|"
            r"suggest\s+(me\s+)?(some\s+)?(propert|flat|apartment)"
            r")\b",
            q,
        )
    )


# Clear off-domain topics we refuse (do NOT match "car parking").
_OFF_TOPIC_RE = re.compile(
    r"\b("
    r"cars?|bikes?|scooters?|motorcycles?|automobiles?|"
    r"phones?|mobiles?|laptops?|tablets?|headphones?|"
    r"restaurants?|food|pizza|movies?|cricket|football|"
    r"stocks?|crypto|bitcoin|mutual\s+funds?"
    r")\b",
    re.I,
)

# Signals that the question is about Noida property / listings / market.
_REAL_ESTATE_RE = re.compile(
    r"\b("
    r"propert\w*|flat|apartment|bhk|bedroom|sector|noida|"
    r"plot|villa|penthouse|builder|rera|circle\s*rate|"
    r"lakh|crore|sq\.?\s*ft|carpet|amenit\w*|gym|pool|"
    r"ready\s+to\s+move|under\s+construction|"
    r"invest(ment)?|resale|rental|possession|metro|"
    r"price|budget|cheapest|expensive|listing"
    r")\b",
    re.I,
)

_OUT_OF_SCOPE_REPLY = (
    "I’m DBG-AI, a Noida real-estate assistant. I can help with flats, "
    "sectors, BHK, prices, amenities, and property market questions — "
    "not cars or other products. Try asking about a sector, BHK, or budget."
)


def is_real_estate_scope(question: str) -> bool:
    """
    True when the question is about property / listings / Noida housing market.
    False for clear off-topic asks (cars, phones, food, etc.).
    """
    q = (question or "").strip().lower()
    if not q:
        return False
    # "car parking" is real-estate amenity — strip that before off-topic check.
    q_check = re.sub(r"\bcar\s*parking\b", "parking", q)
    if _OFF_TOPIC_RE.search(q_check) and not _REAL_ESTATE_RE.search(q_check):
        return False
    if _OFF_TOPIC_RE.search(q_check) and _REAL_ESTATE_RE.search(q_check):
        # Mixed: "best car near my flat" → still refuse if primary noun is off-topic
        # Prefer refuse when an off-topic product noun is the main ask.
        if re.search(
            r"\b(which|best|buy|recommend)\b.{0,40}\b(cars?|bikes?|phones?)\b",
            q_check,
        ):
            return False
    if _REAL_ESTATE_RE.search(q):
        return True
    # No real-estate signal and no off-topic → allow through (inventory intent
    # parser / agent may still handle sector-only style asks).
    return not bool(_OFF_TOPIC_RE.search(q_check))


def out_of_scope_reply() -> str:
    return _OUT_OF_SCOPE_REPLY


# Attributes users ask about that are NOT columns in clean_dataset.csv.
# These cannot be filtered with Pandas, so we answer them via web + LLM.
UNSUPPORTED_ATTRIBUTE_PATTERNS: list[tuple[str, str]] = [
    (r"\bpet[-\s]?friendly\b|\bpets?\s+allowed\b|\bpet\b", "pet-friendly policy"),
    (r"\bgated\s*communit\w*\b|\bgated\b", "gated community"),
    # Soft location asks — never skip Pandas when BHK/price/etc. exist.
    (r"\bdistance\s+to\b|\bhow\s+far\b|\bnearby\b|\bnear\s+by\b", "nearby distance"),
    (
        r"\bnear(?:\s+by)?\s+(?:to\s+)?(?:the\s+)?(?:metro|trans\w*|station|highway|airport|mall|school)\b"
        r"|\b(?:metro|transport)\s+(?:access|connectivity|nearby)\b",
        "nearby transport / landmark",
    ),
    (r"\b(home\s*)?loan\b|\bemi\b|\bmortgage\b|\bfinanc(e|ing)\b", "loan / EMI"),
    (r"\bcorner\s+(flat|unit|plot|property)\b", "corner unit"),
    (r"\b(park|river|lake|garden|pool)\s+view\b", "view / outlook"),
    (r"\bsolar\b|\bev\s+charg\w*\b|\belectric\s+vehicle\b", "solar / EV charging"),
    (r"\bwheelchair\b|\bdisabled\s+access\b|\baccessib\w*\b", "accessibility"),
    (r"\bfeng\s*shui\b", "feng shui"),
    (r"\bcrime\b|\bsafety\s+rating\b", "area safety rating"),
    (r"\bschool\s+district\b|\bschool\s+ranking\b", "school district"),
    (r"\bappreciation\b|\bresale\s+value\b|\brental\s+yield\b", "investment metrics"),
    (r"\bfire\s+safety\b|\bearthquake\s+resistan\w*\b", "safety certification"),
]

# Soft attributes: mention as a note, but do NOT skip Pandas when BHK/price/etc. exist.
_SOFT_UNSUPPORTED_LABELS = {
    "nearby distance",
    "nearby transport / landmark",
    "connectivity / commute",
    "metro connectivity",
}


def references_unsupported_attribute(question: str) -> str | None:
    """
    Return a human label when the query hinges on an attribute that is NOT a
    column in clean_dataset.csv (so Pandas cannot answer it). Else None.
    """
    q = question.lower().strip()
    for pattern, label in UNSUPPORTED_ATTRIBUTE_PATTERNS:
        if re.search(pattern, q):
            return label
    return None


def is_soft_unsupported_attribute(label: str | None) -> bool:
    """Soft attrs (nearby/transport) should not override BHK/price inventory filters."""
    return bool(label) and label in _SOFT_UNSUPPORTED_LABELS


def is_empty_pandas_reply(reply: str | None) -> bool:
    """Heuristic: Pandas path found no usable inventory answer."""
    if not reply or not str(reply).strip():
        return True
    text = str(reply).strip().lower()
    empty_markers = (
        "0 matching properties found",
        "no properties match",
        "no matching properties",
        "could not parse that into exact filters",
        "agent stopped due to iteration limit",
        "execution error:",
        "none list price_in_lakh",
    )
    return any(m in text for m in empty_markers)


def is_web_search_usable(web_data: str | None) -> bool:
    """False when DuckDuckGo returned nothing useful (or failed)."""
    if not web_data or not str(web_data).strip():
        return False
    text = str(web_data).strip().lower()
    fail_markers = (
        "no external web results found",
        "web search fallback unavailable",
        "web search fallback failed",
        "install `ddgs`",
    )
    if any(m in text for m in fail_markers):
        return False
    # Need at least one real snippet body, not just titles.
    return "details:" in text or "http" in text or len(text) > 80
