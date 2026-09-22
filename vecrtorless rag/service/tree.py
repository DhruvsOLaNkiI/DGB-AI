"""Hierarchical tree index over Noida listings (PageIndex-style, no vectors)."""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SECTOR_RE = re.compile(r"sector\s*([0-9]+[a-z]?)", re.I)


@dataclass
class Listing:
    id: str
    address: str
    sector: str
    bedroom: int | None
    price_in_lakh: float | None
    status: str
    type2: str
    text: str


def extract_sector(address: str) -> str:
    match = SECTOR_RE.search(address or "")
    return f"sector {match.group(1).lower()}" if match else "other"


def _num(value: str) -> float | None:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _int(value: str) -> int | None:
    n = _num(value)
    return int(n) if n is not None else None


def listing_text(row: dict[str, str], listing_id: str, sector: str) -> str:
    bedroom = _int(row.get("bedroom", ""))
    price = _num(row.get("price_in_lakh", ""))
    bhk = f"{bedroom} BHK" if bedroom is not None else row.get("type1", "Property")
    price_s = f"{price:.2f} lakh" if price is not None else "price not listed"
    bits = [
        f"Property #{listing_id.replace('listing-', '')}",
        f"Location: {row.get('address') or sector}",
        f"{bhk} {row.get('type2', '')}".strip(),
        f"Price: {price_s}",
        f"Status: {row.get('status', '')}".strip(": "),
        f"Furnishing: {row.get('status.1', '')}".strip(": "),
    ]
    return "\n".join(b for b in bits if b and not b.endswith(": "))


def load_listings(csv_path: Path) -> list[Listing]:
    listings: list[Listing] = []
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            address = (row.get("address") or "").strip()
            sector = extract_sector(address)
            lid = f"listing-{i}"
            listings.append(
                Listing(
                    id=lid,
                    address=address,
                    sector=sector,
                    bedroom=_int(row.get("bedroom", "")),
                    price_in_lakh=_num(row.get("price_in_lakh", "")),
                    status=(row.get("status") or "").strip(),
                    type2=(row.get("type2") or "").strip(),
                    text=listing_text(row, lid, sector),
                )
            )
    return listings


def build_outline(listings: list[Listing]) -> str:
    buckets: dict[tuple[str, int | None], dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "min": None, "max": None}
    )
    for item in listings:
        key = (item.sector, item.bedroom)
        bucket = buckets[key]
        bucket["count"] += 1
        if item.price_in_lakh is not None:
            bucket["min"] = (
                item.price_in_lakh
                if bucket["min"] is None
                else min(bucket["min"], item.price_in_lakh)
            )
            bucket["max"] = (
                item.price_in_lakh
                if bucket["max"] is None
                else max(bucket["max"], item.price_in_lakh)
            )

    by_sector: dict[str, list[tuple[int | None, dict[str, Any]]]] = defaultdict(list)
    for (sector, bedroom), stats in buckets.items():
        by_sector[sector].append((bedroom, stats))

    lines: list[str] = []
    for sector in sorted(by_sector, key=lambda s: (s == "other", s)):
        children = sorted(by_sector[sector], key=lambda x: (x[0] is None, x[0] or 0))
        total = sum(stats["count"] for _, stats in children)
        lines.append(f"# {sector} ({total} listings)")
        for bedroom, stats in children:
            label = f"{bedroom} BHK" if bedroom is not None else "Other"
            price = ""
            if stats["min"] is not None and stats["max"] is not None:
                price = f" · {stats['min']:.1f}–{stats['max']:.1f} lakh"
            lines.append(f"  - {label}: {stats['count']} listings{price}")
    return "\n".join(lines)
