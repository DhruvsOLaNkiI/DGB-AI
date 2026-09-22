"""Load and normalize the Noida listings DataFrame for the Pandas agent."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

# Prefer local copy under langpanda/, else project data/
CANDIDATE_CSVS = [
    ROOT / "clean_dataset.csv",
    REPO / "data" / "clean_dataset.csv",
]

AMENITY_PREFIX = "amenities_"
NUMERIC_COLS = [
    "price_in_lakh",
    "size",
    "carpet_area",
    "rate",
    "bedroom",
    "bathrooms",
]


def resolve_csv_path(csv_path: str | Path | None = None) -> Path:
    if csv_path:
        path = Path(csv_path)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"CSV not found: {path}")

    for candidate in CANDIDATE_CSVS:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "No clean_dataset.csv found. Place it at langpanda/clean_dataset.csv "
        "or data/clean_dataset.csv."
    )


def _to_bool_series(series: pd.Series) -> pd.Series:
    """True only for true/1/yes — False and missing stay non-True."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def load_listings_df(csv_path: str | Path | None = None) -> pd.DataFrame:
    """
    Load the clean structured dataset and normalize types so amenity /
    numeric filters behave predictably under generated Pandas code.
    """
    path = resolve_csv_path(csv_path)
    df = pd.read_csv(path)

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in df.columns:
        name = str(col)
        if name.startswith(AMENITY_PREFIX) or name.startswith("furnish_detail_"):
            df[col] = _to_bool_series(df[col])

    if "address" in df.columns:
        df["address"] = df["address"].astype(str)

    if "age_of_property" in df.columns:
        df["age_of_property"] = pd.to_numeric(df["age_of_property"], errors="coerce")

    return df
