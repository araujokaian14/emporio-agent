"""Loads the store CSVs and turns them into clean, query-ready tables.

Design notes (also documented in the README):
- Raw files in data/ are never modified; all cleaning happens in memory at startup.
- Files are located by suffix (e.g. "*products.csv") so the original names with spaces still work.
- Only promotions with is_active == 1 are applied. If a product had more than one active
  promotion, the highest discount wins (does not happen in the current data, but is handled).
- The 5% PIX discount is NOT applied on top of promotional prices (policy section 6.2).
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

from .config import DATA_DIR

PIX_DISCOUNT = 0.05


def money(value: float, factor: float = 1.0) -> float:
    """value × factor rounded to cents, half-up, in decimal arithmetic.
    Plain float rounding turns 459.90 × 0.95 = 436.905 into 436.90 instead of 436.91."""
    exact = Decimal(str(value)) * Decimal(str(factor))
    return float(exact.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def normalize(text: object) -> str:
    """Lowercase, strip accents and punctuation: 'Violões Elétricos!' -> 'violoes eletricos'."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return text.strip()


def compact(text: object) -> str:
    """Normalized text without spaces, so 'GD-20' and 'GD20' compare equal."""
    return normalize(text).replace(" ", "")


def _find_file(data_dir: Path, suffix: str) -> Path:
    matches = sorted(data_dir.glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"No file ending with '{suffix}' found in {data_dir}")
    return matches[0]


def _parse_specs(raw: object) -> dict:
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


@dataclass
class StoreData:
    products: pd.DataFrame
    categories: pd.DataFrame
    promotions: pd.DataFrame
    orders: pd.DataFrame
    order_items: pd.DataFrame
    customers: pd.DataFrame


def _clean_products(products: pd.DataFrame, categories: pd.DataFrame, promotions: pd.DataFrame) -> pd.DataFrame:
    df = products.copy()
    df["name"] = df["name"].astype(str).str.strip()
    df["price_brl"] = pd.to_numeric(df["price_brl"], errors="coerce")
    df["stock_quantity"] = pd.to_numeric(df["stock_quantity"], errors="coerce").fillna(0).astype(int)
    df["status"] = df["status"].astype(str).str.strip().str.lower()
    df["specs"] = df["specs"].apply(_parse_specs)
    df["category_name"] = df["category_id"].map(categories.set_index("category_id")["name"])

    active = promotions[promotions["is_active"] == 1].sort_values("discount_percent", ascending=False)
    active = active.drop_duplicates("product_id")[["product_id", "discount_percent", "description"]]
    active = active.rename(columns={"description": "promotion_name"})
    df = df.merge(active, on="product_id", how="left")

    has_promo = df["discount_percent"].notna()
    df["promo_price_brl"] = df.apply(
        lambda r: money(r["price_brl"], 1 - r["discount_percent"] / 100) if pd.notna(r["discount_percent"]) else None,
        axis=1).astype(float)
    df["final_price_brl"] = df["promo_price_brl"].fillna(df["price_brl"])
    df["pix_price_brl"] = df["price_brl"].apply(lambda v: money(v, 1 - PIX_DISCOUNT)).where(~has_promo)

    df["is_available"] = (df["status"] == "active") & (df["stock_quantity"] > 0)
    df["search_text"] = (
        df["name"] + " " + df["category_name"].fillna("") + " " + df["description"].fillna("")
        + " " + df["specs"].apply(lambda s: " ".join(map(str, s.values())))
    ).apply(normalize)
    df["name_norm"] = df["name"].apply(normalize)
    df["name_compact"] = df["name"].apply(compact)
    return df.set_index("product_id", drop=False).sort_index()


def _clean_orders(orders: pd.DataFrame) -> pd.DataFrame:
    df = orders.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce").dt.date
    df["estimated_delivery"] = pd.to_datetime(df["estimated_delivery"], errors="coerce").dt.date
    df["status"] = df["status"].astype(str).str.strip().str.lower()
    for col in ("tracking_code", "notes", "payment_method"):
        df[col] = df[col].astype(object).where(df[col].notna(), None)
    return df.set_index("order_id", drop=False)


def _clean_customers(customers: pd.DataFrame) -> pd.DataFrame:
    df = customers.copy()
    df["email_norm"] = df["email"].astype(str).str.strip().str.lower()
    df["phone_digits"] = df["phone"].astype(str).str.replace(r"\D", "", regex=True)
    return df.set_index("customer_id", drop=False)


def load_store_data(data_dir: Path | str = DATA_DIR) -> StoreData:
    data_dir = Path(data_dir)
    read = lambda suffix: pd.read_csv(_find_file(data_dir, suffix))  # noqa: E731

    categories = read("categories.csv")
    promotions = read("promotions.csv")
    products = _clean_products(read("products.csv"), categories, promotions)

    return StoreData(
        products=products,
        categories=categories,
        promotions=promotions,
        orders=_clean_orders(read("orders.csv")),
        order_items=read("order_items.csv"),
        customers=_clean_customers(read("customers.csv")),
    )
