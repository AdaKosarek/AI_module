"""Načítání, čištění a propojování tabulek z Olist datasetu."""

from pathlib import Path
import logging

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

MANUAL_CATEGORY_TRANSLATIONS = {
    "pc_gamer": "pc_gamer",
    "portateis_cozinha_e_preparadores_de_alimentos": "portable_kitchen_food_processors",
}

_FILE_MAP = {
    "products": "olist_products_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


def load_raw_tables(data_dir: Path = RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name, filename in _FILE_MAP.items():
        df = pd.read_csv(data_dir / filename)
        logger.info("Loaded %-25s  shape=%s", name, df.shape)
        tables[name] = df
    return tables


def translate_categories(
    products: pd.DataFrame,
    translation: pd.DataFrame,
) -> pd.DataFrame:
    result = products.merge(
        translation, on="product_category_name", how="left",
    )

    mask = result["product_category_name_english"].isna()
    result.loc[mask, "product_category_name_english"] = (
        result.loc[mask, "product_category_name"].map(MANUAL_CATEGORY_TRANSLATIONS)
    )

    remaining_na = result["product_category_name_english"].isna().sum()
    logger.info("Categories still missing after manual map: %d → filling with 'unknown'", remaining_na)
    result["product_category_name_english"] = result["product_category_name_english"].fillna("unknown")

    result = result.drop(columns=["product_category_name"])

    return result


def filter_delivered_orders(orders: pd.DataFrame) -> pd.DataFrame:
    before = len(orders)
    delivered = orders[orders["order_status"] == "delivered"].copy()
    logger.info("Filtered delivered orders: %d → %d", before, len(delivered))
    return delivered


# Inner join orders + order_items + products
def merge_orders_products(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    result = orders.merge(order_items, on="order_id", how="inner")
    logger.info("After merge with order_items: %d rows", len(result))

    result = result.merge(products, on="product_id", how="inner")
    logger.info("After merge with products: %d rows", len(result))

    return result


def compute_turnover_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    turnover = merged.groupby("product_id").agg(
        order_count=("order_id", "nunique"),
        total_quantity=("order_item_id", "count"),
        avg_price=("price", "mean"),
        avg_freight=("freight_value", "mean"),
        first_order_date=("order_purchase_timestamp", "min"),
        last_order_date=("order_purchase_timestamp", "max"),
    ).reset_index()
    logger.info("Turnover metrics computed for %d products", len(turnover))
    return turnover


def compute_avg_review_score(
    reviews: pd.DataFrame,
    order_items: pd.DataFrame,
) -> pd.DataFrame:
    merged = reviews.merge(order_items[["order_id", "product_id"]], on="order_id", how="inner")
    avg_scores = (
        merged.groupby("product_id")["review_score"]
        .mean()
        .reset_index()
        .rename(columns={"review_score": "avg_review_score"})
    )
    logger.info("Avg review score computed for %d products", len(avg_scores))
    return avg_scores


# load -> translate -> filter -> merge -> turnover + review -> save products_clean.csv
def build_products_clean(data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    tables = load_raw_tables(data_dir)
    products = translate_categories(tables["products"], tables["category_translation"])
    orders = filter_delivered_orders(tables["orders"])
    merged = merge_orders_products(
        orders=orders,
        order_items=tables["order_items"],
        products=products,
    )
    turnover = compute_turnover_metrics(merged)
    review_scores = compute_avg_review_score(tables["reviews"], tables["order_items"])

    result = products.merge(turnover, on="product_id", how="left")
    result = result.merge(review_scores, on="product_id", how="left")

    logger.info(
        "Final dataset: %d products, %d with turnover, %d with reviews",
        len(result),
        result["order_count"].notna().sum(),
        result["avg_review_score"].notna().sum(),
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / "products_clean.csv"
    result.to_csv(out_path, index=False)
    logger.info("Saved to %s  shape=%s", out_path, result.shape)

    return result
