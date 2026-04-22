"""Pravidla pro prirazeni skladovych trid."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


THRESHOLDS = {
    "special_categories": ["electronics"],
    "special_original_categories": ["watches_gifts"],
    "special_other_price_q": 0.75,
    "special_other_max_weight_g": 5000,
    "front_bin_min_order_count_q": 0.75,
    "front_bin_max_weight_g": 10000,
    "front_bin_max_volume_cm3": 40000,
    "pallet_min_weight_g": 8000,
    "pallet_min_volume_cm3": 80000,
    "block_min_volume_cm3": 30000,
    "block_max_weight_g": 8000,
    "block_bulky_categories": ["furniture", "home_garden", "home_appliances"],
    "block_bulky_min_volume_cm3": 25000,
    "block_max_dimension_q": 0.95,
}


def compute_dynamic_thresholds(
    df: pd.DataFrame,
    thresholds: dict | None = None,
) -> dict:
    if thresholds is None:
        thresholds = THRESHOLDS

    resolved = dict(thresholds)

    resolved["resolved_front_bin_min_order_count"] = df["order_count"].quantile(
        thresholds["front_bin_min_order_count_q"]
    )
    resolved["resolved_special_other_price"] = df["avg_price"].quantile(
        thresholds["special_other_price_q"]
    )
    resolved["resolved_block_max_dimension"] = df["product_length_cm"].quantile(
        thresholds["block_max_dimension_q"]
    )

    logger.info(
        "Dynamicke prahy: front_bin_min_order_count=%.1f, "
        "special_other_price=%.1f, block_max_dimension=%.1f",
        resolved["resolved_front_bin_min_order_count"],
        resolved["resolved_special_other_price"],
        resolved["resolved_block_max_dimension"],
    )

    return resolved


# Priradi skladovou tridu jednomu produktu podle poradi pravidel.
def assign_storage_class(row: pd.Series, resolved: dict) -> str:
    category_group = row.get("category_group", "")
    original_cat = row.get("product_category_name_english", "")
    weight = row.get("product_weight_g", 0)
    volume = row.get("volume_cm3", 0)
    order_count = row.get("order_count", 0)
    avg_price = row.get("avg_price", 0)
    length = row.get("product_length_cm", 0)

    # specialni zona > prihradka > paleta > blok > police
    if category_group in resolved["special_categories"]:
        return "special_zone"
    if original_cat in resolved["special_original_categories"]:
        return "special_zone"
    if (
        category_group == "other"
        and avg_price > resolved["resolved_special_other_price"]
        and weight < resolved["special_other_max_weight_g"]
    ):
        return "special_zone"

    if (
        order_count >= resolved["resolved_front_bin_min_order_count"]
        and weight <= resolved["front_bin_max_weight_g"]
        and volume <= resolved["front_bin_max_volume_cm3"]
    ):
        return "front_zone_bin"

    if weight > resolved["pallet_min_weight_g"] or volume > resolved["pallet_min_volume_cm3"]:
        return "pallet_rack"

    if volume > resolved["block_min_volume_cm3"] and weight <= resolved["block_max_weight_g"]:
        return "floor_block"
    if (
        category_group in resolved["block_bulky_categories"]
        and volume > resolved["block_bulky_min_volume_cm3"]
    ):
        return "floor_block"
    if length > resolved["resolved_block_max_dimension"]:
        return "floor_block"

    return "shelf_picking"


def label_products(
    df: pd.DataFrame,
    thresholds: dict | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = THRESHOLDS

    resolved = compute_dynamic_thresholds(df, thresholds)
    df = df.copy()
    df["storage_class"] = df.apply(assign_storage_class, axis=1, resolved=resolved)

    dist = df["storage_class"].value_counts()
    logger.info("Rozlozeni skladovych trid:\n%s", dist.to_string())

    return df


# Nacteni -> olabeluj -> uloz
def label_products_pipeline(
    input_path: str | Path = "data/processed/products_processed.csv",
    output_path: str | Path = "data/processed/products_labeled.csv",
) -> tuple[pd.DataFrame, dict]:
    input_path = Path(input_path)
    output_path = Path(output_path)

    logger.info("Nacitam %s", input_path)
    df = pd.read_csv(input_path)

    df = label_products(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Ulozeno do %s (%d radku)", output_path, len(df))

    dist = df["storage_class"].value_counts()
    total = len(df)
    stats: dict = {}
    for cls_name, count in dist.items():
        stats[f"{cls_name}_count"] = int(count)
        stats[f"{cls_name}_pct"] = round(count / total * 100, 1)

    return df, stats
