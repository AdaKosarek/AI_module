"""Čištění dat"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PHYSICAL_COLS: list[str] = [
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
]

LEAD_DENSITY_GCM3: float = 11.34

CATEGORY_GROUP_MAP: dict[str, str] = {
    "furniture_decor": "furniture",
    "office_furniture": "furniture",
    "furniture_living_room": "furniture",
    "furniture_bedroom": "furniture",
    "furniture_mattress_and_upholstery": "furniture",
    "kitchen_dining_laundry_garden_furniture": "furniture",
    "computers_accessories": "electronics",
    "computers": "electronics",
    "consoles_games": "electronics",
    "electronics": "electronics",
    "telephony": "electronics",
    "fixed_telephony": "electronics",
    "tablets_printing_image": "electronics",
    "pc_gamer": "electronics",
    "audio": "electronics",
    "cine_photo": "electronics",
    "signaling_and_security": "electronics",
    "security_and_services": "electronics",
    "home_appliances": "home_appliances",
    "home_appliances_2": "home_appliances",
    "small_appliances": "home_appliances",
    "small_appliances_home_oven_and_coffee": "home_appliances",
    "portable_kitchen_food_processors": "home_appliances",
    "air_conditioning": "home_appliances",
    "la_cuisine": "home_appliances",
    "health_beauty": "beauty_health",
    "perfumery": "beauty_health",
    "diapers_and_hygiene": "beauty_health",
    "sports_leisure": "sports_leisure",
    "fashion_sport": "sports_leisure",
    "fashion_bags_accessories": "fashion",
    "fashion_shoes": "fashion",
    "fashion_male_clothing": "fashion",
    "fashion_underwear_beach": "fashion",
    "fashio_female_clothing": "fashion",
    "fashion_childrens_clothes": "fashion",
    "watches_gifts": "fashion",
    "cool_stuff": "fashion",
    "luggage_accessories": "fashion",
    "toys": "toys_baby",
    "baby": "toys_baby",
    "party_supplies": "toys_baby",
    "christmas_supplies": "toys_baby",
    "books_general_interest": "books_media",
    "books_technical": "books_media",
    "books_imported": "books_media",
    "dvds_blu_ray": "books_media",
    "cds_dvds_musicals": "books_media",
    "music": "books_media",
    "musical_instruments": "books_media",
    "bed_bath_table": "home_garden",
    "garden_tools": "home_garden",
    "home_confort": "home_garden",
    "home_comfort_2": "home_garden",
    "home_construction": "home_garden",
    "construction_tools_construction": "home_garden",
    "construction_tools_safety": "home_garden",
    "construction_tools_lights": "home_garden",
    "costruction_tools_garden": "home_garden",
    "costruction_tools_tools": "home_garden",
    "flowers": "home_garden",
    "food_drink": "food_drinks",
    "food": "food_drinks",
    "drinks": "food_drinks",
    "housewares": "housewares",
    "pet_shop": "housewares",
    "auto": "housewares",
    "market_place": "housewares",
    "stationery": "stationery",
    "art": "stationery",
    "arts_and_craftmanship": "stationery",
}


def remove_invalid_products(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    mask_all_nan = df[PHYSICAL_COLS].isna().all(axis=1)
    n_removed = int(mask_all_nan.sum())
    df_out = df[~mask_all_nan].copy()
    logger.info("remove_invalid_products: odstraněno %d produktů (NaN ve všech fyzických sloupcích)", n_removed)
    return df_out, n_removed


def impute_missing_physical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    df = df.copy()
    stats: dict[str, int] = {}
    cat_col = "product_category_name_english"

    dim_cols = ["product_length_cm", "product_height_cm", "product_width_cm"]
    zero_weight_mask = (df["product_weight_g"] == 0) & (df[dim_cols].gt(0).any(axis=1))
    n_zero_weight = int(zero_weight_mask.sum())
    df.loc[zero_weight_mask, "product_weight_g"] = np.nan
    stats["weight_zero_fixed"] = n_zero_weight
    if n_zero_weight > 0:
        logger.info("impute_missing_physical: %d produktů s weight=0 a nenulovými rozměry -> NaN", n_zero_weight)

    cat_medians = df.groupby(cat_col)[PHYSICAL_COLS].median()
    global_medians = df[PHYSICAL_COLS].median()

    for col in PHYSICAL_COLS:
        n_missing_before = int(df[col].isna().sum())
        if n_missing_before == 0:
            stats[col] = 0
            continue

        df[col] = df[col].fillna(df[cat_col].map(cat_medians[col]))
        df[col] = df[col].fillna(global_medians[col])

        n_imputed = n_missing_before - int(df[col].isna().sum())
        stats[col] = n_imputed
        logger.info("impute_missing_physical: %s , doplněno %d/%d", col, n_imputed, n_missing_before)

    return df, stats


# Opravi produkty s hustotou vyssi nez olovo (> 11.34 g/cm3). Prepsani hmotnosti medianem kategorie.
def fix_density_anomalies(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    cat_col = "product_category_name_english"

    volume = df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
    density = df["product_weight_g"] / volume.replace(0, np.nan)

    anomaly_mask = density > LEAD_DENSITY_GCM3
    n_fixed = int(anomaly_mask.sum())

    if n_fixed > 0:
        cat_weight_medians = df.groupby(cat_col)["product_weight_g"].median()
        global_weight_median = df["product_weight_g"].median()

        new_weights = df.loc[anomaly_mask, cat_col].map(cat_weight_medians)
        new_weights = new_weights.fillna(global_weight_median)
        df.loc[anomaly_mask, "product_weight_g"] = new_weights

    logger.info("fix_density_anomalies: opraveno %d produktů", n_fixed)
    return df, n_fixed


def fill_missing_turnover(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["order_count", "total_quantity"]:
        n_filled = int(df[col].isna().sum())
        df[col] = df[col].fillna(0)
        if n_filled > 0:
            logger.info("fill_missing_turnover: %s — doplněno %d NaN → 0", col, n_filled)
    return df

def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["volume_cm3"] = (
        df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
    )

    # 0 -> NaN aby deleni vratilo NaN misto ZeroDivisionError/inf
    safe_volume = df["volume_cm3"].replace(0, np.nan)
    safe_weight_kg = (df["product_weight_g"] / 1000).replace(0, np.nan)

    df["volumetric_density"] = df["product_weight_g"] / safe_volume
    df["price_per_kg"] = df["avg_price"] / safe_weight_kg

    logger.info("compute_derived_features: přidány volume_cm3, volumetric_density, price_per_kg")
    return df


def analyze_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    total_outliers = 0

    for col in PHYSICAL_COLS:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_col = f"outlier_{col}"
        df[outlier_col] = (df[col] < lower) | (df[col] > upper)
        n_out = int(df[outlier_col].sum())
        total_outliers += n_out

    logger.info("analyze_outliers: celkem %d outlier flagů", total_outliers)
    return df

def assign_category_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["category_group"] = df["product_category_name_english"].map(CATEGORY_GROUP_MAP).fillna("other")

    group_counts = df["category_group"].value_counts()
    logger.info("assign_category_groups: rozložení skupin:\n%s", group_counts.to_string())
    return df


# load -> clean -> save
def clean_products(
    input_path: str | Path = "data/processed/products_clean.csv",
    output_path: str | Path = "data/processed/products_processed.csv",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    stats: dict[str, Any] = {}

    logger.info("=== Fáze 2: Čištění dat ===")
    logger.info("Vstup: %s", input_path)

    df = pd.read_csv(input_path)
    stats["input_rows"] = len(df)
    logger.info("Načteno %d produktů", len(df))

    df, n_removed = remove_invalid_products(df)
    stats["removed_invalid"] = n_removed

    df, impute_stats = impute_missing_physical(df)
    stats["imputed"] = impute_stats

    df, n_density_fixed = fix_density_anomalies(df)
    stats["density_fixed"] = n_density_fixed

    df = fill_missing_turnover(df)
    df = compute_derived_features(df)
    df = analyze_outliers(df)
    df = assign_category_groups(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    stats["output_rows"] = len(df)
    stats["retention_pct"] = round(100 * len(df) / stats["input_rows"], 2)

    logger.info("Výstup: %s (%d produktů, zachováno %.1f%%)", output_path, len(df), stats["retention_pct"])

    return df, stats
