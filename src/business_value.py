"""Doplněk: Výpočet business hodnoty (úspora docházkové vzdálenosti)."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.features import compute_daily_turnover
from src.labeling import (
    assign_storage_class,
    compute_dynamic_thresholds,
)
from src.models import split_data

logger = logging.getLogger(__name__)

# Topologický model skladu, vzdálenosti jednotlivých zón od balicího pultu
DISTANCES: dict[str, int] = {
    "front_zone_bin": 15,
    "shelf_picking": 45,
    "special_zone": 45,
    "floor_block": 80,
    "pallet_rack": 80,
}

CLASS_NAMES_CZ: dict[str, str] = {
    "shelf_picking": "Police",
    "front_zone_bin": "Přední zóna",
    "special_zone": "Speciální zóna",
    "floor_block": "Bloková zóna",
    "pallet_rack": "Paleta",
}

# Krok skladníka
STEP_LENGTH_M = 0.75
# Předpokládaný počet vychystávek denně
DAILY_PICKS = 1000


def zone_to_distance(zone: str) -> int:
    if zone not in DISTANCES:
        raise KeyError(f"Neznámá skladová třída: {zone!r}")
    return DISTANCES[zone]


def predict_with_model(
    X_test_no_turnover: pd.DataFrame,
    model_path: str | Path,
    le_path: str | Path,
) -> pd.Series:
    model_path = Path(model_path)
    le_path = Path(le_path)

    logger.info("Nacitam model: %s", model_path)
    model = joblib.load(model_path)
    logger.info("Nacitam label encoder: %s", le_path)
    le: LabelEncoder = joblib.load(le_path)

    y_pred_enc = model.predict(X_test_no_turnover)
    y_pred_labels = le.inverse_transform(y_pred_enc)

    return pd.Series(y_pred_labels, index=X_test_no_turnover.index, name="ml_pred")


def predict_with_rules_no_turnover(
    df_test_with_meta: pd.DataFrame,
    full_df: pd.DataFrame,
) -> pd.Series:
    resolved = compute_dynamic_thresholds(full_df)

    df_cs = df_test_with_meta.copy()
    df_cs["order_count"] = 0
    df_cs["daily_turnover"] = 0.0

    logger.info("Aplikuji rules na %d testovacich produktu (cold-start)", len(df_cs))
    preds = df_cs.apply(assign_storage_class, axis=1, resolved=resolved)
    return pd.Series(preds.values, index=df_test_with_meta.index, name="rules_pred")


def compute_per_product_distances(
    y_true: pd.Series,
    y_rules: pd.Series,
    y_ml: pd.Series,
) -> pd.DataFrame:

    idx = y_true.index
    df = pd.DataFrame(
        {
            "product_idx": idx,
            "true_class": y_true.values,
            "rules_class": y_rules.reindex(idx).values,
            "ml_class": y_ml.reindex(idx).values,
        }
    )

    df["true_class_cz"] = df["true_class"].map(CLASS_NAMES_CZ)
    df["rules_class_cz"] = df["rules_class"].map(CLASS_NAMES_CZ)
    df["ml_class_cz"] = df["ml_class"].map(CLASS_NAMES_CZ)

    df["true_distance"] = df["true_class"].map(DISTANCES)
    df["rules_distance"] = df["rules_class"].map(DISTANCES)
    df["ml_distance"] = df["ml_class"].map(DISTANCES)

    df["savings_m"] = df["rules_distance"] - df["ml_distance"]

    df = df[
        [
            "product_idx",
            "true_class",
            "true_class_cz",
            "true_distance",
            "rules_class",
            "rules_class_cz",
            "rules_distance",
            "ml_class",
            "ml_class_cz",
            "ml_distance",
            "savings_m",
        ]
    ]
    return df

##
def select_case_study_sample(
    distances_df: pd.DataFrame,
    X_test: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    def _pick(mask: pd.Series, n: int, label: str) -> pd.DataFrame:
        candidates = distances_df[mask]
        if len(candidates) >= n:
            picked = candidates.sample(n=n, random_state=int(rng.integers(0, 1_000_000)))
        elif len(candidates) > 0:
            logger.warning(
                "Kategorie '%s': pozadovano %d kandidatu, dostupnych %d",
                label,
                n,
                len(candidates),
            )
            picked = candidates.copy()
        else:
            logger.warning(
                "Kategorie '%s': zadni kandidati neodpovidaji presne - fallback",
                label,
            )
            picked = pd.DataFrame(columns=distances_df.columns)
        picked = picked.copy()
        picked["category_label"] = label
        return picked

    parts: list[pd.DataFrame] = []

    mask_hit = (
        (distances_df["true_class"] == "front_zone_bin")
        & (distances_df["ml_class"] == "front_zone_bin")
        & (distances_df["rules_class"] == "shelf_picking")
    )
    parts.append(_pick(mask_hit, 2, "Budoucí hit"))

    mask_missed = (
        (distances_df["true_class"] == "front_zone_bin")
        & (distances_df["ml_class"] == "shelf_picking")
        & (distances_df["rules_class"] == "shelf_picking")
    )
    parts.append(_pick(mask_missed, 1, "Nepoznaný hit"))

    # 3) 1x Těžký produkt
    mask_heavy = (
        (distances_df["true_class"] == "pallet_rack")
        & (distances_df["ml_class"] == "pallet_rack")
        & (distances_df["rules_class"] == "pallet_rack")
    )
    parts.append(_pick(mask_heavy, 1, "Těžký produkt"))

    # 4) 1x Objemný
    mask_bulky = (
        (distances_df["true_class"] == "floor_block")
        & (distances_df["ml_class"] == "floor_block")
        & (distances_df["rules_class"] == "floor_block")
    )
    parts.append(_pick(mask_bulky, 1, "Objemný"))

    # 5) 1x Drahý
    mask_expensive = (
        (distances_df["true_class"] == "special_zone")
        & (distances_df["ml_class"] == "special_zone")
        & (distances_df["rules_class"] == "special_zone")
    )
    parts.append(_pick(mask_expensive, 1, "Drahý"))

    # 6) 1x Běžný produkt
    mask_normal = (
        (distances_df["true_class"] == "shelf_picking")
        & (distances_df["ml_class"] == "shelf_picking")
        & (distances_df["rules_class"] == "shelf_picking")
    )
    parts.append(_pick(mask_normal, 1, "Běžný produkt"))

    case_study = pd.concat(parts, ignore_index=False)

    # Doplneni kategorie produktu z X_test
    if "category_group" in X_test.columns:
        cat_map = X_test.loc[case_study["product_idx"].values, "category_group"]
        case_study["category_group"] = cat_map.values

    cols = [c for c in case_study.columns if c != "category_label"] + ["category_label"]
    case_study = case_study[cols]

    logger.info(
        "Case study sestaveno: %d radku (z pozadovanych 7)",
        len(case_study),
    )
    return case_study


def compute_global_savings(distances_df: pd.DataFrame) -> dict:
    front_mask = distances_df["true_class"] == "front_zone_bin"
    n_front = int(front_mask.sum())
    n_total = int(len(distances_df))

    if n_front == 0:
        logger.warning(
            "V test setu neni zadny produkt s true_class=front_zone_bin"
        )
        avg_rules_dist = float("nan")
        avg_ml_dist = float("nan")
        savings_m = float("nan")
        savings_pct = float("nan")
    else:
        avg_rules_dist = float(distances_df.loc[front_mask, "rules_distance"].mean())
        avg_ml_dist = float(distances_df.loc[front_mask, "ml_distance"].mean())
        savings_m = avg_rules_dist - avg_ml_dist
        savings_pct = (savings_m / avg_rules_dist * 100) if avg_rules_dist > 0 else 0.0

    total_avg_rules = float(distances_df["rules_distance"].mean())
    total_avg_ml = float(distances_df["ml_distance"].mean())
    if total_avg_rules > 0:
        total_savings_pct = (total_avg_rules - total_avg_ml) / total_avg_rules * 100
    else:
        total_savings_pct = 0.0

    # Prevod usporenych metru na pocet kroku skladnika
    if not np.isnan(savings_m):
        steps_before = round(avg_rules_dist / STEP_LENGTH_M * DAILY_PICKS)
        steps_after = round(avg_ml_dist / STEP_LENGTH_M * DAILY_PICKS)
        steps_saved = steps_before - steps_after
    else:
        steps_before = 0
        steps_after = 0
        steps_saved = 0

    summary = {
        "n_front": n_front,
        "n_total": n_total,
        "avg_rules_dist": avg_rules_dist,
        "avg_ml_dist": avg_ml_dist,
        "savings_m": savings_m,
        "savings_pct": savings_pct,
        "total_avg_rules": total_avg_rules,
        "total_avg_ml": total_avg_ml,
        "total_savings_pct": total_savings_pct,
        "steps_before": steps_before,
        "steps_after": steps_after,
        "steps_saved": steps_saved,
    }

    logger.info(
        "Global savings: front_zone_bin n=%d, rules=%.1fm, ml=%.1fm, savings=%.1f%% (%.1fm)",
        n_front,
        avg_rules_dist,
        avg_ml_dist,
        savings_pct,
        savings_m,
    )
    logger.info(
        "Global savings (cely test): rules=%.1fm, ml=%.1fm, savings=%.1f%%",
        total_avg_rules,
        total_avg_ml,
        total_savings_pct,
    )

    return summary


def run_business_value_analysis(
    input_path: str | Path = "data/processed/products_labeled.csv",
    models_dir: str | Path = "models",
    results_dir: str | Path = "results/phase_nadstavba_business_value",
) -> dict:

    input_path = Path(input_path)
    models_dir = Path(models_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = split_data(str(input_path))

    df_full = pd.read_csv(input_path)
    df_full = compute_daily_turnover(df_full)

    if "daily_turnover" in X_test.columns:
        X_test_no_turnover = X_test.drop(columns=["daily_turnover"])
    else:
        X_test_no_turnover = X_test.copy()

    y_ml = predict_with_model(
        X_test_no_turnover,
        models_dir / "best_model_no_turnover.joblib",
        models_dir / "label_encoder_no_turnover.joblib",
    )

    test_indices = y_test.index
    df_test_with_meta = df_full.loc[test_indices].copy()
    y_rules = predict_with_rules_no_turnover(df_test_with_meta, df_full)

    distances_df = compute_per_product_distances(y_test, y_rules, y_ml)

    case_study_df = select_case_study_sample(distances_df, X_test)

    global_savings = compute_global_savings(distances_df)

    distances_csv = results_dir / "per_product_distances.csv"
    case_study_csv = results_dir / "case_study_7_products.csv"
    summary_csv = results_dir / "global_savings_summary.csv"

    distances_df.to_csv(distances_csv, index=False)
    logger.info("Ulozeno: %s", distances_csv)

    case_study_df.to_csv(case_study_csv, index=False)
    logger.info("Ulozeno: %s", case_study_csv)

    pd.DataFrame([global_savings]).to_csv(summary_csv, index=False)
    logger.info("Ulozeno: %s", summary_csv)

    logger.info(
        "SOUHRN ==="
        "\n Test set: n=%d, z toho front_zone_bin: n=%d"
        "\n Pravidla: avg=%.1f m   |   ML: avg=%.1f m"
        "\n Uspora pro rychloobratkove: %.1f %% (%.1f m / vychystani)"
        "\n Globalne pres test set: uspora %.1f %%",
        global_savings["n_total"],
        global_savings["n_front"],
        global_savings["avg_rules_dist"],
        global_savings["avg_ml_dist"],
        global_savings["savings_pct"],
        global_savings["savings_m"],
        global_savings["total_savings_pct"],
    )

    summary = {
        **global_savings,
        "n_case_study": int(len(case_study_df)),
        "outputs": {
            "per_product_distances_csv": str(distances_csv),
            "case_study_csv": str(case_study_csv),
            "summary_csv": str(summary_csv),
        },
    }
    return summary
