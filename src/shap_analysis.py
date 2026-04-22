"""SHAP analyza pro interpretaci XGBoost modelu skladovych trid."""

import logging
from pathlib import Path
from typing import Optional
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.models import split_data, _get_feature_names

logger = logging.getLogger(__name__)



def compute_shap_values(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    cache_path: Optional[Path] = None,
) -> dict:
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            logger.info("Nacitam SHAP cache z %s", cache_path)
            return joblib.load(cache_path)

    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X_test)
    feature_names = _get_feature_names(pipeline)

    logger.info("Vytvarim TreeExplainer a pocitam SHAP hodnoty (%d vzorku)...", len(X_test))
    explainer = shap.TreeExplainer(classifier)
    shap_values_raw = explainer.shap_values(X_transformed)

    # Odlisny format mezi verzemi SHAP
    if isinstance(shap_values_raw, list):
        shap_values = np.stack(shap_values_raw, axis=-1)
    elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
        shap_values = shap_values_raw
    else:
        raise ValueError(f"Neocekavany format SHAP hodnot: type={type(shap_values_raw)}")

    logger.info("SHAP hodnoty shape: %s", shap_values.shape)

    result = {
        "shap_values": shap_values,
        "X_transformed": X_transformed if isinstance(X_transformed, np.ndarray) else X_transformed.toarray(),
        "feature_names": feature_names,
        "explainer": explainer,
        "expected_value": explainer.expected_value,
    }

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result, cache_path)
        logger.info("SHAP cache ulozena do %s", cache_path)

    return result


def plot_feature_importance_bar(
    pipeline: Pipeline,
    feature_names: list,
    output_path: str,
    top_n: int = 15,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    importances = pipeline.named_steps["classifier"].feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(df)), df["importance"].values, color="steelblue")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (Gain)")
    ax.set_title("XGBoost Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance graf ulozen: %s", output_path)


##
def plot_shap_summary_bar(shap_dict: dict, output_path: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shap_values = shap_dict["shap_values"]
    feature_names = shap_dict["feature_names"]
    mean_abs = np.mean(np.abs(shap_values), axis=(0, 2))

    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    df = df.sort_values("mean_abs_shap", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(df)), df["mean_abs_shap"].values, color="coral")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("SHAP Feature Importance (mean |SHAP value|)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("SHAP summary bar graf ulozen: %s", output_path)


def plot_shap_summary_bar_grouped(shap_dict: dict, output_path: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shap_values = shap_dict["shap_values"]  # (n_samples, n_features, n_classes)
    feature_names = shap_dict["feature_names"]

    cat_indices = [i for i, name in enumerate(feature_names) if name.startswith("category_group_")]
    num_indices = [i for i, name in enumerate(feature_names) if not name.startswith("category_group_")]
    num_names = [feature_names[i] for i in num_indices]

    num_importance = np.mean(np.abs(shap_values[:, num_indices, :]), axis=(0, 2))
    cat_shap = shap_values[:, cat_indices, :]
    cat_total = np.mean(np.sum(np.abs(cat_shap), axis=1), axis=(0, 1))

    names = num_names + ["Category (total)"]
    values = list(num_importance) + [cat_total]

    df = pd.DataFrame({"feature": names, "mean_abs_shap": values})
    df = df.sort_values("mean_abs_shap", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(df)), df["mean_abs_shap"].values, color="teal")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Průměrná hodnota SHAP")
    ax.set_title("SHAP (kategorie seskupeny)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("SHAP grouped bar graf ulozen: %s", output_path)


def plot_shap_summary_beeswarm(
    shap_dict: dict,
    class_idx: int,
    class_name: str,
    output_path: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shap_values = shap_dict["shap_values"]
    X_transformed = shap_dict["X_transformed"]
    feature_names = shap_dict["feature_names"]

    shap.summary_plot(
        shap_values[:, :, class_idx],
        X_transformed,
        feature_names=feature_names,
        show=False,
    )
    plt.title(f"SHAP Beeswarm - {class_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("SHAP beeswarm pro tridu '%s' ulozen: %s", class_name, output_path)

def plot_shap_dependence(
    shap_dict: dict,
    feature_name: str,
    class_idx: int,
    class_name: str,
    output_path: str,
    X_raw=None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shap_values = shap_dict["shap_values"]
    X_transformed = shap_dict["X_transformed"]
    feature_names = shap_dict["feature_names"]

    feature_idx = feature_names.index(feature_name)
    shap_col = shap_values[:, feature_idx, class_idx]

    if X_raw is not None and feature_name in X_raw.columns:
        x_values = X_raw[feature_name].values
        x_label = feature_name
    else:
        x_values = X_transformed[:, feature_idx]
        x_label = f"{feature_name} (skalovano)"

    # Obarveni podle category_group_electronics odhaluje interakci kategorie s hlavni feature
    electronics_idx = None
    for i, name in enumerate(feature_names):
        if name == "category_group_electronics":
            electronics_idx = i
            break

    fig, ax = plt.subplots(figsize=(9, 6))
    if electronics_idx is not None:
        colors = X_transformed[:, electronics_idx]
        sc = ax.scatter(x_values, shap_col, c=colors, cmap="coolwarm", alpha=0.6, s=12)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("category_group_electronics")
    else:
        ax.scatter(x_values, shap_col, alpha=0.6, s=12, color="steelblue")

    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"Hodnota SHAP pro {feature_name}")
    ax.set_title(f"SHAP závislost: {feature_name} (třída {class_name})")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("SHAP dependence plot '%s' (%s) ulozen: %s", feature_name, class_name, output_path)


def plot_shap_force_example(
    shap_dict: dict,
    sample_idx: int,
    class_idx: int,
    class_name: str,
    output_path: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shap_values = shap_dict["shap_values"]
    X_transformed = shap_dict["X_transformed"]
    feature_names = shap_dict["feature_names"]
    expected_value = shap_dict["expected_value"]

    shap.force_plot(
        expected_value[class_idx],
        shap_values[sample_idx, :, class_idx],
        X_transformed[sample_idx, :],
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("SHAP force plot (vzorek %d, trida '%s') ulozen: %s", sample_idx, class_name, output_path)

def select_representative_samples(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label_encoder: LabelEncoder,
) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    y_test_encoded = label_encoder.transform(y_test)

    representatives = {}
    for cls_idx, cls_name in enumerate(label_encoder.classes_):
        mask = (y_test_encoded == cls_idx) & (y_pred == cls_idx)
        indices = np.where(mask)[0]

        if len(indices) == 0:
            logger.warning("Zadny spravne klasifikovany vzorek pro tridu '%s'", cls_name)
            continue

        best_local = np.argmax(y_proba[indices, cls_idx])
        representatives[cls_name] = int(indices[best_local])

    logger.info("Reprezentativni vzorky: %s", representatives)
    return representatives


# SHAP vypocet + vsechny grafy
def run_shap_analysis(
    input_path: str = "data/processed/products_labeled.csv",
    models_dir: str = "models",
    results_dir: str = "results/phase7_shap",
) -> dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(models_dir)

    logger.info("=== SHAP analyza ===")
    X_train, X_test, y_train, y_test = split_data(input_path)

    pipeline_path = models_dir / "best_model.joblib"
    logger.info("Nacitam model z %s", pipeline_path)
    pipeline: Pipeline = joblib.load(pipeline_path)

    le = LabelEncoder()
    le.fit(y_train)
    class_names = list(le.classes_)
    logger.info("Tridy (LE poradi): %s", class_names)

    feature_names = _get_feature_names(pipeline)
    logger.info("Pocet features: %d", len(feature_names))

    cache_path = models_dir / "shap_cache.joblib"
    shap_dict = compute_shap_values(pipeline, X_test, cache_path=cache_path)

    plot_feature_importance_bar(
        pipeline, feature_names,
        output_path=results_dir / "feature_importance_gain.png",
    )

    plot_shap_summary_bar(shap_dict, output_path=results_dir / "shap_summary_bar.png")

    plot_shap_summary_bar_grouped(shap_dict, output_path=results_dir / "shap_summary_bar_grouped.png")

    for cls_idx, cls_name in enumerate(class_names):
        plot_shap_summary_beeswarm(
            shap_dict, cls_idx, cls_name,
            output_path=results_dir / f"shap_beeswarm_{cls_name}.png",
        )

    representatives = select_representative_samples(pipeline, X_test, y_test, le)

    for cls_name, sample_idx in representatives.items():
        cls_idx = list(le.classes_).index(cls_name)
        plot_shap_force_example(
            shap_dict, sample_idx, cls_idx, cls_name,
            output_path=results_dir / f"shap_force_{cls_name}.png",
        )

    pallet_rack_idx = class_names.index("pallet_rack")
    shelf_picking_idx = class_names.index("shelf_picking")

    plot_shap_dependence(
        shap_dict, "product_weight_g", pallet_rack_idx, "pallet_rack",
        output_path=results_dir / "shap_dependence_weight_pallet_rack.png",
        X_raw=X_test,
    )
    plot_shap_dependence(
        shap_dict, "volume_cm3", shelf_picking_idx, "shelf_picking",
        output_path=results_dir / "shap_dependence_volume_shelf_picking.png",
        X_raw=X_test,
    )

    summary = {
        "n_samples": shap_dict["shap_values"].shape[0],
        "n_features": shap_dict["shap_values"].shape[1],
        "n_classes": shap_dict["shap_values"].shape[2],
        "class_names": class_names,
        "representatives": representatives,
        "results_dir": str(results_dir),
    }
    logger.info("SHAP analyza dokoncena. Vysledky v %s", results_dir)
    logger.info("Souhrn: %s", summary)
    return summary
