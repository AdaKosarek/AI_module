"""Cold-start simulace"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    compute_daily_turnover,
)
from src.labeling import assign_storage_class, compute_dynamic_thresholds
from src.models import (
    build_model_pipeline,
    evaluate_model,
    get_param_grid,
    split_data,
    train_with_gridsearch,
)

logger = logging.getLogger(__name__)

CLASS_ORDER = [
    "shelf_picking",
    "front_zone_bin",
    "special_zone",
    "floor_block",
    "pallet_rack",
]

NUMERIC_FEATURES_NO_TURNOVER = [f for f in NUMERIC_FEATURES if f != "daily_turnover"]


def _build_pipeline_no_turnover(model_type: str) -> Pipeline:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES_NO_TURNOVER),
            (
                "categorical",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    if model_type == "dt":
        classifier = DecisionTreeClassifier(class_weight="balanced", random_state=42)
    elif model_type == "rf":
        classifier = RandomForestClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1
        )
    elif model_type == "xgb":
        classifier = XGBClassifier(
            objective="multi:softprob",
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
        )
    else:
        raise ValueError(f"Neznamy model_type: {model_type}")

    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def _train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_types: Optional[list[str]] = None,
) -> dict:
    if model_types is None:
        model_types = ["dt", "rf", "xgb"]

    le = LabelEncoder()
    le.fit(sorted(set(y_train.unique()) | set(CLASS_ORDER)))

    trained: dict = {"le": le}

    for mt in model_types:
        logger.info("=== Trenuji model: %s ===", mt)

        if mt == "xgb":
            yt = pd.Series(le.transform(y_train), index=y_train.index)
        else:
            yt = y_train

        pipeline = build_model_pipeline(mt)
        param_grid = get_param_grid(mt)
        grid, train_time = train_with_gridsearch(
            pipeline, param_grid, X_train, yt, model_type=mt
        )
        trained[mt] = grid.best_estimator_
        logger.info("Model %s natrenovan za %.1f s", mt, train_time)

    return trained


def _evaluate_variant(
    trained_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    le: LabelEncoder,
    variant_name: str,
) -> list[dict]:
    results = []

    for mt in ["dt", "rf", "xgb"]:
        model = trained_models[mt]

        if mt == "xgb":
            y_test_enc = pd.Series(le.transform(y_test), index=y_test.index)
            eval_result = evaluate_model(model, X_test, y_test_enc)
            f1_per_class = {
                le.inverse_transform([k])[0]: v
                for k, v in eval_result["f1_per_class"].items()
            }
        else:
            eval_result = evaluate_model(model, X_test, y_test)
            f1_per_class = eval_result["f1_per_class"]

        row = {
            "variant": variant_name,
            "model": mt,
            "accuracy": round(eval_result["accuracy"], 4),
            "f1_macro": round(eval_result["f1_macro"], 4),
        }
        for cls in CLASS_ORDER:
            row[f"f1_{cls}"] = round(f1_per_class.get(cls, 0.0), 4)

        results.append(row)

    return results


# daily_turnover=0
def run_variant_a(
    trained_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    le: LabelEncoder,
) -> list[dict]:
    X_test_mod = X_test.copy()
    X_test_mod["daily_turnover"] = 0.0
    return _evaluate_variant(trained_models, X_test_mod, y_test, le, "variant_a_zero")


# daily_turnover = median trenovacich dat
def run_variant_b(
    trained_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    le: LabelEncoder,
    train_median: float,
) -> list[dict]:
    X_test_mod = X_test.copy()
    X_test_mod["daily_turnover"] = train_median
    return _evaluate_variant(
        trained_models, X_test_mod, y_test, le, "variant_b_median"
    )


# cely model natrenovany BEZ daily_turnover
def run_variant_c(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> list[dict]:
    X_train_no = X_train.drop(columns=["daily_turnover"]).copy()
    X_test_no = X_test.drop(columns=["daily_turnover"]).copy()

    le_c = LabelEncoder()
    le_c.fit(sorted(set(y_train.unique()) | set(CLASS_ORDER)))

    results = []

    for mt in ["dt", "rf", "xgb"]:
        logger.info("=== Varianta C — trenuji model: %s (bez turnover) ===", mt)

        if mt == "xgb":
            yt_train = pd.Series(le_c.transform(y_train), index=y_train.index)
            yt_test = pd.Series(le_c.transform(y_test), index=y_test.index)
        else:
            yt_train = y_train
            yt_test = y_test

        pipeline = _build_pipeline_no_turnover(mt)
        param_grid = get_param_grid(mt)
        grid, train_time = train_with_gridsearch(
            pipeline, param_grid, X_train_no, yt_train, model_type=mt
        )
        best_est = grid.best_estimator_

        eval_result = evaluate_model(best_est, X_test_no, yt_test)

        if mt == "xgb":
            f1_per_class = {
                le_c.inverse_transform([k])[0]: v
                for k, v in eval_result["f1_per_class"].items()
            }
        else:
            f1_per_class = eval_result["f1_per_class"]

        row = {
            "variant": "variant_c_no_feature",
            "model": mt,
            "accuracy": round(eval_result["accuracy"], 4),
            "f1_macro": round(eval_result["f1_macro"], 4),
        }
        for cls in CLASS_ORDER:
            row[f"f1_{cls}"] = round(f1_per_class.get(cls, 0.0), 4)

        results.append(row)
        logger.info(
            "Varianta C [%s]: accuracy=%.4f, f1_macro=%.4f (train=%.1fs)",
            mt, eval_result["accuracy"], eval_result["f1_macro"], train_time,
        )

    return results


def run_rules_no_turnover(
    y_test: pd.Series,
    input_path: str,
) -> dict:
    df_full = pd.read_csv(input_path)
    df_full = compute_daily_turnover(df_full)

    resolved = compute_dynamic_thresholds(df_full)

    df_test = df_full.loc[y_test.index].copy()
    df_test["order_count"] = 0
    df_test["daily_turnover"] = 0.0

    y_pred = df_test.apply(assign_storage_class, axis=1, resolved=resolved)

    acc = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro")
    f1_per = f1_score(y_test, y_pred, average=None, labels=CLASS_ORDER)
    f1_dict = dict(zip(CLASS_ORDER, f1_per))

    logger.info(
        "Rules (no turnover): accuracy=%.4f, f1_macro=%.4f", acc, f1_mac
    )

    row: dict = {
        "variant": "rules_no_turnover",
        "model": "rules",
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_mac, 4),
    }
    for cls in CLASS_ORDER:
        row[f"f1_{cls}"] = round(f1_dict.get(cls, 0.0), 4)

    return row


##
def plot_cold_start_comparison(
    results_df: pd.DataFrame,
    output_path: str,
) -> None:
    model_colors = {
        "dt": "#3274A1",
        "rf": "#3A923A",
        "xgb": "#E1812C",
        "rules": "#888888",
    }

    variants = results_df["variant"].unique()
    models = results_df["model"].unique()
    n_variants = len(variants)
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.8 / n_models
    x = np.arange(n_variants)

    for i, model in enumerate(models):
        mask = results_df["model"] == model
        vals = []
        for v in variants:
            sub = results_df[(results_df["variant"] == v) & (results_df["model"] == model)]
            vals.append(sub["f1_macro"].values[0] if len(sub) > 0 else 0.0)
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, vals, bar_width,
            label=model.upper(),
            color=model_colors.get(model, "#666666"),
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7,
            )

    ax.set_xlabel("Varianta cold-start")
    ax.set_ylabel("F1 Macro")
    ax.set_title("Cold-Start: Porovnani variant a modelu (F1 Macro)")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Ulozen graf: %s", output_path)


def plot_cold_start_per_class_f1(
    results_df: pd.DataFrame,
    class_order: list[str],
    output_path: str,
) -> None:
    f1_cols = [f"f1_{cls}" for cls in class_order]
    row_labels = []
    matrix = []

    for _, row in results_df.iterrows():
        row_labels.append(f"{row['variant']} | {row['model']}")
        matrix.append([row.get(c, 0.0) for c in f1_cols])

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(12, max(6, len(row_labels) * 0.45)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, shrink=0.8, label="F1 Score")

    ax.set_xticks(np.arange(len(class_order)))
    ax.set_xticklabels(class_order, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val < 0.4 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    fzb_idx = class_order.index("front_zone_bin")
    ax.axvline(x=fzb_idx - 0.5, color="red", linewidth=2, linestyle="--")
    ax.axvline(x=fzb_idx + 0.5, color="red", linewidth=2, linestyle="--")

    ax.set_title("Cold-Start: F1 Score per trida a varianta")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Ulozen graf: %s", output_path)


def plot_cold_start_front_zone(
    results_df: pd.DataFrame,
    output_path: str,
) -> None:
    model_colors = {
        "dt": "#3274A1",
        "rf": "#3A923A",
        "xgb": "#E1812C",
        "rules": "#888888",
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    labels = []
    vals = []
    colors = []

    for _, row in results_df.iterrows():
        label = f"{row['variant']}\n{row['model'].upper()}"
        labels.append(label)
        vals.append(row.get("f1_front_zone_bin", 0.0))
        colors.append(model_colors.get(row["model"], "#666666"))

    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    ax.set_ylabel("F1 Score")
    ax.set_title("Cold-Start: front_zone_bin F1 Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="F1 = 0.50")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Ulozen graf: %s", output_path)


# split -> train -> varianty A/B/C + pravidla -> CSV + grafy
def run_cold_start_experiment(
    input_path: str = "data/processed/products_labeled.csv",
    results_dir: str = "results/phase8_cold_start",
) -> dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Cold-Start Experiment: rozdeleni dat ===")
    X_train, X_test, y_train, y_test = split_data(input_path)

    logger.info("=== Cold-Start Experiment: trenovani modelu ===")
    trained = _train_models(X_train, y_train)
    le = trained["le"]

    logger.info("=== Cold-Start Experiment: reference (plne features) ===")
    ref_results = _evaluate_variant(trained, X_test, y_test, le, "reference_full")

    logger.info("=== Cold-Start Experiment: varianta A (turnover=0) ===")
    var_a_results = run_variant_a(trained, X_test, y_test, le)

    train_median = float(X_train["daily_turnover"].median())
    logger.info(
        "=== Cold-Start Experiment: varianta B (turnover=median=%.4f) ===",
        train_median,
    )
    var_b_results = run_variant_b(trained, X_test, y_test, le, train_median)

    logger.info("=== Cold-Start Experiment: varianta C (bez turnover feature) ===")
    var_c_results = run_variant_c(X_train, X_test, y_train, y_test)

    logger.info("=== Cold-Start Experiment: pravidlovy baseline ===")
    rules_result = run_rules_no_turnover(y_test, input_path)

    all_rows = ref_results + var_a_results + var_b_results + var_c_results
    all_rows.append(rules_result)
    results_df = pd.DataFrame(all_rows)

    phase5_path = Path("results/phase5_modeling/model_comparison_table.csv")
    if phase5_path.exists():
        phase5_df = pd.read_csv(phase5_path)
        logger.info("Nactena Phase 5 tabulka z %s", phase5_path)
        phase5_df.to_csv(
            results_dir / "phase5_reference.csv", index=False
        )

    results_df.to_csv(results_dir / "cold_start_results.csv", index=False)
    logger.info(
        "Ulozeno cold_start_results.csv (%d radku)", len(results_df)
    )

    logger.info("=== Cold-Start Experiment: generovani grafu ===")

    plot_cold_start_comparison(
        results_df,
        str(results_dir / "cold_start_comparison.png"),
    )

    plot_cold_start_per_class_f1(
        results_df,
        CLASS_ORDER,
        str(results_dir / "cold_start_per_class_f1.png"),
    )

    plot_cold_start_front_zone(
        results_df,
        str(results_dir / "cold_start_front_zone.png"),
    )

    best_ml = results_df[results_df["model"] != "rules"]
    best_row = best_ml.loc[best_ml["f1_macro"].idxmax()]
    rules_f1 = rules_result["f1_macro"]
    rules_fzb = rules_result.get("f1_front_zone_bin", 0.0)

    summary = {
        "n_variants": len(results_df["variant"].unique()),
        "n_rows": len(results_df),
        "best_ml_variant": best_row["variant"],
        "best_ml_model": best_row["model"],
        "best_ml_f1_macro": best_row["f1_macro"],
        "rules_f1_macro": rules_f1,
        "rules_f1_front_zone_bin": rules_fzb,
        "ml_advantage_f1": round(best_row["f1_macro"] - rules_f1, 4),
        "train_median_turnover": train_median,
        "results_dir": str(results_dir),
    }

    logger.info("Cold-Start Experiment HOTOV")
    logger.info("Nejlepsi ML: %s/%s (F1=%.4f)", summary["best_ml_variant"],
                summary["best_ml_model"], summary["best_ml_f1_macro"])
    logger.info("Pravidla: F1=%.4f, front_zone_bin F1=%.4f",
                rules_f1, rules_fzb)
    logger.info("Vyhoda ML nad pravidly: +%.4f F1 macro", summary["ml_advantage_f1"])

    return summary
