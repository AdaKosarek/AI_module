"""Analyza zavaznosti chyb (vazene chybove skore)."""

import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

from src.models import split_data

logger = logging.getLogger(__name__)

CLASS_ORDER = [
    "shelf_picking",
    "front_zone_bin",
    "special_zone",
    "floor_block",
    "pallet_rack",
]

# Matice zavaznosti zameny
# Hodnoty 0-3 vyjadruji dopad chyby: 0=spravne, 1=tolerovatelne, 2=neefektivni, 3=problematicke.
SEVERITY_MATRIX = np.array(
    [
        [0, 1, 1, 2, 2],
        [1, 0, 2, 2, 3],
        [1, 2, 0, 2, 2],
        [2, 2, 2, 0, 1],
        [3, 3, 2, 1, 0],
    ]
)

DISPLAY_NAMES = ["Shelf", "Front Bin", "Special", "Floor Block", "Pallet"]


def get_severity_matrix() -> tuple[np.ndarray, list[str]]:
    return SEVERITY_MATRIX, CLASS_ORDER

# Nacte DT/RF/XGB modely a spocita confusion matrix pro kazdy. Stejny split jako trenovani.
def compute_confusion_matrices(
    input_path: str = "data/processed/products_labeled.csv",
    models_dir: str = "models",
) -> dict[str, dict]:
    models_dir = Path(models_dir)
    X_train, X_test, y_train, y_test = split_data(input_path)

    le = LabelEncoder()
    le.fit(y_train)

    model_types = ["dt", "rf", "xgb"]
    results = {}

    for mt in model_types:
        model_path = models_dir / f"{mt}_model.joblib"
        logger.info("Nacitam model: %s", model_path)
        model = joblib.load(model_path)

        if mt == "xgb":
            y_test_enc = le.transform(y_test)
            y_pred_enc = model.predict(X_test)
            y_pred = le.inverse_transform(y_pred_enc)
        else:
            y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
        results[mt] = {"cm": cm, "y_pred": y_pred}
        logger.info("[%s] Confusion matrix spoctena.", mt)

    return results


# Spocita vazene chybove metriky (TWE, WER, per-class) z confusion matrix a matice zavaznosti.
def compute_weighted_errors(cm: np.ndarray, severity: np.ndarray) -> dict:
    total_weighted_error = int(np.sum(severity * cm))
    total_predictions = int(np.sum(cm))
    weighted_error_rate = total_weighted_error / total_predictions if total_predictions > 0 else 0.0

    unweighted_error_count = int(np.sum(cm) - np.trace(cm))

    per_class_weighted_error = np.array(
        [np.sum(severity[i, :] * cm[i, :]) for i in range(len(CLASS_ORDER))]
    )

    if total_weighted_error > 0:
        per_class_contribution_pct = per_class_weighted_error / total_weighted_error * 100
    else:
        per_class_contribution_pct = np.zeros(len(CLASS_ORDER))

    return {
        "total_weighted_error": total_weighted_error,
        "total_predictions": total_predictions,
        "weighted_error_rate": round(weighted_error_rate, 4),
        "unweighted_error_count": unweighted_error_count,
        "per_class_weighted_error": per_class_weighted_error,
        "per_class_contribution_pct": per_class_contribution_pct,
    }


# Vrati DataFrame s TWE/WER a per-class TWE pro kazdy model.
def compute_all_model_scores(
    cm_dict: dict[str, dict], severity: np.ndarray
) -> pd.DataFrame:
    rows = []
    for mt, data in cm_dict.items():
        we = compute_weighted_errors(data["cm"], severity)
        row = {
            "model": mt,
            "total_weighted_error": we["total_weighted_error"],
            "weighted_error_rate": we["weighted_error_rate"],
            "unweighted_error_count": we["unweighted_error_count"],
        }
        for i, cls in enumerate(CLASS_ORDER):
            row[f"twe_{cls}"] = int(we["per_class_weighted_error"][i])
        rows.append(row)

    return pd.DataFrame(rows)


##
def plot_severity_matrix(
    severity: np.ndarray,
    class_order: list[str],
    output_path: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(severity, cmap="Reds", vmin=0, vmax=3)
    ax.figure.colorbar(im, ax=ax)

    n = len(class_order)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(DISPLAY_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(DISPLAY_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Misclassification Severity Matrix")

    for i in range(n):
        for j in range(n):
            if i == j:
                text = "OK"
            else:
                text = str(severity[i, j])
            color = "white" if severity[i, j] >= 2 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontweight="bold")

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Ulozeno: %s", output_path)


def plot_severity_confusion(
    cm: np.ndarray,
    severity: np.ndarray,
    class_order: list[str],
    model_name: str,
    output_path: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(class_order)
    weighted = cm * severity

    fig, ax = plt.subplots(figsize=(9, 7))

    masked_weighted = weighted.copy().astype(float)
    for i in range(n):
        masked_weighted[i, i] = np.nan
    off_diag_max = np.nanmax(masked_weighted) if np.nanmax(masked_weighted) > 0 else 1

    diag_vals = np.full((n, n), np.nan)
    for i in range(n):
        diag_vals[i, i] = cm[i, i]
    diag_max = np.nanmax(diag_vals) if np.nanmax(diag_vals) > 0 else 1

    ax.imshow(
        np.where(np.isnan(masked_weighted), 0, masked_weighted).astype(np.float64),
        cmap="Reds",
        vmin=0,
        vmax=off_diag_max,
        alpha=np.where(np.isnan(masked_weighted), 0.0, 1.0).astype(np.float64),
    )
    ax.imshow(
        np.where(np.isnan(diag_vals), 0, diag_vals).astype(np.float64),
        cmap="Blues",
        vmin=0,
        vmax=diag_max,
        alpha=np.where(np.isnan(diag_vals), 0.0, 0.6).astype(np.float64),
    )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(DISPLAY_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(DISPLAY_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Severity-Weighted Confusion Matrix - {model_name.upper()}")

    for i in range(n):
        for j in range(n):
            count = cm[i, j]
            if i == j:
                ax.text(j, i, str(count), ha="center", va="center", fontweight="bold", color="black")
            else:
                sev = severity[i, j]
                ax.text(
                    j, i,
                    f"{count}\n(sev: {sev})",
                    ha="center", va="center",
                    fontsize=9,
                    color="white" if weighted[i, j] > off_diag_max * 0.5 else "black",
                )

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Ulozeno: %s", output_path)


def plot_weighted_error_comparison(
    scores_df: pd.DataFrame,
    output_path: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = scores_df["model"].tolist()
    wer = scores_df["weighted_error_rate"].tolist()
    colors = {"dt": "#1f77b4", "rf": "#2ca02c", "xgb": "#d62728"}
    bar_colors = [colors.get(m, "#333333") for m in models]
    labels = {"dt": "DT", "rf": "RF", "xgb": "XGBoost"}
    bar_labels = [labels.get(m, m) for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(bar_labels, wer, color=bar_colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, wer):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_ylabel("Weighted Error Rate")
    ax.set_title("Weighted Error Rate by Model")
    ax.set_ylim(0, max(wer) * 1.2 if wer else 1)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Ulozeno: %s", output_path)


def plot_per_class_contribution(
    scores_df: pd.DataFrame,
    class_order: list[str],
    output_path: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = scores_df["model"].tolist()
    labels_map = {"dt": "DT", "rf": "RF", "xgb": "XGBoost"}
    model_labels = [labels_map.get(m, m) for m in models]

    class_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = range(len(models))

    left = np.zeros(len(models))
    for idx, cls in enumerate(class_order):
        col = f"twe_{cls}"
        vals = scores_df[col].values.astype(float)
        ax.barh(
            y_pos,
            vals,
            left=left,
            color=class_colors[idx],
            edgecolor="white",
            linewidth=0.5,
            label=DISPLAY_NAMES[idx],
        )
        left += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_labels)
    ax.set_xlabel("Total Weighted Error")
    ax.set_title("Per-Class Contribution to Total Weighted Error")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Ulozeno: %s", output_path)


# confusion matrices -> vazene chyby -> CSV + grafy -> souhrn
def run_severity_analysis(
    input_path: str = "data/processed/products_labeled.csv",
    models_dir: str = "models",
    results_dir: str = "results/phase5c_severity",
    phase5_results_dir: str = "results/phase5_modeling",
) -> dict:
    results_dir = Path(results_dir)
    models_dir = Path(models_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    severity, class_order = get_severity_matrix()
    severity_df = pd.DataFrame(severity, index=class_order, columns=class_order)
    severity_df.to_csv(results_dir / "severity_matrix.csv")
    logger.info("Matice zavaznosti ulozena.")

    logger.info("Pocitam confusion matrices...")
    cm_dict = compute_confusion_matrices(input_path, str(models_dir))

    scores_df = compute_all_model_scores(cm_dict, severity)
    scores_df.to_csv(results_dir / "weighted_error_scores.csv", index=False)
    logger.info("Tabulka vazenych chyb ulozena.")

    plot_severity_matrix(severity, class_order, str(results_dir / "severity_matrix.png"))

    for mt in ["dt", "rf", "xgb"]:
        plot_severity_confusion(
            cm_dict[mt]["cm"],
            severity,
            class_order,
            mt,
            str(results_dir / f"severity_cm_{mt}.png"),
        )

    plot_weighted_error_comparison(scores_df, str(results_dir / "weighted_error_comparison.png"))
    plot_per_class_contribution(scores_df, class_order, str(results_dir / "per_class_contribution.png"))

    best_idx = scores_df["weighted_error_rate"].idxmin()
    best_model = scores_df.loc[best_idx, "model"]
    best_wer = scores_df.loc[best_idx, "weighted_error_rate"]

    summary = {
        "best_model_by_wer": best_model,
        "best_weighted_error_rate": best_wer,
        "scores": scores_df.to_dict(orient="records"),
    }

    logger.info(
        "Souhrn analyzy zavaznosti\n"
        "Nejlepsi model (nejnizsi WER): %s (WER=%.4f)\n%s",
        best_model,
        best_wer,
        scores_df.to_string(index=False),
    )

    return summary
