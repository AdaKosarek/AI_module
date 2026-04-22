"""KNN pro hledani podobnych produktu a validaci predikci XGBoost."""

import logging
from pathlib import Path
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.features import (
    build_preprocessing_pipeline,
    compute_daily_turnover,
    select_features,
)
from src.models import split_data

logger = logging.getLogger(__name__)

CLASS_ORDER = [
    "shelf_picking",
    "front_zone_bin",
    "special_zone",
    "floor_block",
    "pallet_rack",
]

DISPLAY_NAMES = ["Shelf", "Front Bin", "Special", "Floor Block", "Pallet"]


# Nefitnuta pipeline
def build_knn_pipeline(
    n_neighbors: int = 5,
    weights: str = "distance",
    metric: str = "euclidean",
) -> Pipeline:
    preprocessor = build_preprocessing_pipeline("minmax")
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors, weights=weights, metric=metric, n_jobs=-1
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("knn", knn)])
    return pipeline


def find_best_k(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k_values: list[int] | None = None,
    cv: int = 5,
) -> tuple[int, pd.DataFrame]:
    if k_values is None:
        k_values = [3, 5, 7, 10]

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    rows = []

    for k in k_values:
        logger.info("Cross-validace pro K=%d ...", k)
        pipe = build_knn_pipeline(n_neighbors=k)
        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=skf,
            scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"},
            n_jobs=-1,
        )
        rows.append(
            {
                "k": k,
                "mean_f1_macro": np.mean(scores["test_f1_macro"]),
                "std_f1_macro": np.std(scores["test_f1_macro"]),
                "mean_accuracy": np.mean(scores["test_accuracy"]),
                "std_accuracy": np.std(scores["test_accuracy"]),
            }
        )

    cv_df = pd.DataFrame(rows)
    best_idx = cv_df["mean_f1_macro"].idxmax()
    best_k = int(cv_df.loc[best_idx, "k"])
    logger.info(
        "Nejlepsi K=%d (mean_f1_macro=%.4f)",
        best_k,
        cv_df.loc[best_idx, "mean_f1_macro"],
    )
    return best_k, cv_df


# Porovna predikce KNN a XGBoost na test setu
def compute_agreement(
    knn_pipeline: Pipeline,
    xgb_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label_encoder: LabelEncoder,
) -> dict:
    y_knn = knn_pipeline.predict(X_test)

    # XGBoost vraci numericke labely, prevedeme zpet pres LabelEncoder
    y_xgb_enc = xgb_model.predict(X_test)
    y_xgb = label_encoder.inverse_transform(y_xgb_enc)

    agree_mask = y_knn == y_xgb
    overall_agreement = float(np.mean(agree_mask))

    per_class_agreement = {}
    for cls in CLASS_ORDER:
        mask = y_test == cls
        if mask.sum() == 0:
            per_class_agreement[cls] = 0.0
        else:
            per_class_agreement[cls] = float(np.mean(y_knn[mask] == y_xgb[mask]))

    knn_accuracy = float(accuracy_score(y_test, y_knn))
    xgb_accuracy = float(accuracy_score(y_test, y_xgb))

    disagree_idx = np.where(~agree_mask)[0]
    disagreement_df = pd.DataFrame(
        {
            "true_class": y_test.values[disagree_idx],
            "knn_pred": y_knn[disagree_idx],
            "xgb_pred": y_xgb[disagree_idx],
        },
        index=X_test.index[disagree_idx],
    )

    logger.info(
        "Shoda KNN vs. XGB: %.2f%% (%d neshod z %d)",
        overall_agreement * 100,
        len(disagree_idx),
        len(y_test),
    )

    return {
        "overall_agreement": overall_agreement,
        "per_class_agreement": per_class_agreement,
        "knn_accuracy": knn_accuracy,
        "xgb_accuracy": xgb_accuracy,
        "disagreement_df": disagreement_df,
        "y_knn": y_knn,
        "y_xgb": y_xgb,
    }


def find_similar_products(
    knn_pipeline: Pipeline,
    X_query: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k: int = 5,
) -> pd.DataFrame:
    preprocessor = knn_pipeline.named_steps["preprocessor"]
    knn_model = knn_pipeline.named_steps["knn"]

    X_query_scaled = preprocessor.transform(X_query)
    distances, indices = knn_model.kneighbors(X_query_scaled, n_neighbors=k)

    results = []
    for q_idx in range(len(X_query)):
        for rank, (dist, train_pos) in enumerate(
            zip(distances[q_idx], indices[q_idx]), start=1
        ):
            train_row = X_train.iloc[train_pos]
            row = {"query_idx": q_idx, "rank": rank, "distance": dist}
            # Puvodni DataFrame index drzime zvlast — potreba pro pozdejsi product_id
            row["train_original_idx"] = X_train.index[train_pos]
            for col in X_train.columns:
                row[col] = train_row[col]
            row["true_class"] = y_train.iloc[train_pos]
            results.append(row)

    return pd.DataFrame(results)


# Vysvetli XGBoost predikci pro jeden testovaci produkt
def explain_recommendation(
    test_idx: int,
    knn_pipeline: Pipeline,
    xgb_model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    label_encoder: LabelEncoder,
    df_full: pd.DataFrame,
    k: int = 5,
) -> dict:
    query = X_test.iloc[[test_idx]]
    original_idx = X_test.index[test_idx]

    neighbors_df = find_similar_products(knn_pipeline, query, X_train, y_train, k=k)

    knn_vote = knn_pipeline.predict(query)[0]

    xgb_enc = xgb_model.predict(query)[0]
    xgb_pred = label_encoder.inverse_transform([xgb_enc])[0]

    true_class = y_test.iloc[test_idx]

    product_id = None
    if "product_id" in df_full.columns and original_idx in df_full.index:
        product_id = df_full.loc[original_idx, "product_id"]

    return {
        "product_id": product_id,
        "true_class": true_class,
        "xgb_pred": xgb_pred,
        "knn_vote": knn_vote,
        "agree": xgb_pred == knn_vote,
        "neighbors_df": neighbors_df,
    }

##
def generate_sanity_check(
    knn_pipeline: Pipeline,
    xgb_model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    label_encoder: LabelEncoder,
    df_full: pd.DataFrame,
    k: int = 5,
    n_examples: int = 10,
) -> pd.DataFrame:
    y_knn = knn_pipeline.predict(X_test)
    y_xgb_enc = xgb_model.predict(X_test)
    y_xgb = label_encoder.inverse_transform(y_xgb_enc)

    agree_mask = y_knn == y_xgb
    agree_indices = np.where(agree_mask)[0]
    disagree_indices = np.where(~agree_mask)[0]

    rng = np.random.RandomState(42)
    n_agree = min(n_examples // 2, len(agree_indices))
    n_disagree = min(n_examples - n_agree, len(disagree_indices))

    sampled_agree = rng.choice(agree_indices, size=n_agree, replace=False) if n_agree > 0 else np.array([], dtype=int)
    sampled_disagree = rng.choice(disagree_indices, size=n_disagree, replace=False) if n_disagree > 0 else np.array([], dtype=int)
    sampled = np.concatenate([sampled_agree, sampled_disagree])

    rows = []
    for idx in sampled:
        result = explain_recommendation(
            int(idx), knn_pipeline, xgb_model,
            X_train, X_test, y_train, y_test,
            label_encoder, df_full, k=k,
        )
        row = {
            "product_id": result["product_id"],
            "true_class": result["true_class"],
            "xgb_pred": result["xgb_pred"],
            "knn_vote": result["knn_vote"],
            "agree": result["agree"],
        }
        nbrs = result["neighbors_df"]
        for i in range(k):
            if i < len(nbrs):
                nbr = nbrs.iloc[i]
                nbr_orig_idx = nbr.get("train_original_idx", None)
                nbr_product_id = None
                if nbr_orig_idx is not None and "product_id" in df_full.columns:
                    if nbr_orig_idx in df_full.index:
                        nbr_product_id = df_full.loc[nbr_orig_idx, "product_id"]
                row[f"neighbor_{i+1}_id"] = nbr_product_id
                row[f"neighbor_{i+1}_class"] = nbr["true_class"]
                row[f"neighbor_{i+1}_dist"] = round(nbr["distance"], 6)
            else:
                row[f"neighbor_{i+1}_id"] = None
                row[f"neighbor_{i+1}_class"] = None
                row[f"neighbor_{i+1}_dist"] = None
        rows.append(row)

    logger.info(
        "Sanity check: %d prikladu (%d shoda, %d neshoda)",
        len(rows), n_agree, n_disagree,
    )
    return pd.DataFrame(rows)

def plot_accuracy_by_k(cv_results_df: pd.DataFrame, output_path: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ks = cv_results_df["k"].values
    means = cv_results_df["mean_f1_macro"].values
    stds = cv_results_df["std_f1_macro"].values

    best_idx = np.argmax(means)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ks, means, yerr=stds, fmt="-o", capsize=4, label="F1 Macro")
    ax.plot(
        ks[best_idx], means[best_idx], marker="*", markersize=18,
        color="red", zorder=5, label=f"Best K={ks[best_idx]}",
    )

    ax.set_xlabel("K (pocet sousedu)")
    ax.set_ylabel("Mean F1 Macro")
    ax.set_title("KNN Cross-Validation: F1 Macro by K")
    ax.set_xticks(ks)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Ulozeno: %s", output_path)

def plot_agreement_by_class(
    per_class_agreement: dict[str, float],
    overall_agreement: float,
    output_path: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = DISPLAY_NAMES + ["Overall"]
    values = [per_class_agreement.get(cls, 0.0) for cls in CLASS_ORDER] + [overall_agreement]

    colors = []
    for v in values:
        if v >= 0.90:
            colors.append("#2ca02c")
        elif v >= 0.80:
            colors.append("#ff7f0e")
        else:
            colors.append("#d62728")

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Agreement Rate")
    ax.set_title("KNN vs. XGBoost Agreement by Class")
    ax.set_xlim(0, 1.15)
    ax.axvline(x=0.9, color="gray", linestyle="--", alpha=0.5, label="90% threshold")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Ulozeno: %s", output_path)

def plot_knn_vs_classifier(
    y_true: np.ndarray | pd.Series,
    y_knn: np.ndarray,
    y_xgb: np.ndarray,
    output_path: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_true_arr = np.array(y_true)
    knn_acc = []
    xgb_acc = []
    for cls in CLASS_ORDER:
        mask = y_true_arr == cls
        if mask.sum() == 0:
            knn_acc.append(0.0)
            xgb_acc.append(0.0)
        else:
            knn_acc.append(float(np.mean(y_knn[mask] == y_true_arr[mask])))
            xgb_acc.append(float(np.mean(y_xgb[mask] == y_true_arr[mask])))

    x = np.arange(len(CLASS_ORDER))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_knn = ax.bar(x - width / 2, knn_acc, width, label="KNN", color="#1f77b4")
    bars_xgb = ax.bar(x + width / 2, xgb_acc, width, label="XGBoost", color="#d62728")

    for bars in [bars_knn, bars_xgb]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, height + 0.01,
                f"{height:.2f}", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xlabel("Storage Class")
    ax.set_ylabel("Accuracy")
    ax.set_title("KNN vs. XGBoost Accuracy by Class")
    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY_NAMES, rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Ulozeno: %s", output_path)

def plot_disagreement_heatmap(
    y_xgb: np.ndarray,
    y_knn: np.ndarray,
    class_order: list[str],
    output_path: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    disagree_mask = y_xgb != y_knn
    y_xgb_dis = y_xgb[disagree_mask]
    y_knn_dis = y_knn[disagree_mask]

    n = len(class_order)
    heatmap = np.zeros((n, n), dtype=int)
    class_to_idx = {cls: i for i, cls in enumerate(class_order)}

    for xgb_pred, knn_pred in zip(y_xgb_dis, y_knn_dis):
        i = class_to_idx.get(xgb_pred)
        j = class_to_idx.get(knn_pred)
        if i is not None and j is not None:
            heatmap[i, j] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heatmap, cmap="Oranges", interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(DISPLAY_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(DISPLAY_NAMES)
    ax.set_xlabel("KNN Prediction")
    ax.set_ylabel("XGBoost Prediction")
    ax.set_title("Disagreement Heatmap: XGBoost vs. KNN Predictions")

    thresh = heatmap.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(heatmap[i, j]),
                ha="center", va="center",
                color="white" if heatmap[i, j] > thresh else "black",
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Ulozeno: %s", output_path)


# find_best_k -> fit KNN -> agreement vs. XGBoost -> CSV + grafy + sanity check
def run_knn_analysis(
    input_path: str = "data/processed/products_labeled.csv",
    models_dir: str = "models",
    results_dir: str = "results/phase5d_similarity",
) -> dict:
    input_path = Path(input_path)
    models_dir = Path(models_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== KNN Analyza: start ===")
    X_train, X_test, y_train, y_test = split_data(str(input_path))

    df_full = pd.read_csv(input_path)
    df_full = compute_daily_turnover(df_full)

    xgb_path = models_dir / "best_model.joblib"
    logger.info("Nacitam XGBoost model: %s", xgb_path)
    xgb_model = joblib.load(xgb_path)

    le = LabelEncoder()
    le.fit(y_train)

    logger.info("Hledam nejlepsi K...")
    best_k, cv_df = find_best_k(X_train, y_train)
    cv_df.to_csv(results_dir / "knn_cv_results.csv", index=False)
    logger.info("Ulozeno knn_cv_results.csv")

    logger.info("Fituju KNN pipeline s K=%d...", best_k)
    knn_pipeline = build_knn_pipeline(n_neighbors=best_k)
    knn_pipeline.fit(X_train, y_train)

    knn_model_path = models_dir / "knn_model.joblib"
    joblib.dump(knn_pipeline, knn_model_path)
    logger.info("Ulozeno: %s", knn_model_path)

    logger.info("Pocitam shodu KNN vs. XGBoost...")
    agreement = compute_agreement(knn_pipeline, xgb_model, X_test, y_test, le)

    agreement_rows = [
        {"class": cls, "agreement": agreement["per_class_agreement"][cls]}
        for cls in CLASS_ORDER
    ]
    agreement_rows.append(
        {"class": "overall", "agreement": agreement["overall_agreement"]}
    )
    agreement_df = pd.DataFrame(agreement_rows)
    agreement_df["knn_accuracy"] = agreement["knn_accuracy"]
    agreement_df["xgb_accuracy"] = agreement["xgb_accuracy"]
    agreement_df.to_csv(results_dir / "knn_agreement_summary.csv", index=False)
    logger.info("Ulozeno knn_agreement_summary.csv")

    logger.info("Generuji sanity check priklady...")
    sanity_df = generate_sanity_check(
        knn_pipeline, xgb_model,
        X_train, X_test, y_train, y_test,
        le, df_full, k=best_k,
    )
    sanity_df.to_csv(results_dir / "sanity_check_examples.csv", index=False)
    logger.info("Ulozeno sanity_check_examples.csv")

    logger.info("Generuji grafy...")

    plot_accuracy_by_k(cv_df, str(results_dir / "knn_cv_f1_by_k.png"))

    plot_agreement_by_class(
        agreement["per_class_agreement"],
        agreement["overall_agreement"],
        str(results_dir / "knn_agreement_by_class.png"),
    )

    plot_knn_vs_classifier(
        y_test, agreement["y_knn"], agreement["y_xgb"],
        str(results_dir / "knn_vs_xgb_accuracy.png"),
    )

    plot_disagreement_heatmap(
        agreement["y_xgb"], agreement["y_knn"], CLASS_ORDER,
        str(results_dir / "disagreement_heatmap.png"),
    )

    summary = {
        "best_k": best_k,
        "knn_accuracy": agreement["knn_accuracy"],
        "xgb_accuracy": agreement["xgb_accuracy"],
        "overall_agreement": agreement["overall_agreement"],
        "per_class_agreement": agreement["per_class_agreement"],
        "n_disagreements": len(agreement["disagreement_df"]),
        "n_test_samples": len(y_test),
        "cv_results": cv_df.to_dict(orient="records"),
    }

    logger.info(
        "=== Souhrn KNN analyzy ===\n"
        "Nejlepsi K: %d\n"
        "KNN accuracy: %.4f\n"
        "XGB accuracy: %.4f\n"
        "Celkova shoda: %.2f%%\n"
        "Pocet neshod: %d / %d",
        best_k,
        agreement["knn_accuracy"],
        agreement["xgb_accuracy"],
        agreement["overall_agreement"] * 100,
        len(agreement["disagreement_df"]),
        len(y_test),
    )

    return summary
