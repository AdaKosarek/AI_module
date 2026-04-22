"""Trenovani a evaluace klasifikacnich modelu (DT, RF, XGBoost)."""

import logging
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_preprocessing_pipeline,
    compute_daily_turnover,
    select_features,
)
from src.labeling import label_products

logger = logging.getLogger(__name__)


# Nacte CSV, spocita daily_turnover, vybere features a vrati vyvazeny train/test split.
def split_data(
    input_path: str = "data/processed/products_labeled.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    df = pd.read_csv(input_path)
    logger.info("Nacteno %d radku z %s", len(df), input_path)

    df = compute_daily_turnover(df)
    X, y = select_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info("Train: %d, Test: %d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def build_model_pipeline(model_type: str, scaler_type: str = "standard") -> Pipeline:
    preprocessor = build_preprocessing_pipeline(scaler_type)

    if model_type == "dt":
        model = DecisionTreeClassifier(class_weight="balanced", random_state=42)
    elif model_type == "rf":
        model = RandomForestClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1
        )
    elif model_type == "xgb":
        model = XGBClassifier(
            objective="multi:softprob",
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
        )
    else:
        raise ValueError(f"Neznamy model_type: {model_type}")

    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
    return pipeline


def get_param_grid(model_type: str) -> dict:
    if model_type == "dt":
        return {
            "classifier__max_depth": [5, 10, 15, 20, None],
            "classifier__min_samples_split": [2, 5, 10],
        }
    elif model_type == "rf":
        return {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [10, 20, None],
            "classifier__min_samples_leaf": [1, 2, 4],
        }
    elif model_type == "xgb":
        return {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [4, 6, 8],
            "classifier__learning_rate": [0.05, 0.1],
        }
    else:
        raise ValueError(f"Neznamy model_type: {model_type}")


# Trenuje model pres GridSearchCV s f1_macro; vrati (grid_search, training_time_s).
def train_with_gridsearch(
    pipeline: Pipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "dt",
    cv: int = 5,
) -> tuple:
    fit_params = {}
    if model_type == "xgb":
        sample_weights = compute_sample_weight("balanced", y_train)
        fit_params["classifier__sample_weight"] = sample_weights

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
    )

    t0 = time.time()
    grid_search.fit(X_train, y_train, **fit_params)
    training_time = time.time() - t0

    logger.info(
        "[%s] best_params=%s, best_score=%.4f, time=%.1fs",
        model_type,
        grid_search.best_params_,
        grid_search.best_score_,
        training_time,
    )
    return grid_search, training_time


# Evaluace modelu na test setu
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    t0 = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - t0

    acc = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro")
    f1_per = f1_score(y_test, y_pred, average=None, labels=model.classes_)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

    result = {
        "accuracy": acc,
        "f1_macro": f1_mac,
        "f1_per_class": dict(zip(model.classes_, f1_per)),
        "classification_report": report,
        "confusion_matrix": cm,
        "classes": list(model.classes_),
        "predict_time_s": predict_time,
    }
    logger.info("accuracy=%.4f, f1_macro=%.4f, predict_time=%.3fs", acc, f1_mac, predict_time)
    return result


def extract_feature_importance(
    fitted_pipeline: Pipeline, feature_names: list
) -> pd.DataFrame:
    classifier = fitted_pipeline.named_steps["classifier"]
    importances = classifier.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def _get_feature_names(fitted_pipeline: Pipeline) -> list:
    ct = fitted_pipeline.named_steps["preprocessor"]
    ohe = ct.named_transformers_["categorical"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    return list(NUMERIC_FEATURES) + cat_names


def _save_confusion_matrix(
    cm: np.ndarray,
    classes: list,
    output_path: str,
    title: str = "Confusion Matrix",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# split -> train DT/RF/XGB + baseline -> evaluate -> save modely a srovnavaci tabulky
def train_all_models(
    input_path: str = "data/processed/products_labeled.csv",
    models_dir: str = "models",
    results_dir: str = "results/phase5_modeling",
) -> dict:
    models_dir = Path(models_dir)
    results_dir = Path(results_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = split_data(input_path)

    class_order = ["shelf_picking", "front_zone_bin", "special_zone", "floor_block", "pallet_rack"]

    model_types = ["dt", "rf", "xgb"]
    all_results = []
    best_f1 = -1.0
    best_model_name = None
    best_model_obj = None

    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = pd.Series(le.transform(y_train), index=y_train.index)
    y_test_enc = pd.Series(le.transform(y_test), index=y_test.index)

    for mt in model_types:
        logger.info("=== Trenuji model: %s ===", mt)

        yt_train = y_train_enc if mt == "xgb" else y_train
        yt_test = y_test_enc if mt == "xgb" else y_test

        pipeline = build_model_pipeline(mt)
        param_grid = get_param_grid(mt)
        grid, train_time = train_with_gridsearch(
            pipeline, param_grid, X_train, yt_train, model_type=mt
        )
        best_est = grid.best_estimator_

        eval_result = evaluate_model(best_est, X_test, yt_test)

        if mt == "xgb":
            eval_result["classes"] = list(le.inverse_transform(eval_result["classes"]))
            eval_result["f1_per_class"] = {
                le.inverse_transform([k])[0]: v
                for k, v in eval_result["f1_per_class"].items()
            }

        feature_names = _get_feature_names(best_est)
        fi = extract_feature_importance(best_est, feature_names)
        fi.to_csv(results_dir / f"feature_importance_{mt}.csv", index=False)
        logger.info("Ulozeno feature_importance_%s.csv", mt)

        _save_confusion_matrix(
            eval_result["confusion_matrix"],
            eval_result["classes"],
            str(results_dir / f"confusion_matrix_{mt}.png"),
            title=f"Confusion Matrix - {mt.upper()}",
        )
        logger.info("Ulozeno confusion_matrix_%s.png", mt)

        joblib.dump(best_est, models_dir / f"{mt}_model.joblib")
        logger.info("Ulozeno %s_model.joblib", mt)

        f1_per = eval_result["f1_per_class"]
        row = {
            "model": mt,
            "accuracy": round(eval_result["accuracy"], 4),
            "f1_macro": round(eval_result["f1_macro"], 4),
            "f1_shelf_picking": round(f1_per.get("shelf_picking", 0.0), 4),
            "f1_front_zone_bin": round(f1_per.get("front_zone_bin", 0.0), 4),
            "f1_special_zone": round(f1_per.get("special_zone", 0.0), 4),
            "f1_floor_block": round(f1_per.get("floor_block", 0.0), 4),
            "f1_pallet_rack": round(f1_per.get("pallet_rack", 0.0), 4),
            "train_time_s": round(train_time, 2),
            "predict_time_ms": round(eval_result["predict_time_s"] * 1000, 2),
        }
        all_results.append(row)

        if eval_result["f1_macro"] > best_f1:
            best_f1 = eval_result["f1_macro"]
            best_model_name = mt
            best_model_obj = best_est

    logger.info("=== Baseline (rule-based) ===")
    df_full = pd.read_csv(input_path)
    test_indices = y_test.index
    df_test_original = df_full.loc[test_indices].copy()

    if "storage_class" in df_test_original.columns:
        df_test_original = df_test_original.drop(columns=["storage_class"])

    df_test_relabeled = label_products(df_test_original)
    y_baseline = df_test_relabeled["storage_class"]

    acc_bl = accuracy_score(y_test, y_baseline)
    f1_bl = f1_score(y_test, y_baseline, average="macro")
    f1_bl_per = f1_score(
        y_test, y_baseline, average=None,
        labels=class_order,
    )
    f1_bl_dict = dict(zip(class_order, f1_bl_per))
    logger.info("Baseline: accuracy=%.4f, f1_macro=%.4f", acc_bl, f1_bl)

    baseline_row = {
        "model": "baseline",
        "accuracy": round(acc_bl, 4),
        "f1_macro": round(f1_bl, 4),
        "f1_shelf_picking": round(f1_bl_dict.get("shelf_picking", 0.0), 4),
        "f1_front_zone_bin": round(f1_bl_dict.get("front_zone_bin", 0.0), 4),
        "f1_special_zone": round(f1_bl_dict.get("special_zone", 0.0), 4),
        "f1_floor_block": round(f1_bl_dict.get("floor_block", 0.0), 4),
        "f1_pallet_rack": round(f1_bl_dict.get("pallet_rack", 0.0), 4),
        "train_time_s": 0.0,
        "predict_time_ms": 0.0,
    }
    all_results.append(baseline_row)

    df_comp = pd.DataFrame(all_results)
    df_comp.to_csv(results_dir / "model_comparison_table.csv", index=False)
    logger.info("Ulozeno model_comparison_table.csv")

    joblib.dump(best_model_obj, models_dir / "best_model.joblib")
    logger.info("Nejlepsi model: %s (F1 macro=%.4f), ulozeno jako best_model.joblib", best_model_name, best_f1)

    return {
        "best_model": best_model_name,
        "best_f1_macro": best_f1,
        "comparison_table": df_comp.to_dict(orient="records"),
    }
