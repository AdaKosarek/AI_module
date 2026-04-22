"""Faze 9: Priprava cold-start modelu."""

import logging
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

_PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    from src.cold_start import _build_pipeline_no_turnover, CLASS_ORDER
    from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
    from src.models import get_param_grid, split_data, train_with_gridsearch

    logger.info("=== Faze 9: Priprava cold-start modelu pro prototyp ===")

    input_path = "data/processed/products_labeled.csv"
    X_train, X_test, y_train, y_test = split_data(input_path)
    logger.info("Split hotov: train=%d, test=%d", len(X_train), len(X_test))

    X_train_no = X_train.drop(columns=["daily_turnover"]).copy()
    X_test_no = X_test.drop(columns=["daily_turnover"]).copy()

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Trenuji XGBoost bez daily_turnover ===")

    le = LabelEncoder()
    le.fit(sorted(set(y_train.unique()) | set(CLASS_ORDER)))

    y_train_enc = pd.Series(le.transform(y_train), index=y_train.index)
    y_test_enc = pd.Series(le.transform(y_test), index=y_test.index)

    xgb_pipeline = _build_pipeline_no_turnover("xgb")
    xgb_param_grid = get_param_grid("xgb")
    xgb_grid, xgb_time = train_with_gridsearch(
        xgb_pipeline, xgb_param_grid, X_train_no, y_train_enc, model_type="xgb"
    )
    xgb_best = xgb_grid.best_estimator_

    y_pred_xgb_enc = xgb_best.predict(X_test_no)
    xgb_acc = accuracy_score(y_test_enc, y_pred_xgb_enc)
    xgb_f1 = f1_score(y_test_enc, y_pred_xgb_enc, average="macro")

    xgb_path = models_dir / "best_model_no_turnover.joblib"
    joblib.dump(xgb_best, xgb_path)
    logger.info("Ulozen XGBoost (no turnover): %s", xgb_path)
    logger.info("XGBoost accuracy=%.4f, f1_macro=%.4f (train=%.1fs)", xgb_acc, xgb_f1, xgb_time)

    le_path = models_dir / "label_encoder_no_turnover.joblib"
    joblib.dump(le, le_path)
    logger.info("Ulozen LabelEncoder: %s", le_path)

    logger.info("Trenuji KNN bez daily_turnover")

    numeric_no_turn = [f for f in NUMERIC_FEATURES if f != "daily_turnover"]

    preprocessor = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", MinMaxScaler()),
            ]),
            numeric_no_turn,
        ),
        (
            "categorical",
            OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            CATEGORICAL_FEATURES,
        ),
    ])

    knn_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("knn", KNeighborsClassifier(
            n_neighbors=5, weights="distance", metric="euclidean",
        )),
    ])

    knn_pipeline.fit(X_train_no, y_train)

    y_pred_knn = knn_pipeline.predict(X_test_no)
    knn_acc = accuracy_score(y_test, y_pred_knn)
    knn_f1 = f1_score(y_test, y_pred_knn, average="macro")

    knn_path = models_dir / "knn_model_no_turnover.joblib"
    joblib.dump(knn_pipeline, knn_path)
    logger.info("Ulozen KNN (no turnover): %s", knn_path)
    logger.info("KNN accuracy=%.4f, f1_macro=%.4f", knn_acc, knn_f1)

    print("\n" + "=" * 70)
    print("PRIPRAVA COLD-START MODELU — SOUHRN")
    print("=" * 70)
    print(f"XGBoost (no turnover): accuracy={xgb_acc:.4f}, f1_macro={xgb_f1:.4f}")
    print(f"  -> ulozeno: {xgb_path}")
    print(f"KNN (no turnover): accuracy={knn_acc:.4f}, f1_macro={knn_f1:.4f}")
    print(f"  -> ulozeno: {knn_path}")
    print(f"LabelEncoder: ulozeno: {le_path}")


if __name__ == "__main__":
    main()
