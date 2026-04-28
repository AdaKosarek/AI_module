"""Logika API - predikce, KNN, vysvetleni, logy."""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pydantic import ValidationError
from sklearn.preprocessing import LabelEncoder

_API_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _API_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.models import split_data
from src.similarity import find_similar_products

from api.constants import (
    CATEGORY_GROUPS,
    STORAGE_CLASS_CZ,
    CLASS_ORDER_LE,
    NUMERIC_NO_TURNOVER,
)
from api.schemas import (
    ProductInput,
    PredictionResponse,
    SimilarProduct,
    BatchPredictRequest,
    BatchPredictResponse,
    BatchItemOk,
    BatchItemError,
    BatchSummary,
)

logger = logging.getLogger(__name__)


class PredictionService:

    def __init__(self) -> None:
        models_dir = _PROJECT_ROOT / "models"
        data_path = str(
            _PROJECT_ROOT / "data" / "processed" / "products_labeled.csv"
        )

        logger.info("Nacitam modely z %s ...", models_dir)
        self.models = {
            "xgb_full": joblib.load(models_dir / "best_model.joblib"),
            "knn_full": joblib.load(models_dir / "knn_model.joblib"),
            "xgb_no_turnover": joblib.load(
                models_dir / "best_model_no_turnover.joblib"
            ),
            "knn_no_turnover": joblib.load(
                models_dir / "knn_model_no_turnover.joblib"
            ),
        }
        logger.info("Modely nacteny.")

        logger.info("Nacitam trenovaci data pro KNN ...")
        self.X_train, _, self.y_train, _ = split_data(data_path)
        self.median_freight = float(self.X_train["avg_freight"].median())
        logger.info(
            "Trenovaci data: %d radku, median_freight=%.2f",
            len(self.X_train),
            self.median_freight,
        )

        self.le = LabelEncoder()
        self.le.fit(self.y_train)
        assert list(self.le.classes_) == CLASS_ORDER_LE, (
            f"LabelEncoder classes {list(self.le.classes_)} != expected "
            f"{CLASS_ORDER_LE}"
        )

        self._setup_audit_logger()

    def _setup_audit_logger(self) -> None:
        logs_dir = _PROJECT_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        self.audit_logger = logging.getLogger("api.audit")
        self.audit_logger.setLevel(logging.INFO)

        if not self.audit_logger.handlers:
            fh = logging.FileHandler(
                str(logs_dir / "api_audit.log"), encoding="utf-8"
            )
            fh.setFormatter(
                logging.Formatter("%(asctime)s %(message)s")
            )
            self.audit_logger.addHandler(fh)

    def _build_query_df(self, product: ProductInput) -> pd.DataFrame:
        volume = (
            product.product_length_cm
            * product.product_height_cm
            * product.product_width_cm
        )
        volumetric_density = (
            product.product_weight_g / volume if volume > 0 else 0.0
        )
        price_per_kg = (
            product.avg_price / (product.product_weight_g / 1000.0)
            if product.product_weight_g > 0
            else 0.0
        )
        avg_freight = self.median_freight

        if product.cold_start:
            data = {
                "product_weight_g": [product.product_weight_g],
                "product_length_cm": [product.product_length_cm],
                "product_height_cm": [product.product_height_cm],
                "product_width_cm": [product.product_width_cm],
                "volume_cm3": [volume],
                "volumetric_density": [volumetric_density],
                "price_per_kg": [price_per_kg],
                "avg_price": [product.avg_price],
                "avg_freight": [avg_freight],
                "category_group": [product.category_group],
            }
            return pd.DataFrame(
                data, columns=NUMERIC_NO_TURNOVER + CATEGORICAL_FEATURES
            )
        else:
            data = {
                "product_weight_g": [product.product_weight_g],
                "product_length_cm": [product.product_length_cm],
                "product_height_cm": [product.product_height_cm],
                "product_width_cm": [product.product_width_cm],
                "volume_cm3": [volume],
                "volumetric_density": [volumetric_density],
                "price_per_kg": [price_per_kg],
                "avg_price": [product.avg_price],
                "avg_freight": [avg_freight],
                "daily_turnover": [product.daily_turnover],
                "category_group": [product.category_group],
            }
            return pd.DataFrame(
                data, columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES
            )

    def _generate_explanation(
        self,
        pred_cz: str,
        confidence: float,
        neighbors_df: pd.DataFrame,
        predicted_class: str,
    ) -> str:
        majority_class = neighbors_df["true_class"].value_counts().idxmax()
        majority_count = int(
            neighbors_df["true_class"].value_counts().iloc[0]
        )
        majority_cz = STORAGE_CLASS_CZ.get(majority_class, majority_class)

        if majority_cz == pred_cz:
            return (
                f"Doporučení se shoduje s historickými daty, "
                f"{majority_count} z 5 nejpodobnějších produktů je ve "
                f"třídě {majority_cz}."
            )
        else:
            return (
                f"Pozor: KNN sousedé preferují třídu {majority_cz} "
                f"({majority_count} z 5), zatímco model doporučuje "
                f"{pred_cz}."
            )

    def predict(self, product: ProductInput) -> PredictionResponse:
        """Kompletni predikcni pipeline — XGBoost + KNN + vysvetleni."""
        X_query = self._build_query_df(product)

        if product.cold_start:
            xgb_model = self.models["xgb_no_turnover"]
            knn_model = self.models["knn_no_turnover"]
            X_train_knn = self.X_train.drop(columns=["daily_turnover"]).copy()
        else:
            xgb_model = self.models["xgb_full"]
            knn_model = self.models["knn_full"]
            X_train_knn = self.X_train

        y_pred_enc = xgb_model.predict(X_query)
        y_pred = self.le.inverse_transform(y_pred_enc)[0]
        pred_cz = STORAGE_CLASS_CZ[y_pred]

        proba = xgb_model.predict_proba(X_query)[0]
        confidence = float(proba.max())
        all_probabilities = {
            cls: round(float(p), 4)
            for cls, p in zip(self.le.classes_, proba)
        }

        neighbors = find_similar_products(
            knn_model, X_query, X_train_knn, self.y_train, k=5
        )

        similar_products = []
        for _, row in neighbors.iterrows():
            volume = float(row.get("volume_cm3", 0.0))
            similar_products.append(
                SimilarProduct(
                    rank=int(row["rank"]),
                    distance=round(float(row["distance"]), 6),
                    weight_g=float(row["product_weight_g"]),
                    volume_cm3=round(volume, 1),
                    category=str(row["category_group"]),
                    storage_class=str(row["true_class"]),
                    storage_class_cz=STORAGE_CLASS_CZ.get(
                        str(row["true_class"]), str(row["true_class"])
                    ),
                )
            )

        knn_classes = neighbors["true_class"].head(5).tolist()
        knn_agree_count = sum(1 for c in knn_classes if c == y_pred)
        knn_agreement = f"{knn_agree_count}/5"

        explanation = self._generate_explanation(
            pred_cz, confidence, neighbors, y_pred
        )

        response = PredictionResponse(
            recommended_zone=y_pred,
            recommended_zone_cz=pred_cz,
            confidence=round(confidence, 4),
            all_probabilities=all_probabilities,
            similar_products=similar_products,
            knn_agreement=knn_agreement,
            explanation=explanation,
            cold_start_mode=product.cold_start,
            model_version="1.0.0",
        )

        self._log_audit(product, response)

        return response

    def _log_audit(
        self, product: ProductInput, response: PredictionResponse
    ) -> None:
        self.audit_logger.info(
            "PREDICT | id=%s | cold_start=%s | input=%s | "
            "result=%s (%.1f%%) | knn=%s",
            product.product_id or "-",
            product.cold_start,
            {
                "weight": product.product_weight_g,
                "dims": f"{product.product_length_cm}x"
                f"{product.product_height_cm}x"
                f"{product.product_width_cm}",
                "category": product.category_group,
                "price": product.avg_price,
            },
            response.recommended_zone,
            response.confidence * 100,
            response.knn_agreement,
        )

    def predict_batch(self, request: BatchPredictRequest) -> BatchPredictResponse:
        t0 = time.perf_counter()
        n = len(request.products)
        self.audit_logger.info("BATCH_START | n=%d", n)

        results: list = []
        ok_count = 0

        for idx, raw in enumerate(request.products):
            pid = raw.get("product_id") if isinstance(raw, dict) else None
            try:
                product = ProductInput.model_validate(raw)
                prediction = self.predict(product)
                results.append(BatchItemOk(
                    index=idx,
                    product_id=product.product_id,
                    prediction=prediction,
                ))
                ok_count += 1
            except ValidationError as e:
                results.append(BatchItemError(
                    index=idx,
                    product_id=pid,
                    errors=e.errors(),
                ))

        duration_ms = (time.perf_counter() - t0) * 1000
        err_count = n - ok_count
        self.audit_logger.info(
            "BATCH_END | n=%d | ok=%d | err=%d | took=%.1fms",
            n, ok_count, err_count, duration_ms,
        )

        return BatchPredictResponse(
            results=results,
            summary=BatchSummary(
                total=n,
                ok_count=ok_count,
                error_count=err_count,
                duration_ms=round(duration_ms, 1),
            ),
        )
