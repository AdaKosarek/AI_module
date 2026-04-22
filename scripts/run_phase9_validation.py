"""Faze 9: Validacni scenar — 10 produktu"""

import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

_PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    from src.models import split_data
    from src.similarity import find_similar_products

    logger.info("Validacni scenar")
    results_dir = Path("results/phase9_prototype")
    results_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = split_data()
    xgb_model = joblib.load("models/best_model.joblib")
    knn_model = joblib.load("models/knn_model.joblib")

    le = LabelEncoder()
    le.fit(y_train)

    rng = np.random.RandomState(99)
    classes = ["shelf_picking", "front_zone_bin", "special_zone", "floor_block", "pallet_rack"]
    selected_indices = []
    for cls in classes:
        cls_indices = y_test[y_test == cls].index.tolist()
        if len(cls_indices) >= 2:
            picked = rng.choice(cls_indices, size=2, replace=False)
            selected_indices.extend(picked.tolist())
        else:
            selected_indices.extend(cls_indices)

    rows = []
    for idx in selected_indices:
        pos = X_test.index.get_loc(idx)
        X_query = X_test.iloc[[pos]]

        y_pred_enc = xgb_model.predict(X_query)
        y_pred = le.inverse_transform(y_pred_enc)[0]
        proba = xgb_model.predict_proba(X_query)[0]
        confidence = float(proba.max())

        neighbors = find_similar_products(knn_model, X_query, X_train, y_train, k=5)
        knn_classes = neighbors["true_class"].tolist()
        knn_majority = max(set(knn_classes), key=knn_classes.count)
        knn_agreement = sum(1 for c in knn_classes if c == y_pred) / len(knn_classes)

        rows.append({
            "product_idx": idx,
            "true_class": y_test.loc[idx],
            "predicted_class": y_pred,
            "confidence": round(confidence, 4),
            "knn_majority_class": knn_majority,
            "knn_agreement": round(knn_agreement, 2),
            "correct": y_pred == y_test.loc[idx],
            "human_rating": "",
        })

    df = pd.DataFrame(rows)
    df.drop(columns=["human_rating"], inplace=True)
    df.to_csv(results_dir / "validation_table.csv", index=False)
    logger.info("Ulozeno %d radku do %s", len(df), results_dir / "validation_table.csv")

    logger.info("=== Cold-start model (bez turnover) ===")
    xgb_cs = joblib.load("models/best_model_no_turnover.joblib")
    knn_cs = joblib.load("models/knn_model_no_turnover.joblib")

    X_train_no = X_train.drop(columns=["daily_turnover"]).copy()
    X_test_no = X_test.drop(columns=["daily_turnover"]).copy()

    rows_cs = []
    for idx in selected_indices:
        pos = X_test_no.index.get_loc(idx)
        X_query_cs = X_test_no.iloc[[pos]]

        y_pred_enc_cs = xgb_cs.predict(X_query_cs)
        y_pred_cs = le.inverse_transform(y_pred_enc_cs)[0]
        proba_cs = xgb_cs.predict_proba(X_query_cs)[0]
        confidence_cs = float(proba_cs.max())

        neighbors_cs = find_similar_products(knn_cs, X_query_cs, X_train_no, y_train, k=5)
        knn_classes_cs = neighbors_cs["true_class"].tolist()
        knn_majority_cs = max(set(knn_classes_cs), key=knn_classes_cs.count)
        knn_agreement_cs = sum(1 for c in knn_classes_cs if c == y_pred_cs) / len(knn_classes_cs)

        rows_cs.append({
            "product_idx": idx,
            "true_class": y_test.loc[idx],
            "predicted_class": y_pred_cs,
            "confidence": round(confidence_cs, 4),
            "knn_majority_class": knn_majority_cs,
            "knn_agreement": round(knn_agreement_cs, 2),
            "correct": y_pred_cs == y_test.loc[idx],
        })

    df_cs = pd.DataFrame(rows_cs)
    df_cs.to_csv(results_dir / "validation_table_cold_start.csv", index=False)
    logger.info("Ulozeno %d radku do %s", len(df_cs), results_dir / "validation_table_cold_start.csv")

    print("\n" + "=" * 70)
    print("VALIDACNI SCENAR — 10 PRODUKTU")
    print("=" * 70)
    print("\n  S obratkovosti (plny model, F1=0.9866):")
    for _, row in df.iterrows():
        status = "OK" if row["correct"] else "MISS"
        print(f"    [{status}] true={row['true_class']:<20} pred={row['predicted_class']:<20} conf={row['confidence']:.2f} knn={row['knn_agreement']:.0%}")
    print(f"    Presnost: {df['correct'].mean():.0%}")

    print(f"\n  Bez obratkovosti (cold-start model, F1=0.8305):")
    for _, row in df_cs.iterrows():
        status = "OK" if row["correct"] else "MISS"
        print(f"    [{status}] true={row['true_class']:<20} pred={row['predicted_class']:<20} conf={row['confidence']:.2f} knn={row['knn_agreement']:.0%}")
    print(f"    Presnost: {df_cs['correct'].mean():.0%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
