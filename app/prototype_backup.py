"""Streamlit prototyp bez api. Slouzi jako zaloha, logika v aplikaci."""

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

from src.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.models import split_data
from src.similarity import find_similar_products

CLASS_ORDER_LE = ["floor_block", "front_zone_bin", "pallet_rack", "shelf_picking", "special_zone"]

STORAGE_CLASS_CZ = {
    "shelf_picking": "Police na ruční vychystávání",
    "front_zone_bin": "Přihrádka v přední zóně",
    "special_zone": "Speciální zóna",
    "floor_block": "Bloková zóna na zemi",
    "pallet_rack": "Paletová pozice v regálu",
}

CATEGORY_GROUPS = [
    "electronics", "furniture", "home_appliances", "beauty_health",
    "sports_leisure", "fashion", "toys_baby", "books_media",
    "home_garden", "food_drinks", "housewares", "stationery", "other",
]

NUMERIC_NO_TURNOVER = [f for f in NUMERIC_FEATURES if f != "daily_turnover"]

# Staticke kurzy
CURRENCY_RATES = {
    "BRL": 1.0,
    "USD": 5.70,
    "EUR": 6.20,
    "CZK": 0.24,
    "GBP": 7.20,
}

@st.cache_resource
def load_models() -> dict:
    models_dir = _PROJECT_ROOT / "models"
    return {
        "xgb_full": joblib.load(models_dir / "best_model.joblib"),
        "knn_full": joblib.load(models_dir / "knn_model.joblib"),
        "xgb_no_turnover": joblib.load(models_dir / "best_model_no_turnover.joblib"),
        "knn_no_turnover": joblib.load(models_dir / "knn_model_no_turnover.joblib"),
    }


@st.cache_resource
def load_training_data() -> tuple:
    X_train, X_test, y_train, y_test = split_data(
        str(_PROJECT_ROOT / "data" / "processed" / "products_labeled.csv")
    )
    median_freight = float(X_train["avg_freight"].median())
    return X_train, X_test, y_train, y_test, median_freight


@st.cache_resource
def build_label_encoder(_y_train: pd.Series) -> LabelEncoder:
    le = LabelEncoder()
    le.fit(_y_train)
    assert list(le.classes_) == CLASS_ORDER_LE, (
        f"LabelEncoder classes {list(le.classes_)} != expected {CLASS_ORDER_LE}"
    )
    return le


def build_query_df(
    weight: float,
    length: float,
    height: float,
    width: float,
    category: str,
    price: float,
    turnover: float,
    median_freight: float,
    cold_start: bool,
) -> pd.DataFrame:
    """Sestavi DataFrame s jednim radkem pro predikci."""
    volume = length * height * width
    volumetric_density = weight / volume if volume > 0 else 0.0
    price_per_kg = price / (weight / 1000.0) if weight > 0 else 0.0
    avg_freight = median_freight

    if cold_start:
        data = {
            "product_weight_g": [weight],
            "product_length_cm": [length],
            "product_height_cm": [height],
            "product_width_cm": [width],
            "volume_cm3": [volume],
            "volumetric_density": [volumetric_density],
            "price_per_kg": [price_per_kg],
            "avg_price": [price],
            "avg_freight": [avg_freight],
            "category_group": [category],
        }
        return pd.DataFrame(data, columns=NUMERIC_NO_TURNOVER + CATEGORICAL_FEATURES)
    else:
        data = {
            "product_weight_g": [weight],
            "product_length_cm": [length],
            "product_height_cm": [height],
            "product_width_cm": [width],
            "volume_cm3": [volume],
            "volumetric_density": [volumetric_density],
            "price_per_kg": [price_per_kg],
            "avg_price": [price],
            "avg_freight": [avg_freight],
            "daily_turnover": [turnover],
            "category_group": [category],
        }
        return pd.DataFrame(data, columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)


def generate_explanation(
    pred_cz: str, confidence: float, neighbors_df: pd.DataFrame
) -> str:
    majority_class = neighbors_df["true_class"].value_counts().idxmax()
    majority_count = int(neighbors_df["true_class"].value_counts().iloc[0])
    majority_cz = STORAGE_CLASS_CZ.get(majority_class, majority_class)

    if majority_cz == pred_cz:
        return (
            f"Doporučení se shoduje s historickými daty, "
            f"{majority_count} z 5 nejpodobnějších produktů je ve třídě {majority_cz}."
        )
    else:
        return (
            f"Pozor: KNN sousedé preferují třídu {majority_cz} "
            f"({majority_count} z 5), zatímco model doporučuje {pred_cz}."
        )


st.set_page_config(page_title="Skladová lokace", layout="wide", page_icon="\U0001F4E6")

st.sidebar.title("Nastavení")
cold_start = st.sidebar.toggle(
    "Cold-start režim", value=True,
    help="Nový produkt bez historie objednávek",
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model**: XGBoost (F1=0.9866)")
st.sidebar.markdown("**KNN**: K=5, MinMaxScaler")

st.title("Doporučení skladové lokace pro nový produkt")
st.markdown("Zadejte parametry produktu a získejte doporučení typu skladové lokace.")

with st.spinner("Načítám modely a data..."):
    models = load_models()
    X_train, X_test, y_train, y_test, median_freight = load_training_data()
    le = build_label_encoder(y_train)

with st.form("product_form"):
    col1, col2 = st.columns(2)
    with col1:
        weight = st.number_input("Hmotnost (g)", min_value=1, max_value=50000, value=500, step=10)
        length = st.number_input("Délka (cm)", min_value=1, max_value=150, value=20, step=1)
        height = st.number_input("Výška (cm)", min_value=1, max_value=150, value=10, step=1)
    with col2:
        width = st.number_input("Šířka (cm)", min_value=1, max_value=150, value=15, step=1)
        category = st.selectbox("Kategorie", CATEGORY_GROUPS, index=0)
        currency = st.selectbox("Měna", list(CURRENCY_RATES.keys()), index=3)
        price_input = st.number_input(
            f"Cena produktu", min_value=1.0, max_value=10000.0, value=50.0, step=5.0,
        )

    if not cold_start:
        turnover = st.number_input(
            "Denní obrátkovost", min_value=0.0, max_value=10.0, value=0.0, step=0.01,
        )
    else:
        turnover = 0.0
        st.info(
            "**Cold-start režim aktivní**: Používá se model natrénován BEZ obrátkovosti "
            "(Varianta C z Fáze 8, F1=0.83). Pole obrátkovost je skryto — nový produkt nemá historii objednávek."
        )

    submitted = st.form_submit_button("Doporučit skladovou lokaci", type="primary", use_container_width=True)

price = price_input * CURRENCY_RATES[currency]
if currency != "BRL":
    st.caption(f"Převod: {price_input:.1f} {currency} = {price:.1f} BRL")

if submitted:
    X_query = build_query_df(
        weight, length, height, width, category, price, turnover, median_freight, cold_start,
    )

    if cold_start:
        xgb_model = models["xgb_no_turnover"]
        knn_model = models["knn_no_turnover"]
        X_train_knn = X_train.drop(columns=["daily_turnover"]).copy()
    else:
        xgb_model = models["xgb_full"]
        knn_model = models["knn_full"]
        X_train_knn = X_train

    y_pred_enc = xgb_model.predict(X_query)
    y_pred = le.inverse_transform(y_pred_enc)[0]
    pred_cz = STORAGE_CLASS_CZ[y_pred]

    proba = xgb_model.predict_proba(X_query)[0]
    proba_df = pd.DataFrame({
        "Třída": [STORAGE_CLASS_CZ[c] for c in le.classes_],
        "Pravděpodobnost": proba,
    }).sort_values("Pravděpodobnost", ascending=True)

    confidence = proba.max()

    neighbors = find_similar_products(knn_model, X_query, X_train_knn, y_train, k=5)

    st.markdown("---")

    knn_classes = neighbors["true_class"].head(5).tolist()
    knn_agree_count = sum(1 for c in knn_classes if c == y_pred)
    knn_agree_pct = knn_agree_count / len(knn_classes) if knn_classes else 0

    res_col1, res_col2, res_col3 = st.columns([1, 1, 2])
    with res_col1:
        st.subheader("Doporučení")
        st.success(f"**{pred_cz}**")
        st.metric("Confidence (softmax)", f"{confidence:.1%}")
    with res_col2:
        st.subheader("KNN validace")
        if knn_agree_pct >= 0.8:
            st.success(f"Vysoká shoda ({knn_agree_pct:.0%})")
        elif knn_agree_pct >= 0.4:
            st.warning(f"Částečná shoda ({knn_agree_pct:.0%})")
        else:
            st.error(f"Nízká shoda ({knn_agree_pct:.0%}) — doporučena manuální kontrola")
        st.metric("Shoda s modelem", f"{knn_agree_count}/5")
    with res_col3:
        st.subheader("Pravděpodobnosti")
        st.bar_chart(proba_df.set_index("Třída"))

    st.subheader("Podobné produkty (KNN)")
    display_cols = ["rank", "distance", "product_weight_g", "volume_cm3", "category_group", "true_class"]
    available_cols = [c for c in display_cols if c in neighbors.columns]
    st.dataframe(neighbors[available_cols].head(5), width="stretch")

    st.subheader("Vysvětlení")
    explanation = generate_explanation(pred_cz, confidence, neighbors)
    st.info(explanation)
