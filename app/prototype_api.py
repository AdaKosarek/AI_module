"""Streamlit prototyp. Vola REST API misto primych modelu."""

import logging

import streamlit as st
import requests
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000"

STORAGE_CLASS_CZ = {
    "shelf_picking": "Police na ruční vychystávání",
    "front_zone_bin": "Přihrádka v přední zóně",
    "special_zone": "Speciální zóna",
    "floor_block": "Bloková zóna na zemi",
    "pallet_rack": "Paletová pozice v regálu",
}

CURRENCY_RATES = {
    "BRL": 1.0,
    "USD": 5.70,
    "EUR": 6.20,
    "CZK": 0.24,
    "GBP": 7.20,
}

# Nacte seznam kategorii z API endpointu /categories.
@st.cache_data
def load_categories() -> list[str]:

    try:
        resp = requests.get(f"{API_URL}/categories", timeout=5)
        resp.raise_for_status()
        return resp.json()["categories"]
    except requests.RequestException as exc:
        logger.warning("Nelze nacist /categories z API (%s), pouzivam fallback.", exc)
        return [
            "electronics", "furniture", "home_appliances", "beauty_health",
            "sports_leisure", "fashion", "toys_baby", "books_media",
            "home_garden", "food_drinks", "housewares", "stationery", "other",
        ]


st.set_page_config(page_title="Skladová lokace (API)", layout="wide", page_icon="\U0001F310")

st.sidebar.title("Nastavení")
cold_start = st.sidebar.toggle(
    "Cold start režim", value=True,
    help="Nový produkt bez historie objednávek",
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**API**: `{API_URL}`")

try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    st.sidebar.success(f"API: {health['status']} (v{health['version']})")
except requests.RequestException as exc:
    logger.error("API health check selhal: %s", exc)
    st.sidebar.error(
        f"API nedostupné ({type(exc).__name__}). "
        "Spusťte: uv run uvicorn api.main:app"
    )

st.title("Doporučení skladové lokace (API klient)")
st.markdown(
    "Prototyp nevolá modely přímo, "
    "posílá HTTP požadavky na REST API (`/predict`)."
)

categories = load_categories()

with st.form("product_form"):
    col1, col2 = st.columns(2)
    with col1:
        weight = st.number_input("Hmotnost (g)", min_value=1, max_value=50000, value=500, step=10)
        length = st.number_input("Délka (cm)", min_value=1, max_value=150, value=20, step=1)
        height = st.number_input("Výška (cm)", min_value=1, max_value=150, value=10, step=1)
    with col2:
        width = st.number_input("Šířka (cm)", min_value=1, max_value=150, value=15, step=1)
        category = st.selectbox("Kategorie", categories, index=0)
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
            "**Cold start režim aktivní**: API použije model natrénovaný BEZ obrátkovosti."
        )

    submitted = st.form_submit_button("Doporučit skladovou lokaci", type="primary", use_container_width=True)

price = price_input * CURRENCY_RATES[currency]
if currency != "BRL":
    st.caption(f"Převod: {price_input:.1f} {currency} = {price:.1f} BRL")

if submitted:
    payload = {
        "product_weight_g": weight,
        "product_length_cm": length,
        "product_height_cm": height,
        "product_width_cm": width,
        "category_group": category,
        "avg_price": price,
        "daily_turnover": turnover,
        "cold_start": cold_start,
    }

    with st.spinner("Odesílám dotaz na API..."):
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
            resp.raise_for_status()
        except requests.ConnectionError:
            st.error("API nedostupné. Spusťte server: `uv run uvicorn api.main:app`")
            st.stop()
        except requests.HTTPError as e:
            st.error(f"API chyba: {e.response.status_code} — {e.response.text}")
            st.stop()

    data = resp.json()

    st.markdown("---")

    knn_agreement = data.get("knn_agreement", "?/?")
    try:
        agree_num, agree_total = knn_agreement.split("/")
        knn_agree_pct = int(agree_num) / int(agree_total)
    except (ValueError, ZeroDivisionError) as exc:
        logger.warning("Nelze parsovat knn_agreement='%s': %s", knn_agreement, exc)
        knn_agree_pct = 0

    res_col1, res_col2, res_col3 = st.columns([1, 1, 2])
    with res_col1:
        st.subheader("Doporučení")
        st.success(f"**{data['recommended_zone_cz']}**")
        st.metric("Confidence (softmax)", f"{data['confidence']:.1%}")
    with res_col2:
        st.subheader("KNN validace")
        if knn_agree_pct >= 0.8:
            st.success(f"Vysoká shoda ({knn_agree_pct:.0%})")
        elif knn_agree_pct >= 0.4:
            st.warning(f"Částečná shoda ({knn_agree_pct:.0%})")
        else:
            st.error(f"Nízká shoda ({knn_agree_pct:.0%}) — doporučena manuální kontrola")
        st.metric("Shoda s modelem", knn_agreement)
    with res_col3:
        st.subheader("Pravděpodobnosti")
        proba_df = pd.DataFrame({
            "Třída": [STORAGE_CLASS_CZ.get(k, k) for k in data["all_probabilities"]],
            "Pravděpodobnost": list(data["all_probabilities"].values()),
        }).sort_values("Pravděpodobnost", ascending=True)
        st.bar_chart(proba_df.set_index("Třída"))

    st.subheader("Podobné produkty (KNN)")
    if data["similar_products"]:
        neighbors_df = pd.DataFrame(data["similar_products"])
        st.dataframe(neighbors_df, width="stretch")
    else:
        st.write("Žádné podobné produkty nenalezeny.")

    st.subheader("Vysvětlení")
    st.info(data["explanation"])

    with st.expander("Raw API response (JSON)"):
        st.json(data)
