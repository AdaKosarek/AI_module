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

LIMIT_WEIGHT_G = 50000
LIMIT_DIMENSION_CM = 150
LIMIT_PRICE_BRL = 10000
LIMIT_VOLUME_CM3 = 400_000
LIMIT_TURNOVER = 10.0
REQUEST_TIMEOUT_S = 30

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

@st.cache_data
def load_categories() -> tuple[list[str], bool]:
    try:
        resp = requests.get(f"{API_URL}/categories", timeout=5)
        resp.raise_for_status()
        return resp.json()["categories"], False
    except requests.RequestException as exc:
        logger.error("Nelze nacist /categories z API: %s", exc)
        fallback = [
            "electronics", "furniture", "home_appliances", "beauty_health",
            "sports_leisure", "fashion", "toys_baby", "books_media",
            "home_garden", "food_drinks", "housewares", "stationery", "other",
        ]
        return fallback, True


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

categories, categories_from_fallback = load_categories()
if categories_from_fallback:
    st.sidebar.warning(
        "Seznam kategorií se nepodařilo načíst z API — "
        "používá se zástupný seznam. Ověřte, že API běží na "
        f"`{API_URL}`."
    )

with st.form("product_form"):
    col1, col2 = st.columns(2)
    with col1:
        weight = st.number_input(
            "Hmotnost (g)", min_value=1, value=500, step=10,
            help=f"Limit modelu: {LIMIT_WEIGHT_G} g (mimo trénovací distribuci API odmítne).",
        )
        length = st.number_input(
            "Délka (cm)", min_value=1, value=20, step=1,
            help=f"Limit modelu: {LIMIT_DIMENSION_CM} cm.",
        )
        height = st.number_input(
            "Výška (cm)", min_value=1, value=10, step=1,
            help=f"Limit modelu: {LIMIT_DIMENSION_CM} cm.",
        )
    with col2:
        width = st.number_input(
            "Šířka (cm)", min_value=1, value=15, step=1,
            help=f"Limit modelu: {LIMIT_DIMENSION_CM} cm.",
        )
        category = st.selectbox("Kategorie", categories, index=0)
        currency = st.selectbox("Měna", list(CURRENCY_RATES.keys()), index=3)
        price_input = st.number_input(
            "Cena produktu", min_value=1.0, value=50.0, step=5.0,
            help=f"Limit modelu po převodu na BRL: {LIMIT_PRICE_BRL}.",
        )

    if not cold_start:
        turnover = st.number_input(
            "Denní obrátkovost", min_value=0.0, value=0.0, step=0.01,
            help=f"Limit modelu: {LIMIT_TURNOVER}.",
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
    pre_errors = []
    if weight > LIMIT_WEIGHT_G:
        pre_errors.append(
            f"Hmotnost {weight} g překračuje limit modelu ({LIMIT_WEIGHT_G} g)."
        )
    if length > LIMIT_DIMENSION_CM:
        pre_errors.append(
            f"Délka {length} cm překračuje limit modelu ({LIMIT_DIMENSION_CM} cm)."
        )
    if height > LIMIT_DIMENSION_CM:
        pre_errors.append(
            f"Výška {height} cm překračuje limit modelu ({LIMIT_DIMENSION_CM} cm)."
        )
    if width > LIMIT_DIMENSION_CM:
        pre_errors.append(
            f"Šířka {width} cm překračuje limit modelu ({LIMIT_DIMENSION_CM} cm)."
        )
    if price > LIMIT_PRICE_BRL:
        pre_errors.append(
            f"Cena po převodu {price:.0f} BRL překračuje limit modelu ({LIMIT_PRICE_BRL} BRL)."
        )
    if turnover > LIMIT_TURNOVER:
        pre_errors.append(
            f"Denní obrátkovost {turnover} překračuje limit modelu ({LIMIT_TURNOVER})."
        )
    volume_cm3 = length * height * width
    if volume_cm3 > LIMIT_VOLUME_CM3:
        pre_errors.append(
            f"Kombinovaný objem (D x V x Š) {volume_cm3} cm3 překračuje "
            f"limit modelu ({LIMIT_VOLUME_CM3} cm3). "
            f"Limitace vyplývá z rozsahu dat, na kterých byl model vytrénován."
        )
    if pre_errors:
        st.error("Vstup nelze odeslat na API kvůli nepovoleným hodnotám:")
        for msg in pre_errors:
            st.error(f"- {msg}")
        logger.warning("Pre-submit validace selhala: %s", pre_errors)
        st.stop()

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

    resp = None
    with st.spinner("Odesílám dotaz na API..."):
        try:
            resp = requests.post(
                f"{API_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT_S
            )
            resp.raise_for_status()
        except requests.ConnectionError as exc:
            logger.error("API nedostupne (ConnectionError): %s", exc)
            st.error(
                "API nedostupné — spojení se nepodařilo navázat. "
                "Spusťte server: `uv run uvicorn api.main:app`"
            )
            st.stop()
        except requests.Timeout as exc:
            logger.error("API timeout po %ss: %s", REQUEST_TIMEOUT_S, exc)
            st.error(
                f"API neodpovědělo do {REQUEST_TIMEOUT_S} sekund. "
                "Server může být přetížený nebo nedostupný, zkuste to znovu."
            )
            st.stop()
        except requests.HTTPError as exc:
            status_code = exc.response.status_code
            body_text = exc.response.text
            logger.error("API vratil HTTP %s: %s", status_code, body_text)
            if status_code == 422:
                try:
                    details = exc.response.json().get("detail", [])
                except ValueError:
                    details = []
                if details:
                    st.error("API odmítlo požadavek (422 — validační chyba):")
                    for d in details:
                        loc_parts = [
                            str(x) for x in d.get("loc", []) if x != "body"
                        ]
                        loc_str = " > ".join(loc_parts) if loc_parts else "request"
                        msg = d.get("msg", "(bez popisu)")
                        st.error(f"- {loc_str}: {msg}")
                else:
                    st.error(f"API odmítlo požadavek (422): {body_text}")
            else:
                st.error(f"API chyba {status_code}: {body_text}")
            st.stop()
        except requests.RequestException as exc:
            logger.error(
                "API request selhal: %s: %s", type(exc).__name__, exc
            )
            st.error(
                f"Síťová chyba při volání API ({type(exc).__name__}): {exc}"
            )
            st.stop()

    try:
        data = resp.json()
    except ValueError as exc:
        body_preview = (resp.text or "")[:500] if resp is not None else ""
        logger.error("Server vratil neJSON odpoved: %s", body_preview)
        st.error(
            "API vrátilo neplatnou odpověď (není JSON). "
            "Server pravděpodobně neběží správně — zkontrolujte logy."
        )
        st.stop()

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

    with st.expander("Raw API odpověď (JSON)"):
        st.json(data)
