# AI modul pro doporučování typu skladové lokace pro nové zboží

## Úvod

Bakalářská práce řeší problém naskladňování nového zboží bez historie objednávek (tzv. cold start problem). Projekt vytváří AI modul, který na základě fyzických atributů produktu (hmotnost, rozměry, cena, kategorie) doporučí jednu z 5 typů skladových lokací. Model XGBoost v plném režimu dosahuje F1 skóre 98,7 %. V cold-start režimu bez obrátkovosti dokáže u 60 % rychloobrátkových produktů správně rozpoznat, že patří do přední zóny.

Datový zdroj: **Olist Brazilian E-Commerce**: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce. 

Systém je nasaditelný jako REST API (FastAPI) se Streamlit klientem nad ním.

---

## Stromová struktura projektu

```
bakalarka/
├── pyproject.toml              # závislosti projektu (uv)
├── uv.lock                 
├── Dockerfile                  # kontejnerizace API
├── .dockerignore
├── .gitignore
├── README.md
│
├── api/                        # REST API modul (FastAPI)
│   ├── __init__.py
│   ├── constants.py            # sdílené konstanty (kategorie, názvy tříd)
│   ├── main.py                 # HTTP brána — endpointy
│   ├── services.py             # byznys logika (predikce, KNN, vysvětlení, audit)
│   └── schemas/                # Pydantic modely
│       ├── __init__.py
│       ├── request.py          # vstupní model (validace)
│       └── response.py         # výstupní modely
│
├── app/                        # Streamlit prototypy
│   ├── __init__.py
│   ├── prototype_api.py        # hlavní prototyp — klient volající API přes HTTP
│   └── prototype_backup.py     # záloha — přímý přístup k modelům bez API
│
├── src/                        # znovupoužitelné funkce, výpočet
│   ├── __init__.py
│   ├── data_preparation.py     # načtení a propojení surových dat
│   ├── cleaning.py             # čištění a seskupení kategorií
│   ├── labeling.py             # pravidlové přiřazení skladových tříd
│   ├── features.py             # preprocessing pipeline
│   ├── models.py               # trénink a vyhodnocení modelů
│   ├── noise_experiment.py     # experiment s šumem
│   ├── error_severity.py       # analýza závažnosti chyb
│   ├── similarity.py           # KNN vyhledávání podobných produktů
│   ├── shap_analysis.py        # SHAP interpretace modelu
│   └── cold_start.py           # cold start simulace bez obrátkovosti
│
├── scripts/                    # spouštěcí skripty jednotlivých fází
│   ├── run_phase1.py
│   ├── run_phase2.py
│   ├── run_phase3.py
│   ├── run_phase4.py
│   ├── run_phase5.py
│   ├── run_phase5b.py
│   ├── run_phase5c.py
│   ├── run_phase6.py
│   ├── run_phase7.py
│   ├── run_phase8.py
│   ├── run_phase9.py
│   ├── run_phase9_prepare.py
│   ├── run_phase9_validation.py
│   └── verify_api.py           # integrační testy API (19 kontrol)
│
├── notebooks/                  # interaktivní analýza (Jupyter)
│   ├── 01_eda.ipynb 
│   ├── 02_cleaning.ipynb
│   ├── 03_labeling.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_modeling.ipynb
│   ├── 05b_noise_experiment.ipynb
│   ├── 05c_error_severity.ipynb
│   ├── 06_knn_similarity.ipynb
│   ├── 07_shap_analysis.ipynb
│   ├── 08_cold_start.ipynb
│   └── 09_prototype_validation.ipynb
│
├── models/                     # natrénované modely (.joblib)
│   ├── best_model.joblib                    # XGBoost plný
│   ├── best_model_no_turnover.joblib        # XGBoost cold-start (bez obrátkovosti)
│   ├── dt_model.joblib                      # Decision Tree
│   ├── rf_model.joblib                      # Random Forest
│   ├── xgb_model.joblib                     # XGBoost (varianta)
│   ├── knn_model.joblib                     # KNN plný
│   ├── knn_model_no_turnover.joblib         # KNN cold-start
│   ├── label_encoder_no_turnover.joblib     # LabelEncoder pro cold-start (mapování tříd na čísla)
│   ├── preprocessing_pipeline_standard.joblib  # preprocessing pro stromové modely (StandardScaler + OHE)
│   └── preprocessing_pipeline_minmax.joblib    # preprocessing pro KNN (MinMaxScaler + OHE)
│
├── data/
│   └── processed/              # zpracovaná data
│       ├── products_clean.csv
│       ├── products_processed.csv
│       ├── products_labeled.csv
│       └── features_final.csv
│
└── results/                    # grafy a tabulky
    ├── phase1_eda/             
    ├── phase2_cleaning/
    ├── phase3_labeling/
    ├── phase4_features/
    ├── phase5_modeling/
    ├── phase5b_noise/
    ├── phase5c_severity/
    ├── phase6_knn/
    ├── phase7_shap/
    └── phase8_cold_start/
```

---

## Fáze projektu

| Fáze | Název | Stručný popis |
|------|-------|---------------|
| **1** | Načtení dat | Propojení 9 CSV z Olist datasetu, překlad kategorií, výpočet obrátkovosti |
| **2** | Čištění dat | Doplnění chybějících hodnot, seskupení 74 kategorií do 13 skupin |
| **3** | Labeling | Pravidlové přiřazení 5 typů skladových lokací (shelf, front, pallet, floor, special) |
| **4** | Pipeliny | Výběr 10 číselných + 1 příznaků, preprocessing pipeline |
| **5** | Trénink modelů | Decision Tree, Random Forest, XGBoost |
| **5b** | Experiment s šumem | Odolnost modelu proti 10/15/20 % chybným labelům |
| **5c** | Závažnost chyb | Matice závažnosti záměn|
| **6** | KNN podobnost | Validační vrstva, srovnání KNN sousedů s predikcí XGBoost |
| **7** | SHAP interpretace | Vysvětlení predikcí |
| **8** | Cold-start simulace | ML vs pravidla u nových produktů bez historie |
| **9** | Prototyp | Streamlit aplikace + REST API (FastAPI) + validace 10 produktů |
| **10** | Rozšíření API | Batch endpoint `/predict/batch` |

---

## Požadavky

- **Python 3.13+**
- **uv** package manager
- **Docker** (volitelně, pro kontejnerový běh API)

## Instalace

```bash
cd bakalarka
uv sync
```

---

## Spuštění

### A) Docker kontejner s API + Streamlit UI

```bash
# Terminál 1: API v Docker kontejneru
cd bakalarka
docker build -t warehouse-api .
docker run -p 8000:8000 warehouse-api

# Terminál 2: Streamlit klient (volá API)
cd bakalarka
uv run streamlit run app/prototype_api.py
```

- API dostupné na: **http://localhost:8000**
- Swagger UI: **http://localhost:8000/docs**
- Streamlit klient: **http://localhost:8501**

### B) Lokální server (pro vývoj) + Streamlit UI

```bash
# Terminál 1: API přímo přes uvicorn
cd bakalarka
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000

# Terminál 2: Streamlit klient
cd bakalarka
uv run streamlit run app/prototype_api.py
```


### C) Záložní prototyp (bez API)

Pokud by REST API nefungovalo, je k dispozici monolit s přímým načtením modelů:

```bash
cd bakalarka
uv run streamlit run app/prototype_backup.py
```

---

## Spouštění skriptů, notebooků, testů


**Celý postup od nuly:**
```bash
uv run python scripts/run_phase1.py
uv run python scripts/run_phase2.py
uv run python scripts/run_phase3.py
uv run python scripts/run_phase4.py
uv run python scripts/run_phase5.py
uv run python scripts/run_phase5b.py
uv run python scripts/run_phase5c.py
uv run python scripts/run_phase6.py
uv run python scripts/run_phase7.py
uv run python scripts/run_phase8.py
uv run python scripts/run_phase9_prepare.py
uv run python scripts/run_phase9_validation.py
uv run python scripts/run_business_value.py
```

**API integrační testy:**
```bash
# API musí běžet na pozadí
uv run python scripts/verify_api.py
```


**Jupyter Lab:**
```bash
cd bakalarka
uv run jupyter lab
```
