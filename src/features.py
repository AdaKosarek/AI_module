"""Odvozene priznaky, kodovani, skalovani"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
    "volume_cm3",
    "volumetric_density",
    "price_per_kg",
    "avg_price",
    "avg_freight",
    "daily_turnover",
]

CATEGORICAL_FEATURES = ["category_group"]
TARGET = "storage_class"


def compute_daily_turnover(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["first_order_date"] = pd.to_datetime(df["first_order_date"], errors="coerce")
    df["last_order_date"] = pd.to_datetime(df["last_order_date"], errors="coerce")

    # osetreni deleni nulou pro 1 den
    days_active = (df["last_order_date"] - df["first_order_date"]).dt.days
    days_active = days_active.clip(lower=1)

    df["daily_turnover"] = df["order_count"] / days_active

    df.loc[df["order_count"] == 0, "daily_turnover"] = 0.0

    mask_nan_dates = df["first_order_date"].isna() | df["last_order_date"].isna()
    df.loc[mask_nan_dates, "daily_turnover"] = 0.0

    n_positive = int((df["daily_turnover"] > 0).sum())
    logger.info("Produktu s daily_turnover > 0: %d", n_positive)

    return df


def select_features(df: pd.DataFrame) -> tuple:
    if "daily_turnover" not in df.columns:
        df = compute_daily_turnover(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


def build_preprocessing_pipeline(scaler_type: str = "standard") -> ColumnTransformer:
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Neznamy scaler_type: {scaler_type}")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor

##
def create_feature_matrix(
    df: pd.DataFrame, scaler_type: str = "standard"
) -> tuple:
    X, y = select_features(df)
    pipeline = build_preprocessing_pipeline(scaler_type)
    X_transformed = pipeline.fit_transform(X)

    ohe = pipeline.named_transformers_["categorical"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = list(NUMERIC_FEATURES) + cat_names

    return X_transformed, y, pipeline, feature_names


def feature_engineering_pipeline(
    input_path: str = "data/processed/products_labeled.csv",
    output_path: str = "data/processed/features_final.csv",
    models_dir: str = "models",
) -> dict:
    input_path = Path(input_path)
    output_path = Path(output_path)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Nacitam %s", input_path)
    df = pd.read_csv(input_path)
    n_rows_input = len(df)

    df = compute_daily_turnover(df)
    X_raw, y = select_features(df)

    num_imputer = SimpleImputer(strategy="median")
    X_numeric = pd.DataFrame(
        num_imputer.fit_transform(X_raw[NUMERIC_FEATURES]),
        columns=NUMERIC_FEATURES,
        index=X_raw.index,
    )

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = ohe.fit_transform(X_raw[CATEGORICAL_FEATURES])
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    X_cat_df = pd.DataFrame(X_cat, columns=cat_names, index=X_raw.index)

    df_final = pd.concat([X_numeric, X_cat_df], axis=1)
    df_final[TARGET] = y.values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)
    logger.info("Ulozen %s (%d radku, %d sloupcu)", output_path, len(df_final), len(df_final.columns))

    # Ulozeni obou pipeline
    for stype in ("standard", "minmax"):
        _, _, pipeline, _ = create_feature_matrix(df, scaler_type=stype)
        joblib_path = models_dir / f"preprocessing_pipeline_{stype}.joblib"
        joblib.dump(pipeline, joblib_path)
        logger.info("Ulozen %s", joblib_path)

    stats = {
        "n_rows_input": n_rows_input,
        "n_rows_output": len(df_final),
        "n_features": len(df_final.columns) - 1,
        "n_numeric": len(NUMERIC_FEATURES),
        "n_ohe": len(cat_names),
        "output_path": str(output_path),
        "pipelines_saved": [
            str(models_dir / "preprocessing_pipeline_standard.joblib"),
            str(models_dir / "preprocessing_pipeline_minmax.joblib"),
        ],
    }
    return stats
