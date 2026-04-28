"""FastAPI endpointy, REST API."""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

_API_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _API_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, Depends, HTTPException, Header

from api.constants import CATEGORY_GROUPS
from api.schemas import (
    ProductInput,
    PredictionResponse,
    HealthResponse,
    CategoriesResponse,
    BatchPredictRequest,
    BatchPredictResponse,
)
from api.services import PredictionService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

API_KEY = os.environ.get("API_KEY", None)


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if API_KEY is None:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Neplatný API klíč")


service: PredictionService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # nacte modely pri startu, cleanup pri shutdown.

    global service
    logger.info("Inicializuji PredictionService...")
    service = PredictionService()
    logger.info("API server ready.")
    yield
    logger.info("API server shutting down.")


app = FastAPI(
    title="Warehouse Slotting AI",
    description="REST API pro doporučení typu skladové lokace pro nové zboží",
    version="1.0.0",
    lifespan=lifespan,
)


# Endpointy
@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_key)],
)
def predict(product: ProductInput):
    return service.predict(product)


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    dependencies=[Depends(verify_api_key)],
)
def predict_batch(request: BatchPredictRequest):
    return service.predict_batch(request)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model="xgboost",
        version="1.0.0",
        cold_start_available=True,
    )

@app.get("/categories", response_model=CategoriesResponse)
def categories():
    return CategoriesResponse(categories=CATEGORY_GROUPS)
