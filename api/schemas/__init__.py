"""Umožňuje, aby původní importy fungovaly i po změnách."""

from api.schemas.request import ProductInput
from api.schemas.response import (
    SimilarProduct,
    PredictionResponse,
    HealthResponse,
    CategoriesResponse,
)

__all__ = [
    "ProductInput",
    "SimilarProduct",
    "PredictionResponse",
    "HealthResponse",
    "CategoriesResponse",
]
