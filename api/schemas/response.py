"""Pydantic modely."""

from pydantic import BaseModel

class SimilarProduct(BaseModel):
    rank: int
    distance: float
    weight_g: float
    volume_cm3: float
    category: str
    storage_class: str
    storage_class_cz: str


class PredictionResponse(BaseModel):
    recommended_zone: str
    recommended_zone_cz: str
    confidence: float
    all_probabilities: dict[str, float]
    similar_products: list[SimilarProduct]
    knn_agreement: str
    explanation: str
    cold_start_mode: bool
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str
    cold_start_available: bool


class CategoriesResponse(BaseModel):
    categories: list[str]
