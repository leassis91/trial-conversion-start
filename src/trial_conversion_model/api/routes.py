import logging

import pandas as pd
from fastapi import APIRouter
from pydantic import ValidationError

from trial_conversion_model.api.schemas import PredictionRequest, PredictionResponse
from trial_conversion_model.predict import load_model, predict_proba

logger = logging.getLogger(__name__)

router = APIRouter()

# Load the model once, when the service starts, not on every request.
model = load_model()


def to_band(probability: float) -> str:
    """Turn a raw probability into a label a human can act on."""
    if probability < 0.33:
        return "low"
    if probability < 0.66:
        return "medium"
    return "high"


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict conversion probability for a single live trial."""
    # TODO: three steps, roughly one line each:
    #   1. Turn the request into a one-row DataFrame (model_dump gives you a dict).
    #   2. Score it with predict_proba and round to 4 decimals.
    #   3. Turn the probability into a band, and return the PredictionResponse.

    onerow_df = pd.DataFrame([request.model_dump()])  # 1
    conversion_probability_series = predict_proba(model, onerow_df)
    conversion_probability = round(conversion_probability_series.item(), 4)  # 2
    conversion_band = to_band(conversion_probability)  # 3

    return {
        "conversion_probability": conversion_probability,
        "conversion_band": conversion_band,
    }
