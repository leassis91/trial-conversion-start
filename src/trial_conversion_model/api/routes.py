import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import APIRouter

from trial_conversion_model.api.schemas import PredictionRequest, PredictionResponse
from trial_conversion_model.predict import load_model, predict_proba

logger = logging.getLogger(__name__)

router = APIRouter()

LOGS_DATA = Path("logs/")

# Load the model once, when the service starts, not on every request.
model = load_model()


def to_band(probability: float) -> str:
    """Turn a raw probability into a label a human can act on."""
    if probability < 0.33:
        return "low"
    if probability < 0.66:
        return "medium"
    return "high"


def to_jsonl(
    request: PredictionRequest,
    response_model: dict,
    out_path: Path = LOGS_DATA,
) -> None:
    """Materialize the prediction extract into data/01_raw."""
    out_path.mkdir(parents=True, exist_ok=True)
    # Initiliaze JSONL entry
    data = {"timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f%z")}
    data.update(request.model_dump())
    data.update(response_model)

    print(data)

    with open(LOGS_DATA / "predictions.jsonl", "a") as file:
        file.write(json.dumps(data) + "\n")


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.4.0"}


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict conversion probability for a single live trial."""

    onerow_df = pd.DataFrame([request.model_dump()])  # 1
    conversion_probability_series = predict_proba(model, onerow_df)
    conversion_probability = round(conversion_probability_series.item(), 4)  # 2
    conversion_band = to_band(conversion_probability)  # 3

    models_response = {
        "conversion_probability": conversion_probability,
        "conversion_band": conversion_band,
    }

    logger.info(
        "prediction | %s -> probability=%.4f band=%s",
        request.model_dump(),
        conversion_probability,
        conversion_band,
    )

    to_jsonl(request, models_response)

    return models_response
