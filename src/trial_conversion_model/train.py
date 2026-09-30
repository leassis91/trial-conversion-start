import json
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from trial_conversion_model.data import load_processed
from trial_conversion_model.features import TARGET

load_dotenv()

MODEL_DIR = Path("models")
TEST_SIZE = 0.25
RANDOM_STATE = 42

PARAMS = {
    "n_estimators": 1400,
    "max_depth": 3,
    "learning_rate": 0.01,
    "min_child_weight": 8,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "eval_metric": "auc",
}


def train(model_dir: Path = MODEL_DIR) -> dict:
    """Train the trial conversion model from the processed training table."""
    table = load_processed()
    X = table.drop(columns=[TARGET])
    y = table[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    mlflow.set_experiment("trial-conversion-model")  # pick or create the group

    with mlflow.start_run():  # open a run, close it on exit
        mlflow.log_params(PARAMS)
        model = XGBClassifier(**PARAMS)
        model.fit(X_train, y_train)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        mlflow.log_metric("test_auc", auc)  # a number you measured
        mlflow.xgboost.log_model(model, name="model")  # the model itself

    model_dir.mkdir(exist_ok=True)
    model.save_model(model_dir / "model.json")
    metrics = {
        "test_auc": round(float(auc), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": list(X.columns),
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
