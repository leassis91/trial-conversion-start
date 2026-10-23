import json
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from trial_conversion_model.monitoring import REPORT_DIR

load_dotenv()
PREDICTIONS_PREFIX = "predictions/scored_"
PREDICTIONS_DIR = Path("data/04_predictions")
METRICS_FILE = Path("models/metrics.json")


def get_newest_drift_status() -> dict:
    """Return the contents of the most recent drift report."""
    files = sorted(REPORT_DIR.glob("drift_*.json"))
    if not files:
        raise FileNotFoundError(
            f"no drift report files in {REPORT_DIR}; run scripts/check_drift.py first"
        )
    with open(files[-1], "r", encoding="utf-8") as f:
        return json.load(f)


def get_newest_scored() -> pd.DataFrame:
    """Return the most recent scored cohort."""
    files = sorted(PREDICTIONS_DIR.glob("*.csv"))
    if not files:
        st.warning("no scored file found.")
        st.stop()
    return pd.read_csv(files[-1])


def get_test_auc() -> float:
    """Return the test AUC of the published model."""
    if not METRICS_FILE.exists():
        st.warning("no metrics file found; run train.py first.")
        st.stop()
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["test_auc"]


drift_status = get_newest_drift_status()
scored = get_newest_scored()

#################################################################################################
# FIRST LINE OF THE PAGE
#################################################################################################

st.title("Data Drift Monitoring Page")

# Create 3 columns for side-by-side KPI cards
col1, col2, col3 = st.columns(3)

first_col = drift_status["name"].removeprefix("drift_cohort_")
second_col = "DRIFT" if drift_status["drift"] else "OK"
pct_drift = drift_status["drifted_share"]
third_col = get_test_auc()


col1.metric(
    "Latest scored cohort",
    first_col,
    f"{len(scored)} trials, scored yesterday",
    delta_color="off",
    delta_arrow="off",
)
col2.metric(
    "Latest drift verdict",
    second_col,
    f"{pct_drift:.0%} of features",
    delta_color="off",
    delta_arrow="off",
)
col3.metric(
    "Model test AUC",
    third_col,
    "published 2026-08-03",
    delta_color="off",
    delta_arrow="off",
)
