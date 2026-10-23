import os
import sys
from pathlib import Path

import boto3
import pandas as pd
import requests
from botocore import UNSIGNED
from botocore.config import Config

from trial_conversion_model.cohorts import pull_newest_cohort
from trial_conversion_model.monitoring import (
    REPORT_DIR,
    load_reference,
    publish_report,
    run_drift_check,
)


def alert(message: str) -> None:
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook:
        print("SLACK_WEBHOOK_URL not set; alert printed only")
        return
    requests.post(webhook, json={"text": message}, timeout=10)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        # sys.exit("usage: uv run scripts/check_drift.py <cohort file>")
        cohort_path = pull_newest_cohort()
    else:
        cohort_path = Path(sys.argv[1])

    if not cohort_path.exists():
        s3 = boto3.client(
            "s3", region_name="eu-north-1", config=Config(signature_version=UNSIGNED)
        )
        cohort_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(
            Bucket="fpds-trial-cohorts",
            Key=f"cohorts/{cohort_path.name}",
            Filename=str(cohort_path),
        )

    cur = pd.read_parquet(cohort_path)
    snapshot = run_drift_check(cur, load_reference())
    result = publish_report(snapshot=snapshot, name=f"drift_{cohort_path.stem}")

    pct_drift = result["drifted_share"]

    verdict = "DRIFT" if result["drift"] else "OK"
    print(f"{verdict}: {pct_drift:.0%} of features drifted ({cohort_path.name})")

    drift_msg = f"Drift alert: {result['drifted_share']:.0%} of features drifted in {cohort_path.stem}. Latest report in S3 under {REPORT_DIR}/."
    if verdict == "DRIFT":
        alert(message=drift_msg)
