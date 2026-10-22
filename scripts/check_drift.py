import sys
from pathlib import Path

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

from trial_conversion_model.monitoring import (
    load_reference,
    publish_report,
    run_drift_check,
)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: uv run scripts/check_drift.py <cohort file>")
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

    verdict = "DRIFT" if result["drift"] else "OK"
    print(
        f"{verdict}: {result['drifted_share']:.0%} of features drifted ({cohort_path.name})"
    )
