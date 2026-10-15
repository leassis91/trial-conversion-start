from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

RAW_COHORTS = Path("data/01_raw/cohorts")

if __name__ == "__main__":
    s3 = boto3.client(
        "s3", region_name="eu-north-1", config=Config(signature_version=UNSIGNED)
    )
    bucket_cohorts = "fpds-trial-cohorts"
    cohort1 = "cohorts/cohort_2026-09-13.parquet"
    cohort2 = "cohorts/cohort_2026-10-03.parquet"

    RAW_COHORTS.mkdir(parents=True, exist_ok=True)

    # Download the file
    s3.download_file(
        Bucket=bucket_cohorts, Key=cohort1, Filename=Path(RAW_COHORTS / cohort1[8:])
    )
    s3.download_file(
        Bucket=bucket_cohorts, Key=cohort2, Filename=Path(RAW_COHORTS / cohort2[8:])
    )
