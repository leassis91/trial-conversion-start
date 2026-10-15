import sys
from pathlib import Path

import pandas as pd

from trial_conversion_model.monitoring import (
    load_reference,
    publish_report,
    run_drift_check,
)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: uv run scripts/check_drift.py <cohort file>")
    cohort_path = Path(sys.argv[1])
    # TODO: This script needs imports that are not in this file yet, both
    # from your own package and from pandas. Add them as you go.
    #
    # Read the cohort file, check it for drift against the reference,
    # then publish the report. Name it after the cohort file, so you can tell
    # one report from another later. Call what you get back `result`.

    cur = pd.read_parquet(cohort_path)
    snapshot = run_drift_check(cur, load_reference())
    result = publish_report(snapshot=snapshot, name=f"drift_{cohort_path.stem}")

    verdict = "DRIFT" if result["drift"] else "OK"
    print(
        f"{verdict}: {result['drifted_share']:.0%} of features drifted ({cohort_path.name})"
    )
