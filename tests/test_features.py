import pandas as pd

from trial_conversion_model.features import add_features

pd.set_option("future.no_silent_downcasting", True)

VALID_TRIAL = {
    "sessions_day1": None,
    "sessions_day2": None,
    "sessions_day3": None,
    "listen_sessions_3d": None,
    "total_minutes_3d": None,
    "country": "US",
    "device_type": "iOS",
}


def test_zero_session_trial_gets_zero_shares_not_nan():

    onerowdf = pd.DataFrame(VALID_TRIAL, index=[0])
    df = add_features(onerowdf)

    assert df["day1_share"].iloc[0] == 0
    assert df["listen_share"].iloc[0] == 0
    assert df["avg_session_minutes"].iloc[0] == 0
