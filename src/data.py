import pandas as pd

from .config import FREQ, HORIZON, PROCESSED_DATA_PATH, RAW_DATA_PATH, SERIES_ID


def load_raw(path=RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    return df


def validate_series(df: pd.DataFrame) -> dict:
    expected_index = pd.date_range(df["ds"].min(), df["ds"].max(), freq=FREQ)
    missing_timestamps = expected_index.difference(df["ds"])
    duplicate_timestamps = int(df["ds"].duplicated().sum())
    return {
        "rows": int(len(df)),
        "start": df["ds"].min().date().isoformat(),
        "end": df["ds"].max().date().isoformat(),
        "missing_values": int(df["y"].isna().sum()),
        "duplicate_timestamps": duplicate_timestamps,
        "missing_timestamps": [d.date().isoformat() for d in missing_timestamps],
        "freq": FREQ,
    }


def prepare_series(save: bool = True) -> pd.DataFrame:
    df = load_raw().sort_values("ds").reset_index(drop=True)
    df["unique_id"] = SERIES_ID
    df = df[["unique_id", "ds", "y"]]
    if save:
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DATA_PATH, index=False)
    return df


def train_test_split(df: pd.DataFrame, horizon: int = HORIZON) -> tuple[pd.DataFrame, pd.DataFrame]:
    if horizon <= 0 or horizon >= len(df):
        raise ValueError("horizon must be positive and smaller than the series length")
    return df.iloc[:-horizon].copy(), df.iloc[-horizon:].copy()
