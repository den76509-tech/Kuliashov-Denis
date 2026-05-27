import numpy as np
import pandas as pd

from .config import SEASON_LENGTH


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["year"] = result["ds"].dt.year
    result["month"] = result["ds"].dt.month
    result["quarter"] = result["ds"].dt.quarter
    result["t"] = np.arange(len(result))
    result["log_y"] = np.log(result["y"])
    result["diff_1"] = result["y"].diff()
    result["diff_12"] = result["y"].diff(SEASON_LENGTH)
    result["log_diff_1"] = result["log_y"].diff()
    result["log_diff_12"] = result["log_y"].diff(SEASON_LENGTH)
    return result


def seasonality_profile(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(month=df["ds"].dt.month)
        .groupby("month")["y"]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
    )


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    y = df["y"]
    return pd.DataFrame(
        [
            {
                "rows": len(df),
                "start": df["ds"].min().date().isoformat(),
                "end": df["ds"].max().date().isoformat(),
                "mean": y.mean(),
                "median": y.median(),
                "std": y.std(),
                "min": y.min(),
                "max": y.max(),
                "missing_y": int(y.isna().sum()),
            }
        ]
    )


def stationarity_notes(df: pd.DataFrame) -> dict:
    enriched = add_time_features(df)
    return {
        "raw_variation": float(enriched["y"].std() / enriched["y"].mean()),
        "log_variation": float(enriched["log_y"].std() / enriched["log_y"].mean()),
        "seasonal_diff_std": float(enriched["diff_12"].dropna().std()),
        "log_seasonal_diff_std": float(enriched["log_diff_12"].dropna().std()),
        "recommendation": "Use log transform plus seasonal differencing for statistical diagnostics.",
    }
