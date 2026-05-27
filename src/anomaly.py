import numpy as np
import pandas as pd

from .config import SEASON_LENGTH


def seasonal_difference_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    result = df.copy()
    result["score"] = result["y"].diff(SEASON_LENGTH)
    mu = result["score"].mean()
    sigma = result["score"].std()
    result["anomaly_score"] = (result["score"] - mu).abs() / sigma
    result["method"] = "seasonal_diff_zscore"
    result["is_anomaly"] = result["anomaly_score"] > threshold
    return result[["unique_id", "ds", "y", "method", "anomaly_score", "is_anomaly"]]


def robust_iqr(df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    result = df.copy()
    residual = result["y"] - result.groupby(result["ds"].dt.month)["y"].transform("median")
    q1, q3 = residual.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    result["method"] = "monthly_robust_iqr"
    result["anomaly_score"] = residual.abs()
    result["is_anomaly"] = (residual < lower) | (residual > upper)
    return result[["unique_id", "ds", "y", "method", "anomaly_score", "is_anomaly"]]


def isolation_forest(df: pd.DataFrame, contamination: float = 0.03) -> pd.DataFrame:
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError as exc:
        raise ImportError(
            "Install scikit-learn before running IsolationForest: pip install -r requirements.txt"
        ) from exc

    features = pd.DataFrame(
        {
            "y": df["y"],
            "month": df["ds"].dt.month,
            "log_y": np.log(df["y"]),
            "diff_12": df["y"].diff(SEASON_LENGTH).fillna(0),
        }
    )
    model = IsolationForest(contamination=contamination, random_state=42)
    labels = model.fit_predict(features)
    result = df.copy()
    result["method"] = "isolation_forest"
    result["anomaly_score"] = -model.score_samples(features)
    result["is_anomaly"] = labels == -1
    return result[["unique_id", "ds", "y", "method", "anomaly_score", "is_anomaly"]]


def run_anomaly_suite(df: pd.DataFrame) -> pd.DataFrame:
    outputs = [seasonal_difference_zscore(df), robust_iqr(df)]
    try:
        outputs.append(isolation_forest(df))
    except ImportError:
        pass
    return pd.concat(outputs, ignore_index=True)
