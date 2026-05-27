import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def mase(y_true, y_pred, y_train, season_length: int) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    scale = np.mean(np.abs(y_train[season_length:] - y_train[:-season_length]))
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def evaluate_forecasts(test: pd.DataFrame, forecasts: pd.DataFrame, train: pd.DataFrame, season_length: int) -> pd.DataFrame:
    rows = []
    actual = test["y"].to_numpy()
    train_y = train["y"].to_numpy()
    for model in [c for c in forecasts.columns if c not in {"unique_id", "ds"}]:
        pred = forecasts[model].to_numpy()
        rows.append(
            {
                "model": model,
                "MAE": mae(actual, pred),
                "RMSE": rmse(actual, pred),
                "MAPE": mape(actual, pred),
                "sMAPE": smape(actual, pred),
                "MASE": mase(actual, pred, train_y, season_length),
            }
        )
    return pd.DataFrame(rows).sort_values("sMAPE")
