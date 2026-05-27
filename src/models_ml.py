import pandas as pd

from .config import FREQ, SEASON_LENGTH


def fit_predict_mlforecast(train: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forecast with feature-based machine-learning models."""
    try:
        from lightgbm import LGBMRegressor
        from mlforecast import MLForecast
        from mlforecast.lag_transforms import ExpandingMean, RollingMean
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
    except ImportError as exc:
        raise ImportError(
            "Install project requirements before running ML models: "
            "pip install -r requirements.txt"
        ) from exc

    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=500, max_depth=6, random_state=42),
        "LightGBM": LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            random_state=42,
            verbosity=-1,
        ),
    }
    fcst = MLForecast(
        models=models,
        freq=FREQ,
        lags=[1, 2, 3, 6, 12, 24],
        lag_transforms={
            1: [RollingMean(window_size=3), RollingMean(window_size=6), ExpandingMean()],
            12: [RollingMean(window_size=3)],
        },
        date_features=["month", "quarter", "year"],
    )
    fcst.fit(train)
    return fcst.predict(horizon)


def explain_ml_choice() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Ridge", "Линейная регуляризованная модель на лагах; нужна как интерпретируемый ML-бейзлайн."],
            ["RandomForest", "Нелинейная модель, устойчива к выбросам и малым данным при ограниченной глубине."],
            ["LightGBM", "Градиентный бустинг хорошо работает с календарными признаками и лагами."],
        ],
        columns=["model", "rationale"],
    )
