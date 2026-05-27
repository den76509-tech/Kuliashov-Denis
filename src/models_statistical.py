import pandas as pd

from .config import FREQ, SEASON_LENGTH


def fit_predict_statsforecast(train: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forecast with baseline and statistical models from statsforecast."""
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import (
            ARIMA,
            AutoARIMA,
            AutoETS,
            AutoTheta,
            HistoricAverage,
            Naive,
            SeasonalNaive,
            Theta,
        )
    except ImportError as exc:
        raise ImportError(
            "Install project requirements before running statistical models: "
            "pip install -r requirements.txt"
        ) from exc

    models = [
        Naive(),
        SeasonalNaive(season_length=SEASON_LENGTH),
        HistoricAverage(),
        ARIMA(order=(0, 1, 1), seasonal_order=(0, 1, 1), season_length=SEASON_LENGTH),
        AutoARIMA(season_length=SEASON_LENGTH),
        AutoETS(season_length=SEASON_LENGTH),
        Theta(season_length=SEASON_LENGTH),
        AutoTheta(season_length=SEASON_LENGTH),
    ]
    sf = StatsForecast(models=models, freq=FREQ, n_jobs=-1)
    return sf.forecast(df=train, h=horizon)


def explain_statistical_choice() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Naive", "baseline", "Контрольная точка: последний наблюдаемый уровень."],
            ["SeasonalNaive", "baseline", "Сильный сезонный бейзлайн для месячных данных."],
            ["ARIMA(0,1,1)(0,1,1)[12]", "manual", "Ручная SARIMA после логики сезонного дифференцирования."],
            ["AutoARIMA", "auto", "Автоподбор порядков ARIMA по информационному критерию."],
            ["AutoETS", "auto", "Подбирает тренд/сезонность экспоненциального сглаживания."],
            ["Theta", "manual", "Классическая Theta-модель как компактная трендовая альтернатива."],
            ["AutoTheta", "auto", "Автоматический выбор варианта Theta."],
        ],
        columns=["model", "mode", "rationale"],
    )
