import pandas as pd

from .config import FREQ, HORIZON, SEASON_LENGTH


def fit_predict_neuralforecast(train: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Forecast with neuralforecast deep-learning models."""
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import MLP, NBEATS, NHITS
    except ImportError as exc:
        raise ImportError(
            "Install project requirements before running DL models: "
            "pip install -r requirements.txt"
        ) from exc

    models = [
        MLP(h=horizon, input_size=2 * SEASON_LENGTH, max_steps=500, random_seed=42),
        NBEATS(h=horizon, input_size=2 * SEASON_LENGTH, max_steps=500, random_seed=42),
        NHITS(h=horizon, input_size=2 * SEASON_LENGTH, max_steps=500, random_seed=42),
    ]
    nf = NeuralForecast(models=models, freq=FREQ)
    nf.fit(df=train)
    return nf.predict()


def explain_dl_choice() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["MLP", "Нейросетевой бейзлайн: проверяет, дает ли нелинейная функция лагов выигрыш."],
            ["NBEATS", "Модель для трендово-сезонных рядов без ручной спецификации сезонности."],
            ["NHITS", "Иерархическая архитектура, часто сильна на средних и длинных горизонтах."],
        ],
        columns=["model", "rationale"],
    )
