from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from .anomaly import run_anomaly_suite
from .config import HORIZON, REPORT_TABLES_DIR, SEASON_LENGTH
from .data import prepare_series, train_test_split, validate_series
from .eda import add_time_features, seasonality_profile, stationarity_notes, summary_table
from .metrics import evaluate_forecasts


def run_prepare() -> dict:
    df = prepare_series(save=True)
    checks = validate_series(df)
    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary_table(df).to_csv(REPORT_TABLES_DIR / "eda_summary.csv", index=False)
    seasonality_profile(df).to_csv(REPORT_TABLES_DIR / "seasonality_profile.csv", index=False)
    add_time_features(df).to_csv(REPORT_TABLES_DIR / "eda_features_preview.csv", index=False)
    with open(REPORT_TABLES_DIR / "data_quality.json", "w", encoding="utf-8") as fh:
        json.dump(checks, fh, ensure_ascii=False, indent=2)
    with open(REPORT_TABLES_DIR / "stationarity_notes.json", "w", encoding="utf-8") as fh:
        json.dump(stationarity_notes(df), fh, ensure_ascii=False, indent=2)
    try:
        from .visualization import generate_report_figures

        generate_report_figures()
    except ImportError:
        pass
    return checks


def run_baseline_backtest() -> pd.DataFrame:
    df = prepare_series(save=True)
    train, test = train_test_split(df, HORIZON)
    future = test[["unique_id", "ds"]].copy()
    future["Naive"] = train["y"].iloc[-1]
    seasonal_values = train["y"].iloc[-SEASON_LENGTH:].to_numpy()
    future["SeasonalNaive"] = [seasonal_values[i % SEASON_LENGTH] for i in range(HORIZON)]
    future["HistoricAverage"] = train["y"].mean()
    metrics = evaluate_forecasts(test, future, train, SEASON_LENGTH)
    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(REPORT_TABLES_DIR / "quick_baseline_metrics.csv", index=False)
    return metrics


def run_full_pipeline() -> dict[str, pd.DataFrame]:
    from .models_dl import fit_predict_neuralforecast
    from .models_ml import fit_predict_mlforecast
    from .models_statistical import fit_predict_statsforecast

    df = prepare_series(save=True)
    train, test = train_test_split(df, HORIZON)
    outputs = {}

    start = time.perf_counter()
    stat_fcst = fit_predict_statsforecast(train, HORIZON)
    outputs["statistical"] = evaluate_forecasts(test, stat_fcst, train, SEASON_LENGTH)
    outputs["statistical"]["seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    ml_fcst = fit_predict_mlforecast(train, HORIZON)
    outputs["ml"] = evaluate_forecasts(test, ml_fcst, train, SEASON_LENGTH)
    outputs["ml"]["seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    dl_fcst = fit_predict_neuralforecast(train, HORIZON)
    outputs["dl"] = evaluate_forecasts(test, dl_fcst, train, SEASON_LENGTH)
    outputs["dl"]["seconds"] = time.perf_counter() - start

    outputs["anomalies"] = run_anomaly_suite(df)
    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in outputs.items():
        table.to_csv(REPORT_TABLES_DIR / f"{name}_results.csv", index=False)
    try:
        from .visualization import generate_report_figures

        generate_report_figures()
    except ImportError:
        pass
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Final time-series project pipeline")
    parser.add_argument(
        "--mode",
        choices=["prepare", "quick", "full"],
        default="prepare",
        help="prepare: EDA artifacts; quick: local baseline; full: all required frameworks",
    )
    args = parser.parse_args()

    if args.mode == "prepare":
        print(json.dumps(run_prepare(), ensure_ascii=False, indent=2))
    elif args.mode == "quick":
        print(run_baseline_backtest().to_string(index=False))
    else:
        results = run_full_pipeline()
        for name, table in results.items():
            print(f"\n{name}\n{table.head().to_string(index=False)}")


if __name__ == "__main__":
    main()
