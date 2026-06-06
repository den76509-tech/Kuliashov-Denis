import pandas as pd

from .config import HORIZON, REPORT_FIGURES_DIR, REPORT_TABLES_DIR, SEASON_LENGTH
from .data import prepare_series, train_test_split
from .eda import seasonality_profile


def _setup_matplotlib():
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    return plt


def plot_monthly_series(df: pd.DataFrame) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["ds"], df["y"], color="#1f77b4", linewidth=2)
    ax.set_title("AirPassengers: monthly passenger traffic")
    ax.set_xlabel("Date")
    ax.set_ylabel("Passengers")
    fig.tight_layout()
    fig.savefig(REPORT_FIGURES_DIR / "01_monthly_series.png", dpi=160)
    plt.close(fig)


def plot_monthly_seasonality(df: pd.DataFrame) -> None:
    plt = _setup_matplotlib()
    profile = seasonality_profile(df)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(profile["month"], profile["mean"], color="#2ca02c")
    ax.set_title("Average passenger traffic by month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean passengers")
    ax.set_xticks(range(1, 13))
    fig.tight_layout()
    fig.savefig(REPORT_FIGURES_DIR / "02_monthly_seasonality.png", dpi=160)
    plt.close(fig)


def plot_rolling_trend(df: pd.DataFrame) -> None:
    plt = _setup_matplotlib()
    rolling = df["y"].rolling(SEASON_LENGTH).mean()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["ds"], df["y"], color="#9ecae1", linewidth=1.5, label="Monthly values")
    ax.plot(df["ds"], rolling, color="#d62728", linewidth=2.5, label="12-month rolling mean")
    ax.set_title("Trend via 12-month rolling mean")
    ax.set_xlabel("Date")
    ax.set_ylabel("Passengers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_FIGURES_DIR / "03_trend_rolling.png", dpi=160)
    plt.close(fig)


def plot_anomalies(df: pd.DataFrame) -> None:
    plt = _setup_matplotlib()
    path = REPORT_TABLES_DIR / "anomalies_results.csv"
    if not path.exists():
        return
    anomalies = pd.read_csv(path, parse_dates=["ds"])
    flags = anomalies.groupby("ds")["is_anomaly"].sum().reset_index()
    flagged = df.merge(flags, on="ds", how="left").fillna({"is_anomaly": 0})
    flagged_points = flagged[flagged["is_anomaly"] > 0]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["ds"], df["y"], color="#1f77b4", linewidth=2, label="Series")
    ax.scatter(
        flagged_points["ds"],
        flagged_points["y"],
        color="#d62728",
        s=55,
        zorder=3,
        label="Flagged by anomaly methods",
    )
    ax.set_title("Anomaly flags across methods")
    ax.set_xlabel("Date")
    ax.set_ylabel("Passengers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_FIGURES_DIR / "04_anomalies.png", dpi=160)
    plt.close(fig)


def plot_metric_comparison() -> None:
    plt = _setup_matplotlib()
    tables = []
    for path in [
        REPORT_TABLES_DIR / "statistical_results.csv",
        REPORT_TABLES_DIR / "ml_results.csv",
        REPORT_TABLES_DIR / "dl_results.csv",
        REPORT_TABLES_DIR / "quick_baseline_metrics.csv",
    ]:
        if path.exists():
            table = pd.read_csv(path)
            table["source"] = path.stem.replace("_results", "").replace("_baseline_metrics", "")
            tables.append(table)
    if not tables:
        return
    metrics = pd.concat(tables, ignore_index=True)
    metrics = metrics.sort_values("sMAPE").head(12)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(metrics["model"], metrics["sMAPE"], color="#9467bd")
    ax.invert_yaxis()
    ax.set_title("Forecast model comparison by sMAPE")
    ax.set_xlabel("sMAPE, lower is better")
    ax.set_ylabel("Model")
    fig.tight_layout()
    fig.savefig(REPORT_FIGURES_DIR / "05_model_comparison.png", dpi=160)
    plt.close(fig)


def plot_holdout_forecast() -> None:
    plt = _setup_matplotlib()
    df = prepare_series(save=False)
    train, test = train_test_split(df, HORIZON)
    seasonal = train["y"].iloc[-SEASON_LENGTH:].to_numpy()
    naive_forecast = [seasonal[i % SEASON_LENGTH] for i in range(HORIZON)]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(train["ds"].tail(48), train["y"].tail(48), color="#1f77b4", linewidth=2, label="Train")
    ax.plot(test["ds"], test["y"], color="#111111", linewidth=2, label="Actual holdout")
    ax.plot(test["ds"], naive_forecast, color="#ff7f0e", linewidth=2, linestyle="--", label="SeasonalNaive")
    ax.axvline(test["ds"].iloc[0], color="#777777", linestyle=":", linewidth=1.5)
    ax.set_title("Holdout forecast example")
    ax.set_xlabel("Date")
    ax.set_ylabel("Passengers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_FIGURES_DIR / "06_holdout_forecast.png", dpi=160)
    plt.close(fig)


def generate_report_figures() -> None:
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare_series(save=False)
    plot_monthly_series(df)
    plot_monthly_seasonality(df)
    plot_rolling_trend(df)
    plot_anomalies(df)
    plot_metric_comparison()
    plot_holdout_forecast()


if __name__ == "__main__":
    generate_report_figures()
