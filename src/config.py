from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "air_passengers.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "air_passengers_prepared.csv"
REPORT_TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
REPORT_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

SERIES_ID = "air_passengers"
FREQ = "MS"
SEASON_LENGTH = 12
HORIZON = 24
