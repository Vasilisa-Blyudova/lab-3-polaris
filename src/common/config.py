import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
LOGS_DIR = PROJECT_ROOT / "logs"

BRONZE_PATH = DATA_DIR / "bronze" / "flights"
SILVER_PATH = DATA_DIR / "silver" / "flights"
GOLD_ANALYTICS_DIR = DATA_DIR / "gold" / "analytics"
GOLD_ANALYTICS_PATHS = {
    "by_airport": GOLD_ANALYTICS_DIR / "by_airport",
    "by_carrier": GOLD_ANALYTICS_DIR / "by_carrier",
    "by_hour": GOLD_ANALYTICS_DIR / "by_hour",
    "by_season": GOLD_ANALYTICS_DIR / "by_season",
}
GOLD_FEATURES_PATH = DATA_DIR / "gold" / "ml_features"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(PROJECT_ROOT / "mlruns"))
MLFLOW_EXPERIMENT_NAME = "flight-delay-lakehouse"
ML_SAMPLE_FRACTION = float(os.getenv("ML_SAMPLE_FRACTION", "1.0"))


COLUMNS_TO_KEEP = [
    "FL_DATE",
    "OP_UNIQUE_CARRIER",
    "OP_CARRIER_FL_NUM",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "DEP_TIME",
    "DEP_DELAY",
    "CRS_ARR_TIME",
    "ARR_TIME",
    "ARR_DELAY",
    "CANCELLED",
    "DIVERTED",
    "DISTANCE",
]

YEARS = list(range(2018, 2025))
DELAY_THRESHOLD = 15
NA_THRESHOLDS = {"ARR_DELAY": 0.05}
PARTITION_COLUMNS = ["year", "month"]
Z_ORDER_COLUMNS = ["ORIGIN", "OP_UNIQUE_CARRIER", "hour"]
