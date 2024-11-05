import os, mlflow, sqlite3
from globals.logger import LOG


# GLOBALS
# cwd: "app"

DATASET = os.path.join("datasets", "weather.csv")
SAVE_DIR = os.path.join("models")
TRACKING_DB = os.path.join("tracking", "mlflow.db")
TRACKING_URI = f"sqlite:///{TRACKING_DB}"

# Get SQLite Tracking DB:
conn = sqlite3.connect(TRACKING_DB)
conn.close()

# Set tracking URI:
mlflow.set_tracking_uri(TRACKING_URI)
LOG.info(f"MLFlow Tracking URI set to: {TRACKING_URI}")

TEST_PCT = 0.30

FEATURES_TO_REMOVE = ["Unnamed: 0", "Date", "Location", "Evaporation", "Sunshine", "Cloud9am", "Cloud3pm", "RainToday"]
TARGET = "RainTomorrow"

MODELNAME = "RandomForest"

PARAM_GRID = {
    "feature_selection__percentile": [10, 20, 30, 40, 50]
}
