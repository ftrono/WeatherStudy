import os, mlflow
from globals.logger import LOG


# GLOBALS
# cwd: "app"

DATASET = os.path.join("datasets", "weather.csv")
SAVE_DIR = os.path.join("models")
RUNS_DIR = os.path.join("mlruns")
mlflow.set_tracking_uri(RUNS_DIR)

TEST_PCT = 0.30

FEATURES_TO_REMOVE = ["Unnamed: 0", "Date", "Location", "Evaporation", "Sunshine", "Cloud9am", "Cloud3pm", "RainToday"]
TARGET = "RainTomorrow"

MODELNAME = "RandomForest"

PARAM_GRID = {
    "feature_selection__percentile": [10, 20, 30, 40, 50]
}
