import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from globals.globals import *
from utilities.commons import preprocess_data


# TRAINING

def load_and_preprocess_dataset():
    data = pd.read_csv(DATASET)
    data = preprocess_data(data)
    #encode target column (it won't pass through pipeline):
    data[TARGET] = np.where(data[TARGET] == "Yes", 1, 0)
    return data


def separate_labels(data: pd.DataFrame):
    X = data.drop(columns=TARGET)
    Y = data[TARGET]
    return X, Y


def split_dataset(X: pd.DataFrame, Y: pd.Series):
    #split:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_PCT, random_state=42)
    #log:
    LOG.info(f"X.shape, Y.shape, X_train.shape, Y_train.shape, X_test.shape, Y_test.shape")
    LOG.info(f"{X.shape}, {Y.shape}, {X_train.shape}, {Y_train.shape}, {X_test.shape}, {Y_test.shape}")
    return X_train, X_test, Y_train, Y_test


def get_best_model(sk_pipe: Pipeline, X_train: pd.DataFrame, Y_train: pd.Series, cv: int = 5):
    # Find best model:
    n_cands = len(PARAM_GRID["feature_selection__percentile"])
    grid_search = GridSearchCV(sk_pipe, PARAM_GRID, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
    LOG.info(f"Training: fitting {cv} folds for each of {n_cands} candidates, totalling {cv*n_cands} fits")
    grid_search.fit(X_train, Y_train)

    # Get the best model:
    best_percentile = grid_search.best_params_['feature_selection__percentile']
    best_score = grid_search.best_score_
    best_pipe = grid_search.best_estimator_

    LOG.info(f"Best Percentile: {best_percentile}")
    LOG.info(f"Best cross-validated score: {best_score:.4f}")
    return best_pipe


#Serialize:
def save_pipeline(best_pipe: Pipeline, save_path: str, run_id: str):
    joblib.dump(best_pipe, save_path)
    # Register also to Model Registry (production-ready):
    model_uri = f"runs:/{run_id}/{MODELNAME}"
    mlflow.register_model(model_uri, MODELNAME)
    LOG.info(f"Pipeline saved!")


def log_metrics(Y_test: pd.Series, Y_preds: np.ndarray):
    accuracy = accuracy_score(Y_test, Y_preds)
    mlflow.log_metric("Accuracy", round(accuracy, 3))
    LOG.info(f"Accuracy: {accuracy*100:.1f}%")

    metrics = {}

    metrics["Precision"], metrics["Recall"], metrics["F-score"], metrics["Support"] = precision_recall_fscore_support(Y_test, Y_preds)

    df_metrics = pd.DataFrame(data=metrics).T
    df_metrics.columns = ["No", "Yes"]
    df_metrics = df_metrics.round(2)

    for key in df_metrics.keys():
        mlflow.log_metric(f"{key} - No", round(df_metrics[key].iloc[0], 3))
        mlflow.log_metric(f"{key} - Yes", round(df_metrics[key].iloc[1], 3))
        LOG.info(f"{key} - No: {df_metrics[key].iloc[0]}")
        LOG.info(f"{key} - Yes: {df_metrics[key].iloc[1]}")
    

def log_feature_importance(best_pipe: Pipeline, X_train: pd.DataFrame):
    # Get selected features list:
    selected_mask = best_pipe.named_steps['feature_selection'].get_support()
    sel_features = X_train.columns[selected_mask]

    # Build features importance dict:
    imp_scores = best_pipe.named_steps['classifier'].feature_importances_
    total = sum(imp_scores)
    f_importance = {}

    for i in range(len(sel_features)):
        temp = round(float(imp_scores[i]/total)*100,3)
        f_importance[f"FeatImp: {sel_features[i]}"] = temp
    
    mlflow.log_metrics(f_importance)
    LOG.info(f"Feature importance: {f_importance}")
    return f_importance

