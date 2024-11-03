from globals.globals import *
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier


# COMMON UTILITIES

def preprocess_data(data: pd.DataFrame):
    #remove features one by one (for endpoint tests):
    for col in FEATURES_TO_REMOVE:
        try:
            data = data.drop(columns=col)
        except:
            pass
    data = data.dropna().reset_index(drop=True)
    return data
    

def build_pipeline(X: pd.DataFrame):
    # Lists of column names for different data types
    num_feats = X.select_dtypes(['float64']).columns.tolist()
    cat_feats = X.select_dtypes(['object']).columns.tolist()

    # Create transformers for both numeric and categorical features
    num_transform = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    cat_transform = Pipeline(
        steps=[
            ("encoder", OrdinalEncoder())
        ]
    )

    # Combine the transformations in a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transform, num_feats),
            ('cat', cat_transform, cat_feats)
        ]
    )

    #Build pipeline:
    sk_pipe = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ("feature_selection", SelectPercentile(score_func=mutual_info_classif)),
            ('classifier', RandomForestClassifier(random_state=42))
        ]
    )

    return sk_pipe


#Deserialize:
def load_pipeline(save_path: str):
    pipe_loaded = joblib.load(save_path)
    LOG.info(f"Pipeline loaded!")
    return pipe_loaded
