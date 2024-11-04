import pandas as pd
import json, random
from sklearn.pipeline import Pipeline
from classes.WeatherInput import WeatherInput
from globals.globals import *
import utilities.commons as commons


def make_prediction(input_data: dict):
    try:
        LOG.info("WEATHER: Request received. Processing...")
        best_pipe = commons.load_pipeline(os.path.join(SAVE_DIR, MODELNAME))
        LOG.info("Generating prediction...")
        data = pd.DataFrame(data=input_data, index=[0])
        data = commons.preprocess_data(data)
        Y_pred = best_pipe.predict(data)[0]
        response = {"RainTomorrow": int(Y_pred)}
        LOG.info(f"WEATHER: Prediction success!")
        return response

    except Exception as e:
        LOG.error(f"WEATHER: Prediction error!")
        LOG.exception(e)
        return {"error": "endpoint_error"}


#MAIN:
if __name__ == '__main__':
    #perform a random test:
    fname = f"sample_{random.randint(1, 10)}.json"
    test_json = json.load(open(os.path.join("test_json", fname)))
    print(f"Testing with file: {fname}")

    y = make_prediction(test_json["X"])
    print(f"PRED: {y}")
    print(f"ACTUAL: {test_json["Y"]}")
