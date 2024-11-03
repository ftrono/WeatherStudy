from WeatherInput import WeatherInput
from globals.globals import *
import utilities.commons as commons
import utilities.training as training
import pandas as pd
import json, random


def predict(input_data: WeatherInput):
    try:
        LOG.info("Request received. Processing...")
        data = pd.DataFrame(data=input_data.model_dump(), index=[0])
        data = commons.preprocess_data(data)
        best_pipe = commons.load_pipeline(os.path.join(SAVE_DIR, MODELNAME))
        LOG.info("Generating prediction...")
        Y_pred = best_pipe.predict(data)[0]
        LOG.info(f"Prediction success!")
        return Y_pred

    except Exception as e:
        LOG.error(f"Prediction error!")
        LOG.exception(e)
        return -1


#MAIN:
if __name__ == '__main__':
    #perform a random test:
    fname = f"sample_{random.randint(1, 10)}.json"
    test_json = json.load(open(os.path.join("test_json", fname)))
    print(f"Testing with file: {fname}")

    y = predict(WeatherInput(**test_json["X"]))
    print(f"PRED: {y}")
    print(f"ACTUAL: {test_json["Y"]}")
