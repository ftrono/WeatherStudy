# WeatherStudy
Service & model study for a Weather Prediction classification task.

The study has been carried out on:
* Python 3.12.7
* Ubuntu (WSL version)

## Task description

**Problem**

The dataset is a .csv file containing some weather data (temperature, humidity, air pressure, etc.). It is asked to develop a classification algorithm capable of predicting the label contained in the "*RainTomorrow*" column - i.e., whether it rained on the following day - by making use of the values contained in the remaining columns (or in a convenient subset of them).

**Execution**

* Preprocess the data to make them more apt for the subsequent analysis
* Choose one or more algorithms to carry out said classification task
* Pick one or more metrics to assess how algorithms generalise to unseen data, which algorithm displays the best performances and the overall quality of the analysis
* Using *MLFlow*, create a docker image to serve the model on a REST interface

## Task execution comment

To simplify the task, I am assuming that the Weather Prediction endpoint can be called at arbitrary points in time (i.e. an API call will not contain data from the previous days) and that the model shoud be able to generalize the predictions regardless of the location of the input data.

After testing 4 alternative classifiers, the chosen prediction model is a **RandomForest Classifier** trained on the 7 features that were selected by the GridSearch Cross Validation process, which obtained an accuracy of **85.5%** on the test set. A file explaining the process behind model selection can be found at `app/experiments/README.md`.

## Repo structure

All the code is available in the folder `app`. The home directory contains the file `docker-compose.yml` and, for local runs only, two scripts to setup and run the Python venv (`venv_setup.sh` and `venv_run.sh`).

Inside the folder `app`, the following key files and directories can be found:
* `datasets`: contains the file `weather.csv`.
* `experiments`: directory containing all the data science experiments and model tests.
* `globals`: directory containing the Python globals for the app.
* `models`: will contain the trained model after training.
* `test_json`: directory containing some JSON input samples for test.
* `utilities`: directory containing the Python functions used within the training and testing pipelines.
* `main.py`: main app entry point. It executes the training and the FastAPI endpoint.
* `train.py`: executes the training process.
* `predict.py`: executes the prediction process.


## Test samples
A few sample JSONs, extracted from the test set of the original dataset, can be found in the directory `app/test_json`. The JSONs are structured as follows:
```
{
    "X": {json object}
    "Y": int
}
```

 To test the endpoint, please use as input *ONLY* the nested JSON object within the **"X" key**. The "Y" is the label ("RainTomorrow": 1 if Yes, 0, if No).

## Instructions to run the code

### **FastAPI**

**A) With Docker**
* Position the bash terminal inside the ***home*** directory of the repo.
* From the bash terminal, run `docker-compose up --build`. 
* The command will compile the Docker container that will execute the file `app/main.py`.
* The logs are accessible from the `logs` folder that will be automatically created in the ***home*** directory.

**B) Without Docker**
* Position the bash terminal inside the ***home*** directory of the repo.
* Run `source venv_setup.sh` to install and open the Python Environment.
* Now, position the bash terminal inside the folder `app`. This will be the main directory to execute all code.
* From the bash terminal, run `mlflow server --host 127.0.0.1 --port 5000` to start the MLFlow server and UI.
* From the bash terminal, run `python3 main.py`.
* The logs are accessible from the `logs` folder that will be automatically created in the directory `app/logs`. 
* From MLFlow UI, the model run and stored model will be available after training.

**General notes:**

* Both the Docker container and the Python command will execute the file `main.py`, which automatically (in order): 1) trains the model, and 2) activates a FastAPI endpoint, which will be available at this URL: http://localhost:3001/weather (**POST**).
* Once the model is trained, it won't be trained again (unless the Docker container is rebuilt).
* You can test the POST endpoint via Postman using one of the sample JSON files available in `app/test_json` (please use as input *ONLY* the nested JSON object within the "X" key).
