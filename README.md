# WeatherStudy
Service & model study for a Weather Prediction classification task.

The study has been carried out on:
* Python 3.12.7
* Ubuntu (WSL version)

## Task Description

**Problem**

The dataset is a .csv file containing some weather data (temperature, humidity, air pressure, etc.). It is asked to develop a classification algorithm capable of predicting the label contained in the "*RainTomorrow*" column - i.e., whether it rained on the following day - by making use of the values contained in the remaining columns (or in a convenient subset of them).

**Execution**

* Preprocess the data to make them more apt for the subsequent analysis
* Choose one or more algorithms to carry out said classification task
* Pick one or more metrics to assess how algorithms generalise to unseen data, which algorithm displays the best performances and the overall quality of the analysis
* Using *MLFlow*, create a docker image to serve the model on a REST interface

