# Prediction Model Selection

## **Process**

To simplify the task, assumed that the Weather Prediction endpoint can be called at arbitrary points in time (i.e. an API call will not contain data from the previous days) and that the model shoud be able to generalize the predictions regardless of the location of the input data.

## **Data & NaN analysis**

A basic **Data and NaN analysis** can be found in the notebook `DataAnalysis.ipynb`. Here, I show that for 4 features out of the 23 columns (*"Evaporation", "Sunshine", "Cloud9am"* and *"Cloud3pm"*), there are 22 locations out of 49 with more than 30% NaN data. Since data are daily and highly variable, using interpolation or automatic filling through an Imputer would not make sense (I made an experiment anyway in the notebook `tests/DecTree_all_imputer.ipynb`, showing that the imputed actually degrade the performance of the model). Therefore, in `DataAnalysis.ipynb` I show that the best strategy to get rid of the NaN without dropping too much of the dataset is to first remove the 4 features and only then drop the remaining rows with NaN.

## **Feature selection**

The tests here mentioned can be found in the directory `tests`. They have been performed using the *DecTree Classifier* model only, since it was the first model I tested.

The following features have been removed:
* **"Date":** Temporal information is not relevant for this analysis.
* **"Location"**: A test has been performed in `tests/DecTree_all_loc.ipynb` and `tests/DecTree_sel_loc.ipynb`, showing that the feature does not add value to the prediction.
* **"Evaporation", "Sunshine", "Cloud9am", "Cloud3pm"**: As said before, these features have more than 30% NaN values in 22 locations (see `DataAnalysis.ipynb`). Two tests have been made including these features in the prediction (`tests/DecTree_all_dropna.ipynb` and `tests/DecTree_sel_dropna.ipynb`): when these features are included, accuracy score with the DecTree model is 1% higher, but the dataset used would be only 40% of the full dataset (vs 80% without these features). The 1% increase in score is negligible to justify the discarding of so much data.
* **"RainToday"**: This feature (boolean) it is redundant with "Rainfall" (numeric, see `DataAnalysis.ipynb`) and the latter feature appears to be more informative (see test in `tests/DecTree_all_rainfall.ipynb`).

## **Models**

The DataAnalysis and the tests made with 4 prediction models (**DecisionTree Classifier**, **LinearSVC**, **RandomForest Classifier** and **XGBoost Classifier**) can be found in this directory.

I tested the 4 prediction models twice:
* (folder **"all_features"**) one with all the survived features; and 
* (folder **"sel_features"**) one with only a percentage of the most informative features, selected through a Grid Search Cross Validation with Scikit-Learn's SelectPercentile and mutual_info_classif to measure the dependency between variables.

## **Results**

After the tests performed, the chosen prediction model is a **RandomForest Classifier** trained on the 7 features (out of 15) that were selected by the GridSearch Cross Validation, which obtained an accuracy of **85.5%** on the test set.

An overview of the results obtained through the 4 models can be found here.

***1.a) DecTree - All features***: 

* Accuracy: **78.7%**
* Features considered: 15
* Best features: Humidity3pm (30.1%), Pressure3pm (8.2%), WindGustSpeed (7.8%)

|           |  	No  | 	Yes| 
| --------  | ----- | ---- |
| Precision	| 0.87	| 0.53 | 
| Recall	| 0.86	| 0.55 | 
| F-score	| 0.86	| 0.54 | 
| Support	| 26252	| 7626 | 

---

***1.b) DecTree - Selected features***: 

* Accuracy: **78.1%**
* Features considered: 7
* Best features: Humidity3pm (33.0%), Pressure3pm (13.8%), Temp3pm (13.2%)

|           |  	No  | 	Yes| 
| --------  | ----- | ---- |
| Precision	| 0.86	| 0.51 | 
| Recall	| 0.85	| 0.53 | 
| F-score	| 0.86	| 0.52 | 
| Support	| 26252	| 7626 | 

---

***2.a) RandomForest - All features***: 

* Accuracy: **85.5%**
* Features considered: 15
* Best features: Humidity3pm (20.6%), Pressure3pm (7.9%), Humidity9am (7.4%)

|           |  	No  | 	Yes| 
| --------  | ----- | ---- |
| Precision	| 0.87	| 0.77 | 
| Recall	| 0.96	| 0.51 | 
| F-score	| 0.91	| 0.61 | 
| Support	| 26252	| 7626 | 

---

***2.b) RandomForest - Selected features***: 

* Accuracy: **85.1%**
* Features considered: 7
* Best features: Humidity3pm (26.7%), Pressure3pm (13.6%), Temp3pm (13.1%)

|           |  	No  | 	Yes| 
| --------  | ----- | ---- |
| Precision	| 0.87	| 0.75 | 
| Recall	| 0.95	| 0.51 | 
| F-score	| 0.91	| 0.61 | 
| Support	| 26252	| 7626 | 

---

***3.a) LinearSVC - All features***: 

* Accuracy: **84.7%**
* Features considered: 15
* Best features: Pressure3pm (24.5%), Humidity3pm (21.7%), Pressure9am (17.7%), WindGustSpeed (14.3%)

|           |  	No  | 	Yes| 
| --------  | ----- | ---- |
| Precision	| 0.86	| 0.76 | 
| Recall	| 0.96	| 0.48 | 
| F-score	| 0.91	| 0.58 | 
| Support	| 26252	| 7626 | 

---

***3.b) LinearSVC - Selected features***: 

* Accuracy: **84.4%**
* Features considered: 7
* Best features: Pressure3pm (30.2%), Humidity3pm (26.4%), Pressure9am (20.6%), WindGustSpeed (11.8%)

|           |  	No  | 	Yes| 
| --------  | ----- | ---- |
| Precision	| 0.86	| 0.75 | 
| Recall	| 0.96	| 0.46 | 
| F-score	| 0.90	| 0.57 | 
| Support	| 26252	| 7626 | 

---

***4.a) XGBoost - All features***: 

* Accuracy: **84.5%**
* Features considered: 15
* Best features: Humidity3pm (53.1%), WindGustSpeed (9.2%), Pressure3pm (5.2%)

|           |  	No  | 	Yes| 
| --------  | ----- | ---- |
| Precision	| 0.87	| 0.72 | 
| Recall	| 0.94	| 0.52 | 
| F-score	| 0.90	| 0.60 | 
| Support	| 26252	| 7626 |

---

***4.b) XGBoost - Selected features***: 

* Accuracy: **84.5%**
* Features considered: 7
* Best features: Humidity3pm (62.9%), Pressure3pm (8.1%), WindGustSpeed (12.1%)

|           |  	No  | 	Yes| 
| --------  | ----- | ---- |
| Precision	| 0.87	| 0.73 | 
| Recall	| 0.95	| 0.49 | 
| F-score	| 0.90	| 0.59 | 
| Support	| 26252	| 7626 | 
