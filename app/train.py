from globals.globals import *
from datetime import datetime
import utilities.commons as commons
import utilities.training as training


#PERFORM TRAINING:

def perform_training():
    LOG.info("Training process started.")
    mlflow.start_run(run_name=f"{MODELNAME}_{datetime.strftime(datetime.now(), '%Y-%m-%d_%H:%M:%S')}")

    #preprocess dataset:
    data = training.load_and_preprocess_dataset()
    X, Y = training.separate_labels(data)
    X_train, X_test, Y_train, Y_test = training.split_dataset(X, Y)

    #build & cross-validate pipeline:
    sk_pipe = commons.build_pipeline(X_train)
    best_pipe = training.get_best_model(sk_pipe, X_train, Y_train)

    #TRAIN: fit selected model:
    LOG.info("Fitting the best model...")
    best_pipe.fit(X_train, Y_train)
    LOG.info("Training complete.")

    #Serialize:
    training.save_pipeline(best_pipe, os.path.join(SAVE_DIR, MODELNAME))

    #PREDICT:
    LOG.info("Testing trained model...")
    Y_preds = best_pipe.predict(X_test)

    #Log metrics:
    training.log_metrics(Y_test, Y_preds)
    training.log_feature_importance(best_pipe, X_train)

    mlflow.end_run()
    LOG.info("Training process ended.")


#MAIN:
if __name__ == '__main__':
    perform_training()
