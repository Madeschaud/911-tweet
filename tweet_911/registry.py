from colorama import Fore, Style
from tensorflow import keras
import os
import mlflow
import time
import glob
from mlflow.tracking import MlflowClient
from params import *


def save_results(params: dict, metrics: dict) -> None:
    """
    if MODEL_TARGET='mlflow', also persist them on MLflow
    """
    if MODEL_TARGET == "mlflow":
        if params is not None:
            mlflow.log_params(params)
        if metrics is not None:
            mlflow.log_metrics(metrics)

        print("✅ Results saved on mlflow")

def save_model(model: keras.Model, local_model_name) -> None:

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{local_model_name}", f"{timestamp}.h5")
    model.save(model_path)
    print(f"✅ Model saved locally under name: {MLFLOW_MODEL_NAME}/{MLFLOW_EXPERIMENT}/{timestamp}.h5")

    if MODEL_TARGET == "mlflow":
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path='model',
            registered_model_name=MLFLOW_MODEL_NAME
        )
        print("✅ Model saved to MLflow")

    return None

def load_model(stage="Production") -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow'

    Return None (but do not Raise) if no model is found

    """

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

        latest_model = keras.models.load_model(most_recent_model_path_on_disk)

        print("✅ Model loaded from local disk")

        return latest_model


    elif MODEL_TARGET == "mlflow":
        print(Fore.BLUE + f"\nLoading model in {stage} from MLflow..." + Style.RESET_ALL)

        # Load model from MLflow
        model = None
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        try:
            model_version=client.get_latest_versions(name=MLFLOW_MODEL_NAME, stages=[stage])
            model_uri= model_version[0].source
            assert model_uri is not None
        except:
            print(f"No model named {MLFLOW_MODEL_NAME} found in stage {stage}")
            return None
        model = mlflow.tensorflow.load_model(model_uri=model_uri)
        print(Fore.MAGENTA + f"\nModel loaded from MLflow!" + Style.RESET_ALL)
        return model

    else:
        return None





def mlflow_transition_model(current_stage: str, new_stage: str) -> None:
    """
    Transition the latest model from the `current_stage` to the
    `new_stage` and archive the existing model in `new_stage`
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    version = client.get_latest_versions(name=MLFLOW_MODEL_NAME, stages=[current_stage])

    if not version:
        print(f"\n❌ No model found with name {MLFLOW_MODEL_NAME} in stage {current_stage}")
        return None

    client.transition_model_version_stage(
        name=MLFLOW_MODEL_NAME,
        version=version[0].version,
        stage=new_stage,
        archive_existing_versions=True
    )

    print(f"✅ Model {MLFLOW_MODEL_NAME} (version {version[0].version}) transitioned from {current_stage} to {new_stage}")

    return None

def mlflow_run(func):
    """
    Generic function to log params and results to MLflow along with TensorFlow auto-logging

    Args:
        - func (function): Function you want to run within the MLflow run
        - params (dict, optional): Params to add to the run in MLflow. Defaults to None.
        - context (str, optional): Param describing the context of the run. Defaults to "Train".
    """
    def wrapper(*args, **kwargs):
        mlflow.end_run()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)

        with mlflow.start_run():
            mlflow.tensorflow.autolog()
            results = func(*args, **kwargs)

        print("✅ mlflow_run auto-log done")

        return results
    return wrapper


if __name__=='__main__':
    load_model()
