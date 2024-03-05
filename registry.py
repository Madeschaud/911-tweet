from colorama import Fore, Style
from tensorflow import keras
from google.cloud import storage
import os
import mlflow
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

def save_model(model: keras.Model, model_name) -> None:

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{model_name}")
    model.save(model_path)
    print("✅ Model saved locally")

    if MODEL_TARGET == "mlflow":
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path='model',
            registered_model_name=MLFLOW_MODEL_NAME
        )
        print("✅ Model saved to MLflow")

    return None
