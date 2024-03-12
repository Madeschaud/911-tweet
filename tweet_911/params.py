import os
import numpy as np

# Pour s'assurer de save sur MLFLOW
MODEL_TARGET = os.environ.get("MODEL_TARGET")

# Path pour save le model locally
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "911-tweet", "training_outputs")

# Nom du modèle sur MLFLOW
MLFLOW_MODEL_NAME_DISASTER = os.environ.get("MLFLOW_MODEL_NAME_DISASTER")
MLFLOW_MODEL_NAME_ACTIONABLE= os.environ.get("MLFLOW_MODEL_NAME_ACTIONABLE")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")

BUCKET_NAME = os.environ.get("BUCKET_NAME")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
