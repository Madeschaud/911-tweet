import os
import numpy as np

# Pour s'assurer de save sur MLFLOW
MODEL_TARGET = os.environ.get("MODEL_TARGET")

# Path pour save le model locally
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "911-tweet", "training_outputs")

# Nom du modèle sur MLFLOW
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
