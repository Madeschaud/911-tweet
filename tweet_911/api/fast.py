import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Model.bidirection_lstm import model_bidirectional_lstm, split_data, tokenize_data
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# $WIPE_BEGIN
# 💡 Preload the model to accelerate the predictions
# We want to avoid loading the heavy Deep Learning model from MLflow at each `get("/predict")`
# The trick is to load the model in memory when the Uvicorn server starts
# and then store the model in an `app.state.model` global variable, accessible across all routes!
# This will prove very useful for the Demo Day
app.state.model = model_bidirectional_lstm()

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
@app.get("/predict")
def predict(
        tweet: str,  # Tweet we want to predict
    ):
    """
    Make a tweet prediction.
    """
    model = app.state.model
    assert model is not None

    y_pred = model.predict(tweet)

    # # ⚠️ fastapi only accepts simple Python data types as a return value
    # # among them dict, list, str, int, float, bool
    # # in order to be able to convert the api response to JSON
    return dict(tweet_accurate=float(y_pred))
    # $CHA_END


@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Hello")
    # $CHA_END
