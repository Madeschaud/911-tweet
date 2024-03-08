import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tweet_911.load_model_weights import load_model_weights

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# from tensorflow.keras.preprocessing.sequence import pad_sequences

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
app.state.model = load_model_weights()


def padding_tweet(tweet):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([tweet])
    sequences = tokenizer.texts_to_sequences([tweet])
    tweet_padded = pad_sequences(sequences)
    return tweet_padded

# http://127.0.0.1:8000/predict?tweet="HELLOOOO"
@app.get("/predict")
def predict(
        tweet: str,  # Tweet we want to predict
    ):
    """
    Make a tweet prediction.
    """
    model = app.state.model
    assert model is not None



    y_pred = model.predict(padding_tweet(tweet))

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
