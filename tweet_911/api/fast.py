import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tweet_911.registry import load_model

import pickle
import os

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
app.state.model = load_model('Staging')

# def padding_tweet(tweet):
#     tokenizer = Tokenizer()
#     tokenizer.fit_on_texts([tweet])
#     sequences = tokenizer.texts_to_sequences([tweet])
#     tweet_padded = pad_sequences(sequences)
#     return tweet_padded

# http://127.0.0.1:8000/predict?tweet="HELLOOOO"
@app.get("/predict_disaster")
def api_predict_disaster(
        tweet: str,  # Tweet we want to predict
    ):
    """
    Make a tweet prediction.
    """

    model = app.state.model[0]
    assert model is not None
    # loading
    print(os.getcwd())
    with open('tweet_911/Data/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    X_tweet_token = tokenizer.texts_to_sequences([tweet])
    print(X_tweet_token)
    print('tweet = ',tweet)
    X_tweet_pad = pad_sequences(X_tweet_token, dtype='float32', padding='post', maxlen=20)
    print(X_tweet_pad)

    y_pred = model.predict(X_tweet_pad)
    return dict(tweet_disaster=float(y_pred))

@app.get("/predict_actionable")
def api_predict_actionable(
        tweet: str,  # Tweet we want to predict
    ):
    """
    Make a tweet prediction.
    """

    model = app.state.model[1]
    assert model is not None
    # loading
    print(os.getcwd())
    with open('tweet_911/Data/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    X_tweet_token = tokenizer.texts_to_sequences([tweet])
    print(X_tweet_token)
    print('tweet = ',tweet)
    X_tweet_pad = pad_sequences(X_tweet_token, dtype='float32', padding='post', maxlen=20)
    print(X_tweet_pad)

    y_pred = model.predict(X_tweet_pad)
    return dict(tweet_actionable=float(y_pred))


@app.get("/ping")
def root():
    # $CHA_BEGIN
    return dict(greeting="Pong")
    # $CHA_END


@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Hello")
    # $CHA_END
