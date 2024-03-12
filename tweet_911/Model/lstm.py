# Basic Imports
import numpy as np
import pandas as pd


# tensorflow
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras import layers
from keras.regularizers import l2

from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping

from tweet_911.Model.utils import split_data, tokenize_data, pad_data

#sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# import data
data =pd.read_csv('tweet_911/Data/clean_data.csv')
data.head()


def initialize_model(vocab_size, embedding_dim=50):
    # build model
    model = Sequential()
    #embedding
    model.add(Embedding(input_dim=vocab_size+1,output_dim=embedding_dim, mask_zero=True))

    #lstm big -- natural-disaster-tweet_jbach_lstm_big VM
    model.add(LSTM(units=128, return_sequences=True, activation= 'tanh'))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32, activation='tanh'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #lstm small -- natural-disaster-tweet_jbach_lstm_small local
    #model.add(LSTM(units=32, return_sequences=True, activation= 'tanh'))
    #model.add(LSTM(16, return_sequences=True))
    #model.add(LSTM(8, activation='tanh'))
    #model.add(Dense(32, activation='relu'))
    #model.add(Dropout(0.2))
    #model.add(Dense(16, activation='relu'))
    #model.add(Dense(1, activation='sigmoid'))

    #compile
    model.compile(loss='binary_crossentropy', optimizer= 'rmsprop', metrics= ['accuracy', 'Recall', 'Precision'])
    return model
