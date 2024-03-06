from colorama import Fore
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from tempfile import mkdtemp

from tensorflow import keras
from keras import layers
from keras import models
from keras import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping




def split_data(data=pd.read_csv('Data/clean_data.csv', index_col=0)):
    X = data['tweet_text']
    y = data.actionable

    # Split into Train/Test
    return train_test_split(X, y, test_size=0.3)

def tokenize_data():
    X_train, X_test, y_train, y_test = split_data()
    # This initializes a Keras utilities that does all the tokenization for you
    tokenizer = Tokenizer()

    # The tokenization learns a dictionary that maps a token (integer) to each word
    # It can be done only on the train set - we are not supposed to know the test set!
    # This tokenization also lowercases your words, apply some filters, and so on - you can check the doc if you want
    tokenizer.fit_on_texts(X_train)

    vocab_size = len(tokenizer.word_index)

    # We apply the tokenization to the train and test set
    X_train_token = tokenizer.texts_to_sequences(X_train)
    X_test_token = tokenizer.texts_to_sequences(X_test)

    return vocab_size, X_train_token, X_test_token

def pad_data(X_train_token, X_test_token):

    X_train_pad = pad_sequences(X_train_token, dtype='float32', padding='pre')
    X_test_pad = pad_sequences(X_test_token, dtype='float32', padding='pre')

    return X_train_pad, X_test_pad

def initialize_model(vocab_size):

    model = Sequential()

    model.add(layers.Embedding(input_dim=vocab_size+1, output_dim=2))

    model.add(layers.GRU(units=10, activation='tanh'))
    model.add(layers.Dense(10, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))


    model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
    )

    return model


def GRU_model():
    print(Fore.MAGENTA + 'Le GRU est lancé' + Fore.MAGENTA)

    vocab_size, X_train_token, X_test_token = tokenize_data()

    X_train_pad, X_test_pad = pad_data(X_train_token, X_test_token)
    X_train, X_test, y_train, y_test = split_data()

    model = initialize_model(vocab_size)
    es = EarlyStopping(patience=5, restore_best_weights=True)


    history = model.fit(X_train_pad,
                    y_train, batch_size=16,
                    epochs=5,
                    shuffle=True,
                    validation_split = 0.2, #IMPORTANT éviter le data leakage
                    callbacks = [es],
                    verbose = 1)

    print(history)
