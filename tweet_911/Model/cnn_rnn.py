import numpy as np
import pandas as pd
from colorama import Fore

from sklearn.metrics import classification_report

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding, Dense, MaxPool1D, Conv1D, GlobalMaxPool1D, RNN
from keras.callbacks import EarlyStopping
from tweet_911.Model.utils import split_data, tokenize_data, pad_data

def initialize_model(vocab_size, embedding_dim=50):

    model = Sequential()

    model.add(Embedding(input_dim=vocab_size+1, output_dim=embedding_dim, mask_zero=True))

    model.add(Conv1D(20, kernel_size=3, activation='relu'))
    model.add(GlobalMaxPool1D())

    model.add(RNN(units=10, activation='tanh'))

    model.add(Dense(10, activation='relu'))
    model.add(Dense(5, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))


    model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy', 'recall']
    )

    return model


def cnn_rnn():
    print(Fore.MAGENTA + 'CNN + RNN en cours' + Fore.MAGENTA)

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
