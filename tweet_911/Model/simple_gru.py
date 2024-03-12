import numpy as np
import pandas as pd
from colorama import Fore
import os

from sklearn.metrics import classification_report

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding, Dense, MaxPool1D, GRU, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tweet_911.Model.utils import split_data, tokenize_data, pad_data


def initialize_model(vocab_size, embedding_dim=50):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size+1, output_dim=100, mask_zero=True))
    model.add(GRU(units=32, activation='tanh',return_sequences=True))
    model.add(GRU(units=16, activation='tanh',return_sequences=True))
    model.add(GRU(units=16, activation='tanh'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
            loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy', 'Recall', 'Precision']
    )

    return model


def GRU_model():
    print(Fore.YELLOW + '🦾 GRU model loading' + Fore.YELLOW)
    X_train, X_test, y_train, y_test = split_data()
    vocab_size, X_train_token, X_test_token = tokenize_data(X_train, X_test)
    X_train_pad, X_test_pad = pad_data(X_train_token, X_test_token)

    model = initialize_model(vocab_size)
    es = EarlyStopping(patience=10, restore_best_weights=True)

    checkpoint_path = "modelweights/model_gru.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)


    checkpoint = ModelCheckpoint(
        filepath = checkpoint_path,
        monitor = 'val_accuracy',
        verbose = 1,
        save_best_only = True,
        save_weights_only = False,
        mode = 'auto',
        save_freq = 'epoch')

    history = model.fit(X_train_pad,
                    y_train, batch_size=32,
                    epochs=100,
                    shuffle=True,
                    validation_split = 0.2, #IMPORTANT éviter le data leakage
                    callbacks = [es, checkpoint],
                    verbose = 1,
                    )


    print(history)

def report():
    # Generate the classification report
    vocab_size, X_train_token, X_test_token = tokenize_data()
    X_train_pad, X_test_pad = pad_data(X_train_token, X_test_token)
    X_train, X_test, y_train, y_test = split_data()

    y_pred = GRU_model.predict(X_test_pad) # Make cross validated predictions of entire dataset
    print(classification_report(y_test,y_pred)) # Pass predictions and true values to Classification report
