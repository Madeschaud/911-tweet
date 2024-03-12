import numpy as np
import pandas as pd
from colorama import Fore

from sklearn.metrics import classification_report

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding, Dense, MaxPool1D, Conv1D, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tweet_911.Model.utils import split_data, tokenize_data, pad_data

def initialize_model(vocab_size, embedding_dim=50) -> keras.models:
    max_length = 20

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size+1, output_dim=embedding_dim, mask_zero=True, input_length=max_length))
    model.add(Conv1D(128, 3, padding='same', activation="relu"))
    model.add(MaxPool1D(pool_size=3))
    model.add(Conv1D(64, 3, padding='same', activation='relu'))
    model.add(MaxPool1D(pool_size=3))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy', 'Precision', 'Recall'])

    return model

def model_cnn():
    print(Fore.MAGENTA + 'Initializing CNN' + Fore.MAGENTA)

    max_len = 300
    X_train, X_test, y_train, y_test = split_data()
    vocab_size, X_train_token, X_test_token = tokenize_data(X_train, X_test)
    X_train_pad, X_test_pad = pad_data(X_train_token, X_test_token, max_len)

    model = initialize_model(vocab_size=vocab_size)
    print(Fore.MAGENTA + 'CNN initialized' + Fore.MAGENTA)

    es = EarlyStopping(patience=10, restore_best_weights=True)
    filepath = "Data/checkpoint/cnn_model.h5"
    model_checkpoint= ModelCheckpoint(filepath=filepath, save_best_only=True, monitor='val_accuracy')
    history = model.fit(
        X_train_pad, y_train,
        epochs=10,  # Use early stopping in practice
        batch_size=32,
        validation_split=0.2,
        callbacks=[es, model_checkpoint],
        verbose=1)

    print(history)

    y_pred = model.predict(X_test_pad) # Make cross validated predictions of entire dataset
    print(classification_report(y_test,(y_pred > 0.5).astype(int))) # Pass predictions and true values to Classification report
