from colorama import Fore
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import classification_report
from tempfile import mkdtemp

from tensorflow.keras.metrics import Recall
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint





def split_data():
    #identify X,y
    X = data['tweet_clean']
    y = data.actionable

    #split data
    return train_test_split( X, y, test_size=0.30, random_state=42)


def tokenize_data(X_train, X_test):
    # tokenize
    tk =Tokenizer()
    tk.fit_on_texts(X_train)

    vocab_size = len(tk.word_index)
    print(f'There are {vocab_size} different words in your corpus')

    return vocab_size, tk.texts_to_sequences(X_train), tk.texts_to_sequences(X_test)

def pad_data(X_train_token, X_test_token):

    X_train_pad = pad_sequences(X_train_token, dtype='float32', padding='post', maxlen = 20)
    X_test_pad = pad_sequences(X_test_token, dtype='float32', padding='post', maxlen = 20)

    return X_train_pad, X_test_pad

def initialize_model():

    vocab_size, X_train_token, X_test_token = tokenize_data()

    model = Sequential()

    model.add(layers.Embedding(input_dim=vocab_size+1, output_dim=100, mask_zero=True))



    model.add(layers.GRU(units=256, activation='tanh',return_sequences=True))

    model.add(layers.GRU(units=128, activation='tanh',return_sequences=True))

    model.add(layers.GRU(units=64, activation='tanh',return_sequences=True))

    model.add(layers.GRU(units=32, activation='tanh'))


    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(rate=0.2))

    model.add(layers.Dense(1, activation='sigmoid'))




    model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy', 'Recall', 'Precision']
    )

    return model


def GRU_model():
    print(Fore.YELLOW + '🦾 GRU model loading' + Fore.YELLOW)

    # set params
    max_features =10000
    max_len=20
    embedding_dim=50
    batch_size = 32
    patience=20
    validation_split=0.2
    epochs=1

    X_train, X_test,y_train, y_test = split_data()
    vocab_size, X_train_token, X_test_token = tokenize_data(X_train, X_test)

    # Pad the inputs
    X_train_pad = pad_sequences(X_train_token, dtype='float32', padding='post',maxlen=max_len)
    X_test_pad = pad_sequences(X_test_token, dtype='float32', padding='post',maxlen=max_len)


    model = initialize_model()
    es = EarlyStopping(patience=patience, restore_best_weights=True)

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
