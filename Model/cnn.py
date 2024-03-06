import numpy as np
import pandas as pd
from colorama import Fore
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding, Dense, MaxPool1D, Conv1D, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, RMSprop
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.metrics import Accuracy, Recall, Precision

def split_data(data=pd.read_csv('Data/clean_data.csv', index_col=0)) -> tuple:
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

def pad_data(X_train_token, X_test_token, max_len):

    X_train_pad = sequence.pad_sequences(X_train_token, dtype='float32', padding='pre', maxlen=max_len)
    X_test_pad = sequence.pad_sequences(X_test_token, dtype='float32', padding='pre', maxlen=max_len)

    return X_train_pad, X_test_pad

def initialize_model(vocab_size, max_len) -> keras.models:
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size+1, output_dim=50, input_length=max_len))
    model.add(Conv1D(32, 3, padding='same', activation="relu"))
    model.add(MaxPool1D(pool_size=3))
    model.add(Conv1D(32, 3, padding='same', activation='relu'))
    model.add(MaxPool1D(pool_size=3))
    model.add(Conv1D(32, 3, padding='same', activation='relu'))
    model.add(MaxPool1D(pool_size=3))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy', 'Precision', 'Recall'])

    return model

def model_cnn():
    print(Fore.MAGENTA + 'Initializing CNN' + Fore.MAGENTA)

    max_len = 300

    vocab_size, X_train_token, X_test_token = tokenize_data()
    X_train, X_test, y_train, y_test = split_data()
    X_train_pad, X_test_pad = pad_data(X_train_token, X_test_token, max_len)

    model = initialize_model(vocab_size=vocab_size, max_len=max_len)
    print(Fore.MAGENTA + 'CNN initialized' + Fore.MAGENTA)

    es = EarlyStopping(patience=10, restore_best_weights=True)
    filepath = "Data/checkpoint/cnn_model.h5"
    model_checkpoint= ModelCheckpoint(filepath=filepath, save_best_only=True, monitor='val_accuracy')
    history = model.fit(
        X_train_pad, y_train,
        epochs=50,  # Use early stopping in practice
        batch_size=32,
        validation_split=0.2,
        callbacks=[es, model_checkpoint],
        verbose=1)

    print(history)

    y_pred = model.predict(X_test_pad) # Make cross validated predictions of entire dataset
    print(classification_report(y_test,(y_pred > 0.5).astype(int))) # Pass predictions and true values to Classification report

    #save_model()
