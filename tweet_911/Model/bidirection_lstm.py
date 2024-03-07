# Basic Imports
import numpy as np
import pandas as pd
from colorama import Fore

# tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras import layers

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import KFold




#sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict

# import data
data =pd.read_csv('Data/clean_data.csv')
data.head()

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


def initialize_model(vocab_size, embedding_dim):
    # build model
    model = Sequential()
    #embedding
    model.add(Embedding(input_dim=vocab_size+1,output_dim=embedding_dim, mask_zero=True))

    #lstm
    model.add(Bidirectional(LSTM(64, activation='tanh', return_sequences=True)))
    # model.add(Bidirectional(LSTM(64, activation='tanh')))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'Recall', 'Precision'])
    return model

def model_bidirectional_lstm():
    # set params
    max_features =10000
    max_len=300
    embedding_dim=2

    X_train, X_test,y_train, y_test = split_data()
    vocab_size, X_train_token, X_test_token = tokenize_data(X_train, X_test)

    # Pad the inputs
    X_train_pad = pad_sequences(X_train_token, dtype='float32', padding='post',maxlen=max_len)
    X_test_pad = pad_sequences(X_test_token, dtype='float32', padding='post',maxlen=max_len)

    model = initialize_model(vocab_size, embedding_dim)

    #initialize
    print(Fore.MAGENTA + 'Le Bidirectional LSTM est lancé' + Fore.MAGENTA)
    es = EarlyStopping(patience=20, restore_best_weights=True, monitor='val_precision')


    # Train the model
    checkpoint_path = 'Data/checkpoint/model-{epoch:02d}-{val_accuracy:.2f}.hdf5'
    check = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True)
    history = model.fit(X_train_pad,
                    y_train, batch_size=32,
                    epochs=1,
                    shuffle=True,
                    validation_split = 0.2, #IMPORTANT éviter le data leakage
                    callbacks = [es, check],
                    verbose = 1)
    # # Evaluate the model
    # loss, accuracy = model.evaluate(X_test_pad, y_test)
    # print(f'Test loss: {loss:.4f}')
    # print(f'Test accuracy: {accuracy:.4f}')


    y_pred = model.predict(X_test_pad) # Make cross validated predictions of entire dataset
    print(classification_report(y_test,(y_pred > 0.5).astype(int))) # Pass predictions and true values to Classification report
    return model


#  precision    recall  f1-score   support

#            0       0.89      0.86      0.87     25753
#            1       0.65      0.71      0.68      9540

#     accuracy                           0.82     35293
#    macro avg       0.77      0.79      0.78     35293
# weighted avg       0.83      0.82      0.82     35293
