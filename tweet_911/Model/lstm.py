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

    #lstm
    #model.add(LSTM(units=64, return_sequences=True, activation= 'tanh'))
    #model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(8, activation='tanh',kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01)))

    #compile
    model.compile(loss='binary_crossentropy', optimizer= 'rmsprop', metrics= ['accuracy', 'Recall', 'Precision'])
    return model


# def model_lstm():
#     print("ici lstm")
#     # set params
#     max_features =10000
#     max_len=300
#     embedding_dim=50

#     X_train, X_test,y_train, y_test = split_data()
#     vocab_size, X_train_token, X_test_token = tokenize_data(X_train, X_test)


#     # Pad the inputs
#     X_train_pad = pad_sequences(X_train_token, dtype='float32', padding='post',maxlen=max_len)
#     X_test_pad = pad_sequences(X_test_token, dtype='float32', padding='post',maxlen=max_len)
#     model = initialize_model(vocab_size)

#     #initialize
#     #print(Fore.MAGENTA + 'Le lstm est lancé' + Fore.MAGENTA)
#     es = EarlyStopping(patience=20, restore_best_weights=True)


#     # Train the model
#     checkpoint_path = 'Data/checkpoint'
#     check = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True)
#     history = model.fit(X_train_pad,
#                     y_train, batch_size=32,
#                     epochs=50,
#                     shuffle=True,
#                     validation_split = 0.2, #IMPORTANT éviter le data leakage
#                     callbacks = [es, check],
#                     verbose = 1)


#     # Evaluate the model
#     #loss, accuracy = model.evaluate(X_test_pad, y_test)
#     #print(f'Test loss: {loss:.4f}')
#     #print(f'Test accuracy: {accuracy:.4f}')
#     #print(f'history:{history:.4f}')
#     return model, X_test_pad, y_test


# model, X_test_pad,y_test = model_lstm()
# y_pred_proba = model.predict(X_test_pad)
# y_pred= y_pred_proba > 0.6
# print(classification_report(y_test,y_pred)) # Pass predictions and true values to Classification report
