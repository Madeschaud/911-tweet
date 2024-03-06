# Basic Imports
import numpy as np
import pandas as pd



# tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras import layers

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

#sk
from sklearn.model_selection import train_test_split

# import data
data =pd.read_csv('/Users/mathildedeschaud/code/Madeschaud/911-tweet/Data/clean_data.csv')
data.head()

def split_data():
    #identify X,y
    X = data['tweet_clean']
    y = data.actionable

    #split data
    return train_test_split( X, y, test_size=0.30, random_state=42)

def tokenize_data(X_train):
    # tokenize
    tk =Tokenizer()
    tk.fit_on_texts(X_train)
    vocab_size = len(tk.word_index)
    print(f'There are {vocab_size} different words in your corpus')

    return vocab_size, tk.texts_to_sequences(X_train)

def initialize_model(vocab_size):
    # build model
    model = Sequential()
    #embedding
    model.add(Embedding(input_dim=vocab_size+1,output_dim=2, mask_zero=True))

    #lstm
    model.add(LSTM(units=64, return_sequences=True))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(16))
    model.add(Dense(1, activation='sigmoid'))

    #compile
    model.compile(loss='binary_crossentropy', optimizer= 'rmsprop', metrics=['accuracy'])
    return model

def model_lstm():
    # set params
    max_features =10000
    max_len=300
    embedding_dim=50

    X_train, X_test,y_train, y_test = split_data()
    vocab_size, X_train_token = tokenize_data(X_train)


    # Pad the inputs
    X_pad = pad_sequences(X_train_token, dtype='float32', padding='post',maxlen=max_len)

    model = initialize_model(vocab_size)

    # Train the model
    model.fit(X_pad, y_train, batch_size=128, epochs=5, validation_split=0.2)
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test loss: {loss:.4f}')
    print(f'Test accuracy: {accuracy:.4f}')


##Epoch 1/5
#515/515 [==============================] - 215s 410ms/step - loss: 0.3959 - accuracy: 0.8355 - val_loss: 0.3080 - val_accuracy: 0.8773
#Epoch 2/5
#15/515 [==============================] - 215s 418ms/step - loss: 0.2772 - accuracy: 0.8911 - val_loss: 0.2918 - val_accuracy: 0.8822
#Epoch 3/5
#515/515 [==============================] - 221s 428ms/step - loss: 0.2585 - accuracy: 0.8988 - val_loss: 0.2953 - val_accuracy: 0.8813
#Epoch 4/5
#515/515 [==============================] - 218s 424ms/step - loss: 0.2484 - accuracy: 0.9030 - val_loss: 0.2923 - val_accuracy: 0.8821
#Epoch 5/5
#515/515 [==============================] - 215s 418ms/step - loss: 0.2403 - accuracy: 0.9054 - val_loss: 0.2931 - val_accuracy: 0.8811
#2024-03-05 17:18:29.438890: W tensorflow/core/framework/op_kernel.cc:1816] OP_REQUIRES failed at cast_op.cc:122 : UNIMPLEMENTED: Cast string to float is not supported
