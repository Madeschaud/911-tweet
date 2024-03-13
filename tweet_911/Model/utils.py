# from sklearn.model_selection import KFold
import numpy as np
from typing import Tuple
from keras import Model
import mlflow




# from tensorflow import keras
# from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import pandas as pd
from colorama import Fore, Style
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import cross_val_score, train_test_split
# from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.metrics import classification_report

# from tensorflow import keras
# from keras.models import Sequential
# from keras.layers import Embedding, Dense, MaxPool1D, Conv1D, Flatten, Dropout
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.optimizers import Adam, RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
# from keras.metrics import Accuracy, Recall, Precision

# Tentative: Manual cross_val for LSTM model
"""def cross_val_hand(vocab_size, X_pad, y, embedding_dim):
    print('Begin Fold')
    kf = KFold(n_splits=5)
    kf.get_n_splits(X_pad)

    results_eval = []
    all_predictions = []

    for train_index, test_index in kf.split(X_pad):
        # Split the data into train and test
        X_train, X_test = X_pad[train_index], X_pad[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print('Split = ', train_index, test_index)
        # Initialize the model
       #model = initialize_model(vocab_size, embedding_dim)

        es = EarlyStopping(patience=10, restore_best_weights=True)
        checkpoint_path = 'Data/checkpoint/model-{epoch:02d}-{val_accuracy:.2f}.hdf5'

        check = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True)

        history = model.fit(X_train,
                        y_train, batch_size=32,
                        epochs=100,
                        shuffle=True,
                        validation_split = 0.2, #IMPORTANT éviter le data leakage
                        callbacks = [es, check],
                        verbose = 1)


        # Evaluate the model on the testing data
        res = model.evaluate(X_test, y_test, verbose = 0)
        results_eval.append(res)

        #Add prediction on the model
        predictions = model.predict(X_test, verbose=0)
        all_predictions.append(predictions)

    print('End Fold')

    print(results_eval)
    return np.concatenate(all_predictions).ravel() """


# -------- Common functions prior to training model

def split_data() -> tuple:
    data=pd.read_csv('tweet_911/Data/clean_data.csv', index_col=0)
    X = data['tweet_text']
    y = data.disaster_or_not

    # Split into Train/Test
    return train_test_split(X, y, test_size=0.3)

def tokenize_data(X_train, X_test):
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

    return vocab_size, X_train_token, X_test_token, tokenizer

def pad_data(X_train_token, X_test_token, max_len=20):

    X_train_pad = pad_sequences(X_train_token, dtype='float32', padding='post', maxlen=max_len)
    X_test_pad = pad_sequences(X_test_token, dtype='float32', padding='post', maxlen=max_len)

    return X_train_pad, X_test_pad



def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=32
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """
    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)


    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )
    print(metrics)

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]
    recall = metrics["recall"]
    precision = metrics["precision"]

    print(f"✅ results Ok")

    return metrics
