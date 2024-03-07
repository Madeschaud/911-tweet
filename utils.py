from sklearn.model_selection import CrossValidate, cross_val_score, KFold
import numpy as np

from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint

from tweet_911.Model.lstm import initialize_model

def cross_val_hand(vocab_size, X_pad, y, embedding_dim):
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
        model = initialize_model(vocab_size, embedding_dim)

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
    return np.concatenate(all_predictions).ravel()
