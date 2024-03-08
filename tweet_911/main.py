from Preprocessing.preprocessor import preprocessor_all
from colorama import Fore
import os
import pandas
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from Model import cnn, bidirection_lstm, simple_gru, cnn_rnn, lstm, baseline, boost_naive_base
from registry import *
from Model.utils import split_data, tokenize_data, pad_data
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report


def hist_word_distrib(action, data_cleaned):
    # Histogram of Words Distribution
    if action is True:
        plt.figure(figsize=(10, 6))
        sns.histplot(data_cleaned['words_per_tweet'], kde=True)
        plt.title('Histogram of Words Distribution')
        # plt.xlim(0,50)
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.show()
        print('mean = ',data_cleaned['words_per_tweet'].mean())
        print('min = ',data_cleaned['words_per_tweet'].min())
        print('max = ', data_cleaned['words_per_tweet'].max())


def main():
    print(Fore.RED + 'Main' + Fore.WHITE)
    if not os.path.isfile('Data/clean_data.csv'):
        preprocessor_all()
    data_cleaned = pandas.read_csv('Data/clean_data.csv', index_col=0)

    #Turn true or False to activate or disactivate the histplot
    hist_word_distrib(False, data_cleaned)

    # data_cleaned["test_regex"]= data_cleaned["tweet_clean"].str.find('@')
    # print(data_cleaned['test_regex'].value_counts())


#@mlflow_run


def train(
        validation_split: float = 0.2,
        batch_size = 32,
        patience = 20,
        embedding_dim = 50
    ) -> float:

    """
    - Use clean_data.csv as source of data (already processed)
    - Initialize model based on .env selection
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_accuracy, val_precision & val_recall as floats
    """
    print("Launching train function")
    # Create (X_train, y_train, X_test y_test)
    X_train, X_test, y_train, y_test = split_data()
    vocab_size, X_train_token, X_test_token = tokenize_data(X_train, X_test)
    X_train_pad, X_test_pad = pad_data(X_train_token, X_test_token)

    # Train model using `model.py`
    model = load_model()

    if model is None:
        print(Fore.MAGENTA + f'no {MLFLOW_EXPERIMENT} saved, initializing now' + Fore.MAGENTA)
        f'{MLFLOW_EXPERIMENT}.initialize_model(vocab_size=vocab_size, embedding_dim=embedding_dim)'
        model = eval(f'{MLFLOW_EXPERIMENT}.initialize_model(vocab_size=vocab_size, embedding_dim=embedding_dim)')
        print(Fore.MAGENTA + f'{MLFLOW_EXPERIMENT} initialized' + Fore.MAGENTA)

    print(Fore.YELLOW + f'{MLFLOW_EXPERIMENT} is training' + Fore.YELLOW)
    es = EarlyStopping(patience=patience, restore_best_weights=True, monitor='val_precision')

    checkpoint_path = os.path.join(f'Data/checkpoint/{MLFLOW_MODEL_NAME}-model-{MLFLOW_EXPERIMENT}','-{epoch:02d}-{val_accuracy:.2f}.hdf5')
    check = ModelCheckpoint(checkpoint_path, monitor='val_recall ', verbose=1, save_best_only=True)
    epochs = 1
    history = model.fit(
        X_train_pad, y_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        validation_split = validation_split, #IMPORTANT éviter le data leakage
        callbacks = [es, check],
        verbose = 1)

    val_acc = np.min(history.history['val_accuracy'])
    val_rec = np.min(history.history['val_recall'])
    val_precision = np.min(history.history['val_precision'])

    params = dict(
        context="train",
        vocab_size=vocab_size,
        batch_size=batch_size,
        patience=patience,
        validation_split=validation_split,
        epochs=epochs,
        embedding_dim=embedding_dim
    )

    # Save results on the hard drive using taxifare.ml_logic.registry
    save_results(params=params, metrics=dict(accuracy=val_acc, recall=val_rec, precision=val_precision))

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model, local_model_name={MLFLOW_EXPERIMENT})

    # The latest model should be moved to staging
    mlflow_transition_model('None', 'Staging')

    print("✅ train() done \n")

    return val_acc, val_rec, val_precision


#@mlflow_run
def evaluate(stage: str = "Production") -> float:
    """
    Evaluate the performance of the latest production model on processed data
    Return ACCURACY, PRECISION, RECALL as a float
    """
    # Create (X_train, y_train, X_test y_test)
    X_train, X_test, y_train, y_test = split_data()
    vocab_size, X_train_token, X_test_token = tokenize_data()
    X_train_pad, X_test_pad = pad_data(X_train_token, X_test_token)

    model = load_model()
    assert model is not None

    # # Evaluate the model
    # loss, accuracy = model.evaluate(X_test_pad, y_test)
    # print(f'Test loss: {loss:.4f}')
    # print(f'Test accuracy: {accuracy:.4f}')
    y_pred = model.predict(X_test_pad) # Make cross validated predictions of entire dataset
    print(classification_report(y_test,(y_pred > 0.5).astype(int))) # Pass predictions and true values to Classification report

def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    # Check if X_pred exists
    if X_pred is None:
        # DECIDE IF WE PRINT
        pass

    model = load_model()
    assert model is not None

    # Process X_pred
    # Predict y
    pass

if __name__ == '__main__':
    train()
    # evaluate()
    # pred()
