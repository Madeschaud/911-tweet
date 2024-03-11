from tweet_911.Preprocessing.preprocessor import preprocessor_all
from colorama import Fore
import os
import pandas
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tweet_911.Model import cnn, bidirection_lstm, simple_gru, cnn_rnn, lstm, baseline, boost_naive_base
from tweet_911.registry import *
from tweet_911.Model.utils import split_data, tokenize_data, pad_data, evaluate_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report

import pickle

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


# def main():
#     print(Fore.RED + 'Main' + Fore.WHITE)
    # if not os.path.isfile('tweet_911/Data/clean_data.csv'):
    #     preprocessor_all()
    # data_cleaned = pandas.read_csv('tweet_911/Data/clean_data.csv', index_col=0)

    # hist_word_distrib(False, data_cleaned)
    #Turn true or False to activate or disactivate the histplot
        # Create (X_train, y_train, X_test y_test)
    # X_train_pad, X_test_pad, vocab_size = preproc_data()
    # pred(vocab_size)
    # data_cleaned["test_regex"]= data_cleaned["tweet_clean"].str.find('@')
    # print(data_cleaned['test_regex'].value_counts())


@mlflow_run
def train(
        validation_split: float = 0.2,
        batch_size = 32,
        patience = 5,
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
    X_train, X_test, y_train, y_test = split_data()
    vocab_size, X_train_token, X_test_token, tokenizer = tokenize_data(X_train, X_test)
    X_train_pad, X_test_pad = pad_data(X_train_token, X_test_token)

    #Save XTest & Ytest
    pd.DataFrame(X_test_pad).to_csv('tweet_911/Data/Test/xtestpad.csv', index=True)
    y_test.to_csv('tweet_911/Data/Test/ytest.csv', index=True)

    # saving
    with open('tweet_911/Data/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
    check = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True)
    epochs = 10
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
    if MODEL_TARGET == 'mlflow':
        mlflow_transition_model('None', 'Staging')

    print("✅ train() done \n")

    return val_acc, val_rec, val_precision


@mlflow_run
def evaluate(
    stage: str = "Staging"
):
    """
    Evaluate the performance of the latest production model on processed data
    Return MAE as a float
    """
    print(Fore.MAGENTA + "\n⭐️ Launching Eval model" + Style.RESET_ALL)
    model = load_model(stage=stage)
    assert model is not None

    Xtest=pd.read_csv('tweet_911/Data/Test/xtestpad.csv', index_col=0)
    ytest=pd.read_csv('tweet_911/Data/Test/ytest.csv', index_col=0)

    if Xtest.shape[0] == 0 or ytest.shape[0] == 0:
        print("❌ No data to evaluate on")
        return None

    metrics_dict = evaluate_model(model=model, X=Xtest, y=ytest)
    print(metrics_dict)
    # mae = metrics_dict["mae"]

    params = dict(
        context="evaluate", # Package behavior
        row_count=len(Xtest)
    )

    save_results(params=params, metrics=metrics_dict)

    print("✅ evaluate() done \n")

    # return mae

if __name__ == '__main__':
    # train()
    evaluate()
    # pred()
