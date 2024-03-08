from colorama import Fore
from tweet_911.Model.simple_gru import initialize_model, tokenize_data, pad_data, split_data


def load_model_weights(vocab_size):

    '''
    Loading weights
    '''

    print(Fore.CYAN + '🏋️‍♂️🏋️‍♀️ Loading weights ...' + Fore.CYAN)

    # Initialisation des variables et du modèle
    # vocab_size, X_train_token, X_test_token = tokenize_data()
    # X_train_pad, X_test_pad = pad_data(X_train_token, X_test_token)
    # X_train, X_test, y_train, y_test = split_data()

    model = initialize_model(vocab_size)

    # Path du fichier de sauvegarde des poids
    weights_path = 'tweet_911/Data/weights/model_gru_V3.h5'

    # Loading
    model.load_weights(weights_path)
    model.compile(loss = 'binary_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy', 'Recall', 'Precision'])

    # Evaluate
    # result = model.evaluate(X_test_pad, y_test)

    # print(result)
    return model
