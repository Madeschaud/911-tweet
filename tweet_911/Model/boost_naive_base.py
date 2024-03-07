from colorama import Fore
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from tempfile import mkdtemp

def my_tokenizer(X):
    newlist = []
    for alist in X:
        newlist.append(alist[0].split(' '))
    return newlist

def boost_naive_base(data=pd.read_csv('Data/clean_data.csv', index_col=0)):
    print(Fore.BLUE + 'Boost Naive Base' + Fore.WHITE)

    X = data[['tweet_clean']]
    y = data.actionable

    # Split into Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    cachdir=mkdtemp()
    # Create Pipeline for Naive Bayes
    mnb_pipe = make_pipeline(
        # CountVectorizer(tokenizer=my_tokenizer),
        TfidfVectorizer(),
        MultinomialNB(),
        memory=cachdir
    )
    mnb_pipe.get_params()

    # Perform Random Search on pipeline
    params_rand = {
        'multinomialnb__alpha': np.linspace(0.1, 1),
        'tfidfvectorizer__ngram_range': ((1,1), (1,2), (2,2)),
        'tfidfvectorizer__min_df': np.linspace(0.001 , 0.5),
        'tfidfvectorizer__max_df': np.linspace(0.501 , 1)
        }


    scoring = ['accuracy', 'precision', 'recall']

    # Perform grid search on pipeline
    grid_search = GridSearchCV(
        mnb_pipe,
        param_grid=params_rand,
        cv=10,
        scoring=scoring,
        n_jobs=-1,
        refit="accuracy",
        verbose=1
    )

    # Perform Random Search on pipeline
    random_search = RandomizedSearchCV(
        mnb_pipe,
        params_rand,
        cv=10,
        scoring=scoring,
        n_iter=100,
        refit="accuracy",
        n_jobs=-1
    )

    # grid_search.fit(X_train.tweet_clean, y_train)
    random_search.fit(X_train.tweet_clean, y_train)
    i = random_search.best_index_
    best_precision = random_search.cv_results_['mean_test_precision'][i]
    best_recall = random_search.cv_results_['mean_test_recall'][i]


    print('Best score (accuracy): {}'.format(random_search.best_score_))
    print('Mean precision: {}'.format(best_precision))
    print('Mean recall: {}'.format(best_recall))
    print('Best parametes: {}'.format(random_search.best_params_))
