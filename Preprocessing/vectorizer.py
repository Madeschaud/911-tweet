from colorama import Fore
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.naive_bayes import MultinomialNB
from tempfile import mkdtemp



def vectorizer(data=pd.read_csv('Data/clean_data.csv', index_col=0)):
    print(Fore.BLUE + 'Vectorizer in place' + Fore.WHITE)

    X = data[['tweet_clean']]
    y = data.actionable

    # Split into Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    cachdir=mkdtemp()
    # Create Pipeline for Naive Bayes
    mnb_pipe = make_pipeline(
        TfidfVectorizer(),
        MultinomialNB(),
        memory=cachdir
    )
    mnb_pipe.get_params()

    params = {
        'multinomialnb__alpha': (0.15, 0.2, 0.25),
        'tfidfvectorizer__ngram_range': ((1,1), (2,2)),
        'tfidfvectorizer__min_df': (0.003,0.004, 0.005, 0.006),
        'tfidfvectorizer__max_df': (0.6, 0.65,  0.7, 0.75)
    }
    scoring = ['accuracy', 'precision', 'recall']

    # Perform grid search on pipeline
    grid_search = GridSearchCV(
        mnb_pipe,
        param_grid=params,
        cv=10,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )

    random_search = RandomizedSearchCV(
        mnb_pipe,
        params,
        cv=10,
        scoring=scoring,
        n_iter=100,
        refit="accuracy",
        n_jobs=-1
    )

    # grid_search.fit(X_train.tweet_clean, y_train)
    random_search.fit(X_train.tweet_clean, y_train)
    i = random_search.best_index_
    best_precision = random_search.cv_results_['mean_test_precision_macro'][i]
    best_recall = random_search.cv_results_['mean_test_recall_macro'][i]

    print('Best score (accuracy): {}'.format(random_search.best_score_))
    print('Mean precision: {}'.format(best_precision))
    print('Mean recall: {}'.format(best_recall))
    print('Best parametes: {}'.format(random_search.best_params_))
