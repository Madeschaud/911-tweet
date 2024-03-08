from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from colorama import Fore
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import recall_score

def baseline_naive():
    print(Fore.BLUE + 'Baseline Naive' + Fore.WHITE)

    data=pd.read_csv('tweet_911/Data/clean_data.csv', index_col=0)

    # Feature/Target
    X = data['tweet_clean']
    y = data.actionable

    # Pipeline vectorizer + Naive Bayes
    pipeline_naive_bayes = make_pipeline(
        TfidfVectorizer(),
        MultinomialNB()
    )

    # Cross-validation
    cv_results = cross_validate(pipeline_naive_bayes, X, y, cv = 5, scoring = ["recall"])
    average_recall = cv_results["test_recall"].mean()
    print(np.round(average_recall,2))
