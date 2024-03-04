from colorama import Fore
import pandas as pd

def vectorizer(data):
    print(Fore.Blue + 'Vectorizer in place')
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=10000)

    vectorized_documents = vectorizer.fit_transform(data.tweet_clean)
    vectorized_documents = pd.DataFrame(
        vectorized_documents.toarray(),
        columns = vectorizer.get_feature_names_out()
        )

    vectorized_documents
