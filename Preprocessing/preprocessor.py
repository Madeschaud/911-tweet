from Preprocessing.cleaning import cleaning
from colorama import Fore
import pandas

def preprocessor():
    '''Preprocessor function - 1st step : cleaning'''
    print(Fore.GREEN + 'Prepocessor function')
    data = pandas.read_csv('Data/tweet_data_consolidated.csv')
    data['tweet_clean'] = data['tweet_text'].apply(cleaning)
    data.dropna(subset=['tweet_clean'], inplace=True)
    data = data.drop_duplicates()

    data['words_per_tweet'] = data['tweet_clean'].apply(nbwords)
    data = data[~(data['words_per_tweet'] == 0)]
    data.drop(columns=['index'], inplace=True)
    data.to_csv('Data/clean_data.csv', index=True)

def nbwords(sentence):
    if isinstance(sentence, str) :
        return len(sentence.split())
    return 0
