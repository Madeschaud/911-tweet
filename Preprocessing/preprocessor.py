from Preprocessing.cleaning import cleaning
from colorama import Fore
import pandas

def preprocessor():
    '''Preprocessor function - 1st step : cleaning'''
    print(Fore.GREEN + 'Prepocessor function')
    data = pandas.read_csv('Data/disaster_tweet.csv')
    data['tweet_clean'] = data['tweet_text'].apply(cleaning)
    data = data.dropna()
    data = data.drop_duplicates()
    data = data.drop
    data.to_csv('Data/clean_data.csv', index=True)

    print(data)
