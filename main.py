from Preprocessing.preprocessor import preprocessor
from colorama import Fore
import pandas
import os

def main():
    print(Fore.RED + 'Main' + Fore.WHITE)
    if not os.path.isfile('Data/clean_data.csv'):
        preprocessor()

    data_cleaned = pandas.read_csv('Data/clean_data.csv')
    data_cleaned['words_per_tweet'] = data_cleaned['tweet_clean'].apply(nbwords)
    print(data_cleaned)
    print(data_cleaned['words_per_tweet'].mean())

def nbwords(sentence):
    if isinstance(sentence, str) :
        return len(sentence.split())
    return 0
