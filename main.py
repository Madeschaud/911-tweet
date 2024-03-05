from Preprocessing.preprocessor import preprocessor
from colorama import Fore
import pandas
import os

def main():
    print(Fore.RED + 'Main' + Fore.WHITE)
    if not os.path.isfile('Data/clean_data.csv'):
        preprocessor()

    data_cleaned = pandas.read_csv('Data/clean_data.csv')

    print(len(data_cleaned))
    print(data_cleaned['words_per_tweet'].mean())
