from Preprocessing.preprocessor import preprocessor_all
from colorama import Fore
import os
import pandas

def main():
    print(Fore.RED + 'Main' + Fore.WHITE)
    if not os.path.isfile('Data/clean_data.csv'):
        preprocessor_all()

    data_cleaned = pandas.read_csv('Data/clean_data.csv', index_col=0)

    print(data_cleaned.columns)
    print(len(data_cleaned))
    print(data_cleaned['words_per_tweet'].mean())
