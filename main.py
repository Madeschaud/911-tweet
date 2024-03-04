from Preprocessing.preprocessor import preprocessor_all
from colorama import Fore
import pandas
import os
from Add_data.new_data import download_data

def main():
    print(Fore.RED + 'Main' + Fore.WHITE)
    if not os.path.isfile('Data/clean_data.csv'):
        preprocessor_all()

    data_cleaned = pandas.read_csv('Data/clean_data.csv')

    print(data_cleaned)
    print(data_cleaned['words_per_tweet'].mean())
