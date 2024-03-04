from colorama import Fore
from Preprocessing.cleaning import cleaning
from Preprocessing.vectorizer import vectorizer
import os
import pandas

def preprocessor_all():
    '''Preprocessor function - 1st step : cleaning'''
    print(Fore.GREEN + 'Prepocessor function' + Fore.WHITE)

    if not os.path.isfile('Data/clean_data.csv'):
        cleaning()

    data_cleaned = pandas.read_csv('Data/clean_data.csv')
    #vectorizer(data_cleaned)
