from colorama import Fore
from Preprocessing.cleaning import cleaning
from Preprocessing.tokenizer import tokenizer


def preprocessor_all():
    '''Preprocessor function - 1st step : cleaning'''
    print(Fore.GREEN + 'Prepocessor function' + Fore.WHITE)
    cleaning()
    tokenizer()
