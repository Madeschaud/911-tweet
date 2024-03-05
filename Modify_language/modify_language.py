from colorama import Fore
import nltk
from nltk.corpus import words
# Basic Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set Style of Viz
sns.set_style("darkgrid")
# sns.set_palette(palette='dark:#5A9_r')


def modify_language():
    '''Modify the language to only have english'''
    print(Fore.GREEN + 'Modify language function' + Fore.WHITE)
    nltk.download('words')
    data_cleaned = pd.read_csv('Data/clean_data.csv', index_col=0)
    english_vocab = set(words.words())
    only_english = []
    for tweet in data_cleaned['tweet_clean']:
        filtered_words = [word for word in tweet.split() if word.lower() in english_vocab]
        # new_list.append(filtered_words)
        only_english.append(' '.join(filtered_words))
        # print(filtered_words)
    data_cleaned['only_english'] = only_english
    data_cleaned['compare'] = data_cleaned['only_english'] == data_cleaned['tweet_clean']

    # data_cleaned.to_csv('Data/clean_english_data.csv', index=True)
    print(data_cleaned['compare'].value_counts())
    # print(' '.join(words_filtered))
