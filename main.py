from Preprocessing.preprocessor import preprocessor_all
from colorama import Fore
import os
import pandas
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def hist_word_distrib(action, data_cleaned):
    # Histogram of Words Distribution
    if action is True:
        plt.figure(figsize=(10, 6))
        sns.histplot(data_cleaned['words_per_tweet'], kde=True)
        plt.title('Histogram of Words Distribution')
        # plt.xlim(0,50)
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.show()
        print('mean = ',data_cleaned['words_per_tweet'].mean())
        print('min = ',data_cleaned['words_per_tweet'].min())
        print('max = ', data_cleaned['words_per_tweet'].max())


def main():
    print(Fore.RED + 'Main' + Fore.WHITE)
    if not os.path.isfile('Data/clean_data.csv'):
        preprocessor_all()

    data_cleaned = pandas.read_csv('Data/clean_data.csv', index_col=0)

    #Turn true or False to activate or disactivate the histplot
    hist_word_distrib(False, data_cleaned)

    # data_cleaned["test_regex"]= data_cleaned["tweet_clean"].str.find('@')
    # print(data_cleaned['test_regex'].value_counts())
