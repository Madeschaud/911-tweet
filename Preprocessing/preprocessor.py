from colorama import Fore
from Preprocessing.cleaning import sentence_cleaning
from Model.boost_naive_base import vectorizer
import os
import pandas

def preprocessor_all():
    '''Preprocessor function'''
    print(Fore.GREEN + 'Prepocessor function'+Fore.WHITE)

    #Clean the data
    data = pandas.read_csv('Data/tweet_data_consolidated.csv')
    data['tweet_clean'] = data['tweet_text'].apply(sentence_cleaning)
    data.dropna(subset=['tweet_clean'], inplace=True)
    data = data.drop_duplicates()

    #Add Words_per_tweet column
    data['words_per_tweet'] = data['tweet_clean'].apply(nbwords)
    data = data[~(data['words_per_tweet'] == 0)]
    data.drop(columns=['index'], inplace=True)

    #Add Actionable column
    action = ['missing_or_found_people', 'requests_or_urgent_needs', 'injured_or_dead_people', 'infrastructure_and_utility_damage', 'displaced_people_and_evacuations', 'infrastructure_and_utilities_damage', 'affected_individual', 'displaced_and_evacuations', 'missing_and_found_people', 'requests_or_needs']
    data['actionable'] = data['class_label'].apply(lambda x: 1 if x in action else 0)

    data.to_csv('Data/clean_data.csv', index=True)

def nbwords(sentence):
    if isinstance(sentence, str) :
        return len(sentence.split())
    return 0
