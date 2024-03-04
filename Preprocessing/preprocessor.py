from Preprocessing.cleaning import cleaning
from colorama import Fore
import csv
import pandas

def preprocessor():
    '''Preprocessor function - 1st step : cleaning'''
    print(Fore.GREEN + 'Prepocessor function')
    line_count=0
    data = pandas.read_csv('Data/disaster_tweet.csv')
    data['tweet_clean'] = data['tweet_text'].apply(cleaning)
    print(data)
    # with open('Data/disaster_tweet.csv') as file_obj:
    #     reader_obj = csv.reader(file_obj)
        # for row in reader_obj:
        #     #Skip the column line
        #     if line_count == 0:
        #         #Find the index of the tweet_text
        #         tweet_text_index = row.index('tweet_text')
        #     else:
        #         row[tweet_text_index] = cleaning(row[tweet_text_index])

        #     # Kill the function after 3 lines for the test
        #     if line_count == 3:
        #         break

        #     #Count the nb of line
        #     line_count += 1
        #     print(row)
