from colorama import Fore
import string

from nltk import word_tokenize
from nltk.corpus import stopwords

def remove_space_lowercase(sentence, activate):
    if activate is True : return sentence.strip().lower()

def remove_nb(sentence, activate):
    if activate is True : return ''.join(char for char in sentence if not char.isdigit())

def remove_punctuation(sentence, activate):
    if activate is True :
        for punctuation in string.punctuation:
            sentence = sentence.replace(punctuation, '')
        return sentence

def remove_stop_words(sentence, activate):
    if activate is True :
        # Define stopwords
        stop_words = set(stopwords.words('english'))
        # Remove stopwords
        return [w for w in sentence if not w in stop_words]
