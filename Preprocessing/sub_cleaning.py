from colorama import Fore
import string
from nltk.stem import WordNetLemmatizer
import re

from nltk.corpus import stopwords

def remove_space_lowercase(sentence, activate):
    if activate is True : return sentence.strip().lower()
    return sentence

def remove_nb(sentence, activate):
    if activate is True : return ''.join(char for char in sentence if not char.isdigit())
    return sentence

def remove_punctuation(sentence, activate):
    if activate is True :
        for punctuation in string.punctuation:
            sentence = sentence.replace(punctuation, '')
        return sentence

    return sentence

def remove_stop_words(sentence, activate):
    if activate is True :
        # Define stopwords
        stop_words = set(stopwords.words('english'))
        # Remove stopwords
        return [w for w in sentence if not w in stop_words]
    return sentence

def lemmatize(sentence, activate):
    if activate is True :
        return [WordNetLemmatizer().lemmatize(word) for word in sentence]
    return sentence

def remove_hashtags(sentence):
    hashtag_regex = r'#\w+'
    return re.sub(hashtag_regex, '', sentence)

def remove_at(sentence):
    at_regex = r'@\w+'
    return re.sub(at_regex, '', sentence)

def remove_rt(sentence):
    rt_regex = r'\bRT\b\s?:?@\w+'
    if sentence[0:2] == 'RT ':
        return sentence[3:]
    return re.sub(rt_regex, '', sentence)
