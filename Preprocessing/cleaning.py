from Preprocessing.sub_cleaning import Fore, remove_nb, remove_punctuation, remove_space_lowercase, remove_stop_words, lemmatize
from nltk import word_tokenize
def cleaning(sentence):
    '''Cleaning function ! Put True or False to enable or disable some part of the cleaning'''
    # print(Fore.BLUE + 'Cleaning in preprocess')
    sentence = remove_space_lowercase(sentence, True)
    sentence = remove_nb(sentence, False)
    sentence = remove_punctuation(sentence, False)
    sentence = word_tokenize(sentence)
    sentence = remove_stop_words(sentence, False)
    sentence = lemmatize(sentence, False)

    return ' '.join(sentence)
    # return sentence

def nbwords(sentence):
    return len(sentence.split())
