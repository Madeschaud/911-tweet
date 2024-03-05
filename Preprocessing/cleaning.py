from Preprocessing.sub_cleaning import Fore, remove_nb, remove_punctuation, remove_space_lowercase, remove_stop_words, lemmatize, remove_rt, remove_at, remove_hashtags
from nltk import word_tokenize
def sentence_cleaning(sentence):
    '''Cleaning function ! Put True or False to enable or disable some part of the cleaning'''
    # print(Fore.BLUE + 'Cleaning in preprocess')
    sentence = remove_rt(sentence)
    sentence = remove_hashtags(sentence)
    sentence = remove_at(sentence)
    sentence = remove_space_lowercase(sentence, True)
    sentence = remove_nb(sentence, True)
    sentence = remove_punctuation(sentence, True)
    sentence = word_tokenize(sentence)
    sentence = remove_stop_words(sentence, True)
    sentence = lemmatize(sentence, True)
    sentence = ' '.join(sentence)
    return sentence
