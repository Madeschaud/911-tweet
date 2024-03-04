from Preprocessing.cleaning import Fore, remove_nb, remove_punctuation, remove_space_lowercase, remove_stop_words, word_tokenize
from nltk.stem import WordNetLemmatizer



def cleaning(sentence):
    '''Cleaning function ! Put True or False to enable or disable some part of the cleaning'''
    print(Fore.BLUE + 'Cleaning in preprocess')
    sentence = remove_space_lowercase(sentence, True)
    sentence = remove_nb(sentence, True)
    sentence = remove_punctuation(sentence, True)
    sentence = word_tokenize(sentence, True)
    sentence = remove_stop_words(sentence, True)

    # Lemmatize
    sentence = [WordNetLemmatizer().lemmatize(word) for word in sentence]

    return ' '.join(sentence)

def nbwords(sentence):
    return len(sentence.split())
