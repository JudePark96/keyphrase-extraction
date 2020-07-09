__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


from typing import Sequence
from nltk.stem.lancaster import LancasterStemmer


import re


stemmer = LancasterStemmer()


def get_stemed_tokens(sequence: Sequence[str]) -> list:
    """
    stemming given sequence which is in english.
    :param sequence: list of words.
    :return: list of stemming words.
    """
    return [stemmer.stem(word.strip()) for word in sequence]


def get_tokens(text) -> Sequence[str]:
    '''
    The tokenizer used in Meng et al. ACL 2017
    parse the feed-in text, filtering and tokenization
    keep [_<>,\(\)\.\'%], replace digits to <digit>, split by [^a-zA-Z0-9_<>,\(\)\.\'%]
    :param text:
    :return: a list of tokens
    '''
    # remove line breakers
    text = re.sub(r'[\r\n\t]', ' ', text)
    # pad spaces to the left and right of special punctuations
    text = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', text)
    # tokenize by non-letters (new-added + # & *, but don't pad spaces, to make them as one whole word)
    tokens = list(filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,#&\+\*\(\)\.\']', text)))

    return tokens


def replace_numbers_to_DIGIT(tokens, k=1) -> Sequence[str]:
    # replace big numbers (contain more than k digit) with <digit>
    tokens = [w if not re.match('^\d{%d,}$' % k, w) else '<digit>' for w in tokens]

    return tokens


if __name__ == '__main__':
    print(get_stemed_tokens(['hellos', 'worlds']))
    print(get_tokens('hello my name is eunhwan park. (Hey) 412414'))
    print(replace_numbers_to_DIGIT(get_tokens('hello my name is eunhwan park. (Hey) 1')))