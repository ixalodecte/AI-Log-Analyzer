import re
import spacy

nlp = spacy.load('en_core_web_sm')


def camel_to_snake(name):
    """
    https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    """
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def replace_all_blank(value):

    result = re.sub(r'\W+', ' ', value).replace("_", ' ')
    result = re.sub(r'\d', ' ', result)
    return result
# https://github.com/explosion/spaCy
# https://github.com/hamelsmu/Seq2Seq_Tutorial/issues/1


def lemmatize_stop(text):
    """
    https://stackoverflow.com/questions/45605946/how-to-do-text-pre-processing-using-spacy
    """

    document = nlp(text)
    lemmas = [token.text for token in document if not token.is_stop]
    return lemmas
