from ailoganalyzer.dataset.utils_semantic_vec import camel_to_snake, replace_all_blank, lemmatize_stop
import numpy as np
from collections import Counter
import math
import dbm
from functools import lru_cache
import pathlib

vec_file = str(pathlib.Path(__file__).parent.resolve() / "vec")

word_vec = dbm.open(vec_file)


def calcul_TFIDF(template, word_count):

    # TF:
    word_counter = Counter(template)
    l_template = len(template)
    tf_map = {word: (count / l_template) for word, count in word_counter.items()}

    # IDF:
    word_num = sum(word_count.values())
    idf_map = {word: math.log(word_num / word_count[word]) for word in set(template)}

    # TF_IDF:
    tf_idf = {word: tf * idf for word, tf, idf in zip(tf_map, tf_map.values(), idf_map.values())}

    return tf_idf


def template_to_vec(word_vec, template):
    vec = line_to_vec(word_vec, [template], train=False)
    return vec["0"]


@lru_cache(maxsize=2048)
def preprocess_template(template):
    result = {}

    # extract relevant words
    temp = template
    temp = camel_to_snake(temp)
    temp = replace_all_blank(temp)
    temp = " ".join(temp.split())
    temp = lemmatize_stop(temp)
    result = [word for word in temp if word in word_vec]

    return result


def line_to_vec(template, word_count):
    fasttext = word_vec

    # 1 ..
    result = preprocess_template(template)

    # 2 ..

    count = calcul_TFIDF(result, word_count)

    # 3 ..

    template = result

    value_sum = sum(count.values())
    for word in count.keys():
        count[word] = count[word]/value_sum

    semantic_vec = np.zeros(300)
    for word in count.keys():
        fasttext_weight = np.array([float(e) for e in fasttext[word].decode().split()], dtype=float)
        semantic_vec += count[word]*fasttext_weight

    return semantic_vec
