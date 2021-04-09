# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 10:54:57 2019

@author: lidongxu1
"""
import re
import spacy
import json
import numpy as np
import io
from tqdm import tqdm

options = {}
options["preprocess_dir"] = "/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/preprocess"

def data_read(filepath):
    fp = open(filepath, "r")
    datas = []  # 存储处理后的数据
    lines = fp.readlines()  # 读取整个文件数据
    i = 0  # 为一行数据
    for line in lines:
        row = line.strip('\n') # 去除两头的换行符，按空格分割
        if i != 0:
            datas.append(row)
        i = i + 1
    fp.close()
    return datas

def camel_to_snake(name):
    """
    # To handle more advanced cases specially (this is not reversible anymore):
    # Ref: https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    """
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def replace_all_blank(value):
    """
    去除value中的所有非字母内容，包括标点符号、空格、换行、下划线等
    :param value: 需要处理的内容
    :return: 返回处理后的内容
    # https://juejin.im/post/5d50c132f265da03de3af40b
    # \W 表示匹配非数字字母下划线
    """
    result = re.sub('\W+', ' ', value).replace("_", ' ')
    result = re.sub('\d',' ',result)
    return result
# https://github.com/explosion/spaCy
# https://github.com/hamelsmu/Seq2Seq_Tutorial/issues/1
nlp = spacy.load('en_core_web_sm')
def lemmatize_stop(text):
    """
    https://stackoverflow.com/questions/45605946/how-to-do-text-pre-processing-using-spacy
    """
#    nlp = spacy.load('en_core_web_sm')
    document = nlp(text)
    # lemmas = [token.lemma_ for token in document if not token.is_stop]
    lemmas = [token.text for token in document if not token.is_stop]
    return lemmas

def dump_2_json(dump_dict, target_path):
    '''
    :param dump_dict: submits dict
    :param target_path: json dst save path
    :return:
    '''
    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, bytes):
                return str(obj, encoding='utf-8')
            return json.JSONEncoder.default(self, obj)

    file = open(target_path, 'w', encoding='utf-8')
    file.write(json.dumps(dump_dict, cls=MyEncoder, indent=4))
    file.close()

# https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(tokens[1:], dtype=np.float)
    return data





fasttext = load_vectors('/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/cc.en.300.vec')



data = data_read("/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/preprocess/templates.csv")
print(data)
result = {}
for i in range(len(data)):
    temp = data[i]
    temp = camel_to_snake(temp)
    temp = replace_all_blank(temp)
    temp = " ".join(temp.split())
    temp = lemmatize_stop(temp)
    result[i] = temp
print(result)


motok = set(fasttext.keys())
for e in result:
    result[e] = list(set(result[e]) & motok)

dump_2_json(result, '/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/preprocess/eventid2template.json')

# 单独保存需要用到的fasttext词向量
template_set = set()
for key in result.keys():
    for word in result[key]:
        template_set.add(word)

template_fasttext_map = {}

for word in template_set:
    if word in fasttext:
        template_fasttext_map[word] = list(fasttext[word])

dump_2_json(template_fasttext_map,'/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/preprocess/fasttext_map.json')
