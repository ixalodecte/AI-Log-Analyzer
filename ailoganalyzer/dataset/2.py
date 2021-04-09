import os
import json
import numpy as np
import pandas as pd
from collections import Counter
import math

def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict

eventid2template = read_json('/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/preprocess/eventid2template.json')
fasttext_map = read_json('/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/preprocess/fasttext_map.json')
print(eventid2template)
dataset = list()
with open('/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/train/train', 'r') as f:
    for line in f.readlines():
        line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
        dataset.append(line)
print(len(dataset))
idf_matrix = list()
for seq in dataset:
    for event in seq:
        idf_matrix.append(eventid2template[str(event)])
print(len(idf_matrix))
idf_matrix = np.array(idf_matrix)
X_counts = []
for i in range(idf_matrix.shape[0]):
    word_counts = Counter(idf_matrix[i])
    X_counts.append(word_counts)
print(X_counts[1000])
X_df = pd.DataFrame(X_counts)
X_df = X_df.fillna(0)
print(len(X_df))
print(X_df.head())
events = X_df.columns
print(events)
X = X_df.values
num_instance, num_event = X.shape

print('tf-idf here')
df_vec = np.sum(X > 0, axis=0)
print(df_vec)
print('*'*20)
print(num_instance)
# smooth idf like sklearn
idf_vec = np.log((num_instance + 1)  / (df_vec + 1)) + 1
print(idf_vec)
idf_matrix = X * np.tile(idf_vec, (num_instance, 1))
X_new = idf_matrix
print(X_new.shape)
print(X_new[1000])

word2idf = dict()
for i,j in zip(events,idf_vec):
    word2idf[i]=j
    # smooth idf when oov
    word2idf['oov'] = (math.log((num_instance + 1)  / (29+1)) + 1)

print(word2idf)
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

dump_2_json(word2idf,'/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/preprocess/word2idf.json')
