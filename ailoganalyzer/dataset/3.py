import json
import numpy as np
from collections import Counter

def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict

event2template = read_json('/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/preprocess/eventid2template.json')
fasttext = read_json('/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/preprocess/fasttext_map.json')
word2idf = read_json('/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/preprocess/word2idf.json')


event2semantic_vec = dict()
# todo :
# 计算每个seq的tf，然后计算句向量
for event in event2template.keys():
    template = event2template[event]
    tem_len = len(template)
    count = dict(Counter(template))
    for word in count.keys():
        # TF
        TF = count[word]/tem_len
        # IDF
        IDF = word2idf.get(word,word2idf['oov'])
        # print(word)
        # print(TF)
        # print(IDF)
        # print('-'*20)
        count[word] = TF*IDF
    # print(count)
    # print(sum(count.values()))
    value_sum = sum(count.values())
    for word in count.keys():
        count[word] = count[word]/value_sum
    semantic_vec = np.zeros(300)
    for word in count.keys():
        fasttext_weight = np.array(fasttext[word])
        semantic_vec += count[word]*fasttext_weight
    event2semantic_vec[event] = list(semantic_vec)
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

dump_2_json(event2semantic_vec,'/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/preprocess/event2semantic_vec_sameoov.json')
