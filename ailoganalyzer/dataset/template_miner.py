# -*- coding: utf-8 -*-

import os
from collections import defaultdict,Counter
import shelve
import math
from functools import lru_cache
import numpy as np
import re

from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig
from drain3 import TemplateMiner

from ailoganalyzer.dataset.utils_semantic_vec import camel_to_snake, replace_all_blank, lemmatize_stop


class TemplateMinerr():
    def __init__(self, checkpoint_file=None, drain_config = None):
        self.persistence_file = checkpoint_file
        if checkpoint_file is not None:
            persistence = FilePersistence(self.persistence_file)
        else:
            persistence = None
        
        if drain_config is not None:
            config = TemplateMinerConfig()
            config.load(drain_config)

            config.profiling_enabled = False
        else:
            config = None
        self.template_miner = TemplateMiner(persistence_handler=persistence, config=config)
        
        self.modified = {}


    def add_log(self, log):
        cluster_id = self.log_to_key(log)
        #self.template_id.append(cluster_id)
        return cluster_id
        
    def log_to_key(self, log):
        result = self.template_miner.add_log_message(log)
        if result["change_type"] != "none":
            pass
        cluster_id = result["cluster_id"] - 1
        return cluster_id

    def get_templates(self):
        return (c.get_template() for c in self.template_miner.drain.clusters)

    def get_number_classes(self):
        return len(list(self.get_templates()))

    def remove_system(self):
        os.remove(self.persistence_file)
        
    def transform(self, log):
        template = self.template_miner.match(log)
        template_id = template.cluster_id - 1
        params = self.template_miner.extract_parameters(
            template.get_template() , log, exact_matching=True)
        if template is None:
            raise ValueError("log can't be matched to any known template")
        return template_id, params
    
    def get_template_by_id(self, idx):
        return list(self.get_templates())[idx]
    
    def __getitem__(self, idx):
        return list(self.get_templates())[idx]
        
class Vectorizer():
    def __init__(self, db_vec):
        self.word2vec = shelve.open(db_vec, "r")
        
    def get_word_counter(self, template_count):
        d = defaultdict(int)
        
        for template,count in template_count.items():
            for word in self.preprocess_template(template):
                d[word] += count
        return d
    
    def calcul_TFIDF(self, template, word_count):

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


    def template_to_vec(self, template, template_count):
        word_counter = self.get_word_counter(template_count)

        
        vec = self.line_to_vec(template, word_counter)
        return vec


    @lru_cache(maxsize=2048)
    def preprocess_template(self, template):
        result = {}

        # extract relevant words
        temp = template
        temp = re.sub("<:.*?:>", "",temp) # Variables for each template are like "<:*:>"
        temp = camel_to_snake(temp)
        temp = replace_all_blank(temp)
        temp = " ".join(temp.split())
        temp = lemmatize_stop(temp)
        result = [word for word in temp if word in self.word2vec]

        return result


    def line_to_vec(self, template, word_count):
        word2vec = self.word2vec
        
        # 1 Extract relevant words
        result = self.preprocess_template(template)

        # 2 ..
        count = self.calcul_TFIDF(result, word_count)

        # 3 ..

        template = result

        value_sum = sum(count.values())
        for word in count.keys():
            count[word] = count[word]/value_sum

        semantic_vec = np.zeros(300)
        for word in count.keys():
            fasttext_weight = np.array(word2vec[word], dtype=np.float32)
            semantic_vec += count[word]*fasttext_weight

        return semantic_vec
    
    def transform(self, log):
        template, params = self.template_finder.transform(log)
        return self[template], params
    
    def __getitem__(self, template_id):
        template = self.template_finder[template_id]
        word_counter = self.get_word_counter()
        if template not in self.template_calculated.values():
            self.template_id_to_vec_dict[template_id] = self.line_to_vec(template, word_counter)
            self.template_calculated[template_id] = template
        return self.template_id_to_vec_dict[template_id]
