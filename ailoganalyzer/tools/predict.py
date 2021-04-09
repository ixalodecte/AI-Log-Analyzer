#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import os
import sys
import time
from collections import Counter
sys.path.append('../../')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ailoganalyzer.dataset.log import log_dataset
from ailoganalyzer.dataset.sample import session_window, read_json, sliding_window
from ailoganalyzer.tools.utils import (save_parameters, seed_everything,
                                 train_val_split)


def generate(path, name, window_size):
    hdfs = {}
    length = 0
    with open(path + name, 'r') as f:
        for ln in f.readlines():
            # create list of event
            ln = list(map(lambda n: n -1, map(int, ln.strip().split())))
            #ln = ln + [-1] * (window_size + 1 - len(ln))

            # default_dict start at 0 (count ocurence of each line) <amelioration : default_dict>
            hdfs[tuple(ln)] = hdfs.get(tuple(ln), 0) + 1
            length += 1
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs, length


class Predicter():
    def __init__(self, model, options):
        self.data_dir = options['data_dir']
        self.device = options['device']
        self.model = model
        self.model_path = options['model_path']
        self.window_size = options['window_size']
        self.num_candidates = options['num_candidates']
        self.num_classes = options['num_classes']
        self.input_size = options['input_size']
        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.batch_size = options['batch_size']
        if self.semantics: self.event2semantic_vec = read_json(self.data_dir + '../preprocess/event2semantic_vec.json')

    def predict(self, line):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        for i in range(len(line) - self.window_size):
            seq0 = line[i:i + self.window_size]
            if self.semantics:
                Semantic_pattern = []
                for event in seq0:
                    if event == -1:
                        Semantic_pattern.append(tuple([-1] * 300))
                    else:
                        Semantic_pattern.append(tuple(self.event2semantic_vec[str(event)]))
            seq0 = Semantic_pattern
            label = line[i + self.window_size]
            seq1 = [0] * self.num_classes
            log_conuter = Counter(seq0)
            #for key in log_conuter:
            #    seq1[key] = log_conuter[key]

            seq0 = torch.tensor(seq0, dtype=torch.float).view(
                -1, self.window_size, self.input_size).to(self.device)
            #seq1 = torch.tensor(seq1, dtype=torch.float).view(
            #    -1, self.num_classes, self.input_size).to(self.device)
            label = torch.tensor(label).view(-1).to(self.device)
            output = model(features=[seq0, []], device=self.device)
            predicted = torch.argsort(output,
                                      1)[0][-self.num_candidates:]
            if label not in predicted:
                return i
        return -1

    def predict_unsupervised(self):

        print('model_path: {}'.format(self.model_path))
        test_normal_loader, test_normal_length = generate(self.data_dir, 'test_normal', self.window_size)
        test_abnormal_loader, test_abnormal_length = generate(self.data_dir,
            'test_abnormal', self.window_size)
        TP = 0
        FP = 0


        # Test the model
        start_time = time.time()

        with torch.no_grad():
            for line in tqdm(test_normal_loader.keys()):
                #print(line)


                i=self.predict(line)
                if i!=-1:
                    #print(test_normal_loader[line])
                    FP += test_normal_loader[line]
        with torch.no_grad():
            for line in tqdm(test_abnormal_loader.keys()):
                i=self.predict(line)
                if i!=-1:
                    TP += test_abnormal_loader[line]

        # Compute precision, recall and F1-measure
        FN = test_abnormal_length - TP
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        if P+R != 0:
            F1 = 2 * P * R / (P + R)
        else: F1 =0
        print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(FP, FN, P, R, F1))
        print('Finished Predicting')
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))
