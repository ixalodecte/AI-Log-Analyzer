#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
sys.path.append('../')
import os.path

from ailoganalyzer.models.lstm import deeplog, loganomaly, robustlog
from ailoganalyzer.tools.predict import Predicter
from ailoganalyzer.tools.train import Trainer
from ailoganalyzer.tools.utils import *
from ailoganalyzer.extract_template import log2template
from ailoganalyzer.structure import *
from ailoganalyzer.sample import *



# Config Parameters

options = dict()
options['data_dir'] = '/home/kangourou/gestionDeProjet/AILogAnalyzer/data/train/'
options['window_size'] = 10
options['device'] = "cuda"

# Smaple
options['sample'] = "sliding_window"
options['window_size'] = 10  # if fix_window

# Features
options['sequentials'] = True
options['quantitatives'] = False
options['semantics'] = False
options['feature_num'] = sum(
    [options['sequentials'], options['quantitatives'], options['semantics']])

# Model
options['input_size'] = 1
options['hidden_size'] = 64
options['num_layers'] = 2
options['num_classes'] = 1826

# Train
options['batch_size'] = 2048
options['accumulation_step'] = 1

options['optimizer'] = 'adam'
options['lr'] = 0.001
options['max_epoch'] = 370
options['lr_step'] = (300, 350)
options['lr_decay_ratio'] = 0.1

options['resume_path'] = None
options['model_name'] = "deeplog"
options['save_dir'] = "../result/deeplog/"

# Predict
options['model_path'] = "../result/deeplog/deeplog_last.pth"
options['num_candidates'] = 9

seed_everything(seed=1234)


def preprocess():
    # creation des fichier de sequences:
    para = {
        "log_file" : "../data/bgl2_1M",
        "template_file" : "../data/preprocess/templates.csv",
        "structured_file" : "../data/preprocess/structured.csv",
        "window_size" : 0.005,
        "step_size" : 0.001
    }

    log_structure = {
        "separator" : ' ',          # separateur entre les champs d'une ligne
        "time_index" : 4,           # index timestamp
        "message_start_index" : 9,  # debut message
        "message_end_index" : None, # fin message
        "label_index" : 0           # index label (None si aucun)
    }

    # 1. Extraction des templates
    if not os.path.isfile(para["template_file"]):
        num = log2template(para["log_file"], log_structure, para["template_file"])
        options['num_classes'] = num

    # 2. Matching des logs avec les templates.
    log_list = data_read(para["log_file"], log_structure)
    eventmap = match(log_list, log_structure, para["template_file"])
    structure(log_list, log_structure, eventmap, para["structured_file"])

    # 3. Sampling : création des séquences
    log_structured = load_structured_file(para["structured_file"])
    sampling(log_structured,para["window_size"],para["step_size"])


def train():
    Model = deeplog(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'])
    trainer = Trainer(Model, options)
    trainer.start_train()


def predict():
    Model = deeplog(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'])
    predicter = Predicter(Model, options)
    predicter.predict_unsupervised()


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['preprocess','train', 'predict'])
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    elif args.mode == 'preprocess':
        preprocess()
    else:
        predict()
