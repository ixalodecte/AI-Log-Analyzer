#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
sys.path.append('../')
import os.path
from pandas import read_csv
import pickle
from os import makedirs
from ailoganalyzer.models.lstm import deeplog, loganomaly, robustlog
from ailoganalyzer.models.time_series import *
from ailoganalyzer.tools.predict import Predicter
from ailoganalyzer.tools.train import Trainer
from ailoganalyzer.tools.utils import *
from ailoganalyzer.extract_template import log2template
from ailoganalyzer.structure import *
from ailoganalyzer.sample import *
from ailoganalyzer.tools.visualisation import *
from ailoganalyzer.gen_train_data import *
from adtk.detector import SeasonalAD
from ailoganalyzer.dataset.line_to_vec import *
from api.database import LogLoader



# Config Parameters

options = dict()
options['data_dir'] = '/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/train/'
options['window_size'] = 20
options['device'] = "cuda"

# Smaple
#options['sample'] = "sliding_window"
#options['window_size'] = 10  # if fix_window
options['sample'] = "sliding_window"
options['window_size'] = 20

# Features
options['sequentials'] = False
options['quantitatives'] = False
options['semantics'] = True
options['feature_num'] = sum(
    [options['sequentials'], options['quantitatives'], options['semantics']])

# Model
options['input_size'] = 300
options['hidden_size'] = 128
options['num_layers'] = 2
options['num_classes'] = 1826

# Train
options['batch_size'] = 256
options['accumulation_step'] = 1

options['optimizer'] = 'adam'
options['lr'] = 0.01
options['max_epoch'] = 6
options['lr_step'] = (4, 5)
options['lr_decay_ratio'] = 0.1

options['resume_path'] = None
options['model_name'] = "deeplog"
options['save_dir'] = "../result/ailoganalyzer/"


# Predict
options['model_path'] = str(options['save_dir'] + "deeplog_last.pth")
options['model_path_TS'] = str(options['save_dir'] + "time_series.pth")
options['num_candidates'] = 2
options["system"] = ""

para = {
    "log_file" : "../data/bgl2_100k",
    "template_file" : "../data/preprocess/templates.csv",
    "structured_file" : "../data/preprocess/structured.csv",
    "window_size" : 0.1,
    "step_size" : 0.01
}
seed_everything(seed=1234)

def count_num_line(filename):
    with open(filename) as f:
        line_count = 0
        for line in f:
            line_count += 1
        return line_count

def preprocess():



    log_structure = {
        "separator" : ' ',          # separateur entre les champs d'une ligne
        "time_index" : 4,           # index timestamp
        "time_format" : "%Y-%m-%d-%H.%M.%S.%f",
        "message_start_index" : 0,  # debut message
        "message_end_index" : None, # fin message (None si on va jusqu'a la fin de ligne)
        "label_index" : 0           # index label (None si aucun)
    }

    # 1. Extraction des templates
    if not os.path.isfile(para["template_file"]):
        with open(para["log_file"]) as f:
            num = log2template(f, para["template_file"], log_structure)
            options['num_classes'] = num

    # 2. Matching des logs avec les templates.
    log_list = data_read(para["log_file"], log_structure)
    eventmap = structure(log_list, para["template_file"])
    save_structured(log_list, eventmap, para["structured_file"])

    print("\ncreate sequence of event...")
    # 3. Sampling : création des séquences
    log_structured = load_structured_file(para["structured_file"])
    sampling(log_structured,para["window_size"],para["step_size"])

def vectorize():
    trainer_vec(word_vec)

def train():
    #options["num_classes"] = count_num_line("../data/preprocess/templates.csv")
    print(options["num_classes"])
    Model = deeplog(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'])
    trainer = Trainer(Model, options)
    trainer.start_train()





def predict():
    options["num_classes"] = count_num_line("../data/preprocess/templates.csv")
    Model = deeplog(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'])
    predicter = Predicter(Model, options)
    predicter.predict_unsupervised()

def visualisation():
    time_serie = seq_to_time_serie("/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/preprocess/sequence.csv")
    visualize_time_serie(time_serie)

def split():
    gen_train_test(0.3)

def train_TS():
    training_set = pd.read_csv("/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/xaa")
    training_set = training_set.iloc[:,1:2].values
    print(training_set)
    seq, sc = preprocess_TS(training_set,288)
    with open("/home/kangourou/gestionDeProjet/AI-Log-Analyzer/result/deeplog/sc" + options["system"], 'wb') as f1:
        pickle.dump(sc, f1)
    #model = timeSerie(365,22,100).to("cuda")
    train_TS_LSTM(options["model_path_TS"], seq, options)
    val_set = pd.read_csv("/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/val")
    val_set = val_set.iloc[:,1:2].values
    seq_val,_ = preprocess_TS(val_set, 288, sc)
    intervalle = compute_normal_interval_TS(options["model_path_TS"], seq_val)

    #t = TimeSerie()
    #print(s_train)
    #t.fit(s_train)
    #t.visualize("train")
    #print(options['model_path_TS'])
    #save_time_serie(t, options['model_path_TS'])

def test_TS():

    training_set = pd.read_csv("/home/kangourou/gestionDeProjet/AI-Log-Analyzer/data/testt")
    training_set = training_set.iloc[:,1:2].values

    print(training_set)
    with open("/home/kangourou/gestionDeProjet/AI-Log-Analyzer/result/deeplog/sc", 'rb') as f1:
        sc = pickle.load(f1)
    seq_train,_ = preprocess_TS(training_set, 288, sc)

    test_TS_LSTM(options["model_path_TS"], seq_train,sc, intervalle)

    #time_serie.anomaly(s_test)
    #time_serie.visualize("test")

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['preprocess','train', 'predict', 'visualisation', 'split', 'trainTS', 'testTS', 'vectorize'])
    args = parser.parse_args()
    #options["num_classes"] = count_num_line("../data/preprocess/templates.csv")
    if args.mode == 'train':
        train_db("192.168.1.1")
    elif args.mode == 'preprocess':
        preprocess()
    elif args.mode == 'visualisation':
        visualisation()
    elif args.mode == 'split':
        split()
    elif args.mode == 'trainTS':
        train_TS()
    elif args.mode == 'testTS':
        test_TS()
    elif args.mode == 'vectorize':
        vectorize()
    else:
        predict()
