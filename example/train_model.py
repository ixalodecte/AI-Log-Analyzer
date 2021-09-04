import sys
sys.path.append('../')

from ailoganalyzer.anomalyDetection.LSTM import LogAnomaly
import pandas as pd
import numpy as np
from tqdm import tqdm


def preprocess_bgl(line):
    line = line.split()
    label = int(not line[0] == "-")
    date_str = line[4]
    date = pd.to_datetime(date_str, format="%Y-%m-%d-%H.%M.%S.%f")
    message = " ".join(line[9:])
    return message, date, label


bgl_file = "csv/bgl2_100k"

lstm = LogAnomaly("BGL_test_loganomaly")
nb_line = 100000
labels = []
messages = []

with open(bgl_file, "r") as f:
    for i, line in tqdm(enumerate(f), total=nb_line):
        if i == nb_line:
            break
        msg, date, label = preprocess_bgl(line)
        labels.append(label)
        messages.append(msg)

labels = np.array(labels)
sep = int(0.8 * nb_line)
train_seq = messages[:sep]
train_labels = labels[:sep]

for i, msg in tqdm(enumerate(train_seq)):
    if train_labels[i] == 0:
        lstm.add_train_log(msg)
lstm.train()
