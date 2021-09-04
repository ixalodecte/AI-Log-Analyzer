import sys
sys.path.append('../')

from ailoganalyzer.anomalyDetection.LSTM import LogAnomaly
import pandas as pd
import numpy as np
from tqdm import tqdm
from ailoganalyzer.database.sqlite3_persistence import Sqlite3Persistence

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
dates = []

with open(bgl_file, "r") as f:
    for i, line in tqdm(enumerate(f), total=nb_line):
        if i == nb_line:
            break
        msg, date, label = preprocess_bgl(line)
        date.append(date)
        messages.append(msg)

sep = int(0.8 * nb_line)
test_seq = messages[sep:]
test_date = dates[sep:]

# ----------Predict-------------

with Sqlite3Persistence("AI_Log_Analyzer") as db:
    for i, msg in tqdm(enumerate(test_seq)):
        res = lstm.predict(msg)
        msg.save_log(msg, "test_BGL_LogAnomaly", dates[i], res)

with Sqlite3Persistence("AI_Log_Analyzer") as db:
    for log in db.get_logs("test_BGL_LogAnomaly", abnormal=True):
        print(log)
