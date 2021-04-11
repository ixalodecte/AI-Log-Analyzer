# pipeline
# 1. read origin logs
# 2. extract label, time and origin event
# 3. match event to the template
import re
import pandas as pd
from tqdm import tqdm
import fnmatch
from collections import defaultdict
import numpy as np

para = {
    "data":"../data/bgl2_100k",
    "template":"../data/preprocess/templates.csv",
    "structured_file":"../data/preprocess/structured.csv"}
# read origin logs

def information_extractor(line, log_structure):
    info = {
        "message": float("NaN"),
        "time": float("NaN"),
        "label": float("NaN")
    }
    separator = log_structure["separator"]
    if separator:
        line = line.split(separator)
        info["message"] = " ".join(line[log_structure["message_start_index"] : log_structure["message_end_index"]])
        #print(line[log_structure["time_index"]])
        info["time"] = line[log_structure["time_index"]]

        # Si les lignes sont non labelisé : elles sont toutes normale
        if log_structure["label_index"] != None:
            info["label"] = line[log_structure["label_index"]]
        else:
            info["label"] = "-"
    else:
        info["message"] = line
    return info


def data_read(filepath, log_structure):
    fp = open(filepath, "r")
    datas = []
    lines = fp.readlines()

    data = defaultdict(list)

    for line in tqdm(lines, ascii = True, desc = "lecture des logs"):
        log = line.strip("\n")
        infos = information_extractor(log, log_structure)

        for info in infos.keys():
            data[info].append(infos[info])

    data = pd.DataFrame(data = data)
    print(data["time"][0])
    print("parsing dates ...")
    data["time"] = pd.to_datetime(data["time"] ,format = log_structure["time_format"], errors = "coerce")
    print("nombre de date invalide : ", data["time"].isnull().sum(), "sur", len(data["time"]))
    print("")
    data[pd.notnull(data["time"])]
    fp.close()
    return data

def prepare_template(template):
    #print(template)
    if not pd.isna(template) :
        return re.sub('<.*>', '*', template)
    else:
        return ""

def match(line, templates, event2id):
    for i,item in enumerate(templates):
        if fnmatch.fnmatch(line,item): # ajouter longueur
            return event2id[item]
    print("error matching :" , line)
    print(templates)
    return 'error'

def structure(data, template):
    # match event to the template
    template = pd.read_csv(template)

    event = []
    event2id = {}

    # Extract informations from template files
    for i in range(template.shape[0]):
        event_id = template.iloc[i, template.columns.get_loc("EventId")]
        event_template = prepare_template(template.iloc[i, template.columns.get_loc("EventTemplate")])
        event2id[event_template] = event_id
        event.append(event_template)

    error_log = []
    eventmap = []
    print("Matching...")
    for log in tqdm(data["message"]):
        eventmap.append(match(log,event, event2id))
    return eventmap

def save_structured(data, eventmap, filename):
    # extract label, time and origin event
    label = []
    time = []

    data_structured = pd.DataFrame(columns=["label","time","event_id"])
    data_structured["label"] = data["label"]
    data_structured["time"] = data["time"]
    data_structured["event_id"] = eventmap

    # Remove logs which do not match the template(very few logs ......)
    data_structured = data_structured[(-data_structured["event_id"].isin(["error"]))]

    data_structured = data_structured.sort_values(by="time")


    data_structured.to_csv(filename ,date_format = "%Y-%m-%d %H:%M:%S.%f", index=None)

def structure_log(in_log_file, log_structure, out_structured_file, template_file):
    data = data_read(in_log_file, log_structure["separator"])
    eventmap = match(data, template_file)
    structure(data,eventmap, out_structured_file)


if __name__ == "__main__":
    log_structure = {
        "separator" : None,          # separateur entre les champs d'une ligne
        "time_index" : 4,           # index timestamp
        "message_start_index" : 9,  # debut message
        "message_end_index" : None, # fin message
        "label_index" : 0           # index label (None si aucun)
    }
    data = data_read(para["data"], log_structure)

    eventmap = structure(data, log_structure, para["template"])
    save(data, log_structure, eventmap, para["structured_file"])
