# pipeline
# 1. read origin logs
# 2. extract label, time and origin event
# 3. match event to the template
import re
import pandas as pd
from tqdm import tqdm
import fnmatch


para = {
    "bgl":"../data/bgl2_100k",
    "template":"../data/preprocess/templates.csv",
    "structured_file":"../data/preprocess/structured.csv"}
# read origin logs
def data_read(filepath, log_structure):
    fp = open(filepath, "r")
    datas = []
    lines = fp.readlines()
    i = 0
    for line in tqdm(lines):
        # <amelioration> : parser ici la ligne de log?
        row = line.strip("\n")
        datas.append(row)
        i = i + 1
    fp.close()
    return datas

def prepare_template(template):
    #print(template)
    if not pd.isna(template) :
        return re.sub('<.*>', '*', template)
    else:
        return ""

def match(BGL, log_structure, template):
    # match event to the template
    template = pd.read_csv(template)

    event = []
    event2id = {}

    for i in range(template.shape[0]):
        event_id = template.iloc[i, template.columns.get_loc("EventId")]
        event_template = prepare_template(template.iloc[i, template.columns.get_loc("EventTemplate")])
        event2id[event_template] = event_id
        event.append(event_template)
    error_log = []
    eventmap = []
    print("Matching...")

    for log in tqdm(BGL):
        log_event = " ".join(log.split(log_structure["separator"])[log_structure["message_start_index"]:log_structure["message_end_index"]])
        for i,item in enumerate(event):
            if fnmatch.fnmatch(log_event,item): # ajouter longueur
                eventmap.append(event2id[item])
                break
            if i == len(event)-1:
                eventmap.append('error')
                error_log.append(log_event)
                print(log_event)
    return eventmap
def structure(BGL, log_structure, eventmap, structured_file):
    # extract label, time and origin event
    label = []
    time = []
    for line in tqdm(BGL):
        log = line.split(log_structure["separator"])
        label.append(log[log_structure["label_index"]])
        time.append(log[log_structure["time_index"]])

    BGL_structured = pd.DataFrame(columns=["label","time","event_id"])
    BGL_structured["label"] = label
    BGL_structured["time"] = time
    BGL_structured["event_id"] = eventmap
    # Remove logs which do not match the template(very few logs ......)
    BGL_structured = BGL_structured[(-BGL_structured["event_id"].isin(["error"]))]
    BGL_structured.to_csv(structured_file,index=None)

def structure_log(in_log_file, log_structure, out_structured_file, template_file):
    BGL = data_read(in_log_file, log_structure["separator"])
    eventmap = match(BGL, template_file)
    structure(BGL,eventmap, out_structured_file)


if __name__ == "__main__":
    log_structure = {
        "separator" : ' ',          # separateur entre les champs d'une ligne
        "time_index" : 4,           # index timestamp
        "message_start_index" : 9,  # debut message
        "message_end_index" : None, # fin message
        "label_index" : 0           # index label (None si aucun)
    }
    BGL = data_read(para["bgl"], log_structure)

    eventmap = match(BGL, log_structure, para["template"])
    structure(BGL, log_structure, eventmap, para["structured_file"])
