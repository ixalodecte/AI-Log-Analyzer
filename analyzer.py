from datetime import datetime
from datetime import timedelta
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from ailoganalyzer.structure import match
from pandas import date_range
from api.database import LogLoader
import pickle
import pandas as pd
import time
from numpy import array
from torch.multiprocessing import Process, set_start_method
from multiprocessing import Lock
from sklearn.neighbors import NearestNeighbors
from ailoganalyzer.models.lstm import deeplog
from ailoganalyzer.models.time_series import train_TS_LSTM, preprocess_TS, compute_normal_interval_TS
from ailoganalyzer.tools.predict import Predicter
from ailoganalyzer.tools.train import Trainer
from ailoganalyzer.tools.utils import count_num_line
from ailoganalyzer.extract_template import log2template
from ailoganalyzer.structure import structure, save_structured, prepare_template
from ailoganalyzer.sample import load_structured_file, sampling
from ailoganalyzer.gen_train_data import gen_train_test
from ailoganalyzer.dataset.line_to_vec import read_json, import_word_vec,line_to_vec, dump_2_json, data_read_template, template_to_vec

options = dict()
options['data_dir'] = 'data/train/'
options['preprocess_dir'] = "data/preprocess/"
options["train_dir"] = "data/train/"
options['window_size'] = 20
options['device'] = "cpu"

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
options['save_dir'] = "result/ailoganalyzer/"


# Predict
options['model_path'] = str(options['save_dir'] + "deeplog_last.pth")
options['model_path_TS'] = str(options['save_dir'] + "time_series.pth")
options['num_candidates'] = 2
options["system"] = ""

para = {
    "window_size" : 0.1,
    "step_size" : 0.01
}

def nearest_template(template_vec, system):
    event2semantic_vec = read_json('data/preprocess/event2semantic_vec_'+system+'.json')
    knn = NearestNeighbors(n_neighbors=1, metric='cosine')
    knn.fit(array([array(e) for e in event2semantic_vec.values()]))
    _, id = knn.kneighbors(array([(array(template_vec))]))
    return id[0][0]



def train(options):
    #options["num_classes"] = count_num_line("../data/preprocess/templates.csv")
    print(options["num_classes"])
    Model = deeplog(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'])
    trainer = Trainer(Model, options)
    trainer.start_train()


def train_TS(options, system):
    training_set = pd.read_csv("data/train/train_TS")
    training_set = training_set.iloc[:,1:2].values
    print(training_set)
    seq, sc = preprocess_TS(training_set,288)
    with open(options['save_dir'] + options["system"] + "_sc", 'wb') as f1:
        pickle.dump(sc, f1)
    #model = timeSerie(365,22,100).to("cuda")
    train_TS_LSTM(options["save_dir"] + system + "_TS.pth", seq, options)
    val_set = pd.read_csv("data/train/normal_TS")
    val_set = val_set.iloc[:,1:2].values
    seq_val,_ = preprocess_TS(val_set, 288, sc)
    intervalle = compute_normal_interval_TS(options["save_dir"] + system + "_TS.pth", seq_val)
    with open("/home/kangourou/gestionDeProjet/AI-Log-Analyzer/result/ailoganalyzer/intervalle" + options["system"], 'wb') as f1:
        pickle.dump(intervalle, f1)


def train_db(l,system, options, word_vec, date_debut = None, date_fin = None):
    l.acquire()
    print("hello")
    template_file = "data/preprocess/templates_" + system + ".csv"
    structured_file = "data/preprocess/structured_" + system + ".csv"
    param = {"system" : system}
    if (date_debut != None) and (date_fin != None):
        param = {
            "start_time" : date_debut,
            "end_time" : date_fin
        }
    db = LogLoader("ailoganalyzer_db")
    date = []
    logs = []
    for log in db.find("logs",param):
        date.append(log["time"])
        logs.append(log["message"])

    log_date_message = pd.DataFrame(columns=["time","message", "label"])
    log_date_message["time"] = date
    log_date_message["message"] = logs
    log_date_message["label"] = ["-"]*len(logs)

    #log_date_message.to_csv("../data/preprocess/templates.csv", index = None)
    num = log2template(log_date_message["message"], template_file, options)
    options['num_classes'] = num
    eventmap = structure(log_date_message, template_file)
    save_structured(log_date_message, eventmap, structured_file)
    print("\ncreate sequence of event...")

    # 3. Sampling : création des séquences
    log_structured = load_structured_file(structured_file)
    sampling(log_structured,para["window_size"],para["step_size"])
    gen_train_test(1,options)

    # 4. Vectorisation des templates

    data = data_read_template(template_file)
    vec = line_to_vec(word_vec, data)
    dump_2_json(vec, "data/preprocess/event2semantic_vec_" +system +".json")
    options["system"] = system
    options["num_classes"] = count_num_line(template_file)
    options['model_path'] = str(options['save_dir'] + "deeplog_last" + system + ".pth")
    options['model_path_TS'] = str(options['save_dir'] + "time_series" + system + ".pth")
    train(options)

    # -- entrainement time Series
    #start, end = db.start_end_date(system)
    #dates = date_range(start=start, end=end, periods="4min")
    #train_dates = dates[:int(len(dates) * 0.7)]
    #val_dates = dates[:int(len(dates) * 0.3)]

    #time_serie_train = db.time_serie("logs", train_dates, {"system":system})
    #time_serie_val = db.time_serie("logs", val_dates, {"system":system})


    #TS_trainn = pd.DataFrame(columns=["time","number"])
    #TS_trainn["time"] = list(time_serie_train.keys())
    #TS_trainn["number"] = list(time_serie_train.values())
    #TS_trainn.to_csv("data/train/train_TS", index = None)

    #TS_val = pd.DataFrame(columns=["time","number"])
    #TS_val["time"] = list(time_serie_val.keys())
    #TS_val["number"] = list(time_serie_val.values())
    #TS_val.to_csv("data/train/normal_TS", index = None)

    #train_TS(options)
    db.set_trained(system)
    l.release()


def analyzer_semantic(system, options, word_vec):
    print("l'analyse commence")
    template_file = "data/preprocess/templates_" + system+".csv"
    templates=pd.read_csv(template_file)
    template_str = templates["EventTemplate"]
    template_str = [prepare_template(e) for e in template_str]
    event2id = {}
    for i,e in enumerate(template_str):
        event2id[e] = i


    options["num_classes"] = count_num_line(template_file)

    persistence = FilePersistence("data/preprocess/templates_persist_"+options["system"]+".bin")
    template_miner = TemplateMiner(persistence)


    window_size = 1800
    step_size = 30
    db = LogLoader("ailoganalyzer_db")

    last_date = datetime(1970, 1, 1)
    while(True):
        end_time = datetime.now()
        start_time = end_time - timedelta(seconds=window_size)
        filters = {
            "start_time": start_time,
            "end_time": end_time,
            "system": system
        }
        sequence_data = db.find("logs", filters)
        print("seq len", len(sequence_data))

        indice = -1

        #On recherche une sequence qu'on a pas deja analyser
        for i,elt in enumerate(sequence_data):
            print(elt["time"])
            if elt["time"] > last_date:
                print("ok")
                indice = i
                break
        if indice != -1:
            sequence_message = [e["message"] for e in sequence_data]

            for line in sequence_message[indice:]:
                result = template_miner.add_log_message(line)

                # -> si on trouve un nouveau template:
                if result["change_type"] != "none":
                    print("new template :",end=" ")
                    new_template = prepare_template(result["template_mined"])
                    print(new_template)
                    vec_new_t = template_to_vec(word_vec, new_template)
                    new_t = nearest_template(vec_new_t, options["system"])
                    event2id[new_template] = new_t
                    sorted_clusters = sorted(template_miner.drain.clusters, key=lambda it: it.size, reverse=True)
                    cluster_template_str = []
                    for e in sorted_clusters:
                        cluster_template_str.append(prepare_template(e.get_template()))

                    print( event2id)

                    for e in event2id:
                        if e not in cluster_template_str:
                            event2id.pop(e)
                            break
                    print("---------- del ancien ---------")
                    print( event2id)

            sequence_message = sequence_message[max((indice - options["window_size"]), 0):]
            print("get sequence de longueur :", len(sequence_message))
            sequence = []

            # sequence de log en sequence d'event
            for e in sequence_message:
                sequence.append(int(match(e, template_str, event2id)))
            print("seq:", sequence)



            # testing de la sequence:
            Model = deeplog(input_size=options['input_size'],
                            hidden_size=options['hidden_size'],
                            num_layers=options['num_layers'],
                            num_keys=options['num_classes'])
            predicter = Predicter(Model, options)
            indice_abnormal = predicter.predict(sequence)
            if indice_abnormal !=- 1:
                ab_line = sequence_data[indice_abnormal]
                print(ab_line)
                db.set_abnormal_log(ab_line)
        last_date = end_time
        time.sleep(step_size)

def analyzer_TS(system, options):
    now = datetime.now()
    dates = date_range(start =now - timedelta(1,0), end = now, periods = "4min")
    time_serie = db.time_serie("logs", dates, {"system":system})
    TS = []

try:
    set_start_method('spawn')
except RuntimeError:
    pass
if __name__ == '__main__':
    db = LogLoader("ailoganalyzer_db")
    word_vec = import_word_vec()
    #word_vec = []
    train_delta = timedelta(days = 1)
    my_lock = Lock()
    process_predict = []
    is_trained = []



    # a retirer
    db.set_trained("192.168.1.1")

    while(True):
        time.sleep(5)
        for system in db.get_systems():
            if not db.is_trained(system) and system not in is_trained:
                if (datetime.now() - db.start_end_date(system)[0]) > train_delta:
                    options["system"] = system
                    print("lets go")
                    train_db(my_lock,system,options, word_vec, date_debut=datetime.now() - timedelta(days = 2), date_fin=datetime.now())
                    is_trained.append(system)

            else:
                if system not in process_predict:
                    options["system"] = system
                    Process(target = analyzer_semantic, args = (system, options, word_vec)).start()
                    process_predict.append(system)
        print(process_predict)
