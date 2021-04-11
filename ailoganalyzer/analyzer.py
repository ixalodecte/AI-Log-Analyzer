from datetime import datetime
from datetime import timedelta
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from ailoganalyzer.structure import match
from pandas import date_range

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
    training_set = pd.read_csv("../data/train/train_TS")
    training_set = training_set.iloc[:,1:2].values
    print(training_set)
    seq, sc = preprocess_TS(training_set,288)
    with open(options['save_dir'] + options["system"] + "_sc", 'wb') as f1:
        pickle.dump(sc, f1)
    #model = timeSerie(365,22,100).to("cuda")
    train_TS_LSTM(options["save_dir"] + system + "_TS.pth", seq, options)
    val_set = pd.read_csv("../data/train/normal_TS")
    val_set = val_set.iloc[:,1:2].values
    seq_val,_ = preprocess_TS(val_set, 288, sc)
    intervalle = compute_normal_interval_TS(options["save_dir"] + system + "_TS.pth", seq_val)

def train_db(system, options, date_debut = None, date_fin = None):
    template_file = "../data/preprocess/templates_" + system + ".csv"
    structured_file = "../data/preprocess/structured_" + system + ".csv"
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
    log2template(logs, template_file)
    eventmap = structure(log_date_message, template_file)
    save_structured(log_date_message, eventmap, structured_file)
    print("\ncreate sequence of event...")

    # 3. Sampling : création des séquences
    log_structured = load_structured_file(structured_file)
    sampling(log_structured,para["window_size"],para["step_size"])
    gen_train_test(1)

    # 4. Vectorisation des templates
    word_vec = import_word_vec()
    data = data_read_template(template_file)
    vec = line_to_vec(word_vec, data)
    dump_2_json(vec, "../data/preprocess/event2semantic_vec_" +system +".json")
    options["system"] = system
    options["num_classes"] = count_num_line(template_file)
    options['model_path'] = str(options['save_dir'] + "deeplog_last" + system + ".pth")
    options['model_path_TS'] = str(options['save_dir'] + "time_series" + system + ".pth")
    train(options)

    # -- entrainement time Series
    start, end = db.start_end_date(system)
    dates = date_range(start =start, end = end, periods = "4min")
    train_dates = dates[:int(len(dates) * 0.7)]
    val_dates = dates[:int(len(dates) * 0.3)]

    time_serie_train = db.time_serie("logs", train_dates, {"system":system})
    time_serie_val = db.time_serie("logs", val_dates, {"system":system})


    TS_trainn = pd.DataFrame(columns=["time","number"])
    TS_trainn["time"] = list(time_serie_train.keys())
    TS_trainn["number"] = list(time_serie_train.values())
    TS_trainn.to_csv("../data/train/train_TS", index = None)

    TS_val = pd.DataFrame(columns=["time","number"])
    TS_val["time"] = list(time_serie_val.keys())
    TS_val["number"] = list(time_serie_val.values())
    TS_val.to_csv("../data/train/normal_TS", index = None)

    train_TS(options)
    db.set_trained(system)

def analyzer_semantic(system, options):
    template_file = "../data/preprocess/templates_" + system+".csv"
    options["num_classes"] = count_num_line(template_file)

    persistence = FilePersistence("../data/preprocess/templates_persist.bin")
    template_miner = TemplateMiner(persistence)

    sorted_clusters = sorted(template_miner.drain.clusters, key=lambda it: it.size, reverse=True)

    for cluster in sorted_clusters:
        print(str(cluster.cluster_id) + "," + '"' + cluster.get_template() + '"')
        cluster_id.append(cluster.cluster_id)
        template_str.append(cluster.get_template())
    event2id = dict(zip(template_str, cluster_id))


    window_size = 1800
    step_size = 30
    db = LogLoader("ailoganalyzer_db")

    last_date = datetime.date(1970, 1, 1)
    while(True):
        end_time = datetime.now()
        start_time = end_time - timedelta(0,window_size)
        filters = {
            "start_time": start_time,
            "end_time":end_time,
            "system":system
        }
        sequence_message = db.find("logs", filters)

        indice = -1
        for i,elt in enumerate(sequence_message):
            if elt["time"] >= last_date:
                indice = i
                break
        if indice != -1:
            for line in sequence_message[indice:]:
                result = template_miner.add_log_message(line)
                if result["change_type"] != "none":
                    sorted_clusters = sorted(template_miner.drain.clusters, key=lambda it: it.size, reverse=True)
                    for cluster in sorted_clusters:
                        cluster_id.append(cluster.cluster_id)
                        template_str.append(cluster.get_template())
                    event2id = dict(zip(template_str, cluster_id))

            sequence_message = sequence_message[max((indice - options["window_size"]), 0):]
            sequence = []
            for e in sequence_message:
                sequence.append(match(e, template_str, event2id) -1)



            # testing de la sequence:
            Model = deeplog(input_size=options['input_size'],
                            hidden_size=options['hidden_size'],
                            num_layers=options['num_layers'],
                            num_keys=options['num_classes'])
            predicter = Predicter(Model, options)
            indice_abnormal = predicter.predict(line)
            if indice_abnormal !=- 1:
                ab_line = sequence_message[indice_abnormal]
                db.set_abnormal_log(ab_line)
        time.sleep(step_size)

def analyzer_TS(system, options):
    now = datetime.now()
    dates = date_range(start =now - timedelta(1,0), end = now, periods = "4min")
    time_serie = db.time_serie("logs", dates, {"system":system})
    TS = []
