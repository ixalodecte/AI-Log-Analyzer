import json
import numpy as np
from tqdm import tqdm

def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict


def trp(l, n):
    """ Truncate or pad a list """
    r = l[:n]
    if len(r) < n:
        r.extend(list([0]) * (n - len(r)))
    return r


def down_sample(logs, labels, sample_ratio):
    print('sampling...')
    total_num = len(labels)
    all_index = list(range(total_num))
    sample_logs = {}
    for key in logs.keys():
        sample_logs[key] = []
    sample_labels = []
    sample_num = int(total_num * sample_ratio)
    print(total_num, sample_num)

    for i in tqdm(range(sample_num)):
        random_index = int(np.random.uniform(0, len(all_index)))
        for key in logs.keys():
            sample_logs[key].append(logs[key][random_index])
        sample_labels.append(labels[random_index])
        del all_index[random_index]
    return sample_logs, sample_labels


def sliding_window(log_loader, sequences, num_classes, window_size, sample_ratio=1, semantic=True, system=""):
    '''
    dataset structure
        result_logs(dict):
            result_logs['feature0'] = list()
            result_logs['feature1'] = list()
            ...
        labels(list)
    '''

    num_sessions = 0
    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []
    labels = []

    for line in tqdm(sequences):
        num_sessions += 1
        line = [int(e) - 1 for e in line]
        #sys.exit(0)
        for i in range(len(line) - window_size):
            Sequential_pattern = line[i:i + window_size]

            #Quantitative_pattern = [0] * num_classes
            #log_counter = Counter(Sequential_pattern)

            #for key in log_counter:
                #Quantitative_pattern[key] = log_counter[key]
            if semantic:
                Semantic_pattern = []
                for event in Sequential_pattern:
                    if event == -1:
                        Semantic_pattern.append(np.array([-1] * 300))
                    else:
                        #start_time = time.perf_counter()
                        Semantic_pattern.append(log_loader.template_to_vec(system,event))
                        #end_time = time.perf_counter()
                        #print("ex time:", end_time-start_time)

            Sequential_pattern = np.array(Sequential_pattern)[:,
                                                  np.newaxis]
            Quantitative_pattern = []
            Quantitative_pattern = np.array(
                Quantitative_pattern)[:, np.newaxis]
            result_logs['Sequentials'].append(Sequential_pattern)
            result_logs['Quantitatives'].append(Quantitative_pattern)
            if semantic:
                result_logs['Semantics'].append(Semantic_pattern)

            labels.append(line[i + window_size])
    #print(labels)
    if sample_ratio != 1:
        result_logs, labels = down_sample(result_logs, labels, sample_ratio)

    print('System {}, number of sessions {}'.format(system, num_sessions))
    print('System {}, number of seqs {}'.format(system,
                                              len(result_logs['Sequentials'])))

    return result_logs, labels


def session_window(data_dir,num_classes, datatype, sample_ratio=1, semantic = False):
    raise NotImplementedError
