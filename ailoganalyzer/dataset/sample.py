import json
import numpy as np
from tqdm import tqdm
from itertools import islice
from torch.utils.data import Dataset
from numpy import array, stack, vectorize, amax

import torch


class sliddingWindowDataset(Dataset):
    def __init__(self, log_seq, labels, event2vec, windows_size, seq=True, quan=False, sem=False):
        self.seq = seq
        self.quan = quan
        self.sem = sem
        self.Sequentials = log_seq
        self.event2vec = event2vec
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        log = dict()
        seq = self.Sequentials[idx]
        if self.seq:
            log['Sequentials'] = torch.tensor(seq,
                                              dtype=torch.float)
        # TODO : quantitative
        if self.quan:
            log['Quantitatives'] = torch.tensor(self.Quantitatives[idx],
                                                dtype=torch.float)
        if self.sem:
            sem = array([self.event2vec[e] for e in seq])
            log['Semantics'] = torch.tensor(sem,
                                            dtype=torch.float)
        return log, self.labels[idx]


def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict


def trp(ls, n):
    """ Truncate or pad a list """
    r = ls[:n]
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


def window_seq(seqs, n):
    for seq in seqs:
        if len(seq) > n:
            seq = seq[:-1]
        seq = map(int, seq)
        it = iter(seq)
        result = list(islice(it, n))
        if len(result) == n:
            yield array(result)
        else:
            yield result + list([0]) * (n - len(result))
        for elem in it:
            result = result[1:] + [elem]
            yield array(result)


#class slidding_window_dataset(Dataset)


def sliding_window(log_loader, sequences, window_size, sample_ratio=1, semantic=True, system=""):
    '''
    dataset structure
        result_logs(dict):
            result_logs['feature0'] = list()
            result_logs['feature1'] = list()
            ...
        labels(list)
    '''
    print("slidding windows")
    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []
    print("get template to vec")
    print("nb seq", len(sequences))

    # Remove empty sequences
    sequences = [seq for seq in sequences if len(seq) > 0]
    print("nb seq", len(sequences))

    event2vec = log_loader.template_to_vec_all(system)
    print("got template to vec")
    seq_it = window_seq(sequences, window_size)

    labels = []
    for seq in tqdm(sequences):
        if len(seq) > window_size:
            labels.extend(seq[window_size:])
        else:
            labels.append(0)
    print("start ee")
    result_logs["Sequentials"] = stack(seq_it)
    print(result_logs["Sequentials"].shape)
    print(len(labels))
    print("max ::", amax(result_logs["Sequentials"]))
    print("max lab :: ", amax(labels))
    dataset = sliddingWindowDataset(result_logs["Sequentials"], labels, event2vec, window_size, seq=False, quan=False, sem=True)
    #vfunc = vectorize(event2vec.__getitem__)
    #result_logs["Semantics"] = vfunc(result_logs["Sequentials"])
    print("end ee")

    return dataset


def session_window(data_dir, num_classes, datatype, sample_ratio=1, semantic = False):
    raise NotImplementedError
