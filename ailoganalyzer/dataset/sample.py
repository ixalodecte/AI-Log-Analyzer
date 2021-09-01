from collections import Counter
from torch.utils.data import Dataset
from numpy import array, zeros, newaxis
import torch


class sliddingWindowDataset(Dataset):
    def __init__(self, log_seq, labels, windows_size, event2vec={}, num_classes = 0, seq=True, quan=False, sem=False):
        self.event2vec = event2vec
        self.num_classes = num_classes

        self.seq = seq
        self.quan = quan
        self.sem = sem

        self.labels = labels
        self.Sequentials = log_seq

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        log = dict()
        seq = self.Sequentials[idx]
        if self.seq:
            log['Sequentials'] = torch.tensor(seq[:, newaxis],
                                              dtype=torch.float)
        if self.quan:
            quan = zeros(self.num_classes)
            counter = Counter(seq)
            for i, e in enumerate(counter):
                if i < self.num_classes:
                    quan[i] = counter[e]
            quan = quan[:, newaxis]
            log['Quantitatives'] = torch.tensor(quan,
                                                dtype=torch.float)
        if self.sem:
            sem = array([self.event2vec[e] for e in seq])
            log['Semantics'] = torch.tensor(sem,
                                            dtype=torch.float)
        return log, self.labels[idx]
