from torch.utils.data import Dataset
from numpy import array
import torch


class sliddingWindowDataset(Dataset):
    def __init__(self, log_seq, labels, windows_size, event2vec={}, seq=True, quan=False, sem=False):
        self.event2vec = event2vec

        self.seq = seq
        self.quan = quan
        self.sem = sem

        self.labels = labels
        self.Sequentials = log_seq
        print(self.labels.shape, self.Sequentials.shape)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        log = dict()
        seq = self.Sequentials[idx]
        if self.seq:
            log['Sequentials'] = torch.tensor(seq,
                                              dtype=torch.float)
        # TODO : quantitative (Counter)
        if self.quan:
            log['Quantitatives'] = torch.tensor(self.Quantitatives[idx],
                                                dtype=torch.float)
        if self.sem:
            sem = array([self.event2vec[e] for e in seq])
            log['Semantics'] = torch.tensor(sem,
                                            dtype=torch.float)
        return log, self.labels[idx]
