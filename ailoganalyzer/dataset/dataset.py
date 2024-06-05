from collections import defaultdict
from torch.utils.data import Dataset
import torch
import numpy as np

from ailoganalyzer.dataset.template_miner import TemplateMinerr, Vectorizer
    
class LogDataset(Dataset):
    def __init__(self, semantic_vector=None, window_size=10, seq_label=True, template_miner_persistence=None):
        
        self.window_size = window_size
        self.template_miner = TemplateMinerr(template_miner_persistence)
        
        self.semantic=False
        self.semantic_vector=None
        
        # Return semantic feature if semantic feature is set
        if semantic_vector is not None:
            self.semantic_vector = Vectorizer(semantic_vector)
            self.semantic = True
        
        # Always return sequential and quantitative features
        self.sequential = True
        self.quantitative = True
        
        self.ls_seq = []
        self.ls_sem = []
        self.ls_quant = []
        
        # template_counter count the number of logs corresponding to a template
        # Each times a log is added to the dataset, increment the template_counter attribute
        self.template_counter = defaultdict(int)
        self.seq_label = seq_label
        
        self.id_max = None
    
    def count_template(self):
        return self.template_counter
        # if self.ls_seq_change_counter or True:
        #     self.mem_counter = Counter(self.ls_seq)
        #     {t:self.ls_seq.count(i) for i,t in enumerate()}
        #     self.ls_seq_change_counter = False
        # return self.mem_counter
    
    def train_log(self, line):
        
        idx = self.template_miner.add_log(line)
        self.add_seq_id(idx)
        self.ls_seq_change_counter=True
        
    def set_num_classes(self, num_classes):
        # For inference we dont want to have idx that exceed the number of classes
        # (will raise an exception in the pytorch side)
        self.id_max = num_classes - 1
    
    def add_seq_id(self, idx):
        self.template_counter[self.template_miner.get_template_by_id(idx)] += 1
        self.ls_seq.append(idx)
    
    def get_num_classes(self):
        return len(set(self.ls_seq))
    
    def template_id_to_vec(self, idx):
        if self.semantic_vector is None:
            raise RuntimeError("This dataset can't transform a template to a vector. Please add a semantic vector.")
        vec = self.semantic_vector.template_to_vec(self.template_miner.get_template_by_id(idx),self.count_template())
        return vec
    
    def __len__(self):
        # if we generate label, minus 1
        return len(self.ls_seq) - self.window_size - int(self.seq_label)

    def __getitem__(self, idx):
        features = {}
        if self.sequential or self.semantic or self.quantitative:
            X_seq = self.ls_seq[idx:idx + self.window_size]
            Y_seq = self.ls_seq[idx+ self.window_size + 1]
            
            # Try to find the closest template id for each id that exceed id_max
            if self.id_max is not None:
                seq = np.append(X_seq, Y_seq)
                if max(seq) > self.id_max:
                    if self.semantic_vector is not None:
                        # TODO test this block of code
                        candidate_vectors = np.array([self.template_id_to_vec(i) for i in range(self.id_max)])
                        
                        modif_seq = []
                        for e in seq:
                            if e > self.id_max:
                                template_vector = self.template_id_to_vec(e)
                                distances = np.linalg.norm(candidate_vectors - template_vector, axis=1)
                                e = np.argmin(distances)
                            modif_seq.append(e)
                        
                        X_seq = np.array(e[:-1])
                        Y_seq = e[-1]
                    else:
                        raise RuntimeError()
        
        if self.sequential:
            features["sequential"] = torch.tensor(X_seq, dtype=torch.float).unsqueeze(-1)
            
        if self.semantic:
            # Transform a sequence of event id into a sequence of semantic vectors
            
            #set_keys = sorted(set(X_seq + [Y_seq]))
            #keys_vect = {i:self.template_id_to_vec(i) for i in set_keys}
                        
            X_sem = np.array([self.template_id_to_vec(i) for i in X_seq])
            features["semantic"] = torch.tensor(X_sem, dtype=torch.float)
        
        if self.quantitative:
            # Counter for X_seq
            
            u, c = np.unique(X_seq, return_counts=True)
            d = dict(zip(u, c))
            quant = [d.get(e, 0) for e in range(self.get_num_classes())]
            features["quantitative"] = torch.tensor(quant, dtype=torch.float).unsqueeze(-1)
            
        if self.seq_label:
            return features, Y_seq
        else:
            return features
    

class LogFileDataset(LogDataset):
    def __init__(self, log_file, **kwargs):
        super().__init__(**kwargs)
        with open(log_file, "r") as f:
            for line in f:
                self.train_log(line)


