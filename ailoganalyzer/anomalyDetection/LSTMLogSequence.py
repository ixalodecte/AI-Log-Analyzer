from datetime import datetime
from datetime import timedelta
from sklearn.model_selection import train_test_split
import torch
from torch.multiprocessing import Process, set_start_method
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np

from ailoganalyzer.anomalyDetection.lstmModels.lstm import robustlog, loganomaly, deeplog
from ailoganalyzer.tools.train import Trainer
from ailoganalyzer.anomalyDetection.AnomalyDetector import AnomalyDetector
from ailoganalyzer.dataset.sample import sliddingWindowDataset

try:
    set_start_method('spawn')
except RuntimeError:
    pass


class LSTMLogSequence(AnomalyDetector):
    def __init__(self, num_candidates=9, model="robustLog", device="auto"):
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super().__init__(model)
        self.semantic = False
        self.sequentials = False
        self.quantitatives = False
        self.window_size = 20
        self.num_candidates = num_candidates

        self.train_delta = timedelta(days=0)

        self.device = device
        self.model = None

    def initialize_model(self):
        self.model_path = "data/" + self.model_name + "_" + self.system + "_" + "last.pth"
        state = None
        if os.path.isfile(self.model_path):
            state = torch.load(self.model_path)
            num_classes = state["num_keys"]
        else:
            num_classes = self.log_loader.get_number_classes(self.system)
        self.num_classes = num_classes

        #num_classes = 16

        if self.model_name == "robustLog":
            self.model = robustlog(input_size=300,
                                   hidden_size=128,
                                   num_layers=2,
                                   num_keys=num_classes)
            self.input_size = 300
            self.semantic = True
        elif self.model_name == "loganomaly":
            self.model = loganomaly(input_size=300,
                                    hidden_size=128,
                                    num_layers=2,
                                    num_keys=num_classes)
            self.input_size = 300
            self.semantic = True
            self.quantitatives = True
        elif self.model_name == "deeplog":
            self.model = loganomaly(input_size=1,
                                    hidden_size=64,
                                    num_layers=2,
                                    num_keys=num_classes)
            self.input_size = 1
            self.sequentials = True
            self.quantitatives = True
        else:
            raise NotImplementedError

        if state is not None:
            self.model.load_state_dict(state["state_dict"])

    def is_trainable(self):
        if not Trainer.is_available:
            print("another model already training")
            return False
        start, end = self.log_loader.start_end_date(self.system)
        return datetime.now() - start > self.train_delta and not self.log_loader.is_trained(self.system, self.model_name)

    def train(self):
        trainer = Trainer(self.model,
                          self.train_loader,
                          self.valid_loader,
                          self.num_classes,
                          self.system,
                          self.model_name,
                          self.window_size,
                          max_epoch=60,
                          lr_step=(40, 50),
                          model_path=self.model_path,
                          device=self.device
                          )
        #Trainer.is_available = False
        p = Process(target=trainer.start_train)
        p.start()
        #Trainer.is_available = True
        return p

    def set_dataLoader_training(self):
        self.initialize_model()

        sequences, labels, _ = self.log_loader.get_last_sequence(self.system, self.model_name, self.window_size)

        train_seq, val_seq, train_label, val_label = train_test_split(sequences, labels, train_size=0.8)
        print("number train sequences :", train_seq.shape)
        print("number val sequences :", val_seq.shape)
        self.num_classes = self.log_loader.get_number_classes(self.system)
        event2vec = self.log_loader.template_to_vec_all(self.system)

        train_dataset = sliddingWindowDataset(train_seq,
                                              train_label,
                                              self.window_size,
                                              event2vec,
                                              num_classes=self.num_classes,
                                              seq=self.sequentials,
                                              quan=self.quantitatives,
                                              sem=self.semantic)
        valid_dataset = sliddingWindowDataset(val_seq,
                                              val_label,
                                              self.window_size,
                                              event2vec,
                                              num_classes=self.num_classes,
                                              seq=self.sequentials,
                                              quan=self.quantitatives,
                                              sem=self.semantic)

        print("num_classes :: ", self.num_classes)

        print("end slidding")

        self.train_loader = DataLoader(train_dataset,
                                       batch_size=2048,
                                       shuffle=True,
                                       pin_memory=True)
        self.valid_loader = DataLoader(valid_dataset,
                                       batch_size=2048,
                                       shuffle=False,
                                       pin_memory=True)

    def predict(self):
        print("prediction")
        line, label, ids = self.log_loader.get_last_sequence(self.system,
                                                             self.model_name,
                                                             self.window_size,
                                                             get_ids=True)
        event2vec = self.log_loader.template_to_vec_all(self.system)
        if self.model is None:
            self.initialize_model()
        dataset = sliddingWindowDataset(line,
                                        label,
                                        self.window_size,
                                        event2vec,
                                        num_classes=self.num_classes,
                                        seq=self.sequentials,
                                        quan=self.quantitatives,
                                        sem=self.semantic)

        model = self.model.to(self.device)
        model.eval()

        abnormal = list()
        for i in tqdm(range(len(dataset))):
            data, label = dataset[i]
            features = []
            for value in data.values():
                features.append(value[np.newaxis].to(self.device))

            # seq1 = torch.tensor(seq1, dtype=torch.float).view(
            #    -1, self.num_classes, self.input_size).to(self.device)
            label = torch.tensor(label).view(-1).to(self.device)
            output = model(features=features, device=self.device)
            predicted = torch.argsort(output,
                                      1)[0][-self.num_candidates:]
            if label not in predicted:
                abnormal.append(i+self.window_size)
        for i in abnormal:
            self.log_loader.set_abnormal_log(ids[i], self.model_name)
        return abnormal
