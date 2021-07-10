from datetime import datetime
from datetime import timedelta
from sklearn.model_selection import train_test_split

from ailoganalyzer.dataset.log import log_dataset
from ailoganalyzer.dataset.sample import sliding_window
from torch.utils.data import DataLoader

from ailoganalyzer.anomalyDetection.lstmModels.lstm import robustlog
from ailoganalyzer.database.dataLoader import LogLoader
from ailoganalyzer.tools.train import Trainer
from ailoganalyzer.anomalyDetection.AnomalyDetector import AnomalyDetector
import torch
from torch.multiprocessing import Process, set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass


class LSTMLogSequence(AnomalyDetector):
    def __init__(self, log_loader, num_candidates=5, model="robustLog", device="auto"):
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super().__init__(model, log_loader)
        self.semantic = True
        self.sequentials = False
        self.quantitatives = False
        self.window_size = 10
        self.num_candidates = num_candidates

        self.train_delta = timedelta(days=0)

        self.device = device

    def initialize_model(self):
        if self.model_name == "robustLog":
            self.model = robustlog(input_size=300,
                                   hidden_size=128,
                                   num_layers=2,
                                   num_keys=self.log_loader.get_number_classes(self.system))
        self.model_path = "result/" + self.model_name + "_" + self.system + "_" + "last.pth"

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
                          model_path=self.model_path,
                          device=self.device
                          )
        #Trainer.is_available = False
        p = Process(target=trainer.start_train)
        p.start()
        #Trainer.is_available = True
        return p

    def set_dataLoader_training(self):
        sequences = self.log_loader.get_sequences(self.system, 300)
        train_seq, val_seq = train_test_split(sequences, train_size=0.8)
        self.num_classes = self.log_loader.get_number_classes(self.system)
        train_dataset = sliding_window(
                                        self.log_loader,
                                        train_seq,
                                        self.window_size,
                                        system=self.system)
        valid_dataset = sliding_window(
                                        self.log_loader,
                                        val_seq,
                                        self.window_size,
                                        system=self.system)
        print("num_classes :: ", self.num_classes)

        print("end slidding")

        self.train_loader = DataLoader(train_dataset,
                                       batch_size=256,
                                       shuffle=True,
                                       pin_memory=True)
        self.valid_loader = DataLoader(valid_dataset,
                                       batch_size=256,
                                       shuffle=False,
                                       pin_memory=True)

    def predict(self):
        line, ids = self.log_loader.get_last_sequence(self.system, 1800)

        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()

        for i in range(len(line) - self.window_size):
            seq0 = line[i:i + self.window_size]
            if self.semantics:
                Semantic_pattern = []
                for event in seq0:
                    if event == -1:
                        Semantic_pattern.append(tuple([-1] * 300))
                    else:
                        self.event2semantic_vec = self.log_loader.template_to_vec_all(self.system)
                        Semantic_pattern.append(tuple(self.event2semantic_vec[str(event)]))
                seq0 = Semantic_pattern

            label = line[i + self.window_size]
            # quantitatives
            #for key in log_conuter:
            #    seq1[key] = log_conuter[key]

            seq0 = torch.tensor(seq0, dtype=torch.float).view(
                -1, self.window_size, self.input_size).to(self.device)
            #seq1 = torch.tensor(seq1, dtype=torch.float).view(
            #    -1, self.num_classes, self.input_size).to(self.device)
            label = torch.tensor(label).view(-1).to(self.device)
            output = model(features=[seq0, []], device=self.device)
            predicted = torch.argsort(output,
                                      1)[0][-self.num_candidates:]
            if label not in predicted:
                self.log_loader.set_abnormal_log(self.system, ids[i+self.window_size])
