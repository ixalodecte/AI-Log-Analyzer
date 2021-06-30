from datetime import datetime
from datetime import timedelta


from ailoganalyzer.anomalyDetection.lstmModels.lstm import robustlog
from ailoganalyzer.database.dataLoader import LogLoader
from ailoganalyzer.tools.train import Trainer
from ailoganalyzer.anomalyDetection.AnomalyDetector import AnomalyDetector
import torch


class LSTMLogSequence(AnomalyDetector):
    def __init__(self, log_loader : LogLoader, num_candidates = 5, model = "robustLog", device="cuda"):
        super().__init__(log_loader,model)
        self.semantic = True
        self.sequentials = False
        self.quantitatives = False
        self.window_size = 10
        self.num_candidates = num_candidates

        self.train_delta = timedelta(days = 0)

        self.device = device

    def initialize_model(self):

        if self.model_name == "robustLog":
            self.model = robustlog(input_size=300,
                            hidden_size=128,
                            num_layers=2,
                            num_keys=self.log_loader.get_number_classes(self.system))
        self.model_path = "result/" + self.model_name + "_" + self.system + "_" + "last.pth"

    def is_trainable(self):
        start, end = self.log_loader.start_end_date(self.system)
        return datetime.now() - start > self.train_delta and not self.log_loader.is_trained(self.system, self.model_name)


    def train(self):
        trainer = Trainer(self.model,
                          self.log_loader,
                          self.system,
                          self.model_name,
                          self.window_size,
                          model_path = self.model_path
                          )
        trainer.start_train()
        self.log_loader
        self.log_loader.set_trained(self.system, self.model_name)

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
                #return True
        #return False
