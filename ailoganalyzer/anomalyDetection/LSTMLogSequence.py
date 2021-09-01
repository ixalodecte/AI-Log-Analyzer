from datetime import timedelta
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig
from drain3 import TemplateMiner
from ailoganalyzer.dataset.line_to_vec import line_to_vec, preprocess_template
from collections import defaultdict
import re
import random

from ailoganalyzer.anomalyDetection.lstmModels.lstm import loganomaly, deeplog
from ailoganalyzer.tools.train import Trainer
from ailoganalyzer.anomalyDetection.AnomalyDetector import AnomalyDetector
from ailoganalyzer.dataset.sample import sliddingWindowDataset


class LSTMLogSequence(AnomalyDetector):
    def __init__(self, system, num_candidates=9, model="robustLog", device="auto"):
        Path("data").mkdir(parents=True, exist_ok=True)
        self.persistence_path = "data/templates_persist_" + model + "_" + system + ".bin"
        persistence = FilePersistence(self.persistence_path)
        config = TemplateMinerConfig()
        config.load("ailoganalyzer/drain3.ini")
        config.profiling_enabled = False
        self.template_miner = TemplateMiner(persistence, config)
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super().__init__(model)

        self.system = system
        self.semantic = False
        self.sequentials = False
        self.quantitatives = False
        self.window_size = 10
        self.num_candidates = num_candidates

        self.train_delta = timedelta(days=0)

        self.device = device
        self.model = None

        self.sequence = []
        self.train_seq = []
        self.train_loader = None
        self.valid_loader = None
        self.model_path = "data/" + self.model_name + "_" + self.system + "_" + "last.pth"

    def add_log(self, log):
        result = self.template_miner.add_log_message(log)
        if result["change_type"] != "none":
            pass
        cluster_id = result["cluster_id"] - 1

        if self.mode == "train":
            self.train_seq.append(cluster_id)

        else:
            if len(self.train_seq) > 0:
                self.sequence = self.train_seq[-self.window_size:]
                self.train_seq = []

            self.sequence = np.array(self.sequence)
            label = np.array([cluster_id])
            if len(self.sequence) == self.window_size:
                res = self.predict(self.sequence, label)
            else:
                res = False

            if len(self.sequence) == self.window_size:
                self.sequence = self.sequence[1:]
            self.sequence = np.append(self.sequence, cluster_id)
            return res

    def initialize_model(self):
        state = None
        if os.path.isfile(self.model_path):
            state = torch.load(self.model_path)
            num_classes = state["num_keys"]
        else:
            num_classes = self.get_number_classes()
        self.num_classes = num_classes

        if self.model_name == "loganomaly":
            self.model = loganomaly(hidden_size=128,
                                    num_layers=2,
                                    num_keys=num_classes)
            self.input_size = 300
            self.semantic = True
            self.quantitatives = True
            self.batch_size = 256
            self.nb_epoch = 60
            self.lr_step = (40, 50)

        elif self.model_name == "deeplog":
            self.model = deeplog(hidden_size=64,
                                 num_layers=2,
                                 num_keys=num_classes)
            self.input_size = 1
            self.sequentials = True
            self.batch_size = 2048
            self.nb_epoch = 370
            self.lr_step = (300, 350)

        elif self.model_name == "robustlog":
            raise NotImplementedError

        else:
            raise NotImplementedError

        if state is not None:
            self.model.load_state_dict(state["state_dict"])

    def train(self):
        if self.train_loader is None or self.valid_loader is None:
            self.set_dataLoader_training()

        print("num classes:", self.num_classes)
        trainer = Trainer(self.model,
                          self.train_loader,
                          self.valid_loader,
                          self.num_classes,
                          self.system,
                          self.model_name,
                          self.window_size,
                          max_epoch=self.nb_epoch,
                          lr_step=self.lr_step,
                          model_path=self.model_path,
                          device=self.device
                          )
        trainer.start_train()

    def set_dataLoader_training(self):
        self.train_seq = np.array(self.train_seq)
        labels = self.train_seq[self.window_size:]
        sequences = sliding_window_view(self.train_seq[:-1], self.window_size)
        self.set_dataLoader_training_1(sequences, labels)

    def set_dataLoader_training_1(self, sequences, labels):
        self.initialize_model()

        train_seq, val_seq, train_label, val_label = train_test_split(sequences, labels, train_size=0.8)
        print("number train sequences :", len(train_seq))
        print("number val sequences :", len(val_seq))
        self.num_classes = self.get_number_classes()
        event2vec = self.template_to_vec_all()

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

        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       pin_memory=True)
        self.valid_loader = DataLoader(valid_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       pin_memory=True)

    def predict(self, sequence, label):
        sequence = sequence[np.newaxis]
        event2vec = self.template_to_vec_all()
        if self.model is None:
            self.initialize_model()
        self.model = self.model.eval().to(self.device)

        label = np.array([label])
        dataset = sliddingWindowDataset(sequence,
                                        label,
                                        self.window_size,
                                        event2vec,
                                        num_classes=self.num_classes,
                                        seq=self.sequentials,
                                        quan=self.quantitatives,
                                        sem=self.semantic)

        data, label = dataset[0]
        features = []
        for value in data.values():
            features.append(value[np.newaxis].to(self.device))

        label = torch.tensor(label).view(-1).to(self.device)
        output = self.model(features=features, device=self.device)
        predicted = torch.argsort(output,
                                  1)[0][-self.num_candidates:]

        if label not in predicted:
            return True
        else:
            return False

    def evaluate_HDFS(self, train=True):
        config = TemplateMinerConfig()
        config.load("ailoganalyzer/drain3.ini")
        config.profiling_enabled = False
        self.template_miner = TemplateMiner(config=config)

        hdfs_log = "../../Documents/HDFS_1/HDFS.log"
        hdfs_anomaly_label = "../../Documents/HDFS_1/anomaly_label.csv"
        nb_block = 30000

        with open(hdfs_anomaly_label, "r") as f:
            hdfs_labels = {}
            for i, line in tqdm(enumerate(f), total=nb_block):
                label = line.strip().split(",")
                hdfs_labels[label[0]] = (label[1] == "Anomaly")
        keys = random.sample(list(hdfs_labels), nb_block)
        values = [hdfs_labels[k] for k in keys]
        hdfs_labels = dict(zip(keys, values))
        #print("\n".join(self.get_templates()))

        blk_finder_2 = re.compile("(blk_-?\d+)")
        with open(hdfs_log, "r") as f:
            data_dict = {key: [] for key in hdfs_labels.keys()}
            for line in tqdm(f):
                blk = re.search(blk_finder_2, line).group()
                if blk in data_dict:
                    msg = " ".join(line.strip().split()[5:])
                    result = self.template_miner.add_log_message(msg)
                    cluster_id = result["cluster_id"] - 1
                    data_dict[blk].append(cluster_id)

        abnormal = []
        normal = []
        abnormal_label = []
        normal_label = []
        abnormal_blk = []

        for blk, seq in data_dict.items():
            if len(seq) > self.window_size:
                labels = seq[self.window_size:]
                seqs = sliding_window_view(seq[:-1], self.window_size)
                if hdfs_labels[blk]:
                    abnormal.append(seqs)
                    abnormal_label.append(labels)
                    abnormal_blk.append(blk)
                else:
                    normal.append(seqs)
                    normal_label.append(labels)

        print("normal : ", len(normal))
        print("abnormal : ", len(abnormal))
        train_seq, test_seq, train_label, test_label = train_test_split(normal, normal_label, train_size=0.8)
        train_seq = np.concatenate(train_seq)
        train_label = np.concatenate(train_label)

        if train:
            self.set_dataLoader_training_1(train_seq, train_label)
            self.train()

        # predict

        FP = 0
        TP = 0
        mem = {}
        for seqs, labels in tqdm(zip(test_seq, test_label), total=len(test_seq)):
            for seq, label in zip(seqs, labels):
                seq_tuple = tuple(seq + [label])
                if seq_tuple in mem:
                    result = mem[seq_tuple]
                else:
                    result = self.predict(seq, label)
                    mem[seq_tuple] = result
                if result:
                    FP += 1
                    break
        for seqs, labels in tqdm(zip(abnormal, abnormal_label), total=len(abnormal)):
            for seq, label in zip(seqs, labels):
                seq_tuple = tuple(seq + [label])
                if seq_tuple in mem:
                    result = mem[seq_tuple]
                else:
                    result = self.predict(seq, label)
                    mem[seq_tuple] = result
                if result:
                    TP += 1
                    break
        FN = len(abnormal) - TP
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(FP, FN, P, R, F1))

    # -------------- drain3 function -----------------

    def get_templates(self):
        return (c.get_template() for c in self.template_miner.drain.clusters)

    def get_number_classes(self):
        return len(list(self.get_templates()))

    def get_word_counter(self):
        d = defaultdict(int)
        for cluster in self.template_miner.drain.clusters:
            for word in preprocess_template(cluster.get_template()):
                d[word] += cluster.size
        return d

    def template_to_vec_all(self):
        d = {}
        d[0] = np.array([-1] * 300)
        word_counter = self.get_word_counter()
        for cluster in self.template_miner.drain.clusters:
            template, template_id = cluster.get_template(), cluster.cluster_id
            d[template_id] = line_to_vec(template, word_counter)
        return d

    def template_to_vec(self, templateID):
        if templateID == 0:
            return np.array([-1] * 300)
        for cluster in self.template_miner.drain.clusters:
            if cluster.cluster_id == templateID:
                word_counter = self.get_word_counter()
                return line_to_vec(cluster.get_template(), word_counter)

        print(templateID)
        raise RuntimeError

    def remove_system(self):
        os.remove("data/templates_persist_" + self.model_name + "_" + self.system + ".bin")
