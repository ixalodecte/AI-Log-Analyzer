#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import os
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from ailoganalyzer.dataset.log import log_dataset
from ailoganalyzer.dataset.sample import sliding_window, session_window


class Trainer():
    def __init__(self, model, log_loader, system, model_name, window_size, model_path, device = "cuda", lr_step = (4,5), lr_decay_ratio = 0.1):
        print("start training for ", system, "model", model_name)
        self.log_loader = log_loader
        self.model_name = model_name
        self.save_dir = "result"
        self.window_size = window_size
        self.batch_size = 256
        self.lr = 0.001

        self.optimizer_fun = 'sgd'

        self.device = device
        self.lr_step = lr_step
        self.lr_decay_ratio = lr_decay_ratio
        self.accumulation_step = 1
        self.max_epoch = 6

        self.sequentials = False
        self.quantitatives = False
        self.semantics = True
        self.sample = "sliding_window"
        #self.feature_num = options['feature_num']
        self.num_classes = log_loader.get_number_classes(system)
        self.system = system
        self.model_path = model_path


        os.makedirs(self.save_dir, exist_ok=True)
        sequences = log_loader.get_sequences(system, 1800)
        train_seq, val_seq = train_test_split(sequences, train_size=0.8)
        if self.sample == 'sliding_window':
            train_logs, train_labels = sliding_window(
                                            self.log_loader,
                                            train_seq,
                                            self.num_classes,
                                            self.window_size,
                                            system = self.system)
            val_logs, val_labels = sliding_window(
                                            self.log_loader,
                                            val_seq,
                                            self.num_classes,
                                            self.window_size,
                                            system = self.system)
            print("end slidding")
        elif self.sample == 'session_window':
            train_logs, train_labels = session_window(self.data_dir,
                                                      self.num_classes,
                                                      datatype='train')
            val_logs, val_labels = session_window(self.data_dir,
                                                  self.num_classes,
                                                  datatype='val')
        else:
            raise NotImplementedError

        train_dataset = log_dataset(logs=train_logs,
                                    labels=train_labels,
                                    seq=self.sequentials,
                                    quan=self.quantitatives,
                                    sem=self.semantics)
        valid_dataset = log_dataset(logs=val_logs,
                                    labels=val_labels,
                                    seq=self.sequentials,
                                    quan=self.quantitatives,
                                    sem=self.semantics)

        del train_logs
        del val_logs
        gc.collect()

        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       pin_memory=True)
        self.valid_loader = DataLoader(valid_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       pin_memory=True)

        self.num_train_log = len(train_dataset)
        self.num_valid_log = len(valid_dataset)

        print('Find %d train logs, %d validation logs' %
              (self.num_train_log, self.num_valid_log))
        print('Train batch size %d ,Validation batch size %d' %
              (self.batch_size, self.batch_size))

        self.model = model.to(self.device)

        if self.optimizer_fun == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.lr,
                                             momentum=0.9)
        elif self.optimizer_fun == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                betas=(0.9, 0.999),
            )
        else:
            raise NotImplementedError

        self.start_epoch = 0
        self.best_loss = 1e10
        self.best_score = -1
        #save_parameters(options, self.save_dir + "parameters.txt")
        self.log = {
            "train": {key: []
                      for key in ["epoch", "lr", "time", "loss"]},
            "valid": {key: []
                      for key in ["epoch", "lr", "time", "loss"]}
        }
        #if options['resume_path'] is not None:
        #    if os.path.isfile(options['resume_path']):
        #        self.resume(options['resume_path'], load_optimizer=True)
        #    else:
        #        print("Checkpoint not found")

    def resume(self, path, load_optimizer=True):
        print("Resuming from {}".format(path))
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.log = checkpoint['log']
        self.best_f1_score = checkpoint['best_f1_score']
        self.model.load_state_dict(checkpoint['state_dict'])
        if "optimizer" in checkpoint.keys() and load_optimizer:
            print("Loading optimizer state dict")
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save_checkpoint(self, epoch, save_optimizer=True, suffix="", saveLast=False):
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "best_loss": self.best_loss,
            "log": self.log,
            "best_score": self.best_score
        }
        if save_optimizer:
            checkpoint['optimizer'] = self.optimizer.state_dict()
        if saveLast:
            save_path = self.model_path
        else:
            save_path = self.save_dir + self.model_name + "_" + self.system + "_" + suffix + ".pth"
        torch.save(checkpoint, save_path)
        print("Save model checkpoint at {}".format(save_path))

    def save_log(self):
        try:
            for key, values in self.log.items():
                pd.DataFrame(values).to_csv(self.save_dir + key + "_log.csv",
                                            index=False)
            print("Log saved")
        except:
            print("Failed to save logs")

    def train(self, epoch):
        self.log['train']['epoch'].append(epoch)
        start = time.strftime("%H:%M:%S")
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        print("Starting epoch: %d | phase: train | ⏰: %s | Learning rate: %f" %
              (epoch, start, lr))
        self.log['train']['lr'].append(lr)
        self.log['train']['time'].append(start)
        self.model.train()
        self.optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        tbar = tqdm(self.train_loader, desc="\r")
        num_batch = len(self.train_loader)
        total_losses = 0
        for i, (log, label) in enumerate(tbar):
            features = []
            for value in log.values():
                features.append(value.clone().detach().to(self.device))
            output = self.model(features=features, device=self.device)
            loss = criterion(output, label.to(self.device))
            total_losses += float(loss)
            loss /= self.accumulation_step
            loss.backward()
            if (i + 1) % self.accumulation_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            tbar.set_description("Train loss: %.5f" % (total_losses / (i + 1)))

        self.log['train']['loss'].append(total_losses / num_batch)

    def valid(self, epoch):
        self.model.eval()
        self.log['valid']['epoch'].append(epoch)
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.log['valid']['lr'].append(lr)
        start = time.strftime("%H:%M:%S")
        print("Starting epoch: %d | phase: valid | ⏰: %s " % (epoch, start))
        self.log['valid']['time'].append(start)
        total_losses = 0
        criterion = nn.CrossEntropyLoss()
        tbar = tqdm(self.valid_loader, desc="\r")
        num_batch = len(self.valid_loader)
        for i, (log, label) in enumerate(tbar):
            with torch.no_grad():
                features = []
                for value in log.values():
                    features.append(value.clone().detach().to(self.device))
                output = self.model(features=features, device=self.device)
                loss = criterion(output, label.to(self.device))
                total_losses += float(loss)
        print("Validation loss:", total_losses / num_batch)
        self.log['valid']['loss'].append(total_losses / num_batch)

        if total_losses / num_batch < self.best_loss:
            self.best_loss = total_losses / num_batch
            self.save_checkpoint(epoch,
                                 save_optimizer=False,
                                 suffix="bestloss")

    def start_train(self):
        for epoch in range(self.start_epoch, self.max_epoch):
            if epoch == 0:
                self.optimizer.param_groups[0]['lr'] /= 32
            if epoch in [1, 2, 3, 4, 5]:
                self.optimizer.param_groups[0]['lr'] *= 2
            if epoch in self.lr_step:
                self.optimizer.param_groups[0]['lr'] *= self.lr_decay_ratio
            self.train(epoch)
            if epoch >= self.max_epoch // 2 and epoch % 2 == 0:
                self.valid(epoch)
                self.save_checkpoint(epoch,
                                     save_optimizer=True,
                                     suffix="epoch" + str(epoch))
            self.save_checkpoint(epoch, save_optimizer=True, saveLast=True)
            self.save_log()
