from copy import copy
import time
import threading


class Controler():
    def __init__(self, log_loader, predict_frequence=5):
        self.log_loader = log_loader
        self.anomaly_detectors_all = []
        self.anomaly_detectors = []
        self.predict_frequence = predict_frequence
        self.systems = []
        #self.anomaly_detection()
        threading.Thread(target=self.anomaly_detection).start()

    def add_anomaly_detector(self, anomaly_detector, system="all"):
        if system == "all":
            self.anomaly_detectors_all.append(anomaly_detector)
            for system in self.log_loader.get_systems():
                self.add_anomaly_detector(anomaly_detector, system)
                self.systems.append(system)
            return

        print("add model", anomaly_detector.model_name, "for system", system)
        anomaly_detector.set_system(system)
        self.anomaly_detectors.append(copy(anomaly_detector))

    def check_new_system(self):
        for system in self.log_loader.get_systems():
            if system not in self.systems:
                for anomaly_detector in self.anomaly_detectors_all:
                    self.add_anomaly_detector(anomaly_detector, system)

    def anomaly_detection(self):
        is_training = {}
        while(1):
            for system, val in is_training.items():
                process, model_name = val
                if not process.is_alive():
                    self.log_loader.set_trained(system, model_name)

            is_training = {key: val for key, val in is_training.items() if val[0].is_alive()}

            time.sleep(self.predict_frequence)
            self.last_time = time.time()
            self.check_new_system()
            for elt in self.anomaly_detectors:
                print("trainable:", elt.is_trainable())
                if elt.is_trainable() and elt.system not in is_training:
                    elt.set_dataLoader_training()
                    p = elt.train()
                    if p is not None:
                        is_training[elt.system] = (p, elt.model_name)
                    else:
                        self.log_loader.set_trained(elt.system, elt.model_name)
                if self.log_loader.is_trained(elt.system, elt.model_name):
                    elt.predict()

    def add_log(self, system: str, line: str, date):
        self.log_loader.add_log(system, line, date)
