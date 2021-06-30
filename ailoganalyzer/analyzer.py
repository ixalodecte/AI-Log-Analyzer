from copy import copy

class Analyzer():
    def __init__(self, log_loader):
        self.log_loader = log_loader
        self.anomaly_detectors_all = []
        self.anomaly_detectors = []

    def add_anomaly_detector(self, anomaly_detector, system="all"):
        if system == "all":
            for system in self.log_loader.get_systems():
                self.add_anomaly_detector(anomaly_detector, system)


        anomaly_detector.set_system(system)
        self.anomaly_detectors.append(copy(anomaly_detector))

    # Pb : log loader pas dans le meme processus (fichier)
    def check_new_system(self):
        for system in self.log_loader.new_system:
            for anomaly_detector in self.anomaly_detectors_all():
                self.add_anomaly_detector(anomaly_detector, system)

    def anomaly_detection(self):
        self.check_new_system()
        for elt in self.anomaly_detectors:
            if elt.is_trainable():
                elt.train()
            if self.log_loader.is_trained(elt.system, elt.model_name):
                elt.predict()
