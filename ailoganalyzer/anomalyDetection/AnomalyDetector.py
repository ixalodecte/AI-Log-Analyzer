class AnomalyDetector():
    def __init__(self, model_name, log_loader):
        self.log_loader = log_loader
        self.model_name = model_name

    def set_system(self, system):
        self.system = system
        self.initialize_model()

    def is_trainable(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def initialize_model(self):
        raise NotImplementedError

    def set_dataLoader_training(self, log_loader):
        pass
