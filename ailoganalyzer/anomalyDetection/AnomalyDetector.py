class AnomalyDetector():
    def __init__(self, model_name):
        self.model_name = model_name
        self.mode = "train"

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def initialize_model(self):
        raise NotImplementedError

    def set_mode(self, mode):
        if mode == "train" or mode == "predict":
            self.mode = mode
            return True
        return False
