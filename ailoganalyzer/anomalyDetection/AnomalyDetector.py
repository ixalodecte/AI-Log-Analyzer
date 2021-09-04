class AnomalyDetector():
    def __init__(self, model_name):
        self.model_name = model_name
        self.is_trained = False

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def initialize_model(self):
        raise NotImplementedError
