from ailoganalyzer.database.mongodb import LogLoaderMongoDb
from ailoganalyzer.anomalyDetection.LSTMLogSequence import LSTMLogSequence
from ailoganalyzer.analyzer import Analyzer


file = "data/bgl2_1M"

log_loader = LogLoaderMongoDb("ailoganalyzer_db")

seq = LSTMLogSequence(log_loader)
analyzer = Analyzer(log_loader)
analyzer.add_anomaly_detector(seq)
analyzer.anomaly_detection()
