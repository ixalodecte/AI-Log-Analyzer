from ailoganalyzer.anomalyDetection.LSTMLogSequence import LSTMLogSequence


class DeepLog(LSTMLogSequence):
    def __init__(self, prefix_file, num_candidates=9, window_size=10,
                 device="auto", lr=0.001, lr_step=(300, 350),
                 lr_decay_ratio=0.1, max_iter=370):
        super().__init__(prefix_file, "deeplog", num_candidates, window_size,
                         device, lr, lr_step,
                         lr_decay_ratio, max_iter)


class LogAnomaly(LSTMLogSequence):
    def __init__(self, prefix_file, num_candidates=9, window_size=10,
                 device="auto", lr=0.001, lr_step=(50, 60),
                 lr_decay_ratio=0.1, max_iter=70):
        super().__init__(prefix_file, "loganomaly", num_candidates, window_size,
                         device, lr, lr_step,
                         lr_decay_ratio, max_iter)
