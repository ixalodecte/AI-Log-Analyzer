class LogLoader():

    def __init__(self):
        self.systems = self.get_systems()
        self.template_miners = {}
        self.new_system = []

    def save_log(self, system, line, time=None, abnormal=None):
        raise NotImplementedError

    def get_systems(self):
        raise NotImplementedError

    def start_end_date(self, system: str):
        raise NotImplementedError

    def time_serie(self, system, dates, param={}) -> dict:
        time_serie = {}
        for a, b in zip(dates[:-1], dates[1:]):
            time_serie[b.to_pydatetime().isoformat()] = self.count_log(a, b)
        return time_serie
