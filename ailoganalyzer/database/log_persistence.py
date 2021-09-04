class LogPersistence():
    """An abstract class for implementing a "log saver".

    ...

    Attributes
    ----------

    Methods
    -------
    save_log(line, system, time=None, abnormal=None)
        save a log and informations related to it.

    get_systems()
        return all systems that have log(s) in the database.

    start_end_date(system)
        return the datetime of the first log and the datetime of the last log

    get_logs(self, system, limit=None, abnormal=None)
        return the logs of a system.
    """

    def __init__(self):
        pass

    def save_log(self, line, system, time=None, abnormal=None):
        raise NotImplementedError

    def get_systems(self):
        raise NotImplementedError

    def start_end_date(self, system: str):
        raise NotImplementedError

    def get_logs(self, system, limit=None, abnormal=None):
        raise NotImplementedError

    def time_serie(self, system, dates, param={}) -> dict:
        time_serie = {}
        for a, b in zip(dates[:-1], dates[1:]):
            time_serie[b.to_pydatetime().isoformat()] = self.count_log(a, b)
        return time_serie
