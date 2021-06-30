from dataLoader import LogLoader
import sqlite3

class SqliteLogLoader(LogLoader):
    def __init__(self, system : str, db_name):
        super(system)
        self.db_name = db_name
        self.conn = sqlite3.connect(self.table_name)
        self.conn.execute('''create table if not exists LOG (
                                  ID INTEGER PRIMARY KEY AUTOINCREMENT,
                                  DATE DATE''')



    def insert_line(self, line, date):
        raise NotImplementedError

    def insert_data(self, data):
        for line,date in zip(data["line"],data["date"]):
            self.insert_line(line,date)

    def find(self, filters_d = {}, select_ls = None, count=False, limit=None):
        raise NotImplementedError

    def count_log(start_time, end_time):
        raise NotImplementedError

    def time_serie(self, dates, param = {}) -> dict:
        time_serie = {}
        for a, b in zip(dates[:-1], dates[1:]):
            time_serie[b.to_pydatetime().isoformat()] = self.count_log(a,b)
        return time_serie

    def set_trained(self, system):
        raise NotImplementedError

    def is_trained(self, system : str) -> bool:
        raise NotImplementedError

    def get_systems(self) -> list[str]:
         raise NotImplementedError

    def set_abnormal_log(self, line):
        raise NotImplementedError

    def start_end_date(self, system : str):
        raise NotImplementedError
