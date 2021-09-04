from ailoganalyzer.database.log_persistence import LogPersistence
import sqlite3


class Sqlite3Persistence(LogPersistence):
    """A wrappers around Sqlite3 to save logs.

    ...

    Attributes
    ----------
    db_name : str
        the name of the sqlite3 database

    """
    def __init__(self, db_name):
        # TODO: manage time
        self.db_name = db_name
        super().__init__()

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_name)
        self.conn.execute('''create table if not exists log (
                                  system TEXT,
                                  date TEXT,
                                  log TEXT,
                                  anomaly INTEGER)''')
        #self.datetime_format = "%y-%m-%d-%H.%M.%S.%f"
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.conn.commit()
        self.conn.close()

    def save_log(self, line, system, time=None, abnormal=None):
        #if time is not None:
        #    time = time.strftime(self.datetime_format)
        param = (system, time, line, abnormal)
        self.conn.execute("INSERT INTO log VALUES (?, ?, ?, ?)", param)

    def get_systems(self):
        return self.conn.execute("SELECT DISTINCT system FROM log").fetchall()

    def start_end_date(self, system : str):
        raise NotImplementedError

    def get_logs(self, system, limit=None, abnormal=None):
        query = "SELECT * FROM log WHERE system = ?"
        param = [system]
        if abnormal is not None:
            query += " AND abnormal = ?"
            param.append(int(abnormal))
        if limit is not None:
            query += " LIMIT ?"
            param.append(limit)
        return self.conn.execute(query, param).fetchall()
