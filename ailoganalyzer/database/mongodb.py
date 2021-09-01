import pymongo
from ailoganalyzer.database.dataLoader import LogLoader
from pylab import array
from numpy.lib.stride_tricks import sliding_window_view


class LogLoaderMongoDb(LogLoader):
    def __init__(self, database_name):
        self.database_name = database_name
        self.client = pymongo.MongoClient('localhost', 27017)
        self.db = self.client[database_name]
        super().__init__()

    def save_log(self, system, line, time=None, abnormal=None):
        collection = self.db["log"]
        collection.insert({"message": line,
                           "system": system,
                           "time": time,
                           "abnormal": abnormal()})

    def remove_system_db(self, system):
        self.db["log"].delete_many({"system": system})

    def get_systems(self):
        systems = self.db["log"]
        return [i["system"] for i in systems.distinct("system")]

    def start_end_date(self, system):
        logs = self.db["log"]
        debut = logs.find({"system": system}).sort("time", pymongo.ASCENDING).limit(1)
        end = logs.find({"system": system}).sort("time", pymongo.DESCENDING).limit(1)
        return list(debut)[0]["time"], list(end)[0]["time"]
