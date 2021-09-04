import pymongo
from ailoganalyzer.database.log_persistence import LogPersistence


class MongoPersistence(LogPersistence):
    def __init__(self, database_name):
        self.database_name = database_name
        self.client = pymongo.MongoClient('localhost', 27017)
        self.db = self.client[database_name]
        super().__init__()

    def save_log(self, line, system, time=None, abnormal=None):
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

    def get_logs(self, system, limit):
        log = self.db["log"]
        res = log.find().sort("time", pymongo.DESCENDING).limit(limit)
        return list(res)

    def start_end_date(self, system):
        logs = self.db["log"]
        debut = logs.find({"system": system}).sort("time", pymongo.ASCENDING).limit(1)
        end = logs.find({"system": system}).sort("time", pymongo.DESCENDING).limit(1)
        return list(debut)[0]["time"], list(end)[0]["time"]
