import pymongo
from datetime import datetime, timedelta
from ailoganalyzer.database.dataLoader import LogLoader
from pylab import array
from numpy.lib.stride_tricks import sliding_window_view


class LogLoaderMongoDb(LogLoader):
    def __init__(self, database_name):
        self.database_name = database_name
        self.client = pymongo.MongoClient('localhost', 27017)
        self.db = self.client[database_name]
        super().__init__()

    def insert_line(self, system, line, date, template):
        collection = self.db["log"]
        collection.insert({"message": line,
                           "system": system,
                           "time": date,
                           "template": template})

    def remove_system_db(self, system):
        self.db["system"].delete_one({"system": system})
        self.db["log"].delete_many({"system": system})
        self.db["train"].delete_many({"system": system})
        self.db["last_predicted"].delete_many({"system": system})

    def set_trained(self, system, model_name):
        systems = self.db["train"]
        systems.insert({"system": system, "model": model_name})

    def unset_trained(self, system, model_name):
        raise NotImplementedError

    def is_trained(self, system, model_name):
        trained = self.db["train"]
        res = trained.find_one({"system": system, "model": model_name})
        return not (res is None)

    def get_systems(self):
        systems = self.db["system"]
        return [i["system"] for i in systems.find()]

    def add_system(self, system):
        collection = self.db["system"]
        collection.insert({"system": system})

    def set_abnormal_log(self, line, model_name):
        # TODO : a list of model that detect an anomaly
        logs = self.db["log"]
        logs.update_one({"_id": line}, {"$set": {"abnormal": True, "model": model_name}})

    def start_end_date(self, system):
        logs = self.db["log"]
        debut = logs.find({"system": system}).sort("time", pymongo.ASCENDING).limit(1)
        end = logs.find({"system": system}).sort("time", pymongo.DESCENDING).limit(1)
        return list(debut)[0]["time"], list(end)[0]["time"]

    def get_last_sequence(self, system, model_name, window_size, get_ids=False):
        logs = self.db["log"]
        filters = {}
        filters["system"] = system
        last_predicted = self.db["last_predicted"].find_one({
            "system": system,
            "model": model_name
        }, {"time": 1})

        if last_predicted is not None:
            last_predicted = last_predicted["time"]
            filters["time"] = {"gte": last_predicted}

        res = list(logs.find(filters, {"template": 1, "_id": 1, "time": 1}))

        sequence = array([r["template"] for r in res])
        labels = sequence[window_size:]
        sequence = sliding_window_view(sequence[:-1], window_size)
        print(sequence)
        if get_ids:
            ids = [r["_id"] for r in res]
        else:
            ids = []
        times = [r["time"] for r in res]
        assert len(times) >= 10

        self.db["last_predicted"].replace_one({
            "system": system,
            "model": model_name
        }, {"time": times[-window_size]}, upsert=True)

        return sequence, labels, ids
