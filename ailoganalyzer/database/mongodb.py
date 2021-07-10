import pymongo
from datetime import datetime, timedelta
from pandas import date_range
from ailoganalyzer.database.dataLoader import LogLoader
from pylab import array


class LogLoaderMongoDb(LogLoader):
    def __init__(self, database_name):
        self.database_name = database_name
        self.client = pymongo.MongoClient('localhost', 27017)
        self.db = self.client[database_name]
        super().__init__()

    def insert_line(self, system, line, date, template=-1):
        collection = self.db["log"]
        collection.insert({"message": line,
                           "system": system,
                           "time": date,
                           "template": template})

    def insert_data(self, data):
        collection = self.db["log"]
        collection.insert_many(data)

    def drop_table(self, collection):
        self.db.drop_collection(collection)

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
        # modif : ajoute le model a une liste
        logs = self.db["log"]
        logs.update_one({"_id": line}, {"$set": {"abnormal": True, "model": model_name}})

    def start_end_date(self, system):
        logs = self.db["log"]
        debut = logs.find({"system": system}).sort("time", pymongo.ASCENDING).limit(1)
        end = logs.find({"system": system}).sort("time", pymongo.DESCENDING).limit(1)
        return list(debut)[0]["time"], list(end)[0]["time"]

    def get_last_sequence(self, system, window_size):
        logs = self.db["log"]
        end_time = datetime.now()
        start_time = end_time - timedelta(seconds=window_size)
        filters = {}
        filters["$and"] = [
                 {"time": {"$gte": start_time}},
                 {"time": {"$lte": end_time}}
        ]
        filters["system"] = system
        res = logs.find(filters, {"template": 1, "_id": 1})
        sequence = [r["template"] for r in res]
        ids = [r["_id"] for r in res]
        return sequence, ids

    def get_sequences(self, system, window_size):
        logs = self.db["log"]

        start, end = self.start_end_date(system)
        r = date_range(start, end, freq=str(window_size) + "S")
        sequences = []
        for s, e in zip(r[:-1], r[1:]):
            filters = {}
            filters["$and"] = [
                     {"time": {"$gte": s}},
                     {"time": {"$lte": e}}
            ]
            filters["system"] = system
            sequences.append(array([e["template"] for e in logs.find(filters, {"template": 1})], dtype=int))
        return sequences
