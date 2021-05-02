import pymongo
from datetime import datetime


def information_extractor(line, log_structure):
    info = {
        "message": float("NaN"),
        "time": float("NaN"),
        "abnormal": float("NaN")
    }
    separator = log_structure["separator"]
    if separator:
        line = line.split(separator)
        info["message"] = " ".join(line[log_structure["message_start_index"] : log_structure["message_end_index"]])
        # print(line[log_structure["time_index"]])
        try:
            info["time"] = datetime.strptime(line[log_structure["time_index"]], log_structure["time_format"])
        except ValueError:
            print("format invalide")
            return {}
        if log_structure["label_index"] != None:
            info["abnormal"] = line[log_structure["label_index"]] != "-" 

    else:
        info["message"] = line
    return info


class LogLoader():
    def __init__(self, database_name):
        self.database_name = database_name
        self.client = pymongo.MongoClient('localhost',27017)
        self.db = self.client[database_name]
        self.fields = ["abnormal","time","message", "system"]

    def insert_data(self, collection_str, data):
        collection = self.db[collection_str]
        collection.insert_many(data)
        print("hehe")
        if collection_str == "logs":
            print("hello")
            if self.is_trained(data[0]["system"]) is None:
                print("insertion")
                self.insert_data("system", [{"system": data[0]["system"], "trained": False}])

    def drop_table(self, collection):
        self.db.drop_collection(collection)

    # Fonctions a utiliser pour inserer un fichier de logs
    def insert_raw_log(self, collection, f_stream, log_structure):
        log_data = []
        for line in f_stream:
            d = information_extractor(line.strip(), log_structure)
            d["system"] = "192.168.1.1"
            if d != {}:
                log_data.append(d)
        self.insert_data(collection, log_data)

    def find(self, collection, filters_d = {}, select_ls = None, count=False, limit=None):
        filters = filters_d.copy()
        collection_str = collection
        collection = self.db[collection]
        field = None
        if collection_str == "logs":
            if select_ls is None:
                select_ls = self.fields
            field = dict(zip(select_ls, [1] * len(select_ls)))
            field["_id"] = 0
        else:
            if select_ls is not None:
                field = dict(zip(select_ls, [1] * len(select_ls)))
                field["_id"] = 0


        # recherche entre deux dates
        if "start_time" in filters or "end_time" in filters:
            if "start_time" in filters:
                start_time = filters["start_time"]
                filters.pop("start_time")
            else:
                start_time = datetime(1970, 1, 1)
            if "end_time" in filters:
                end_time = filters["end_time"]
                filters.pop("end_time")
            else:
                end_time = datetime.now()
            filters["$and"] = [
                     {"time": {"$gte": start_time}},
                     {"time": {"$lte": end_time}}
            ]
        print("hello filterssss",filters)

        if field is not None:
            print(field)
            # sys.exit(1)

            if limit is not None:
                cursor = collection.find(filters,field).sort("time", pymongo.DESCENDING).limit(limit)
            else:
                cursor = collection.find(filters,field).sort("time", pymongo.ASCENDING)
        else:
            cursor = collection.find(filters)
        if count:
            return cursor.count(True)
        return list(cursor)

    def time_serie(self, collection, dates, param = {}):
        time_serie = {}
        for a, b in zip(dates[:-1], dates[1:]):
            param["start_time"] = a.to_pydatetime()
            param["end_time"] = b.to_pydatetime()

            time_serie[b.to_pydatetime().isoformat()] = self.find("logs", param, count = True)
        return time_serie

    def set_trained(self, system):
        systems = self.db["system"]
        systems.update_one({"system": system}, {"$set": {"trained": True}})

    def unset_trained(self, system):
        systems = self.db["system"]
        systems.update_one({"system": system}, {"$set": {"trained": False}})

    def is_trained(self, system):
        systems = self.db["system"]
        res = systems.find_one({"system": system})
        if res is None:
            return res
        return res["trained"]

    def get_systems(self):
         return [i["system"] for i in self.find("system", select_ls=["system"])]

    def set_abnormal_log(self, line):
        logs = self.db["logs"]
        logs.update_one(line, {"$set": {"abnormal": True}})

    def start_end_date(self, system):
        logs = self.db["logs"]
        debut = logs.find({"system":system}).sort("time", pymongo.ASCENDING).limit(1)
        end = logs.find({"system":system}).sort("time", pymongo.DESCENDING).limit(1)
        return list(debut)[0]["time"], list(end)[0]["time"]


if __name__ == "__main__":
    log_structure = {
        "separator": ' ',          # separateur entre les champs d'une ligne
        "time_index": 4,           # index timestamp
        "time_format": "%Y-%m-%d-%H.%M.%S.%f",
        "message_start_index": 6,  # debut message
        "message_end_index": None, # fin message (None si on va jusqu'a la fin de ligne)
        "label_index": 0           # index label (None si aucun)
    }

    db = LogLoader("ailoganalyzer_db")

    db.drop_table("logs")
    db.drop_table("system")
    # --> Décommente les 2 lignes suivantes pour inserer le fichier bgl2_100k dans la base de donnée
    with open("../data/bgl2_100k_modif", "r") as f:
        db.insert_raw_log("logs", f, log_structure)

    # equivalent des parametres where dans le select
    filters = {
        "abnormal": False,
        "start_time": datetime.strptime("2005-06-03-16.12.34.557453", "%Y-%m-%d-%H.%M.%S.%f"),
        "end_time": datetime.strptime("2005-06-03-16.17.40.435081", "%Y-%m-%d-%H.%M.%S.%f")

    }

    # Selection des champs
    select_ls = [
        "message",
        "time",
        "system"
    ]
    print(db.find("logs", filters))
    print()

    print((db.find("logs", limit=2)))
    print(db.start_end_date("192.168.1.1"))
    #db.set_trained("192.168.1.1")
    print(db.is_trained("192.168.1.1"))
    print(db.get_systems())
