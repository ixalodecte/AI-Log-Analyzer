from drain3.file_persistence import FilePersistence
from drain3 import TemplateMiner
from ailoganalyzer.dataset.line_to_vec import line_to_vec, preprocess_template
from collections import defaultdict
import numpy as np
from tqdm import tqdm


class LogLoader():

    def __init__(self):
        self.systems = self.get_systems()
        self.template_miners = {}
        self.new_system = []
        for system in self.systems:
            persistence = FilePersistence("data/preprocess/templates_persist_" + system + ".bin")
            self.template_miners[system] = TemplateMiner(persistence)

    def add_system(self, system):
        raise NotImplementedError

    def insert_line(self, system, line, date, template=-1):
        raise NotImplementedError

    def count_log(self, system, start_time, end_time):
        raise NotImplementedError

    def set_trained(self, system : str):
        raise NotImplementedError

    def is_trained(self, system : str) -> bool:
        raise NotImplementedError

    def get_systems(self):
        raise NotImplementedError

    def set_abnormal_log(self, system : str, line):
        raise NotImplementedError

    def start_end_date(self, system : str):
        raise NotImplementedError

    def get_last_sequence(self, system, windows_size):
        raise NotImplementedError

    def get_sequences(self, windows_size):
        raise NotImplementedError

    def get_templates(self, system : str):
        return (c.get_template() for c in self.template_miners[system].drain.clusters)

    def get_number_classes(self, system):
        return len(list(self.get_templates(system))) + 1

    def add_log(self, system : str, line : str, date):
        if system not in self.systems:
            self.systems.append(system)

            persistence = FilePersistence("data/preprocess/templates_persist_" + system + ".bin")
            self.template_miners[system] = TemplateMiner(persistence)
            self.add_system(system)
            self.new_system.append(system)

        result = self.template_miners[system].add_log_message(line)
        if result["change_type"] != "none":
            pass
            #print(result)
            #result_json = json.dumps(result)
            #logger.info(f"Input ({line_count}): " + line)
            #logger.info("Result: " + result_json)
        #template_mined = result["template_mined"].split()

        cluster_id = result["cluster_id"]
        self.insert_line(system, line, date, cluster_id)

    def insert_data(self, system, data):
        for line, date in zip(data["line"], data["date"]):
            self.insert_line(line, date, system)

    def time_serie(self, system, dates, param={}) -> dict:
        time_serie = {}
        for a, b in zip(dates[:-1], dates[1:]):
            time_serie[b.to_pydatetime().isoformat()] = self.count_log(a, b)
        return time_serie

    def get_word_counter(self, system):
        d = defaultdict(int)
        for cluster in self.template_miners[system].drain.clusters:
            for word in preprocess_template(cluster.get_template()):
                d[word] += cluster.size
            print("new template")
        # print(d)
        return d

    def template_to_vec_all(self, system):
        d = {}
        d[0] = np.array([-1] * 300)
        print("h")
        word_counter = self.get_word_counter(system)
        print("hello")
        for cluster in tqdm(self.template_miners[system].drain.clusters):
            template, template_id = cluster.get_template(), cluster.cluster_id
            d[template_id] = line_to_vec(template, word_counter)
        return d

    def template_to_vec(self, system, templateID):
        if templateID == 0:
            return np.array([-1] * 300)
        for cluster in self.template_miners[system].drain.clusters:
            if cluster.cluster_id == templateID:
                word_counter = self.get_word_counter(system)
                return line_to_vec(cluster.get_template(), word_counter)

        print(templateID)
        raise RuntimeError
