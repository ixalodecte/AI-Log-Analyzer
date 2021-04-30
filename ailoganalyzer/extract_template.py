import json
import logging
import sys
import time
import pandas as pd

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence


def log2template(lines, out_file, options, log_structure = None):
    persistence = FilePersistence(options["preprocess_dir"] + "templates_persist_"+options["system"]+".bin")

    logger = logging.getLogger(__name__)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')

    template_miner = TemplateMiner(persistence)

    line_count = 0
    start_time = time.time()
    batch_start_time = start_time
    batch_size = 10000
    for line in lines:
        line = line.rstrip()
        if log_structure:
            if log_structure["separator"]:
                line = " ".join(line.split(log_structure["separator"])[log_structure["message_start_index"]:log_structure["message_end_index"]])
        result = template_miner.add_log_message(line)
        line_count += 1
        if line_count % batch_size == 0:
            time_took = time.time() - batch_start_time
            rate = batch_size / time_took
            logger.info(f"Processing line: {line_count}, rate {rate:.1f} lines/sec, "
                        f"{len(template_miner.drain.clusters)} clusters so far.")
            batch_start_time = time.time()
        if result["change_type"] != "none":
            print(result)
            result_json = json.dumps(result)
            logger.info(f"Input ({line_count}): " + line)
            logger.info("Result: " + result_json)

    time_took = time.time() - start_time
    rate = line_count / time_took
    logger.info(f"--- Done processing file. Total of {line_count} lines, rate {rate:.1f} lines/sec, "
                f"{len(template_miner.drain.clusters)} clusters")
    sorted_clusters = sorted(template_miner.drain.clusters, key=lambda it: it.size, reverse=True)

    template_df = pd.DataFrame(columns=["EventId","EventTemplate"])
    cluster_id = []
    template_str = []
    for cluster in sorted_clusters:
        print(str(cluster.cluster_id) + "," + '"' + cluster.get_template() + '"')
        cluster_id.append(cluster.cluster_id)
        template_str.append(cluster.get_template())
    template_df["EventId"] = cluster_id
    template_df["EventTemplate"] = template_str
    template_df = template_df.sort_values(by = ["EventId"])
    template_df.to_csv(out_file, index = None)
    return len(sorted_clusters)
