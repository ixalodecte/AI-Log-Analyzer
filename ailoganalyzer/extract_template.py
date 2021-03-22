import json
import logging
import os
import subprocess
import sys
import time
import fnmatch

from drain3 import TemplateMiner



def log2template(in_log_file,log_structure, out_file):
    logger = logging.getLogger(__name__)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')



    template_miner = TemplateMiner()

    line_count = 0
    start_time = time.time()
    batch_start_time = start_time
    batch_size = 10000
    with open(in_log_file) as f:
        for line in f:
            line = line.rstrip()
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
                result_json = json.dumps(result)
                logger.info(f"Input ({line_count}): " + line)
                logger.info("Result: " + result_json)

    time_took = time.time() - start_time
    rate = line_count / time_took
    logger.info(f"--- Done processing file. Total of {line_count} lines, rate {rate:.1f} lines/sec, "
                f"{len(template_miner.drain.clusters)} clusters")
    sorted_clusters = sorted(template_miner.drain.clusters, key=lambda it: it.size, reverse=True)

    with open(out_file, 'a') as the_file:
        the_file.write('EventId,EventTemplate\n')
        for cluster in sorted_clusters:
            print(str(cluster.cluster_id) + "," + '"' + cluster.get_template() + '"')
            the_file.write('"'+str(cluster.cluster_id)+ '",' + '"' + cluster.get_template() + '"' + "\n")
    return len(sorted_clusters)

if __name__ == '__main__':
    in_log_file = "../data/bgl2_100k"
    out_file = "../data/preprocess/templates.csv"
    log_structure = {
        "separator" : ' ',          # separateur entre les champs d'une ligne
        "time_index" : 4,           # index timestamp
        "message_start_index" : 9,  # debut message
        "message_end_index" : None, # fin message
        "label_index" : 0           # index label (None si aucun)
    }
    num=log2template(in_log_file, log_structure, out_file)
    print(num)
