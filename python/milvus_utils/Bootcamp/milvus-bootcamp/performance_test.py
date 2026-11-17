import os
import random
import time
from datetime import datetime

import numpy as np

from common import load_gt_ids, get_search_params, get_nq_vec
from config import PERFORMANCE_RESULTS_PATH, NQ_SCOPE, TOPK_SCOPE, PERCENTILE_NUM, RED, RESET, CYAN, MAGENTA
from logs import LOGGER
from recall_test import compute_recall


def performance(client, collection_name, search_param, query_file_path, gt_file, metric_type,
                result_path, nq_scope, topk_scope, refine_k, beamwidth, vectors_beamwidth, pq_read_page_cache_size):
    gt_ids = load_gt_ids(gt_file)
    index_type = client.get_index_params(collection_name)
    if index_type:
        index_type = index_type[0]['index_type']
    else:
        index_type = 'FLAT'
    search_params = get_search_params(search_param, index_type, metric_type, refine_k, beamwidth, vectors_beamwidth, pq_read_page_cache_size)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    result_filename = collection_name + '_' + str(search_param) + '_performance.csv'
    performance_result_file = os.path.join(result_path, result_filename)

    with open(performance_result_file, 'w+', encoding='utf-8') as f:
        f.write("nq,topk,total_time,avg_time" + '\n')
        for nq in nq_scope:
            print("\n")
            data = get_nq_vec(-1, query_file_path)
            if len(data) < nq:
                print(f"{RED}Error:{RESET}The {CYAN}nq{RESET} value is {MAGENTA}{nq}{RESET}, "
                      f"but number of queries in the file is {MAGENTA}{len(data)}{RESET}")
                exit(0)
            seed = datetime.now().timestamp()
            random.seed(seed)
            rand = sorted(random.sample(range(0, len(data)), nq))
            query_list = []
            for i in rand:
                query_list.append(data[i])

            LOGGER.info(f"begin to search, nq = {len(query_list)}")
            for topk in topk_scope:
                time_start = time.time()
                results = client.search_vectors(collection_name, query_list, topk, search_params)
                time_cost = time.time() - time_start
                print(nq, topk, time_cost)
                line = str(nq) + ',' + str(topk) + ',' + str(round(time_cost, 4)) + ',' + str(
                    round(time_cost / nq, 4)) + '\n'
                f.write(line)
                compute_recall(collection_name, nq, results, search_param, rand, gt_ids, [topk])
            f.write('\n')
    LOGGER.info("search_vec_list done !")


def percentile_test(client, collection_name, search_param, percentile):
    index_type = client.get_index_params(collection_name)
    if index_type:
        index_type = index_type[0]['index_type']
    else:
        index_type = 'FLAT'
    search_params = get_search_params(search_param, index_type, None)

    if not os.path.exists(PERFORMANCE_RESULTS_PATH):
        os.mkdir(PERFORMANCE_RESULTS_PATH)

    result_filename = collection_name + '_' + str(search_param) + '_percentile.csv'
    performance_file = os.path.join(PERFORMANCE_RESULTS_PATH, result_filename)

    with open(performance_file, 'w+', encoding='utf-8') as f:
        f.write("nq,topk,total_time" + '\n')
        for nq in NQ_SCOPE:
            query_list = get_nq_vec(nq)
            LOGGER.info(f"begin to search, nq = {len(query_list)}")
            for topk in TOPK_SCOPE:
                time_cost = []
                for _ in range(PERCENTILE_NUM):
                    time_start = time.time()
                    client.search_vectors(collection_name, query_list, topk, search_params)
                    time_cost.append(time.time() - time_start)
                time_cost = np.array(time_cost)
                time_cost = np.percentile(time_cost, float(percentile))
                print(nq, topk, round(time_cost, 4))
                line = str(nq) + ',' + str(topk) + ',' + str(round(time_cost, 4)) + '\n'
                f.write(line)
            f.write('\n')
    LOGGER.info("search_vec_list done !")
