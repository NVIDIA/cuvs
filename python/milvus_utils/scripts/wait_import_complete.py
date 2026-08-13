import math
import os
import sys
import time
import logging
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import json
import requests
from pymilvus.bulk_writer import RemoteBulkWriter, BulkFileType
import multiprocessing
from pymilvus import DataType, FieldSchema, CollectionSchema
from tqdm import tqdm

MILVUS_PORT = 19530
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'


def call_milvus_api(host, api, json_dict, port=MILVUS_PORT, print_response=True):
    json_data = json.dumps(json_dict)
    headers = {
        'Content-Type': 'application/json',
    }
    post_uri = 'http://' + host + ':' +str(port) + api

    try:
        response = requests.post(post_uri, headers=headers, data=json_data)
        response.raise_for_status()
        json_response = response.json()
        if print_response:
            print("Response:")
            json_formatted_str = json.dumps(json_response, indent=2)
            print(json_formatted_str)
        return json_response
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')

def wait_import_complete(host, job_id, port=MILVUS_PORT):
    get_progress_dict = {
        "jobId": str(job_id),
    }
    start_time = time.time()
    response_json = call_milvus_api(host, '/v2/vectordb/jobs/import/get_progress', get_progress_dict, port, print_response=False)
    if response_json['code'] != 0:
        print(f"{RED}Error:{RESET} Failed to get import progress")
        json_formatted_str = json.dumps(response_json, indent=2)
        print(json_formatted_str)
        return 2
    if response_json['data']['totalRows'] == 0:
        print("import not available")
        return 2
    with tqdm(total=int(response_json['data']['totalRows']), desc="Importing vectors") as pbar:
        while True:
            response_json = call_milvus_api(host, '/v2/vectordb/jobs/import/get_progress', get_progress_dict, port, print_response=False)
            if response_json['code'] != 0:
                print(f"{RED}Error:{RESET} Failed to get import progress")
                json_formatted_str = json.dumps(response_json, indent=2)
                print(json_formatted_str)
                return 1
            pbar.n = int(response_json['data']['importedRows'])
            pbar.refresh()
            if int(response_json['data']['importedRows']) == int(response_json['data']['totalRows']):
                break
            time.sleep(5)
    with tqdm(total=100, desc="Completing Import") as pbar:
        while True:
            response_json = call_milvus_api(host, '/v2/vectordb/jobs/import/get_progress', get_progress_dict, port, print_response=False)
            if response_json['code'] != 0:
                print(f"{RED}Error:{RESET} Failed to get import progress")
                json_formatted_str = json.dumps(response_json, indent=2)
                print(json_formatted_str)
                return 1
            pbar.n = int(response_json['data']['progress'])
            pbar.refresh()
            if int(response_json['data']['progress']) == 100:
                break
            time.sleep(5)
    print(f"Import completed in {time.time() - start_time} seconds")
    return 0

if __name__ == "__main__":
    if os.environ.get('MILVUS_IP') is None:
        print(f"{RED}Error:{RESET} MILVUS_IP is not set")
        exit(1)
    if os.environ.get('IMPORT_JOBID') is None:
        print(f"{RED}Error:{RESET} IMPORT_JOBID is not set")
        exit(1)
    exit_code = wait_import_complete(os.environ['MILVUS_IP'], os.environ['IMPORT_JOBID'], os.environ.get('MILVUS_PORT', MILVUS_PORT))
    exit(exit_code)
