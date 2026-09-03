import struct
import sys
import traceback
import re
import os

import numpy as np
import pandas as pd

from config import METRIC_TYPE, GROUNDTRUTH_FILE, RESET, CYAN, YELLOW, RED, QUERY_FILE, EF, REFINE_K
from logs import LOGGER

def load_gt_ids(gt_file=GROUNDTRUTH_FILE):
    gt_ids = []
    result = []
    if gt_file.endswith('csv'):
        with open(gt_file, 'r',encoding='utf-8') as f:
            for line in f.readlines():
                data = line.split()
                if data:
                    result.append(int(data[0]))
                else:
                    gt_ids.append(result)
                    result = []
    if gt_file.endswith('fbin'):
        with open(gt_file, 'rb') as f:
            num = struct.unpack('i', f.read(4))
            d = struct.unpack('i', f.read(4))
            data = np.fromfile(f, dtype='uint32')
            for i in range(num[0]):
                for j in range(d[0]):
                    val = data[i*d[0]+j]
                    result.append(val)
                gt_ids.append(result)
                result = []
    if gt_file.endswith('ivecs') or gt_file.endswith('ivec'):
        a = np.fromfile(gt_file, dtype='int32')
        d = a[0]
        gt_ids = a.reshape(-1, d + 1)[:, 1:].copy()
    if gt_file.endswith('parquet'):
        df = pd.read_parquet(gt_file)
        substring = 'neighbors_id'
        matching_columns = [col for col in df.columns if substring in col]
        df_matching = df[matching_columns]
        gt_lists = df_matching.values.tolist()
        for a_list in gt_lists:
            gt_ids.append(a_list[0])
    return gt_ids


def get_search_params(search_param, index_type, metric_type, refine_k=None, beamwidth=None, vectors_beamwidth=None, pq_read_page_cache_size=None):
    if index_type == 'FLAT':
        search_params = {"metric_type": metric_type}
    elif search_param == -1:
        print(f"{RED}Error:{RESET} The {CYAN}'--search_param'{RESET} option is mandatory for performance test")
        sys.exit(0)
    elif index_type == 'RNSG':
        search_params = {"metric_type": metric_type, "params": {'search_length': search_param}}
    elif index_type == 'HNSW':
        search_params = {"metric_type": metric_type, "params": {'ef': search_param}}
    elif index_type == 'HNSW_PQ' or index_type == 'HNSW_SQ':
        if refine_k is not None and refine_k > 0:
            params = {'ef': search_param, "override_faiss_search": False, "refine_k": refine_k}
        else:
            params = {'ef': search_param, "override_faiss_search": False}
        search_params = {"metric_type": metric_type, "params": params}
    elif index_type == 'ANNOY':
        search_params = {"metric_type": metric_type, "params": {"search_k": search_param}}
    elif index_type in ('DISKANN', 'AISAQ'):
        params = {}
        if search_param is not None:
            params.update({'search_list': search_param})
        if beamwidth is not None:
            params.update({'beamwidth': beamwidth})
        if index_type == 'AISAQ':
            if pq_read_page_cache_size is not None:
                params.update({'pq_read_page_cache_size': str(pq_read_page_cache_size)})
            if vectors_beamwidth is not None:
                params.update({'vectors_beamwidth': vectors_beamwidth})
        search_params = {"metric_type": metric_type, "params": params}
    else:
        search_params = {"metric_type": metric_type, "params": {"nprobe": search_param}}
    print(search_params)
    return search_params


def get_nq_vec(num_queries, file_path=QUERY_FILE):
    if file_path.endswith('npy'):
        data = np.load(file_path)
        if len(data) > num_queries and num_queries > 0:
            return data[0:num_queries].tolist()
        else:
            if num_queries > 0:
                LOGGER.info(f'There is only {len(data)} vectors')
            return data.tolist()
    elif file_path.endswith('fvec') or file_path.endswith('fvecs'):
            data = np.memmap(file_path, dtype='uint8', mode='r')
            d = data[:4].view('int32')[0]
            if num_queries > 0:
                data = data.view('float32').reshape(-1, d + 1)[0:num_queries, 1:]
            else:
                data = data.view('float32').reshape(-1, d + 1)[0:, 1:]
            return data.tolist()
    elif file_path.endswith(("fbin", "fbins")):
        # FBIN layout (common): int32 n, int32 d, then n*d float32 values
        header = np.memmap(file_path, dtype="int32", mode="r", shape=(2,))
        n, d = int(header[0]), int(header[1])

        # Map only the vector payload (skip 8-byte header)
        payload = np.memmap(file_path, dtype="float32", mode="r", offset=8)

        # Defensive: avoid reshape errors if file is shorter than header claims
        expected = n * d
        if payload.size < expected:
            raise ValueError(
                f"FBIN payload too small for header: header expects {n}x{d}={expected} "
                f"float32 values, file has {payload.size}."
            )

        mat = payload[:expected].reshape(n, d)
        if num_queries > 0:
            mat = mat[:min(num_queries, n)]
        return mat.tolist()
    else:
        print(f'Error: Bootcump failed to process file {file_path}. '
              f'Files of this format are not supported in this version. Make sure to use fvecs or npy files')
        exit(0)


def read_fvecs(file_path):
    with open(file_path, 'rb') as f:
        while True:
            dim = np.fromfile(f, dtype=np.int32, count=1)
            if not dim.size > 0:
                break
            vector = np.fromfile(f, dtype=np.float32, count=dim[0])
            yield vector.tolist()


def get_collection_dim(collection):
    schema = collection.schema
    for field in schema.fields:
        if 'dim' in field.params:
        # if field.dtype == "FLOAT_VECTOR" or field.dtype == "BINARY_VECTOR":
            return field.params['dim']
    return None


def validate_collection_exists(client, collection_name):
    if client.has_collection(collection_name):
        return True
    else:
        print(f"{YELLOW}Warning: {RESET}Collection {CYAN}{collection_name}{RESET} does not exist")
        return False


def get_index_metric_type(collection):
    try:
        # Get the index information for the specified field
        indexes = collection.indexes
        for idx in indexes:
            return idx.params.get('metric_type')
        print("The collection is not indexed")
    except Exception as e:
        print(f"An error occurred in get_index_metric_type: {e}")
        traceback.print_exc()
        sys.exit(-1)


def print_version():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, 'version.py')
    try:
        with open(filename, 'r') as file:
            line = file.readline().strip()
            match = re.match(r'^__version__\s*=\s*([0-9.]+)$', line)
            if match:
                version_number = match.group(1)
                if all(char.isdigit() or char == '.' for char in version_number):
                    print(version_number)
                else:
                    print("Not a version")
            else:
                print(f"{RED}Error: {YELLOW}Bad version file format{RESET}")
    except FileNotFoundError:
        print(f"{RED}Error: {YELLOW}Version file not found{RESET}")
    except Exception as e:
        print(f"An error occurred: {e}")
