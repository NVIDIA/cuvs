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
from sklearn.preprocessing import normalize

from milvus_helpers import generate_additional_fields
from common import read_fvecs
from config import (FILE_TYPE, BASE_FILE_PATH, IS_UINT8, IF_NORMALIZE, TOTAL_VECTOR_COUNT, IMPORT_CHUNK_SIZE, \
                    PQ_M, N_TREE, RED, RESET, SHARDS_NUM, VECTOR_FIELD, BULK_FILE_TYPE, OBJECT_STORAGE_ACCESS_KEY,
                    OBJECT_STORAGE_SECRET_KEY, MILVUS_PORT, OBJECT_STORAGE_PORT, UPLOAD_CHUNK_SIZE_MB, MEMORY_USAGE_THRESHOLD)
from logs import LOGGER
from pymilvus.bulk_writer import RemoteBulkWriter, BulkFileType
import multiprocessing
from pymilvus import DataType, FieldSchema, CollectionSchema
from tqdm import tqdm
import urllib3
from urllib3.util import Timeout  # For fine-grained control
from pathlib import Path



def load_csv_data(filename):
    # filename = BASE_FILE_PATH + "/" + filename
    data = pd.read_csv(filename, header=None)
    data = np.array(data)
    if IS_UINT8:
        data = (data + 0.5) / 255
    if IF_NORMALIZE:
        data = normalize(data)
    data = data.tolist()
    return data


def csv_to_milvus(collection_name, client, shards_num=SHARDS_NUM):
    filenames = os.listdir(BASE_FILE_PATH)
    filenames.sort()
    total_insert_time = 0
    collection_rows = 0
    for filename in filenames:
        fname = os.path.join(BASE_FILE_PATH, filename)
        vectors = load_csv_data(fname)
        vectors_ids = list(id for id in range(collection_rows, collection_rows + len(vectors)))
        time_add_start = time.time()
        ids = client.insert(collection_name, vectors, vectors_ids, shards_num)
        total_insert_time = total_insert_time + time.time() - time_add_start
        collection_rows = collection_rows + len(ids)
        print(filename, " insert time: ", time.time() - time_add_start)
    print("total insert time: ", total_insert_time)


def load_fvecs_data(base_len, idx, fname, start_index=TOTAL_VECTOR_COUNT):
    if start_index == TOTAL_VECTOR_COUNT: # not provided
        begin_num = base_len * idx
    else:
        begin_num = start_index
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    data = x.view('float32').reshape(-1, d+1)[begin_num:(begin_num + base_len), 1:]
    if IF_NORMALIZE:
        data = normalize(data)
    data = data.tolist()
    return data


def bin_file_to_milvus(file_type, collection_name, client, base_file_name, total_vectors_count=TOTAL_VECTOR_COUNT,
                       import_chunk_size=IMPORT_CHUNK_SIZE, additional_columns_files=None):
    count = 0
    total_insert_time = 0
    fbin_base_file = None
    dimension = 0

    if file_type == 'fbin':
        fbin_base_file = open(base_file_name, 'rb')
        dimension = get_dim_from_vectors_file(base_file_name, 'fbin')
    mapped_arrays = []
    column_names = []
    columns_npy_chunks = {}
    start_index = 0
    columns_files = None
    if additional_columns_files is not None:
        columns_files = additional_columns_files.split(",")
    if columns_files is not None:
        for ind, column_file in enumerate(columns_files):
            column_name = os.path.basename(column_file).split('.')[0]
            column_names.append(column_name)
            mapped_arrays.append(np.load(column_file, mmap_mode='r'))

    num_chunks = math.ceil(total_vectors_count / import_chunk_size)
    left_to_insert = total_vectors_count
    vectors = None

    while count < num_chunks:
        import_chunk_size = min(import_chunk_size, left_to_insert)

        if file_type == 'fbin':
            vectors = load_fbin_data(fbin_base_file, dimension, import_chunk_size, start_index)
        elif file_type == 'fvecs':
            vectors = load_fvecs_data(import_chunk_size, count, base_file_name)
        else:
            print(f"{RED}Error:{RESET} Only fbin or fvecs file format is supported")
            sys.exit(0)

        actual_chunk_size = min(len(vectors), import_chunk_size)
        if actual_chunk_size == 0:
            break
        if len(mapped_arrays) > 0:
            for i, mapped_array in enumerate(mapped_arrays):
                column_name = column_names[i]
                npy_chunk = load_npy_chunk_data(mapped_array, start_index, import_chunk_size)
                columns_npy_chunks[column_name] = npy_chunk
        vectors_ids = list(id for id in range(count * actual_chunk_size, (count + 1) * actual_chunk_size))
        # Prepare Data for Insertion
        data = [ vectors_ids, vectors ]
        if bool(columns_npy_chunks):
            for key, value in columns_npy_chunks.items():
                data.append(value)
        time_add_start = time.time()
        # Insert Data into Milvus
        client.insert_data(collection_name, data)
        delta_time = time.time() - time_add_start
        total_insert_time = total_insert_time + delta_time
        print(count * actual_chunk_size, (count + 1) * actual_chunk_size, 'time:', delta_time)
        count = count + 1
        start_index = start_index + actual_chunk_size
        left_to_insert -= actual_chunk_size
    print("total insert time: ", total_insert_time)


def load_npy_data(filename):
    data = np.load(filename)
    if IS_UINT8:
        data = (data + 0.5) / 255
    if IF_NORMALIZE:
        data = normalize(data)
    data = data.tolist()
    return data


def load_npy_chunk_data(mapped_array, start_index, chunk_size):
    end = start_index + chunk_size
    chunk = mapped_array[start_index:end]
    # print(f"Processing chunk from index {start_index} to {end - 1}")
    return chunk


def npy_to_milvus(collection_name, client, shards_num=SHARDS_NUM):
    filenames = os.listdir(BASE_FILE_PATH)
    filenames.sort()
    total_insert_time = 0
    collection_rows = 0
    for filename in filenames:
        vectors = load_npy_data(os.path.join(BASE_FILE_PATH, filename))
        vectors_ids =list(id for id in range(collection_rows, collection_rows + len(vectors)))
        #vectors_ids = [id for id in range(collection_rows, collection_rows + len(vectors))]
        time_add_start = time.time()
        ids = client.insert(collection_name, vectors, vectors_ids, shards_num)
        total_insert_time = total_insert_time + time.time() - time_add_start
        print(filename, "insert rows", len(ids), " insert milvus time: ", time.time() - time_add_start)
        collection_rows = collection_rows + len(ids)
    print("total insert time: ", total_insert_time)


def load_bvecs_data(base_len, idx, fname):
    begin_num = base_len * idx
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    data = x.reshape(-1, d + 4)[begin_num:(begin_num + base_len), 4:]
    data = (data + 0.5) / 255
    if IF_NORMALIZE:
        data = normalize(data)
    data = data.tolist()
    return data


def bvecs_to_milvus(collection_name, client, fname=BASE_FILE_PATH, shards_num=SHARDS_NUM):
    count = 0
    total_insert_time = 0
    collection_rows = 0
    while count < (TOTAL_VECTOR_COUNT // IMPORT_CHUNK_SIZE):
        vectors = load_bvecs_data(IMPORT_CHUNK_SIZE, count, fname)
        vectors_ids = list(id for id in range(collection_rows, collection_rows + len(vectors)))
        # vectors_ids = [id for id in range(collection_rows, collection_rows + len(vectors))]
        time_add_start = time.time()
        ids = client.insert(collection_name, vectors, vectors_ids, shards_num)
        print(count * IMPORT_CHUNK_SIZE, (count + 1) * IMPORT_CHUNK_SIZE, 'time:',
              time.time() - time_add_start)
        total_insert_time = total_insert_time + time.time() - time_add_start
        count = count + 1
        collection_rows = collection_rows + len(ids)
    print(f"total insert time: {total_insert_time}")


def insert_data(client, collection_name, base_file=BASE_FILE_PATH, total_vectors_count=TOTAL_VECTOR_COUNT,
                    import_chunk_size=IMPORT_CHUNK_SIZE, shards_num=SHARDS_NUM, additional_columns_files=None):
    base_file_extension = os.path.splitext(base_file)[1][1:]
    if base_file_extension == 'fbin':
        bin_file_to_milvus('fbin', collection_name, client, base_file, total_vectors_count=total_vectors_count,
                           import_chunk_size=import_chunk_size, additional_columns_files=additional_columns_files)
        return
    if FILE_TYPE[0] == 'npy':
        npy_to_milvus(collection_name, client, shards_num=shards_num)
    if FILE_TYPE[0] == 'csv':
        csv_to_milvus(collection_name, client, shards_num=shards_num)
    if FILE_TYPE[0] == 'bvecs':
        bvecs_to_milvus(collection_name, client, fname=base_file, shards_num=shards_num)
    if FILE_TYPE[0] == 'fvecs':
        bin_file_to_milvus('fvecs', collection_name, client, base_file, total_vectors_count=total_vectors_count,
                           import_chunk_size=import_chunk_size, additional_columns_files=additional_columns_files)


def search_vectors(client, collection_name, input_file, k, mt, probe):
    if FILE_TYPE[0] == 'fvecs':
        query_vectors = list(read_fvecs(input_file))
        search_params = {
            "metric_type": mt,
            "params": {"nprobe": probe},
        }
        time_search_start = time.time()
        res = client.search_vectors(collection_name, query_vectors, k, search_params)
        search_time = time.time() - time_search_start
        print("total search time: ", search_time)
        return res
    else:
        print(f"{RED}Error:{RESET} Only fvecs file format is supported")
        sys.exit(0)


def get_index_params(index_type, metric_type, hnsw_m, efconstr, max_degree, search_list_size, pq_code_budget_gb_ratio,
                     search_cache_budget_gb_ratio, disk_pq_code_budget_gb_ratio, nbits, sq_type, refine, refine_type, pq_m,
                     nlist=None, inline_pq=None, rearrange=False, num_entry_points=None, pq_cache_size=None):
    if index_type == 'FLAT':
        index_param = {"index_type": index_type}
    elif index_type == 'IVF_FLAT':
        params = {"nlist": nlist}
        index_param = {"index_type": index_type, "metric_type": metric_type, "params": params}
    elif index_type in ('DISKANN', 'AISAQ'):
        params = {}
        if max_degree is not None:
            params.update({'max_degree': max_degree})
        if search_list_size is not None:
            params.update({'search_list_size': search_list_size})
        if pq_code_budget_gb_ratio is not None:
            params.update({'pq_code_budget_gb_ratio': pq_code_budget_gb_ratio})
        if search_cache_budget_gb_ratio is not None:
            params.update({'search_cache_budget_gb_ratio': search_cache_budget_gb_ratio})
        if disk_pq_code_budget_gb_ratio is not None:
            params.update({'disk_pq_code_budget_gb_ratio': disk_pq_code_budget_gb_ratio})
        if index_type == 'AISAQ':
            if inline_pq is not None:
                params.update({'inline_pq': inline_pq})
            if rearrange is not None:
                params.update({'rearrange': rearrange})
            if num_entry_points is not None:
                params.update({'num_entry_points': num_entry_points})
            if pq_cache_size is not None:
                params.update({'pq_cache_size': pq_cache_size})
        index_param = {"index_type": index_type, "metric_type": metric_type, "params": params}
    elif index_type == 'HNSW':
        params = {"M": hnsw_m, "efConstruction": efconstr}
        index_param = {"index_type": index_type, "metric_type": metric_type, "params": params}
    elif index_type == 'ANNOY':
        params = {"n_trees": N_TREE}
        index_param = {"index_type": index_type, "metric_type": metric_type, "params": params}
    elif index_type == 'IVF_PQ':
        params = {"nlist": nlist, "m": PQ_M}
        index_param = {"index_type": index_type, "metric_type": metric_type, "params": params}
    elif index_type == 'HNSW_PQ':
        params = {"M": hnsw_m, "efConstruction": efconstr, "m": pq_m, "nbits": nbits}
        if refine:
            params.update({'refine': refine})
            params.update({'refine_type': refine_type})
        index_param = {"index_type": index_type, "metric_type": metric_type, "params": params}
    elif index_type == 'HNSW_SQ':
        params = {"M": hnsw_m, "efConstruction": efconstr, "refine_type": refine_type, "refine": refine,
                  "sq_type": sq_type}
        index_param = {"index_type": index_type, "metric_type": metric_type, "params": params}
    else:
        params = {"nlist": nlist}
        index_param = {"index_type": index_type, "metric_type": metric_type, "params": params}
    LOGGER.info(index_param)
    return index_param


def create_index(client, collection_name, index_type, metric_type, hnsw_m, efconstr,
                 max_degree, search_list_size, pq_code_budget_gb_ratio,
                 search_cache_budget_gb_ratio, disk_pq_code_budget_gb_ratio, nbits, sq_type, refine, refine_type, pq_m,
                 nlist, inline_pq, rearrange, num_entry_points, pq_cache_size):
    index_param = get_index_params(index_type, metric_type, hnsw_m, efconstr,
                                   max_degree, search_list_size, pq_code_budget_gb_ratio,
                                   search_cache_budget_gb_ratio, disk_pq_code_budget_gb_ratio, nbits, sq_type, refine,
                                   refine_type, pq_m, nlist, inline_pq, rearrange, num_entry_points, pq_cache_size)
    time1 = time.time()
    client.create_index(collection_name, index_param)
    LOGGER.info(f"create index total cost time: {time.time() - time1}")

def remove_keys(obj, exclude_keys):
    if isinstance(obj, dict):
        return {
            k: remove_keys(v, exclude_keys)
            for k, v in obj.items()
            if k not in exclude_keys
        }
    elif isinstance(obj, list):
        return [remove_keys(item, exclude_keys) for item in obj]
    else:
        return obj

def call_milvus_api(host, api, json_dict, port=MILVUS_PORT, exclude_keys=None):
    json_data = json.dumps(json_dict)
    headers = {
        'Content-Type': 'application/json',
    }
    post_uri = 'http://' + host + ':' +str(port) + api

    try:
        response = requests.post(post_uri, headers=headers, data=json_data)
        response.raise_for_status()
        json_response = response.json()
        if exclude_keys:
            json_response = remove_keys(json_response, exclude_keys)
        print("Response:")
        json_formatted_str = json.dumps(json_response, indent=2)
        print(json_formatted_str)
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')


def list_imports(host, collection_name, port=MILVUS_PORT):
    create_dict = {
        "collectionName": collection_name,
    }
    call_milvus_api(host, '/v2/vectordb/jobs/import/list', create_dict, port)


def get_import_progress(host, job_id, port=MILVUS_PORT, exclude_keys=None):
    get_progress_dict = {
        "jobId": str(job_id),
    }
    call_milvus_api(host, '/v2/vectordb/jobs/import/get_progress', get_progress_dict, port, exclude_keys)


def import_data(host, collection_name, upload_desc_file_name, port=MILVUS_PORT):
    files_to_import = []
    with open(upload_desc_file_name, 'r') as file:
        for line in file:
            files_to_import.append(line.strip())

    files_to_import_main_list = []
    for file_name in enumerate(files_to_import):
        file_path = '/' + file_name[1]
        one_file_list = [file_path]
        files_to_import_main_list.append(one_file_list)

    create_dict = {
        "files": files_to_import_main_list,
        "collectionName": collection_name,
    }
    call_milvus_api(host, '/v2/vectordb/jobs/import/create', create_dict, port)


def get_dim_from_vectors_file(base_file_name, base_file_extension):
    if base_file_extension == 'fbin':
        fbin_base_file = open(base_file_name, 'rb')
        fbin_attrs = np.fromfile(fbin_base_file, count=2, dtype=np.int32)
        dim = fbin_attrs[1]
    else: # fvecs file
        x = np.memmap(base_file_name, dtype='uint8', mode='r')
        dim = x[:4].view('int32')[0]
    assert dim > 0
    return dim


def upload_data(collection, base_file_name, num_vectors_provided, results_file_name, object_storage_ip, bucket_name,
                bulk_file_type, num_tasks=0, upload_chunk_size_mb=UPLOAD_CHUNK_SIZE_MB, file_start_index=0,
                object_storage_port=OBJECT_STORAGE_PORT, collection_start_index=0, additional_columns_files=None,
                additional_fields_types=None):

    base_file_extension = os.path.splitext(base_file_name)[1][1:]
    t0 = time.time()
    dim = get_dim_from_vectors_file(base_file_name, base_file_extension)
    mapped_arrays = []
    column_names = []
    columns_files = None
    if additional_columns_files is not None:
        columns_files = additional_columns_files.split(",")
    if columns_files is not None:
        for ind, column_file in enumerate(columns_files):
            column_name = os.path.basename(column_file).split('.')[0]
            column_names.append(column_name)
            mapped_arrays.append(np.load(column_file, mmap_mode='r'))

    num_vectors, total_tasks, fbin_base_file = create_total_tasks_list(num_vectors_provided, dim, base_file_extension,
                                                                       base_file_name, num_tasks, file_start_index,
                                                                       collection_start_index)
    print('num_vectors: {}, total_tasks: {}'.format(num_vectors, total_tasks))
    conn, schema = create_schema_and_connection_param(dim, object_storage_ip, bucket_name, object_storage_port,
                                                      column_names, additional_fields_types)
    load_and_upload(collection, conn, schema, num_vectors, base_file_name, results_file_name,
                    total_tasks, dim, fbin_base_file, bulk_file_type, upload_chunk_size_mb, column_names, mapped_arrays)
    if base_file_extension == 'fbin':
        fbin_base_file.close()
    t4 = time.time()
    print('Total Time ({:.2f}sec).'.format(t4 - t0))


def load_fbin_data(base_file, dimension, import_chunk_size, start_index):
    vector_size = dimension * 4
    file_offset = 8 + (int(start_index) * int(vector_size)) # 8 first bytes are fbin_attrs [num vectors, dimension]
    base_file.seek(file_offset)
    data = np.fromfile(base_file, count=import_chunk_size * dimension, dtype=np.float32)
    actual_num_vectors = int(len(data) / dimension)
    actual_chunk_size = min(actual_num_vectors, import_chunk_size)
    arr_base = np.reshape(data, [actual_chunk_size, dimension])
    block_to_write = np.float32(arr_base)
    return block_to_write


def load_data(r_lock, fbin_base_file, dimension, fname, import_chunk_size, start_index):

    r_lock.acquire()
    try:
        if fbin_base_file is not None:
            vectors = load_fbin_data(fbin_base_file, dimension, import_chunk_size, start_index)
        else:
            vectors = load_fvecs_data(import_chunk_size, None, fname, start_index)
    finally:
        r_lock.release()
    return vectors


def handle_bulk_vectors_task(r_lock, wr_lock, collection, fbin_base_file, chunk_max_num_vectors, dimension, task,
                             base_file_name, conn, schema, results_file,
                             progress_counter, writer_chunk_size_mb=UPLOAD_CHUNK_SIZE_MB, bulk_file_type=BulkFileType.PARQUET,
                             column_names = None, mapped_arrays = None):
    collection_start_index = task[3]
    task_num_vectors = task[2]
    start_index = task[1]
    num_uploaded = 0
    num_vectors_left = task_num_vectors
    remote_path = '/' + collection + '/'
    columns_npy_chunks = {}
    vectors = None

    writer = RemoteBulkWriter(
        schema=schema,
        remote_path=remote_path,
        connect_param=conn,
        file_type=bulk_file_type,
        chunk_size=writer_chunk_size_mb * 1024 * 1024,  # the default value sets the maximum file segment size to 1 GB
    )

    while num_uploaded < task_num_vectors:
        if num_vectors_left >= chunk_max_num_vectors:
            num_to_load = chunk_max_num_vectors
        else:
            num_to_load = num_vectors_left
        vectors = load_data(r_lock, fbin_base_file, dimension, base_file_name, num_to_load, start_index)
        actual_chunk_size = min(len(vectors), num_to_load)
        if actual_chunk_size == 0:
            return

        if mapped_arrays is not None and len(mapped_arrays) > 0:
            for i, mapped_array in enumerate(mapped_arrays):
                column_name = column_names[i]
                npy_chunk = load_npy_chunk_data(mapped_array, start_index, num_to_load)
                columns_npy_chunks[column_name] = npy_chunk

        end_index = collection_start_index + actual_chunk_size
        vectors_ids = list(id for id in range(collection_start_index, end_index))
        upload_vectors(writer, vectors_ids, vectors, progress_counter, columns_npy_chunks)
        collection_start_index += actual_chunk_size
        num_vectors_left -= actual_chunk_size
        num_uploaded += actual_chunk_size
        start_index += actual_chunk_size

    wr_lock.acquire()
    try:
        for batch in writer.batch_files:
            for file in batch:
                file_record = f"{file}\n"
                results_file.write(file_record)
        results_file.flush()
    finally:
        wr_lock.release()
    vectors=None


def upload_vectors(writer, vectors_ids, vectors, progress_counter, columns_npy_chunks):
    for idx, vec in enumerate(vectors):
        vectors_id = vectors_ids[idx]
        row = {"id": vectors_id, VECTOR_FIELD: vec, }
        if bool(columns_npy_chunks):
            for key, value in columns_npy_chunks.items():
                row[key] = value[idx]
        writer.append_row(row)
    writer.commit()
    progress_counter.value += len(vectors)

def get_free_memory_mb():
    meminfo = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, value = line.split(":", 1)
            meminfo[key] = int(value.strip().split()[0])  # kB
    return meminfo["MemAvailable"] / 1024  # MB

def load_and_upload(collection, conn, schema, num_vectors, base_file_name, results_file_name,
                    total_tasks, dimension, fbin_base_file=None, bulk_file_type=BULK_FILE_TYPE,
                    upload_chunk_size_mb=UPLOAD_CHUNK_SIZE_MB, column_names=None,
                    mapped_arrays = None):
    jobs = []
    logging.basicConfig(level=logging.ERROR)
    bulk_buffer_logger = logging.getLogger("pymilvus.bulk_buffer")
    bulk_buffer_logger.setLevel(logging.WARNING)
    bulk_writer_logger = logging.getLogger("pymilvus.bulk_writer")
    bulk_writer_logger.setLevel(logging.WARNING)
    local_bulk_writer_logger = logging.getLogger("pymilvus.local_bulk_writer")
    local_bulk_writer_logger.setLevel(logging.WARNING)
    remote_bulk_writer_logger = logging.getLogger("pymilvus.remote_bulk_writer")
    remote_bulk_writer_logger.setLevel(logging.WARNING)
    progress_counter = multiprocessing.Value('i', 0)
    vector_size = dimension * 4
    num_tasks = len(total_tasks)
    upload_chunk_size = upload_chunk_size_mb * 1024 * 1024
    chunk_max_num_vectors = int(upload_chunk_size / vector_size)
    print('chunk_max_num_vectors: {}'.format(chunk_max_num_vectors))
    free_memory_mb = get_free_memory_mb()
    print(f"Free memory: {free_memory_mb:.2f} MB")
    requested_memory = num_tasks * upload_chunk_size_mb * MEMORY_USAGE_THRESHOLD
    print(f"Estimated consume memory: {requested_memory} MB")
    max_expected_file_size_mb = free_memory_mb / (num_tasks * MEMORY_USAGE_THRESHOLD)
    if requested_memory >= free_memory_mb:
        print(f"Warning: Estimated consume memory too high. please use lower --max_file_size_mb, suggested size is {int(max_expected_file_size_mb)} MB")
        sys.exit(1)

    with open(results_file_name, 'a') as results_file:

        wr_lock = multiprocessing.Lock()
        r_lock = multiprocessing.Lock()
        with tqdm(total=num_vectors, desc="Uploading Vectors") as pbar:
            for task in total_tasks:
                p = multiprocessing.Process(target=handle_bulk_vectors_task, args=(r_lock, wr_lock, collection,
                                                                                   fbin_base_file,
                                                                                   chunk_max_num_vectors,
                                                                                   dimension, task,
                                                                                   base_file_name, conn, schema,
                                                                                   results_file,
                                                                                   progress_counter, upload_chunk_size_mb,
                                                                                   bulk_file_type,
                                                                                   column_names, mapped_arrays,))
                jobs.append(p)
                p.start()
            # Update the tqdm progress bar
            while any(p.is_alive() for p in jobs):
                pbar.n = progress_counter.value
                pbar.refresh()
                time.sleep(0.2)

            jobs = [job for job in jobs if job.is_alive()]
            for job in jobs:
                job.join()
            results_file.close()
    print('{} vectors uploaded.'.format(progress_counter.value))


def create_schema_and_connection_param(dim, object_storage_ip, bucket_name, object_storage_port=OBJECT_STORAGE_PORT,
                                       additional_fields=None, additional_fields_types=None):

    field1 = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True)
    field2 = FieldSchema(name=VECTOR_FIELD, dtype=DataType.FLOAT_VECTOR, dim=dim)
    schema_fields = [field1, field2]
    print('additional_fields: {}'.format(additional_fields))
    if additional_fields is not None and len(additional_fields) > 0:
        additional_schema_fields = generate_additional_fields(additional_fields, additional_fields_types, dim)
        schema_fields.extend(additional_schema_fields)

    schema = CollectionSchema(
        auto_id=False,
        enable_dynamic_field=False,
        fields=schema_fields
    )
    schema.verify()

    object_storage_endpoint = object_storage_ip + ':' + str(object_storage_port)

    # Custom HTTP client with timeout (key fix for 499 errors)
    http_timeout = Timeout(connect=600.0, read=600.0)  # 5 minutes for connect and read
    http_client = urllib3.PoolManager(
        timeout=http_timeout,
        maxsize=100,  # Connection pool size
        retries=urllib3.Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 499, 500, 502, 503, 504],  # Retry on these server errors
        ),
    )
    # Connections parameters to access the remote bucket
    conn = RemoteBulkWriter.S3ConnectParam(
        endpoint=object_storage_endpoint,  # the default Object Storage service started along with Milvus
        access_key=OBJECT_STORAGE_ACCESS_KEY,
        secret_key=OBJECT_STORAGE_SECRET_KEY,
        bucket_name=bucket_name,
        secure=False,
        http_client=http_client
    )

    return conn, schema


def create_total_tasks_list(num_vectors_provided, dim, base_file_extension, base_file_name, num_tasks=0, file_start_index=0,
                            collection_start_index=0):
    fbin_base_file = None

    if base_file_extension == 'fbin':
        fbin_base_file = open(base_file_name, 'rb')
        fbin_attrs = np.fromfile(fbin_base_file, count=2, dtype=np.int32)
        fbin_num_base_vectors, dim = fbin_attrs[0], fbin_attrs[1]
        if num_vectors_provided == 0: # num_vectors not provided
            num_vectors = fbin_num_base_vectors - file_start_index
        else:
            num_vectors = num_vectors_provided
    else: # fvecs file
        if num_vectors_provided == 0: # num_vectors not provided
            vector_size = 4 * dim
            record_size = vector_size + 4
            file_stats = os.stat(base_file_name)
            file_size = file_stats.st_size
            num_vectors_in_file = int(file_size / record_size)
            num_vectors = num_vectors_in_file - file_start_index
        else:
            num_vectors = num_vectors_provided
    num_tasks = multiprocessing.cpu_count() if num_tasks == 0 else num_tasks
    print('num_vectors: {}'.format(num_vectors))
    print('num_tasks: {}'.format(num_tasks))
    chunk_size = math.ceil(num_vectors / num_tasks)
    total_tasks = []
    num_vectors_left = num_vectors
    for task_id in range(num_tasks):
        if num_vectors_left >= chunk_size:
            task_num_vectors = chunk_size
        else:
            task_num_vectors = num_vectors_left
        task_vecs = (task_id, file_start_index, task_num_vectors, collection_start_index)
        file_start_index = file_start_index + task_num_vectors
        collection_start_index = collection_start_index + task_num_vectors
        num_vectors_left = num_vectors_left - task_num_vectors
        total_tasks.append(task_vecs)

    return num_vectors, total_tasks, fbin_base_file


def load_descriptor_lines(path):
    if not Path(path).exists():
        return set()
    with open(path, "r") as f:
        return {line.strip() for line in f if line.strip()}

def append_to_descriptor(path, new_keys):
    # append-only is safe here since keys are immutable
    with open(path, "a") as f:
        for k in sorted(new_keys):
            f.write(k + "\n")
        f.flush()
        os.fsync(f.fileno())

def complete_upload(
    bulk_writer_dir,
    descriptor_path,
    collection_name,
    minio_client,
    bucket,
):
    uploaded_keys = load_descriptor_lines(descriptor_path)
    newly_uploaded = []
    bulk_writer_dir = Path(bulk_writer_dir)

    for uuid_dir in bulk_writer_dir.iterdir():
        if not uuid_dir.is_dir():
            continue
        uuid = uuid_dir.name
        for parquet in uuid_dir.glob("*.parquet"):
            object_key = f"{collection_name}/{uuid}/{parquet.name}"
            if object_key in uploaded_keys:
                continue
            try:
                size = parquet.stat().st_size
                print(
                    f"[UPLOAD][START] {object_key} "
                    f"size={size}"
                )
                minio_client.fput_object(
                    bucket,
                    object_key,
                    str(parquet),
                )
                print(f"[UPLOAD][DONE ] {object_key}")
            except Exception as e:
                print(f"[WARN] failed upload {object_key}: {e}")
                continue
            newly_uploaded.append(object_key)
            uploaded_keys.add(object_key)
            # optional: cleanup
            # parquet.unlink()
    if newly_uploaded:
        append_to_descriptor(descriptor_path, newly_uploaded)
    print(
        f"complete_upload: retried={len(newly_uploaded)} "
        f"total_uploaded={len(uploaded_keys)}"
    )
    return newly_uploaded

