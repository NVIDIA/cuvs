import argparse
import os
import struct
import sys
import numpy
import time

from pymilvus.bulk_writer import BulkFileType

from minio_client import list_buckets, list_objects, disk_usage, remove_objects, create_object_storage_client
from common import validate_collection_exists, print_version
from config import MILVUS_PORT, METRIC_TYPE, PROBE, K, \
    MAGENTA, RESET, CYAN, HNSW_M, EFCONSTRUCTION, RED, PERFORMANCE_RESULTS_PATH, NQ_SCOPE, \
    TOTAL_VECTOR_COUNT, IMPORT_CHUNK_SIZE, TOPK_SCOPE, EF, DETAILS, \
    NBITS, REFINE, REFINE_K, PQ_M, SHARDS_NUM, INDEX_TYPE, BULK_FILE_TYPE, \
    NOT_CLUSTERING_KEY, UPLOAD_CHUNK_SIZE_MB, BULK_WRITER_DIR
from load import insert_data, create_index, search_vectors, upload_data, import_data, get_import_progress, list_imports, complete_upload
from milvus_helpers import MilvusHelper, additional_types
from performance_test import performance


def valid_base_file(file_path):
    if not file_path:
        return None
    if not os.path.isfile(file_path):
        raise argparse.ArgumentTypeError(f"The file '{file_path}' does not exist.\n\n")
    if not (file_path.endswith('.fvecs') or file_path.endswith('.bvecs') or file_path.endswith('.fbin')):
        raise argparse.ArgumentTypeError(f"The file '{file_path}' must have a .fvecs or .bvecs extension.\n\n")
    return file_path


def valid_int_array(arg):
    if isinstance(arg, str):
        items = arg.split(',')
        res = [int(item) for item in items]
        return res
    return input


def valid_gt_file(file_path):
    if not file_path:
        return None
    if not os.path.isfile(file_path):
        raise argparse.ArgumentTypeError(f"The file '{file_path}' does not exist.\n\n")
    if not (file_path.endswith('.csv') or file_path.endswith('.ivec') or file_path.endswith('.ivecs') or file_path.endswith('.fbin') or file_path.endswith('.parquet')):
        raise argparse.ArgumentTypeError(f"The file '{file_path}' must have either .csv, .fbin, .ivec, or .ivecs extension.\n\n")
    return file_path


def valid_qry_file(file_path):
    if not file_path:
        return None
    if not os.path.isfile(file_path):
        raise argparse.ArgumentTypeError(f"The file '{file_path}' does not exist.\n\n")
    if not file_path.endswith(('.fvecs', '.fbin')):
        raise argparse.ArgumentTypeError(f"The file '{file_path}' must have a .fvecs extension.\n\n")
    return file_path


def valid_result_path(path):
    if sys.argv[-1] != 'performance' and sys.argv[-1] != 'P':
        return None
    if os.path.exists(path) and os.path.isdir(path):
        return path  # Pass if the path exists and is a directory
    
    parent_dir = os.path.dirname(path)
    if len(parent_dir) == 0:
        parent_dir = '.'
    
    if os.path.exists(parent_dir) and os.path.isdir(parent_dir):
        # If the parent directory exists, create the missing directory
        try:
            os.makedirs(path)
            return path
        except OSError as e:
            raise argparse.ArgumentTypeError(f"Failed to create directory '{path}': {str(e)}\n\n")
    else:
        raise argparse.ArgumentTypeError(f"The path '{parent_dir}' does not exist, unable to create '{path}'.\n\n")


def count_rows_in_npy_file(file_path):
    """Count the number of rows in a .npy file."""
    data = numpy.load(file_path, mmap_mode='r')
    # Check the shape of the array
    if len(data.shape) == 1:
        return data.shape[0]  # Return the number of rows
    else:
        print("The data is not a 1D array.")
        return None

def get_count_rows_and_dim_from_fvecs_file(fvecs_file):
    with open(fvecs_file, 'rb') as binary_fvecs_file:
        vector_dim = struct.unpack('i', binary_fvecs_file.read(4))[0]
    file_size = os.stat(fvecs_file)
    one_row_size = 4 + (4 * vector_dim)
    num_vectors = int(file_size.st_size / one_row_size)
    print('file_size: {} one_row_size: {}'.format(file_size, one_row_size))
    return num_vectors, vector_dim

def get_count_rows_and_dim_from_fbin_file(file_path):
    fbin_base_file = open(file_path, 'rb')
    fbin_attrs = numpy.fromfile(fbin_base_file, count=2, dtype=numpy.int32)
    fbin_num_base_vectors, dim = fbin_attrs[0], fbin_attrs[1]

    return fbin_num_base_vectors, dim

def get_count_rows_and_dim_from_bin_file(file_type, bin_base_file):
    if file_type == 'fbin':
        base_num_records, dim = get_count_rows_and_dim_from_fbin_file(bin_base_file)
    elif file_type == 'fvecs':
        base_num_records, dim = get_count_rows_and_dim_from_fvecs_file(bin_base_file)
    else:
        print(f"{RED}Error:{RESET} Only fbin or fvecs file format is supported")
        sys.exit(0)
    return base_num_records, dim

def validate_additional_columns_files_for_upload(file_type, additional_columns_files, bin_base_file):
    base_num_records, _ = get_count_rows_and_dim_from_bin_file(file_type, bin_base_file)
    print('fbin_base_file {} has {} records'.format(bin_base_file, base_num_records))
    additional_columns_files_list = additional_columns_files.split(",")
    for column_file in additional_columns_files_list:
        column_file_num_records = count_rows_in_npy_file(column_file)
        print('column_file {} has {} records'.format(column_file, column_file_num_records))
        if column_file_num_records != base_num_records:
            raise argparse.ArgumentTypeError(f"{column_file} has {column_file_num_records} which is different from "
                                             f"{bin_base_file} {base_num_records} number of records.\n\n")

def validate_additional_columns_files(file_type, client, collection_name, additional_columns_files, bin_base_file):
    base_num_records, _ = get_count_rows_and_dim_from_bin_file(file_type, bin_base_file)
    print('bin_base_file {} has {} records'.format(bin_base_file, base_num_records))
    collections_fields = client.get_collection_fields(collection_name)
    additional_columns_files_list = additional_columns_files.split(",")
    for column_file in additional_columns_files_list:
        column_field_name = os.path.basename(column_file).split('.')[0]
        if column_field_name not in collections_fields:
            raise argparse.ArgumentTypeError(f"There is no such column {column_field_name} in {collection_name} "
                                             f"collection schema.\n\n")
        column_file_num_records = count_rows_in_npy_file(column_file)
        print('column_file {} has {} records'.format(column_file, column_file_num_records))
        if column_file_num_records != base_num_records:
            raise argparse.ArgumentTypeError(f"{column_file} has {column_file_num_records} which is different from "
                                             f"{bin_base_file} {base_num_records} number of records.\n\n")


def validate_additional_fields(additional_fields, additional_fields_types):
    if additional_fields is None and additional_fields_types is None:
        return
    if additional_fields is None or additional_fields_types is None:
        missing_arg = "additional_fields" if additional_fields is None else "additional_fields_types"
        raise argparse.ArgumentTypeError(f"The {missing_arg} argument is required for additional_fields_types.\n\n")
    if not validate_additional_fields_types(additional_fields_types.split(",")):
        raise argparse.ArgumentTypeError(f"Invalid additional_fields_types!\n\n")
    if len(additional_fields.split(",")) != len(additional_fields_types.split(",")):
        raise argparse.ArgumentTypeError(f"The number of additional_fields and additional_fields_types "
                                         f"must be the same\n\n")


def validate_additional_fields_types(additional_fields_types):

    for field_type in additional_fields_types:
        if field_type in ['int', 'vector']:
            continue

        valid_field = False
        for base_type, valid_range in additional_types.items():
            if field_type.startswith(base_type):
                valid_field = True
                size = field_type[len(base_type):]
                if not size.isdigit():
                    raise argparse.ArgumentTypeError(f"Invalid field type: {field_type}. Size must be a valid integer.\n")
                if int(size) not in valid_range:
                    raise argparse.ArgumentTypeError(f"Invalid varchar field size: {size}.")
                break
        if not valid_field:
            raise argparse.ArgumentTypeError(f"Invalid field type: {field_type}.\n")
    return True


def default_search_param():
    return EF


def parse_arguments():
    parser = argparse.ArgumentParser(description="Bootcamp CLI. For more detail refer: "
                                                 "https://dssd-bitbucket.us.kioxia.com/projects/TES/repos/bootcamp"
                                                 "/browse/benchmark_test/scripts/README.md",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('command', choices=[
             "s", "load_progress",
             "S", "search",
             "c", "create",
             "a", "insert",
             "u", "upload",
             "im", "import",
             "imp", "import_progress",
             "lim", "list_imports",
             "lib", "mc_list_buckets",
             "lio", "mc_list_objects",
             "du", "mc_disk_usage",
             "ro", "mc_remove_objects",
             "P", "performance",
             "I", "create_index",
             "i", "index_info",
             "p", "index_progress",
             "h", "has",
             "R", "release",
             "D", "drop",
             "L", "load",
             "l", "list",
             "r", "rows",
             "d", "desc",
             "C", "compact",
             "O", "drop_index",
             "version",
             "n", "rename",
             "g", "segments",
             "cu", "complete_upload",
             ], help="Command (must be the last argument)\n\n")

    parser.add_argument('-a', '--host', type=str, help=f"Host name or IP where the Milvus server is "
                                                           f"located. {CYAN}Default: None{RESET}\n\n", default = None)
    parser.add_argument('-p', '--port', type=int, help=f"Port number the Milvus data node "
                                                       f"listens to. {CYAN}Default: {MILVUS_PORT}{RESET}\n\n", default=MILVUS_PORT)
    parser.add_argument('-c', '--collection', type=str, help="Name of the collection to be operated on\n\n",
                        default=None)

    parser.add_argument('--shards_num', type=int, help=f"Number of shards to split the collection. "
                                                       f"{CYAN}Default: {SHARDS_NUM}{RESET}\n\n", default=SHARDS_NUM)

    # Index related parameters
    parser.add_argument('-i', '--index_type', type=str, default="DISKANN", help=f"When creating an index, you need to specify the index type. {CYAN}Default: {INDEX_TYPE}{RESET}\n\n",
                        choices=['DISKANN', 'AISAQ', 'HNSW', 'FLAT', 'IVF_FLAT', 'HNSW_PQ', 'HNSW_SQ'])
    parser.add_argument('-m', '--metric_type', type=str, help=f"Index metric type (L2 for Euclidean Distance, IP for Inner Product). {CYAN}Default: {METRIC_TYPE}{RESET}\n\n", choices=['L2', 'IP'], default=METRIC_TYPE)
    parser.add_argument('-d', '--dim', type=int, help=f'Vector dimension. {CYAN}Default: None{RESET}\n\n', default=None)

    parser.add_argument('--hnsw_m', type=int, help=f'The max number of connections that each vector in the graph can have during the HNSW index construction phase. {CYAN}Default: {HNSW_M}{RESET}\n\n', default=HNSW_M)
    parser.add_argument('--nlist', type=int, help=f'The number of buckets during the IVF_FLAT/IVF_PQ index clustering construction phase. {CYAN}Default: None{RESET}\n\n', default=None)
    parser.add_argument('--efconstr', type=int, help=f'Size of the priority queue that is used to determine which vectors will be connected to a new vector being inserted into the graph of HNSW index. {CYAN}Default: {EFCONSTRUCTION}{RESET}\n\n', default=EFCONSTRUCTION)
    parser.add_argument('--max_degree', type=int, help=f'Max number of edges that each node in the DISKANN/AISAQ graph can have. {CYAN}Default: None{RESET}\n\n', default=None)
    parser.add_argument('--pq_code_budget_gb_ratio', type=float, help=f'Ratio of memory allocated to storing PQ (Product Quantization) codes relative to the total memory budget for the AISAQ index. {CYAN}Default: None{RESET}\n\n', default=None)
    parser.add_argument('--search_cache_budget_gb_ratio', type=float, help=f'Ratio of cached node numbers to raw data for the DISKANN/AISAQ index. {CYAN}Default: None{RESET}\n\n', default=None)
    parser.add_argument('--disk_pq_code_budget_gb_ratio', type=float, help=f'Size limit on the vector code for the DISKANN/AISAQ index. The 0 value means no compression. {CYAN}Default: None{RESET}\n\n', default=None)
    parser.add_argument('--pq_m', type=int, help=f'Number of subquantizers for the HNSW_PQ index. {CYAN}Default: {PQ_M}{RESET}\n\n', default=PQ_M)
    parser.add_argument('--nbits', type=int, help=f'Number of bits per subquantizer for the HNSW_PQ index. {CYAN}Default: {NBITS}{RESET}\n\n', default=NBITS)
    parser.add_argument('--sq_type', type=str, help=f'Scalar quantizer type the HNSW_SQ index {CYAN}Default: None{RESET}', default=None, choices=["sq6", "sq8", "fp16", "bf16"])
    parser.add_argument('--refine', type=bool, help=f'Determines whether the refine is used during the train for the HNSW_PQ and HNSW_SQ indexes. {CYAN}Default: {REFINE}{RESET}\n\n', default=REFINE, action=argparse.BooleanOptionalAction)
    parser.add_argument('--refine_type', type=str, help=f'The type of refine for the HNSW_PQ and HNSW_SQ indexes. {CYAN}Default: None{RESET}', default=None, choices=["sq6", "sq8", "fp16", "bf16", "fp32", "flat"])
    parser.add_argument('--search_list_size', type=int, help=f'Size of the candidate list used for the DISKANN/AISAQ index. {CYAN}Default: None{RESET}', default=None)

    parser.add_argument('-b', '--base', type=valid_base_file, help="Path to the fvecs, fbin or bvecs file. Mandatory for 'insert', 'upload' and 'update' operations\n\n")
    parser.add_argument('-t', '--total_vectors', type=int, help=f'Number of vectors to insert/upload from the base file. {CYAN}Default: {TOTAL_VECTOR_COUNT}{RESET}\n\n', default=TOTAL_VECTOR_COUNT)
    parser.add_argument('-s', '--chunk_size', type=int, help=f'When the data format is bvecs or fvecs, the amount of data written into milvus each time. Unless the gRPC is configured differently, chunk size must not exceed 256MB. To calculate chunk size in bytes, use the formula: chunk size * vector dimension * 4.  Default: {IMPORT_CHUNK_SIZE}{RESET}\n\n', default=IMPORT_CHUNK_SIZE)

    # Performance test related parameters
    parser.add_argument('-q', '--query_file', type=valid_qry_file, help="Path to the query fvecs or fbin file. "
                                                                        "Mandatory for 'search' and 'performance' operations\n\n")
    parser.add_argument('-g', '--gt_file', type=valid_gt_file, help="Path to the ground truth file. "
                                                                    "Mandatory for 'performance' operation\n\n")
    parser.add_argument('-r', '--result_path', type=valid_result_path, help=f'Path to the performance '
                                                                            f'test result directory. {CYAN}Default: '
                                                                            f'{PERFORMANCE_RESULTS_PATH}{RESET}\n\n', default=PERFORMANCE_RESULTS_PATH)

    parser.add_argument('--nq', type=valid_int_array, help=f'Number of queries to be tested. {CYAN}Default: {NQ_SCOPE}{RESET}\n\n', default=NQ_SCOPE)
    parser.add_argument('--search_param', type=int, help=f'When querying, specify the parameter value '
                                                         f'when querying (When the index is of type Ivf, '
                                                         f'this parameter refers to Nprobe. When indexing by Rnsg, '
                                                         f'this parameter refers to Search_Length. '
                                                         f'When the index is HNSW, this parameter refers to EF)\n\n')
    parser.add_argument('--topk', type=valid_int_array, help=f'The topk value to be tested in each np '
                                                            f'(Here means testing multiple topk values). {CYAN}Default: {TOPK_SCOPE}{RESET}\n\n', default=TOPK_SCOPE)
    parser.add_argument('-k', '--k', type=int, help=f'The topk value to be tested in each np '
                                                    f'(Here means testing multiple topk values). {CYAN}Default: {K}{RESET}\n\n', default=K)
  
    parser.add_argument('--probe', type=int, help=f'Determines how many clusters are visited during '
                                                  f'the search in IVF Flat index. {CYAN}Default: {PROBE}{RESET}\n\n', default=PROBE)
    parser.add_argument('--refine_k', type=float, help=f'Refine factor (0 value leads to a search without a refine) '
                                                     f'search in FAISS HNSW PQ/SQ indexes. {CYAN}Default: {REFINE_K}{RESET}\n\n',
                        default=REFINE_K)

    parser.add_argument('--bucket_name', type=str, help=f'Object Storage bucket name to upload the file vectors to. {CYAN}Default: None{RESET}',
                        default=None)

    parser.add_argument('--upload_desc_file', type=str, help=f'File (full path) for the upload output list of parquet/json files to. {CYAN}Default: None{RESET}', default=None)

    parser.add_argument('--additional_fields', type=str,
                        help=f'List of additional fields. {CYAN}Default: None{RESET}',
                        default=None)

    parser.add_argument('--additional_types', type=str,
                        help=f'List of additional fields types correlated to the List of additional fields. {CYAN}Default: None{RESET}',
                        default=None)

    parser.add_argument('--additional_columns_files', type=str,
                        help=f'List of additional column npy files correlated to the List of additional fields. {CYAN}Default: None{RESET}',
                        default=None)

    parser.add_argument('--bulk_file_type', type=str, help=f'Parquet (default) / Json {CYAN}Default: {BULK_FILE_TYPE}{RESET}', default=BULK_FILE_TYPE, choices=["parquet", "json"])

    parser.add_argument('--file_start_index', type=int, help=f'Start index on the base vectors file to upload from. {CYAN}Default: 0{RESET}\n\n', default=0)

    parser.add_argument('--collection_start_index', type=int,
                        help=f'Collection index for first vector to upload. {CYAN}Default: 0{RESET}\n\n',
                        default=0)
    parser.add_argument('--job_id', type=int, help=f'The ID of the bulk-import job of your interest. {CYAN}Default: None{RESET}', default=None)

    parser.add_argument('--prefix', type=str, help=f'Object name starts with prefix. {CYAN}Default: None{RESET}', default=None)
    parser.add_argument('--dry_run', action="store_true", help=f'Outputs the results of a command without actually removing any files.')
    # parser.add_argument('-', '--', type=..., help=f'. {CYAN}Default: {}', default=...)
    parser.add_argument('-w', '--beamwidth', type=int, help="Beam width of the indexes (only for DISKANN family index types)\n\n.",
                        default=None)
    parser.add_argument('-v', '--vectors_beamwidth', type=int, help="Beam width of the compressed vectors (only for AISAQ index type)\n\n",
                        default=None)
    parser.add_argument('--inline_pq', type=int,
                        help="Set the number of pq vectors to be stored inline as part of the index node(only for AISAQ index type)\n\n",
                        default=None)
    parser.add_argument('--num_entry_points', type=int, required=False,
                        help="Number of entry points (1-1000) that should be generated to be used as a search start points (only for AISAQ index type)\n\n",
                        default=None)
    parser.add_argument( '--rearrange', type=str,
                        help="Enable vectors rearrangement during build (only for AISAQ index type)\n\n",
                        default=None, choices=["True", "False"])
    parser.add_argument('--pq_cache_size', type=int,
                        help="PQ vectors cache DRAM size in bytes (only for AISAQ index type)\n\n",
                        default=None)
    parser.add_argument('--pq_read_page_cache_size', type=int,
                        help="PQ vectors read page cache DRAM size in bytes - per thread (only for AISAQ index type)\n\n",
                        default=None)
    parser.add_argument('--not_clustering_key', type=bool,
                        help=f'Determines whether the collection vectors field is not clustering key in Milvus open source contribution collections. {CYAN}Default: {NOT_CLUSTERING_KEY}{RESET}\n\n',
                        default=NOT_CLUSTERING_KEY, action=argparse.BooleanOptionalAction)
    parser.add_argument('--new_name', type=str, help='The new name for the collection', default=None)
    parser.add_argument('--sum_by', type=str, help='Sum segment info by node or by collection', default=None)
    parser.add_argument('--exclude', type=str,
                        help='Comma separated list of fields to exclude from collection loading', default=None)
    parser.add_argument('--num_tasks', type=int, help="Number of parallel tasks for upload\n\n", default=0)
    parser.add_argument('--max_file_size_mb', type=int, choices=range(64, 5120), metavar='[64-5120]', help="The maximum size of each uploaded file in MB\n\n", default=UPLOAD_CHUNK_SIZE_MB)
    parser.add_argument('--details', type=bool,
                        help=f'Use this flag if you would like to get a more detailed output. {CYAN}Default: {DETAILS}{RESET}\n\n',
                        default=DETAILS, action=argparse.BooleanOptionalAction)

    # Parse the arguments
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    if not args.host and args.command != "version":
        print(f"{RED}Error:{RESET} The {CYAN}--host{RESET} parameter is mandatory")
        exit(1)
    # create collection
    if args.command in ("c", "create"):
        if not args.collection:
            print(f"{RED}Error:{RESET} The {CYAN}--collection{RESET} parameter is mandatory")
            exit(1)
        if not args.dim:
            print(f"{RED}Error:{RESET} The {CYAN}--dim{RESET} parameter is mandatory")
            exit(1)
        validate_additional_fields(args.additional_fields, args.additional_types)
        client = MilvusHelper(milvus_host=args.host, milvus_port=args.port)
        if client.has_collection(args.collection):
            print(f"Collection {CYAN}{args.collection}{RESET} already exists")
        else:
            is_clustering_key = True
            if args.not_clustering_key:
                is_clustering_key = False
            if client.create_collection(args.collection, vector_dimension=args.dim, shards_num=args.shards_num,
                                        additional_fields=args.additional_fields,
                                        additional_fields_types=args.additional_types,
                                        is_clustering_key=is_clustering_key):
                print("ok")
        sys.exit(0)

    if args.command in ("S", "search"):
        if not args.collection:
            print(f"{RED}Error:{RESET} The {CYAN}--collection{RESET} parameter is mandatory")
            exit(1)
        if len(args.query_file) == 0:
            print(f"{RED}Error:{RESET} The {CYAN}--query_file{RESET} parameter is mandatory")
            exit(1)
        client = MilvusHelper(milvus_host=args.host, milvus_port=args.port)
        if validate_collection_exists(client, args.collection):
            res = search_vectors(client, args.collection, args.query_file, args.k, args.metric_type, args.probe)
            print(f"Found vectors: {res}")
        sys.exit(0)

    # insert data to milvus
    if args.command in ("a", "insert"):
        if not args.base:
            print(f"{RED}Error:{RESET} The {CYAN}--base{RESET} parameter is mandatory")
            exit(1)
        if not args.collection:
            print(f"{RED}Error:{RESET} The {CYAN}--collection{RESET} parameter is mandatory")
            exit(1)
        client = MilvusHelper(milvus_host=args.host, milvus_port=args.port)
        if args.additional_columns_files is not None:
            base_file_extension = os.path.splitext(args.base)[1][1:]
            validate_additional_columns_files(base_file_extension, client, args.collection, args.additional_columns_files, args.base)
        insert_data(client, args.collection, args.base, args.total_vectors, args.chunk_size, shards_num=args.shards_num,
                    additional_columns_files=args.additional_columns_files)
        sys.exit(0)

    # upload data to milvus
    if args.command in ("u", "upload"):
        if not args.base:
            print(f"{RED}Error:{RESET} The {CYAN}--base{RESET} parameter is mandatory")
            exit(1)
        if not args.bucket_name:
            print(f"{RED}Error:{RESET} The {CYAN}--bucket_name{RESET} parameter is mandatory")
            exit(1)
        if not args.upload_desc_file:
            print(f"{RED}Error:{RESET} The {CYAN}--upload_desc_file{RESET} parameter is mandatory")
            exit(1)
        if not args.collection:
            print(f"{RED}Error:{RESET} The {CYAN}--collection{RESET} parameter is mandatory")
            exit(1)
        if args.bulk_file_type == 'json':
            bulk_file_type = BulkFileType.JSON
        else:
            bulk_file_type = BulkFileType.PARQUET
        if args.total_vectors == TOTAL_VECTOR_COUNT:
            total_vectors = 0
        else:
            total_vectors = args.total_vectors
        if args.port == MILVUS_PORT:
            object_storage_port = 9000
        else:
            object_storage_port = args.port
        if args.additional_columns_files is not None:
            base_file_extension = os.path.splitext(args.base)[1][1:]
            validate_additional_columns_files_for_upload(base_file_extension, args.additional_columns_files, args.base)
            validate_additional_fields(args.additional_columns_files, args.additional_types)
        upload_data(args.collection, args.base, total_vectors, args.upload_desc_file, args.host,
                    args.bucket_name, bulk_file_type, args.num_tasks, args.max_file_size_mb, args.file_start_index,
                    object_storage_port, args.collection_start_index, additional_columns_files=args.additional_columns_files,
                    additional_fields_types=args.additional_types)
        sys.exit(0)

    # import data to milvus
    if args.command in ("im", "import"):
        if not args.upload_desc_file:
            print(f"{RED}Error:{RESET} The {CYAN}--upload_desc_file{RESET} parameter is mandatory")
            exit(1)
        if not args.collection:
            print(f"{RED}Error:{RESET} The {CYAN}--collection{RESET} parameter is mandatory")
            exit(1)
        import_data(args.host, args.collection, args.upload_desc_file, args.port)
        sys.exit(0)

    # get list imports
    if args.command in ("lim", "list_imports"):
        if not args.collection:
            print(f"{RED}Error:{RESET} The {CYAN}--collection{RESET} parameter is mandatory")
            exit(1)
        list_imports(args.host, args.collection, args.port)
        sys.exit(0)

    # get import progress
    if args.command in ("imp", "import_progress"):
        if not args.job_id:
            print(f"{RED}Error:{RESET} The {CYAN}--job_id{RESET} parameter is mandatory")
            exit(1)
        if args.details:
            exclude_keys = None
        else:
            exclude_keys = {"details"}
        get_import_progress(args.host, args.job_id, args.port, exclude_keys)
        sys.exit(0)

    # get list buckets
    if args.command in ("lib", "mc_list_buckets"):
        if args.port == MILVUS_PORT:
            object_storage_port = 9000
        else:
            object_storage_port = args.port
        list_buckets(args.host, object_storage_port)
        sys.exit(0)

    # get list objects
    if args.command in ("lio", "mc_list_objects"):
        if not args.bucket_name:
            print(f"{RED}Error:{RESET} The {CYAN}--bucket_name{RESET} parameter is mandatory")
            exit(1)
        if args.port == MILVUS_PORT:
            object_storage_port = 9000
        else:
            object_storage_port = args.port
        list_objects(args.host, args.bucket_name, object_storage_port, args.prefix)
        sys.exit(0)

    # get disk usage
    if args.command in ("du", "mc_disk_usage"):
        if not args.bucket_name:
            print(f"{RED}Error:{RESET} The {CYAN}--bucket_name{RESET} parameter is mandatory")
            exit(1)
        if args.port == MILVUS_PORT:
            object_storage_port = 9000
        else:
            object_storage_port = args.port
        disk_usage(args.host, args.bucket_name, object_storage_port, args.prefix)
        sys.exit(0)

    # remove objects
    if args.command in ("ro", "mc_remove_objects"):
        if not args.bucket_name:
            print(f"{RED}Error:{RESET} The {CYAN}--bucket_name{RESET} parameter is mandatory")
            exit(1)
        if args.port == MILVUS_PORT:
            object_storage_port = 9000
        else:
            object_storage_port = args.port
        remove_objects(args.host, args.bucket_name, object_storage_port, args.dry_run, args.prefix)
        sys.exit(0)

    # build index
    if args.command in ("I", "create_index"):
        if not args.collection:
            print(f"{RED}Error:{RESET} The {CYAN}--collection{RESET} parameter is mandatory")
            exit(1)
        if (args.index_type== 'IVF_FLAT' or args.index_type== 'IVF_PQ') and not args.nlist:
            print(f"{RED}Error:{RESET} The {CYAN}--nlist{RESET} parameter is mandatory for IVF index")
            exit(1)
        client = MilvusHelper(milvus_host=args.host, milvus_port=args.port)
        if validate_collection_exists(client, args.collection):
            create_index(client, args.collection, args.index_type, args.metric_type, args.hnsw_m,
                         args.efconstr, args.max_degree, args.search_list_size, args.pq_code_budget_gb_ratio,
                         args.search_cache_budget_gb_ratio, args.disk_pq_code_budget_gb_ratio, args.nbits, args.sq_type,
                         args.refine, args.refine_type, args.pq_m, args.nlist, args.inline_pq, args.rearrange,
                         args.num_entry_points, args.pq_cache_size)
            print("ok")
        sys.exit(0)

    # Drop collection
    if args.command in ("D", "drop"):
        if not args.collection:
            print(f"{RED}Error:{RESET} The {CYAN}--collection{RESET} parameter is mandatory")
            exit(1)
        client = MilvusHelper(milvus_host=args.host, milvus_port=args.port)
        if validate_collection_exists(client, args.collection):
            status = client.delete_collection(args.collection)
            print(status)
        sys.exit(0)

    if args.command in ("L", "load"):
        if not args.collection:
            print(f"{RED}Error:{RESET} The {CYAN}--collection{RESET} parameter is mandatory")
            exit(1)
        client = MilvusHelper(milvus_host=args.host, milvus_port=args.port)
        if validate_collection_exists(client, args.collection):
            start_time = time.time()
            client.load_data(args.collection, args.exclude)
            end_time = time.time()
            latency = end_time - start_time
            print(f"Load collection finished in {latency*1000:,.4f} ms")
        sys.exit(0)

    if args.command in ("R", "release"):
        if not args.collection:
            print(f"{RED}Error:{RESET} The {CYAN}--collection{RESET} parameter is mandatory")
            exit(1)
        client = MilvusHelper(milvus_host=args.host, milvus_port=args.port)
        if validate_collection_exists(client, args.collection):
            client.release_data(args.collection)
        sys.exit(0)

    # Drop index
    if args.command in ("O", "drop_index"):
        if not args.collection:
            print(f"{RED}Error:{RESET} The {CYAN}--collection{RESET} parameter is mandatory")
            exit(1)
        client = MilvusHelper(milvus_host=args.host, milvus_port=args.port)
        if validate_collection_exists(client, args.collection):
            client.delete_index(args.collection)
        sys.exit(0)

    # present collection info
    if args.command in ("i", "index_info"):
        if not args.collection:
            print(f"{RED}Error:{RESET} The {CYAN}--collection{RESET} parameter is mandatory")
            exit(1)
        client = MilvusHelper(milvus_host=args.host, milvus_port=args.port)
        if validate_collection_exists(client, args.collection):
            print(client.get_index_params(args.collection))
        sys.exit(0)

    if args.command in ("p", "index_progress"):
        if not args.collection:
            print(f"{RED}Error:{RESET} The {CYAN}--collection{RESET} parameter is mandatory")
            exit(1)
        client = MilvusHelper(milvus_host=args.host, milvus_port=args.port)
        if validate_collection_exists(client, args.collection):
            print(client.get_index_progress(args.collection))
        sys.exit(0)

    # Show if collection exists
    if args.command in ("h", "has"):
        if not args.collection:
            print(f"{RED}Error:{RESET} The {CYAN}--collection{RESET} parameter is mandatory")
            exit(1)
        client = MilvusHelper(milvus_host=args.host, milvus_port=args.port)
        print(client.has_collection(args.collection))
        sys.exit(0)

    # Get collection row count
    if args.command in ("r", "rows"):
        if not args.collection:
            print(f"{RED}Error:{RESET} The {CYAN}--collection{RESET} parameter is mandatory")
            exit(1)
        client = MilvusHelper(milvus_host=args.host, milvus_port=args.port)
        print(client.count(args.collection))
        sys.exit(0)

    if args.command in ("C", "compact"):
        if not args.collection:
            print(f"{RED}Error:{RESET} The {CYAN}--collection{RESET} parameter is mandatory")
            exit(1)
        client = MilvusHelper(milvus_host=args.host, milvus_port=args.port)
        client.compact(args.collection)
        sys.exit(0)

    if args.command in ("l", "list"):
        client = MilvusHelper(milvus_host=args.host, milvus_port=args.port)
        print(client.list_collection())
        sys.exit(0)

    if args.command in ("s", "load_progress"):
        if not args.collection:
            print(f"{RED}Error:{RESET} The {CYAN}--collection{RESET} parameter is mandatory")
            exit(1)
        client = MilvusHelper(milvus_host=args.host, milvus_port=args.port)
        print(client.get_loading_progress(args.collection))
        sys.exit(0)

    if args.command in ("d", "desc"):
        if not args.collection:
            print(f"{RED}Error:{RESET} The {CYAN}--collection{RESET} parameter is mandatory")
            exit(1)
        client = MilvusHelper(milvus_host=args.host, milvus_port=args.port)
        client.print_collection_metadata(args.collection)
        sys.exit(0)

    # test search performance
    if args.command in ("P", "performance"):
        if not args.collection:
            print(f"{RED}Error:{RESET} The {CYAN}--collection{RESET} parameter is mandatory")
            exit(1)
        if len(args.query_file) == 0:
            print(f"{RED}Error:{RESET} The {CYAN}--query_file{RESET} parameter is mandatory")
            exit(1)
        if not args.query_file or len(args.query_file) == 0:
            print(f"{RED}Error:{RESET} The {CYAN}--gt_file{RESET} parameter is mandatory")
            exit(1)
        client = MilvusHelper(milvus_host=args.host, milvus_port=args.port)
        performance(client, args.collection, args.search_param, args.query_file, args.gt_file, args.metric_type,
                    args.result_path, args.nq, args.topk, args.refine_k,
                    args.beamwidth, args.vectors_beamwidth, args.pq_read_page_cache_size)
        sys.exit(0)

    if args.command == "version":
        print_version()
        sys.exit(0)

    if args.command in ("n", "rename"):
        if not args.collection:
            print(f"{RED}Error:{RESET} The {CYAN}--collection{RESET} parameter is mandatory")
            exit(1)
        if not args.new_name:
            print(f"{RED}Error:{RESET} The {CYAN}--new_name{RESET} parameter is mandatory")
            exit(1)
        client = MilvusHelper(milvus_host=args.host, milvus_port=args.port)
        client.rename_collection(args.collection, args.new_name)
        sys.exit(0)

    if args.command in ("g", "segments"):
        client = MilvusHelper(milvus_host=args.host, milvus_port=args.port)
        client.print_segments(args.collection, args.sum_by)
        sys.exit(0)

    if args.command in ("cu", "complete_upload"):
        if not args.bucket_name:
            print(f"{RED}Error:{RESET} The {CYAN}--bucket_name{RESET} parameter is mandatory")
            exit(1)
        if not args.upload_desc_file:
            print(f"{RED}Error:{RESET} The {CYAN}--upload_desc_file{RESET} parameter is mandatory")
            exit(1)
        if not args.collection:
            print(f"{RED}Error:{RESET} The {CYAN}--collection{RESET} parameter is mandatory")
            exit(1)

        if args.port == MILVUS_PORT:
            object_storage_port = 9000
        else:
            object_storage_port = args.port
        minio_client = create_object_storage_client(args.host, object_storage_port)
        complete_upload(BULK_WRITER_DIR, args.upload_desc_file, args.collection, minio_client, args.bucket_name)
        sys.exit(0)

    print(f"Unknown operation {args.command}.")
    usage()


def usage():
    print(
        "For parameter descriptions, please refer to "
        "https://dssd-bitbucket.us.kioxia.com/projects/TES/repos/bootcamp/browse/benchmark_test/scripts")
    print(f"{MAGENTA}Usage:{RESET} 'python3 main.py [options] procedure'")


if __name__ == '__main__':
    main()
