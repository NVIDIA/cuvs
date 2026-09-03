import sys
from pymilvus.bulk_writer import BulkFileType

MILVUS_PORT = 19530

OBJECT_STORAGE_ACCESS_KEY = "minioadmin"
OBJECT_STORAGE_SECRET_KEY = "minioadmin"
OBJECT_STORAGE_PORT = 9000
##################### Collection Parameters ########################################################

METRIC_TYPE = 'L2'
SHARDS_NUM = 1
INDEX_TYPE="DISKANN"
VECTOR_FIELD="embedding"
##################### Search Parameters ##########################################################
K = 4
PROBE = 10

##################### Indexing Parameters ##########################################################

# Index IVF parameters
NLIST = 4096
PQ_M = 12  # number of subquantizers

# Index NSG parameters
SEARCH_LENGTH = 45
OUT_DEGREE = 50
KNNG = 100

#Index DISKANN parameters
# Index HNSW parameters
HNSW_M = 16
EFCONSTRUCTION = 500
# Index HNSW Flat parameters
EF = 0  # should be larger than k
# Index HNSW PQ parameters
NBITS = 8  # number of bits per subquantizer
# Index HNSW SQ parameters
SQ_TYPE = None  # scalar quantizer type {"sq6", "sq8", "fp16", "bf16"}
# Index HNSW PQ/SQ parameters
REFINE = False  # whether the refine is used during the train (in index built)
DETAILS = False  # whether to display more details in output
REFINE_K = 0  # undefined value leads to a search without a refine
REFINE_TYPE = None  # the type of refine index {"sq6", "sq8", "fp16", "bf16", "fp32", "flat"}

# DISKANN AISAQ related parameters
BEAMWIDTH = 1
VECTORS_BEAMWIDTH = 1
NOT_CLUSTERING_KEY = False  # whether the refine is used during the train (in index built)

# Index ANNOY parameters
N_TREE = 8

##################### Insert Parameters ############################################################

# File type used for base and query
FILE_TYPE = [
    #'npy',
    # 'csv',
     'fvecs',
    # 'bvecs',
]

QUERY_FILE = '/home/zeev/work/git/milvus-client/data/siftsmall_query.fvecs'
# Point to directory of file data.
BASE_FILE_PATH = '/mnt/csd-card-root/base.50M.fvec'
#BASE_FILE_PATH = '/mnt/gili-index/sift_base.fvecs'
# Does the data need to be normalized before insertion
IF_NORMALIZE = False
# If dealing with bvecs or fvecs files. Import chunk size must be <= 256mb
TOTAL_VECTOR_COUNT = sys.maxsize
IMPORT_CHUNK_SIZE = 20000
UPLOAD_CHUNK_SIZE_MB = 1024 # 1GB
MEMORY_USAGE_THRESHOLD = 13
BULK_FILE_TYPE = BulkFileType.PARQUET

##################### Performance Test Parameters ##################################################

# Location of the query files

# Path to put performance results to, based on current directory.
PERFORMANCE_RESULTS_PATH = 'performance'

# Scope of performance results. For each NQ_Scope, all the TOPK values will be tested
# NQ_SCOPE = [1, 10, 100, 500, 1000]
NQ_SCOPE = [10000]
TOPK_SCOPE = [1, 1, 10]

PERCENTILE_NUM = 100


##################### Recall Test Parameters #######################################################

# Number of queries to be searched for in test
RECALL_NQ = 1000

# TopK value to be computed for each query
RECALL_TOPK = 10

# Recall accuracies to be calculated, largest number must by < RECALL_TOPK
RECALL_CALC_SCOPE = [1, 10]

IS_CSV = False
IS_UINT8 = False

# Location of ground truth file
GROUNDTRUTH_FILE = '/mnt/csd-card-root/gt50M_100.fbin'
#GROUNDTRUTH_FILE = '/mnt/gili-index/sift_groundtruth.ivecs'

# Result locations
RECALL_RES = 'recall_result'
RECALL_RES_TOPK = 'recall_result/recall_compare_out'

# The number of log files will be saved
LOGS_NUM = 1

####################################################################
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'
BULK_WRITER_DIR = 'bulk_writer'
