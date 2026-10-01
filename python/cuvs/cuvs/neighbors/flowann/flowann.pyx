#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

import numpy as np

cimport cuvs.common.cydlpack

from libc.stdint cimport int64_t
from libcpp.string cimport string

from cuvs.common cimport cydlpack
from cuvs.common.c_api cimport cuvsResources_t
from cuvs.common.dataset cimport Dataset, make_device_padded_dataset_handle
from cuvs.common.exceptions import check_cuvs
from cuvs.common.resources import auto_sync_resources
from cuvs.distance import DISTANCE_TYPES
from cuvs.distance_type cimport cuvsDistanceType
from cuvs.neighbors.cagra.cagra cimport (
    AUTO_SELECT,
    IVF_PQ,
    NN_DESCENT,
    cuvsCagraHashMode,
    cuvsCagraSearchAlgo,
)
from cuvs.neighbors.common import _check_input_array

from pylibraft.common import auto_convert_output, device_ndarray
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.interruptible import cuda_interruptible


cdef class IndexParams:
    """Parameters for building a dense FlowANN index.

    Parameters
    ----------
    metric : {"sqeuclidean", "inner_product", "cosine"}
    intermediate_graph_degree : int, default 128
    graph_degree : int, default 64
    build_algo : {"auto", "ivf_pq", "nn_descent"}, default "auto"
    device_graph_budget_bytes : int, default 0
        Maximum bytes used by the packed resident graph. A zero budget keeps
        graph neighbors in host memory.
    node_per_cacheline : int, default 2
    n_groups : int, default 0
        Zero derives the group count from the dataset size.
    n_bits : int, default 0
        Zero derives an aligned local-ID width from the dataset size.
    num_seeds : int, default 0
        Number of medoid seeds generated with the index.
    """

    def __cinit__(self):
        self.params = NULL
        check_cuvs(cuvsFlowannIndexParamsCreate(&self.params))

    def __dealloc__(self):
        if self.params != NULL:
            cuvsFlowannIndexParamsDestroy(self.params)

    def __init__(self, *, metric="sqeuclidean",
                 intermediate_graph_degree=128,
                 graph_degree=64,
                 build_algo="auto",
                 nn_descent_niter=20,
                 device_graph_budget_bytes=0,
                 node_per_cacheline=2,
                 grouping_enabled=True,
                 n_groups=0,
                 n_bits=0,
                 balance_tolerance=0.10,
                 training_rows=0,
                 assignment_batch_rows=0,
                 kmeans_n_iters=20,
                 validate=False,
                 num_seeds=0,
                 seed_training_rows=0,
                 seed=0x9e3779b97f4a7c15):
        if metric not in DISTANCE_TYPES:
            raise ValueError(f"Unknown metric '{metric}'")
        self.params.metric = <cuvsDistanceType>DISTANCE_TYPES[metric]
        self.params.intermediate_graph_degree = intermediate_graph_degree
        self.params.graph_degree = graph_degree
        if build_algo == "auto":
            self.params.build_algo = AUTO_SELECT
        elif build_algo == "ivf_pq":
            self.params.build_algo = IVF_PQ
        elif build_algo == "nn_descent":
            self.params.build_algo = NN_DESCENT
        else:
            raise ValueError("build_algo must be 'auto', 'ivf_pq', or 'nn_descent'")
        self.params.nn_descent_niter = nn_descent_niter
        self.params.device_graph_budget_bytes = device_graph_budget_bytes
        self.params.node_per_cacheline = node_per_cacheline
        self.params.grouping_enabled = grouping_enabled
        self.params.n_groups = n_groups
        self.params.n_bits = n_bits
        self.params.balance_tolerance = balance_tolerance
        self.params.training_rows = training_rows
        self.params.assignment_batch_rows = assignment_batch_rows
        self.params.kmeans_n_iters = kmeans_n_iters
        self.params.validate = validate
        self.params.num_seeds = num_seeds
        self.params.seed_training_rows = seed_training_rows
        self.params.seed = seed


cdef class SearchParams:
    """FlowANN search and host queue parameters."""

    def __cinit__(self):
        self.params = NULL
        check_cuvs(cuvsFlowannSearchParamsCreate(&self.params))

    def __dealloc__(self):
        if self.params != NULL:
            cuvsFlowannSearchParamsDestroy(self.params)

    def __init__(self, *, max_queries=0, itopk_size=64, max_iterations=0,
                 algo="auto", team_size=0, search_width=1,
                 min_iterations=0, thread_block_size=0,
                 hashmap_mode="auto", hashmap_min_bitlen=0,
                 hashmap_max_fill_rate=0.5, num_random_samplings=1,
                 rand_xor_mask=0x128394,
                 num_seeds=0, sync_window_scale=10.0,
                 sync_drop_threshold=51, num_queues=1,
                 empty_pause=64, collect_statistics=False):
        self.params.cagra.max_queries = max_queries
        self.params.cagra.itopk_size = itopk_size
        self.params.cagra.max_iterations = max_iterations
        if algo == "auto":
            self.params.cagra.algo = cuvsCagraSearchAlgo.AUTO
        elif algo == "single_cta":
            self.params.cagra.algo = cuvsCagraSearchAlgo.SINGLE_CTA
        elif algo == "multi_cta":
            self.params.cagra.algo = cuvsCagraSearchAlgo.MULTI_CTA
        else:
            raise ValueError("algo must be 'auto', 'single_cta', or 'multi_cta'")
        self.params.cagra.team_size = team_size
        self.params.cagra.search_width = search_width
        self.params.cagra.min_iterations = min_iterations
        self.params.cagra.thread_block_size = thread_block_size
        if hashmap_mode == "auto":
            self.params.cagra.hashmap_mode = cuvsCagraHashMode.AUTO_HASH
        elif hashmap_mode == "hash":
            self.params.cagra.hashmap_mode = cuvsCagraHashMode.HASH
        elif hashmap_mode == "small":
            self.params.cagra.hashmap_mode = cuvsCagraHashMode.SMALL
        else:
            raise ValueError("hashmap_mode must be 'auto', 'hash', or 'small'")
        self.params.cagra.hashmap_min_bitlen = hashmap_min_bitlen
        self.params.cagra.hashmap_max_fill_rate = hashmap_max_fill_rate
        self.params.cagra.num_random_samplings = num_random_samplings
        self.params.cagra.rand_xor_mask = rand_xor_mask
        self.params.num_seeds = num_seeds
        self.params.sync_window_scale = sync_window_scale
        self.params.sync_drop_threshold = sync_drop_threshold
        self.params.num_queues = num_queues
        self.params.empty_pause = empty_pause
        self.params.collect_statistics = collect_statistics


cdef class Index:
    """Owning dense FlowANN index."""

    def __cinit__(self):
        self.index = NULL
        self._trained = False
        check_cuvs(cuvsFlowannIndexCreate(&self.index))

    def __dealloc__(self):
        if self.index != NULL:
            cuvsFlowannIndexDestroy(self.index)

    @property
    def trained(self):
        return self._trained

    @property
    def size(self):
        cdef int64_t value
        if not self._trained:
            return 0
        check_cuvs(cuvsFlowannIndexGetSize(self.index, &value))
        return value

    @property
    def dim(self):
        cdef int64_t value
        if not self._trained:
            return 0
        check_cuvs(cuvsFlowannIndexGetDims(self.index, &value))
        return value

    @property
    def graph_degree(self):
        cdef int64_t value
        if not self._trained:
            return 0
        check_cuvs(cuvsFlowannIndexGetGraphDegree(self.index, &value))
        return value


@auto_sync_resources
def build(IndexParams index_params, dataset, resources=None):
    """Build an owning dense FlowANN index from a host or device matrix.

    The matrix dtype must be float32, int8, or uint8.
    """
    dataset_ai = wrap_array(dataset)
    _check_input_array(dataset_ai, [np.dtype("float32"),
                                    np.dtype("int8"),
                                    np.dtype("uint8")])
    cdef cydlpack.DLManagedTensor* dataset_dlpack = cydlpack.dlpack_c(dataset_ai)
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    cdef Dataset padded
    cdef Index index = Index()
    with cuda_interruptible():
        padded = make_device_padded_dataset_handle(res, dataset_dlpack)
        check_cuvs(cuvsFlowannBuild(res, index_params.params,
                                    padded.dataset, index.index))
    index._trained = True
    return index


@auto_sync_resources
@auto_convert_output
def search(SearchParams search_params, Index index, queries, k,
           neighbors=None, distances=None, resources=None):
    """Find the ``k`` nearest neighbors for each query."""
    if not index._trained:
        raise ValueError("Index needs to be built or loaded before search")
    queries_ai = wrap_array(queries)
    _check_input_array(queries_ai, [np.dtype("float32"),
                                    np.dtype("int8"),
                                    np.dtype("uint8")],
                       exp_cols=index.dim)
    cdef int64_t n_queries = queries_ai.shape[0]
    if neighbors is None:
        neighbors = device_ndarray.empty((n_queries, k), dtype="uint32")
    if distances is None:
        distances = device_ndarray.empty((n_queries, k), dtype="float32")
    neighbors_ai = wrap_array(neighbors)
    distances_ai = wrap_array(distances)
    _check_input_array(neighbors_ai, [np.dtype("uint32")],
                       exp_rows=n_queries, exp_cols=k)
    _check_input_array(distances_ai, [np.dtype("float32")],
                       exp_rows=n_queries, exp_cols=k)
    cdef cydlpack.DLManagedTensor* queries_dlpack = cydlpack.dlpack_c(queries_ai)
    cdef cydlpack.DLManagedTensor* neighbors_dlpack = cydlpack.dlpack_c(neighbors_ai)
    cdef cydlpack.DLManagedTensor* distances_dlpack = cydlpack.dlpack_c(distances_ai)
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    with cuda_interruptible():
        check_cuvs(cuvsFlowannSearch(res, search_params.params, index.index,
                                     queries_dlpack, neighbors_dlpack,
                                     distances_dlpack))
    return distances, neighbors


@auto_sync_resources
def save(filename, Index index, resources=None):
    """Serialize a self-contained dense FlowANN index."""
    if not index._trained:
        raise ValueError("Index needs to be built or loaded before save")
    cdef string c_filename = str(filename).encode("utf-8")
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    check_cuvs(cuvsFlowannSerialize(res, c_filename.c_str(), index.index))


@auto_sync_resources
def load(filename, dtype=np.float32, resources=None):
    """Load a self-contained dense FlowANN index."""
    cdef cydlpack.DLDataType dl_dtype
    dtype = np.dtype(dtype)
    if dtype == np.dtype("float32"):
        dl_dtype.code = cydlpack.kDLFloat
        dl_dtype.bits = 32
    elif dtype == np.dtype("int8"):
        dl_dtype.code = cydlpack.kDLInt
        dl_dtype.bits = 8
    elif dtype == np.dtype("uint8"):
        dl_dtype.code = cydlpack.kDLUInt
        dl_dtype.bits = 8
    else:
        raise ValueError("dtype must be float32, int8, or uint8")
    dl_dtype.lanes = 1
    cdef string c_filename = str(filename).encode("utf-8")
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    cdef Index index = Index()
    check_cuvs(cuvsFlowannDeserialize(res, c_filename.c_str(),
                                      dl_dtype, index.index))
    index._trained = True
    return index
