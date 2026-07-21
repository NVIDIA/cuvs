#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

import warnings

import numpy as np

cimport cuvs.common.cydlpack

from cuvs.common.resources import auto_sync_resources

from cython.operator cimport dereference as deref
from libcpp cimport bool, cast
from libcpp.string cimport string

from cuvs.common cimport cydlpack

from cuvs.common.device_tensor_view import DeviceTensorView

from cuvs.distance_type cimport cuvsDistanceType

from pylibraft.common import auto_convert_output, cai_wrapper, device_ndarray
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.interruptible import cuda_interruptible

from cuvs.common.cydlpack import dl_data_type_to_numpy
from cuvs.distance import DISTANCE_NAMES, DISTANCE_TYPES
from cuvs.neighbors.common import _check_input_array

from libc.stdint cimport (
    int8_t,
    int32_t,
    int64_t,
    uint8_t,
    uint32_t,
    uint64_t,
    uintptr_t,
)
from libc.stdlib cimport free, malloc
from libc.string cimport strdup

from cuvs.common.exceptions import check_cuvs
from cuvs.neighbors import ivf_pq
from cuvs.neighbors.filters import no_filter


cdef class AceParams:
    """
    Parameters for ACE (Augmented Core Extraction) graph building algorithm.

    ACE enables building indexes for datasets too large to fit in GPU memory by
    partitioning the dataset using balanced k-means and building sub-indexes
    for each partition independently.

    Parameters
    ----------
    npartitions : int, default = 0
        Number of partitions for ACE partitioned build. When set to 0 (default),
        the number of partitions is automatically derived based on available
        host and GPU memory to maximize partition size while ensuring the build
        fits in memory.

        Small values might improve recall but potentially degrade performance
        and increase memory usage. Partitions should not be too small to prevent
        issues in KNN graph construction. The partition size is on average 2 *
        (n_rows / npartitions) * dim * sizeof(T). 2 is because of the core and
        augmented vectors. Please account for imbalance in the partition sizes
        (up to 3x in our tests).

        If the specified number of partitions results in partitions that exceed
        available memory, the value will be automatically increased to fit
        memory constraints and a warning will be issued.
    ef_construction : int, default = 120
        The index quality for the ACE build. Bigger values increase the index
        quality. At some point, increasing this will no longer improve the
        quality.
    build_dir : str, default = "/tmp/ace_build"
        Directory to store ACE build artifacts (e.g., KNN graph, optimized
        graph). Used when `use_disk` is true or when the graph does not fit
        in host and GPU memory. This should be the fastest disk in the system
        and hold enough space for twice the dataset, final graph, and label
        mapping.
    use_disk : bool, default = False
        Whether to use disk-based storage for ACE build. When true, enables
        disk-based operations for memory-efficient graph construction.
    max_host_memory_gb : float, default = 0
        Maximum host memory to use for ACE build in GiB. When set to 0
        (default), uses available host memory. Useful for testing or
        when running alongside other memory-intensive processes.
    max_gpu_memory_gb : float, default = 0
        Maximum GPU memory to use for ACE build in GiB. When set to 0
        (default), uses available GPU memory. Useful for testing or
        when running alongside other memory-intensive processes.
    """
    cdef cuvsAceParams* params
    cdef bytes _build_dir_bytes  # Keep Python bytes alive for property access

    def __cinit__(self):
        check_cuvs(cuvsAceParamsCreate(&self.params))
        self._build_dir_bytes = b""

    def __dealloc__(self):
        if self.params != NULL:
            check_cuvs(cuvsAceParamsDestroy(self.params))

    def __init__(self, *,
                 npartitions=0,
                 ef_construction=120,
                 build_dir="/tmp/ace_build",
                 use_disk=False,
                 max_host_memory_gb=0,
                 max_gpu_memory_gb=0):
        self.params.npartitions = npartitions
        self.params.ef_construction = ef_construction
        self.params.use_disk = use_disk
        self.params.max_host_memory_gb = max_host_memory_gb
        self.params.max_gpu_memory_gb = max_gpu_memory_gb

        # Need to replace the default build_dir allocated by
        # cuvsAceParamsCreate
        # First free the old C string, then allocate new one
        if self.params.build_dir != NULL:
            free(<void*>self.params.build_dir)

        # Store Python bytes for property access
        self._build_dir_bytes = build_dir.encode('utf-8')
        # Allocate C memory and copy the string (strdup-like behavior)
        self.params.build_dir = strdup(self._build_dir_bytes)

    @property
    def npartitions(self):
        return self.params.npartitions

    @property
    def ef_construction(self):
        return self.params.ef_construction

    @property
    def build_dir(self):
        if self._build_dir_bytes:
            return self._build_dir_bytes.decode('utf-8')
        else:
            return ""

    @property
    def use_disk(self):
        return self.params.use_disk

    @property
    def max_host_memory_gb(self):
        return self.params.max_host_memory_gb

    @property
    def max_gpu_memory_gb(self):
        return self.params.max_gpu_memory_gb

    def get_handle(self):
        return <size_t>self.params


cdef class IndexParams:
    """
    Parameters to build index for CAGRA nearest neighbor search

    Parameters
    ----------

    metric : str, default = "sqeuclidean"
        String denoting the metric type, valid values for metric are
        ["sqeuclidean", "inner_product", "cosine"], where:

            - sqeuclidean is the euclidean distance without the square root
              operation, i.e.: distance(a,b) = \\sum_i (a_i - b_i)^2
            - inner_product distance is defined as
              distance(a, b) = \\sum_i a_i * b_i.
            - cosine distance is defined as
              distance(a, b) = 1 - \\sum_i a_i * b_i / ( ||a||_2 * ||b||_2).

    intermediate_graph_degree : int, default = 128
    graph_degree : int, default = 64
    build_algo: str, default = "ivf_pq"
        string denoting the graph building algorithm to use. Valid values for
        algo: ["ivf_pq", "nn_descent", "iterative_cagra_search", "ace"], where

            - ivf_pq will use the IVF-PQ algorithm for building the knn graph
            - nn_descent (experimental) will use the NN-Descent algorithm for
              building the knn graph. It is expected to be generally
              faster than ivf_pq.
            - iterative_cagra_search will iteratively build the knn graph using
              CAGRA's search() and optimize()
            - ace will use ACE (Augmented Core Extraction) for building indices
              for datasets too large to fit in GPU memory

    ivf_pq_build_params: cuvs.neighbors.ivf_pq.IndexParams, optional
        Parameters for IVF-PQ algorithm. If provided, it will be used for
        building the graph.
    ivf_pq_search_params: cuvs.neighbors.ivf_pq.SearchParams, optional
        Parameters for IVF-PQ search. If provided, it will be used for
        searching the graph.
    ace_params: AceParams, optional
        Parameters for ACE algorithm. If provided, it will be used for
        building the graph with ACE partitioning.
    refinement_rate: float, default = 1.0

    """

    def __cinit__(self):
        check_cuvs(cuvsCagraIndexParamsCreate(&self.params))
        self.ivf_pq_build_params = None
        self.ivf_pq_search_params = None
        self.ace_params = None

    def __dealloc__(self):
        if self.params != NULL:
            check_cuvs(cuvsCagraIndexParamsDestroy(self.params))

    def __init__(self, *,
                 metric="sqeuclidean",
                 intermediate_graph_degree=128,
                 graph_degree=64,
                 build_algo="ivf_pq",
                 nn_descent_niter=20,
                 ivf_pq_build_params: ivf_pq.IndexParams = None,
                 ivf_pq_search_params: ivf_pq.SearchParams = None,
                 ace_params: AceParams = None,
                 refinement_rate: float = 1.0):
        # Declare cdef variables at the top of the function
        cdef cuvsIvfPqParams_t ivf_pq_params_ptr
        cdef cuvsAceParams_t new_ace_params

        self.params.metric = <cuvsDistanceType>DISTANCE_TYPES[metric]
        self.params.intermediate_graph_degree = intermediate_graph_degree
        self.params.graph_degree = graph_degree
        if build_algo == "ivf_pq":
            self.params.build_algo = cuvsCagraGraphBuildAlgo.IVF_PQ
        elif build_algo == "nn_descent":
            self.params.build_algo = cuvsCagraGraphBuildAlgo.NN_DESCENT
        elif build_algo == "iterative_cagra_search":
            self.params.build_algo = \
                cuvsCagraGraphBuildAlgo.ITERATIVE_CAGRA_SEARCH
        elif build_algo == "ace":
            self.params.build_algo = cuvsCagraGraphBuildAlgo.ACE
        else:
            raise ValueError(f"Unknown build_algo '{build_algo}'")

        self.params.nn_descent_niter = nn_descent_niter

        # Handle graph build params based on build algorithm
        if build_algo == "ace":
            if ace_params is None:
                ace_params = AceParams()
            self.ace_params = ace_params

            # Create a new C-allocated cuvsAceParams that the C API will own
            # We cannot pass the Python object's pointer directly because
            # cuvsCagraIndexParamsDestroy will try to delete it
            check_cuvs(cuvsAceParamsCreate(&new_ace_params))

            # Copy values from Python object to new C struct
            new_ace_params.npartitions = ace_params.params.npartitions
            new_ace_params.ef_construction = ace_params.params.ef_construction
            new_ace_params.use_disk = ace_params.params.use_disk
            new_ace_params.max_host_memory_gb = ace_params.params.max_host_memory_gb
            new_ace_params.max_gpu_memory_gb = ace_params.params.max_gpu_memory_gb

            # Copy the build_dir string
            if new_ace_params.build_dir != NULL:
                free(<void*>new_ace_params.build_dir)
            new_ace_params.build_dir = strdup(ace_params.params.build_dir)

            # Pass the new C struct to the index params
            self.params.graph_build_params = <void*>new_ace_params
        else:
            # For IVF-PQ algorithm, handle ivf_pq params
            # Cast the void* back to cuvsIvfPqParams_t
            ivf_pq_params_ptr = (
                <cuvsIvfPqParams_t>self.params.graph_build_params
            )

            if ivf_pq_build_params is not None:
                if ivf_pq_build_params.metric != self.metric:
                    raise ValueError(
                        "Metric mismatch with IVF-PQ build params"
                    )
                self.ivf_pq_build_params = ivf_pq_build_params
                ivf_pq_params_ptr.ivf_pq_build_params = (
                    <cuvsIvfPqIndexParams_t><size_t>
                    ivf_pq_build_params.get_handle()
                )

            if ivf_pq_search_params is not None:
                self.ivf_pq_search_params = ivf_pq_search_params
                ivf_pq_params_ptr.ivf_pq_search_params = (
                    <cuvsIvfPqSearchParams_t><size_t>
                    ivf_pq_search_params.get_handle()
                )

            ivf_pq_params_ptr.refinement_rate = refinement_rate

    def get_handle(self):
        return <size_t> self.params

    @property
    def metric(self):
        return DISTANCE_NAMES[self.params.metric]

    @property
    def intermediate_graph_degree(self):
        return self.params.intermediate_graph_degree

    @property
    def graph_degree(self):
        return self.params.graph_degree

    @property
    def build_algo(self):
        return self.params.build_algo

    @property
    def nn_descent_niter(self):
        return self.params.nn_descent_niter

    @property
    def refinement_rate(self):
        # refinement_rate only applies to IVF-PQ builds
        if self.params.build_algo == cuvsCagraGraphBuildAlgo.IVF_PQ:
            return (
                (<cuvsIvfPqParams_t>self.params.graph_build_params)
                .refinement_rate
            )
        else:
            # For ACE and other algorithms, refinement_rate doesn't apply
            return 1.0


cdef class Index:
    def __cinit__(self):
        self.trained = False
        self.active_index_type = None
        check_cuvs(cuvsCagraIndexCreate(&self.index))

    def __dealloc__(self):
        if self.index is not NULL:
            check_cuvs(cuvsCagraIndexDestroy(self.index))

    @property
    def trained(self):
        return self.trained

    @property
    def dim(self):
        cdef int64_t dim
        check_cuvs(cuvsCagraIndexGetDims(self.index, &dim))
        return dim

    @property
    def graph_degree(self):
        cdef int64_t degree
        check_cuvs(cuvsCagraIndexGetGraphDegree(self.index, &degree))
        return degree

    def __len__(self):
        cdef int64_t size
        check_cuvs(cuvsCagraIndexGetSize(self.index, &size))
        return size

    @property
    def dtype(self):
        return dl_data_type_to_numpy(self.index.dtype)

    @property
    def dataset(self):
        if not self.trained:
            raise ValueError("Index needs to be built before getting dataset")

        # get the cagra dataset from the index without copying as dlpack
        output = DeviceTensorView()
        cdef cydlpack.DLManagedTensor * tensor = \
            <cydlpack.DLManagedTensor*><size_t>output.get_handle()
        check_cuvs(cuvsCagraIndexGetDataset(self.index, tensor))

        # since we're referencing memory internal to this cagra index in the
        # output view, keep the cagra index alive as long as the output view
        # is to avoid segfaulting
        output.parent = self

        return output

    @property
    def graph(self):
        if not self.trained:
            raise ValueError("Index needs to be built before getting graph")

        output = DeviceTensorView()
        cdef cydlpack.DLManagedTensor * tensor = \
            <cydlpack.DLManagedTensor*><size_t>output.get_handle()
        check_cuvs(cuvsCagraIndexGetGraph(self.index, tensor))
        output.parent = self
        return output

    def __repr__(self):
        # todo(dgd): update repr as we expose data through C API
        attr_str = []
        return "Index(type=CAGRA, metric=L2" + (", ".join(attr_str)) + ")"


cdef class PaddedDataset:
    def __cinit__(self):
        self.dataset = NULL

    def __dealloc__(self):
        if self.dataset != NULL:
            check_cuvs(cuvsDatasetPaddedDestroy(self.dataset))


cdef class StandardDataset:
    def __cinit__(self):
        self.dataset = NULL

    def __dealloc__(self):
        if self.dataset != NULL:
            check_cuvs(cuvsDatasetStandardDestroy(self.dataset))


cdef class PaddedDatasetView:
    def __cinit__(self):
        self.view = NULL

    def __dealloc__(self):
        if self.view != NULL:
            check_cuvs(cuvsDatasetPaddedViewDestroy(self.view))


@auto_sync_resources
def build(IndexParams index_params, dataset, resources=None):
    """
    Build the CAGRA index from the dataset for efficient search.

    The build performs two different steps- first an intermediate knn-graph is
    constructed, then it's optimized it to create the final graph. The
    index_params object controls the node degree of these graphs.

    It is required that both the dataset and the optimized graph fit the
    GPU memory.

    Note: When using ACE (Augmented Core Extraction) build algorithm, the
    dataset must be in host memory (CPU). The ACE algorithm is designed for
    datasets too large to fit in GPU memory.

    The following distance metrics are supported:
        - L2
        - InnerProduct
        - Cosine

    Parameters
    ----------
    index_params : IndexParams object
    dataset : CUDA array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float, half, int8, uint8]
        **Note:** For ACE build algorithm, the dataset MUST be in host memory.
        Use NumPy arrays or call .get() on CuPy arrays before passing.
    {resources_docstring}

    Returns
    -------
    index: cuvs.cagra.Index

    Examples
    --------

    >>> import cupy as cp
    >>> from cuvs.neighbors import cagra
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> k = 10
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> build_params = cagra.IndexParams(metric="sqeuclidean")
    >>> index = cagra.build(build_params, dataset)
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> distances, neighbors = cagra.search(cagra.SearchParams(),
    ...                                      index, queries,
    ...                                      k)
    >>> distances = cp.asarray(distances)
    >>> neighbors = cp.asarray(neighbors)
    """

    # Check if ACE build is requested
    is_ace_build = (
        index_params.params.build_algo == cuvsCagraGraphBuildAlgo.ACE
    )

    # todo(dgd): we can make the check of dtype a parameter of wrap_array
    # in RAFT to make this a single call
    dataset_ai = wrap_array(dataset)
    _check_input_array(dataset_ai, [np.dtype('float32'),
                                    np.dtype('float16'),
                                    np.dtype('byte'),
                                    np.dtype('ubyte')])

    # For ACE, verify dataset is on host
    if is_ace_build:
        # Check if data is on device (has __cuda_array_interface__)
        if hasattr(dataset, '__cuda_array_interface__'):
            raise ValueError(
                "ACE build requires dataset to be in host memory. "
                "Please use NumPy arrays or transfer CuPy arrays to host with "
                "dataset.get() before calling build()."
            )

    cdef Index idx = Index()
    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_ai)
    cdef cuvsCagraIndexParams* params = index_params.params
    cdef cuvsDatasetPaddedView_t padded_view = NULL
    cdef cuvsDatasetStandardView_t standard_view = NULL
    cdef cuvsDatasetViewKind_t view_kind

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    with cuda_interruptible():
        try:
            check_cuvs(cuvsCagraGetDatasetViewKind(dataset_dlpack, &view_kind))
            if view_kind == CUVS_DATASET_VIEW_KIND_DEVICE_PADDED:
                check_cuvs(cuvsDatasetMakeDevicePaddedView(res, dataset_dlpack, &padded_view))
                check_cuvs(cuvsCagraBuildDevicePadded(res, params, padded_view, idx.index))
            elif view_kind == CUVS_DATASET_VIEW_KIND_DEVICE_STANDARD:
                check_cuvs(cuvsDatasetMakeDeviceStandardView(res, dataset_dlpack, &standard_view))
                check_cuvs(cuvsCagraBuildDeviceStandard(res, params, standard_view, idx.index))
            elif view_kind == CUVS_DATASET_VIEW_KIND_HOST_PADDED:
                check_cuvs(cuvsDatasetMakeHostPaddedView(res, dataset_dlpack, &padded_view))
                check_cuvs(cuvsCagraBuildHostPadded(res, params, padded_view, idx.index))
            else:
                check_cuvs(cuvsDatasetMakeHostStandardView(res, dataset_dlpack, &standard_view))
                check_cuvs(cuvsCagraBuildHostStandard(res, params, standard_view, idx.index))
            idx.trained = True
            idx.active_index_type = dataset_ai.dtype.name
        finally:
            if padded_view != NULL:
                cuvsDatasetPaddedViewDestroy(padded_view)
            if standard_view != NULL:
                cuvsDatasetStandardViewDestroy(standard_view)

    return idx


def get_dataset_view_kind(dataset):
    """
    Return dataset view kind as one of:
    "device_padded", "device_standard", "host_padded", "host_standard".
    """
    dataset_ai = wrap_array(dataset)
    _check_input_array(dataset_ai, [np.dtype('float32'),
                                    np.dtype('float16'),
                                    np.dtype('byte'),
                                    np.dtype('ubyte')])
    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_ai)
    cdef cuvsDatasetViewKind_t view_kind
    with cuda_interruptible():
        check_cuvs(cuvsCagraGetDatasetViewKind(dataset_dlpack, &view_kind))

    if view_kind == CUVS_DATASET_VIEW_KIND_DEVICE_PADDED:
        return "device_padded"
    elif view_kind == CUVS_DATASET_VIEW_KIND_DEVICE_STANDARD:
        return "device_standard"
    elif view_kind == CUVS_DATASET_VIEW_KIND_HOST_PADDED:
        return "host_padded"
    elif view_kind == CUVS_DATASET_VIEW_KIND_HOST_STANDARD:
        return "host_standard"
    raise RuntimeError("unexpected dataset view kind")


@auto_sync_resources
def make_device_padded_dataset(dataset, resources=None):
    """
    Create an owning device padded dataset handle via explicit C API factory.
    """
    dataset_ai = wrap_array(dataset)
    _check_input_array(dataset_ai, [np.dtype('float32'),
                                    np.dtype('float16'),
                                    np.dtype('byte'),
                                    np.dtype('ubyte')])
    if not hasattr(dataset, '__cuda_array_interface__'):
        raise ValueError("dataset must be in device memory")

    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_ai)
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    cdef PaddedDataset padded = PaddedDataset()
    with cuda_interruptible():
        check_cuvs(cuvsDatasetMakeDevicePadded(
            res,
            dataset_dlpack,
            &padded.dataset
        ))
    return padded


def make_view_from_owning_padded(PaddedDataset padded_dataset):
    """
    Create a padded dataset view handle from an owning padded dataset handle.
    """
    if padded_dataset is None or padded_dataset.dataset == NULL:
        raise ValueError("padded_dataset is uninitialized")
    cdef PaddedDatasetView padded_view = PaddedDatasetView()
    check_cuvs(cuvsDatasetMakeViewFromOwningPadded(
        padded_dataset.dataset,
        &padded_view.view
    ))
    return padded_view


@auto_sync_resources
def attach_device_dataset_on_host_index(Index index, device_dataset, resources=None):
    """
    Convert a host-built CAGRA index to device index by attaching a device dataset.
    """
    if not index.trained:
        raise ValueError("Index needs to be built before attaching device dataset.")
    if not hasattr(device_dataset, '__cuda_array_interface__'):
        raise ValueError("device_dataset must be in device memory")

    dataset_ai = wrap_array(device_dataset)
    _check_input_array(dataset_ai, [np.dtype('float32'),
                                    np.dtype('float16'),
                                    np.dtype('byte'),
                                    np.dtype('ubyte')])
    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_ai)
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    with cuda_interruptible():
        check_cuvs(cuvsCagraAttachDeviceDatasetOnHostIndex(
            res,
            dataset_dlpack,
            index.index
        ))
    return index


@auto_sync_resources
def attach_padded_dataset_for_search(Index index,
                                     PaddedDatasetView padded_dataset_view,
                                     resources=None):
    """
    Attach a padded dataset view to a standard CAGRA index before search.
    """
    if not index.trained:
        raise ValueError("Index needs to be built before attaching padded dataset.")
    if padded_dataset_view is None or padded_dataset_view.view == NULL:
        raise ValueError("padded_dataset_view is uninitialized")
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    with cuda_interruptible():
        check_cuvs(cuvsCagraAttachPaddedDatasetForSearch(
            res,
            padded_dataset_view.view,
            index.index
        ))
    return index


def build_index(IndexParams index_params, dataset, resources=None):
    warnings.warn("cagra.build_index is deprecated, use cagra.build instead",
                  FutureWarning)
    return build(index_params, dataset, resources=resources)


cdef class SearchParams:
    """
    CAGRA search parameters

    Parameters
    ----------

    max_queries: int, default = 0
        Maximum number of queries to search at the same time (batch size).
        Auto select when 0.
    itopk_size: int, default = 64
        Number of intermediate search results retained during the search.
        This is the main knob to adjust trade off between accuracy and
        search speed. Higher values improve the search accuracy.
    max_iterations: int, default = 0
        Upper limit of search iterations. Auto select when 0.
    algo: str, default = "auto"
        String denoting the search algorithm to use
        Valid values for algo: ["auto", "single_cta", "multi_cta"], where:

            - auto will automatically select the best value based on query size
            - single_cta is better when query contains larger number of
              vectors (e.g >10)
            - multi_cta is better when query contains only a few vectors

    team_size: int, default = 0
        Number of threads used to calculate a single distance. 4, 8, 16,
        or 32.
    search_width: int, default = 1
        Number of graph nodes to select as the starting point for the
        search in each iteration.
    min_iterations: int, default = 0
        Lower limit of search iterations.
    thread_block_size: int, default = 0
        Thread block size. 0, 64, 128, 256, 512, 1024.
        Auto selection when 0.
    hashmap_mode: str, default = "auto"
        String denoting the type of hash map to use.
        It's usually better to allow the algorithm to select this value,
        Valid values for hashmap_mode: ["auto", "small", "hash"], where:

            - auto will automatically select the best value based on algo
            - small will use the small shared memory hash table with resetting.
            - hash will use a single hash table in global memory.

    hashmap_min_bitlen: int, default = 0
        Upper limit of hashmap fill rate. More than 0.1, less than 0.9.
    hashmap_max_fill_rate: float, default = 0.5
        Upper limit of hashmap fill rate. More than 0.1, less than 0.9.
    num_random_samplings: int, default = 1
        Number of iterations of initial random seed node selection. 1 or
        more.
    rand_xor_mask: int, default = 0x128394
        Bit mask used for initial random seed node selection.
    persistent: bool, default = false
        Whether to use the persistent version of the kernel
    persistent_lifetime: float
        Persistent kernel: time in seconds before the kernel stops if no
        requests are received.
    persistent_device_usage : float
        Sets the fraction of maximum grid size used by persistent kernel.

    """

    def __cinit__(self):
        check_cuvs(cuvsCagraSearchParamsCreate(&self.params))

    def __dealloc__(self):
        if self.params != NULL:
            check_cuvs(cuvsCagraSearchParamsDestroy(self.params))

    def __init__(self, *,
                 max_queries=0,
                 itopk_size=64,
                 max_iterations=0,
                 algo="auto",
                 team_size=0,
                 search_width=1,
                 min_iterations=0,
                 thread_block_size=0,
                 hashmap_mode="auto",
                 hashmap_min_bitlen=0,
                 hashmap_max_fill_rate=0.5,
                 num_random_samplings=1,
                 rand_xor_mask=0x128394,
                 persistent=False,
                 persistent_lifetime=None,
                 persistent_device_usage=None
                 ):
        self.params.max_queries = max_queries
        self.params.itopk_size = itopk_size
        self.params.max_iterations = max_iterations
        if algo == "single_cta":
            self.params.algo = cuvsCagraSearchAlgo.SINGLE_CTA
        elif algo == "multi_cta":
            self.params.algo = cuvsCagraSearchAlgo.MULTI_CTA
        elif algo == "multi_kernel":
            self.params.algo = cuvsCagraSearchAlgo.MULTI_KERNEL
        elif algo == "auto":
            self.params.algo = cuvsCagraSearchAlgo.AUTO
        else:
            raise ValueError("`algo` value not supported.")

        self.params.team_size = team_size
        self.params.search_width = search_width
        self.params.min_iterations = min_iterations
        self.params.thread_block_size = thread_block_size
        if hashmap_mode == "hash":
            self.params.hashmap_mode = cuvsCagraHashMode.HASH
        elif hashmap_mode == "small":
            self.params.hashmap_mode = cuvsCagraHashMode.SMALL
        elif hashmap_mode == "auto":
            self.params.hashmap_mode = cuvsCagraHashMode.AUTO_HASH
        else:
            raise ValueError("`hashmap_mode` value not supported.")

        self.params.hashmap_min_bitlen = hashmap_min_bitlen
        self.params.hashmap_max_fill_rate = hashmap_max_fill_rate
        self.params.num_random_samplings = num_random_samplings
        self.params.rand_xor_mask = rand_xor_mask
        self.params.persistent = persistent
        if persistent_lifetime is not None:
            self.params.persistent_lifetime = persistent_lifetime
        if persistent_device_usage is not None:
            self.params.persistent_device_usage = persistent_device_usage

    def __repr__(self):
        attr_str = [attr + "=" + str(getattr(self, attr))
                    for attr in [
                        "max_queries", "itopk_size", "max_iterations", "algo",
                        "team_size", "search_width", "min_iterations",
                        "thread_block_size", "hashmap_mode",
                        "hashmap_min_bitlen", "hashmap_max_fill_rate",
                        "num_random_samplings", "rand_xor_mask"]]
        return "SearchParams(type=CAGRA, " + (", ".join(attr_str)) + ")"

    def get_handle(self):
        return <size_t> self.params

    @property
    def max_queries(self):
        return self.params.max_queries

    @property
    def itopk_size(self):
        return self.params.itopk_size

    @property
    def max_iterations(self):
        return self.params.max_iterations

    @property
    def algo(self):
        return self.params.algo

    @property
    def team_size(self):
        return self.params.team_size

    @property
    def search_width(self):
        return self.params.search_width

    @property
    def min_iterations(self):
        return self.params.min_iterations

    @property
    def thread_block_size(self):
        return self.params.thread_block_size

    @property
    def hashmap_mode(self):
        return self.params.hashmap_mode

    @property
    def hashmap_min_bitlen(self):
        return self.params.hashmap_min_bitlen

    @property
    def hashmap_max_fill_rate(self):
        return self.params.hashmap_max_fill_rate

    @property
    def num_random_samplings(self):
        return self.params.num_random_samplings

    @property
    def rand_xor_mask(self):
        return self.params.rand_xor_mask


@auto_sync_resources
@auto_convert_output
def search(SearchParams search_params,
           Index index,
           queries,
           k,
           neighbors=None,
           distances=None,
           resources=None,
           filter=None):
    """
    Find the k nearest neighbors for each query.

    Parameters
    ----------

    search_params : SearchParams
    index : Index
        Trained CAGRA index.
    queries : CUDA array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float, int8, uint8]
    k : int
        The number of neighbors.
    neighbors : Optional CUDA array interface compliant matrix shape
                (n_queries, k), dtype int64_t. If supplied, neighbor
                indices will be written here in-place. (default None)
    distances : Optional CUDA array interface compliant matrix shape
                (n_queries, k) If supplied, the distances to the
                neighbors will be written here in-place. (default None)
    filter:     Optional cuvs.neighbors.cuvsFilter can be used to filter
                neighbors based on a given bitset.
                (default None)
    {resources_docstring}

    Examples
    --------

    >>> import cupy as cp
    >>> from cuvs.neighbors import cagra
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build index
    >>> index = cagra.build(cagra.IndexParams(), dataset)
    >>> # Search using the built index
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> k = 10
    >>> search_params = cagra.SearchParams(
    ...     max_queries=100,
    ...     itopk_size=64
    ... )
    >>> # Using a pooling allocator reduces overhead of temporary array
    >>> # creation during search. This is useful if multiple searches
    >>> # are performed with same query size.
    >>> distances, neighbors = cagra.search(search_params, index, queries,
    ...                                     k)
    >>> neighbors = cp.asarray(neighbors)
    >>> distances = cp.asarray(distances)

    """
    if not index.trained:
        raise ValueError("Index needs to be built before calling search.")

    # todo(dgd): we can make the check of dtype a parameter of wrap_array
    # in RAFT to make this a single call
    queries_cai = wrap_array(queries)
    _check_input_array(queries_cai, [np.dtype('float32'),
                                     np.dtype('float16'),
                                     np.dtype('byte'),
                                     np.dtype('ubyte')])

    cdef uint32_t n_queries = queries_cai.shape[0]

    if neighbors is None:
        neighbors = device_ndarray.empty((n_queries, k), dtype='uint32')

    neighbors_cai = wrap_array(neighbors)
    _check_input_array(neighbors_cai, [np.dtype('uint32')],
                       exp_rows=n_queries, exp_cols=k)

    if distances is None:
        distances = device_ndarray.empty((n_queries, k), dtype='float32')

    distances_cai = wrap_array(distances)
    _check_input_array(distances_cai, [np.dtype('float32')],
                       exp_rows=n_queries, exp_cols=k)

    if filter is None:
        filter = no_filter()

    cdef cuvsCagraSearchParams* params = search_params.params
    cdef cydlpack.DLManagedTensor* queries_dlpack = \
        cydlpack.dlpack_c(queries_cai)
    cdef cydlpack.DLManagedTensor* neighbors_dlpack = \
        cydlpack.dlpack_c(neighbors_cai)
    cdef cydlpack.DLManagedTensor* distances_dlpack = \
        cydlpack.dlpack_c(distances_cai)
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    with cuda_interruptible():
        check_cuvs(cuvsCagraSearch(
            res,
            params,
            index.index,
            queries_dlpack,
            neighbors_dlpack,
            distances_dlpack,
            filter.prefilter
        ))

    return (distances, neighbors)


@auto_sync_resources
def save(filename, Index index, bool include_dataset=True, resources=None):
    """
    Saves the index to a file.

    Saving / loading the index is experimental. The serialization format is
    subject to change.

    Parameters
    ----------
    filename : string
        Name of the file.
    index : Index
        Trained CAGRA index.
    include_dataset : bool
        Whether or not to write out the dataset along with the index. Including
        the dataset in the serialized index will use extra disk space, and
        might not be desired if you already have a copy of the dataset on
        disk. If this option is set to false, you will have to call
        `index.update_dataset(dataset)` after loading the index.
    {resources_docstring}

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.neighbors import cagra
    >>> n_samples = 50000
    >>> n_features = 50
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build index
    >>> index = cagra.build(cagra.IndexParams(), dataset)
    >>> # Serialize and deserialize the cagra index built
    >>> cagra.save("my_index.bin", index)
    >>> index_loaded = cagra.Index()
    >>> out_dataset = cagra.PaddedDataset()
    >>> cagra.load(index_loaded, "my_index.bin", out_dataset=out_dataset)
    """
    cdef string c_filename = filename.encode('utf-8')
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    check_cuvs(cuvsCagraSerialize(res,
                                  c_filename.c_str(),
                                  index.index,
                                  include_dataset))


@auto_sync_resources
def load(index, filename, out_dataset=None, resources=None):
    """
    Deserialize CAGRA index into an existing index handle.

    Saving / loading the index is experimental. The serialization format is
    subject to change, therefore loading an index saved with a previous
    version of cuvs is not guaranteed to work.

    Parameters
    ----------
    index : Index
        Pre-created index object to populate.
    filename : string
        Name of the file.
    out_dataset : PaddedDataset or StandardDataset, optional
        Optional pre-created owning dataset output handle. The concrete type
        dispatches to the corresponding deserialize entrypoint.
    {resources_docstring}
    """
    cdef Index idx = index
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    cdef string c_filename = filename.encode('utf-8')
    cdef cuvsDatasetPadded_t out_padded_dataset = NULL
    cdef cuvsDatasetStandard_t out_standard_dataset = NULL
    cdef cuvsDatasetPadded_t* out_padded_ptr
    cdef cuvsDatasetStandard_t* out_standard_ptr

    if isinstance(out_dataset, PaddedDataset):
        out_padded_ptr = &out_padded_dataset
        check_cuvs(cuvsCagraDeserializePadded(
            res,
            c_filename.c_str(),
            idx.index,
            out_padded_ptr
        ))
        (<PaddedDataset>out_dataset).dataset = out_padded_dataset
    elif isinstance(out_dataset, StandardDataset):
        out_standard_ptr = &out_standard_dataset
        check_cuvs(cuvsCagraDeserializeStandard(
            res,
            c_filename.c_str(),
            idx.index,
            out_standard_ptr
        ))
        (<StandardDataset>out_dataset).dataset = out_standard_dataset
    else:
        raise TypeError(
            "out_dataset must be a PaddedDataset or StandardDataset "
            "to choose deserialize target layout"
        )
    idx.trained = True


@auto_sync_resources
def from_graph(graph, dataset, metric="sqeuclidean", resources=None):
    """
    Construct a cagra index from an existing graph and dataset

    Parameters
    ----------
    graph : Array interface compliant matrix with shape (n_samples,
        graph_degree)
    dataset : Array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float32, float16, int8, uint8]
    metric : str
    {resources_docstring}

    Returns
    -------
    index: cuvs.cagra.Index
    """
    cdef cuvsDistanceType c_metric = <cuvsDistanceType>DISTANCE_TYPES[metric]

    cdef Index idx = Index()
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    dataset_ai = wrap_array(dataset)
    _check_input_array(dataset_ai, [np.dtype('float32'),
                                    np.dtype('float16'),
                                    np.dtype('byte'),
                                    np.dtype('ubyte')])
    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_ai)

    graph_ai = wrap_array(graph)
    _check_input_array(graph_ai, [np.dtype('uint32')])
    cdef cydlpack.DLManagedTensor* graph_dlpack = \
        cydlpack.dlpack_c(graph_ai)

    check_cuvs(cuvsCagraIndexFromArgs(
        res,
        c_metric,
        graph_dlpack,
        dataset_dlpack,
        idx.index))
    idx.trained = True
    return idx


cdef class ExtendParams:
    """ Supplemental parameters to extend CAGRA Index

    Parameters
    ----------
    max_chunk_size : int
        The additional dataset is divided into chunks and added to the graph.
        This is the knob to adjust the tradeoff between the recall and
        operation throughput. Large chunk sizes can result in high throughput,
        but use more working memory (O(max_chunk_size*degree^2)). This can also
        degrade recall because no edges are added between the nodes in the same
        chunk. Auto select when 0.
    """

    cdef cuvsCagraExtendParams* params

    def __cinit__(self):
        check_cuvs(cuvsCagraExtendParamsCreate(&self.params))

    def __dealloc__(self):
        check_cuvs(cuvsCagraExtendParamsDestroy(self.params))

    def __init__(self, *,
                 max_chunk_size=None):
        if max_chunk_size is not None:
            self.params.max_chunk_size = max_chunk_size

    @property
    def max_chunk_size(self):
        return self.params.max_chunk_size


@auto_sync_resources
def extend(ExtendParams params, Index index, additional_dataset,
           resources=None):
    """
    Extend a CAGRA index with additional vectors

    Parameters
    ----------
    params : ExtendParams object
    index: Index
       Existing cagra index to extend
    additional_dataset : PaddedDatasetView
        Explicit padded dataset view for vectors to append.
    {resources_docstring}

    """
    if not isinstance(additional_dataset, PaddedDatasetView):
        raise TypeError(
            "additional_dataset must be a PaddedDatasetView. "
            "Create it explicitly via make_device_padded_dataset() + "
            "make_view_from_owning_padded()."
        )
    cdef cuvsDatasetPaddedView_t padded_view = (<PaddedDatasetView>additional_dataset).view
    if padded_view == NULL:
        raise ValueError("additional_dataset padded view is uninitialized")

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    cdef cuvsDatasetStorage_t extended_dataset = NULL

    check_cuvs(cuvsMakeExtendedStorage(
        res,
        padded_view,
        index.index,
        &extended_dataset
    ))

    try:
        with cuda_interruptible():
            check_cuvs(cuvsCagraExtend(
                res,
                params.params,
                padded_view,
                extended_dataset,
                index.index
            ))
    finally:
        if extended_dataset != NULL:
            check_cuvs(cuvsDatasetStorageDestroy(extended_dataset))

    return index
