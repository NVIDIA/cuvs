#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from libc.stdint cimport uint32_t
from libcpp cimport bool
from libcpp.string cimport string

from cuvs.common.exceptions import check_cuvs
from cuvs.common.resources import auto_sync_resources
from cuvs.neighbors.common import _check_input_array

from cuvs.common cimport cydlpack

import numpy as np

from cuvs.distance import DISTANCE_TYPES

from cuvs.neighbors.cagra cimport cagra

import os
import uuid

from pylibraft.common import auto_convert_output
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.interruptible import cuda_interruptible


cdef class AceParams:
    """
    Parameters for ACE (Augmented Core Extraction) graph build for HNSW.

    ACE enables building HNSW indices for datasets too large to fit in GPU
    memory by partitioning the dataset and building sub-indices for each
    partition independently.

    Parameters
    ----------
    npartitions : int, default = 0 (optional)
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
    build_dir : string, default = "/tmp/hnsw_ace_build" (optional)
        Directory to store ACE build artifacts (KNN graph, optimized graph).
        Used when `use_disk` is true or when the graph does not fit in memory.
    use_disk : bool, default = False (optional)
        Whether to use disk-based storage for ACE build. When true, enables
        disk-based operations for memory-efficient graph construction.
    max_host_memory_gb : float, default = 0 (optional)
        Maximum host memory to use for ACE build in GiB. When set to 0
        (default), uses available host memory. Useful for testing or
        when running alongside other memory-intensive processes.
    max_gpu_memory_gb : float, default = 0 (optional)
        Maximum GPU memory to use for ACE build in GiB. When set to 0
        (default), uses available GPU memory. Useful for testing or
        when running alongside other memory-intensive processes.
    """

    cdef cuvsHnswAceParams* params
    cdef object _build_dir_bytes

    def __cinit__(self):
        check_cuvs(cuvsHnswAceParamsCreate(&self.params))
        self._build_dir_bytes = None

    def __dealloc__(self):
        if self.params is not NULL:
            check_cuvs(cuvsHnswAceParamsDestroy(self.params))

    def __init__(self, *,
                 npartitions=0,
                 build_dir="/tmp/hnsw_ace_build",
                 use_disk=False,
                 max_host_memory_gb=0,
                 max_gpu_memory_gb=0):
        self.params.npartitions = npartitions
        self._build_dir_bytes = build_dir.encode('utf-8')
        self.params.build_dir = self._build_dir_bytes
        self.params.use_disk = use_disk
        self.params.max_host_memory_gb = max_host_memory_gb
        self.params.max_gpu_memory_gb = max_gpu_memory_gb

    @property
    def npartitions(self):
        return self.params.npartitions

    @property
    def build_dir(self):
        if self.params.build_dir is not NULL:
            return self.params.build_dir.decode('utf-8')
        return "/tmp/hnsw_ace_build"

    @property
    def use_disk(self):
        return self.params.use_disk

    @property
    def max_host_memory_gb(self):
        return self.params.max_host_memory_gb

    @property
    def max_gpu_memory_gb(self):
        return self.params.max_gpu_memory_gb


cdef class IndexParams:
    """
    Parameters to build index for HNSW nearest neighbor search

    Parameters
    ----------
    hierarchy : string, default = "gpu" (optional)
        The hierarchy of the HNSW index. Valid values are ["none", "cpu", "gpu"].
        - "none": No hierarchy is built.
        - "cpu": Hierarchy is built using CPU.
        - "gpu": Hierarchy is built using GPU.
    ef_construction : int, default = 200 (optional)
        Maximum candidate list size used during index construction.
    num_threads : int, default = 0 (optional)
        Number of CPU threads used to increase construction parallelism
        when hierarchy is `cpu` or `gpu`. When the value is 0, the number of
        threads is automatically determined to the maximum number of threads
        available.
        NOTE: When hierarchy is `gpu`, while the majority of the work is done
        on the GPU, initialization of the HNSW index itself and some other
        work is parallelized with the help of CPU threads.
    M : int, default = 32 (optional)
        HNSW M parameter: number of bi-directional links per node. When the
        graph is built on the GPU, this parameter is used to derive the
        internal CAGRA graph build parameters.
    metric : string, default = "sqeuclidean" (optional)
        Distance metric to use. Valid values: ["sqeuclidean", "inner_product"]
    ace_params : AceParams, default = None (optional)
        Explicit ACE parameters for out-of-core graph construction. When not
        set, the graph build algorithm is selected automatically.
    """

    cdef cuvsHnswIndexParams* params
    cdef AceParams _ace_params

    def __cinit__(self):
        check_cuvs(cuvsHnswIndexParamsCreate(&self.params))
        self._ace_params = None

    def __dealloc__(self):
        check_cuvs(cuvsHnswIndexParamsDestroy(self.params))

    def __init__(self, *,
                 hierarchy="gpu",
                 ef_construction=200,
                 num_threads=0,
                 M=32,
                 metric="sqeuclidean",
                 ace_params=None):
        if hierarchy == "none":
            self.params.hierarchy = cuvsHnswHierarchy.NONE
        elif hierarchy == "cpu":
            self.params.hierarchy = cuvsHnswHierarchy.CPU
        elif hierarchy == "gpu":
            self.params.hierarchy = cuvsHnswHierarchy.GPU
        else:
            raise ValueError("Invalid hierarchy type."
                             " Valid values are 'none', 'cpu', and 'gpu'.")
        self.params.ef_construction = ef_construction
        self.params.num_threads = num_threads
        self.params.M = M
        self.params.metric = DISTANCE_TYPES[metric]

        if ace_params is not None:
            if not isinstance(ace_params, AceParams):
                raise ValueError("ace_params must be an instance of AceParams")
            self._ace_params = ace_params
            self.params.ace_params = self._ace_params.params
        else:
            self.params.ace_params = NULL

    @property
    def hierarchy(self):
        if self.params.hierarchy == cuvsHnswHierarchy.NONE:
            return "none"
        elif self.params.hierarchy == cuvsHnswHierarchy.CPU:
            return "cpu"
        elif self.params.hierarchy == cuvsHnswHierarchy.GPU:
            return "gpu"

    @property
    def ef_construction(self):
        return self.params.ef_construction

    @property
    def num_threads(self):
        return self.params.num_threads

    @property
    def m(self):
        return self.params.M

    @property
    def ace_params(self):
        return self._ace_params


cdef class Index:
    """
    HNSW index object. This object stores the trained HNSW index state
    which can be used to perform nearest neighbors searches.
    """

    cdef cuvsHnswIndex_t index
    cdef bool trained

    def __cinit__(self):
        self.trained = False
        check_cuvs(cuvsHnswIndexCreate(&self.index))

    def __dealloc__(self):
        if self.index is not NULL:
            check_cuvs(cuvsHnswIndexDestroy(self.index))

    @property
    def trained(self):
        return self.trained

    def __repr__(self):
        # todo(dgd): update repr as we expose data through C API
        attr_str = []
        return "Index(type=HNSW, metric=L2" + (", ".join(attr_str)) + ")"


cdef class ExtendParams:
    """
    Parameters to extend the HNSW index with new data

    Parameters
    ----------
    num_threads : int, default = 0 (optional)
        Number of CPU threads used to increase construction parallelism.
        When set to 0, the number of threads is automatically determined.
    """

    cdef cuvsHnswExtendParams* params

    def __cinit__(self):
        check_cuvs(cuvsHnswExtendParamsCreate(&self.params))

    def __dealloc__(self):
        check_cuvs(cuvsHnswExtendParamsDestroy(self.params))

    def __init__(self, *,
                 num_threads=0):
        self.params.num_threads = num_threads

    @property
    def num_threads(self):
        return self.params.num_threads


@auto_sync_resources
def save(filename, Index index, resources=None):
    """
    Saves the CAGRA index to a file as an hnswlib index.
    If the index was constructed with `hnsw.IndexParams(hierarchy="none")`,
    then the saved index is immutable and can only be searched by the hnswlib
    wrapper in cuVS, as the format is not compatible with the original hnswlib.
    However, if the index was constructed with
    `hnsw.IndexParams(hierarchy="cpu")`, then the saved index is mutable and
    compatible with the original hnswlib.

    Saving / loading the index is experimental. The serialization format is
    subject to change.

    Parameters
    ----------
    filename : string
        Name of the file.
    index : Index
        Trained HNSW index.
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
    >>> cagra_index = cagra.build(cagra.IndexParams(), dataset)
    >>> # Serialize and deserialize the cagra index built
    >>> hnsw_index = hnsw.from_cagra(hnsw.IndexParams(), cagra_index)
    >>> hnsw.save("my_index.bin", hnsw_index)
    """
    cdef string c_filename = filename.encode('utf-8')
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    check_cuvs(cuvsHnswSerialize(res,
                                 c_filename.c_str(),
                                 index.index))


@auto_sync_resources
def load(IndexParams index_params, filename, dim, dtype, metric="sqeuclidean",
         resources=None):
    """
    Loads an HNSW index.
    If the index was constructed with `hnsw.IndexParams(hierarchy="none")`,
    then the loaded index is immutable and can only be searched by the hnswlib
    wrapper in cuVS, as the format is not compatible with the original hnswlib.
    However, if the index was constructed with
    `hnsw.IndexParams(hierarchy="cpu")`, then the loaded index is mutable and
    compatible with the original hnswlib.

    Saving / loading the index is experimental. The serialization format is
    subject to change, therefore loading an index saved with a previous
    version of cuVS is not guaranteed to work.

    Parameters
    ----------
    index_params : IndexParams
        Parameters that were used to convert CAGRA index to HNSW index.
    filename : string
        Name of the file.
    dim : int
        Dimensions of the training dataest
    dtype : np.dtype of the saved index
        Valid values for dtype: [np.float32, np.byte, np.ubyte]
    metric : string denoting the metric type, default="sqeuclidean"
        Valid values for metric: ["sqeuclidean", "inner_product"], where
            - sqeuclidean is the euclidean distance without the square root
              operation, i.e.: distance(a,b) = \\sum_i (a_i - b_i)^2,
            - inner_product distance is defined as
              distance(a, b) = \\sum_i a_i * b_i.
    {resources_docstring}

    Returns
    -------
    index : HnswIndex

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.neighbors import cagra
    >>> from cuvs.neighbors import hnsw
    >>> n_samples = 50000
    >>> n_features = 50
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build index
    >>> index = cagra.build(cagra.IndexParams(), dataset)
    >>> # Serialize the CAGRA index to hnswlib base layer only index format
    >>> hnsw.save("my_index.bin", index)
    >>> index = hnsw.load("my_index.bin", n_features, np.float32,
    ...                   "sqeuclidean")
    """
    cdef Index idx = Index()
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    cdef string c_filename = filename.encode('utf-8')
    cdef cydlpack.DLDataType dl_dtype
    if dtype == np.float32:
        dl_dtype.code = cydlpack.kDLFloat
        dl_dtype.bits = 32
        dl_dtype.lanes = 1
    elif dtype == np.float16:
        dl_dtype.code = cydlpack.kDLFloat
        dl_dtype.bits = 16
        dl_dtype.lanes = 1
    elif dtype == np.ubyte:
        dl_dtype.code = cydlpack.kDLUInt
        dl_dtype.bits = 8
        dl_dtype.lanes = 1
    elif dtype == np.byte:
        dl_dtype.code = cydlpack.kDLInt
        dl_dtype.bits = 8
        dl_dtype.lanes = 1
    else:
        raise ValueError("Only float32, float16, uint8, and int8 are supported"
                         " for dtype")

    idx.index.dtype = dl_dtype
    cdef cuvsDistanceType distance_type = DISTANCE_TYPES[metric]

    check_cuvs(cuvsHnswDeserialize(
        res,
        index_params.params,
        c_filename.c_str(),
        dim,
        distance_type,
        idx.index
    ))
    idx.trained = True
    return idx


@auto_sync_resources
def from_cagra(IndexParams index_params, cagra.Index cagra_index,
               temporary_index_path=None, resources=None):
    """
    Returns an HNSW index from a CAGRA index.

    NOTE: When `index_params.hierarchy` is:

    1. `NONE`: This method uses the filesystem to write the CAGRA index in
    `/tmp/<random_number>.bin` before reading it as an hnswlib index, then
    deleting the temporary file. The returned index is immutable and can only
    be searched by the hnswlib wrapper in cuVS, as the format is not
    compatible with the original hnswlib.
    2. `CPU`: The returned index is mutable and can be extended with additional
    vectors. The serialized index is also compatible with the original hnswlib
    library.

    Saving / loading the index is experimental. The serialization format is
    subject to change.

    Parameters
    ----------

    index_params : IndexParams
        Parameters to convert the CAGRA index to HNSW index.
    cagra_index : cagra.Index
        Trained CAGRA index.
    temporary_index_path : string, default = None
        Path to save the temporary index file. If None, the temporary file
        will be saved in `/tmp/<random_number>.bin`.
    {resources_docstring}

    Examples
    --------

    >>> import cupy as cp
    >>> from cuvs.neighbors import cagra
    >>> from cuvs.neighbors import hnsw
    >>> n_samples = 50000
    >>> n_features = 50
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build index
    >>> index = cagra.build(cagra.IndexParams(), dataset)
    >>> # Serialize the CAGRA index to hnswlib base layer only index format
    >>> hnsw_index = hnsw.from_cagra(hnsw.IndexParams(), index)

    """

    cdef Index hnsw_index = Index()
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    check_cuvs(cuvsHnswFromCagra(
        res,
        index_params.params,
        cagra_index.index,
        hnsw_index.index
    ))

    hnsw_index.trained = True
    return hnsw_index


@auto_sync_resources
def build(IndexParams index_params, dataset, resources=None):
    """
    Build an HNSW index from HNSW parameters.

    The graph is built on the GPU and converted to an HNSW index that can be
    searched on the CPU. The graph build algorithm is selected automatically
    unless explicit ACE parameters are provided.

    Parameters
    ----------
    index_params : IndexParams
        Parameters for the HNSW index.
    dataset : Host array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float32, float16, int8, uint8]
    {resources_docstring}

    Returns
    -------
    index : Index
        Trained HNSW index ready for search.

    Examples
    --------
    >>> import numpy as np
    >>> from cuvs.neighbors import hnsw
    >>>
    >>> n_samples = 50000
    >>> n_features = 50
    >>> dataset = np.random.random_sample((n_samples, n_features),
    ...                                   dtype=np.float32)
    >>>
    >>> # Create HNSW index parameters
    >>> index_params = hnsw.IndexParams(
    ...     hierarchy="gpu",
    ...     ef_construction=120,
    ...     M=32,
    ...     metric="sqeuclidean"
    ... )
    >>>
    >>> # Build the index
    >>> index = hnsw.build(index_params, dataset)
    >>>
    >>> # Search the index
    >>> queries = np.random.random_sample((10, n_features), dtype=np.float32)
    >>> distances, neighbors = hnsw.search(
    ...     hnsw.SearchParams(ef=200),
    ...     index,
    ...     queries,
    ...     k=10
    ... )
    """
    dataset_ai = wrap_array(dataset)
    _check_input_array(dataset_ai, [np.dtype('float32'),
                                    np.dtype('float16'),
                                    np.dtype('uint8'),
                                    np.dtype('int8')])

    cdef cydlpack.DLManagedTensor* dataset_dlpack = cydlpack.dlpack_c(dataset_ai)
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef Index hnsw_index = Index()
    check_cuvs(cuvsHnswBuild(
        res,
        index_params.params,
        dataset_dlpack,
        hnsw_index.index
    ))

    hnsw_index.trained = True
    return hnsw_index


@auto_sync_resources
def extend(ExtendParams extend_params, Index index, data, resources=None):
    """
    Extends the HNSW index with new data.

    Parameters
    ----------
    extend_params : ExtendParams
    index : Index
        Trained HNSW index.
    data : Host array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float32, float16, int8, uint8]
    {resources_docstring}

    Examples
    --------
    >>> import numpy as np
    >>> from cuvs.neighbors import hnsw, cagra
    >>>
    >>> n_samples = 50000
    >>> n_features = 50
    >>> dataset = np.random.random_sample((n_samples, n_features))
    >>>
    >>> # Build index
    >>> index = cagra.build(hnsw.IndexParams(), dataset)
    >>> # Load index
    >>> hnsw_index = hnsw.from_cagra(hnsw.IndexParams(hierarchy="cpu"), index)
    >>> # Extend the index with new data
    >>> new_data = np.random.random_sample((n_samples, n_features))
    >>> hnsw.extend(hnsw.ExtendParams(), hnsw_index, new_data)
    """

    data_ai = wrap_array(data)
    _check_input_array(data_ai, [np.dtype('float32'),
                                 np.dtype('float16'),
                                 np.dtype('uint8'),
                                 np.dtype('int8')])

    cdef cydlpack.DLManagedTensor* data_dlpack = cydlpack.dlpack_c(data_ai)
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    check_cuvs(cuvsHnswExtend(
        res,
        extend_params.params,
        data_dlpack,
        index.index
    ))


cdef class SearchParams:
    """
    HNSW search parameters

    Parameters
    ----------
    ef: int, default = 200
        Maximum number of candidate list size used during search.
    num_threads: int, default = 0
        Number of CPU threads used to increase search parallelism.
        When set to 0, the number of threads is automatically determined
        using OpenMP's `omp_get_max_threads()`.
    """

    cdef cuvsHnswSearchParams params

    def __init__(self, *,
                 ef=200,
                 num_threads=0):
        self.params.ef = ef
        self.params.num_threads = num_threads

    def __repr__(self):
        attr_str = [attr + "=" + str(getattr(self, attr))
                    for attr in [
                        "ef", "num_threads"]]
        return "SearchParams(type=HNSW, " + (", ".join(attr_str)) + ")"

    @property
    def ef(self):
        return self.params.ef

    @property
    def num_threads(self):
        return self.params.num_threads


@auto_sync_resources
@auto_convert_output
def search(SearchParams search_params,
           Index index,
           queries,
           k,
           neighbors=None,
           distances=None,
           resources=None):
    """
    Find the k nearest neighbors for each query.

    Parameters
    ----------
    search_params : SearchParams
    index : Index
        Trained HNSW index.
    queries : CPU array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float, int]
    k : int
        The number of neighbors.
    neighbors : Optional CPU array interface compliant matrix shape
                (n_queries, k), dtype uint64_t. If supplied, neighbor
                indices will be written here in-place. (default None)
    distances : Optional CPU array interface compliant matrix shape
                (n_queries, k) If supplied, the distances to the
                neighbors will be written here in-place. (default None)
    {resources_docstring}

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.neighbors import cagra, hnsw
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
    >>> search_params = hnsw.SearchParams(
    ...     ef=200,
    ...     num_threads=0
    ... )
    >>> # Convert CAGRA index to HNSW
    >>> hnsw_index = hnsw.from_cagra(hnsw.IndexParams(), index)
    >>> # Using a pooling allocator reduces overhead of temporary array
    >>> # creation during search. This is useful if multiple searches
    >>> # are performed with same query size.
    >>> distances, neighbors = hnsw.search(search_params, index, queries,
    ...                                     k)
    >>> neighbors = cp.asarray(neighbors)
    >>> distances = cp.asarray(distances)
    """
    if not index.trained:
        raise ValueError("Index needs to be built before calling search.")

    # todo(dgd): we can make the check of dtype a parameter of wrap_array
    # in RAFT to make this a single call
    queries_ai = wrap_array(queries)
    _check_input_array(queries_ai, [np.dtype('float32'),
                                    np.dtype('float16'),
                                    np.dtype('uint8'),
                                    np.dtype('int8')])

    cdef uint32_t n_queries = queries_ai.shape[0]

    if neighbors is None:
        neighbors = np.empty((n_queries, k), dtype='uint64')

    neighbors_ai = wrap_array(neighbors)
    _check_input_array(neighbors_ai, [np.dtype('uint64')],
                       exp_rows=n_queries, exp_cols=k)

    if distances is None:
        distances = np.empty((n_queries, k), dtype='float32')

    distances_ai = wrap_array(distances)
    _check_input_array(distances_ai, [np.dtype('float32')],
                       exp_rows=n_queries, exp_cols=k)

    cdef cuvsHnswSearchParams* params = &search_params.params
    cdef cydlpack.DLManagedTensor* queries_dlpack = \
        cydlpack.dlpack_c(queries_ai)
    cdef cydlpack.DLManagedTensor* neighbors_dlpack = \
        cydlpack.dlpack_c(neighbors_ai)
    cdef cydlpack.DLManagedTensor* distances_dlpack = \
        cydlpack.dlpack_c(distances_ai)
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    with cuda_interruptible():
        check_cuvs(cuvsHnswSearch(
            res,
            params,
            index.index,
            queries_dlpack,
            neighbors_dlpack,
            distances_dlpack
        ))

    return (distances, neighbors)
