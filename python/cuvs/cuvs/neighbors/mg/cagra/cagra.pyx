#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

import numpy as np

from libc.stdint cimport uint32_t
from libcpp.string cimport string

from pylibraft.common import auto_convert_output
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.interruptible import cuda_interruptible

from cuvs.common.exceptions import check_cuvs
from cuvs.common.mg_resources import auto_sync_multi_gpu_resources
from cuvs.neighbors.common import _check_input_array, _check_memory_location

from cuvs.common cimport cydlpack
from cuvs.common.c_api cimport cuvsResources_t
from cuvs.common.dataset cimport Dataset, cuvsDataset_t
from cuvs.common.dataset import make_device_padded_dataset
from cuvs.neighbors.cagra.cagra cimport (
    IndexParams as SingleGpuIndexParams,
    SearchParams as SingleGpuSearchParams,
    cuvsCagraIndexParams_t,
    cuvsCagraIndexParamsDestroy,
    cuvsCagraSearchParams_t,
    cuvsCagraSearchParamsDestroy,
)

from .cagra cimport (
    cuvsMultiGpuCagraBuild,
    cuvsMultiGpuCagraDeserialize,
    cuvsMultiGpuCagraDistribute,
    cuvsMultiGpuCagraExtend,
    cuvsMultiGpuCagraIndex_t,
    cuvsMultiGpuCagraIndexCreate,
    cuvsMultiGpuCagraIndexDestroy,
    cuvsMultiGpuCagraIndexParams_t,
    cuvsMultiGpuCagraIndexParamsCreate,
    cuvsMultiGpuCagraIndexParamsDestroy,
    cuvsMultiGpuCagraSearch,
    cuvsMultiGpuCagraSearchParams_t,
    cuvsMultiGpuCagraSearchParamsCreate,
    cuvsMultiGpuCagraSearchParamsDestroy,
    cuvsMultiGpuCagraSerialize,
    cuvsMultiGpuCagraUpdateDataset,
    cuvsMultiGpuDistributionMode,
    cuvsMultiGpuReplicatedSearchMode,
    cuvsMultiGpuShardedMergeMode,
)


cdef class IndexParams(SingleGpuIndexParams):
    """
    Parameters to build multi-GPU CAGRA index for efficient search.
    Extends single-GPU IndexParams with multi-GPU specific parameters.

    Parameters
    ----------
    distribution_mode : str, default = "sharded"
        Distribution mode for multi-GPU setup.
        Valid values: ["replicated", "sharded"]
    **kwargs : Additional parameters passed to single-GPU IndexParams

    Note
    ----
    CAGRA currently only supports "sqeuclidean" and "inner_product" metrics.
    """

    def __cinit__(self):
        # Base class __cinit__ has already created self.params
        # We need to destroy it and use our embedded params instead
        if self.params != NULL:
            check_cuvs(cuvsCagraIndexParamsDestroy(self.params))

        # Create multi-GPU params which includes embedded base params
        check_cuvs(cuvsMultiGpuCagraIndexParamsCreate(&self.mg_params))
        # Replace base pointer with embedded base params
        self.params = self.mg_params.base_params

    def __dealloc__(self):
        # Only destroy the mg_params, which will handle base_params cleanup
        check_cuvs(cuvsMultiGpuCagraIndexParamsDestroy(self.mg_params))
        self.mg_params = NULL
        self.params = NULL

    def __init__(self, *, distribution_mode="sharded", **kwargs):
        super().__init__(**kwargs)
        if distribution_mode == "replicated":
            self.mg_params.mode = CUVS_NEIGHBORS_MG_REPLICATED
        elif distribution_mode == "sharded":
            self.mg_params.mode = CUVS_NEIGHBORS_MG_SHARDED
        else:
            raise ValueError(
                "distribution_mode must be 'replicated' or 'sharded'")

    def get_handle(self):
        return <size_t> self.mg_params

    @property
    def distribution_mode(self):
        return ("replicated" if self.mg_params.mode ==
                CUVS_NEIGHBORS_MG_REPLICATED else "sharded")


cdef class Index:
    """
    Multi-GPU CAGRA index object. Stores the trained multi-GPU CAGRA index
    state which can be used to perform nearest neighbors searches across
    multiple GPUs.
    """

    def __cinit__(self):
        # Initialize multi-GPU index
        check_cuvs(cuvsMultiGpuCagraIndexCreate(&self.mg_index))
        # Initialize multi-GPU trained state
        self.mg_trained = False

    def __dealloc__(self):
        check_cuvs(cuvsMultiGpuCagraIndexDestroy(self.mg_index))

    def __repr__(self):
        return "Index(type=MultiGpuCagra)"

    @property
    def trained(self):
        return self.mg_trained


@auto_sync_multi_gpu_resources
def build(IndexParams index_params, dataset, resources=None):
    """
    Build the multi-GPU CAGRA index from the dataset for efficient search.

    Parameters
    ----------
    index_params : :py:class:`cuvs.neighbors.cagra.IndexParams`
    dataset : Array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float32, float16, int8, uint8]
        **IMPORTANT**: For multi-GPU CAGRA, the dataset MUST be in host
        memory (CPU). If using CuPy/device arrays, transfer to host with
        array.get() or cp.asnumpy(array).
    {resources_docstring}

    Returns
    -------
    index: py:class:`cuvs.neighbors.cagra.Index`

    Examples
    --------

    >>> import numpy as np
    >>> from pylibraft.common import device_ndarray
    >>> from cuvs.common import make_device_padded_dataset
    >>> from cuvs.neighbors import cagra as sg_cagra
    >>> from cuvs.neighbors.mg import cagra
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> k = 10
    >>> # For multi-GPU CAGRA, use host (NumPy) arrays
    >>> dataset = np.random.random_sample((n_samples, n_features)).astype(
    ...     np.float32)
    >>> build_params = cagra.IndexParams(metric="sqeuclidean")
    >>> index = cagra.build(build_params, dataset)
    >>> device_dataset = device_ndarray(dataset)
    >>> padded_dataset = make_device_padded_dataset(device_dataset)
    >>> _ = cagra.update_dataset(index, padded_dataset)
    >>> distances, neighbors = cagra.search(cagra.SearchParams(),
    ...                                         index, dataset, k)
    >>> # Results are already in host memory (NumPy arrays)
    """

    dataset_ai = wrap_array(dataset)
    _check_input_array(dataset_ai, [np.dtype('float32'), np.dtype('float16'),
                                    np.dtype('byte'), np.dtype('ubyte')])

    # Multi-GPU CAGRA requires dataset in host memory
    _check_memory_location(dataset, expected_host=True, name="dataset")

    cdef Index idx = Index()
    cdef cydlpack.DLManagedTensor* dataset_dlpack = (
        cydlpack.dlpack_c(dataset_ai))
    cdef cuvsMultiGpuCagraIndexParams_t params = index_params.mg_params

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    # Build the multi-GPU index
    with cuda_interruptible():
        check_cuvs(cuvsMultiGpuCagraBuild(
            res, params, dataset_dlpack, idx.mg_index))
        idx.mg_trained = True

    return idx


@auto_sync_multi_gpu_resources
def update_dataset(Index index, padded_dataset, resources=None):
    """
    Update a multi-GPU CAGRA index with a padded dataset.

    Accepts a ``Dataset`` or array.
    """
    if not index.trained:
        raise ValueError("Index needs to be built before updating the dataset.")

    cdef Dataset dataset_obj
    if isinstance(padded_dataset, Dataset):
        dataset_obj = padded_dataset
    else:
        dataset_obj = make_device_padded_dataset(padded_dataset, resources=resources)

    if dataset_obj.dataset == NULL:
        raise ValueError("padded_dataset is uninitialized")
    if dataset_obj.layout != "padded":
        raise TypeError("padded_dataset must have padded layout")

    cdef cuvsDataset_t dataset_handle = dataset_obj.dataset
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    with cuda_interruptible():
        check_cuvs(cuvsMultiGpuCagraUpdateDataset(
            res,
            dataset_handle,
            index.mg_index
        ))
    return index


cdef class SearchParams(SingleGpuSearchParams):
    """
    Parameters to search multi-GPU CAGRA index.
    """

    def __cinit__(self):
        # Base class __cinit__ has already created self.params
        # We need to destroy it and use our embedded params instead
        if self.params != NULL:
            check_cuvs(cuvsCagraSearchParamsDestroy(self.params))

        # Create multi-GPU search params which includes embedded base params
        check_cuvs(cuvsMultiGpuCagraSearchParamsCreate(&self.mg_params))
        # Replace base pointer with embedded base params
        self.params = self.mg_params.base_params

    def __dealloc__(self):
        # Only destroy the mg_params, which will handle base_params cleanup
        check_cuvs(cuvsMultiGpuCagraSearchParamsDestroy(self.mg_params))
        self.mg_params = NULL
        self.params = <cuvsCagraSearchParams_t>NULL

    def __init__(self, *, search_mode="load_balancer",
                 merge_mode="merge_on_root_rank",
                 n_rows_per_batch=1000, **kwargs):
        super().__init__(**kwargs)
        # Use the property setters for consistent validation
        self.search_mode = search_mode
        self.merge_mode = merge_mode
        self.n_rows_per_batch = n_rows_per_batch

    def get_handle(self):
        return <size_t> self.mg_params

    @property
    def search_mode(self):
        """Get the search mode for multi-GPU search."""
        return ("load_balancer" if self.mg_params.search_mode ==
                CUVS_NEIGHBORS_MG_LOAD_BALANCER else "round_robin")

    @search_mode.setter
    def search_mode(self, value):
        """Set the search mode for multi-GPU search."""
        if value == "load_balancer":
            self.mg_params.search_mode = CUVS_NEIGHBORS_MG_LOAD_BALANCER
        elif value == "round_robin":
            self.mg_params.search_mode = CUVS_NEIGHBORS_MG_ROUND_ROBIN
        else:
            raise ValueError(
                "search_mode must be 'load_balancer' or 'round_robin'")

    @property
    def merge_mode(self):
        """Get the merge mode for multi-GPU search."""
        return ("merge_on_root_rank" if self.mg_params.merge_mode ==
                CUVS_NEIGHBORS_MG_MERGE_ON_ROOT_RANK else "tree_merge")

    @merge_mode.setter
    def merge_mode(self, value):
        """Set the merge mode for multi-GPU search."""
        if value == "merge_on_root_rank":
            self.mg_params.merge_mode = CUVS_NEIGHBORS_MG_MERGE_ON_ROOT_RANK
        elif value == "tree_merge":
            self.mg_params.merge_mode = CUVS_NEIGHBORS_MG_TREE_MERGE
        else:
            raise ValueError(
                "merge_mode must be 'merge_on_root_rank' or 'tree_merge'")

    @property
    def n_rows_per_batch(self):
        """Get the number of rows per batch for multi-GPU search."""
        return self.mg_params.n_rows_per_batch

    @n_rows_per_batch.setter
    def n_rows_per_batch(self, value):
        """Set the number of rows per batch for multi-GPU search."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("n_rows_per_batch must be a positive integer")
        self.mg_params.n_rows_per_batch = value


@auto_sync_multi_gpu_resources
@auto_convert_output
def search(SearchParams search_params, Index index, queries,
           k, neighbors=None, distances=None, resources=None):
    """
    Search the multi-GPU CAGRA index for the k-nearest neighbors of each query.

    Parameters
    ----------
    search_params : :py:class:`cuvs.neighbors.cagra.SearchParams`
    index : :py:class:`cuvs.neighbors.cagra.Index`
    queries : Array interface compliant matrix shape (n_queries, dim)
        Supported dtype [float32, float16, int8, uint8]
        **IMPORTANT**: For multi-GPU CAGRA, queries MUST be in host memory
        (CPU). If using CuPy/device arrays, transfer to host with
        array.get() or cp.asnumpy(array).
    k : int
        The number of neighbors to search for each query.
    neighbors : Array interface compliant matrix shape (n_queries, k), optional
        If provided, this array will be filled with the indices of
        the k-nearest neighbors.
        If not provided, a new host array will be allocated.
        **IMPORTANT**: Must be in host memory (CPU) for multi-GPU CAGRA.
        Expected dtype: int64
    distances : Array interface compliant matrix shape (n_queries, k), optional
        If provided, this array will be filled with the distances
        to the k-nearest neighbors.
        If not provided, a new host array will be allocated.
        **IMPORTANT**: Must be in host memory (CPU) for multi-GPU CAGRA.
    {resources_docstring}

    Returns
    -------
    distances : numpy.ndarray
        The distances to the k-nearest neighbors for each query
        (in host memory).
    neighbors : numpy.ndarray
        The indices of the k-nearest neighbors for each query
        (in host memory).

    Examples
    --------

    >>> import numpy as np
    >>> from cuvs.neighbors.mg import cagra
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> k = 10
    >>> # For multi-GPU CAGRA, use host (NumPy) arrays
    >>> dataset = np.random.random_sample((n_samples, n_features)).astype(
    ...     np.float32)
    >>> queries = np.random.random_sample((n_queries, n_features)).astype(
    ...     np.float32)
    >>> build_params = cagra.IndexParams(metric="sqeuclidean")
    >>> index = cagra.build(build_params, dataset)
    >>> distances, neighbors = cagra.search(cagra.SearchParams(),
    ...                                         index, queries, k)
    >>> # Results are already in host memory (NumPy arrays)
    """

    if not index.trained:
        raise ValueError("Index needs to be built before searching")

    queries_ai = wrap_array(queries)
    _check_input_array(queries_ai, [np.dtype('float32'), np.dtype('float16'),
                                    np.dtype('byte'), np.dtype('ubyte')])

    # Multi-GPU CAGRA requires queries in host memory
    _check_memory_location(queries, expected_host=True, name="queries")

    # Get resources
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    # Prepare output arrays
    cdef uint32_t n_queries = queries.shape[0]
    if neighbors is None:
        # For multi-GPU, create host arrays instead of device arrays
        neighbors = np.empty((n_queries, k), dtype='int64')
    if distances is None:
        # For multi-GPU, create host arrays instead of device arrays
        distances = np.empty((n_queries, k), dtype='float32')

    neighbors_ai = wrap_array(neighbors)
    _check_input_array(neighbors_ai, [np.dtype('int64')],
                       exp_rows=n_queries, exp_cols=k)
    distances_ai = wrap_array(distances)
    _check_input_array(distances_ai, [np.dtype('float32')],
                       exp_rows=n_queries, exp_cols=k)

    # Multi-GPU CAGRA requires output arrays in host memory
    _check_memory_location(neighbors, expected_host=True,
                           name="neighbors")
    _check_memory_location(distances, expected_host=True,
                           name="distances")

    cdef cydlpack.DLManagedTensor* queries_dlpack = (
        cydlpack.dlpack_c(queries_ai))
    cdef cydlpack.DLManagedTensor* neighbors_dlpack = (
        cydlpack.dlpack_c(neighbors_ai))
    cdef cydlpack.DLManagedTensor* distances_dlpack = (
        cydlpack.dlpack_c(distances_ai))

    # Perform search
    with cuda_interruptible():
        check_cuvs(cuvsMultiGpuCagraSearch(
            res, search_params.mg_params, index.mg_index, queries_dlpack,
            neighbors_dlpack, distances_dlpack))

    return (distances, neighbors)


@auto_sync_multi_gpu_resources
def extend(Index index, new_vectors, new_indices=None, resources=None):
    """
    Extend the multi-GPU CAGRA index with new vectors.

    Parameters
    ----------
    index : :py:class:`cuvs.neighbors.cagra.Index`
    new_vectors : Array interface compliant matrix shape (n_new_vectors, dim)
        Supported dtype [float32, float16, int8, uint8]
        **IMPORTANT**: For multi-GPU CAGRA, new_vectors MUST be in host
        memory (CPU). If using CuPy/device arrays, transfer to host with
        array.get() or cp.asnumpy(array).
    new_indices : Array interface compliant matrix shape (n_new_vectors,),
                  optional
        If provided, these indices will be used for the new vectors.
        If not provided, indices will be automatically assigned.
        **IMPORTANT**: Must be in host memory (CPU) for multi-GPU CAGRA.
        Expected dtype: uint32
    {resources_docstring}

    Examples
    --------

    >>> import numpy as np
    >>> from cuvs.neighbors.mg import cagra
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_new_vectors = 1000
    >>> # For multi-GPU CAGRA, use host (NumPy) arrays
    >>> dataset = np.random.random_sample((n_samples, n_features)).astype(
    ...     np.float32)
    >>> new_vectors = np.random.random_sample(
    ...     (n_new_vectors, n_features)).astype(np.float32)
    >>> new_indices = np.arange(n_samples, n_samples + n_new_vectors,
    ...                         dtype=np.uint32)
    >>> build_params = cagra.IndexParams(metric="sqeuclidean")
    >>> index = cagra.build(build_params, dataset)
    >>> cagra.extend(index, new_vectors, new_indices)  # doctest: +SKIP
    """

    if not index.trained:
        raise ValueError("Index needs to be built before extending")

    new_vectors_ai = wrap_array(new_vectors)
    _check_input_array(new_vectors_ai,
                       [np.dtype('float32'), np.dtype('float16'),
                        np.dtype('byte'), np.dtype('ubyte')])

    # Multi-GPU CAGRA requires new_vectors in host memory
    _check_memory_location(new_vectors, expected_host=True, name="new_vectors")

    # Get resources
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef cydlpack.DLManagedTensor* new_vectors_dlpack = \
        cydlpack.dlpack_c(new_vectors_ai)
    cdef cydlpack.DLManagedTensor* new_indices_dlpack = NULL

    if new_indices is not None:
        new_indices_ai = wrap_array(new_indices)
        _check_input_array(new_indices_ai, [np.dtype('uint32')])
        # Multi-GPU CAGRA requires new_indices in host memory
        _check_memory_location(new_indices, expected_host=True,
                               name="new_indices")
        new_indices_dlpack = cydlpack.dlpack_c(new_indices_ai)

    with cuda_interruptible():
        check_cuvs(cuvsMultiGpuCagraExtend(res, index.mg_index,
                                           new_vectors_dlpack,
                                           new_indices_dlpack))


@auto_sync_multi_gpu_resources
def save(Index index, filename, resources=None):
    """
    Serialize the multi-GPU CAGRA index to a file.

    Parameters
    ----------
    index : :py:class:`cuvs.neighbors.cagra.Index`
    filename : str
        The filename to serialize the index to.
    {resources_docstring}

    Examples
    --------

    >>> import numpy as np
    >>> from cuvs.neighbors.mg import cagra
    >>> n_samples = 50000
    >>> n_features = 50
    >>> # For multi-GPU CAGRA, use host (NumPy) arrays
    >>> dataset = np.random.random_sample((n_samples, n_features)).astype(
    ...     np.float32)
    >>> build_params = cagra.IndexParams(metric="sqeuclidean")
    >>> index = cagra.build(build_params, dataset)
    >>> cagra.save(index, "index.bin")
    """

    if not index.trained:
        raise ValueError("Index needs to be built before serializing")

    # Get resources
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef string filename_str = filename.encode('utf-8')
    check_cuvs(cuvsMultiGpuCagraSerialize(
        res, index.mg_index, filename_str.c_str()))


@auto_sync_multi_gpu_resources
def load(filename, resources=None):
    """
    Deserialize the multi-GPU CAGRA index from a file.

    Parameters
    ----------
    filename : str
        The filename to deserialize the index from.
    {resources_docstring}

    Returns
    -------
    index : Index
        The deserialized index.

    Examples
    --------

    >>> from cuvs.neighbors.mg import cagra
    >>> index = cagra.load("index.bin")  # doctest: +SKIP
    """

    cdef Index index = Index()
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef string filename_str = filename.encode('utf-8')
    check_cuvs(cuvsMultiGpuCagraDeserialize(
        res, filename_str.c_str(), index.mg_index))
    index.mg_trained = True

    return index


@auto_sync_multi_gpu_resources
def distribute(filename, resources=None):
    """
    Distribute a single-GPU CAGRA index across multiple GPUs from a file.

    Parameters
    ----------
    filename : str
        The filename to distribute the index from.
    {resources_docstring}

    Returns
    -------
    index : Index
        The distributed index.

    Examples
    --------

    >>> from cuvs.neighbors.mg import cagra
    >>> index = cagra.distribute("single_gpu_index.bin")  # doctest: +SKIP
    """

    cdef Index index = Index()
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef string filename_str = filename.encode('utf-8')
    check_cuvs(cuvsMultiGpuCagraDistribute(
        res, filename_str.c_str(), index.mg_index))
    index.mg_trained = True

    return index
