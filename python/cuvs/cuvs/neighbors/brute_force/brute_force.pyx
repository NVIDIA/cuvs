#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

import cupy as cp
import numpy as np

cimport cuvs.common.cydlpack

from cuvs.common.resources import auto_sync_resources

from cython.operator cimport dereference as deref
from libc.stdint cimport uint32_t
from libcpp cimport bool
from libcpp.string cimport string

from cuvs.common cimport cydlpack
from cuvs.distance_type cimport cuvsDistanceType

from pylibraft.common import auto_convert_output, cai_wrapper, device_ndarray
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.interruptible import cuda_interruptible

from cuvs.distance import DISTANCE_TYPES
from cuvs.neighbors.common import _check_input_array

from cuvs.common.c_api cimport cuvsResources_t

from cuvs.common.exceptions import check_cuvs
from cuvs.neighbors.filters import no_filter


cdef class Index:
    """
    Brute Force index object. This object stores the trained Brute Force
    which can be used to perform nearest neighbors searches.
    """

    cdef cuvsBruteForceIndex_t index
    cdef bool trained
    cdef object dataset
    cdef object metric
    cdef float metric_arg

    def __cinit__(self):
        self.trained = False
        check_cuvs(cuvsBruteForceIndexCreate(&self.index))

    def __dealloc__(self):
        if self.index is not NULL:
            check_cuvs(cuvsBruteForceIndexDestroy(self.index))

    @property
    def trained(self):
        return self.trained

    def __repr__(self):
        return "Index(type=BruteForce)"


@auto_sync_resources
def build(dataset, metric="sqeuclidean", metric_arg=2.0, resources=None):
    """
    Build the Brute Force index from the dataset for efficient search.

    Parameters
    ----------
    dataset : CUDA array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float32, float16]
    metric : Distance metric to use. Default is sqeuclidean
    metric_arg : value of 'p' for Minkowski distances
    {resources_docstring}

    Returns
    -------
    index: cuvs.neighbors.brute_force.Index

    Examples
    --------

    >>> import cupy as cp
    >>> from cuvs.neighbors import brute_force
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> k = 10
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> index = brute_force.build(dataset, metric="cosine")
    >>> distances, neighbors = brute_force.search(index, dataset, k)
    >>> distances = cp.asarray(distances)
    >>> neighbors = cp.asarray(neighbors)
    """

    dataset_ai = wrap_array(dataset)
    _check_input_array(dataset_ai,
                       [np.dtype('float32'), np.dtype('float16')],
                       exp_row_major=False)

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef cuvsDistanceType c_metric = <cuvsDistanceType>DISTANCE_TYPES[metric]
    cdef Index idx = Index()
    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_ai)

    with cuda_interruptible():
        check_cuvs(cuvsBruteForceBuild(
            res,
            dataset_dlpack,
            c_metric,
            <float>metric_arg,
            idx.index
        ))
        idx.trained = True
        idx.dataset = dataset
        idx.metric = metric
        idx.metric_arg = metric_arg

    return idx


def _maybe_apply_two_pass_precision(queries, distances, neighbors,
                                    dataset, metric, metric_arg):
    if not (
        metric == "euclidean" or
        metric == "sqeuclidean" or
        metric == "l2" or
        (metric == "minkowski" and metric_arg == 2.0) or
        (metric == "lp" and metric_arg == 2.0)
    ):
        return distances, neighbors

    # Wrap arrays in cupy
    queries_cp = cp.asarray(queries)
    neighbors_indices_cp = cp.asarray(neighbors)
    dataset_cp = cp.asarray(dataset)

    # Get the neighbor vectors
    # neighbors_indices_cp has shape (n_queries, k)
    # dataset_cp has shape (n_samples, dim)
    # neighbor_vectors will have shape (n_queries, k, dim)
    neighbor_vectors = dataset_cp[neighbors_indices_cp]

    # Calculate exact L2 distances
    # queries_cp[:, cp.newaxis, :] has shape (n_queries, 1, dim)
    diff = neighbor_vectors - queries_cp[:, cp.newaxis, :]
    new_distances_cp = cp.sum(diff * diff, axis=2)

    if metric == "euclidean" or metric == "l2":
        new_distances_cp = cp.sqrt(new_distances_cp)

    # Re-sort
    sort_indices = cp.argsort(new_distances_cp, axis=1)
    new_distances_cp = cp.take_along_axis(new_distances_cp, sort_indices, axis=1)
    neighbors_indices_cp = cp.take_along_axis(neighbors_indices_cp,
                                              sort_indices, axis=1)

    # Update in-place
    # Since these are device_ndarray (CAI), we can use cupy to copy data back
    distances_out = cp.asarray(distances)
    neighbors_out = cp.asarray(neighbors)

    distances_out[:] = new_distances_cp
    neighbors_out[:] = neighbors_indices_cp

    return distances, neighbors


@auto_sync_resources
@auto_convert_output
def search(Index index,
           queries,
           k,
           neighbors=None,
           distances=None,
           resources=None,
           prefilter=None,
           two_pass_precision=False):
    """
    Find the k nearest neighbors for each query.

    Parameters
    ----------
    index : Index
        Trained Brute Force index.
    queries : CUDA array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float32, float16]
    k : int
        The number of neighbors.
    neighbors : Optional CUDA array interface compliant matrix shape
                (n_queries, k), dtype int64_t. If supplied, neighbor
                indices will be written here in-place. (default None)
    distances : Optional CUDA array interface compliant matrix shape
                (n_queries, k) If supplied, the distances to the
                neighbors will be written here in-place. (default None)
    prefilter : Optional, cuvs.neighbors.cuvsFilter
                An optional filter to exclude certain query-neighbor
                pairs using a bitmap or bitset. The filter function should
                have a row-major layout with logical shape
                `(n_prefilter_rows, n_samples)`, where:
                - `n_prefilter_rows == n_queries` when using a bitmap filter.
                - `n_prefilter_rows == 1` when using a bitset prefilter.
                Each bit in `n_samples` determines whether `queries[i]`
                should be considered for distance computation with the index.
                (default None)
    two_pass_precision : bool, optional (default = False)
                When set to True, a slow second pass will be used to improve
                the precision of results returned for searches using
                L2-derived metrics.
    {resources_docstring}

    Examples
    --------

    >>> # Example without pre-filter
    >>> import cupy as cp
    >>> from cuvs.neighbors import brute_force
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build index
    >>> index = brute_force.build(dataset, metric="sqeuclidean")
    >>> # Search using the built index
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> k = 10
    >>> # Using a pooling allocator reduces overhead of temporary array
    >>> # creation during search. This is useful if multiple searches
    >>> # are performed with same query size.
    >>> distances, neighbors = brute_force.search(index, queries, k)
    >>> neighbors = cp.asarray(neighbors)
    >>> distances = cp.asarray(distances)

    >>> # Example with pre-filter
    >>> import numpy as np
    >>> import cupy as cp
    >>> from cuvs.neighbors import brute_force, filters
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build index
    >>> index = brute_force.build(dataset, metric="sqeuclidean")
    >>> # Search using the built index
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build filters
    >>> n_bitmap = np.ceil(n_samples * n_queries / 32).astype(int)
    >>> # Create your own bitmap as the filter by replacing the random one.
    >>> bitmap = cp.random.randint(1, 100, size=(n_bitmap,), dtype=cp.uint32)
    >>> bitmap_prefilter = filters.from_bitmap(bitmap)
    >>>
    >>> # or Build bitset prefilter:
    >>> # n_bitset = np.ceil(n_samples * 1 / 32).astype(int)
    >>> # # Create your own bitset as the filter by replacing the random one.
    >>> # bitset = cp.random.randint(1, 100, size=(n_bitset,), dtype=cp.uint32)
    >>> # bitset_prefilter = filters.from_bitset(bitset)
    >>>
    >>> k = 10
    >>> # Using a pooling allocator reduces overhead of temporary array
    >>> # creation during search. This is useful if multiple searches
    >>> # are performed with same query size.
    >>> distances, neighbors = brute_force.search(index, queries, k,
    ...                                           prefilter=bitmap_prefilter)
    >>> neighbors = cp.asarray(neighbors)
    >>> distances = cp.asarray(distances)

    """
    if not index.trained:
        raise ValueError("Index needs to be built before calling search.")

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    queries_cai = wrap_array(queries)
    _check_input_array(queries_cai,
                       [np.dtype('float32'), np.dtype('float16')],
                       exp_row_major=False)

    cdef uint32_t n_queries = queries_cai.shape[0]

    if neighbors is None:
        neighbors = device_ndarray.empty((n_queries, k), dtype='int64')

    neighbors_cai = wrap_array(neighbors)
    _check_input_array(neighbors_cai, [np.dtype('int64')],
                       exp_rows=n_queries, exp_cols=k)

    if distances is None:
        distances = device_ndarray.empty((n_queries, k), dtype='float32')

    distances_cai = wrap_array(distances)
    _check_input_array(distances_cai, [np.dtype('float32')],
                       exp_rows=n_queries, exp_cols=k)

    cdef cydlpack.DLManagedTensor* queries_dlpack = \
        cydlpack.dlpack_c(queries_cai)
    cdef cydlpack.DLManagedTensor* neighbors_dlpack = \
        cydlpack.dlpack_c(neighbors_cai)
    cdef cydlpack.DLManagedTensor* distances_dlpack = \
        cydlpack.dlpack_c(distances_cai)

    if prefilter is None:
        prefilter = no_filter()

    with cuda_interruptible():
        check_cuvs(cuvsBruteForceSearch(
            res,
            index.index,
            queries_dlpack,
            neighbors_dlpack,
            distances_dlpack,
            prefilter.prefilter
        ))

    if two_pass_precision:
        if index.dataset is None:
            raise ValueError("two_pass_precision requires the dataset to be "
                             "available in the Index object. This is "
                             "currently only supported if the index was "
                             "built in this session.")
        distances, neighbors = _maybe_apply_two_pass_precision(
            queries, distances, neighbors,
            index.dataset, index.metric, index.metric_arg
        )

    return (distances, neighbors)


@auto_sync_resources
def save(filename, Index index, bool include_dataset=True, resources=None):
    """
    Saves the index to a file.

    The serialization format can be subject to changes, therefore loading
    an index saved with a previous version of cuvs is not guaranteed
    to work.

    Parameters
    ----------
    filename : string
        Name of the file.
    index : Index
        Trained Brute Force index.
    {resources_docstring}

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.neighbors import brute_force
    >>> n_samples = 50000
    >>> n_features = 50
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build index
    >>> index = brute_force.build(dataset)
    >>> # Serialize and deserialize the brute_force index built
    >>> brute_force.save("my_index.bin", index)
    >>> index_loaded = brute_force.load("my_index.bin")
    """
    cdef string c_filename = filename.encode('utf-8')
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    check_cuvs(cuvsBruteForceSerialize(res,
                                       c_filename.c_str(),
                                       index.index))


@auto_sync_resources
def load(filename, resources=None):
    """
    Loads index from file.

    The serialization format can be subject to changes, therefore loading
    an index saved with a previous version of cuvs is not guaranteed
    to work.


    Parameters
    ----------
    filename : string
        Name of the file.
    {resources_docstring}

    Returns
    -------
    index : Index

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.neighbors import brute_force
    >>> n_samples = 50000
    >>> n_features = 50
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build index
    >>> index = brute_force.build(dataset)
    >>> # Serialize and deserialize the brute_force index built
    >>> brute_force.save("my_index.bin", index)
    >>> index_loaded = brute_force.load("my_index.bin")
    """
    cdef Index idx = Index()
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    cdef string c_filename = filename.encode('utf-8')

    check_cuvs(cuvsBruteForceDeserialize(
        res,
        c_filename.c_str(),
        idx.index
    ))
    idx.trained = True
    return idx
