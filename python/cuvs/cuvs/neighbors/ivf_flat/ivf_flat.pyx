#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

import numpy as np

cimport cuvs.common.cydlpack

from cuvs.common.resources import auto_sync_resources

from cython.operator cimport dereference as deref
from libcpp cimport bool, cast
from libcpp.string cimport string

from cuvs.common cimport cydlpack
from cuvs.distance_type cimport cuvsDistanceType

from pylibraft.common import auto_convert_output, cai_wrapper, device_ndarray
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.interruptible import cuda_interruptible

from cuvs.common.device_tensor_view import DeviceTensorView
from cuvs.distance import DISTANCE_NAMES, DISTANCE_TYPES
from cuvs.neighbors.common import _check_input_array
from cuvs.neighbors.filters import no_filter

from libc.stdint cimport (
    int8_t,
    int64_t,
    uint8_t,
    uint32_t,
    uint64_t,
    uintptr_t,
)

from cuvs.common.exceptions import check_cuvs


cdef class IndexParams:
    """
    Parameters to build index for IvfFlat nearest neighbor search

    Parameters
    ----------
    n_lists : int, default = 1024
        The number of clusters used in the coarse quantizer.
    metric : str, default = "sqeuclidean"
        String denoting the metric type.
        Valid values for metric: ["sqeuclidean", "inner_product",
        "euclidean", "cosine"], where

            - sqeuclidean is the euclidean distance without the square root
              operation, i.e.: distance(a,b) = \\sum_i (a_i - b_i)^2,
            - euclidean is the euclidean distance
            - inner product distance is defined as
              distance(a, b) = \\sum_i a_i * b_i.
            - cosine distance is defined as
              distance(a, b) = 1 - \\sum_i a_i * b_i / ( ||a||_2 * ||b||_2).

    kmeans_n_iters : int, default = 20
        The number of iterations searching for kmeans centers during index
        building.
        The default setting is often fine, but this parameter can be decreased
        to improve training time wih larger trainset fractions (10M+ vectors)
        or increased for smaller trainset fractions (very small number of
        vectors) to improve recall.
    kmeans_trainset_fraction : int, default = 0.5
        If kmeans_trainset_fraction is less than 1, then the dataset is
        subsampled, and only n_samples * kmeans_trainset_fraction rows
        are used for training.
    add_data_on_build : bool, default = True
        After training the coarse and fine quantizers, we will populate
        the index with the dataset if add_data_on_build == True, otherwise
        the index is left empty, and the extend method can be used
        to add new vectors to the index.
    adaptive_centers : bool, default = False
        By default (adaptive_centers = False), the cluster centers are
        trained in `ivf_flat.build`, and and never modified in
        `ivf_flat.extend`. The alternative behavior (adaptive_centers
        = true) is to update the cluster centers for new data when it is
        added. In this case, `index.centers()` are always exactly the
        centroids of the data in the corresponding clusters. The drawback
        of this behavior is that the centroids depend on the order of
        adding new data (through the classification of the added data);
        that is, `index.centers()` "drift" together with the changing
        distribution of the newly added data.
    """

    def __cinit__(self):
        cuvsIvfFlatIndexParamsCreate(&self.params)

    def __dealloc__(self):
        if self.params != NULL:
            check_cuvs(cuvsIvfFlatIndexParamsDestroy(self.params))

    def __init__(self, *,
                 n_lists=1024,
                 metric="sqeuclidean",
                 metric_arg=2.0,
                 kmeans_n_iters=20,
                 kmeans_trainset_fraction=0.5,
                 adaptive_centers=False,
                 add_data_on_build=True,
                 conservative_memory_allocation=False):
        self.params.metric = <cuvsDistanceType>DISTANCE_TYPES[metric]
        self.params.metric_arg = metric_arg
        self.params.add_data_on_build = add_data_on_build
        self.params.n_lists = n_lists
        self.params.kmeans_n_iters = kmeans_n_iters
        self.params.kmeans_trainset_fraction = kmeans_trainset_fraction
        self.params.adaptive_centers = adaptive_centers
        self.params.conservative_memory_allocation = \
            conservative_memory_allocation

    def get_handle(self):
        return <size_t> self.params

    @property
    def metric(self):
        return DISTANCE_NAMES[self.params.metric]

    @property
    def metric_arg(self):
        return self.params.metric_arg

    @property
    def add_data_on_build(self):
        return self.params.add_data_on_build

    @property
    def n_lists(self):
        return self.params.n_lists

    @property
    def kmeans_n_iters(self):
        return self.params.kmeans_n_iters

    @property
    def kmeans_trainset_fraction(self):
        return self.params.kmeans_trainset_fraction

    @property
    def adaptive_centers(self):
        return self.params.adaptive_centers

    @property
    def conservative_memory_allocation(self):
        return self.params.conservative_memory_allocation


cdef class Index:
    """
    IvfFlat index object. This object stores the trained IvfFlat index state
    which can be used to perform nearest neighbors searches.
    """

    cdef cuvsIvfFlatIndex_t index
    cdef bool trained

    def __cinit__(self):
        self.trained = False
        check_cuvs(cuvsIvfFlatIndexCreate(&self.index))

    def __dealloc__(self):
        check_cuvs(cuvsIvfFlatIndexDestroy(self.index))

    @property
    def trained(self):
        return self.trained

    def __repr__(self):
        return "Index(type=IvfFlat)"

    @property
    def n_lists(self):
        """ The number of inverted lists (clusters) """
        cdef int64_t n_lists = 0
        cuvsIvfFlatIndexGetNLists(self.index, &n_lists)
        return n_lists

    @property
    def dim(self):
        """ dimensionality of the cluster centers """
        cdef int64_t dim = 0
        cuvsIvfFlatIndexGetDim(self.index, &dim)
        return dim

    @property
    def centers(self):
        """ Get the cluster centers corresponding to the lists in the
        original space """
        if not self.trained:
            raise ValueError("Index needs to be built before getting centers")

        output = DeviceTensorView()
        cdef cydlpack.DLManagedTensor * tensor = \
            <cydlpack.DLManagedTensor*><size_t>output.get_handle()
        check_cuvs(cuvsIvfFlatIndexGetCenters(self.index, tensor))
        output.parent = self

        return output


@auto_sync_resources
def build(IndexParams index_params, dataset, resources=None):
    """
    Build the IvfFlat index from the dataset for efficient search.

    Parameters
    ----------
    index_params : :py:class:`cuvs.neighbors.ivf_flat.IndexParams`
    dataset : CUDA array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float32, float16, int8, uint8]
    {resources_docstring}

    Returns
    -------
    index: py:class:`cuvs.neighbors.ivf_flat.Index`

    Examples
    --------

    >>> import cupy as cp
    >>> from cuvs.neighbors import ivf_flat
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> k = 10
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> build_params = ivf_flat.IndexParams(metric="sqeuclidean")
    >>> index = ivf_flat.build(build_params, dataset)
    >>> distances, neighbors = ivf_flat.search(ivf_flat.SearchParams(),
    ...                                        index, dataset,
    ...                                        k)
    >>> distances = cp.asarray(distances)
    >>> neighbors = cp.asarray(neighbors)
    """

    dataset_ai = wrap_array(dataset)
    _check_input_array(dataset_ai, [np.dtype('float32'), np.dtype('float16'),
                                    np.dtype('byte'), np.dtype('ubyte')])

    cdef Index idx = Index()
    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_ai)
    cdef cuvsIvfFlatIndexParams* params = index_params.params

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    with cuda_interruptible():
        check_cuvs(cuvsIvfFlatBuild(
            res,
            params,
            dataset_dlpack,
            idx.index
        ))
        idx.trained = True

    return idx


cdef class SearchParams:
    """
    Supplemental parameters to search IVF-Flat index

    Parameters
    ----------
    n_probes: int
        The number of clusters to search.
    """

    def __cinit__(self):
        cuvsIvfFlatSearchParamsCreate(&self.params)

    def __dealloc__(self):
        if self.params != NULL:
            check_cuvs(cuvsIvfFlatSearchParamsDestroy(self.params))

    def __init__(self, *, n_probes=20):
        self.params.n_probes = n_probes

    def get_handle(self):
        return <size_t> self.params

    @property
    def n_probes(self):
        return self.params.n_probes


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
    search_params : py:class:`cuvs.neighbors.ivf_flat.SearchParams`
    index : py:class:`cuvs.neighbors.ivf_flat.Index`
        Trained IvfFlat index.
    queries : CUDA array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float32, float16, int8, uint8]
    k : int
        The number of neighbors.
    neighbors : Optional CUDA array interface compliant matrix shape
                (n_queries, k), dtype int64_t. If supplied, neighbor
                indices will be written here in-place. (default None)
    distances : Optional CUDA array interface compliant matrix shape
                (n_queries, k) If supplied, the distances to the
                neighbors will be written here in-place. (default None)
    filter:     Optional cuvs.neighbors.cuvsFilter can be used to filter
                neighbors based on a given bitset. (default None)
    {resources_docstring}

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.neighbors import ivf_flat
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build the index
    >>> index = ivf_flat.build(ivf_flat.IndexParams(), dataset)
    >>>
    >>> # Search using the built index
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> k = 10
    >>> search_params = ivf_flat.SearchParams(n_probes=20)
    >>>
    >>> distances, neighbors = ivf_flat.search(search_params, index, queries,
    ...                                     k)
    """
    if not index.trained:
        raise ValueError("Index needs to be built before calling search.")

    queries_cai = wrap_array(queries)
    _check_input_array(queries_cai, [np.dtype('float32'), np.dtype('float16'),
                                     np.dtype('byte'), np.dtype('ubyte')])

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

    if filter is None:
        filter = no_filter()

    cdef cuvsIvfFlatSearchParams* params = search_params.params
    cdef cuvsError_t search_status
    cdef cydlpack.DLManagedTensor* queries_dlpack = \
        cydlpack.dlpack_c(queries_cai)
    cdef cydlpack.DLManagedTensor* neighbors_dlpack = \
        cydlpack.dlpack_c(neighbors_cai)
    cdef cydlpack.DLManagedTensor* distances_dlpack = \
        cydlpack.dlpack_c(distances_cai)
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    with cuda_interruptible():
        check_cuvs(cuvsIvfFlatSearch(
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
        Trained IVF-Flat index.
    {resources_docstring}

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.neighbors import ivf_flat
    >>> n_samples = 50000
    >>> n_features = 50
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build index
    >>> index = ivf_flat.build(ivf_flat.IndexParams(), dataset)
    >>> # Serialize and deserialize the ivf_flat index built
    >>> ivf_flat.save("my_index.bin", index)
    >>> index_loaded = ivf_flat.load("my_index.bin")
    """
    cdef string c_filename = filename.encode('utf-8')
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    check_cuvs(cuvsIvfFlatSerialize(res,
                                    c_filename.c_str(),
                                    index.index))


@auto_sync_resources
def load(filename, resources=None):
    """
    Loads index from file.

    Saving / loading the index is experimental. The serialization format is
    subject to change, therefore loading an index saved with a previous
    version of cuvs is not guaranteed to work.

    Parameters
    ----------
    filename : string
        Name of the file.
    {resources_docstring}

    Returns
    -------
    index : Index

    """
    cdef Index idx = Index()
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    cdef string c_filename = filename.encode('utf-8')

    check_cuvs(cuvsIvfFlatDeserialize(
        res,
        c_filename.c_str(),
        idx.index
    ))
    idx.trained = True
    return idx


@auto_sync_resources
def extend(Index index, new_vectors, new_indices, resources=None):
    """
    Extend an existing index with new vectors.

    The input array can be either CUDA array interface compliant matrix or
    array interface compliant matrix in host memory.


    Parameters
    ----------
    index : ivf_flat.Index
        Trained ivf_flat object.
    new_vectors : array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float32, float16, int8, uint8]
    new_indices : array interface compliant vector shape (n_samples)
        Supported dtype [int64]
    {resources_docstring}

    Returns
    -------
    index: py:class:`cuvs.neighbors.ivf_flat.Index`

    Examples
    --------

    >>> import cupy as cp
    >>> from cuvs.neighbors import ivf_flat
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> index = ivf_flat.build(ivf_flat.IndexParams(), dataset)
    >>> n_rows = 100
    >>> more_data = cp.random.random_sample((n_rows, n_features),
    ...                                     dtype=cp.float32)
    >>> indices = n_samples + cp.arange(n_rows, dtype=cp.int64)
    >>> index = ivf_flat.extend(index, more_data, indices)
    >>> # Search using the built index
    >>> queries = cp.random.random_sample((n_queries, n_features),
    ...                                   dtype=cp.float32)
    >>> distances, neighbors = ivf_flat.search(ivf_flat.SearchParams(),
    ...                                      index, queries,
    ...                                      k=10)
    """

    new_vectors_ai = wrap_array(new_vectors)
    _check_input_array(new_vectors_ai,
                       [np.dtype('float32'), np.dtype('float16'),
                        np.dtype('byte'), np.dtype('ubyte')])

    new_indices_ai = wrap_array(new_indices)
    _check_input_array(new_indices_ai, [np.dtype('int64')])
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef cydlpack.DLManagedTensor* new_vectors_dlpack = \
        cydlpack.dlpack_c(new_vectors_ai)

    cdef cydlpack.DLManagedTensor* new_indices_dlpack = \
        cydlpack.dlpack_c(new_indices_ai)

    with cuda_interruptible():
        check_cuvs(cuvsIvfFlatExtend(
            res,
            new_vectors_dlpack,
            new_indices_dlpack,
            index.index
        ))

    return index
