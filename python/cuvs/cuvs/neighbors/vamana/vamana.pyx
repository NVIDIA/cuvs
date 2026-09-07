#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

import warnings

import numpy as np

cimport cuvs.common.cydlpack

from cuvs.common.resources import auto_sync_resources

from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.string cimport string

from cuvs.common cimport cydlpack
from cuvs.distance_type cimport cuvsDistanceType

from pylibraft.common import auto_convert_output, cai_wrapper, device_ndarray
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.interruptible import cuda_interruptible

from cuvs.distance import DISTANCE_NAMES, DISTANCE_TYPES
from cuvs.neighbors.common import _check_input_array

from libc.stdint cimport uint32_t, uintptr_t

from cuvs.common.exceptions import check_cuvs


cdef class Index:
    """
    Vamana index object. This object stores the trained Vamana index state
    which can be used to perform nearest neighbors searches.
    """

    def __cinit__(self):
        self.trained = False
        check_cuvs(cuvsVamanaIndexCreate(&self.index))

    def __dealloc__(self):
        if self.index is not NULL:
            check_cuvs(cuvsVamanaIndexDestroy(self.index))

    @property
    def trained(self):
        return self.trained

    def __repr__(self):
        attr_str = [attr + "=" + str(getattr(self, attr))
                    for attr in ["trained"]]
        return "Index(type=Vamana, " + (", ".join(attr_str)) + ")"


cdef class IndexParams:
    """
    Parameters for building a Vamana index

    Parameters
    ----------
    metric : str, default="sqeuclidean"
        String denoting the metric type. Supported metrics include:
        - "sqeuclidean"
        - "l2"
    graph_degree : int, default=32
        Maximum degree of graph; corresponds to the R parameter of
        Vamana algorithm in the literature.
    visited_size : int, default=64
        Maximum number of visited nodes per search during Vamana algorithm.
        Loosely corresponds to the L parameter in the literature.
    vamana_iters : float, default=1
        Number of Vamana vector insertion iterations (each iteration inserts
        all vectors).
    alpha : float, default=1.2
        Alpha for pruning parameter. Used to determine how aggressive the
        pruning will be.
    max_fraction : float, default=0.06
        Maximum fraction of dataset inserted per batch. Larger max batch
        decreases graph quality, but improves speed.
    batch_base : float, default=2.0
        Base of growth rate of batch sizes.
    queue_size : int, default=127
        Size of candidate queue structure - should be (2^x)-1.
    reverse_batchsize : int, default=1000000
        Max batchsize of reverse edge processing (reduces memory footprint).
    """

    def __cinit__(self):
        check_cuvs(cuvsVamanaIndexParamsCreate(&self.params))

    def __dealloc__(self):
        check_cuvs(cuvsVamanaIndexParamsDestroy(self.params))

    def __init__(self, *,
                 metric="sqeuclidean",
                 graph_degree=32,
                 visited_size=64,
                 vamana_iters=1,
                 alpha=1.2,
                 max_fraction=0.06,
                 batch_base=2.0,
                 queue_size=127,
                 reverse_batchsize=1000000):
        if metric in DISTANCE_TYPES:
            self.params.metric = <cuvsDistanceType>DISTANCE_TYPES[metric]
        else:
            raise ValueError("metric %s not supported" % metric)

        self.params.graph_degree = graph_degree
        self.params.visited_size = visited_size
        self.params.vamana_iters = vamana_iters
        self.params.alpha = alpha
        self.params.max_fraction = max_fraction
        self.params.batch_base = batch_base
        self.params.queue_size = queue_size
        self.params.reverse_batchsize = reverse_batchsize

    @property
    def metric(self):
        return DISTANCE_NAMES[self.params.metric]

    @property
    def graph_degree(self):
        return self.params.graph_degree

    @property
    def visited_size(self):
        return self.params.visited_size

    @property
    def vamana_iters(self):
        return self.params.vamana_iters

    @property
    def alpha(self):
        return self.params.alpha

    @property
    def max_fraction(self):
        return self.params.max_fraction

    @property
    def batch_base(self):
        return self.params.batch_base

    @property
    def queue_size(self):
        return self.params.queue_size

    @property
    def reverse_batchsize(self):
        return self.params.reverse_batchsize


@auto_sync_resources
def build(IndexParams index_params, dataset, resources=None):
    """
    Build the Vamana index from the dataset for efficient search.

    The build utilities the Vamana insertion-based algorithm to create
    the graph. The algorithm starts with an empty graph and iteratively
    inserts batches of nodes. Each batch involves performing a greedy
    search for each vector to be inserted, and inserting it with edges to
    all nodes traversed during the search. Reverse edges are also inserted
    and robustPrune is applied to improve graph quality. The index_params
    struct controls the degree of the final graph.

    The following distance metrics are supported:
        - L2Expanded (sqeuclidean)
        - L2SqrtExpanded (l2 / euclidean distance)

    Parameters
    ----------
    index_params : IndexParams object
    dataset : CUDA array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float, float16, int8, uint8]
    {resources_docstring}

    Returns
    -------
    index: cuvs.vamana.Index

    Examples
    --------

    >>> import cupy as cp
    >>> from cuvs.neighbors import vamana
    >>> n_samples = 50000
    >>> n_features = 50
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> build_params = vamana.IndexParams(metric="sqeuclidean")
    >>> index = vamana.build(build_params, dataset)
    >>> # Serialize index to file for later use with CPU DiskANN
    >>> vamana.save("my_index.bin", index)
    """

    # todo(dgd): we can make the check of dtype a parameter of wrap_array
    # in RAFT to make this a single call
    dataset_ai = wrap_array(dataset)
    _check_input_array(dataset_ai, [np.dtype('float32'),
                                    np.dtype('float16'),
                                    np.dtype('int8'),
                                    np.dtype('uint8')])

    cdef Index idx = Index()
    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_ai)
    cdef cuvsVamanaIndexParams* params = index_params.params

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    with cuda_interruptible():
        check_cuvs(cuvsVamanaBuild(
            res,
            params,
            dataset_dlpack,
            idx.index
        ))
        idx.trained = True
        idx.active_index_type = dataset_ai.dtype.name

    return idx


@auto_sync_resources
def save(filename, Index index, bool include_dataset=True, resources=None):
    """
    Saves the index to a file.

    Matches the file format used by the DiskANN open-source repository,
    allowing cross-compatibility.

    Parameters
    ----------
    filename : string
        Name of the file.
    index : Index
        Trained Vamana index.
    include_dataset : bool
        Whether or not to write out the dataset along with the index. Including
        the dataset in the serialized index will use extra disk space, and
        might not be desired if you already have a copy of the dataset on
        disk.
    {resources_docstring}

    Examples
    --------
    >>> import cupy as cp
    >>> from cuvs.neighbors import vamana
    >>> n_samples = 50000
    >>> n_features = 50
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build index
    >>> index = vamana.build(vamana.IndexParams(), dataset)
    >>> # Serialize and save the vamana index
    >>> vamana.save("my_index.bin", index)
    """
    cdef string c_filename = filename.encode('utf-8')
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    check_cuvs(cuvsVamanaSerialize(res,
                                   c_filename.c_str(),
                                   index.index,
                                   include_dataset))


# Note: Vamana index currently only supports build and serialize operations.
# Search functionality is not yet implemented in cuVS and should be performed
# using the DiskANN library with the serialized index.
