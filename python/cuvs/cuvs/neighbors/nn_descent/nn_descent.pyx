#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from cuvs.distance import DISTANCE_NAMES, DISTANCE_TYPES
from cuvs.neighbors.common import _check_input_array

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
    Parameters to build NN-Descent Index

    Parameters
    ----------
    metric : str, default = "sqeuclidean"
        String denoting the metric type.
        Supported metrics are `l2`, `euclidean`, `sqeuclidean`,
        `inner_product`, `cosine`, and `bitwise_hamming`
        (`bitwise_hamming` is for int8 and uint8 data types only)
    graph_degree :  int
        For an input dataset of dimensions (N, D), determines the final
        dimensions of the all-neighbors knn graph which turns out to be of
        dimensions (N, graph_degree)
    intermediate_graph_degree : int
        Internally, nn-descent builds an all-neighbors knn graph of dimensions
        (N, intermediate_graph_degree) before selecting the final
        `graph_degree` neighbors. It's recommended that
        `intermediate_graph_degree` >= 1.5 * graph_degree
    max_iterations : int
        The number of iterations that nn-descent will refine the graph for.
        More iterations produce a better quality graph at cost of performance
    termination_threshold : float, default = 0.001
        NN-descent terminates when the number of graph updates in an iteration
        is less than ``termination_threshold * graph_degree * dataset_size``.
        Set to ``0.0`` to disable early termination.
    return_distances : bool
        Whether to return distances array
    dist_comp_dtype : str, default = "auto"
        Dtype to use for distance computation.
        Supported dtypes are `auto`, `fp32`, and `fp16`
        `auto` automatically determines the best dtype for distance computation based on the dataset dimensions.
        `fp32` uses fp32 distance computation for better precision at the cost of performance and memory usage. This option is only valid when data type is fp32.
        `fp16` uses fp16 distance computation for better performance and memory usage at the cost of precision.
    """

    cdef cuvsNNDescentIndexParams* params
    cdef object _metric

    def __cinit__(self):
        cuvsNNDescentIndexParamsCreate(&self.params)

    def __dealloc__(self):
        check_cuvs(cuvsNNDescentIndexParamsDestroy(self.params))

    def __init__(self, *,
                 metric=None,
                 metric_arg=None,
                 graph_degree=None,
                 intermediate_graph_degree=None,
                 max_iterations=None,
                 termination_threshold=None,
                 return_distances=None,
                 dist_comp_dtype="auto"
                 ):
        if metric is not None:
            self.params.metric = <cuvsDistanceType>DISTANCE_TYPES[metric]
        if graph_degree is not None:
            self.params.graph_degree = graph_degree
        if intermediate_graph_degree is not None:
            self.params.intermediate_graph_degree = intermediate_graph_degree
        if max_iterations is not None:
            self.params.max_iterations = max_iterations
        if termination_threshold is not None:
            self.params.termination_threshold = termination_threshold
        if return_distances is not None:
            self.params.return_distances = return_distances

        if dist_comp_dtype is "auto":
            self.params.dist_comp_dtype = cuvsNNDescentDistCompDtype.NND_DIST_COMP_AUTO
        elif dist_comp_dtype is "fp32":
            self.params.dist_comp_dtype = cuvsNNDescentDistCompDtype.NND_DIST_COMP_FP32
        elif dist_comp_dtype is "fp16":
            self.params.dist_comp_dtype = cuvsNNDescentDistCompDtype.NND_DIST_COMP_FP16
        else:
            raise ValueError(f"Invalid dist_comp_dtype: {dist_comp_dtype}. Supported options are 'auto', 'fp32', and 'fp16'.")

    @property
    def metric(self):
        return DISTANCE_NAMES[self.params.metric]

    @property
    def metric_arg(self):
        return self.params.metric_arg

    @property
    def graph_degree(self):
        return self.params.graph_degree

    @property
    def intermediate_graph_degree(self):
        return self.params.intermediate_graph_degree

    @property
    def max_iterations(self):
        return self.params.max_iterations

    @property
    def termination_threshold(self):
        return self.params.termination_threshold

    def get_handle(self):
        """Get a pointer to the underlying C object."""
        return <size_t>self.params

cdef class Index:
    """
    NN-Descent index object. This object stores the trained NN-Descent index,
    which can be used to get the NN-Descent graph and distances after
    building
    """

    cdef cuvsNNDescentIndex_t index
    cdef bool trained
    cdef int64_t num_rows
    cdef size_t graph_degree

    def __cinit__(self):
        self.trained = False
        self.num_rows = 0
        self.graph_degree = 0
        check_cuvs(cuvsNNDescentIndexCreate(&self.index))

    def __dealloc__(self):
        check_cuvs(cuvsNNDescentIndexDestroy(self.index))

    @property
    def trained(self):
        return self.trained

    @property
    def graph(self):
        return self._get_graph()

    @property
    def distances(self):
        return self._get_distances()

    @auto_sync_resources
    def _get_graph(self, resources=None):
        if not self.trained:
            raise ValueError("Index needs to be built before getting graph")

        cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

        output = np.empty((self.num_rows, self.graph_degree), dtype='uint32')
        ai = wrap_array(output)
        cdef cydlpack.DLManagedTensor* output_dlpack = cydlpack.dlpack_c(ai)
        check_cuvs(cuvsNNDescentIndexGetGraph(res, self.index, output_dlpack))
        return output

    @auto_sync_resources
    def _get_distances(self, resources=None):
        if not self.trained:
            raise ValueError("Index needs to be built before getting"
                             " distances")

        cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

        output = np.empty((self.num_rows, self.graph_degree), dtype='float32')
        ai = wrap_array(output)
        cdef cydlpack.DLManagedTensor* output_dlpack = cydlpack.dlpack_c(ai)
        check_cuvs(cuvsNNDescentIndexGetDistances(res,
                                                  self.index,
                                                  output_dlpack))
        return output

    def __repr__(self):
        return "Index(type=NNDescent)"


@auto_sync_resources
def build(IndexParams index_params, dataset, graph=None, resources=None):
    """
    Build KNN graph from the dataset

    Parameters
    ----------
    index_params : :py:class:`cuvs.neighbors.nn_descent.IndexParams`
    dataset : Array interface compliant matrix, on either host or device memory
        Supported dtype [float, int8, uint8]
    graph : Optional host matrix for storing output graph
    {resources_docstring}

    Returns
    -------
    index: py:class:`cuvs.neighbors.nn_descent.Index`

    Examples
    --------

    >>> import cupy as cp
    >>> from cuvs.neighbors import nn_descent
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> k = 10
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> build_params = nn_descent.IndexParams(metric="sqeuclidean")
    >>> index = nn_descent.build(build_params, dataset)
    >>> graph = index.graph
    """
    dataset_ai = wrap_array(dataset)
    _check_input_array(dataset_ai, [np.dtype('float32'), np.dtype('float16'), np.dtype('byte'),
                                    np.dtype('ubyte')])

    cdef Index idx = Index()
    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_ai)
    cdef cuvsNNDescentIndexParams* params = index_params.params

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef cydlpack.DLManagedTensor* graph_dlpack = NULL
    if graph is not None:
        if params.return_distances:
            # When using a pre-existing graph - having return_distances set to
            # true will cause an exception in the C++ api
            # (`Using return_distances set to true requires distance view to
            # be allocated.`). Raise a more informative error here instead of
            # the C++ exception
            raise ValueError("Can't use return_distances with an existing"
                             " graph. Either set params.return_distances to"
                             " False, or set graph to None")

        graph_ai = wrap_array(graph)
        graph_dlpack = cydlpack.dlpack_c(graph_ai)

    with cuda_interruptible():
        check_cuvs(cuvsNNDescentBuild(
            res,
            params,
            dataset_dlpack,
            graph_dlpack,
            idx.index
        ))
        idx.trained = True
        idx.num_rows = dataset_ai.shape[0]
        idx.graph_degree = params.graph_degree

    return idx
