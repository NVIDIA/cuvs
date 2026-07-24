#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

import warnings
from collections import namedtuple

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

INIT_METHOD_TYPES = {
    "KMeansPlusPlus" : cuvsKMeansInitMethod.KMeansPlusPlus,
    "Random" : cuvsKMeansInitMethod.Random,
    "Array" : cuvsKMeansInitMethod.Array}

INIT_METHOD_NAMES = {v: k for k, v in INIT_METHOD_TYPES.items()}

cdef class KMeansParams:
    """
    Hyper-parameters for the kmeans algorithm

    Parameters
    ----------
    metric : str
        String denoting the metric type.
    n_clusters : int
        The number of clusters to form as well as the number of centroids
        to generate
    init_method : str
        Method for initializing clusters. One of:
        "KMeansPlusPlus" : Use scalable k-means++ algorithm to select initial
        cluster centers
        "Random" : Choose 'n_clusters' observations at random from the input
        data
        "Array" : Use centroids as initial cluster centers
    max_iter : int
        Maximum number of iterations of the k-means algorithm for a single run
    tol : float
        Relative tolerance with regards to inertia to declare convergence.
    n_init : int
        Number of instance k-means algorithm will be run with different seeds
    oversampling_factor : double
        Oversampling factor for use in the k-means|| algorithm
    batch_samples : int
        Number of samples to process in each batch for tiled 1NN computation.
        Useful to optimize/control memory footprint. Default tile is
        [batch_samples x n_clusters].
    batch_centroids : int
        Number of centroids to process in each batch. If 0, uses n_clusters.
    inertia_check : bool
        Deprecated and ignored. Will be
        removed in a future release. Inertia-based convergence checking
        always runs.
    init_size : int
        Number of samples to draw for KMeansPlusPlus initialization with
        host (out-of-core) data. When set to 0, uses the heuristic
        min(3 * n_clusters, n_samples). Default: 0.
    device_buffer_samples : int
        Number of samples to process per GPU batch when fitting with host
        (numpy) data. When set to 0, defaults to n_samples (process all
        at once). Only used by the batched (host-data) code path. Reducing
        device_buffer_samples can help reduce GPU memory pressure but increases
        overhead as the number of times centroid adjustments are computed
        increases.

        Default: 0 (process all data at once).
    hierarchical : bool
        Whether to use hierarchical (balanced) kmeans or not
    hierarchical_n_iters : int
        For hierarchical k-means , defines the number of training iterations
    """

    def __cinit__(self):
        cuvsKMeansParamsCreate(&self.params)

    def __dealloc__(self):
        check_cuvs(cuvsKMeansParamsDestroy(self.params))

    def __init__(self, *,
                 metric=None,
                 n_clusters=None,
                 init_method=None,
                 max_iter=None,
                 tol=None,
                 n_init=None,
                 oversampling_factor=None,
                 batch_samples=None,
                 batch_centroids=None,
                 inertia_check=None,
                 init_size=None,
                 device_buffer_samples=None,
                 hierarchical=None,
                 hierarchical_n_iters=None):
        if metric is not None:
            self.params.metric = <cuvsDistanceType>DISTANCE_TYPES[metric]
        if n_clusters is not None:
            self.params.n_clusters = n_clusters
        if init_method is not None:
            c_method = INIT_METHOD_TYPES[init_method]
            self.params.init = <cuvsKMeansInitMethod>c_method
        if max_iter is not None:
            self.params.max_iter = max_iter
        if tol is not None:
            self.params.tol = tol
        if n_init is not None:
            self.params.n_init = n_init
        if oversampling_factor is not None:
            self.params.oversampling_factor = oversampling_factor
        if batch_samples is not None:
            self.params.batch_samples = batch_samples
        if batch_centroids is not None:
            self.params.batch_centroids = batch_centroids
        if inertia_check is not None:
            warnings.warn(
                "KMeansParams `inertia_check` is deprecated and ignored; "
                "inertia-based convergence checking always runs.",
                FutureWarning
            )
        if init_size is not None:
            self.params.init_size = init_size
        if device_buffer_samples is not None:
            self.params.device_buffer_samples = device_buffer_samples
        if hierarchical is not None:
            self.params.hierarchical = hierarchical
        if hierarchical_n_iters is not None:
            if not self.params.hierarchical:
                raise ValueError("Setting hierarchical_n_iters requires"
                                 " `hierarchical` to be also set to True")
            self.params.hierarchical_n_iters = hierarchical_n_iters

    @property
    def metric(self):
        return DISTANCE_NAMES[self.params.metric]

    @property
    def n_clusters(self):
        return self.params.n_clusters

    @property
    def init_method(self):
        return INIT_METHOD_NAMES[self.params.init]

    @property
    def max_iter(self):
        return self.params.max_iter

    @property
    def tol(self):
        return self.params.tol

    @property
    def n_init(self):
        return self.params.n_init

    @property
    def oversampling_factor(self):
        return self.params.oversampling_factor

    @property
    def batch_samples(self):
        return self.params.batch_samples

    @property
    def batch_centroids(self):
        return self.params.batch_centroids

    @property
    def init_size(self):
        return self.params.init_size

    @property
    def device_buffer_samples(self):
        return self.params.device_buffer_samples

    @property
    def hierarchical(self):
        return self.params.hierarchical

    @property
    def hierarchical_n_iters(self):
        return self.params.hierarchical_n_iters


FitOutput = namedtuple("FitOutput", "centroids inertia n_iter")


@auto_sync_resources
@auto_convert_output
def fit(
    KMeansParams params, X, centroids=None, sample_weights=None, resources=None
):
    """
    Find clusters with the k-means algorithm

    When X is a device array (CUDA array interface), standard on-device
    k-means is used.  When X is a host array (numpy ndarray or
    ``__array_interface__``), data is streamed to the GPU in batches
    controlled by ``params.device_buffer_samples``. For large host datasets, consider
    reducing ``device_buffer_samples`` to reduce GPU memory usage.

    Parameters
    ----------

    params : KMeansParams
        Parameters to use to fit KMeans model.  For host data,
        ``params.device_buffer_samples`` controls how many samples are sent to the
        GPU per batch.
    X : array-like
        Training instances, shape (m, k).  Accepts both device arrays
        (cupy / CUDA array interface) and host arrays (numpy).
    centroids : Optional writable CUDA array interface compliant matrix
                shape (n_clusters, k)
    sample_weights : Optional weights per observation.  Must reside on
                     the same memory space as X (device or host).
                     default: None
    {resources_docstring}

    Returns
    -------
    centroids : raft.device_ndarray
        The computed centroids for each cluster
    inertia : float
       Sum of squared distances of samples to their closest cluster center
    n_iter : int
        The number of iterations used to fit the model

    Examples
    --------

    >>> import cupy as cp
    >>>
    >>> from cuvs.cluster.kmeans import fit, KMeansParams
    >>>
    >>> n_samples = 5000
    >>> n_features = 50
    >>> n_clusters = 3
    >>>
    >>> X = cp.random.random_sample((n_samples, n_features),
    ...                             dtype=cp.float32)

    >>> params = KMeansParams(n_clusters=n_clusters)
    >>> centroids, inertia, n_iter = fit(params, X)

    Host-data (batched) example:

    >>> import numpy as np
    >>> X_host = np.random.random((10_000_000, 128)).astype(np.float32)
    >>> params = KMeansParams(n_clusters=1000, device_buffer_samples=1_000_000)
    >>> centroids, inertia, n_iter = fit(params, X_host)
    """

    is_host = isinstance(X, np.ndarray) or (
        hasattr(X, '__array_interface__') and
        not hasattr(X, '__cuda_array_interface__')
    )

    if sample_weights is not None:
        is_sample_weight_host = isinstance(sample_weights, np.ndarray) or (
            hasattr(sample_weights, '__array_interface__') and
            not hasattr(sample_weights, '__cuda_array_interface__')
        )
        if is_host != is_sample_weight_host:
            raise ValueError(
                "X and sample_weights must have the same memory residency "
                "(both host or both device). X is {}, sample_weights is {}.".format(
                    "host" if is_host else "device",
                    "host" if is_sample_weight_host else "device"
                )
            )

    if is_host:
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not X.flags['C_CONTIGUOUS']:
            raise ValueError("X must have C contiguous layout")
        if sample_weights is not None:
            if not isinstance(sample_weights, np.ndarray):
                sample_weights = np.asarray(sample_weights)
            if not sample_weights.flags['C_CONTIGUOUS']:
                raise ValueError("sample_weights must have C contiguous layout")

    x_ai = wrap_array(X)
    _check_input_array(
        x_ai, [np.dtype('float32'), np.dtype('float64')]
    )

    cdef cydlpack.DLManagedTensor* x_dlpack = cydlpack.dlpack_c(x_ai)
    cdef cydlpack.DLManagedTensor* sample_weight_dlpack = NULL

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef double inertia = 0
    cdef int n_iter = 0

    if centroids is None:
        centroids = device_ndarray.empty(
            (params.n_clusters, x_ai.shape[1]), dtype=x_ai.dtype
        )

    centroids_ai = wrap_array(centroids)
    cdef cydlpack.DLManagedTensor* centroids_dlpack = \
        cydlpack.dlpack_c(centroids_ai)

    if sample_weights is not None:
        sample_weight_dlpack = \
            cydlpack.dlpack_c(wrap_array(sample_weights))

    with cuda_interruptible():
        check_cuvs(cuvsKMeansFit(
            res,
            params.params,
            x_dlpack,
            sample_weight_dlpack,
            centroids_dlpack,
            &inertia,
            &n_iter))

    return FitOutput(centroids, inertia, n_iter)


PredictOutput = namedtuple("PredictOutput", "labels inertia")


@auto_sync_resources
@auto_convert_output
def predict(
    KMeansParams params, X, centroids, sample_weights=None, labels=None,
    normalize_weight=True, resources=None
):
    """
    Predict clusters with the k-means algorithm

    Parameters
    ----------

    params : KMeansParams
        Parameters to used in fitting KMeans model
    X : Input CUDA array interface compliant matrix shape (m, k)
    centroids : CUDA array interface compliant matrix, calculated by fit
                shape (n_clusters, k)
    sample_weights : Optional input CUDA array interface compliant matrix shape
                     (n_clusters, 1) default: None
    labels : Optional preallocated CUDA array interface matrix shape (m, 1)
        to hold the output
    normalize_weight: bool
        True if the weights should be normalized
    {resources_docstring}

    Returns
    -------
    labels : raft.device_ndarray
        The label for each datapoint in X
    inertia : float
       Sum of squared distances of samples to their closest cluster center

    Examples
    --------

    >>> import cupy as cp
    >>>
    >>> from cuvs.cluster.kmeans import fit, predict, KMeansParams
    >>>
    >>> n_samples = 5000
    >>> n_features = 50
    >>> n_clusters = 3
    >>>
    >>> X = cp.random.random_sample((n_samples, n_features),
    ...                             dtype=cp.float32)

    >>> params = KMeansParams(n_clusters=n_clusters)
    >>> centroids, inertia, n_iter = fit(params, X)
    >>>
    >>> labels, inertia = predict(params, X, centroids)
    """

    x_ai = wrap_array(X)
    _check_input_array(x_ai, [np.dtype('float32'), np.dtype('float64')])
    cdef cydlpack.DLManagedTensor* x_dlpack = cydlpack.dlpack_c(x_ai)

    cdef cydlpack.DLManagedTensor* sample_weight_dlpack = NULL
    if sample_weights is not None:
        sample_weight_dlpack = cydlpack.dlpack_c(wrap_array(sample_weights))

    if labels is None:
        labels = device_ndarray.empty((x_ai.shape[0]), dtype='int32')

    labels_ai = wrap_array(labels)
    _check_input_array(labels_ai, [np.dtype('int32')])
    cdef cydlpack.DLManagedTensor * labels_dlpack = \
        cydlpack.dlpack_c(labels_ai)

    centroids_ai = wrap_array(centroids)
    _check_input_array(centroids_ai, [np.dtype('float32'),
                                      np.dtype('float64')])
    cdef cydlpack.DLManagedTensor * centroids_dlpack = \
        cydlpack.dlpack_c(centroids_ai)

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    cdef double inertia = 0

    with cuda_interruptible():
        check_cuvs(cuvsKMeansPredict(
            res,
            params.params,
            x_dlpack,
            sample_weight_dlpack,
            centroids_dlpack,
            labels_dlpack,
            normalize_weight,
            &inertia))

    return PredictOutput(labels, inertia)


@auto_sync_resources
@auto_convert_output
def cluster_cost(X, centroids, resources=None):
    """
    Compute cluster cost given an input matrix and existing centroids

    Parameters
    ----------
    X : Input CUDA array interface compliant matrix shape (m, k)
    centroids : Input CUDA array interface compliant matrix shape
                    (n_clusters, k)
    {resources_docstring}

    Returns
    -------
    inertia : float
        The cluster cost between the input matrix and existing centroids

    Examples
    --------

    >>> import cupy as cp
    >>>
    >>> from cuvs.cluster.kmeans import cluster_cost
    >>>
    >>> n_samples = 5000
    >>> n_features = 50
    >>> n_clusters = 3
    >>>
    >>> X = cp.random.random_sample((n_samples, n_features),
    ...                             dtype=cp.float32)

    >>> centroids = cp.random.random_sample((n_clusters, n_features),
    ...                                      dtype=cp.float32)

    >>> inertia = cluster_cost(X, centroids)
    """

    x_ai = wrap_array(X)
    _check_input_array(x_ai, [np.dtype('float32'), np.dtype('float64')])
    cdef cydlpack.DLManagedTensor* x_dlpack = cydlpack.dlpack_c(x_ai)

    centroids_ai = wrap_array(centroids)
    _check_input_array(centroids_ai, [np.dtype('float32'),
                                      np.dtype('float64')])
    cdef cydlpack.DLManagedTensor* centroids_dlpack = \
        cydlpack.dlpack_c(centroids_ai)

    cdef double inertia = 0
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    with cuda_interruptible():
        check_cuvs(cuvsKMeansClusterCost(
            res,
            x_dlpack,
            centroids_dlpack,
            &inertia))

    return inertia
