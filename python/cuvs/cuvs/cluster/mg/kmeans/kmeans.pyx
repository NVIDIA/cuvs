#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from collections import namedtuple

import numpy as np
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.interruptible import cuda_interruptible

from cuvs.common.exceptions import check_cuvs
from cuvs.common.mg_resources import auto_sync_multi_gpu_resources
from cuvs.neighbors.common import _check_input_array

from cuvs.cluster.kmeans.kmeans cimport KMeansParams, cuvsKMeansParams_v2
from cuvs.common cimport cydlpack
from cuvs.common.c_api cimport cuvsResources_t

from .kmeans cimport cuvsMultiGpuKMeansFit

FitOutput = namedtuple("FitOutput", "centroids inertia n_iter")


def _as_host_array(array, name, *, writable=False):
    if hasattr(array, "__cuda_array_interface__") and not isinstance(
        array, np.ndarray
    ):
        raise ValueError(
            f"SNMG KMeans requires {name} to be in host memory"
        )

    if not isinstance(array, np.ndarray):
        array = np.asarray(array)

    if array.ndim == 0:
        raise ValueError(f"{name} must be an array")

    if not array.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{name} must have C contiguous layout")

    if writable and not array.flags["WRITEABLE"]:
        raise ValueError(f"{name} must be writable")

    return array


@auto_sync_multi_gpu_resources
def fit(
    KMeansParams params, X, centroids=None, sample_weights=None,
    resources=None
):
    """
    Find clusters with single-node multi-GPU k-means using host data.

    Parameters
    ----------
    params : KMeansParams
        Parameters to use to fit KMeans model.
    X : host array-like
        Training instances, shape (m, k). Must be C-contiguous float32 or
        float64 host data.
    centroids : host array-like, optional
        Initial centroids when ``params.init_method == "Array"`` and output
        centroids for all init methods. If omitted, a host NumPy output array
        is allocated unless ``init_method == "Array"``.
    sample_weights : host array-like, optional
        Optional weights per observation. Must be C-contiguous and have the
        same dtype as X.
    {resources_docstring}

    Returns
    -------
    FitOutput
        ``centroids`` is a host NumPy array containing the computed centroids,
        ``inertia`` is the final objective value, and ``n_iter`` is the number
        of iterations run.
    """

    if params.hierarchical:
        raise ValueError("SNMG KMeans does not support hierarchical KMeans")

    X = _as_host_array(X, "X")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    x_ai = wrap_array(X)
    _check_input_array(
        x_ai, [np.dtype("float32"), np.dtype("float64")]
    )

    if centroids is None:
        if params.init_method == "Array":
            raise ValueError(
                "centroids must be provided when init_method is 'Array'"
            )
        centroids = np.empty(
            (params.n_clusters, X.shape[1]), dtype=X.dtype
        )
    else:
        centroids = _as_host_array(centroids, "centroids", writable=True)

    if centroids.ndim != 2:
        raise ValueError("centroids must be a 2D array")
    if centroids.dtype != X.dtype:
        raise TypeError("centroids dtype must match X dtype")

    centroids_ai = wrap_array(centroids)
    _check_input_array(
        centroids_ai,
        [x_ai.dtype],
        exp_rows=params.n_clusters,
        exp_cols=X.shape[1],
    )

    cdef cydlpack.DLManagedTensor* sample_weight_dlpack = NULL
    if sample_weights is not None:
        sample_weights = _as_host_array(sample_weights, "sample_weights")
        if sample_weights.ndim != 1:
            raise ValueError("sample_weights must be a 1D array")
        if sample_weights.dtype != X.dtype:
            raise TypeError("sample_weights dtype must match X dtype")

        sample_weights_ai = wrap_array(sample_weights)
        _check_input_array(
            sample_weights_ai,
            [x_ai.dtype],
            exp_rows=X.shape[0],
            exp_row_major=False,
        )
        sample_weight_dlpack = cydlpack.dlpack_c(sample_weights_ai)

    cdef cydlpack.DLManagedTensor* x_dlpack = cydlpack.dlpack_c(x_ai)
    cdef cydlpack.DLManagedTensor* centroids_dlpack = (
        cydlpack.dlpack_c(centroids_ai)
    )
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef double inertia = 0
    cdef int n_iter = 0
    cdef cuvsKMeansParams_v2 params_v2
    params_v2.metric = params.params.metric
    params_v2.n_clusters = params.params.n_clusters
    params_v2.init = params.params.init
    params_v2.max_iter = params.params.max_iter
    params_v2.tol = params.params.tol
    params_v2.n_init = params.params.n_init
    params_v2.oversampling_factor = params.params.oversampling_factor
    params_v2.batch_samples = params.params.batch_samples
    params_v2.batch_centroids = params.params.batch_centroids
    params_v2.hierarchical = params.params.hierarchical
    params_v2.hierarchical_n_iters = params.params.hierarchical_n_iters
    params_v2.device_buffer_samples = params.params.device_buffer_samples
    params_v2.init_size = params.params.init_size

    with cuda_interruptible():
        check_cuvs(cuvsMultiGpuKMeansFit(
            res,
            &params_v2,
            x_dlpack,
            sample_weight_dlpack,
            centroids_dlpack,
            &inertia,
            &n_iter))

    return FitOutput(centroids, inertia, n_iter)
