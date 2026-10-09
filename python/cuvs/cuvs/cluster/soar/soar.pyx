#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

import numpy as np

from cuvs.common cimport cydlpack

from pylibraft.common import auto_convert_output, device_ndarray
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.interruptible import cuda_interruptible

from cuvs.common.exceptions import check_cuvs
from cuvs.common.resources import auto_sync_resources
from cuvs.neighbors.common import _check_input_array

LABEL_DTYPES = [np.dtype("uint32"), np.dtype("int32")]


cdef class Params:
    """
    Hyper-parameters for SOAR assignment.

    Parameters
    ----------
    lambda_ : float
        Weight of the projection of the secondary residual onto the primary
        residual in the SOAR loss. Larger values penalize secondary centroids
        whose residual is aligned with the primary residual, favoring
        complementary assignments. ``0`` reduces the loss to plain squared
        distance, which the primary centroid itself minimizes, so nothing is
        spilled. Named with a trailing underscore because ``lambda`` is a
        Python keyword (default: 1.0).
    """

    cdef cuvsSoarParams* params

    def __cinit__(self):
        check_cuvs(cuvsSoarParamsCreate(&self.params))

    def __dealloc__(self):
        check_cuvs(cuvsSoarParamsDestroy(self.params))

    def __init__(self, *, lambda_=None):
        if lambda_ is not None:
            self.params.lambda_ = lambda_

    @property
    def lambda_(self):
        return self.params.lambda_


@auto_sync_resources
@auto_convert_output
def predict(Params params, dataset, centroids, labels, soar_labels=None,
            resources=None):
    """
    Assign a secondary ("spilled") cluster to each row of the dataset.

    SOAR (Spilling with Orthogonality-Amplified Residuals) picks, for each
    vector, a second centroid that complements the primary assignment instead
    of merely being the next-closest one. It minimizes the loss of Theorem 3.1
    of https://arxiv.org/abs/2404.00774: for a vector ``x`` with primary
    residual ``r = x - centroids[labels[i]]``,

        ``score(c) = ||x - c||^2 + lambda * (dot(r / ||r||, x - c))^2``

    and ``soar_labels[i]`` is the centroid minimizing that score. Indexing a
    vector under both its primary and its secondary centroid improves recall
    for queries near a partition boundary.

    The primary centroid is not excluded from the search, so
    ``soar_labels[i] == labels[i]`` is a possible (and meaningful) result: it
    says that no other centroid is worth spilling to, which is the common case
    for vectors in the interior of a cluster. Callers that treat SOAR as a
    strictly second posting list should test for this case and skip those rows.

    Scratch memory scales as ``n_rows * n_clusters * 4`` bytes because scores
    against all centroids are materialized at once and are not tiled. Process
    the dataset in row batches to bound the peak device memory usage.

    Parameters
    ----------
    params : Params
        Parameters for SOAR assignment.
    dataset : CUDA array interface compliant matrix, row major, float32
        shape (n_rows, n_features)
    centroids : CUDA array interface compliant matrix, row major, float32
        Cluster centroids, shape (n_clusters, n_features)
    labels : CUDA array interface compliant vector, uint32 or int32
        Index of the primary cluster each row belongs to, as produced by
        k-means prediction. Every value must be in ``[0, n_clusters)``.
        shape (n_rows,)
    soar_labels : Optional preallocated CUDA array interface vector to hold
        the output, shape (n_rows,). Must have the same dtype as ``labels``.
        When None, an array matching the dtype of ``labels`` is allocated.
    {resources_docstring}

    Returns
    -------
    soar_labels : raft.device_ndarray
        Index of the secondary cluster each row is spilled to.

    Examples
    --------

    >>> import cupy as cp
    >>>
    >>> from cuvs.cluster.kmeans import KMeansParams
    >>> from cuvs.cluster.kmeans import fit as kmeans_fit
    >>> from cuvs.cluster.kmeans import predict as kmeans_predict
    >>> from cuvs.cluster.soar import Params, predict
    >>>
    >>> n_samples = 5000
    >>> n_features = 50
    >>> n_clusters = 50
    >>>
    >>> X = cp.random.random_sample((n_samples, n_features),
    ...                             dtype=cp.float32)
    >>>
    >>> # SOAR needs centroids and a primary label per row, so k-means runs
    >>> # first
    >>> kmeans_params = KMeansParams(n_clusters=n_clusters)
    >>> centroids, inertia, n_iter = kmeans_fit(kmeans_params, X)
    >>> labels, inertia = kmeans_predict(kmeans_params, X, centroids)
    >>>
    >>> soar_labels = predict(Params(), X, centroids, labels)
    """

    dataset_ai = wrap_array(dataset)
    _check_input_array(dataset_ai, [np.dtype("float32")])

    centroids_ai = wrap_array(centroids)
    _check_input_array(centroids_ai, [np.dtype("float32")],
                       exp_cols=dataset_ai.shape[1])

    labels_ai = wrap_array(labels)
    _check_input_array(labels_ai, LABEL_DTYPES,
                       exp_rows=dataset_ai.shape[0])

    if soar_labels is None:
        soar_labels = device_ndarray.empty((dataset_ai.shape[0],),
                                           dtype=labels_ai.dtype)

    soar_labels_ai = wrap_array(soar_labels)
    _check_input_array(soar_labels_ai, [labels_ai.dtype],
                       exp_rows=dataset_ai.shape[0])

    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_ai)
    cdef cydlpack.DLManagedTensor* centroids_dlpack = \
        cydlpack.dlpack_c(centroids_ai)
    cdef cydlpack.DLManagedTensor* labels_dlpack = \
        cydlpack.dlpack_c(labels_ai)
    cdef cydlpack.DLManagedTensor* soar_labels_dlpack = \
        cydlpack.dlpack_c(soar_labels_ai)

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    with cuda_interruptible():
        check_cuvs(cuvsSoarPredict(
            res,
            params.params,
            dataset_dlpack,
            centroids_dlpack,
            labels_dlpack,
            soar_labels_dlpack))

    return soar_labels
