#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

import numpy as np

cimport cuvs.common.cydlpack

from cuvs.common.mg_resources import MultiGpuResources
from cuvs.common.resources import Resources

from cython.operator cimport dereference as deref

from cuvs.common cimport cydlpack
from cuvs.common.c_api cimport cuvsResources_t
from cuvs.distance_type cimport cuvsDistanceType

from pylibraft.common import auto_convert_output, device_ndarray
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.interruptible import cuda_interruptible

from cuvs.distance import DISTANCE_TYPES
from cuvs.neighbors.common import _check_input_array

from libc.stdint cimport int64_t

from cuvs.common.exceptions import check_cuvs

from cuvs.neighbors.ivf_pq.ivf_pq cimport cuvsIvfPqIndexParams_t
from cuvs.neighbors.nn_descent.nn_descent cimport cuvsNNDescentIndexParams_t

from .all_neighbors cimport (
    CUVS_ALL_NEIGHBORS_ALGO_BRUTE_FORCE,
    CUVS_ALL_NEIGHBORS_ALGO_IVF_PQ,
    CUVS_ALL_NEIGHBORS_ALGO_NN_DESCENT,
    cuvsAllNeighborsAlgo,
    cuvsAllNeighborsBuild,
    cuvsAllNeighborsIndexParams,
    cuvsAllNeighborsIndexParams_t,
)

from cuvs.neighbors.ivf_pq.ivf_pq import IndexParams as IvfPqIndexParams
from cuvs.neighbors.nn_descent.nn_descent import (
    IndexParams as NNDescentIndexParams,
)


cdef inline cuvsAllNeighborsAlgo _algo_from_str(object algo):
    if isinstance(algo, str):
        if algo == "brute_force":
            return CUVS_ALL_NEIGHBORS_ALGO_BRUTE_FORCE
        elif algo == "ivf_pq":
            return CUVS_ALL_NEIGHBORS_ALGO_IVF_PQ
        elif algo == "nn_descent":
            return CUVS_ALL_NEIGHBORS_ALGO_NN_DESCENT
        else:
            raise ValueError(f"Invalid algo: {algo}")
    elif isinstance(algo, int):
        return <cuvsAllNeighborsAlgo>algo
    else:
        raise ValueError(f"Invalid algo type: {type(algo)}")


cdef class AllNeighborsParams:
    """
    Parameters for all-neighbors k-NN graph building.

    Parameters
    ----------
    algo : str or cuvsAllNeighborsAlgo
        Algorithm to use for local k-NN graph building.
        Options: "brute_force", "ivf_pq", "nn_descent"
    overlap_factor : int, default=2
        Number of clusters each point is assigned to (must be < n_clusters)
    n_clusters : int, default=1
        Number of clusters/batches to partition the dataset into
        (> overlap_factor). Use n_clusters>1 to distribute the work
        across GPUs.
    metric : str or cuvsDistanceType, default="sqeuclidean"
        Distance metric to use for graph construction
    ivf_pq_params : cuvs.neighbors.ivf_pq.IndexParams, optional
        IVF-PQ specific parameters (used when algo="ivf_pq")
    nn_descent_params : cuvs.neighbors.nn_descent.IndexParams, optional
        NN-Descent specific parameters (used when algo="nn_descent")
    """

    cdef cuvsAllNeighborsIndexParams params
    cdef object _ivf_pq_params
    cdef object _nn_descent_params

    def __init__(self, *,
                 algo="nn_descent",
                 overlap_factor=2,
                 n_clusters=1,
                 metric="sqeuclidean",
                 ivf_pq_params=None,
                 nn_descent_params=None):

        self.params.algo = _algo_from_str(algo)
        self.params.overlap_factor = overlap_factor
        self.params.n_clusters = n_clusters
        self.params.metric = <cuvsDistanceType>DISTANCE_TYPES[metric]

        if ivf_pq_params is not None:
            if not isinstance(ivf_pq_params, IvfPqIndexParams):
                raise TypeError(
                    "ivf_pq_params must be an instance of "
                    "cuvs.neighbors.ivf_pq.IndexParams"
                )

            # Check metric consistency
            metric_type = DISTANCE_TYPES[metric]
            ivf_pq_metric_type = DISTANCE_TYPES[ivf_pq_params.metric]
            if ivf_pq_metric_type != metric_type:
                raise ValueError(
                    f"Metric conflict: AllNeighborsParams metric '{metric}' "
                    f"does not match IVF-PQ metric '{ivf_pq_params.metric}'. Please "
                    f"ensure both use the same metric."
                )

        if nn_descent_params is not None:
            if not isinstance(nn_descent_params, NNDescentIndexParams):
                raise TypeError(
                    "nn_descent_params must be an instance of "
                    "cuvs.neighbors.nn_descent.IndexParams"
                )

            # Check metric consistency
            metric_type = DISTANCE_TYPES[metric]
            nn_descent_metric_type = DISTANCE_TYPES[nn_descent_params.metric]
            if nn_descent_metric_type != metric_type:
                raise ValueError(
                    f"Metric conflict: AllNeighborsParams metric '{metric}' "
                    f"does not match NN-Descent metric '{nn_descent_params.metric}'. "
                    f"Please ensure both use the same metric."
                )

        # Store references to prevent garbage collection
        self._ivf_pq_params = ivf_pq_params
        self._nn_descent_params = nn_descent_params

        # Set algorithm-specific parameter pointers
        if ivf_pq_params is not None:
            self.params.ivf_pq_params = (
                <cuvsIvfPqIndexParams_t><size_t>ivf_pq_params.get_handle()
            )
        else:
            self.params.ivf_pq_params = <cuvsIvfPqIndexParams_t>NULL

        if nn_descent_params is not None:
            self.params.nn_descent_params = (
                <cuvsNNDescentIndexParams_t><size_t>(
                    nn_descent_params.get_handle()
                )
            )
        else:
            self.params.nn_descent_params = <cuvsNNDescentIndexParams_t>NULL

    def get_handle(self):
        """Get a pointer to the underlying C object."""
        return <size_t>&self.params

    @property
    def algo(self):
        """Algorithm used for local k-NN graph building."""
        if self.params.algo == CUVS_ALL_NEIGHBORS_ALGO_BRUTE_FORCE:
            return "brute_force"
        elif self.params.algo == CUVS_ALL_NEIGHBORS_ALGO_IVF_PQ:
            return "ivf_pq"
        elif self.params.algo == CUVS_ALL_NEIGHBORS_ALGO_NN_DESCENT:
            return "nn_descent"
        else:
            return self.params.algo

    @property
    def overlap_factor(self):
        """Number of clusters each point is assigned to."""
        return self.params.overlap_factor

    @property
    def n_clusters(self):
        """Number of clusters/batches to partition the dataset into."""
        return self.params.n_clusters

    @property
    def metric(self):
        """Distance metric used for graph construction."""
        # Reverse lookup in DISTANCE_TYPES
        for name, value in DISTANCE_TYPES.items():
            if value == self.params.metric:
                return name
        return self.params.metric


@auto_convert_output
def build(dataset, k, params, *,
          indices=None,
          distances=None,
          core_distances=None,
          alpha=1.0,
          resources=None):
    """
    All-neighbors allows building an approximate all-neighbors knn graph.
    Given a full dataset, it finds nearest neighbors for all the training
    vectors in the dataset.

    Parameters
    ----------
    dataset : array_like
        Training dataset to build the k-NN graph for. Can be provided
        on host (for multi-GPU build) or device (for single-GPU build).
        Host vs device location is automatically detected.
        Supported dtype: float32
    k : int
        Number of nearest neighbors to find for each point
    params : AllNeighborsParams
        Parameters object containing all build settings including algorithm
        choice and algorithm-specific parameters.
    indices : array_like, optional
        Optional output buffer for indices [num_rows x k] on device
        (int64). If not provided, will be allocated automatically.
    distances : array_like, optional
        Optional output buffer for distances [num_rows x k] on device
        (float32)
    core_distances : array_like, optional
        Optional output buffer for core distances [num_rows] on device
        (float32). Requires distances parameter to be provided.
    alpha : float, default=1.0
        Mutual-reachability scaling; used only when core_distances is
        provided
    resources : Resources or MultiGpuResources, optional
        CUDA resources to use for the operation. If not provided, a default
        Resources object will be created. Use MultiGpuResources to enable
        multi-GPU execution across multiple devices.

    Returns
    -------
    indices : array_like
        k-NN indices for each point [num_rows x k], always on device.
        If indices buffer was provided, returns the same array filled
        with results.
    distances : array_like or None
        k-NN distances if distances buffer was provided, None otherwise
    core_distances : array_like or None
        Core distances if core_distances buffer was provided, None otherwise
    """
    if not isinstance(params, AllNeighborsParams):
        raise TypeError("params must be an instance of AllNeighborsParams")

    # Check if data is on device for validation purposes
    on_device = hasattr(dataset, "__cuda_array_interface__")

    if on_device and params.n_clusters > 1:
        raise ValueError(
            "Batched all-neighbors build is not supported with data on "
            "device. Put data on host for batch build."
        )

    if not isinstance(resources, (Resources, MultiGpuResources)):
        resources = Resources()

    resources.sync()

    dataset_ai = wrap_array(dataset)
    _check_input_array(dataset_ai, [np.dtype('float32')])
    n_rows, n_cols = dataset_ai.shape

    # Check dependencies between parameters
    if core_distances is not None and distances is None:
        raise ValueError(
            "distances must be provided when core_distances is provided"
        )

    # Validate user-provided outputs (must be device arrays if provided)
    if indices is not None and not hasattr(
        indices, "__cuda_array_interface__"
    ):
        raise ValueError(
            "indices must be a device array (CUDA array interface)"
        )
    if distances is not None and not hasattr(
        distances, "__cuda_array_interface__"
    ):
        raise ValueError(
            "distances must be a device array (CUDA array interface)"
        )
    if core_distances is not None and not hasattr(
        core_distances, "__cuda_array_interface__"
    ):
        raise ValueError(
            "core_distances must be a device array (CUDA array interface)"
        )

    # Handle indices array (create if not provided)
    if indices is None:
        indices = device_ndarray.empty((n_rows, k), dtype="int64")

    indices_out = wrap_array(indices)
    _check_input_array(
        indices_out, [np.dtype("int64")], exp_rows=n_rows, exp_cols=k
    )

    distances_out = None
    if distances is not None:
        distances_out = wrap_array(distances)
        _check_input_array(
            distances_out, [np.dtype("float32")], exp_rows=n_rows, exp_cols=k
        )

    core_out = None
    if core_distances is not None:
        core_out = wrap_array(core_distances)
        _check_input_array(
            core_out, [np.dtype("float32")], exp_rows=n_rows, exp_cols=None
        )

    cdef cydlpack.DLManagedTensor* indices_dlpack = cydlpack.dlpack_c(
        indices_out
    )

    cdef cydlpack.DLManagedTensor* distances_dlpack = NULL
    if distances_out is not None:
        distances_dlpack = cydlpack.dlpack_c(distances_out)

    cdef cydlpack.DLManagedTensor* core_dlpack = NULL
    if core_out is not None:
        core_dlpack = cydlpack.dlpack_c(core_out)

    cdef cydlpack.DLManagedTensor* dataset_dlpack = cydlpack.dlpack_c(
        dataset_ai
    )

    cdef cuvsAllNeighborsIndexParams_t params_ptr = (
        <cuvsAllNeighborsIndexParams_t><size_t>params.get_handle()
    )

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    with cuda_interruptible():
        # Use unified function that auto-detects host vs device
        check_cuvs(cuvsAllNeighborsBuild(
            res,
            params_ptr,
            dataset_dlpack,
            indices_dlpack,
            distances_dlpack,
            core_dlpack,
            <float>alpha,
        ))

    # Build return tuple based on provided parameters
    result = [indices]
    if distances is not None:
        result.append(distances)
    if core_distances is not None:
        result.append(core_distances)

    # Return single element if only indices, otherwise return tuple
    if len(result) == 1:
        return result[0]
    else:
        return tuple(result)
