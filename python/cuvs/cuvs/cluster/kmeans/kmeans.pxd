#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from libc.stdint cimport int64_t, uintptr_t
from libcpp cimport bool

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor
from cuvs.distance_type cimport cuvsDistanceType


cdef extern from "cuvs/cluster/kmeans.h" nogil:
    ctypedef enum cuvsKMeansInitMethod:
        KMeansPlusPlus
        Random
        Array

    ctypedef enum cuvsKMeansType:
        CUVS_KMEANS_TYPE_KMEANS
        CUVS_KMEANS_TYPE_KMEANS_BALANCED

    # NOTE: The Python binding currently targets the unsuffixed cuvsKMeansParams
    # ABI (which still carries the deprecated `inertia_check` field). In cuVS
    # 26.08 this struct/entry-point set will be replaced by the contents of
    # cuvsKMeansParams_v2 -- once that lands, the `inertia_check` field below
    # should be deleted.
    ctypedef struct cuvsKMeansParams:
        cuvsDistanceType metric,
        int n_clusters,
        cuvsKMeansInitMethod init,
        int max_iter,
        double tol,
        int n_init,
        double oversampling_factor,
        int batch_samples,
        int batch_centroids,
        bool inertia_check,
        bool hierarchical,
        int hierarchical_n_iters,
        int64_t streaming_batch_size,
        int64_t init_size

    ctypedef struct cuvsKMeansParams_v2:
        cuvsDistanceType metric,
        int n_clusters,
        cuvsKMeansInitMethod init,
        int max_iter,
        double tol,
        int n_init,
        double oversampling_factor,
        int batch_samples,
        int batch_centroids,
        bool hierarchical,
        int hierarchical_n_iters,
        int64_t device_buffer_batch_size,
        int64_t init_size,
        bool device_buffer_prefetch

    ctypedef cuvsKMeansParams* cuvsKMeansParams_t
    ctypedef cuvsKMeansParams_v2* cuvsKMeansParams_v2_t

    cuvsError_t cuvsKMeansParamsCreate(cuvsKMeansParams_t* index)

    cuvsError_t cuvsKMeansParamsDestroy(cuvsKMeansParams_t index)

    cuvsError_t cuvsKMeansFit(cuvsResources_t res,
                              cuvsKMeansParams_t params,
                              DLManagedTensor* X,
                              DLManagedTensor* sample_weight,
                              DLManagedTensor * centroids,
                              double * inertia,
                              int * n_iter) except +

    cuvsError_t cuvsKMeansPredict(cuvsResources_t res,
                                  cuvsKMeansParams_t params,
                                  DLManagedTensor* X,
                                  DLManagedTensor* sample_weight,
                                  DLManagedTensor * centroids,
                                  DLManagedTensor * labels,
                                  bool normalize_weight,
                                  double * inertia)

    cuvsError_t cuvsKMeansClusterCost(cuvsResources_t res,
                                      DLManagedTensor* X,
                                      DLManagedTensor* centroids,
                                      double* cost)


cdef class KMeansParams:
    cdef cuvsKMeansParams* params
    cdef bool _device_buffer_prefetch
