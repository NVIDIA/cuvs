#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from libc.stdint cimport int64_t, uint32_t, uintptr_t
from libcpp cimport bool

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor
from cuvs.distance_type cimport cuvsDistanceType
from cuvs.neighbors.filters.filters cimport cuvsFilter


cdef extern from "cuvs/neighbors/ivf_sq.h" nogil:

    ctypedef struct cuvsIvfSqIndexParams:
        cuvsDistanceType metric
        float metric_arg
        bool add_data_on_build
        uint32_t n_lists
        uint32_t kmeans_n_iters
        uint32_t max_train_points_per_cluster
        bool conservative_memory_allocation

    ctypedef cuvsIvfSqIndexParams* cuvsIvfSqIndexParams_t

    ctypedef struct cuvsIvfSqSearchParams:
        uint32_t n_probes

    ctypedef cuvsIvfSqSearchParams* cuvsIvfSqSearchParams_t

    ctypedef struct cuvsIvfSqIndex:
        uintptr_t addr
        DLDataType dtype

    ctypedef cuvsIvfSqIndex* cuvsIvfSqIndex_t

    cuvsError_t cuvsIvfSqIndexParamsCreate(cuvsIvfSqIndexParams_t* params)

    cuvsError_t cuvsIvfSqIndexParamsDestroy(cuvsIvfSqIndexParams_t index)

    cuvsError_t cuvsIvfSqSearchParamsCreate(
        cuvsIvfSqSearchParams_t* params)

    cuvsError_t cuvsIvfSqSearchParamsDestroy(cuvsIvfSqSearchParams_t index)

    cuvsError_t cuvsIvfSqIndexCreate(cuvsIvfSqIndex_t* index)

    cuvsError_t cuvsIvfSqIndexDestroy(cuvsIvfSqIndex_t index)

    cuvsError_t cuvsIvfSqIndexGetNLists(cuvsIvfSqIndex_t index,
                                        int64_t * n_lists)

    cuvsError_t cuvsIvfSqIndexGetDim(cuvsIvfSqIndex_t index, int64_t * dim)

    cuvsError_t cuvsIvfSqIndexGetSize(cuvsIvfSqIndex_t index, int64_t * size)

    cuvsError_t cuvsIvfSqIndexGetCenters(cuvsIvfSqIndex_t index,
                                         DLManagedTensor * centers)

    cuvsError_t cuvsIvfSqBuild(cuvsResources_t res,
                               cuvsIvfSqIndexParams* params,
                               DLManagedTensor* dataset,
                               cuvsIvfSqIndex_t index) except +

    cuvsError_t cuvsIvfSqSearch(cuvsResources_t res,
                                cuvsIvfSqSearchParams* params,
                                cuvsIvfSqIndex_t index,
                                DLManagedTensor* queries,
                                DLManagedTensor* neighbors,
                                DLManagedTensor* distances,
                                cuvsFilter filter) except +

    cuvsError_t cuvsIvfSqSerialize(cuvsResources_t res,
                                   const char * filename,
                                   cuvsIvfSqIndex_t index) except +

    cuvsError_t cuvsIvfSqDeserialize(cuvsResources_t res,
                                     const char * filename,
                                     cuvsIvfSqIndex_t index) except +

    cuvsError_t cuvsIvfSqExtend(cuvsResources_t res,
                                DLManagedTensor* new_vectors,
                                DLManagedTensor* new_indices,
                                cuvsIvfSqIndex_t index) except +


cdef class IndexParams:
    cdef cuvsIvfSqIndexParams* params

cdef class SearchParams:
    cdef cuvsIvfSqSearchParams* params
