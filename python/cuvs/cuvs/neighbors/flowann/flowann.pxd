#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from libc.stdint cimport int64_t, uint16_t, uint32_t, uint64_t, uintptr_t
from libc.stddef cimport size_t
from libcpp cimport bool

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor
from cuvs.common.dataset cimport cuvsDataset_t
from cuvs.distance_type cimport cuvsDistanceType
from cuvs.neighbors.cagra.cagra cimport (
    cuvsCagraGraphBuildAlgo,
    cuvsCagraSearchParams,
)

cdef extern from "cuvs/neighbors/flowann.h" nogil:
    ctypedef struct cuvsFlowannIndexParams:
        cuvsDistanceType metric
        size_t intermediate_graph_degree
        size_t graph_degree
        cuvsCagraGraphBuildAlgo build_algo
        size_t nn_descent_niter
        size_t device_graph_budget_bytes
        uint16_t node_per_cacheline
        bool grouping_enabled
        uint32_t n_groups
        uint16_t n_bits
        double balance_tolerance
        size_t training_rows
        size_t assignment_batch_rows
        uint32_t kmeans_n_iters
        bool validate
        uint32_t num_seeds
        size_t seed_training_rows
        uint64_t seed

    ctypedef cuvsFlowannIndexParams* cuvsFlowannIndexParams_t

    ctypedef struct cuvsFlowannSearchParams:
        cuvsCagraSearchParams cagra
        uint32_t num_seeds
        float sync_window_scale
        uint32_t sync_drop_threshold
        uint32_t num_queues
        uint32_t empty_pause
        bool collect_statistics

    ctypedef cuvsFlowannSearchParams* cuvsFlowannSearchParams_t

    ctypedef struct cuvsFlowannIndex:
        uintptr_t addr
        DLDataType dtype

    ctypedef cuvsFlowannIndex* cuvsFlowannIndex_t

    cuvsError_t cuvsFlowannIndexParamsCreate(cuvsFlowannIndexParams_t* params)
    cuvsError_t cuvsFlowannIndexParamsDestroy(cuvsFlowannIndexParams_t params)
    cuvsError_t cuvsFlowannSearchParamsCreate(cuvsFlowannSearchParams_t* params)
    cuvsError_t cuvsFlowannSearchParamsDestroy(cuvsFlowannSearchParams_t params)
    cuvsError_t cuvsFlowannIndexCreate(cuvsFlowannIndex_t* index)
    cuvsError_t cuvsFlowannIndexDestroy(cuvsFlowannIndex_t index)
    cuvsError_t cuvsFlowannIndexGetDims(cuvsFlowannIndex_t index, int64_t* dim)
    cuvsError_t cuvsFlowannIndexGetSize(cuvsFlowannIndex_t index, int64_t* size)
    cuvsError_t cuvsFlowannIndexGetGraphDegree(cuvsFlowannIndex_t index, int64_t* degree)
    cuvsError_t cuvsFlowannBuild(cuvsResources_t res,
                                 cuvsFlowannIndexParams_t params,
                                 cuvsDataset_t dataset,
                                 cuvsFlowannIndex_t index)
    cuvsError_t cuvsFlowannSearch(cuvsResources_t res,
                                  cuvsFlowannSearchParams_t params,
                                  cuvsFlowannIndex_t index,
                                  DLManagedTensor* queries,
                                  DLManagedTensor* neighbors,
                                  DLManagedTensor* distances)
    cuvsError_t cuvsFlowannSerialize(cuvsResources_t res,
                                     const char* filename,
                                     cuvsFlowannIndex_t index)
    cuvsError_t cuvsFlowannDeserialize(cuvsResources_t res,
                                       const char* filename,
                                       DLDataType dtype,
                                       cuvsFlowannIndex_t index)

cdef class Index:
    cdef cuvsFlowannIndex_t index
    cdef bool _trained

cdef class IndexParams:
    cdef cuvsFlowannIndexParams_t params

cdef class SearchParams:
    cdef cuvsFlowannSearchParams_t params
