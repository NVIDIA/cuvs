#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from libc.stdint cimport int32_t, uint32_t, uintptr_t
from libcpp cimport bool

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor
from cuvs.distance_type cimport cuvsDistanceType


cdef extern from "cuvs/neighbors/vamana.h" nogil:

    ctypedef struct cuvsVamanaIndexParams:
        cuvsDistanceType metric
        uint32_t graph_degree
        uint32_t visited_size
        float vamana_iters
        float alpha
        float max_fraction
        float batch_base
        uint32_t queue_size
        uint32_t reverse_batchsize
        DLManagedTensor* initial_graph

    ctypedef cuvsVamanaIndexParams* cuvsVamanaIndexParams_t

    ctypedef struct cuvsVamanaIndex:
        uintptr_t addr
        DLDataType dtype

    ctypedef cuvsVamanaIndex* cuvsVamanaIndex_t

    cuvsError_t cuvsVamanaIndexParamsCreate(cuvsVamanaIndexParams_t* params)

    cuvsError_t cuvsVamanaIndexParamsDestroy(cuvsVamanaIndexParams_t params)

    cuvsError_t cuvsVamanaIndexCreate(cuvsVamanaIndex_t* index)

    cuvsError_t cuvsVamanaIndexDestroy(cuvsVamanaIndex_t index)

    cuvsError_t cuvsVamanaIndexGetDims(cuvsVamanaIndex_t index, int* dim)

    cuvsError_t cuvsVamanaBuild(cuvsResources_t res,
                                cuvsVamanaIndexParams_t params,
                                DLManagedTensor* dataset,
                                cuvsVamanaIndex_t index)

    cuvsError_t cuvsVamanaSerialize(cuvsResources_t res,
                                    const char* filename,
                                    cuvsVamanaIndex_t index,
                                    bool include_dataset)


cdef class Index:
    """
    Vamana index object. This object stores the trained Vamana index state
    which can be used to perform nearest neighbors searches.
    """

    cdef cuvsVamanaIndex_t index
    cdef bool trained
    cdef str active_index_type


cdef class IndexParams:
    cdef cuvsVamanaIndexParams* params
