#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from libc.stddef cimport size_t
from libc.stdint cimport int32_t, int64_t, uint32_t, uintptr_t
from libcpp cimport bool

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor
from cuvs.distance_type cimport cuvsDistanceType
from cuvs.neighbors.filters.filters cimport cuvsFilter



cdef extern from "cuvs/core/device_udf.h" nogil:

    ctypedef enum cuvsDeviceUDFPayloadKind:
        CUVS_DEVICE_UDF_PAYLOAD_LTOIR
        CUVS_DEVICE_UDF_PAYLOAD_CUDA_SOURCE

    cdef enum:
        CUVS_UDF_CAPTURE_READONLY

    ctypedef struct cuvsUDFCapture:
        const char* name
        const char* dtype
        const int64_t* shape
        const int64_t* strides
        int32_t ndim
        int32_t device_id
        uintptr_t pointer
        uint32_t flags

    ctypedef struct cuvsDeviceUDF:
        const char* abi
        cuvsDeviceUDFPayloadKind payload_kind
        const void* payload
        size_t payload_size
        const char* symbol_name
        const cuvsUDFCapture* captures
        size_t n_captures
        const char* cache_key
        uint32_t flags

cdef extern from "cuvs/neighbors/ivf_flat.h" nogil:

    ctypedef struct cuvsIvfFlatIndexParams:
        cuvsDistanceType metric
        float metric_arg
        bool add_data_on_build
        uint32_t n_lists
        uint32_t kmeans_n_iters
        double kmeans_trainset_fraction
        bool adaptive_centers
        bool conservative_memory_allocation

    ctypedef cuvsIvfFlatIndexParams* cuvsIvfFlatIndexParams_t

    ctypedef struct cuvsIvfFlatSearchParams:
        uint32_t n_probes
        const cuvsDeviceUDF* metric_udf

    ctypedef cuvsIvfFlatSearchParams* cuvsIvfFlatSearchParams_t

    ctypedef struct cuvsIvfFlatIndex:
        uintptr_t addr
        DLDataType dtype

    ctypedef cuvsIvfFlatIndex* cuvsIvfFlatIndex_t

    cuvsError_t cuvsIvfFlatIndexParamsCreate(cuvsIvfFlatIndexParams_t* params)

    cuvsError_t cuvsIvfFlatIndexParamsDestroy(cuvsIvfFlatIndexParams_t index)

    cuvsError_t cuvsIvfFlatSearchParamsCreate(
        cuvsIvfFlatSearchParams_t* params)

    cuvsError_t cuvsIvfFlatSearchParamsDestroy(cuvsIvfFlatSearchParams_t index)

    cuvsError_t cuvsIvfFlatIndexCreate(cuvsIvfFlatIndex_t* index)

    cuvsError_t cuvsIvfFlatIndexDestroy(cuvsIvfFlatIndex_t index)

    cuvsError_t cuvsIvfFlatIndexGetNLists(cuvsIvfFlatIndex_t index,
                                          int64_t * n_lists)

    cuvsError_t cuvsIvfFlatIndexGetDim(cuvsIvfFlatIndex_t index, int64_t * dim)

    cuvsError_t cuvsIvfFlatIndexGetCenters(cuvsIvfFlatIndex_t index,
                                           DLManagedTensor * centers)

    cuvsError_t cuvsIvfFlatBuild(cuvsResources_t res,
                                 cuvsIvfFlatIndexParams* params,
                                 DLManagedTensor* dataset,
                                 cuvsIvfFlatIndex_t index) except +

    cuvsError_t cuvsIvfFlatSearch(cuvsResources_t res,
                                  cuvsIvfFlatSearchParams* params,
                                  cuvsIvfFlatIndex_t index,
                                  DLManagedTensor* queries,
                                  DLManagedTensor* neighbors,
                                  DLManagedTensor* distances,
                                  cuvsFilter filter) except +

    cuvsError_t cuvsIvfFlatSerialize(cuvsResources_t res,
                                     const char * filename,
                                     cuvsIvfFlatIndex_t index) except +

    cuvsError_t cuvsIvfFlatDeserialize(cuvsResources_t res,
                                       const char * filename,
                                       cuvsIvfFlatIndex_t index) except +

    cuvsError_t cuvsIvfFlatExtend(cuvsResources_t res,
                                  DLManagedTensor* new_vectors,
                                  DLManagedTensor* new_indices,
                                  cuvsIvfFlatIndex_t index)


cdef class IndexParams:
    cdef cuvsIvfFlatIndexParams* params

cdef class SearchParams:
    cdef cuvsIvfFlatSearchParams* params
    cdef object _metric
    cdef bytes _metric_payload
    cdef bytes _metric_abi
    cdef bytes _metric_symbol_name
    cdef bytes _metric_cache_key
    cdef bytes _metric_capture_0_name
    cdef bytes _metric_capture_0_dtype
    cdef cuvsDeviceUDF _metric_udf_desc
    cdef cuvsUDFCapture _metric_captures[1]
    cdef int64_t _metric_capture_0_shape[8]
    cdef int64_t _metric_capture_0_strides[8]
