#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3


from cuda.bindings.cyruntime cimport cudaStream_t
from libc.stdint cimport int64_t, uintptr_t

from cuvs.common.cydlpack cimport DLManagedTensor


cdef extern from "cuvs/core/c_api.h":
    ctypedef uintptr_t cuvsResources_t

    ctypedef enum cuvsError_t:
        CUVS_ERROR,
        CUVS_SUCCESS

    cuvsError_t cuvsResourcesCreate(cuvsResources_t* res)
    cuvsError_t cuvsResourcesCreateWithMemoryTracking(
        cuvsResources_t* res,
        const char* csv_path,
        int64_t sample_interval_ms)
    cuvsError_t cuvsResourcesDestroy(cuvsResources_t res)
    cuvsError_t cuvsStreamSet(cuvsResources_t res, cudaStream_t stream)
    cuvsError_t cuvsStreamSync(cuvsResources_t res)
    const char * cuvsGetLastErrorText()

    cuvsError_t cuvsMultiGpuResourcesCreate(cuvsResources_t* res)
    cuvsError_t cuvsMultiGpuResourcesCreateWithDeviceIds(
        cuvsResources_t* res,
        DLManagedTensor* device_ids)
    cuvsError_t cuvsMultiGpuResourcesDestroy(cuvsResources_t res)
    cuvsError_t cuvsMultiGpuResourcesSetMemoryPool(cuvsResources_t res,
                                                   int percent_of_free_memory)

    cuvsError_t cuvsMatrixCopy(cuvsResources_t res, DLManagedTensor * src,
                               DLManagedTensor * dst)

    cuvsError_t cuvsMatrixSliceRows(cuvsResources_t res, DLManagedTensor* src,
                                    int64_t start, int64_t end,
                                    DLManagedTensor* dst)
