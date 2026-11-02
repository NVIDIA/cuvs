#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from libcpp cimport bool

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor


cdef extern from "cuvs/core/dataset.h" nogil:
    ctypedef enum cuvsDatasetLayout_t:
        CUVS_DATASET_LAYOUT_STANDARD
        CUVS_DATASET_LAYOUT_PADDED

    ctypedef enum cuvsDatasetMemType_t:
        CUVS_DATASET_MEM_TYPE_HOST
        CUVS_DATASET_MEM_TYPE_DEVICE

    cdef struct cuvsDataset:
        pass
    ctypedef cuvsDataset* cuvsDataset_t

    cuvsError_t cuvsDatasetCreate(cuvsDataset_t* dataset)

    cuvsError_t cuvsDatasetMakePadded(cuvsResources_t res,
                                      DLManagedTensor* dataset,
                                      cuvsDatasetMemType_t target_mem_type,
                                      cuvsDataset_t* padded_dataset)

    cuvsError_t cuvsDatasetMakePaddedView(cuvsResources_t res,
                                          DLManagedTensor* dataset,
                                          cuvsDataset_t* padded_dataset)

    cuvsError_t cuvsDatasetMakeStandardView(cuvsResources_t res,
                                            DLManagedTensor* dataset,
                                            cuvsDataset_t* standard_dataset)

    cuvsError_t cuvsDatasetDestroy(cuvsDataset_t dataset)

    cuvsError_t cuvsDatasetGetMemType(cuvsDataset_t dataset,
                                      cuvsDatasetMemType_t* mem_type)

    cuvsError_t cuvsDatasetGetLayout(cuvsDataset_t dataset,
                                     cuvsDatasetLayout_t* layout)

    cuvsError_t cuvsDatasetGetIsOwning(cuvsDataset_t dataset, bool* is_owning)

    cuvsError_t cuvsDatasetGetDtype(cuvsDataset_t dataset, DLDataType* dtype)


cdef class Dataset:
    cdef cuvsDataset_t dataset
    cdef object _source


cdef Dataset make_device_padded_dataset_handle(
    cuvsResources_t res,
    DLManagedTensor* dataset_dlpack)
