#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from libc.stdint cimport uint32_t, uintptr_t
from libcpp cimport bool

from cuvs.cluster.kmeans.kmeans cimport cuvsKMeansType
from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor


cdef extern from "cuvs/preprocessing/quantize/pq.h" nogil:

    ctypedef struct cuvsProductQuantizerParams:
        uint32_t pq_bits
        uint32_t pq_dim
        bool use_subspaces
        bool train_coarse
        uint32_t vq_n_centers
        uint32_t kmeans_n_iters
        cuvsKMeansType pq_kmeans_type
        uint32_t max_train_points_per_pq_code
        uint32_t max_train_points_per_vq_cluster

    ctypedef cuvsProductQuantizerParams* cuvsProductQuantizerParams_t

    ctypedef struct cuvsProductQuantizer:
        uintptr_t addr
        DLDataType dtype

    ctypedef cuvsProductQuantizer* cuvsProductQuantizer_t

    cuvsError_t cuvsProductQuantizerParamsCreate(
        cuvsProductQuantizerParams_t* params)

    cuvsError_t cuvsProductQuantizerParamsDestroy(
        cuvsProductQuantizerParams_t params)

    cuvsError_t cuvsProductQuantizerCreate(cuvsProductQuantizer_t* quantizer)

    cuvsError_t cuvsProductQuantizerDestroy(cuvsProductQuantizer_t quantizer)

    cuvsError_t cuvsProductQuantizerTransform(cuvsResources_t res,
                                              cuvsProductQuantizer_t quantizer,
                                              DLManagedTensor* dataset,
                                              DLManagedTensor* codes_out,
                                              DLManagedTensor* vq_labels)
    cuvsError_t cuvsProductQuantizerInverseTransform(
        cuvsResources_t res, cuvsProductQuantizer_t quantizer,
        DLManagedTensor* pq_codes, DLManagedTensor* out,
        DLManagedTensor* vq_labels)

    cuvsError_t cuvsProductQuantizerBuild(cuvsResources_t res,
                                          cuvsProductQuantizerParams_t params,
                                          DLManagedTensor* dataset,
                                          cuvsProductQuantizer_t quantizer)

    cuvsError_t cuvsProductQuantizerGetPqBits(cuvsProductQuantizer_t quantizer,
                                              uint32_t* pq_bits)

    cuvsError_t cuvsProductQuantizerGetPqDim(cuvsProductQuantizer_t quantizer,
                                             uint32_t* pq_dim)

    cuvsError_t cuvsProductQuantizerGetPqCodebook(
        cuvsProductQuantizer_t quantizer, DLManagedTensor* pq_codebook)

    cuvsError_t cuvsProductQuantizerGetVqCodebook(
        cuvsProductQuantizer_t quantizer, DLManagedTensor* vq_codebook)

    cuvsError_t cuvsProductQuantizerGetEncodedDim(
        cuvsProductQuantizer_t quantizer, uint32_t* encoded_dim)

    cuvsError_t cuvsProductQuantizerGetTrainCoarse(
        cuvsProductQuantizer_t quantizer, bool* train_coarse)
