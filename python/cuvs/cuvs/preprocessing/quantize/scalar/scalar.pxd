#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor


cdef extern from "cuvs/preprocessing/quantize/scalar.h" nogil:
    ctypedef struct cuvsScalarQuantizerParams:
        float quantile

    ctypedef cuvsScalarQuantizerParams* cuvsScalarQuantizerParams_t

    cuvsError_t cuvsScalarQuantizerParamsCreate(
        cuvsScalarQuantizerParams_t* params)

    cuvsError_t cuvsScalarQuantizerParamsDestroy(
        cuvsScalarQuantizerParams_t params)

    ctypedef struct cuvsScalarQuantizer:
        pass

    ctypedef cuvsScalarQuantizer* cuvsScalarQuantizer_t

    cuvsError_t cuvsScalarQuantizerCreate(
        cuvsScalarQuantizer_t* quantizer)

    cuvsError_t cuvsScalarQuantizerDestroy(
        cuvsScalarQuantizer_t quantizer)

    cuvsError_t cuvsScalarQuantizerGetBounds(
        cuvsScalarQuantizer_t quantizer, double* min, double* max)

    cuvsError_t cuvsScalarQuantizerTrain(cuvsResources_t res,
                                         cuvsScalarQuantizerParams_t params,
                                         DLManagedTensor* dataset,
                                         cuvsScalarQuantizer_t quantizer)

    cuvsError_t cuvsScalarQuantizerTransform(cuvsResources_t res,
                                             cuvsScalarQuantizer_t quantizer,
                                             DLManagedTensor* dataset,
                                             DLManagedTensor* out)

    cuvsError_t cuvsScalarQuantizerInverseTransform(cuvsResources_t res,
                                                    cuvsScalarQuantizer_t q,
                                                    DLManagedTensor* dataset,
                                                    DLManagedTensor* out)
