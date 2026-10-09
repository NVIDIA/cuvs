#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLManagedTensor


cdef extern from "cuvs/cluster/soar.h" nogil:
    ctypedef struct cuvsSoarParams:
        float lambda_ "lambda"

    ctypedef cuvsSoarParams* cuvsSoarParams_t

    cuvsError_t cuvsSoarParamsCreate(cuvsSoarParams_t* params)

    cuvsError_t cuvsSoarParamsDestroy(cuvsSoarParams_t params)

    cuvsError_t cuvsSoarPredict(cuvsResources_t res,
                                cuvsSoarParams_t params,
                                DLManagedTensor* dataset,
                                DLManagedTensor* centroids,
                                DLManagedTensor* labels,
                                DLManagedTensor* soar_labels)
