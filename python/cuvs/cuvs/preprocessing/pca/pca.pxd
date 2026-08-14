#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from libcpp cimport bool

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLManagedTensor


cdef extern from "cuvs/preprocessing/pca.h" nogil:
    ctypedef enum cuvsPcaSolver:
        CUVS_PCA_COV_EIG_DQ
        CUVS_PCA_COV_EIG_JACOBI

    ctypedef struct cuvsPcaParams:
        int n_components
        bool copy
        bool whiten
        cuvsPcaSolver algorithm
        float tol
        int n_iterations

    ctypedef cuvsPcaParams* cuvsPcaParams_t

    cuvsError_t cuvsPcaParamsCreate(cuvsPcaParams_t* params)

    cuvsError_t cuvsPcaParamsDestroy(cuvsPcaParams_t params)

    cuvsError_t cuvsPcaFit(cuvsResources_t res,
                           cuvsPcaParams_t params,
                           DLManagedTensor* input,
                           DLManagedTensor* components,
                           DLManagedTensor* explained_var,
                           DLManagedTensor* explained_var_ratio,
                           DLManagedTensor* singular_vals,
                           DLManagedTensor* mu,
                           DLManagedTensor* noise_vars,
                           bool flip_signs_based_on_U)

    cuvsError_t cuvsPcaFitTransform(cuvsResources_t res,
                                    cuvsPcaParams_t params,
                                    DLManagedTensor* input,
                                    DLManagedTensor* trans_input,
                                    DLManagedTensor* components,
                                    DLManagedTensor* explained_var,
                                    DLManagedTensor* explained_var_ratio,
                                    DLManagedTensor* singular_vals,
                                    DLManagedTensor* mu,
                                    DLManagedTensor* noise_vars,
                                    bool flip_signs_based_on_U)

    cuvsError_t cuvsPcaTransform(cuvsResources_t res,
                                 cuvsPcaParams_t params,
                                 DLManagedTensor* input,
                                 DLManagedTensor* components,
                                 DLManagedTensor* singular_vals,
                                 DLManagedTensor* mu,
                                 DLManagedTensor* trans_input)

    cuvsError_t cuvsPcaInverseTransform(cuvsResources_t res,
                                        cuvsPcaParams_t params,
                                        DLManagedTensor* trans_input,
                                        DLManagedTensor* components,
                                        DLManagedTensor* singular_vals,
                                        DLManagedTensor* mu,
                                        DLManagedTensor* output)
