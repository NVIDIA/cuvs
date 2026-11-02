#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from cuvs.cluster.kmeans.kmeans cimport cuvsKMeansParams_t
from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLManagedTensor


cdef extern from "cuvs/cluster/mg_kmeans.h" nogil:
    cuvsError_t cuvsMultiGpuKMeansFit(cuvsResources_t res,
                                      cuvsKMeansParams_t params,
                                      DLManagedTensor* X,
                                      DLManagedTensor* sample_weight,
                                      DLManagedTensor* centroids,
                                      double* inertia,
                                      int* n_iter) except +
