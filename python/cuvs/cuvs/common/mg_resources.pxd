#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from cuda.bindings.cyruntime cimport cudaStream_t

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t


cdef class MultiGpuResources:
    cdef cuvsResources_t c_obj
    cdef bint _owns_resource
