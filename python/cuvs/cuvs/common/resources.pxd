#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from cuvs.common.c_api cimport cuvsResources_t


cdef class Resources:
    cdef cuvsResources_t c_obj
    cdef bint _owns_resource
