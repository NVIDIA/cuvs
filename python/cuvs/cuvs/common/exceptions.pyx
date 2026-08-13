#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from cuvs.common.c_api cimport cuvsError_t, cuvsGetLastErrorText


class CuvsException(Exception):
    pass


def get_last_error_text():
    """ returns the last error description from the cuvs c-api """
    cdef const char* c_err = cuvsGetLastErrorText()
    if c_err is not NULL:
        return c_err.decode("utf8", "ignore")


def check_cuvs(status: cuvsError_t):
    """ Converts a status code into an exception """
    if status == cuvsError_t.CUVS_ERROR:
        raise CuvsException(get_last_error_text())
