#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


from __future__ import absolute_import, division, print_function

import sys

import numpy as np

from cuvs_bench._bin_format import read_bin_header, write_bin_header


def read_fbin(fname):
    itemsize = np.dtype(np.float32).itemsize
    n_rows, n_cols, header_bytes = read_bin_header(fname, itemsize)
    shape = (n_rows, n_cols)
    if float(n_rows) * n_cols * itemsize > 2_000_000_000:
        data = np.memmap(
            fname, dtype=np.float32, offset=header_bytes, mode="r"
        ).reshape(shape)
    else:
        data = np.fromfile(
            fname, dtype=np.float32, offset=header_bytes
        ).reshape(shape)
    return data


def write_bin(fname, data, *, force_uint64=False):
    with open(fname, "wb") as f:
        write_bin_header(f, data.shape[0], data.shape[1])
        data.tofile(f)


if len(sys.argv) != 3:
    print(
        "usage: %s input.fbin output.f16bin" % (sys.argv[0]),
        file=sys.stderr,
    )
    sys.exit(-1)

data = read_fbin(sys.argv[1]).astype(np.float16)
write_bin(sys.argv[2], data)
