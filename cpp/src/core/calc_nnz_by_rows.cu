/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/sparse/convert/csr.cuh>

template void raft::sparse::convert::detail::calc_nnz_by_rows<unsigned int, long, long>(
  raft::resources const&, unsigned int const*, long, long, long*, size_t&, long&);
