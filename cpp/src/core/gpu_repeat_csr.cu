/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/sparse/convert/csr.cuh>

template void raft::sparse::convert::detail::gpu_repeat_csr<long, long>(
  raft::resources const&, long const*, long const*, long, long, long*, long*);
