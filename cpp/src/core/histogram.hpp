/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>
#include <raft/stats/stats_types.hpp>

namespace cuvs::detail {

/** Identity-bin histograms shared by graph and IVF construction. */
template <typename DataT, typename IdxT>
void histogram(raft::stats::HistType type,
               int* bins,
               IdxT nbins,
               const DataT* data,
               IdxT nrows,
               IdxT ncols,
               cudaStream_t stream);

}  // namespace cuvs::detail
