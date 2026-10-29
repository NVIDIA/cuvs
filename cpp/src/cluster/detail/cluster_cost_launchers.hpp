/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <rmm/device_uvector.hpp>

namespace cuvs::cluster::kmeans::detail {
template <typename InputIterator,
          typename OutputT,
          typename MainOpT,
          typename ReductionOpT,
          typename IndexT = int>
void computeClusterCost(raft::resources const& handle,
                        InputIterator input,
                        IndexT size,
                        rmm::device_uvector<char>& workspace,
                        raft::device_scalar_view<OutputT> clusterCost,
                        MainOpT main_op,
                        ReductionOpT reduction_op);
}  // namespace cuvs::cluster::kmeans::detail
