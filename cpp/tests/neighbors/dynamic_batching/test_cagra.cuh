/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../dynamic_batching.cuh"

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>

namespace cuvs::neighbors::dynamic_batching {

namespace {

template <typename T, typename IdxT>
auto build_cagra_with_dataset(raft::resources const& res,
                              cagra::index_params const& params,
                              raft::device_matrix_view<const T, int64_t, raft::row_major> dataset)
  -> cagra::device_padded_index<T, IdxT>
{
  auto padded = cuvs::neighbors::make_device_padded_dataset_view(res, dataset);
  auto index  = cagra::build(res, params, padded);
  index       = cagra::update_dataset(res, std::move(index), padded);
  return index;
}

}  // namespace

using cagra_F32 = dynamic_batching_test<float,
                                        uint32_t,
                                        cagra::device_padded_index<float, uint32_t>,
                                        build_cagra_with_dataset<float, uint32_t>,
                                        cagra::search>;

using cagra_U8 = dynamic_batching_test<uint8_t,
                                       uint32_t,
                                       cagra::device_padded_index<uint8_t, uint32_t>,
                                       build_cagra_with_dataset<uint8_t, uint32_t>,
                                       cagra::search>;

template <typename fixture>
static void set_default_cagra_params(fixture& that)
{
  that.build_params_upsm.intermediate_graph_degree = 128;
  that.build_params_upsm.graph_degree              = 64;
  that.search_params_upsm.itopk_size =
    std::clamp<int64_t>(raft::bound_by_power_of_two(that.ps.k) * 16, 128, 512);
}

}  // namespace cuvs::neighbors::dynamic_batching
