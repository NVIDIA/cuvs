/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute_distance.hpp"

#include <cuvs/distance/distance.hpp>
#include <raft/util/cudart_utils.hpp>

#include <type_traits>

namespace cuvs::neighbors::cagra::detail {

inline auto select_supported_vpq_smem_dtype(const cagra::search_params& params)
  -> cuvs::neighbors::cagra::internal_dtype
{
  if (params.smem_dtype == cuvs::neighbors::cagra::internal_dtype::E5M2 &&
      raft::getComputeCapability().first < 9) {
    return cuvs::neighbors::cagra::internal_dtype::F16;
  }
  return params.smem_dtype;
}

template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PqBits,
          uint32_t PqLen,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          cuvs::neighbors::cagra::internal_dtype SmemDType>
struct vpq_descriptor_spec : public instance_spec<DataT, IndexT, DistanceT> {
  using base_type = instance_spec<DataT, IndexT, DistanceT>;
  using typename base_type::data_type;
  using typename base_type::distance_type;
  using typename base_type::host_type;
  using typename base_type::index_type;

  template <typename DatasetT>
  constexpr static inline auto accepts_dataset()
    -> std::enable_if_t<is_vpq_dataset_v<DatasetT>, bool>
  {
    return std::is_same_v<typename DatasetT::math_type, CodebookT>;
  }

  template <typename DatasetT>
  constexpr static inline auto accepts_dataset()
    -> std::enable_if_t<!is_vpq_dataset_v<DatasetT>, bool>
  {
    return false;
  }

  template <typename DatasetT>
  static auto init(const cagra::search_params& params,
                   const DatasetT& dataset,
                   cuvs::distance::DistanceType metric,
                   const DistanceT* dataset_norms = nullptr) -> host_type
  {
    return init_(params,
                 dataset.data.data_handle(),
                 dataset.encoded_row_length(),
                 dataset.vq_code_book.data_handle(),
                 dataset.pq_code_book.data_handle(),
                 IndexT(dataset.n_rows()),
                 dataset.dim());
  }

  template <typename DatasetT>
  static auto priority(const cagra::search_params& params,
                       const DatasetT& dataset,
                       cuvs::distance::DistanceType metric) -> double
  {
    // If explicit team_size is specified and doesn't match the instance, discard it
    if (params.team_size != 0 && TeamSize != params.team_size) { return -1.0; }
    if (cuvs::distance::DistanceType::L2Expanded != metric) { return -1.0; }
    // Match codebook params
    if (dataset.pq_bits() != PqBits) { return -1.0; }
    if (dataset.pq_len() != PqLen) { return -1.0; }
    if (select_supported_vpq_smem_dtype(params) != SmemDType) { return -1.0; }
    // Keep auto-selection on the tuned VPQ diagonal while allowing explicit team_size requests to
    // use the expanded team_size / dataset_block_dim grid.
    constexpr std::uint32_t auto_dataset_block_dim_per_team = PqLen == 8 ? 32 : 16;
    if (params.team_size == 0 && DatasetBlockDim != TeamSize * auto_dataset_block_dim_per_team) {
      return -1.0;
    }
    // Otherwise, favor the closest dataset dimensionality.
    constexpr std::uint32_t preferred_load_elmes_per_thread =
      16; /*magic number that is good based on experiments.*/
    return 1.0 / (0.1 + std::abs(double(dataset.dim()) - double(DatasetBlockDim))) * TeamSize +
           1.0 / (0.1 + std::abs(double(dataset.dim()) / TeamSize / PqLen -
                                 preferred_load_elmes_per_thread));
  }

 private:
  static dataset_descriptor_host<DataT, IndexT, DistanceT> init_(
    const cagra::search_params& params,
    const std::uint8_t* encoded_dataset_ptr,
    uint32_t encoded_dataset_dim,
    const CodebookT* vq_code_book_ptr,
    const CodebookT* pq_code_book_ptr,
    IndexT size,
    uint32_t dim);
};

}  // namespace cuvs::neighbors::cagra::detail
