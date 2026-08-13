/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "compute_distance_standard.hpp"

#include <cuvs/distance/distance.hpp>
#include <raft/util/pow2_utils.cuh>

#include <type_traits>

namespace cuvs::neighbors::cagra::detail {

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename QueryT>
struct standard_dataset_descriptor_t : public dataset_descriptor_base_t<DataT, IndexT, DistanceT> {
  using base_type = dataset_descriptor_base_t<DataT, IndexT, DistanceT>;
  using QUERY_T   = QueryT;
  using base_type::args;
  using base_type::smem_ws_size_in_bytes;
  using typename base_type::args_t;
  using typename base_type::DATA_T;
  using typename base_type::DISTANCE_T;
  using typename base_type::INDEX_T;
  using typename base_type::LOAD_T;
  constexpr static inline auto kTeamSize        = TeamSize;
  constexpr static inline auto kDatasetBlockDim = DatasetBlockDim;

  static constexpr RAFT_INLINE_FUNCTION auto ptr(const args_t& args) noexcept
    -> const DATA_T* const&
  {
    return (const DATA_T* const&)(args.extra_ptr1);
  }
  static constexpr RAFT_INLINE_FUNCTION auto ptr(args_t& args) noexcept -> const DATA_T*&
  {
    return (const DATA_T*&)(args.extra_ptr1);
  }

  static constexpr RAFT_INLINE_FUNCTION auto dataset_norms_ptr(const args_t& args) noexcept
    -> const DISTANCE_T* const&
  {
    return (const DISTANCE_T* const&)(args.extra_ptr2);
  }
  static constexpr RAFT_INLINE_FUNCTION auto dataset_norms_ptr(args_t& args) noexcept
    -> const DISTANCE_T*&
  {
    return (const DISTANCE_T*&)(args.extra_ptr2);
  }

  static constexpr RAFT_INLINE_FUNCTION auto ld(const args_t& args) noexcept -> const uint32_t&
  {
    return args.extra_word1;
  }
  static constexpr RAFT_INLINE_FUNCTION auto ld(args_t& args) noexcept -> uint32_t&
  {
    return args.extra_word1;
  }

  _RAFT_HOST_DEVICE standard_dataset_descriptor_t(const DATA_T* ptr,
                                                  INDEX_T size,
                                                  uint32_t dim,
                                                  uint32_t ld,
                                                  const DISTANCE_T* dataset_norms = nullptr)
    : base_type(size, dim, raft::Pow2<TeamSize>::Log2, get_smem_ws_size_in_bytes(dim))
  {
    standard_dataset_descriptor_t::ptr(args)               = ptr;
    standard_dataset_descriptor_t::ld(args)                = ld;
    standard_dataset_descriptor_t::dataset_norms_ptr(args) = dataset_norms;
    static_assert(sizeof(*this) == sizeof(base_type));
    static_assert(alignof(standard_dataset_descriptor_t) == alignof(base_type));
  }

  // Public static method to compute smem size without constructing descriptor
  RAFT_INLINE_FUNCTION constexpr static auto get_smem_ws_size_in_bytes(uint32_t dim) -> uint32_t
  {
    return sizeof(standard_dataset_descriptor_t) +
           raft::round_up_safe<uint32_t>(dim, DatasetBlockDim) * sizeof(QUERY_T);
  }
};

template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT>
RAFT_KERNEL __launch_bounds__(1, 1)
  standard_dataset_descriptor_init_kernel(dataset_descriptor_base_t<DataT, IndexT, DistanceT>* out,
                                          const DataT* ptr,
                                          IndexT size,
                                          uint32_t dim,
                                          uint32_t ld,
                                          const DistanceT* dataset_norms = nullptr)
{
  using query_t =
    std::conditional_t<Metric == cuvs::distance::DistanceType::BitwiseHamming, DataT, float>;
  using desc_type =
    standard_dataset_descriptor_t<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT, query_t>;
  using base_type = typename desc_type::base_type;

  new (out) desc_type(ptr, size, dim, ld, dataset_norms);
}

template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT>
dataset_descriptor_host<DataT, IndexT, DistanceT>
standard_descriptor_spec<Metric, TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>::init_(
  const cagra::search_params& params,
  const DataT* ptr,
  IndexT size,
  uint32_t dim,
  uint32_t ld,
  const DistanceT* dataset_norms)
{
  using query_t =
    std::conditional_t<Metric == cuvs::distance::DistanceType::BitwiseHamming, DataT, float>;
  using desc_type =
    standard_dataset_descriptor_t<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT, query_t>;
  using base_type = typename desc_type::base_type;

  RAFT_EXPECTS(Metric != cuvs::distance::DistanceType::CosineExpanded || dataset_norms != nullptr,
               "Dataset norms must be provided for CosineExpanded metric");

  return host_type{desc_type{ptr, size, dim, ld, dataset_norms},
                   [=](dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dev_ptr,
                       rmm::cuda_stream_view stream) {
                     standard_dataset_descriptor_init_kernel<Metric,
                                                             TeamSize,
                                                             DatasetBlockDim,
                                                             DataT,
                                                             IndexT,
                                                             DistanceT>
                       <<<1, 1, 0, stream>>>(dev_ptr, ptr, size, dim, ld, dataset_norms);
                     RAFT_CUDA_TRY(cudaPeekAtLastError());
                   },
                   Metric,
                   DatasetBlockDim,
                   false,  // is_vpq
                   0,      // pq_bits
                   0};     // pq_len
}

}  // namespace cuvs::neighbors::cagra::detail
