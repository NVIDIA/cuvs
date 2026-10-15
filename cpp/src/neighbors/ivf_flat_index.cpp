/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <cuvs/core/export.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <limits>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>
#include <type_traits>

namespace cuvs::neighbors::ivf_flat {
namespace {
uint32_t expanded_binary_dim(uint32_t dim)
{
  RAFT_EXPECTS(dim <= std::numeric_limits<uint32_t>::max() / 8,
               "binary dimensionality is too large for expanded center statistics");
  return dim * 8;
}
}  // namespace

template <typename T, typename IdxT>
index<T, IdxT>::index(raft::resources const& res)
  : index(res, cuvs::distance::DistanceType::L2Expanded, 0, false, false, 0)
{
}

template <typename T, typename IdxT>
index<T, IdxT>::index(raft::resources const& res, const index_params& params, uint32_t dim)
  : index(res,
          params.metric,
          params.n_lists,
          params.adaptive_centers,
          params.conservative_memory_allocation,
          dim)
{
}

template <typename T, typename IdxT>
index<T, IdxT>::index(raft::resources const& res,
                      cuvs::distance::DistanceType metric,
                      uint32_t n_lists,
                      bool adaptive_centers,
                      bool conservative_memory_allocation,
                      uint32_t dim)
  : cuvs::neighbors::index(),
    veclen_(calculate_veclen(dim)),
    metric_(metric),
    adaptive_centers_(adaptive_centers),
    conservative_memory_allocation_{conservative_memory_allocation},
    lists_{n_lists},
    list_sizes_{raft::make_device_vector<uint32_t, uint32_t>(res, n_lists)},
    centers_(metric != cuvs::distance::DistanceType::BitwiseHamming
               ? raft::make_device_matrix<float, uint32_t>(res, n_lists, dim)
               : raft::make_device_matrix<float, uint32_t>(res, 0, 0)),
    binary_centers_(metric != cuvs::distance::DistanceType::BitwiseHamming
                      ? raft::make_device_matrix<uint8_t, int64_t>(res, 0, 0)
                      : raft::make_device_matrix<uint8_t, int64_t>(res, n_lists, dim)),
    binary_center_counts_(
      metric == cuvs::distance::DistanceType::BitwiseHamming && adaptive_centers
        ? raft::make_device_matrix<uint32_t, int64_t>(res, n_lists, expanded_binary_dim(dim))
        : raft::make_device_matrix<uint32_t, int64_t>(res, 0, 0)),
    center_norms_(std::nullopt),
    binary_index_(metric == cuvs::distance::DistanceType::BitwiseHamming),
    data_ptrs_{raft::make_device_vector<T*, uint32_t>(res, n_lists)},
    inds_ptrs_{raft::make_device_vector<IdxT*, uint32_t>(res, n_lists)},
    accum_sorted_sizes_{raft::make_host_vector<IdxT, uint32_t>(n_lists + 1)}
{
  if (metric == cuvs::distance::DistanceType::BitwiseHamming && !std::is_same_v<T, uint8_t>) {
    RAFT_FAIL("BitwiseHamming distance is only supported with uint8_t data type, got %s",
              typeid(T).name());
  }

  if (binary_center_counts_.size() != 0) {
    RAFT_CUDA_TRY(cudaMemsetAsync(binary_center_counts_.data_handle(),
                                  0,
                                  binary_center_counts_.size() * sizeof(uint32_t),
                                  raft::resource::get_cuda_stream(res).get()));
  }
  check_consistency();
  accum_sorted_sizes_(n_lists) = 0;
}

template <typename T, typename IdxT>
uint32_t index<T, IdxT>::veclen() const noexcept
{
  return veclen_;
}

template <typename T, typename IdxT>
cuvs::distance::DistanceType index<T, IdxT>::metric() const noexcept
{
  return metric_;
}

template <typename T, typename IdxT>
bool index<T, IdxT>::adaptive_centers() const noexcept
{
  return adaptive_centers_;
}

template <typename T, typename IdxT>
raft::device_vector_view<uint32_t, uint32_t> index<T, IdxT>::list_sizes() noexcept
{
  return list_sizes_.view();
}

template <typename T, typename IdxT>
raft::device_vector_view<const uint32_t, uint32_t> index<T, IdxT>::list_sizes() const noexcept
{
  return list_sizes_.view();
}

template <typename T, typename IdxT>
raft::device_matrix_view<float, uint32_t, raft::row_major> index<T, IdxT>::centers() noexcept
{
  return centers_.view();
}

template <typename T, typename IdxT>
raft::device_matrix_view<const float, uint32_t, raft::row_major> index<T, IdxT>::centers()
  const noexcept
{
  return centers_.view();
}

template <typename T, typename IdxT>
raft::device_matrix_view<uint8_t, int64_t, raft::row_major>
index<T, IdxT>::binary_centers() noexcept
{
  return binary_centers_.view();
}

template <typename T, typename IdxT>
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> index<T, IdxT>::binary_centers()
  const noexcept
{
  return binary_centers_.view();
}
template <typename T, typename IdxT>
raft::device_matrix_view<uint32_t, int64_t, raft::row_major>
index<T, IdxT>::binary_center_counts() noexcept
{
  return binary_center_counts_.view();
}

template <typename T, typename IdxT>
raft::device_matrix_view<const uint32_t, int64_t, raft::row_major>
index<T, IdxT>::binary_center_counts() const noexcept
{
  return binary_center_counts_.view();
}

template <typename T, typename IdxT>
std::optional<raft::device_vector_view<float, uint32_t>> index<T, IdxT>::center_norms() noexcept
{
  if (center_norms_.has_value()) {
    return std::make_optional<raft::device_vector_view<float, uint32_t>>(center_norms_->view());
  } else {
    return std::nullopt;
  }
}

template <typename T, typename IdxT>
std::optional<raft::device_vector_view<const float, uint32_t>> index<T, IdxT>::center_norms()
  const noexcept
{
  if (center_norms_.has_value()) {
    return std::make_optional<raft::device_vector_view<const float, uint32_t>>(
      center_norms_->view());
  } else {
    return std::nullopt;
  }
}

template <typename T, typename IdxT>
auto index<T, IdxT>::accum_sorted_sizes() noexcept -> raft::host_vector_view<IdxT, uint32_t>
{
  return accum_sorted_sizes_.view();
}

template <typename T, typename IdxT>
[[nodiscard]] auto index<T, IdxT>::accum_sorted_sizes() const noexcept
  -> raft::host_vector_view<const IdxT, uint32_t>
{
  return accum_sorted_sizes_.view();
}

template <typename T, typename IdxT>
IdxT index<T, IdxT>::size() const noexcept
{
  return accum_sorted_sizes()(n_lists());
}

template <typename T, typename IdxT>
uint32_t index<T, IdxT>::dim() const noexcept
{
  if (binary_index_) {
    return binary_centers_.extent(1);
  } else {
    return centers_.extent(1);
  }
}

template <typename T, typename IdxT>
uint32_t index<T, IdxT>::n_lists() const noexcept
{
  return lists_.size();
}

template <typename T, typename IdxT>
raft::device_vector_view<T*, uint32_t> index<T, IdxT>::data_ptrs() noexcept
{
  return data_ptrs_.view();
}

template <typename T, typename IdxT>
raft::device_vector_view<T* const, uint32_t> index<T, IdxT>::data_ptrs() const noexcept
{
  return data_ptrs_.view();
}

template <typename T, typename IdxT>
raft::device_vector_view<IdxT*, uint32_t> index<T, IdxT>::inds_ptrs() noexcept
{
  return inds_ptrs_.view();
}

template <typename T, typename IdxT>
raft::device_vector_view<IdxT* const, uint32_t> index<T, IdxT>::inds_ptrs() const noexcept
{
  return inds_ptrs_.view();
}

template <typename T, typename IdxT>
bool index<T, IdxT>::conservative_memory_allocation() const noexcept
{
  return conservative_memory_allocation_;
}

template <typename T, typename IdxT>
void index<T, IdxT>::allocate_center_norms(raft::resources const& res)
{
  switch (metric_) {
    case cuvs::distance::DistanceType::L2Expanded:
    case cuvs::distance::DistanceType::L2SqrtExpanded:
    case cuvs::distance::DistanceType::L2Unexpanded:
    case cuvs::distance::DistanceType::L2SqrtUnexpanded:
    case cuvs::distance::DistanceType::CosineExpanded:
      center_norms_ = raft::make_device_vector<float, uint32_t>(res, n_lists());
      break;
    default: center_norms_ = std::nullopt;
  }
}

template <typename T, typename IdxT>
std::vector<std::shared_ptr<list_data<T, IdxT>>>& index<T, IdxT>::lists() noexcept
{
  return lists_;
}

template <typename T, typename IdxT>
const std::vector<std::shared_ptr<list_data<T, IdxT>>>& index<T, IdxT>::lists() const noexcept
{
  return lists_;
}

template <typename T, typename IdxT>
void index<T, IdxT>::check_consistency()
{
  auto n_lists = lists_.size();
  RAFT_EXPECTS(dim() % veclen_ == 0, "dimensionality is not a multiple of the veclen");
  RAFT_EXPECTS(list_sizes_.extent(0) == n_lists, "inconsistent list size");
  RAFT_EXPECTS(data_ptrs_.extent(0) == n_lists, "inconsistent list size");
  RAFT_EXPECTS(inds_ptrs_.extent(0) == n_lists, "inconsistent list size");
  if (binary_index_) {
    RAFT_EXPECTS(binary_centers_.extent(0) == list_sizes_.extent(0),
                 "inconsistent number of lists (clusters)");
    RAFT_EXPECTS(!adaptive_centers_ || (binary_center_counts_.extent(0) == int64_t(n_lists) &&
                                        binary_center_counts_.extent(1) == int64_t(dim()) * 8),
                 "inconsistent binary center counts");
  } else {
    RAFT_EXPECTS(                                       //
      (centers_.extent(0) == list_sizes_.extent(0)) &&  //
        (!center_norms_.has_value() || centers_.extent(0) == center_norms_->extent(0)),
      "inconsistent number of lists (clusters)");
  }
}

template <typename T, typename IdxT>
bool index<T, IdxT>::binary_index() const noexcept
{
  return binary_index_;
}

template struct CUVS_EXPORT index<float, uint32_t>;  // Used for refine function
template struct CUVS_EXPORT index<float, int64_t>;
template struct CUVS_EXPORT index<half, int64_t>;
template struct CUVS_EXPORT index<int8_t, int64_t>;
template struct CUVS_EXPORT index<uint8_t, int64_t>;

}  // namespace cuvs::neighbors::ivf_flat
