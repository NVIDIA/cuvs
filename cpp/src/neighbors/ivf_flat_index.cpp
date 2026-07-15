/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/export.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>

namespace cuvs::neighbors::ivf_flat {

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
    centers_(raft::make_device_matrix<float, uint32_t>(res, n_lists, dim)),
    center_norms_(std::nullopt),
    data_ptrs_{raft::make_device_vector<T*, uint32_t>(res, n_lists)},
    inds_ptrs_{raft::make_device_vector<IdxT*, uint32_t>(res, n_lists)},
    accum_sorted_sizes_{raft::make_host_vector<IdxT, uint32_t>(n_lists + 1)}
{
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
  return centers_.extent(1);
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
  RAFT_EXPECTS(                                       //
    (centers_.extent(0) == list_sizes_.extent(0)) &&  //
      (!center_norms_.has_value() || centers_.extent(0) == center_norms_->extent(0)),
    "inconsistent number of lists (clusters)");
}

template struct CUVS_EXPORT index<float, uint32_t>;  // Used for refine function
template struct CUVS_EXPORT index<float, int64_t>;
template struct CUVS_EXPORT index<half, int64_t>;
template struct CUVS_EXPORT index<int8_t, int64_t>;
template struct CUVS_EXPORT index<uint8_t, int64_t>;

}  // namespace cuvs::neighbors::ivf_flat
