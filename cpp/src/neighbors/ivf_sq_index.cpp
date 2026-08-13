/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/ivf_sq.hpp>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <cstring>

namespace cuvs::neighbors::ivf_sq {

template <typename CodeT>
index<CodeT>::index(raft::resources const& res)
  : index(res, cuvs::distance::DistanceType::L2Expanded, 0, 0, false)
{
}

template <typename CodeT>
index<CodeT>::index(raft::resources const& res, const index_params& params, uint32_t dim)
  : index(res, params.metric, params.n_lists, dim, params.conservative_memory_allocation)
{
}

template <typename CodeT>
index<CodeT>::index(raft::resources const& res,
                    cuvs::distance::DistanceType metric,
                    uint32_t n_lists,
                    uint32_t dim,
                    bool conservative_memory_allocation)
  : cuvs::neighbors::index(),
    metric_(metric),
    conservative_memory_allocation_(conservative_memory_allocation),
    lists_{n_lists},
    list_sizes_{raft::make_device_vector<uint32_t, uint32_t>(res, n_lists)},
    centers_(raft::make_device_matrix<float, uint32_t>(res, n_lists, dim)),
    center_norms_(std::nullopt),
    sq_vmin_{raft::make_device_vector<float, uint32_t>(res, dim)},
    sq_delta_{raft::make_device_vector<float, uint32_t>(res, dim)},
    data_ptrs_{raft::make_device_vector<CodeT*, uint32_t>(res, n_lists)},
    inds_ptrs_{raft::make_device_vector<int64_t*, uint32_t>(res, n_lists)},
    accum_sorted_sizes_{raft::make_host_vector<int64_t, uint32_t>(n_lists + 1)}
{
  check_consistency();
  auto stream = raft::resource::get_cuda_stream(res);
  std::memset(accum_sorted_sizes_.data_handle(), 0, accum_sorted_sizes_.size() * sizeof(int64_t));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(list_sizes_.data_handle(), 0, list_sizes_.size() * sizeof(uint32_t), stream));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(data_ptrs_.data_handle(), 0, data_ptrs_.size() * sizeof(CodeT*), stream));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(inds_ptrs_.data_handle(), 0, inds_ptrs_.size() * sizeof(int64_t*), stream));
}

template <typename CodeT>
cuvs::distance::DistanceType index<CodeT>::metric() const noexcept
{
  return metric_;
}

template <typename CodeT>
int64_t index<CodeT>::size() const noexcept
{
  return accum_sorted_sizes()(n_lists());
}

template <typename CodeT>
uint32_t index<CodeT>::dim() const noexcept
{
  return centers_.extent(1);
}

template <typename CodeT>
uint32_t index<CodeT>::n_lists() const noexcept
{
  return lists_.size();
}

template <typename CodeT>
bool index<CodeT>::conservative_memory_allocation() const noexcept
{
  return conservative_memory_allocation_;
}

template <typename CodeT>
raft::device_vector_view<uint32_t, uint32_t> index<CodeT>::list_sizes() noexcept
{
  return list_sizes_.view();
}

template <typename CodeT>
raft::device_vector_view<const uint32_t, uint32_t> index<CodeT>::list_sizes() const noexcept
{
  return list_sizes_.view();
}

template <typename CodeT>
raft::device_matrix_view<float, uint32_t, raft::row_major> index<CodeT>::centers() noexcept
{
  return centers_.view();
}

template <typename CodeT>
raft::device_matrix_view<const float, uint32_t, raft::row_major> index<CodeT>::centers()
  const noexcept
{
  return centers_.view();
}

template <typename CodeT>
std::optional<raft::device_vector_view<float, uint32_t>> index<CodeT>::center_norms() noexcept
{
  if (center_norms_.has_value()) {
    return std::make_optional<raft::device_vector_view<float, uint32_t>>(center_norms_->view());
  } else {
    return std::nullopt;
  }
}

template <typename CodeT>
std::optional<raft::device_vector_view<const float, uint32_t>> index<CodeT>::center_norms()
  const noexcept
{
  if (center_norms_.has_value()) {
    return std::make_optional<raft::device_vector_view<const float, uint32_t>>(
      center_norms_->view());
  } else {
    return std::nullopt;
  }
}

template <typename CodeT>
void index<CodeT>::allocate_center_norms(raft::resources const& res)
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

template <typename CodeT>
raft::device_vector_view<float, uint32_t> index<CodeT>::sq_vmin() noexcept
{
  return sq_vmin_.view();
}

template <typename CodeT>
raft::device_vector_view<const float, uint32_t> index<CodeT>::sq_vmin() const noexcept
{
  return sq_vmin_.view();
}

template <typename CodeT>
raft::device_vector_view<float, uint32_t> index<CodeT>::sq_delta() noexcept
{
  return sq_delta_.view();
}

template <typename CodeT>
raft::device_vector_view<const float, uint32_t> index<CodeT>::sq_delta() const noexcept
{
  return sq_delta_.view();
}

template <typename CodeT>
raft::host_vector_view<int64_t, uint32_t> index<CodeT>::accum_sorted_sizes() noexcept
{
  return accum_sorted_sizes_.view();
}

template <typename CodeT>
raft::host_vector_view<const int64_t, uint32_t> index<CodeT>::accum_sorted_sizes() const noexcept
{
  return accum_sorted_sizes_.view();
}

template <typename CodeT>
raft::device_vector_view<CodeT*, uint32_t> index<CodeT>::data_ptrs() noexcept
{
  return data_ptrs_.view();
}

template <typename CodeT>
raft::device_vector_view<CodeT* const, uint32_t> index<CodeT>::data_ptrs() const noexcept
{
  return data_ptrs_.view();
}

template <typename CodeT>
raft::device_vector_view<int64_t*, uint32_t> index<CodeT>::inds_ptrs() noexcept
{
  return inds_ptrs_.view();
}

template <typename CodeT>
raft::device_vector_view<int64_t* const, uint32_t> index<CodeT>::inds_ptrs() const noexcept
{
  return inds_ptrs_.view();
}

template <typename CodeT>
std::vector<std::shared_ptr<list_data<CodeT, int64_t>>>& index<CodeT>::lists() noexcept
{
  return lists_;
}

template <typename CodeT>
const std::vector<std::shared_ptr<list_data<CodeT, int64_t>>>& index<CodeT>::lists() const noexcept
{
  return lists_;
}

template <typename CodeT>
void index<CodeT>::check_consistency()
{
  auto n_lists = lists_.size();
  RAFT_EXPECTS(list_sizes_.extent(0) == n_lists, "inconsistent list size");
  RAFT_EXPECTS(data_ptrs_.extent(0) == n_lists, "inconsistent list size");
  RAFT_EXPECTS(inds_ptrs_.extent(0) == n_lists, "inconsistent list size");
  RAFT_EXPECTS((centers_.extent(0) == list_sizes_.extent(0)) &&
                 (!center_norms_.has_value() || centers_.extent(0) == center_norms_->extent(0)),
               "inconsistent number of lists (clusters)");
  RAFT_EXPECTS(sq_vmin_.extent(0) == centers_.extent(1),
               "sq_vmin size (%u) does not match dim (%u)",
               static_cast<uint32_t>(sq_vmin_.extent(0)),
               static_cast<uint32_t>(centers_.extent(1)));
  RAFT_EXPECTS(sq_delta_.extent(0) == centers_.extent(1),
               "sq_delta size (%u) does not match dim (%u)",
               static_cast<uint32_t>(sq_delta_.extent(0)),
               static_cast<uint32_t>(centers_.extent(1)));
}

template struct CUVS_EXPORT index<uint8_t>;

}  // namespace cuvs::neighbors::ivf_sq
