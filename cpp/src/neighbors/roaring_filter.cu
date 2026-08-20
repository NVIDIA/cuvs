/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/roaring_filter_data.cuh"

#include <cuvs/core/roaring_allowlist.hpp>
#include <cuvs/neighbors/common.hpp>

#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace cuvs::neighbors::filtering {
namespace {

using data_type = cuvs::neighbors::detail::roaring_filter_data_t<std::uint32_t>;
using ref_type  = data_type::ref_type;

std::size_t validate_views(std::span<const cuvs::core::roaring_allowlist_view> allowlists)
{
  RAFT_EXPECTS(!allowlists.empty(), "roaring_filter requires at least one query allowlist.");
  RAFT_EXPECTS(allowlists.front().valid(), "roaring_filter requires valid allowlist views.");
  auto const dataset_rows = allowlists.front().dataset_rows();
  for (auto const& allowlist : allowlists) {
    RAFT_EXPECTS(allowlist.valid(), "roaring_filter requires valid allowlist views.");
    RAFT_EXPECTS(allowlist.dataset_rows() == dataset_rows,
                 "Every roaring_filter allowlist must have the same dataset_rows.");
    RAFT_EXPECTS(allowlist.empty() || allowlist.device_reference() != nullptr,
                 "A nonempty Roaring allowlist must carry a device reference.");
  }
  RAFT_EXPECTS(allowlists.size() <= std::numeric_limits<std::uint32_t>::max(),
               "roaring_filter has too many query allowlists.");
  return dataset_rows;
}

float estimate_filtering_rate(std::span<const cuvs::core::roaring_allowlist_view> allowlists,
                              std::size_t dataset_rows)
{
  auto minimum_cardinality = dataset_rows;
  for (auto const& allowlist : allowlists) {
    minimum_cardinality = std::min(minimum_cardinality, allowlist.cardinality());
  }
  auto const rejected =
    1.0 - static_cast<double>(minimum_cardinality) / static_cast<double>(dataset_rows);
  return std::clamp(static_cast<float>(rejected), 0.0f, 0.999f);
}

}  // namespace

struct roaring_filter::impl {
  std::vector<cuvs::core::roaring_allowlist_view> allowlists;
  rmm::device_uvector<ref_type const*> refs;
  rmm::device_uvector<std::uint8_t> empty_rows;
  rmm::device_uvector<data_type> payload;
  std::size_t dataset_rows_{};
  float filtering_rate_{};

  impl(raft::resources const& res,
       std::span<const cuvs::core::roaring_allowlist_view> input_allowlists)
    : allowlists(input_allowlists.begin(), input_allowlists.end()),
      refs(input_allowlists.size(), raft::resource::get_cuda_stream(res)),
      empty_rows(input_allowlists.size(), raft::resource::get_cuda_stream(res)),
      payload(1, raft::resource::get_cuda_stream(res)),
      dataset_rows_(validate_views(input_allowlists)),
      filtering_rate_(estimate_filtering_rate(input_allowlists, dataset_rows_))
  {
    auto stream = raft::resource::get_cuda_stream(res);
    std::vector<ref_type const*> host_refs;
    std::vector<std::uint8_t> host_empty;
    host_refs.reserve(allowlists.size());
    host_empty.reserve(allowlists.size());
    for (auto const& allowlist : allowlists) {
      host_refs.push_back(static_cast<ref_type const*>(allowlist.device_reference()));
      host_empty.push_back(allowlist.empty() ? 1 : 0);
    }

    RAFT_CUDA_TRY(cudaMemcpyAsync(refs.data(),
                                  host_refs.data(),
                                  host_refs.size() * sizeof(ref_type const*),
                                  cudaMemcpyHostToDevice,
                                  stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(empty_rows.data(),
                                  host_empty.data(),
                                  host_empty.size() * sizeof(std::uint8_t),
                                  cudaMemcpyHostToDevice,
                                  stream));
    auto const host_payload = data_type{refs.data(),
                                        empty_rows.data(),
                                        static_cast<std::uint32_t>(allowlists.size()),
                                        static_cast<std::uint64_t>(dataset_rows_)};
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      payload.data(), &host_payload, sizeof(host_payload), cudaMemcpyHostToDevice, stream));

    // Construction establishes a stream-independent ready object. Search only reads these stable
    // allocations and therefore needs no event, copy, initialization kernel, or synchronization.
    raft::resource::sync_stream(res);
  }

  void recompute_filtering_rate()
  {
    filtering_rate_ = estimate_filtering_rate(allowlists, dataset_rows_);
  }
};

roaring_filter::roaring_filter(raft::resources const& res,
                               std::span<const cuvs::core::roaring_allowlist_view> allowlists)
  : impl_(std::make_shared<impl>(res, allowlists))
{
}

bool roaring_filter::valid() const noexcept { return impl_ != nullptr; }

std::size_t roaring_filter::num_queries() const noexcept
{
  return valid() ? impl_->allowlists.size() : 0;
}

std::size_t roaring_filter::dataset_rows() const noexcept
{
  return valid() ? impl_->dataset_rows_ : 0;
}

std::size_t roaring_filter::cardinality(std::size_t query_id) const
{
  RAFT_EXPECTS(valid(), "roaring_filter is not initialized.");
  RAFT_EXPECTS(query_id < num_queries(), "roaring_filter query_id is out of range.");
  return impl_->allowlists[query_id].cardinality();
}

bool roaring_filter::empty(std::size_t query_id) const { return cardinality(query_id) == 0; }

float roaring_filter::filtering_rate() const noexcept
{
  return valid() ? impl_->filtering_rate_ : 0.0f;
}

std::size_t roaring_filter::size_bytes() const noexcept
{
  if (!valid()) { return 0; }
  return impl_->refs.size() * sizeof(ref_type const*) +
         impl_->empty_rows.size() * sizeof(std::uint8_t) +
         impl_->payload.size() * sizeof(data_type);
}

void roaring_filter::set_allowlist(raft::resources const& res,
                                   std::size_t query_id,
                                   cuvs::core::roaring_allowlist_view replacement)
{
  RAFT_EXPECTS(valid(), "roaring_filter is not initialized.");
  RAFT_EXPECTS(query_id < num_queries(), "roaring_filter query_id is out of range.");
  RAFT_EXPECTS(replacement.valid(), "roaring_filter requires a valid replacement allowlist view.");
  RAFT_EXPECTS(replacement.dataset_rows() == dataset_rows(),
               "Replacement Roaring allowlist must have the filter's dataset_rows.");
  RAFT_EXPECTS(replacement.empty() || replacement.device_reference() != nullptr,
               "A nonempty replacement Roaring allowlist must carry a device reference.");

  auto stream       = raft::resource::get_cuda_stream(res);
  auto const ref    = static_cast<ref_type const*>(replacement.device_reference());
  auto const empty_ = static_cast<std::uint8_t>(replacement.empty() ? 1 : 0);
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    impl_->refs.data() + query_id, &ref, sizeof(ref), cudaMemcpyHostToDevice, stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    impl_->empty_rows.data() + query_id, &empty_, sizeof(empty_), cudaMemcpyHostToDevice, stream));
  raft::resource::sync_stream(res);

  impl_->allowlists[query_id] = replacement;
  impl_->recompute_filtering_rate();
}

void* roaring_filter::device_payload() const noexcept
{
  return valid() ? const_cast<data_type*>(impl_->payload.data()) : nullptr;
}

}  // namespace cuvs::neighbors::filtering
