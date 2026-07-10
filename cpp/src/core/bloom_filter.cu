/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../neighbors/detail/sample_filter_data.cuh"
#include <cuvs/core/bloom_filter.hpp>

#include <cuco/bloom_filter.cuh>

#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>

#include <rmm/device_uvector.hpp>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <cmath>
#include <new>
#include <optional>
#include <utility>

namespace cuvs::core {

std::size_t compute_num_blocks_from_rates(std::size_t dataset_rows,
                                          float filtering_rate,
                                          float target_false_positive_rate)
{
  RAFT_EXPECTS(dataset_rows > 0,
               "dataset_rows must be greater than zero when deriving bloom size.");
  RAFT_EXPECTS(filtering_rate > 0.0f && filtering_rate <= 1.0f,
               "filtering_rate must be in (0, 1].");
  RAFT_EXPECTS(target_false_positive_rate > 0.0f && target_false_positive_rate < 1.0f,
               "target_false_positive_rate must be in (0, 1).");

  // Bloom sizing: m = -n * ln(p) / (ln(2)^2), then blocks = ceil(m / 256 bits-per-block).
  constexpr double kBitsPerBlock = 256.0;
  constexpr double kLn2          = 0.6931471805599453;
  constexpr double kLn2Sq        = kLn2 * kLn2;

  auto expected_insertions = std::max<std::size_t>(
    1, static_cast<std::size_t>(std::ceil(static_cast<double>(dataset_rows) * filtering_rate)));
  auto required_bits = -static_cast<double>(expected_insertions) *
                       std::log(static_cast<double>(target_false_positive_rate)) / kLn2Sq;
  return std::max<std::size_t>(1,
                               static_cast<std::size_t>(std::ceil(required_bits / kBitsPerBlock)));
}

struct bloom_filter::impl {
  using key_type              = bloom_filter::key_type;
  using cuco_filter_type      = cuco::bloom_filter<key_type>;
  using sample_filter_payload = cuvs::neighbors::detail::bloom_filter_data_t<key_type>;

  cuco_filter_type filter;
  std::optional<std::size_t> configured_dataset_rows;
  float filtering_rate;
  float target_false_positive_rate;

  impl(raft::resources const& res,
       std::size_t num_blocks,
       float filtering_rate_,
       float target_false_positive_rate_)
    : filter(num_blocks, {}, {}, {}, raft::resource::get_cuda_stream(res)),
      filtering_rate(filtering_rate_),
      target_false_positive_rate(target_false_positive_rate_)
  {
  }

  void rebuild_filter(std::size_t num_blocks, cudaStream_t stream)
  {
    filter = cuco_filter_type(num_blocks, {}, {}, {}, stream);
  }

  void configure_or_validate_dataset_rows(raft::device_vector_view<const key_type, int64_t> keys,
                                          cudaStream_t stream)
  {
    if (keys.extent(0) == 0) { return; }

    auto inferred_dataset_rows = static_cast<std::size_t>(keys.extent(0));
    if (!configured_dataset_rows.has_value()) {
      configured_dataset_rows  = inferred_dataset_rows;
      auto required_num_blocks = compute_num_blocks_from_rates(
        inferred_dataset_rows, filtering_rate, target_false_positive_rate);
      auto target_num_blocks = std::max<std::size_t>(filter.block_extent(), required_num_blocks);
      if (target_num_blocks != filter.block_extent()) { rebuild_filter(target_num_blocks, stream); }
      return;
    }

    RAFT_EXPECTS(
      inferred_dataset_rows == *configured_dataset_rows,
      "keys.size() must match dataset_rows established by the first add/add_async call.");
  }
};

bloom_filter::bloom_filter(raft::resources const& res,
                           float filtering_rate,
                           float target_false_positive_rate,
                           std::size_t num_blocks)
  : impl_(std::make_unique<impl>(
      res, std::max<std::size_t>(1, num_blocks), filtering_rate, target_false_positive_rate))
{
  RAFT_EXPECTS(filtering_rate > 0.0f && filtering_rate <= 1.0f,
               "filtering_rate must be in (0, 1].");
  RAFT_EXPECTS(target_false_positive_rate > 0.0f && target_false_positive_rate < 1.0f,
               "target_false_positive_rate must be in (0, 1).");
}

bloom_filter::~bloom_filter()                                  = default;
bloom_filter::bloom_filter(bloom_filter&&) noexcept            = default;
bloom_filter& bloom_filter::operator=(bloom_filter&&) noexcept = default;

void bloom_filter::clear(raft::resources const& res)
{
  impl_->filter.clear(raft::resource::get_cuda_stream(res));
}

void bloom_filter::clear_async(raft::resources const& res)
{
  impl_->filter.clear_async(raft::resource::get_cuda_stream(res));
}

void bloom_filter::add(raft::resources const& res,
                       raft::device_vector_view<const key_type, int64_t> keys)
{
  auto stream = raft::resource::get_cuda_stream(res);
  impl_->configure_or_validate_dataset_rows(keys, stream);
  impl_->filter.add(keys.data_handle(), keys.data_handle() + keys.extent(0), stream);
}

void bloom_filter::add_async(raft::resources const& res,
                             raft::device_vector_view<const key_type, int64_t> keys)
{
  auto stream = raft::resource::get_cuda_stream(res);
  impl_->configure_or_validate_dataset_rows(keys, stream);
  impl_->filter.add_async(keys.data_handle(), keys.data_handle() + keys.extent(0), stream);
}

void bloom_filter::contains(raft::resources const& res,
                            raft::device_vector_view<const key_type, int64_t> keys,
                            raft::device_vector_view<std::uint8_t, int64_t> output) const
{
  impl_->filter.contains(keys.data_handle(),
                         keys.data_handle() + keys.extent(0),
                         output.data_handle(),
                         raft::resource::get_cuda_stream(res));
}

void bloom_filter::contains_async(raft::resources const& res,
                                  raft::device_vector_view<const key_type, int64_t> keys,
                                  raft::device_vector_view<std::uint8_t, int64_t> output) const
{
  impl_->filter.contains_async(keys.data_handle(),
                               keys.data_handle() + keys.extent(0),
                               output.data_handle(),
                               raft::resource::get_cuda_stream(res));
}

std::size_t bloom_filter::num_blocks() const noexcept { return impl_->filter.block_extent(); }

float bloom_filter::estimate_filtering_rate(raft::resources const& res,
                                            std::size_t dataset_rows) const
{
  if (dataset_rows == 0) { return 0.0f; }
  auto stream = raft::resource::get_cuda_stream(res);
  auto policy = raft::resource::get_thrust_policy(res);

  rmm::device_uvector<std::uint8_t> hits(dataset_rows, stream);

  auto first_id = thrust::counting_iterator<key_type>(0);
  impl_->filter.contains_async(first_id, first_id + dataset_rows, hits.data(), stream);

  auto positives = thrust::count_if(
    policy, hits.begin(), hits.end(), [] __device__(std::uint8_t v) { return v != 0; });
  raft::resource::sync_stream(res);
  auto filtering_rate = static_cast<float>(dataset_rows - static_cast<std::size_t>(positives)) /
                        static_cast<float>(dataset_rows);
  return std::clamp(filtering_rate, 0.0f, 0.999f);
}

void bloom_filter::export_payload(void* payload_out, std::size_t payload_bytes) const
{
  using sample_filter_payload = impl::sample_filter_payload;
  RAFT_EXPECTS(payload_out != nullptr, "payload_out must not be null.");
  RAFT_EXPECTS(payload_bytes == sizeof(sample_filter_payload),
               "payload_bytes must match bloom filter payload size.");

  ::new (payload_out) sample_filter_payload{impl_->filter.ref()};
}

}  // namespace cuvs::core
