/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../neighbors/detail/sample_filter_data.cuh"
#include <cuvs/core/bloom_filter.hpp>

#include <cuco/bloom_filter.cuh>

#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace cuvs::core {

namespace {

using default_filter_policy = cuco::default_filter_policy<bloom_filter::key_type>;

constexpr auto kPatternBits   = default_filter_policy::pattern_bits;
constexpr auto kWordsPerBlock = default_filter_policy::words_per_block;
constexpr auto kBitsPerWord =
  std::numeric_limits<typename default_filter_policy::word_type>::digits;

static_assert(kPatternBits == kWordsPerBlock,
              "Bloom FPR sizing assumes the default policy sets one bit in each word.");

/**
 * Expected FPR for the default sectorized policy.
 *
 * A queried key selects one of `num_blocks` blocks. The number of inserted keys in that block is
 * binomially distributed. Given `x` keys in the block, each of the policy's eight queried bits is
 * set with probability `1 - (31 / 32)^x`, so the conditional FPR is that value to the eighth power.
 */
long double expected_false_positive_rate(std::size_t num_insertions, std::size_t num_blocks)
{
  if (num_insertions == 0) { return 0.0L; }

  auto const block_probability = 1.0L / static_cast<long double>(num_blocks);
  auto const bit_miss_probability =
    static_cast<long double>(kBitsPerWord - 1) / static_cast<long double>(kBitsPerWord);

  // Inclusion-exclusion evaluates the expectation over the binomial block occupancy directly:
  // E[(1 - q^X)^k] = sum_j (-1)^j C(k,j) (1 - (1 - q^j) / B)^n.
  long double result               = 0.0L;
  long double binomial_coefficient = 1.0L;
  for (std::uint32_t j = 0; j <= kPatternBits; ++j) {
    if (j > 0) {
      binomial_coefficient *=
        static_cast<long double>(kPatternBits - j + 1) / static_cast<long double>(j);
    }
    auto const avoids_selected_bits = std::pow(bit_miss_probability, j);
    auto const insertion_misses     = 1.0L - block_probability * (1.0L - avoids_selected_bits);
    auto const term = binomial_coefficient * std::pow(insertion_misses, num_insertions);
    result += (j % 2 == 0) ? term : -term;
  }

  return std::clamp(result, 0.0L, 1.0L);
}

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

  auto expected_insertions = std::max<std::size_t>(
    1, static_cast<std::size_t>(std::ceil(static_cast<double>(dataset_rows) * filtering_rate)));
  auto const target = static_cast<long double>(target_false_positive_rate);

  std::size_t upper = 1;
  while (expected_false_positive_rate(expected_insertions, upper) > target) {
    if (upper > default_filter_policy::max_filter_blocks / 2) {
      upper = default_filter_policy::max_filter_blocks;
      RAFT_EXPECTS(expected_false_positive_rate(expected_insertions, upper) <= target,
                   "Requested Bloom filter false-positive rate requires too many blocks.");
      break;
    }
    upper *= 2;
  }

  std::size_t lower = 1;
  while (lower < upper) {
    auto const midpoint = lower + (upper - lower) / 2;
    if (expected_false_positive_rate(expected_insertions, midpoint) <= target) {
      upper = midpoint;
    } else {
      lower = midpoint + 1;
    }
  }
  return lower;
}

}  // namespace

struct bloom_filter::impl {
  using key_type              = bloom_filter::key_type;
  using cuco_filter_type      = cuco::bloom_filter<key_type>;
  using sample_filter_payload = cuvs::neighbors::detail::bloom_filter_data_t<key_type>;

  cuco_filter_type filter;
  std::size_t dataset_rows;
  float estimated_filtering_rate;

  impl(raft::resources const& res,
       std::size_t num_blocks,
       std::size_t dataset_rows_,
       float filtering_rate_)
    : filter(num_blocks, {}, {}, {}, raft::resource::get_cuda_stream(res)),
      dataset_rows(dataset_rows_),
      estimated_filtering_rate(0.0f)
  {
    auto expected_insertions = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::ceil(static_cast<double>(dataset_rows) * filtering_rate_)));
    auto expected_fpr = expected_false_positive_rate(expected_insertions, num_blocks);
    auto rejected_fraction =
      (1.0L - static_cast<long double>(filtering_rate_)) * (1.0L - expected_fpr);
    estimated_filtering_rate = static_cast<float>(std::clamp(rejected_fraction, 0.0L, 0.999L));
  }

  void validate_keys(raft::device_vector_view<const key_type, int64_t> keys) const
  {
    if (keys.extent(0) == 0) { return; }
    auto num_insertions = static_cast<std::size_t>(keys.extent(0));
    RAFT_EXPECTS(num_insertions <= dataset_rows,
                 "Number of keys to add must not exceed dataset_rows.");
  }
};

bloom_filter::bloom_filter(raft::resources const& res,
                           std::size_t dataset_rows,
                           float filtering_rate,
                           float target_false_positive_rate)
  : impl_(std::make_unique<impl>(
      res,
      compute_num_blocks_from_rates(dataset_rows, filtering_rate, target_false_positive_rate),
      dataset_rows,
      filtering_rate))
{
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
  impl_->validate_keys(keys);
  impl_->filter.add(keys.data_handle(), keys.data_handle() + keys.extent(0), stream);
}

void bloom_filter::add_async(raft::resources const& res,
                             raft::device_vector_view<const key_type, int64_t> keys)
{
  auto stream = raft::resource::get_cuda_stream(res);
  impl_->validate_keys(keys);
  impl_->filter.add_async(keys.data_handle(), keys.data_handle() + keys.extent(0), stream);
}

void bloom_filter::contains(raft::resources const& res,
                            raft::device_vector_view<const key_type, int64_t> keys,
                            raft::device_vector_view<std::uint8_t, int64_t> output) const
{
  RAFT_EXPECTS(output.extent(0) >= keys.extent(0),
               "Bloom filter contains output size must be at least keys.size().");
  impl_->filter.contains(keys.data_handle(),
                         keys.data_handle() + keys.extent(0),
                         output.data_handle(),
                         raft::resource::get_cuda_stream(res));
}

void bloom_filter::contains_async(raft::resources const& res,
                                  raft::device_vector_view<const key_type, int64_t> keys,
                                  raft::device_vector_view<std::uint8_t, int64_t> output) const
{
  RAFT_EXPECTS(output.extent(0) >= keys.extent(0),
               "Bloom filter contains output size must be at least keys.size().");
  impl_->filter.contains_async(keys.data_handle(),
                               keys.data_handle() + keys.extent(0),
                               output.data_handle(),
                               raft::resource::get_cuda_stream(res));
}

std::size_t bloom_filter::num_blocks() const noexcept { return impl_->filter.block_extent(); }

float bloom_filter::estimate_filtering_rate() const noexcept
{
  return impl_->estimated_filtering_rate;
}

auto get_bloom_filter_impl(bloom_filter const& filter) noexcept -> bloom_filter::impl const&
{
  return *filter.impl_;
}

}  // namespace cuvs::core

namespace cuvs::neighbors::detail {

bloom_filter_data_t<std::uint32_t> bloom_filter_factory::make(
  cuvs::core::bloom_filter const& filter)
{
  return bloom_filter_data_t<std::uint32_t>{get_bloom_filter_impl(filter).filter.ref()};
}

}  // namespace cuvs::neighbors::detail
