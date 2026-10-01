/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace CUVS_EXPORT cuvs {
namespace core {

/**
 * @brief cuVS-owned Bloom filter wrapper with opaque implementation.
 *
 * This class intentionally hides cuCollections types from the cuVS public API.
 * The wrapper supports the expected bulk host APIs used by ANN workflows.
 */
class CUVS_EXPORT bloom_filter {
 private:
  struct impl;

 public:
  using key_type = std::uint32_t;

  /**
   * @brief Construct a Bloom filter with user-facing quality knobs.
   *
   * @p dataset_rows is the number of rows in the indexed dataset. The filter uses it with
   * @p filtering_rate to estimate the number of inserted valid ids and compute a target filter
   * size that satisfies the requested false-positive rate.
   *
   * The primary tuning knobs are:
   * - @p filtering_rate: expected fraction of dataset rows that will be inserted as valid ids.
   * - @p target_false_positive_rate: desired Bloom filter false-positive probability.
   *
   * Sizing math used internally:
   * - `expected_insertions = ceil(dataset_rows * filtering_rate)`
   * - The default policy uses 256-bit blocks split into eight 32-bit words and sets one bit in each
   *   word per inserted key.
   * - For each candidate block count, the expected false-positive rate accounts for the binomial
   *   distribution of inserted keys across blocks and the fixed eight-bit fingerprint.
   * - The smallest block count whose expected false-positive rate meets
   *   @p target_false_positive_rate is selected.
   *
   * Practical knob behavior:
   * - Lower @p target_false_positive_rate -> larger filter, fewer false positives, typically higher
   *   filtered-search recall.
   * - Higher @p filtering_rate -> larger filter for the same target false-positive rate.
   */
  bloom_filter(raft::resources const& res,
               std::size_t dataset_rows,
               float filtering_rate             = 1.0f,
               float target_false_positive_rate = 0.01f);
  ~bloom_filter();

  bloom_filter(bloom_filter const&)            = delete;
  bloom_filter& operator=(bloom_filter const&) = delete;
  bloom_filter(bloom_filter&&) noexcept;
  bloom_filter& operator=(bloom_filter&&) noexcept;

  void clear(raft::resources const& res);
  void clear_async(raft::resources const& res);

  void add(raft::resources const& res, raft::device_vector_view<const key_type, int64_t> keys);
  void add_async(raft::resources const& res,
                 raft::device_vector_view<const key_type, int64_t> keys);

  void contains(raft::resources const& res,
                raft::device_vector_view<const key_type, int64_t> keys,
                raft::device_vector_view<std::uint8_t, int64_t> output) const;
  void contains_async(raft::resources const& res,
                      raft::device_vector_view<const key_type, int64_t> keys,
                      raft::device_vector_view<std::uint8_t, int64_t> output) const;

  [[nodiscard]] std::size_t num_blocks() const noexcept;

  /**
   * @brief Return the estimated fraction of dataset rows rejected by this filter.
   *
   * The estimate is derived at construction from the configured valid-row fraction and the
   * expected false-positive rate of the selected filter geometry. It performs no device work.
   */
  [[nodiscard]] float estimate_filtering_rate() const noexcept;

 private:
  friend impl const& get_bloom_filter_impl(bloom_filter const& filter) noexcept;

  std::unique_ptr<impl> impl_;
};

}  // namespace core
}  // namespace CUVS_EXPORT cuvs
