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
 public:
  using key_type = std::uint32_t;

  /**
   * @brief Construct a Bloom filter with user-facing quality knobs.
   *
   * The first add/add_async call sets `dataset_rows` from `keys.size()` and then computes a target
   * filter size from the first two parameters. The filter keeps at least `num_blocks` blocks, and
   * may grow above that floor to satisfy the requested false-positive rate.
   *
   * The primary tuning knobs are:
   * - @p expected_valid_rate: expected fraction of dataset rows that will be inserted as valid ids.
   * - @p target_false_positive_rate: desired Bloom filter false-positive probability.
   *
   * Sizing math used internally:
   * - `expected_insertions = ceil(dataset_rows * expected_valid_rate)`
   * - `required_bits = -expected_insertions * ln(target_false_positive_rate) / (ln(2)^2)`
   * - `required_blocks = ceil(required_bits / 256)` (default cuco policy uses 256-bit blocks)
   * - `final_blocks = max(num_blocks, required_blocks)`
   *
   * Practical knob behavior:
   * - Lower @p target_false_positive_rate -> larger filter, fewer false positives, typically higher
   *   filtered-search recall.
   * - Higher @p expected_valid_rate -> larger filter for the same target false-positive rate.
   * - @p num_blocks is an expert floor; keep default unless you need a hard minimum memory budget.
   */
  bloom_filter(raft::resources const& res,
               float expected_valid_rate        = 1.0f,
               float target_false_positive_rate = 0.01f,
               std::size_t num_blocks           = 256);
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
   * @brief Device pointer to the CAGRA JIT sample-filter payload.
   *
   * The pointed object is device memory owned by this wrapper and remains valid while this object
   * is alive.
   */
  [[nodiscard]] void* filter_data() const noexcept;

 private:
  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace core
}  // namespace CUVS_EXPORT cuvs
