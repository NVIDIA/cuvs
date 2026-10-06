/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace cuvs::bench {

namespace detail {

class splitmix64 {
 public:
  explicit splitmix64(std::uint64_t seed) : state_(seed) {}

  auto operator()() -> std::uint64_t
  {
    auto value = (state_ += 0x9e3779b97f4a7c15ULL);
    value      = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value      = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  }

 private:
  std::uint64_t state_;
};

inline auto bounded_random(splitmix64& generator, std::uint64_t bound) -> std::uint64_t
{
  if (bound == 0) { throw std::invalid_argument("Random bound must be positive"); }
  const auto threshold = -bound % bound;
  while (true) {
    const auto value = generator();
    if (value >= threshold) { return value % bound; }
  }
}

}  // namespace detail

/** Select a deterministic, uniformly distributed sample without replacement. */
inline auto uniform_sample_ids(std::size_t population_size,
                               std::size_t sample_size,
                               std::uint64_t seed) -> std::vector<std::size_t>
{
  if (sample_size > population_size) {
    throw std::invalid_argument("Training sample cannot exceed the dataset size");
  }

  std::vector<std::size_t> ids(sample_size);
  std::iota(ids.begin(), ids.end(), std::size_t{0});
  if (sample_size == population_size) { return ids; }

  // Algorithm R reservoir sampling gives every sample_size-subset equal probability.
  detail::splitmix64 generator(seed);
  for (std::size_t row = sample_size; row < population_size; ++row) {
    const auto candidate = static_cast<std::size_t>(
      detail::bounded_random(generator, static_cast<std::uint64_t>(row) + 1));
    if (candidate < sample_size) { ids[candidate] = row; }
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

template <typename T>
auto make_training_sample(const T* dataset,
                          std::size_t n_rows,
                          std::size_t dimension,
                          std::size_t sample_size,
                          std::uint64_t seed) -> std::vector<T>
{
  if (dataset == nullptr) { throw std::invalid_argument("Training dataset must not be null"); }
  if (dimension == 0) { throw std::invalid_argument("Training dimension must be positive"); }
  if (sample_size > std::numeric_limits<std::size_t>::max() / dimension) {
    throw std::overflow_error("Training sample size overflow");
  }

  const auto ids = uniform_sample_ids(n_rows, sample_size, seed);
  std::vector<T> sample(sample_size * dimension);
  for (std::size_t output_row = 0; output_row < ids.size(); ++output_row) {
    const auto input_offset  = ids[output_row] * dimension;
    const auto output_offset = output_row * dimension;
    std::copy_n(dataset + input_offset, dimension, sample.data() + output_offset);
  }
  return sample;
}

inline auto training_sample_size(std::size_t n_rows,
                                 std::size_t n_lists,
                                 std::size_t max_rows_per_list) -> std::size_t
{
  if (n_lists == 0 || max_rows_per_list == 0) {
    throw std::invalid_argument("Training sample parameters must be positive");
  }
  if (n_lists > std::numeric_limits<std::size_t>::max() / max_rows_per_list) { return n_rows; }
  return std::min(n_rows, n_lists * max_rows_per_list);
}

}  // namespace cuvs::bench
