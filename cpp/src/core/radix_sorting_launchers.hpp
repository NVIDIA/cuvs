/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstddef>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <cuda_runtime_api.h>
#include <cuvs/core/export.hpp>
#include <iterator>

namespace cuvs::detail {
template <typename KeyIn, typename KeyOut, typename ValueIn, typename ValueOut, typename NumItems>
cudaError_t cub_sort_pairs(void* storage,
                           size_t& bytes,
                           KeyIn keys_in,
                           KeyOut keys_out,
                           ValueIn values_in,
                           ValueOut values_out,
                           NumItems count,
                           int begin_bit,
                           int end_bit,
                           cudaStream_t stream);
template <typename InputIt, typename OutputIt, typename NumItems>
cudaError_t cub_sort_keys_descending(void* storage,
                                     size_t& bytes,
                                     InputIt input,
                                     OutputIt output,
                                     NumItems count,
                                     int begin_bit,
                                     int end_bit,
                                     cudaStream_t stream);
}  // namespace cuvs::detail

namespace cuvs::detail {
template <typename Policy, typename Iterator>
CUVS_EXPORT void thrust_sort(Policy const& policy, Iterator first, Iterator last);

template <typename Policy, typename Iterator, typename Compare>
CUVS_EXPORT void thrust_sort(Policy const& policy, Iterator first, Iterator last, Compare compare);

template <typename Policy, typename Key, typename Value>
CUVS_EXPORT void thrust_sort_by_key(Policy const& policy, Key first, Key last, Value values);

template <typename Policy, typename Key, typename Value, typename Compare>
CUVS_EXPORT void thrust_sort_by_key(
  Policy const& policy, Key first, Key last, Value values, Compare compare);
}  // namespace cuvs::detail

namespace cuvs::detail {
template <typename Key, typename Value>
CUVS_EXPORT void thrust_sort_by_key(Key first, Key last, Value values);
}

namespace cuvs::sparse::neighbors::detail {
/**
 * Assumes 3-iterator tuple containing COO rows, cols, and
 * a cub keyvalue pair object. Sorts the 3 arrays in
 * ascending order: row->col->keyvaluepair
 */
struct TupleComp {
  template <typename one, typename two>
  __host__ __device__ bool operator()(const one& t1, const two& t2)
  {
    // sort first by each sample's color,
    if (cuda::std::get<0>(t1) < cuda::std::get<0>(t2)) return true;
    if (cuda::std::get<0>(t1) > cuda::std::get<0>(t2)) return false;

    // then by the color of each sample's closest neighbor,
    if (cuda::std::get<1>(t1) < cuda::std::get<1>(t2)) return true;
    if (cuda::std::get<1>(t1) > cuda::std::get<1>(t2)) return false;

    // then sort by value in descending order
    return cuda::std::get<2>(t1).value < cuda::std::get<2>(t2).value;
  }
};

}  // namespace cuvs::sparse::neighbors::detail
