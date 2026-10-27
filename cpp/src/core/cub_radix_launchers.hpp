/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <core/radix_sorting_launchers.hpp>

#include <raft/core/kvp.hpp>

#include <thrust/iterator/zip_iterator.h>

#include <cub/device/device_merge_sort.cuh>

#include <cub/device/device_radix_sort.cuh>

namespace cuvs::detail::radix_launch {

using f32_keys_u64_ascending = cub::detail::radix_sort::DeviceRadixSortKernelSource<
  cub::detail::radix_sort::policy_selector_from_types<float, cub::NullType, unsigned long long>,
  cub::SortOrder::Ascending,
  float,
  cub::NullType,
  unsigned long long,
  cub::detail::identity_decomposer_t>;

using f32_keys_u64_descending = cub::detail::radix_sort::DeviceRadixSortKernelSource<
  cub::detail::radix_sort::policy_selector_from_types<float, cub::NullType, unsigned long long>,
  cub::SortOrder::Descending,
  float,
  cub::NullType,
  unsigned long long,
  cub::detail::identity_decomposer_t>;

using i32_keys_u64_ascending = cub::detail::radix_sort::DeviceRadixSortKernelSource<
  cub::detail::radix_sort::policy_selector_from_types<int, cub::NullType, unsigned long long>,
  cub::SortOrder::Ascending,
  int,
  cub::NullType,
  unsigned long long,
  cub::detail::identity_decomposer_t>;

using i32_keys_u64_descending = cub::detail::radix_sort::DeviceRadixSortKernelSource<
  cub::detail::radix_sort::policy_selector_from_types<int, cub::NullType, unsigned long long>,
  cub::SortOrder::Descending,
  int,
  cub::NullType,
  unsigned long long,
  cub::detail::identity_decomposer_t>;

using f32_keys_u64_ascending_RadixSortExclusiveSumKernel_pointer = void (*)(unsigned long long*);

using f32_keys_u64_descending_RadixSortExclusiveSumKernel_pointer = void (*)(unsigned long long*);

using i32_keys_u64_ascending_RadixSortExclusiveSumKernel_pointer = void (*)(unsigned long long*);

using i32_keys_u64_descending_RadixSortExclusiveSumKernel_pointer = void (*)(unsigned long long*);

using f32_keys_u64_ascending_RadixSortHistogramKernel_pointer =
  void (*)(unsigned long long*,
           float const*,
           unsigned long long,
           int,
           int,
           cub::detail::identity_decomposer_t);

using f32_keys_u64_ascending_RadixSortInitLookbackKernel_pointer = void (*)(int*,
                                                                            unsigned long,
                                                                            int*,
                                                                            unsigned long);

using f32_keys_u64_descending_RadixSortInitLookbackKernel_pointer = void (*)(int*,
                                                                             unsigned long,
                                                                             int*,
                                                                             unsigned long);

using f32_keys_u64_ascending_RadixSortInitBinsAndCountersKernel_pointer =
  void (*)(int*, unsigned long, unsigned long long*, unsigned long);

using f32_keys_u64_descending_RadixSortInitBinsAndCountersKernel_pointer =
  void (*)(int*, unsigned long, unsigned long long*, unsigned long);

using i32_keys_u64_ascending_RadixSortInitLookbackKernel_pointer = void (*)(int*,
                                                                            unsigned long,
                                                                            int*,
                                                                            unsigned long);

using i32_keys_u64_descending_RadixSortInitLookbackKernel_pointer = void (*)(int*,
                                                                             unsigned long,
                                                                             int*,
                                                                             unsigned long);

using i32_keys_u64_ascending_RadixSortInitBinsAndCountersKernel_pointer =
  void (*)(int*, unsigned long, unsigned long long*, unsigned long);

using i32_keys_u64_descending_RadixSortInitBinsAndCountersKernel_pointer =
  void (*)(int*, unsigned long, unsigned long long*, unsigned long);

using f32_keys_u64_ascending_RadixSortOnesweepKernel_pointer =
  void (*)(int*,
           int*,
           unsigned long long*,
           unsigned long long const*,
           float*,
           float const*,
           cub::NullType*,
           cub::NullType const*,
           int,
           int,
           int,
           cub::detail::identity_decomposer_t);

using f32_keys_u64_ascending_RadixSortSingleTileKernel_pointer =
  void (*)(float const*,
           float*,
           cub::NullType const*,
           cub::NullType*,
           unsigned long long,
           int,
           int,
           cub::detail::identity_decomposer_t);

}  // namespace cuvs::detail::radix_launch

template <>
__host__ __device__
  cuvs::detail::radix_launch::f32_keys_u64_ascending_RadixSortExclusiveSumKernel_pointer
  cuvs::detail::radix_launch::f32_keys_u64_ascending::RadixSortExclusiveSumKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::f32_keys_u64_descending_RadixSortExclusiveSumKernel_pointer
  cuvs::detail::radix_launch::f32_keys_u64_descending::RadixSortExclusiveSumKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::i32_keys_u64_ascending_RadixSortExclusiveSumKernel_pointer
  cuvs::detail::radix_launch::i32_keys_u64_ascending::RadixSortExclusiveSumKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::i32_keys_u64_descending_RadixSortExclusiveSumKernel_pointer
  cuvs::detail::radix_launch::i32_keys_u64_descending::RadixSortExclusiveSumKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::f32_keys_u64_ascending_RadixSortHistogramKernel_pointer
  cuvs::detail::radix_launch::f32_keys_u64_ascending::RadixSortHistogramKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::f32_keys_u64_ascending_RadixSortInitLookbackKernel_pointer
  cuvs::detail::radix_launch::f32_keys_u64_ascending::RadixSortInitLookbackKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::f32_keys_u64_descending_RadixSortInitLookbackKernel_pointer
  cuvs::detail::radix_launch::f32_keys_u64_descending::RadixSortInitLookbackKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::f32_keys_u64_ascending_RadixSortInitBinsAndCountersKernel_pointer
  cuvs::detail::radix_launch::f32_keys_u64_ascending::RadixSortInitBinsAndCountersKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::f32_keys_u64_descending_RadixSortInitBinsAndCountersKernel_pointer
  cuvs::detail::radix_launch::f32_keys_u64_descending::RadixSortInitBinsAndCountersKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::i32_keys_u64_ascending_RadixSortInitLookbackKernel_pointer
  cuvs::detail::radix_launch::i32_keys_u64_ascending::RadixSortInitLookbackKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::i32_keys_u64_descending_RadixSortInitLookbackKernel_pointer
  cuvs::detail::radix_launch::i32_keys_u64_descending::RadixSortInitLookbackKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::i32_keys_u64_ascending_RadixSortInitBinsAndCountersKernel_pointer
  cuvs::detail::radix_launch::i32_keys_u64_ascending::RadixSortInitBinsAndCountersKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::i32_keys_u64_descending_RadixSortInitBinsAndCountersKernel_pointer
  cuvs::detail::radix_launch::i32_keys_u64_descending::RadixSortInitBinsAndCountersKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::f32_keys_u64_ascending_RadixSortOnesweepKernel_pointer
  cuvs::detail::radix_launch::f32_keys_u64_ascending::RadixSortOnesweepKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::f32_keys_u64_ascending_RadixSortSingleTileKernel_pointer
  cuvs::detail::radix_launch::f32_keys_u64_ascending::RadixSortSingleTileKernel();

namespace cuvs::detail::radix_launch {
using u32_keys_u64_ascending = cub::detail::radix_sort::DeviceRadixSortKernelSource<
  cub::detail::radix_sort::
    policy_selector_from_types<unsigned int, cub::NullType, unsigned long long>,
  cub::SortOrder::Ascending,
  unsigned int,
  cub::NullType,
  unsigned long long,
  cub::detail::identity_decomposer_t>;
using u32_keys_u64_descending = cub::detail::radix_sort::DeviceRadixSortKernelSource<
  cub::detail::radix_sort::
    policy_selector_from_types<unsigned int, cub::NullType, unsigned long long>,
  cub::SortOrder::Descending,
  unsigned int,
  cub::NullType,
  unsigned long long,
  cub::detail::identity_decomposer_t>;
using u32_keys_u64_ascending_RadixSortExclusiveSumKernel_pointer  = void (*)(unsigned long long*);
using u32_keys_u64_descending_RadixSortExclusiveSumKernel_pointer = void (*)(unsigned long long*);
using u32_keys_u64_ascending_RadixSortHistogramKernel_pointer =
  void (*)(unsigned long long*,
           unsigned int const*,
           unsigned long long,
           int,
           int,
           cub::detail::identity_decomposer_t);
using u32_keys_u64_ascending_RadixSortInitLookbackKernel_pointer  = void (*)(int*,
                                                                            unsigned long,
                                                                            int*,
                                                                            unsigned long);
using u32_keys_u64_descending_RadixSortInitLookbackKernel_pointer = void (*)(int*,
                                                                             unsigned long,
                                                                             int*,
                                                                             unsigned long);
using u32_keys_u64_ascending_RadixSortInitBinsAndCountersKernel_pointer =
  void (*)(int*, unsigned long, unsigned long long*, unsigned long);
using u32_keys_u64_descending_RadixSortInitBinsAndCountersKernel_pointer =
  void (*)(int*, unsigned long, unsigned long long*, unsigned long);
using u32_keys_u64_ascending_RadixSortOnesweepKernel_pointer =
  void (*)(int*,
           int*,
           unsigned long long*,
           unsigned long long const*,
           unsigned int*,
           unsigned int const*,
           cub::NullType*,
           cub::NullType const*,
           int,
           int,
           int,
           cub::detail::identity_decomposer_t);
using u32_keys_u64_ascending_RadixSortSingleTileKernel_pointer =
  void (*)(unsigned int const*,
           unsigned int*,
           cub::NullType const*,
           cub::NullType*,
           unsigned long long,
           int,
           int,
           cub::detail::identity_decomposer_t);
}  // namespace cuvs::detail::radix_launch

template <>
__host__ __device__
  cuvs::detail::radix_launch::u32_keys_u64_ascending_RadixSortExclusiveSumKernel_pointer
  cuvs::detail::radix_launch::u32_keys_u64_ascending::RadixSortExclusiveSumKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::u32_keys_u64_descending_RadixSortExclusiveSumKernel_pointer
  cuvs::detail::radix_launch::u32_keys_u64_descending::RadixSortExclusiveSumKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::u32_keys_u64_ascending_RadixSortHistogramKernel_pointer
  cuvs::detail::radix_launch::u32_keys_u64_ascending::RadixSortHistogramKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::u32_keys_u64_ascending_RadixSortInitLookbackKernel_pointer
  cuvs::detail::radix_launch::u32_keys_u64_ascending::RadixSortInitLookbackKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::u32_keys_u64_descending_RadixSortInitLookbackKernel_pointer
  cuvs::detail::radix_launch::u32_keys_u64_descending::RadixSortInitLookbackKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::u32_keys_u64_ascending_RadixSortInitBinsAndCountersKernel_pointer
  cuvs::detail::radix_launch::u32_keys_u64_ascending::RadixSortInitBinsAndCountersKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::u32_keys_u64_descending_RadixSortInitBinsAndCountersKernel_pointer
  cuvs::detail::radix_launch::u32_keys_u64_descending::RadixSortInitBinsAndCountersKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::u32_keys_u64_ascending_RadixSortOnesweepKernel_pointer
  cuvs::detail::radix_launch::u32_keys_u64_ascending::RadixSortOnesweepKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::u32_keys_u64_ascending_RadixSortSingleTileKernel_pointer
  cuvs::detail::radix_launch::u32_keys_u64_ascending::RadixSortSingleTileKernel();

namespace cuvs::detail::radix_launch {
using pairs_i32_i32_u64_ascending = cub::detail::radix_sort::DeviceRadixSortKernelSource<
  cub::detail::radix_sort::policy_selector_from_types<int, int, unsigned long long>,
  cub::SortOrder::Ascending,
  int,
  int,
  unsigned long long,
  cub::detail::identity_decomposer_t>;
using pairs_i64_i64_u64_ascending = cub::detail::radix_sort::DeviceRadixSortKernelSource<
  cub::detail::radix_sort::policy_selector_from_types<long, long, unsigned long long>,
  cub::SortOrder::Ascending,
  long,
  long,
  unsigned long long,
  cub::detail::identity_decomposer_t>;
using mst_zip_i64_u64 = cub::detail::merge_sort::DeviceMergeSortKernelSource<
  cub::detail::merge_sort::policy_selector_from_types<
    thrust::zip_iterator<cuda::std::tuple<long*, long*, raft::KeyValuePair<long, float>*>>>,
  thrust::zip_iterator<cuda::std::tuple<long*, long*, raft::KeyValuePair<long, float>*>>,
  thrust::zip_iterator<cuda::std::tuple<long*>>,
  thrust::zip_iterator<cuda::std::tuple<long*, long*, raft::KeyValuePair<long, float>*>>,
  thrust::zip_iterator<cuda::std::tuple<long*>>,
  unsigned long long,
  cuvs::sparse::neighbors::detail::TupleComp>;
using mst_zip_i64_u32 = cub::detail::merge_sort::DeviceMergeSortKernelSource<
  cub::detail::merge_sort::policy_selector_from_types<
    thrust::zip_iterator<cuda::std::tuple<long*, long*, raft::KeyValuePair<long, float>*>>>,
  thrust::zip_iterator<cuda::std::tuple<long*, long*, raft::KeyValuePair<long, float>*>>,
  thrust::zip_iterator<cuda::std::tuple<long*>>,
  thrust::zip_iterator<cuda::std::tuple<long*, long*, raft::KeyValuePair<long, float>*>>,
  thrust::zip_iterator<cuda::std::tuple<long*>>,
  unsigned int,
  cuvs::sparse::neighbors::detail::TupleComp>;
using mst_zip_i32_u64 = cub::detail::merge_sort::DeviceMergeSortKernelSource<
  cub::detail::merge_sort::policy_selector_from_types<
    thrust::zip_iterator<cuda::std::tuple<int*, int*, raft::KeyValuePair<int, float>*>>>,
  thrust::zip_iterator<cuda::std::tuple<int*, int*, raft::KeyValuePair<int, float>*>>,
  thrust::zip_iterator<cuda::std::tuple<int*>>,
  thrust::zip_iterator<cuda::std::tuple<int*, int*, raft::KeyValuePair<int, float>*>>,
  thrust::zip_iterator<cuda::std::tuple<int*>>,
  unsigned long long,
  cuvs::sparse::neighbors::detail::TupleComp>;
using mst_zip_i32_u32 = cub::detail::merge_sort::DeviceMergeSortKernelSource<
  cub::detail::merge_sort::policy_selector_from_types<
    thrust::zip_iterator<cuda::std::tuple<int*, int*, raft::KeyValuePair<int, float>*>>>,
  thrust::zip_iterator<cuda::std::tuple<int*, int*, raft::KeyValuePair<int, float>*>>,
  thrust::zip_iterator<cuda::std::tuple<int*>>,
  thrust::zip_iterator<cuda::std::tuple<int*, int*, raft::KeyValuePair<int, float>*>>,
  thrust::zip_iterator<cuda::std::tuple<int*>>,
  unsigned int,
  cuvs::sparse::neighbors::detail::TupleComp>;
using pairs_i32_i32_u64_ascending_RadixSortOnesweepKernel_pointer =
  void (*)(int*,
           int*,
           unsigned long long*,
           unsigned long long const*,
           int*,
           int const*,
           int*,
           int const*,
           int,
           int,
           int,
           cub::detail::identity_decomposer_t);
using pairs_i32_i32_u64_ascending_RadixSortInitLookbackKernel_pointer = void (*)(int*,
                                                                                 unsigned long,
                                                                                 int*,
                                                                                 unsigned long);
using pairs_i32_i32_u64_ascending_RadixSortInitBinsAndCountersKernel_pointer =
  void (*)(int*, unsigned long, unsigned long long*, unsigned long);
using pairs_i32_i32_u64_ascending_RadixSortExclusiveSumKernel_pointer =
  void (*)(unsigned long long*);
using pairs_i32_i32_u64_ascending_RadixSortHistogramKernel_pointer =
  void (*)(unsigned long long*,
           int const*,
           unsigned long long,
           int,
           int,
           cub::detail::identity_decomposer_t);
using pairs_i32_i32_u64_ascending_RadixSortSingleTileKernel_pointer =
  void (*)(int const*,
           int*,
           int const*,
           int*,
           unsigned long long,
           int,
           int,
           cub::detail::identity_decomposer_t);
using pairs_i64_i64_u64_ascending_RadixSortOnesweepKernel_pointer =
  void (*)(int*,
           int*,
           unsigned long long*,
           unsigned long long const*,
           long*,
           long const*,
           long*,
           long const*,
           int,
           int,
           int,
           cub::detail::identity_decomposer_t);
using pairs_i64_i64_u64_ascending_RadixSortInitLookbackKernel_pointer = void (*)(int*,
                                                                                 unsigned long,
                                                                                 int*,
                                                                                 unsigned long);
using pairs_i64_i64_u64_ascending_RadixSortInitBinsAndCountersKernel_pointer =
  void (*)(int*, unsigned long, unsigned long long*, unsigned long);
using pairs_i64_i64_u64_ascending_RadixSortExclusiveSumKernel_pointer =
  void (*)(unsigned long long*);
using pairs_i64_i64_u64_ascending_RadixSortHistogramKernel_pointer =
  void (*)(unsigned long long*,
           long const*,
           unsigned long long,
           int,
           int,
           cub::detail::identity_decomposer_t);
using pairs_i64_i64_u64_ascending_RadixSortSingleTileKernel_pointer =
  void (*)(long const*,
           long*,
           long const*,
           long*,
           unsigned long long,
           int,
           int,
           cub::detail::identity_decomposer_t);
using mst_zip_i64_u64_MergeSortMergeKernel_pointer =
  void (*)(bool,
           thrust::zip_iterator<cuda::std::tuple<long*, long*, raft::KeyValuePair<long, float>*>>,
           thrust::zip_iterator<cuda::std::tuple<long*>>,
           unsigned long long,
           cuda::std::tuple<long, long, raft::KeyValuePair<long, float>>*,
           cuda::std::tuple<long>*,
           cuvs::sparse::neighbors::detail::TupleComp,
           unsigned long long*,
           unsigned long long,
           cub::detail::vsmem_t);
using mst_zip_i64_u64_MergeSortPartitionKernel_pointer =
  void (*)(bool,
           thrust::zip_iterator<cuda::std::tuple<long*, long*, raft::KeyValuePair<long, float>*>>,
           cuda::std::tuple<long, long, raft::KeyValuePair<long, float>>*,
           unsigned long long,
           unsigned long long,
           unsigned long long*,
           cuvs::sparse::neighbors::detail::TupleComp,
           unsigned long long,
           int);
using mst_zip_i64_u64_MergeSortBlockSortKernel_pointer =
  void (*)(bool,
           thrust::zip_iterator<cuda::std::tuple<long*, long*, raft::KeyValuePair<long, float>*>>,
           thrust::zip_iterator<cuda::std::tuple<long*>>,
           thrust::zip_iterator<cuda::std::tuple<long*, long*, raft::KeyValuePair<long, float>*>>,
           thrust::zip_iterator<cuda::std::tuple<long*>>,
           unsigned long long,
           cuda::std::tuple<long, long, raft::KeyValuePair<long, float>>*,
           cuda::std::tuple<long>*,
           cuvs::sparse::neighbors::detail::TupleComp,
           cub::detail::vsmem_t);
using mst_zip_i64_u32_MergeSortMergeKernel_pointer =
  void (*)(bool,
           thrust::zip_iterator<cuda::std::tuple<long*, long*, raft::KeyValuePair<long, float>*>>,
           thrust::zip_iterator<cuda::std::tuple<long*>>,
           unsigned int,
           cuda::std::tuple<long, long, raft::KeyValuePair<long, float>>*,
           cuda::std::tuple<long>*,
           cuvs::sparse::neighbors::detail::TupleComp,
           unsigned int*,
           unsigned int,
           cub::detail::vsmem_t);
using mst_zip_i64_u32_MergeSortPartitionKernel_pointer =
  void (*)(bool,
           thrust::zip_iterator<cuda::std::tuple<long*, long*, raft::KeyValuePair<long, float>*>>,
           cuda::std::tuple<long, long, raft::KeyValuePair<long, float>>*,
           unsigned int,
           unsigned int,
           unsigned int*,
           cuvs::sparse::neighbors::detail::TupleComp,
           unsigned int,
           int);
using mst_zip_i64_u32_MergeSortBlockSortKernel_pointer =
  void (*)(bool,
           thrust::zip_iterator<cuda::std::tuple<long*, long*, raft::KeyValuePair<long, float>*>>,
           thrust::zip_iterator<cuda::std::tuple<long*>>,
           thrust::zip_iterator<cuda::std::tuple<long*, long*, raft::KeyValuePair<long, float>*>>,
           thrust::zip_iterator<cuda::std::tuple<long*>>,
           unsigned int,
           cuda::std::tuple<long, long, raft::KeyValuePair<long, float>>*,
           cuda::std::tuple<long>*,
           cuvs::sparse::neighbors::detail::TupleComp,
           cub::detail::vsmem_t);
using mst_zip_i32_u64_MergeSortMergeKernel_pointer =
  void (*)(bool,
           thrust::zip_iterator<cuda::std::tuple<int*, int*, raft::KeyValuePair<int, float>*>>,
           thrust::zip_iterator<cuda::std::tuple<int*>>,
           unsigned long long,
           cuda::std::tuple<int, int, raft::KeyValuePair<int, float>>*,
           cuda::std::tuple<int>*,
           cuvs::sparse::neighbors::detail::TupleComp,
           unsigned long long*,
           unsigned long long,
           cub::detail::vsmem_t);
using mst_zip_i32_u64_MergeSortPartitionKernel_pointer =
  void (*)(bool,
           thrust::zip_iterator<cuda::std::tuple<int*, int*, raft::KeyValuePair<int, float>*>>,
           cuda::std::tuple<int, int, raft::KeyValuePair<int, float>>*,
           unsigned long long,
           unsigned long long,
           unsigned long long*,
           cuvs::sparse::neighbors::detail::TupleComp,
           unsigned long long,
           int);
using mst_zip_i32_u64_MergeSortBlockSortKernel_pointer =
  void (*)(bool,
           thrust::zip_iterator<cuda::std::tuple<int*, int*, raft::KeyValuePair<int, float>*>>,
           thrust::zip_iterator<cuda::std::tuple<int*>>,
           thrust::zip_iterator<cuda::std::tuple<int*, int*, raft::KeyValuePair<int, float>*>>,
           thrust::zip_iterator<cuda::std::tuple<int*>>,
           unsigned long long,
           cuda::std::tuple<int, int, raft::KeyValuePair<int, float>>*,
           cuda::std::tuple<int>*,
           cuvs::sparse::neighbors::detail::TupleComp,
           cub::detail::vsmem_t);
using mst_zip_i32_u32_MergeSortMergeKernel_pointer =
  void (*)(bool,
           thrust::zip_iterator<cuda::std::tuple<int*, int*, raft::KeyValuePair<int, float>*>>,
           thrust::zip_iterator<cuda::std::tuple<int*>>,
           unsigned int,
           cuda::std::tuple<int, int, raft::KeyValuePair<int, float>>*,
           cuda::std::tuple<int>*,
           cuvs::sparse::neighbors::detail::TupleComp,
           unsigned int*,
           unsigned int,
           cub::detail::vsmem_t);
using mst_zip_i32_u32_MergeSortPartitionKernel_pointer =
  void (*)(bool,
           thrust::zip_iterator<cuda::std::tuple<int*, int*, raft::KeyValuePair<int, float>*>>,
           cuda::std::tuple<int, int, raft::KeyValuePair<int, float>>*,
           unsigned int,
           unsigned int,
           unsigned int*,
           cuvs::sparse::neighbors::detail::TupleComp,
           unsigned int,
           int);
using mst_zip_i32_u32_MergeSortBlockSortKernel_pointer =
  void (*)(bool,
           thrust::zip_iterator<cuda::std::tuple<int*, int*, raft::KeyValuePair<int, float>*>>,
           thrust::zip_iterator<cuda::std::tuple<int*>>,
           thrust::zip_iterator<cuda::std::tuple<int*, int*, raft::KeyValuePair<int, float>*>>,
           thrust::zip_iterator<cuda::std::tuple<int*>>,
           unsigned int,
           cuda::std::tuple<int, int, raft::KeyValuePair<int, float>>*,
           cuda::std::tuple<int>*,
           cuvs::sparse::neighbors::detail::TupleComp,
           cub::detail::vsmem_t);
}  // namespace cuvs::detail::radix_launch

template <>
__host__ __device__
  cuvs::detail::radix_launch::pairs_i32_i32_u64_ascending_RadixSortOnesweepKernel_pointer
  cuvs::detail::radix_launch::pairs_i32_i32_u64_ascending::RadixSortOnesweepKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::pairs_i32_i32_u64_ascending_RadixSortInitLookbackKernel_pointer
  cuvs::detail::radix_launch::pairs_i32_i32_u64_ascending::RadixSortInitLookbackKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::pairs_i32_i32_u64_ascending_RadixSortInitBinsAndCountersKernel_pointer
  cuvs::detail::radix_launch::pairs_i32_i32_u64_ascending::RadixSortInitBinsAndCountersKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::pairs_i32_i32_u64_ascending_RadixSortExclusiveSumKernel_pointer
  cuvs::detail::radix_launch::pairs_i32_i32_u64_ascending::RadixSortExclusiveSumKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::pairs_i32_i32_u64_ascending_RadixSortHistogramKernel_pointer
  cuvs::detail::radix_launch::pairs_i32_i32_u64_ascending::RadixSortHistogramKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::pairs_i32_i32_u64_ascending_RadixSortSingleTileKernel_pointer
  cuvs::detail::radix_launch::pairs_i32_i32_u64_ascending::RadixSortSingleTileKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::pairs_i64_i64_u64_ascending_RadixSortOnesweepKernel_pointer
  cuvs::detail::radix_launch::pairs_i64_i64_u64_ascending::RadixSortOnesweepKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::pairs_i64_i64_u64_ascending_RadixSortInitLookbackKernel_pointer
  cuvs::detail::radix_launch::pairs_i64_i64_u64_ascending::RadixSortInitLookbackKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::pairs_i64_i64_u64_ascending_RadixSortInitBinsAndCountersKernel_pointer
  cuvs::detail::radix_launch::pairs_i64_i64_u64_ascending::RadixSortInitBinsAndCountersKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::pairs_i64_i64_u64_ascending_RadixSortExclusiveSumKernel_pointer
  cuvs::detail::radix_launch::pairs_i64_i64_u64_ascending::RadixSortExclusiveSumKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::pairs_i64_i64_u64_ascending_RadixSortHistogramKernel_pointer
  cuvs::detail::radix_launch::pairs_i64_i64_u64_ascending::RadixSortHistogramKernel();

template <>
__host__ __device__
  cuvs::detail::radix_launch::pairs_i64_i64_u64_ascending_RadixSortSingleTileKernel_pointer
  cuvs::detail::radix_launch::pairs_i64_i64_u64_ascending::RadixSortSingleTileKernel();

template <>
__host__ __device__ cuvs::detail::radix_launch::mst_zip_i64_u64_MergeSortMergeKernel_pointer
cuvs::detail::radix_launch::mst_zip_i64_u64::MergeSortMergeKernel();

template <>
__host__ __device__ cuvs::detail::radix_launch::mst_zip_i64_u64_MergeSortPartitionKernel_pointer
cuvs::detail::radix_launch::mst_zip_i64_u64::MergeSortPartitionKernel();

template <>
__host__ __device__ cuvs::detail::radix_launch::mst_zip_i64_u64_MergeSortBlockSortKernel_pointer
cuvs::detail::radix_launch::mst_zip_i64_u64::MergeSortBlockSortKernel();

template <>
__host__ __device__ cuvs::detail::radix_launch::mst_zip_i64_u32_MergeSortMergeKernel_pointer
cuvs::detail::radix_launch::mst_zip_i64_u32::MergeSortMergeKernel();

template <>
__host__ __device__ cuvs::detail::radix_launch::mst_zip_i64_u32_MergeSortPartitionKernel_pointer
cuvs::detail::radix_launch::mst_zip_i64_u32::MergeSortPartitionKernel();

template <>
__host__ __device__ cuvs::detail::radix_launch::mst_zip_i64_u32_MergeSortBlockSortKernel_pointer
cuvs::detail::radix_launch::mst_zip_i64_u32::MergeSortBlockSortKernel();

template <>
__host__ __device__ cuvs::detail::radix_launch::mst_zip_i32_u64_MergeSortMergeKernel_pointer
cuvs::detail::radix_launch::mst_zip_i32_u64::MergeSortMergeKernel();

template <>
__host__ __device__ cuvs::detail::radix_launch::mst_zip_i32_u64_MergeSortPartitionKernel_pointer
cuvs::detail::radix_launch::mst_zip_i32_u64::MergeSortPartitionKernel();

template <>
__host__ __device__ cuvs::detail::radix_launch::mst_zip_i32_u64_MergeSortBlockSortKernel_pointer
cuvs::detail::radix_launch::mst_zip_i32_u64::MergeSortBlockSortKernel();

template <>
__host__ __device__ cuvs::detail::radix_launch::mst_zip_i32_u32_MergeSortMergeKernel_pointer
cuvs::detail::radix_launch::mst_zip_i32_u32::MergeSortMergeKernel();

template <>
__host__ __device__ cuvs::detail::radix_launch::mst_zip_i32_u32_MergeSortPartitionKernel_pointer
cuvs::detail::radix_launch::mst_zip_i32_u32::MergeSortPartitionKernel();

template <>
__host__ __device__ cuvs::detail::radix_launch::mst_zip_i32_u32_MergeSortBlockSortKernel_pointer
cuvs::detail::radix_launch::mst_zip_i32_u32::MergeSortBlockSortKernel();
