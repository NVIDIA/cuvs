/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../compute_distance.hpp"
#include "../shared_launcher_jit.hpp"
#include "sample_filter_udf.cuh"
#include "search_multi_cta_planner.hpp"
#include "search_multi_kernel_planner.hpp"
#include "search_single_cta_planner.hpp"

#include <cuvs/detail/jit_lto/AlgorithmLauncher.hpp>
#include <cuvs/detail/jit_lto/cagra/cagra_fragments.hpp>
#include <cuvs/distance/distance.hpp>

#include <memory>
#include <type_traits>

namespace cuvs::neighbors::cagra::detail {

namespace cagra_jit_launcher_factory_detail {

template <typename DataTag,
          typename IndexTag,
          typename DistTag,
          typename SourceTag,
          typename QueryTag,
          typename CodebookTag,
          typename SampleFilterJitTag,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT>
std::shared_ptr<AlgorithmLauncher> build_single_cta_launcher(
  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
  bool topk_by_bitonic_sort,
  bool bitonic_sort_and_merge_multi_warps,
  bool persistent,
  std::unique_ptr<UDFFatbinFragment> sample_filter_udf_fragment)
{
  single_cta_search::CagraSingleCtaSearchPlanner<DataTag,
                                                 IndexTag,
                                                 DistTag,
                                                 SourceTag,
                                                 QueryTag,
                                                 CodebookTag,
                                                 SampleFilterJitTag>
    planner(dataset_desc.metric,
            topk_by_bitonic_sort,
            bitonic_sort_and_merge_multi_warps,
            dataset_desc.team_size,
            dataset_desc.dataset_block_dim,
            dataset_desc.is_vpq,
            dataset_desc.pq_bits,
            dataset_desc.pq_len,
            persistent);

  if constexpr (std::is_same_v<CodebookTag, tag_codebook_half>) {
    planner.add_setup_workspace_device_function(
      dataset_desc.team_size, dataset_desc.dataset_block_dim, dataset_desc.pq_len);
    planner.add_compute_distance_device_function(
      dataset_desc.team_size, dataset_desc.dataset_block_dim, dataset_desc.pq_len);
  } else {
    planner.add_setup_workspace_device_function(dataset_desc.team_size,
                                                dataset_desc.dataset_block_dim);
    planner.add_compute_distance_device_function(
      dataset_desc.metric, dataset_desc.team_size, dataset_desc.dataset_block_dim);
  }
  planner.add_search_kernel_fragment(
    topk_by_bitonic_sort, bitonic_sort_and_merge_multi_warps, persistent);
  planner.add_sample_filter_device_function(std::move(sample_filter_udf_fragment));
  return planner.get_launcher();
}

template <typename DataTag,
          typename IndexTag,
          typename DistTag,
          typename SourceTag,
          typename QueryTag,
          typename CodebookTag,
          typename SampleFilterJitTag,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT>
std::shared_ptr<AlgorithmLauncher> build_multi_cta_launcher(
  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
  std::unique_ptr<UDFFatbinFragment> sample_filter_udf_fragment)
{
  multi_cta_search::CagraMultiCtaSearchPlanner<DataTag,
                                               IndexTag,
                                               DistTag,
                                               SourceTag,
                                               QueryTag,
                                               CodebookTag,
                                               SampleFilterJitTag>
    planner(dataset_desc.metric,
            dataset_desc.team_size,
            dataset_desc.dataset_block_dim,
            dataset_desc.is_vpq,
            dataset_desc.pq_bits,
            dataset_desc.pq_len);

  if constexpr (std::is_same_v<CodebookTag, tag_codebook_half>) {
    planner.add_setup_workspace_device_function(
      dataset_desc.team_size, dataset_desc.dataset_block_dim, dataset_desc.pq_len);
    planner.add_compute_distance_device_function(
      dataset_desc.team_size, dataset_desc.dataset_block_dim, dataset_desc.pq_len);
  } else {
    planner.add_setup_workspace_device_function(dataset_desc.team_size,
                                                dataset_desc.dataset_block_dim);
    planner.add_compute_distance_device_function(
      dataset_desc.metric, dataset_desc.team_size, dataset_desc.dataset_block_dim);
  }
  planner.add_search_multi_cta_kernel_fragment();
  planner.add_sample_filter_device_function(std::move(sample_filter_udf_fragment));
  return planner.get_launcher();
}

template <typename DataTag,
          typename IndexTag,
          typename DistTag,
          typename SourceTag,
          typename QueryTag,
          typename CodebookTag,
          typename SampleFilterJitTag,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT>
std::shared_ptr<AlgorithmLauncher> build_multi_kernel_launcher(
  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
  const char* linked_kernel_name,
  std::unique_ptr<UDFFatbinFragment> sample_filter_udf_fragment)
{
  multi_kernel_search::CagraMultiKernelSearchPlanner<DataTag,
                                                     IndexTag,
                                                     DistTag,
                                                     SourceTag,
                                                     QueryTag,
                                                     CodebookTag,
                                                     SampleFilterJitTag>
    planner(dataset_desc.metric,
            linked_kernel_name,
            dataset_desc.team_size,
            dataset_desc.dataset_block_dim,
            dataset_desc.is_vpq,
            dataset_desc.pq_bits,
            dataset_desc.pq_len);
  if constexpr (std::is_same_v<CodebookTag, tag_codebook_half>) {
    planner.add_setup_workspace_device_function(
      dataset_desc.team_size, dataset_desc.dataset_block_dim, dataset_desc.pq_len);
    planner.add_compute_distance_device_function(
      dataset_desc.team_size, dataset_desc.dataset_block_dim, dataset_desc.pq_len);
  } else {
    planner.add_setup_workspace_device_function(dataset_desc.team_size,
                                                dataset_desc.dataset_block_dim);
    planner.add_compute_distance_device_function(
      dataset_desc.metric, dataset_desc.team_size, dataset_desc.dataset_block_dim);
  }
  planner.add_sample_filter_device_function(std::move(sample_filter_udf_fragment));
  planner.add_linked_kernel(linked_kernel_name);
  return planner.get_launcher();
}

/// `apply_filter_kernel.cu.in` does not link `setup_workspace` / `compute_distance`; only
/// `sample_filter` + the apply-filter kernel fragment (see `fragment_tag_apply_filter_kernel`).
template <typename DataTag,
          typename IndexTag,
          typename DistTag,
          typename SourceTag,
          typename QueryTag,
          typename CodebookTag,
          typename SampleFilterJitTag,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT>
std::shared_ptr<AlgorithmLauncher> build_apply_filter_only_launcher(
  std::unique_ptr<UDFFatbinFragment> sample_filter_udf_fragment)
{
  multi_kernel_search::CagraMultiKernelSearchPlanner<DataTag,
                                                     IndexTag,
                                                     DistTag,
                                                     SourceTag,
                                                     QueryTag,
                                                     CodebookTag,
                                                     SampleFilterJitTag>
    planner("apply_filter_kernel");
  planner.add_sample_filter_device_function(std::move(sample_filter_udf_fragment));
  planner.add_linked_kernel("apply_filter_kernel");
  return planner.get_launcher();
}

}  // namespace cagra_jit_launcher_factory_detail

/// Build a JIT AlgorithmLauncher for single-CTA CAGRA search (runtime VPQ / metric → tag
/// dispatch). `SampleFilterJitTag` is `cuvs::neighbors::detail::tag_filter_none`,
/// `tag_filter_bitset`, or use `sample_filter_jit_tag_t<SAMPLE_FILTER_T>`.
template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT,
          typename SampleFilterJitTag>
std::shared_ptr<AlgorithmLauncher> make_cagra_single_cta_jit_launcher(
  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
  bool topk_by_bitonic_sort,
  bool bitonic_sort_and_merge_multi_warps,
  bool persistent,
  std::unique_ptr<UDFFatbinFragment> sample_filter_udf_fragment = nullptr)
{
  using DataTag   = decltype(get_data_type_tag<DataT>());
  using IndexTag  = decltype(get_index_type_tag<IndexT>());
  using DistTag   = decltype(get_distance_type_tag<DistanceT>());
  using SourceTag = decltype(get_source_index_type_tag<SourceIndexT>());

  if (dataset_desc.is_vpq) {
    using QueryTag    = query_type_tag_vpq_t<DataTag>;
    using CodebookTag = codebook_tag_vpq_t;
    return cagra_jit_launcher_factory_detail::build_single_cta_launcher<DataTag,
                                                                        IndexTag,
                                                                        DistTag,
                                                                        SourceTag,
                                                                        QueryTag,
                                                                        CodebookTag,
                                                                        SampleFilterJitTag,
                                                                        DataT,
                                                                        IndexT,
                                                                        DistanceT,
                                                                        SourceIndexT>(
      dataset_desc,
      topk_by_bitonic_sort,
      bitonic_sort_and_merge_multi_warps,
      persistent,
      std::move(sample_filter_udf_fragment));
  }
  using CodebookTag = codebook_tag_standard_t;
  if (dataset_desc.metric == cuvs::distance::DistanceType::BitwiseHamming) {
    using QueryTag =
      query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::BitwiseHamming>;
    return cagra_jit_launcher_factory_detail::build_single_cta_launcher<DataTag,
                                                                        IndexTag,
                                                                        DistTag,
                                                                        SourceTag,
                                                                        QueryTag,
                                                                        CodebookTag,
                                                                        SampleFilterJitTag,
                                                                        DataT,
                                                                        IndexT,
                                                                        DistanceT,
                                                                        SourceIndexT>(
      dataset_desc,
      topk_by_bitonic_sort,
      bitonic_sort_and_merge_multi_warps,
      persistent,
      std::move(sample_filter_udf_fragment));
  }
  using QueryTag = query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::L2Expanded>;
  return cagra_jit_launcher_factory_detail::build_single_cta_launcher<DataTag,
                                                                      IndexTag,
                                                                      DistTag,
                                                                      SourceTag,
                                                                      QueryTag,
                                                                      CodebookTag,
                                                                      SampleFilterJitTag,
                                                                      DataT,
                                                                      IndexT,
                                                                      DistanceT,
                                                                      SourceIndexT>(
    dataset_desc,
    topk_by_bitonic_sort,
    bitonic_sort_and_merge_multi_warps,
    persistent,
    std::move(sample_filter_udf_fragment));
}

/// Build a JIT AlgorithmLauncher for multi-CTA CAGRA search.
template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT,
          typename SampleFilterJitTag>
std::shared_ptr<AlgorithmLauncher> make_cagra_multi_cta_jit_launcher(
  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
  std::unique_ptr<UDFFatbinFragment> sample_filter_udf_fragment = nullptr)
{
  using DataTag   = decltype(get_data_type_tag<DataT>());
  using IndexTag  = decltype(get_index_type_tag<IndexT>());
  using DistTag   = decltype(get_distance_type_tag<DistanceT>());
  using SourceTag = decltype(get_source_index_type_tag<SourceIndexT>());

  if (dataset_desc.is_vpq) {
    using QueryTag    = query_type_tag_vpq_t<DataTag>;
    using CodebookTag = codebook_tag_vpq_t;
    return cagra_jit_launcher_factory_detail::build_multi_cta_launcher<DataTag,
                                                                       IndexTag,
                                                                       DistTag,
                                                                       SourceTag,
                                                                       QueryTag,
                                                                       CodebookTag,
                                                                       SampleFilterJitTag,
                                                                       DataT,
                                                                       IndexT,
                                                                       DistanceT,
                                                                       SourceIndexT>(
      dataset_desc, std::move(sample_filter_udf_fragment));
  }
  using CodebookTag = codebook_tag_standard_t;
  if (dataset_desc.metric == cuvs::distance::DistanceType::BitwiseHamming) {
    using QueryTag =
      query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::BitwiseHamming>;
    return cagra_jit_launcher_factory_detail::build_multi_cta_launcher<DataTag,
                                                                       IndexTag,
                                                                       DistTag,
                                                                       SourceTag,
                                                                       QueryTag,
                                                                       CodebookTag,
                                                                       SampleFilterJitTag,
                                                                       DataT,
                                                                       IndexT,
                                                                       DistanceT,
                                                                       SourceIndexT>(
      dataset_desc, std::move(sample_filter_udf_fragment));
  }
  using QueryTag = query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::L2Expanded>;
  return cagra_jit_launcher_factory_detail::build_multi_cta_launcher<DataTag,
                                                                     IndexTag,
                                                                     DistTag,
                                                                     SourceTag,
                                                                     QueryTag,
                                                                     CodebookTag,
                                                                     SampleFilterJitTag,
                                                                     DataT,
                                                                     IndexT,
                                                                     DistanceT,
                                                                     SourceIndexT>(
    dataset_desc, std::move(sample_filter_udf_fragment));
}

/// Build a JIT AlgorithmLauncher for multi-kernel CAGRA helpers that need `setup_workspace` and
/// `compute_distance` linked (e.g. `random_pickup`, `compute_distance_to_child_nodes`). For
/// `apply_filter_kernel` only, use `make_cagra_apply_filter_jit_launcher` instead. Use
/// `SampleFilterJitTag = tag_cagra_jit_sample_filter_link_absent` (default) when the kernel does
/// not link `sample_filter`; otherwise `sample_filter_jit_tag_t<SAMPLE_FILTER_T>` or a
/// `tag_filter_*` from `common_fragments.hpp`.
template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT,
          typename SampleFilterJitTag = tag_cagra_jit_sample_filter_link_absent>
std::shared_ptr<AlgorithmLauncher> make_cagra_multi_kernel_jit_launcher(
  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
  const char* linked_kernel_name,
  std::unique_ptr<UDFFatbinFragment> sample_filter_udf_fragment = nullptr)
{
  using DataTag   = decltype(get_data_type_tag<DataT>());
  using IndexTag  = decltype(get_index_type_tag<IndexT>());
  using DistTag   = decltype(get_distance_type_tag<DistanceT>());
  using SourceTag = decltype(get_source_index_type_tag<SourceIndexT>());

  if (dataset_desc.is_vpq) {
    using QueryTag    = query_type_tag_vpq_t<DataTag>;
    using CodebookTag = codebook_tag_vpq_t;
    return cagra_jit_launcher_factory_detail::build_multi_kernel_launcher<DataTag,
                                                                          IndexTag,
                                                                          DistTag,
                                                                          SourceTag,
                                                                          QueryTag,
                                                                          CodebookTag,
                                                                          SampleFilterJitTag,
                                                                          DataT,
                                                                          IndexT,
                                                                          DistanceT,
                                                                          SourceIndexT>(
      dataset_desc, linked_kernel_name, std::move(sample_filter_udf_fragment));
  }
  using CodebookTag = codebook_tag_standard_t;
  if (dataset_desc.metric == cuvs::distance::DistanceType::BitwiseHamming) {
    using QueryTag =
      query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::BitwiseHamming>;
    return cagra_jit_launcher_factory_detail::build_multi_kernel_launcher<DataTag,
                                                                          IndexTag,
                                                                          DistTag,
                                                                          SourceTag,
                                                                          QueryTag,
                                                                          CodebookTag,
                                                                          SampleFilterJitTag,
                                                                          DataT,
                                                                          IndexT,
                                                                          DistanceT,
                                                                          SourceIndexT>(
      dataset_desc, linked_kernel_name, std::move(sample_filter_udf_fragment));
  }
  using QueryTag = query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::L2Expanded>;
  return cagra_jit_launcher_factory_detail::build_multi_kernel_launcher<DataTag,
                                                                        IndexTag,
                                                                        DistTag,
                                                                        SourceTag,
                                                                        QueryTag,
                                                                        CodebookTag,
                                                                        SampleFilterJitTag,
                                                                        DataT,
                                                                        IndexT,
                                                                        DistanceT,
                                                                        SourceIndexT>(
    dataset_desc, linked_kernel_name, std::move(sample_filter_udf_fragment));
}

/// JIT launcher for the post-search `apply_filter_kernel` only (no workspace / distance fragments).
/// Tag dispatch matches `make_cagra_multi_kernel_jit_launcher` so VPQ / metric / codebook stay
/// aligned with the search descriptor; `fragment_tag_apply_filter_kernel` only keys on index,
/// distance, and source-index tags (see `apply_filter_matrix.json`).
template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT,
          typename SampleFilterJitTag>
std::shared_ptr<AlgorithmLauncher> make_cagra_apply_filter_jit_launcher(
  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
  std::unique_ptr<UDFFatbinFragment> sample_filter_udf_fragment = nullptr)
{
  using DataTag   = decltype(get_data_type_tag<DataT>());
  using IndexTag  = decltype(get_index_type_tag<IndexT>());
  using DistTag   = decltype(get_distance_type_tag<DistanceT>());
  using SourceTag = decltype(get_source_index_type_tag<SourceIndexT>());

  if (dataset_desc.is_vpq) {
    using QueryTag    = query_type_tag_vpq_t<DataTag>;
    using CodebookTag = codebook_tag_vpq_t;
    return cagra_jit_launcher_factory_detail::build_apply_filter_only_launcher<DataTag,
                                                                               IndexTag,
                                                                               DistTag,
                                                                               SourceTag,
                                                                               QueryTag,
                                                                               CodebookTag,
                                                                               SampleFilterJitTag,
                                                                               DataT,
                                                                               IndexT,
                                                                               DistanceT,
                                                                               SourceIndexT>(
      std::move(sample_filter_udf_fragment));
  }
  using CodebookTag = codebook_tag_standard_t;
  if (dataset_desc.metric == cuvs::distance::DistanceType::BitwiseHamming) {
    using QueryTag =
      query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::BitwiseHamming>;
    return cagra_jit_launcher_factory_detail::build_apply_filter_only_launcher<DataTag,
                                                                               IndexTag,
                                                                               DistTag,
                                                                               SourceTag,
                                                                               QueryTag,
                                                                               CodebookTag,
                                                                               SampleFilterJitTag,
                                                                               DataT,
                                                                               IndexT,
                                                                               DistanceT,
                                                                               SourceIndexT>(
      std::move(sample_filter_udf_fragment));
  }
  using QueryTag = query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::L2Expanded>;
  return cagra_jit_launcher_factory_detail::build_apply_filter_only_launcher<DataTag,
                                                                             IndexTag,
                                                                             DistTag,
                                                                             SourceTag,
                                                                             QueryTag,
                                                                             CodebookTag,
                                                                             SampleFilterJitTag,
                                                                             DataT,
                                                                             IndexT,
                                                                             DistanceT,
                                                                             SourceIndexT>(
    std::move(sample_filter_udf_fragment));
}

}  // namespace cuvs::neighbors::cagra::detail
