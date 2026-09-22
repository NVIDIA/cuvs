/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "kernel.cuh"
#include "planner.hpp"

#include <neighbors/detail/cagra/jit_lto_kernels/sample_filter_udf.cuh>
#include <neighbors/detail/cagra/shared_launcher_jit.hpp>

#include <memory>
#include <type_traits>

namespace cuvs::neighbors::cagra::experimental::flowann::detail {

template <typename DataTag,
          typename IndexTag,
          typename DistanceTag,
          typename SourceIndexTag,
          typename QueryTag,
          typename CodebookTag,
          typename SampleFilterJitTag,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT>
auto build_single_cta_launcher(
  const cuvs::neighbors::cagra::detail::dataset_descriptor_host<DataT, IndexT, DistanceT>&
    dataset_desc,
  bool topk_by_bitonic_sort,
  bool bitonic_sort_and_merge_multi_warps,
  std::unique_ptr<rtcx::udf_fatbin_fragment> sample_filter_udf_fragment)
  -> std::shared_ptr<rtcx::algorithm_launcher>
{
  single_cta_planner<DataTag,
                     IndexTag,
                     DistanceTag,
                     SourceIndexTag,
                     QueryTag,
                     CodebookTag,
                     SampleFilterJitTag>
    planner;

  if constexpr (std::is_same_v<CodebookTag, cuvs::neighbors::cagra::detail::tag_codebook_half>) {
    planner.add_setup_workspace_device_function(dataset_desc.team_size,
                                                dataset_desc.dataset_block_dim,
                                                dataset_desc.pq_len,
                                                dataset_desc.smem_dtype);
    planner.add_compute_distance_device_function(dataset_desc.team_size,
                                                 dataset_desc.dataset_block_dim,
                                                 dataset_desc.pq_len,
                                                 dataset_desc.smem_dtype);
  } else {
    planner.add_setup_workspace_device_function(dataset_desc.team_size,
                                                dataset_desc.dataset_block_dim);
    planner.add_compute_distance_device_function(
      dataset_desc.metric, dataset_desc.team_size, dataset_desc.dataset_block_dim);
  }
  planner.add_search_kernel(topk_by_bitonic_sort, bitonic_sort_and_merge_multi_warps);
  planner.add_sample_filter_device_function(std::move(sample_filter_udf_fragment));
  return planner.get_launcher();
}

template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT,
          typename SampleFilterJitTag>
auto make_single_cta_launcher(
  const cuvs::neighbors::cagra::detail::dataset_descriptor_host<DataT, IndexT, DistanceT>&
    dataset_desc,
  bool topk_by_bitonic_sort,
  bool bitonic_sort_and_merge_multi_warps,
  std::unique_ptr<rtcx::udf_fatbin_fragment> sample_filter_udf_fragment = nullptr)
  -> std::shared_ptr<rtcx::algorithm_launcher>
{
  using namespace cuvs::neighbors::cagra::detail;
  using data_tag         = decltype(get_data_type_tag<DataT>());
  using index_tag        = decltype(get_index_type_tag<IndexT>());
  using distance_tag     = decltype(get_distance_type_tag<DistanceT>());
  using source_index_tag = decltype(get_source_index_type_tag<SourceIndexT>());

  if (dataset_desc.is_vpq) {
    using query_tag    = query_type_tag_vpq_t<data_tag>;
    using codebook_tag = codebook_tag_vpq_t;
    return build_single_cta_launcher<data_tag,
                                     index_tag,
                                     distance_tag,
                                     source_index_tag,
                                     query_tag,
                                     codebook_tag,
                                     SampleFilterJitTag,
                                     DataT,
                                     IndexT,
                                     DistanceT,
                                     SourceIndexT>(dataset_desc,
                                                   topk_by_bitonic_sort,
                                                   bitonic_sort_and_merge_multi_warps,
                                                   std::move(sample_filter_udf_fragment));
  }

  using codebook_tag = codebook_tag_standard_t;
  if (dataset_desc.metric == cuvs::distance::DistanceType::BitwiseHamming) {
    using query_tag =
      query_type_tag_standard_t<data_tag, cuvs::distance::DistanceType::BitwiseHamming>;
    return build_single_cta_launcher<data_tag,
                                     index_tag,
                                     distance_tag,
                                     source_index_tag,
                                     query_tag,
                                     codebook_tag,
                                     SampleFilterJitTag,
                                     DataT,
                                     IndexT,
                                     DistanceT,
                                     SourceIndexT>(dataset_desc,
                                                   topk_by_bitonic_sort,
                                                   bitonic_sort_and_merge_multi_warps,
                                                   std::move(sample_filter_udf_fragment));
  }
  using query_tag = query_type_tag_standard_t<data_tag, cuvs::distance::DistanceType::L2Expanded>;
  return build_single_cta_launcher<data_tag,
                                   index_tag,
                                   distance_tag,
                                   source_index_tag,
                                   query_tag,
                                   codebook_tag,
                                   SampleFilterJitTag,
                                   DataT,
                                   IndexT,
                                   DistanceT,
                                   SourceIndexT>(dataset_desc,
                                                 topk_by_bitonic_sort,
                                                 bitonic_sort_and_merge_multi_warps,
                                                 std::move(sample_filter_udf_fragment));
}

template <typename DataTag,
          typename IndexTag,
          typename DistanceTag,
          typename SourceIndexTag,
          typename QueryTag,
          typename CodebookTag,
          typename SampleFilterJitTag,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT>
auto build_multi_cta_launcher(
  const cuvs::neighbors::cagra::detail::dataset_descriptor_host<DataT, IndexT, DistanceT>&
    dataset_desc,
  std::unique_ptr<rtcx::udf_fatbin_fragment> sample_filter_udf_fragment)
  -> std::shared_ptr<rtcx::algorithm_launcher>
{
  multi_cta_planner<DataTag,
                    IndexTag,
                    DistanceTag,
                    SourceIndexTag,
                    QueryTag,
                    CodebookTag,
                    SampleFilterJitTag>
    planner;
  if constexpr (std::is_same_v<CodebookTag, cuvs::neighbors::cagra::detail::tag_codebook_half>) {
    planner.add_setup_workspace_device_function(dataset_desc.team_size,
                                                dataset_desc.dataset_block_dim,
                                                dataset_desc.pq_len,
                                                dataset_desc.smem_dtype);
    planner.add_compute_distance_device_function(dataset_desc.team_size,
                                                 dataset_desc.dataset_block_dim,
                                                 dataset_desc.pq_len,
                                                 dataset_desc.smem_dtype);
  } else {
    planner.add_setup_workspace_device_function(dataset_desc.team_size,
                                                dataset_desc.dataset_block_dim);
    planner.add_compute_distance_device_function(
      dataset_desc.metric, dataset_desc.team_size, dataset_desc.dataset_block_dim);
  }
  planner.add_search_kernel();
  planner.add_sample_filter_device_function(std::move(sample_filter_udf_fragment));
  return planner.get_launcher();
}

template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT,
          typename SampleFilterJitTag>
auto make_multi_cta_launcher(
  const cuvs::neighbors::cagra::detail::dataset_descriptor_host<DataT, IndexT, DistanceT>&
    dataset_desc,
  std::unique_ptr<rtcx::udf_fatbin_fragment> sample_filter_udf_fragment = nullptr)
  -> std::shared_ptr<rtcx::algorithm_launcher>
{
  using namespace cuvs::neighbors::cagra::detail;
  using data_tag         = decltype(get_data_type_tag<DataT>());
  using index_tag        = decltype(get_index_type_tag<IndexT>());
  using distance_tag     = decltype(get_distance_type_tag<DistanceT>());
  using source_index_tag = decltype(get_source_index_type_tag<SourceIndexT>());
  if (dataset_desc.is_vpq) {
    using query_tag    = query_type_tag_vpq_t<data_tag>;
    using codebook_tag = codebook_tag_vpq_t;
    return build_multi_cta_launcher<data_tag,
                                    index_tag,
                                    distance_tag,
                                    source_index_tag,
                                    query_tag,
                                    codebook_tag,
                                    SampleFilterJitTag,
                                    DataT,
                                    IndexT,
                                    DistanceT,
                                    SourceIndexT>(dataset_desc,
                                                  std::move(sample_filter_udf_fragment));
  }
  using codebook_tag = codebook_tag_standard_t;
  if (dataset_desc.metric == cuvs::distance::DistanceType::BitwiseHamming) {
    using query_tag =
      query_type_tag_standard_t<data_tag, cuvs::distance::DistanceType::BitwiseHamming>;
    return build_multi_cta_launcher<data_tag,
                                    index_tag,
                                    distance_tag,
                                    source_index_tag,
                                    query_tag,
                                    codebook_tag,
                                    SampleFilterJitTag,
                                    DataT,
                                    IndexT,
                                    DistanceT,
                                    SourceIndexT>(dataset_desc,
                                                  std::move(sample_filter_udf_fragment));
  }
  using query_tag = query_type_tag_standard_t<data_tag, cuvs::distance::DistanceType::L2Expanded>;
  return build_multi_cta_launcher<data_tag,
                                  index_tag,
                                  distance_tag,
                                  source_index_tag,
                                  query_tag,
                                  codebook_tag,
                                  SampleFilterJitTag,
                                  DataT,
                                  IndexT,
                                  DistanceT,
                                  SourceIndexT>(dataset_desc,
                                                std::move(sample_filter_udf_fragment));
}

}  // namespace cuvs::neighbors::cagra::experimental::flowann::detail
