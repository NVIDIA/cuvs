/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "search_multi_cta.cuh"
#include "search_multi_kernel.cuh"
#include "search_plan.cuh"
#include "search_single_cta.cuh"
#include <neighbors/detail/cagra/compute_distance-ext.cuh>

#include <raft/core/resource/custom_resource.hpp>
#include <raft/util/cache.hpp>

#include <cuvs/neighbors/common.hpp>

namespace cuvs::neighbors::cagra::detail {

template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename CagraSampleFilterT = cuvs::neighbors::filtering::none_sample_filter,
          typename SourceIndexT       = IndexT,
          typename OutputIndexT       = SourceIndexT>
class factory {
 public:
  /**
   * Create a search structure for dataset with dim features.
   */
  static std::unique_ptr<
    search_plan_impl<DataT, IndexT, DistanceT, CagraSampleFilterT, SourceIndexT, OutputIndexT>>
  create(raft::resources const& res,
         search_params const& params,
         const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
         int64_t dim,
         int64_t dataset_size,
         int64_t graph_degree,
         uint32_t topk)
  {
    search_plan_impl_base plan(params, dim, dataset_size, graph_degree, topk);
    return dispatch_kernel(res, plan, dataset_desc);
  }

 private:
  static std::unique_ptr<
    search_plan_impl<DataT, IndexT, DistanceT, CagraSampleFilterT, SourceIndexT, OutputIndexT>>
  dispatch_kernel(raft::resources const& res,
                  search_plan_impl_base& plan,
                  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc)
  {
    if (plan.algo == search_algo::SINGLE_CTA) {
      return std::make_unique<
        single_cta_search::
          search<DataT, IndexT, DistanceT, CagraSampleFilterT, SourceIndexT, OutputIndexT>>(
        res, plan, dataset_desc, plan.dim, plan.dataset_size, plan.graph_degree, plan.topk);
    } else if (plan.algo == search_algo::MULTI_CTA) {
      return std::make_unique<
        multi_cta_search::
          search<DataT, IndexT, DistanceT, CagraSampleFilterT, SourceIndexT, OutputIndexT>>(
        res, plan, dataset_desc, plan.dim, plan.dataset_size, plan.graph_degree, plan.topk);
    } else {
      return std::make_unique<
        multi_kernel_search::
          search<DataT, IndexT, DistanceT, CagraSampleFilterT, SourceIndexT, OutputIndexT>>(
        res, plan, dataset_desc, plan.dim, plan.dataset_size, plan.graph_degree, plan.topk);
    }
  }
};

/*
Caching of dataset/distance descriptor initialization
  (see `dataset_descriptor_init_with_cache` below).
 */
namespace descriptor_cache {

/**
 * The key for caching consists of a minimal set of fields that uniquely define the descriptor.
 * The key field names are the same as of the descriptor and the contents are not relevant for
 * caching.
 */
struct key {
  uint64_t data_ptr;
  uint64_t n_rows;
  uint32_t dim;
  uint32_t extra_val;  // this one has different meanings for different descriptor types
  uint32_t team_size;
  uint32_t metric;
};

template <typename DatasetT>
auto make_key(const cagra::search_params& params,
              const DatasetT& dataset,
              cuvs::distance::DistanceType metric)
  -> std::enable_if_t<is_strided_dataset_v<DatasetT>, key>
{
  return key{reinterpret_cast<uint64_t>(dataset.view().data_handle()),
             uint64_t(dataset.n_rows()),
             dataset.dim(),
             dataset.stride(),
             uint32_t(params.team_size),
             uint32_t(metric)};
}

template <typename DatasetT>
auto make_key(const cagra::search_params& params,
              const DatasetT& dataset,
              cuvs::distance::DistanceType metric)
  -> std::enable_if_t<cuvs::preprocessing::quantize::pq::is_vpq_dataset_v<DatasetT>, key>
{
  return key{reinterpret_cast<uint64_t>(dataset.data.data_handle()),
             uint64_t(dataset.n_rows()),
             dataset.dim(),
             uint32_t(reinterpret_cast<uint64_t>(dataset.pq_code_book().data_handle()) >> 6),
             uint32_t(params.team_size),
             uint32_t(metric)};
}

inline auto operator==(const key& a, const key& b) -> bool
{
  return a.data_ptr == b.data_ptr && a.n_rows == b.n_rows && a.dim == b.dim &&
         a.extra_val == b.extra_val && a.team_size == b.team_size && a.metric == b.metric;
}

struct key_hash {
  inline auto operator()(const key& x) const noexcept -> std::size_t
  {
    return size_t{x.data_ptr} + size_t{x.n_rows} * size_t{x.dim} * size_t{x.extra_val} +
           (size_t{x.team_size} ^ size_t{x.metric});
  }
};

template <typename DataT, typename IndexT, typename DistanceT>
struct store {
  /** Number of descriptors to cache. */
  static constexpr size_t kDefaultSize = 100;
  raft::cache::
    lru<key, key_hash, std::equal_to<>, dataset_descriptor_host<DataT, IndexT, DistanceT>>
      value{kDefaultSize};
};

}  // namespace descriptor_cache

/**
 * Call `dataset_descriptor_init` with memoization.
 * (NB: `dataset_descriptor_init` is a function in a generated header file
 * `neighbors/detail/cagra/compute_distance-ext.cuh`).
 *
 * `dataset_descriptor_init`  involves calling a CUDA kernel to resolve device symbols before the
 * main search kernel runs. This adds an extra unwanted latency.
 * Caching the the descriptor helps to hide this latency for repeated searches.
 *
 */
template <typename DataT, typename IndexT, typename DistanceT, typename DatasetT>
auto dataset_descriptor_init_with_cache(const raft::resources& res,
                                        const cagra::search_params& params,
                                        const DatasetT& dataset,
                                        cuvs::distance::DistanceType metric,
                                        const DistanceT* dataset_norms = nullptr)
  -> dataset_descriptor_host<DataT, IndexT, DistanceT>
{
  auto key = descriptor_cache::make_key(params, dataset, metric);
  auto& cache =
    raft::resource::get_custom_resource<descriptor_cache::store<DataT, IndexT, DistanceT>>(res)
      ->value;
  dataset_descriptor_host<DataT, IndexT, DistanceT> desc;
  if (!cache.get(key, &desc)) {
    desc =
      dataset_descriptor_init<DataT, IndexT, DistanceT>(params, dataset, metric, dataset_norms);
    cache.set(key, desc);
  }
  return desc;
}

};  // namespace cuvs::neighbors::cagra::detail
