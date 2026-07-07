/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../../core/nvtx.hpp"
#include "factory.cuh"
#include "sample_filter_utils.cuh"
#include "search_plan.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuvs/distance/distance.hpp>

#include <cuvs/neighbors/cagra.hpp>

// TODO: Fix these when ivf methods are moved over
#include "../../ivf_common.cuh"
#include "../../ivf_pq/ivf_pq_search.cuh"
#include <cuvs/neighbors/common.hpp>
#include <cuvs/preprocessing/quantize/vpq_dataset.hpp>

// TODO: This shouldn't be calling spatial/knn apis
#include "../ann_utils.cuh"

#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/reduce.cuh>

// All includes are done before opening namespace to avoid nested namespace issues
namespace cuvs::neighbors::cagra::detail {

template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename CagraSampleFilterT,
          typename SourceIdxT = IndexT,
          typename OutputIdxT = SourceIdxT>
void search_main_core(
  raft::resources const& res,
  search_params params,
  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
  raft::device_matrix_view<const IndexT, int64_t, raft::row_major> graph,
  std::optional<raft::device_vector_view<const SourceIdxT, int64_t>> source_indices,
  raft::device_matrix_view<const DataT, int64_t, raft::row_major> queries,
  raft::device_matrix_view<OutputIdxT, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<DistanceT, int64_t, raft::row_major> distances,
  CagraSampleFilterT sample_filter = CagraSampleFilterT())
{
  static_assert(std::is_same_v<IndexT, uint32_t>,
                "Only uint32_t is supported as the graph element type (internal index type)");
  RAFT_LOG_DEBUG("# dataset size = %lu, dim = %lu\n",
                 static_cast<size_t>(graph.extent(0)),
                 static_cast<size_t>(queries.extent(1)));
  RAFT_LOG_DEBUG("# query size = %lu, dim = %lu\n",
                 static_cast<size_t>(queries.extent(0)),
                 static_cast<size_t>(queries.extent(1)));
  const uint32_t topk = neighbors.extent(1);

  cudaDeviceProp deviceProp = raft::resource::get_device_properties(res);
  if (params.max_queries == 0) {
    params.max_queries = std::min<size_t>(queries.extent(0), deviceProp.maxGridSize[1]);
  }

  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "cagra::search(max_queries = %u, k = %u, dim = %zu)",
    params.max_queries,
    topk,
    queries.extent(1));

  using CagraSampleFilterT_s = typename CagraSampleFilterT_Selector<CagraSampleFilterT>::type;
  std::unique_ptr<
    search_plan_impl<DataT, IndexT, DistanceT, CagraSampleFilterT_s, SourceIdxT, OutputIdxT>>
    plan = factory<DataT, IndexT, DistanceT, CagraSampleFilterT_s, SourceIdxT, OutputIdxT>::create(
      res, params, dataset_desc, queries.extent(1), graph.extent(0), graph.extent(1), topk);

  plan->check(topk);

  RAFT_LOG_DEBUG("Cagra search");
  const uint32_t max_queries = plan->max_queries;
  const uint32_t query_dim   = queries.extent(1);

  for (unsigned qid = 0; qid < queries.extent(0); qid += max_queries) {
    const uint32_t n_queries = std::min<std::size_t>(max_queries, queries.extent(0) - qid);
    auto _topk_indices_ptr   = neighbors.data_handle() + (topk * qid);
    auto _topk_distances_ptr = distances.data_handle() + (topk * qid);
    // todo(tfeher): one could keep distances optional and pass nullptr
    const auto* _query_ptr = queries.data_handle() + (query_dim * qid);
    const auto* _seed_ptr =
      plan->num_seeds > 0
        ? reinterpret_cast<const IndexT*>(plan->dev_seed.data()) + (plan->num_seeds * qid)
        : nullptr;
    uint32_t* _num_executed_iterations = nullptr;

    (*plan)(res,
            graph,
            source_indices,
            _topk_indices_ptr,
            _topk_distances_ptr,
            _query_ptr,
            n_queries,
            _seed_ptr,
            _num_executed_iterations,
            topk,
            set_offset(sample_filter, qid));
  }
}

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [build](#build) documentation for a usage example.
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the CAGRA graph
 * @tparam OutputIdxT type of the returned indices
 *
 * @param[in] handle
 * @param[in] params configure the search
 * @param[in] idx ivf-pq constructed index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 */
template <typename T,
          typename OutputIdxT,
          typename CagraSampleFilterT,
          typename IdxT      = uint32_t,
          typename DistanceT = float>
void search_main(raft::resources const& res,
                 search_params params,
                 const index<T, IdxT>& index,
                 raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
                 raft::device_matrix_view<OutputIdxT, int64_t, raft::row_major> neighbors,
                 raft::device_matrix_view<DistanceT, int64_t, raft::row_major> distances,
                 CagraSampleFilterT sample_filter = CagraSampleFilterT())
{
  RAFT_EXPECTS(!index.dataset_fd().has_value(),
               "Cannot search a CAGRA index that is stored on disk. "
               "Use cuvs::neighbors::hnsw::from_cagra() to convert the index and "
               "cuvs::neighbors::hnsw::deserialize() to load it into memory before searching.");

  // n_rows has the same type as the dataset index (the array extents type)
  using ds_idx_type    = decltype(index.data().n_rows());
  using graph_idx_type = uint32_t;
  // Dispatch search parameters based on the dataset kind.
  if (auto* strided_dset = dynamic_cast<const strided_dataset<T, ds_idx_type>*>(&index.data());
      strided_dset != nullptr) {
    // Search using a plain (strided) row-major dataset
    RAFT_EXPECTS(index.metric() != cuvs::distance::DistanceType::CosineExpanded ||
                   index.dataset_norms().has_value(),
                 "Dataset norms must be provided for CosineExpanded metric");

    const float* dataset_norms_ptr = nullptr;
    if (index.metric() == cuvs::distance::DistanceType::CosineExpanded) {
      dataset_norms_ptr = index.dataset_norms().value().data_handle();
    }
    auto desc = dataset_descriptor_init_with_cache<T, graph_idx_type, DistanceT>(
      res, params, *strided_dset, index.metric(), dataset_norms_ptr);
    search_main_core<T, graph_idx_type, DistanceT, CagraSampleFilterT, IdxT, OutputIdxT>(
      res,
      params,
      desc,
      index.graph(),
      index.source_indices(),
      queries,
      neighbors,
      distances,
      sample_filter);
  } else if (auto* vpq_dset = dynamic_cast<
               const cuvs::preprocessing::quantize::pq::vpq_dataset<float, ds_idx_type>*>(
               &index.data());
             vpq_dset != nullptr) {
    // Search using a compressed dataset
    RAFT_FAIL("FP32 VPQ dataset support is coming soon");
  } else if (auto* vpq_dset = dynamic_cast<
               const cuvs::preprocessing::quantize::pq::vpq_dataset<half, ds_idx_type>*>(
               &index.data());
             vpq_dset != nullptr) {
    auto desc = dataset_descriptor_init_with_cache<T, graph_idx_type, DistanceT>(
      res, params, *vpq_dset, index.metric(), nullptr);
    search_main_core<T, graph_idx_type, DistanceT, CagraSampleFilterT, IdxT, OutputIdxT>(
      res,
      params,
      desc,
      index.graph(),
      index.source_indices(),
      queries,
      neighbors,
      distances,
      sample_filter);
  } else if (auto* empty_dset = dynamic_cast<const empty_dataset<ds_idx_type>*>(&index.data());
             empty_dset != nullptr) {
    // Forgot to add a dataset.
    RAFT_FAIL(
      "Attempted to search without a dataset. Please call index.update_dataset(...) first.");
  } else {
    // This is a logic error.
    RAFT_FAIL("Unrecognized dataset format");
  }

  static_assert(std::is_same_v<DistanceT, float>,
                "only float distances are supported at the moment");
  float* dist_out          = distances.data_handle();
  const DistanceT* dist_in = distances.data_handle();
  // We're converting the data from T to DistanceT during distance computation
  // and divide the values by kDivisor. Here we restore the original scale.
  constexpr float kScale = cuvs::spatial::knn::detail::utils::config<T>::kDivisor /
                           cuvs::spatial::knn::detail::utils::config<DistanceT>::kDivisor;

  if (index.metric() == cuvs::distance::DistanceType::CosineExpanded) {
    auto stream      = raft::resource::get_cuda_stream(res);
    auto query_norms = raft::make_device_vector<DistanceT, int64_t>(res, queries.extent(0));

    // first scale the queries and then compute norms
    auto scaled_sq_op = raft::compose_op(
      raft::sq_op{}, raft::div_const_op<DistanceT>{DistanceT(kScale)}, raft::cast_op<DistanceT>());
    raft::linalg::reduce<raft::Apply::ALONG_ROWS>(
      res,
      raft::make_device_matrix_view<const T, int64_t, raft::row_major>(
        queries.data_handle(), queries.extent(0), queries.extent(1)),
      query_norms.view(),
      (DistanceT)0,
      false,
      scaled_sq_op,
      raft::add_op(),
      raft::sqrt_op{});

    const auto n_queries = distances.extent(0);
    const auto k         = distances.extent(1);
    auto query_norms_ptr = query_norms.data_handle();

    raft::linalg::matrix_vector_op<raft::Apply::ALONG_COLUMNS>(
      res,
      raft::make_const_mdspan(distances),
      raft::make_const_mdspan(query_norms.view()),
      distances,
      raft::compose_op(raft::add_const_op<DistanceT>{DistanceT(1)}, raft::div_checkzero_op{}));
  } else {
    cuvs::neighbors::ivf::detail::postprocess_distances(res,
                                                        dist_out,
                                                        dist_in,
                                                        index.metric(),
                                                        distances.extent(0),
                                                        distances.extent(1),
                                                        kScale,
                                                        true);
  }
}
/** @} */  // end group cagra

}  // namespace cuvs::neighbors::cagra::detail
