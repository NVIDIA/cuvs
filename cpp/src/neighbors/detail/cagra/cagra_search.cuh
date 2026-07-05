/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../../core/nvtx.hpp"
#include "factory.cuh"
#include "sample_filter_utils.cuh"
#include "search_multi_cta.cuh"
#include "search_plan.cuh"
#include "search_single_cta.cuh"

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

// TODO: This shouldn't be calling spatial/knn apis
#include "../ann_utils.cuh"

#include <raft/linalg/map.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/matrix/select_k.cuh>

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
  } else if (auto* vpq_dset = dynamic_cast<const vpq_dataset<float, ds_idx_type>*>(&index.data());
             vpq_dset != nullptr) {
    // Search using a compressed dataset
    RAFT_FAIL("FP32 VPQ dataset support is coming soon");
  } else if (auto* vpq_dset = dynamic_cast<const vpq_dataset<half, ds_idx_type>*>(&index.data());
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

/**
 * @brief Search all partitions concurrently and return the global top-k per query.
 *
 * For each query row in @p queries, the kernel searches all partitions in parallel
 * (blockIdx.z = partition_id, blockIdx.y = query_id) into an internal intermediate buffer.
 * Per-partition distance post-processing is applied, then a batched select_k merges across
 * partitions and a small decode pass writes the final outputs.
 *
 * @param indices         CAGRA index objects, one per partition (strided datasets only)
 * @param queries         queries matrix [n_queries, dim]; searched against every partition
 * @param partition_ids   output: which partition each neighbor came from, shape [n_queries, k]
 * @param neighbors       output: ordinal in partition[i]'s dataset, shape [n_queries, k]
 * @param distances       output: post-processed distance, shape [n_queries, k]
 */
template <typename T,
          typename OutputIdxT         = uint32_t,
          typename IdxT               = uint32_t,
          typename DistanceT          = float,
          typename CagraSampleFilterT = cuvs::neighbors::filtering::none_sample_filter>
void search_multi_partition(
  raft::resources const& res,
  search_params params,
  const std::vector<const index<T, IdxT>*>& indices,
  raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> partition_ids,
  raft::device_matrix_view<OutputIdxT, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<DistanceT, int64_t, raft::row_major> distances,
  CagraSampleFilterT sample_filter = CagraSampleFilterT{})
{
  static_assert(std::is_same_v<IdxT, uint32_t>, "Only uint32_t graph index type is supported");
  static_assert(std::is_same_v<DistanceT, float>, "Only float distances are supported");

  const uint32_t num_partitions = static_cast<uint32_t>(indices.size());

  const uint32_t n_queries = static_cast<uint32_t>(queries.extent(0));
  const int64_t dim        = queries.extent(1);
  const uint32_t topk      = static_cast<uint32_t>(neighbors.extent(1));

  // All partitions must share one metric and one graph degree. The shared plan descriptor (and its
  // shared-memory / layout sizing) and the cross-partition select_k direction are all derived from
  // indices[0], so a differing metric would be merged in the wrong order and a differing graph
  // degree would be sized incorrectly. Dataset sizes may still differ (e.g. skewed splits).
  const cuvs::distance::DistanceType metric = indices[0]->metric();
  const int64_t graph_degree                = indices[0]->graph().extent(1);

  int64_t max_dataset_size = 0;
  for (uint32_t i = 0; i < num_partitions; i++) {
    RAFT_EXPECTS(!indices[i]->dataset_fd().has_value(),
                 "Disk-based datasets are not supported for multi-partition search");
    RAFT_EXPECTS(indices[i]->metric() == metric,
                 "All partitions must use the same distance metric for multi-partition search");
    RAFT_EXPECTS(indices[i]->graph().extent(1) == graph_degree,
                 "All partitions must use the same graph degree for multi-partition search");
    max_dataset_size = std::max(max_dataset_size, indices[i]->data().n_rows());
  }

  // Query norms are needed only for CosineExpanded (uniform across partitions, checked above).
  const bool needs_query_norms = metric == cuvs::distance::DistanceType::CosineExpanded;

  if (params.max_queries == 0) {
    cudaDeviceProp deviceProp = raft::resource::get_device_properties(res);
    params.max_queries =
      std::min<size_t>(static_cast<size_t>(n_queries), deviceProp.maxGridSize[1]);
  }

  // Persistent kernels are not used in multi-partition search regardless of which algo runs.
  params.persistent = false;

  // MULTI_KERNEL is a reference implementation and is substantially slower than SINGLE_CTA /
  // MULTI_CTA in practice; multi-partition deliberately does not route to it.
  if (params.algo == search_algo::MULTI_KERNEL) {
    RAFT_FAIL("MULTI_KERNEL is not supported for multi-partition search");
  }

  // AUTO resolution. Mirrors single-partition's heuristic in search_plan_impl_base, with the
  // occupancy gate scaled by num_partitions (multi-partition grids already have a partition
  // axis, so each query produces num_partitions CTAs on SINGLE_CTA). SINGLE_CTA's
  // itopk_size <= 512 hard cap is enforced in its plan constructor (search_single_cta.cuh);
  // above that, AUTO must route to MULTI_CTA. Below the cap, SINGLE_CTA wins only if there
  // are enough (query, partition) CTAs to fill the GPU; otherwise MULTI_CTA's
  // ceildiv(itopk_size, 32) CTAs per query recover occupancy.
  if (params.algo == search_algo::AUTO) {
    const size_t num_sm = raft::getMultiProcessorCount();
    if (params.itopk_size <= 512 &&
        static_cast<size_t>(params.max_queries) * num_partitions >= num_sm * 2lu) {
      params.algo = search_algo::SINGLE_CTA;
    } else {
      params.algo = search_algo::MULTI_CTA;
    }
  }

  // Build a single plan_desc for the (uniform) graph_degree. The smem layout in the descriptor is
  // type-dependent only, so any partition's descriptor (we pick indices[0]) is representative for
  // the plan's smem/sizing calculations.
  using graph_idx_type = uint32_t;
  auto* strided_dset0  = dynamic_cast<const strided_dataset<T, int64_t>*>(&indices[0]->data());
  RAFT_EXPECTS(strided_dset0 != nullptr,
               "Multi-partition search only supports strided (non-compressed) datasets");

  RAFT_EXPECTS(metric != cuvs::distance::DistanceType::CosineExpanded ||
                 indices[0]->dataset_norms().has_value(),
               "Dataset norms must be provided for CosineExpanded metric");
  const float* dataset_norms_ptr0 = nullptr;
  if (metric == cuvs::distance::DistanceType::CosineExpanded) {
    dataset_norms_ptr0 = indices[0]->dataset_norms().value().data_handle();
  }
  auto plan_desc = dataset_descriptor_init_with_cache<T, graph_idx_type, DistanceT>(
    res, params, *strided_dset0, metric, dataset_norms_ptr0);

  cudaStream_t stream = raft::resource::get_cuda_stream(res);

  // Cap the per-launch query count. num_queries maps to grid.y in the multi-partition kernels,
  // which is bounded by maxGridSize[1]; chunking also bounds the intermediate workspaces, which
  // scale with (queries * num_partitions * per_partition_topk). Mirrors the single-partition path.
  const uint32_t max_queries = params.max_queries;

  using CagraSampleFilterT_s = typename CagraSampleFilterT_Selector<CagraSampleFilterT>::type;

  // Word-aligned (64-bit) prefix sum of per-partition sizes: the starting bit offset of partition i
  // in the combined filter bitset. Recomputed here (rather than transported) and must match the
  // host-side packing; consumed by both algo branches via multi_partition_desc_t::bit_offset.
  std::vector<int64_t> partition_bit_offsets(num_partitions);
  int64_t total_filter_bits = 0;
  for (uint32_t i = 0; i < num_partitions; i++) {
    partition_bit_offsets[i] = total_filter_bits;
    total_filter_bits += ((indices[i]->data().n_rows() + 63) / 64) * 64;
  }
  // A bitset filter must be large enough to cover every partition's slice (catches a mismatch
  // between the caller's packed bitset and the index sizes being searched).
  if constexpr (requires { sample_filter.view(); }) {
    RAFT_EXPECTS(
      total_filter_bits <= static_cast<int64_t>(sample_filter.view().size()),
      "Combined filter bitset (%ld bits) is too small for the partition sizes (%ld bits)",
      static_cast<int64_t>(sample_filter.view().size()),
      total_filter_bits);
  }

  constexpr float kScale = cuvs::spatial::knn::detail::utils::config<T>::kDivisor /
                           cuvs::spatial::knn::detail::utils::config<DistanceT>::kDivisor;

  // Number of candidates each partition contributes to the cross-partition merge below.
  // SINGLE_CTA's kernel produces exactly `topk` per partition; MULTI_CTA's kernel emits
  // `num_cta_per_query * itopk_size` per partition (no per-partition merge — rely on the
  // cross-partition select_k below to pick the final global top-k). Set once the plan is built.
  uint32_t per_partition_topk = 0;

  // Workspaces reused across chunks; sized to one max_queries-sized chunk once per_partition_topk
  // is known. The intermediate buffers are laid out [num_partitions, chunk_queries,
  // per_partition_topk]; transposed_distances and positions_buf hold one chunk's merge scratch.
  lightweight_uvector<graph_idx_type> intermediate_neighbors(res);
  lightweight_uvector<DistanceT> intermediate_distances(res);
  lightweight_uvector<DistanceT> transposed_distances(res);
  lightweight_uvector<uint32_t> positions_buf(res);
  auto query_norms = raft::make_device_vector<DistanceT, int64_t>(res, max_queries);

  // Merge stage shared by both algos, run once per query chunk [qid, qid + chunk_queries):
  // post-process each partition's per-chunk distance slice, transpose into
  // [chunk_queries, num_partitions * per_partition_topk], run a batched select_k for the global
  // top-k per query, and decode positions into partition_ids / neighbors. Writes the chunk's slice
  // of the caller outputs; the intermediate/scratch buffers are chunk-local (start at offset 0).
  auto merge_chunk = [&](uint32_t qid, uint32_t chunk_queries) {
    const size_t chunk_partition_stride = static_cast<size_t>(chunk_queries) * per_partition_topk;

    // Query norms, needed only by CosineExpanded. Each chunk covers a disjoint set of queries, so
    // its norms are computed over just this chunk's rows (each query's norm computed once overall).
    auto chunk_query_norms =
      raft::make_device_vector_view<DistanceT, int64_t>(query_norms.data_handle(), chunk_queries);
    if (needs_query_norms) {
      auto scaled_sq_op = raft::compose_op(raft::sq_op{},
                                           raft::div_const_op<DistanceT>{DistanceT(kScale)},
                                           raft::cast_op<DistanceT>());
      raft::linalg::reduce<raft::Apply::ALONG_ROWS>(
        res,
        raft::make_device_matrix_view<const T, int64_t, raft::row_major>(
          queries.data_handle() + static_cast<size_t>(qid) * dim, chunk_queries, dim),
        chunk_query_norms,
        (DistanceT)0,
        false,
        scaled_sq_op,
        raft::add_op(),
        raft::sqrt_op{});
    }

    // Per-partition distance post-processing (scale + metric transform). Each partition's slice in
    // intermediate_distances has shape [chunk_queries, per_partition_topk], contiguous row-major.
    for (uint32_t i = 0; i < num_partitions; i++) {
      DistanceT* slice_ptr =
        intermediate_distances.data() + static_cast<size_t>(i) * chunk_partition_stride;
      if (indices[i]->metric() == cuvs::distance::DistanceType::CosineExpanded) {
        auto slice_view = raft::make_device_matrix_view<DistanceT, int64_t, raft::row_major>(
          slice_ptr, chunk_queries, per_partition_topk);
        raft::linalg::matrix_vector_op<raft::Apply::ALONG_COLUMNS>(
          res,
          raft::make_const_mdspan(slice_view),
          raft::make_const_mdspan(chunk_query_norms),
          slice_view,
          raft::compose_op(raft::add_const_op<DistanceT>{DistanceT(1)}, raft::div_checkzero_op{}));
      } else {
        cuvs::neighbors::ivf::detail::postprocess_distances(res,
                                                            slice_ptr,
                                                            slice_ptr,
                                                            indices[i]->metric(),
                                                            chunk_queries,
                                                            per_partition_topk,
                                                            kScale,
                                                            true);
      }
    }

    // Transpose intermediate_distances from [num_partitions, chunk_queries, per_partition_topk] to
    // [chunk_queries, num_partitions * per_partition_topk] so batched select_k can pick the global
    // top-k per query. (raft::matrix::select_k requires row-major contiguous input.)
    {
      const DistanceT* src     = intermediate_distances.data();
      const int64_t row_stride = static_cast<int64_t>(num_partitions) * per_partition_topk;
      const int64_t partition_stride_i64   = static_cast<int64_t>(chunk_partition_stride);
      const int64_t per_partition_topk_i64 = per_partition_topk;
      auto transposed_view = raft::make_device_matrix_view<DistanceT, int64_t, raft::row_major>(
        transposed_distances.data(), static_cast<int64_t>(chunk_queries), row_stride);
      raft::linalg::map_offset(
        res,
        transposed_view,
        [src, row_stride, partition_stride_i64, per_partition_topk_i64] __device__(int64_t idx) {
          const int64_t q   = idx / row_stride;
          const int64_t rem = idx % row_stride;
          const int64_t p   = rem / per_partition_topk_i64;
          const int64_t j   = rem % per_partition_topk_i64;
          return src[p * partition_stride_i64 + q * per_partition_topk_i64 + j];
        });
    }

    // Batched select_k: for each query row, find the global top-k across all partition slots.
    // Writes the chunk's `distances` rows directly; positions in
    // [0, num_partitions * per_partition_topk) go to positions_buf for decoding below.
    auto positions_view = raft::make_device_matrix_view<uint32_t, int64_t, raft::row_major>(
      positions_buf.data(), chunk_queries, topk);
    auto distances_slice = raft::make_device_matrix_view<DistanceT, int64_t, raft::row_major>(
      distances.data_handle() + static_cast<size_t>(qid) * topk, chunk_queries, topk);

    // Post-processing above restores each metric's natural ordering: distance metrics (L2, and the
    // Cosine transform applied here) are smaller-is-closer, while InnerProduct is larger-is-closer.
    const bool select_min = cuvs::distance::is_min_close(metric);

    raft::matrix::select_k<DistanceT, uint32_t>(
      res,
      raft::make_device_matrix_view<const DistanceT, int64_t, raft::row_major>(
        transposed_distances.data(),
        static_cast<int64_t>(chunk_queries),
        static_cast<int64_t>(num_partitions) * per_partition_topk),
      std::nullopt,
      distances_slice,
      positions_view,
      select_min);

    // Decode positions into partition_ids and neighbors (chunk slices):
    //   partition_ids[q, j] = pos / per_partition_topk
    //   neighbors[q, j]     = intermediate_neighbors[(pos / per_partition_topk) * partition_stride
    //                           + q * per_partition_topk + (pos % per_partition_topk)]
    // The output buffers have stride `topk`; the intermediate buffer has per-partition stride
    // `per_partition_topk`. The two differ when the kernel emits more than `topk` candidates per
    // partition (e.g. MULTI_CTA mp).
    auto partition_ids_slice = raft::make_device_matrix_view<uint32_t, int64_t, raft::row_major>(
      partition_ids.data_handle() + static_cast<size_t>(qid) * topk, chunk_queries, topk);
    {
      const uint32_t per_partition_topk_u32 = per_partition_topk;
      raft::linalg::map(
        res,
        partition_ids_slice,
        [per_partition_topk_u32] __device__(uint32_t pos) { return pos / per_partition_topk_u32; },
        raft::make_const_mdspan(positions_view));
    }
    auto neighbors_slice = raft::make_device_matrix_view<OutputIdxT, int64_t, raft::row_major>(
      neighbors.data_handle() + static_cast<size_t>(qid) * topk, chunk_queries, topk);
    {
      const graph_idx_type* intermediate_neighbors_ptr = intermediate_neighbors.data();
      const uint32_t* positions_ptr                    = positions_buf.data();
      const int64_t partition_stride_i64   = static_cast<int64_t>(chunk_partition_stride);
      const int64_t per_partition_topk_i64 = per_partition_topk;
      const int64_t topk_i64               = topk;
      raft::linalg::map_offset(
        res,
        neighbors_slice,
        [intermediate_neighbors_ptr,
         positions_ptr,
         partition_stride_i64,
         per_partition_topk_i64,
         topk_i64] __device__(int64_t idx) {
          const int64_t q     = idx / topk_i64;
          const int64_t j_out = idx % topk_i64;
          const uint32_t pos  = positions_ptr[q * topk_i64 + j_out];
          const int64_t p     = pos / static_cast<uint32_t>(per_partition_topk_i64);
          const int64_t j_in  = pos % static_cast<uint32_t>(per_partition_topk_i64);
          return static_cast<OutputIdxT>(
            intermediate_neighbors_ptr[p * partition_stride_i64 + q * per_partition_topk_i64 +
                                       j_in]);
        });
    }
  };

  if (params.algo == search_algo::SINGLE_CTA) {
    single_cta_search::
      search<T, graph_idx_type, DistanceT, CagraSampleFilterT_s, graph_idx_type, graph_idx_type>
        plan(res, params, plan_desc, dim, max_dataset_size, graph_degree, topk);

    RAFT_EXPECTS(topk <= plan.itopk_size,
                 "topk = %u must be smaller than itopk_size = %lu",
                 topk,
                 plan.itopk_size);

    per_partition_topk = topk;
    const size_t chunk_capacity =
      static_cast<size_t>(num_partitions) * max_queries * per_partition_topk;
    intermediate_neighbors.resize(chunk_capacity, stream);
    intermediate_distances.resize(chunk_capacity, stream);
    transposed_distances.resize(chunk_capacity, stream);
    positions_buf.resize(static_cast<size_t>(max_queries) * topk, stream);

    // Build per-partition descriptors on the host (partition-dependent and query-independent, so
    // built once and reused across chunks). Queries and result buffers are passed to the kernel
    // as separate per-chunk parameters.
    using part_desc_t = single_cta_search::multi_partition_desc_t<T, graph_idx_type, DistanceT>;
    std::vector<part_desc_t> host_part_descs(num_partitions);

    // Collect per-partition dataset descriptors (may trigger lazy device init on `stream`).
    std::vector<dataset_descriptor_host<T, graph_idx_type, DistanceT>> part_dataset_descs;
    part_dataset_descs.reserve(num_partitions);

    for (uint32_t i = 0; i < num_partitions; i++) {
      auto* strided_dset = dynamic_cast<const strided_dataset<T, int64_t>*>(&indices[i]->data());
      RAFT_EXPECTS(strided_dset != nullptr,
                   "All partitions must have strided (non-compressed) datasets");
      const float* norms_ptr = nullptr;
      if (indices[i]->metric() == cuvs::distance::DistanceType::CosineExpanded) {
        RAFT_EXPECTS(indices[i]->dataset_norms().has_value(),
                     "Dataset norms required for CosineExpanded metric (partition %u)",
                     i);
        norms_ptr = indices[i]->dataset_norms().value().data_handle();
      }
      part_dataset_descs.push_back(dataset_descriptor_init_with_cache<T, graph_idx_type, DistanceT>(
        res, params, *strided_dset, indices[i]->metric(), norms_ptr));

      host_part_descs[i].dataset_desc = part_dataset_descs.back().dev_ptr(stream);
      host_part_descs[i].graph        = indices[i]->graph().data_handle();
      host_part_descs[i].graph_degree = static_cast<uint32_t>(indices[i]->graph().extent(1));
      host_part_descs[i].bit_offset   = partition_bit_offsets[i];
    }

    lightweight_uvector<part_desc_t> dev_part_descs_buf(res);
    dev_part_descs_buf.resize(num_partitions, stream);
    RAFT_CUDA_TRY(cudaMemcpyAsync(dev_part_descs_buf.data(),
                                  host_part_descs.data(),
                                  num_partitions * sizeof(part_desc_t),
                                  cudaMemcpyHostToDevice,
                                  stream));

    for (uint32_t qid = 0; qid < n_queries; qid += max_queries) {
      const uint32_t chunk_queries = std::min<uint32_t>(max_queries, n_queries - qid);
      plan.run_multi_partition(res,
                               dev_part_descs_buf.data(),
                               num_partitions,
                               queries.data_handle() + static_cast<size_t>(qid) * dim,
                               chunk_queries,
                               intermediate_neighbors.data(),
                               intermediate_distances.data(),
                               per_partition_topk,
                               set_offset(sample_filter, qid));
      merge_chunk(qid, chunk_queries);
    }
  } else /* MULTI_CTA */ {
    multi_cta_search::
      search<T, graph_idx_type, DistanceT, CagraSampleFilterT_s, graph_idx_type, graph_idx_type>
        plan(res, params, plan_desc, dim, max_dataset_size, graph_degree, topk);

    // MULTI_CTA splits the global itopk pool across num_cta_per_query CTAs of 32 candidates
    // each. The kernel emits all num_cta_per_query * itopk_size candidates per (query,
    // partition) and lets the cross-partition select_k pick the final global top-k.
    per_partition_topk =
      static_cast<uint32_t>(plan.num_cta_per_query) * static_cast<uint32_t>(plan.itopk_size);
    const size_t chunk_capacity =
      static_cast<size_t>(num_partitions) * max_queries * per_partition_topk;
    intermediate_neighbors.resize(chunk_capacity, stream);
    intermediate_distances.resize(chunk_capacity, stream);
    transposed_distances.resize(chunk_capacity, stream);
    positions_buf.resize(static_cast<size_t>(max_queries) * topk, stream);

    using part_desc_t = multi_cta_search::multi_partition_desc_t<T, graph_idx_type, DistanceT>;
    std::vector<part_desc_t> host_part_descs(num_partitions);

    std::vector<dataset_descriptor_host<T, graph_idx_type, DistanceT>> part_dataset_descs;
    part_dataset_descs.reserve(num_partitions);

    for (uint32_t i = 0; i < num_partitions; i++) {
      auto* strided_dset = dynamic_cast<const strided_dataset<T, int64_t>*>(&indices[i]->data());
      RAFT_EXPECTS(strided_dset != nullptr,
                   "All partitions must have strided (non-compressed) datasets");
      const float* norms_ptr = nullptr;
      if (indices[i]->metric() == cuvs::distance::DistanceType::CosineExpanded) {
        RAFT_EXPECTS(indices[i]->dataset_norms().has_value(),
                     "Dataset norms required for CosineExpanded metric (partition %u)",
                     i);
        norms_ptr = indices[i]->dataset_norms().value().data_handle();
      }
      part_dataset_descs.push_back(dataset_descriptor_init_with_cache<T, graph_idx_type, DistanceT>(
        res, params, *strided_dset, indices[i]->metric(), norms_ptr));

      host_part_descs[i].dataset_desc = part_dataset_descs.back().dev_ptr(stream);
      host_part_descs[i].graph        = indices[i]->graph().data_handle();
      host_part_descs[i].graph_degree = static_cast<uint32_t>(indices[i]->graph().extent(1));
      host_part_descs[i].bit_offset   = partition_bit_offsets[i];
    }

    lightweight_uvector<part_desc_t> dev_part_descs_buf(res);
    dev_part_descs_buf.resize(num_partitions, stream);
    RAFT_CUDA_TRY(cudaMemcpyAsync(dev_part_descs_buf.data(),
                                  host_part_descs.data(),
                                  num_partitions * sizeof(part_desc_t),
                                  cudaMemcpyHostToDevice,
                                  stream));

    for (uint32_t qid = 0; qid < n_queries; qid += max_queries) {
      const uint32_t chunk_queries = std::min<uint32_t>(max_queries, n_queries - qid);
      plan.run_multi_partition(res,
                               dev_part_descs_buf.data(),
                               num_partitions,
                               static_cast<uint32_t>(graph_degree),
                               queries.data_handle() + static_cast<size_t>(qid) * dim,
                               chunk_queries,
                               intermediate_neighbors.data(),
                               intermediate_distances.data(),
                               set_offset(sample_filter, qid));
      merge_chunk(qid, chunk_queries);
    }
  }
}

}  // namespace cuvs::neighbors::cagra::detail
