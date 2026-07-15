/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/distance/distance.hpp>
#include <raft/core/copy.cuh>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/linalg/map.cuh>
#include <raft/matrix/detail/select_warpsort.cuh>  // matrix::detail::select::warpsort::warp_sort_distributed

namespace cuvs::neighbors::ivf::detail {

// Forward declaration of helper function to avoid including cub/device/* in header
void sort_cluster_sizes_descending(uint32_t* input,
                                   uint32_t* output,
                                   uint32_t n_lists,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref tmp_res);

/**
 * Default value returned by `search` when the `n_probes` is too small and top-k is too large.
 * One may encounter it if the combined size of probed clusters is smaller than the requested
 * number of results per query.
 */
template <typename IdxT>
constexpr static IdxT kOutOfBoundsRecord = std::numeric_limits<IdxT>::max();

template <typename T, typename IdxT, bool Ascending = true>
struct dummy_block_sort_t {
  using queue_t = raft::matrix::detail::select::warpsort::
    warp_sort_distributed<raft::WarpSize, Ascending, T, IdxT>;
  template <typename... Args>
  __device__ dummy_block_sort_t(int k, Args...) {};
};

/**
 * Struct to configure and launch calc_chunk_indices_kernel.
 *
 * Both configure() and operator() are defined in ivf_common.cu to comply
 * with CUDA whole compilation rules - the kernel pointer must be obtained
 * and used within the same translation unit. See
 * https://developer.nvidia.com/blog/cuda-c-compiler-updates-impacting-elf-visibility-and-linkage/
 */
struct calc_chunk_indices {
 public:
  struct configured {
    dim3 block_dim;
    dim3 grid_dim;
    uint32_t n_probes;

    void operator()(const uint32_t* cluster_sizes,
                    const uint32_t* clusters_to_probe,
                    uint32_t* chunk_indices,
                    uint32_t* n_samples,
                    rmm::cuda_stream_view stream);
  };

  static inline auto configure(uint32_t n_probes, uint32_t n_queries) -> configured
  {
    return try_block_dim<1024>(n_probes, n_queries);
  }

 private:
  template <int BlockDim>
  static auto try_block_dim(uint32_t n_probes, uint32_t n_queries) -> configured
  {
    if constexpr (BlockDim >= raft::WarpSize * 2) {
      if (BlockDim >= n_probes * 2) { return try_block_dim<(BlockDim / 2)>(n_probes, n_queries); }
    }
    return {dim3(BlockDim, 1, 1), dim3(n_queries, 1, 1), n_probes};
  }
};

/**
 * Look up the chunk id corresponding to the sample index.
 *
 * Each query vector was compared to all the vectors from n_probes clusters, and sample_ix is an
 * ordered number of one of such vectors. This function looks up to which chunk it belongs,
 * and returns the index within the chunk (which is also an index within a cluster).
 *
 * @param[inout] sample_ix
 *   input: the offset of the sample in the batch;
 *   output: the offset inside the chunk (probe) / selected cluster.
 * @param[in] n_probes number of probes
 * @param[in] chunk_indices offsets of the chunks within the batch [n_probes]
 * @return chunk index (== n_probes when the input index is not in the valid range,
 *    which can happen if there is not enough data to output in the selected clusters).
 */
__device__ inline auto find_chunk_ix(uint32_t& sample_ix,  // NOLINT
                                     uint32_t n_probes,
                                     const uint32_t* chunk_indices) -> uint32_t
{
  uint32_t ix_min = 0;
  uint32_t ix_max = n_probes;
  do {
    uint32_t i = (ix_min + ix_max) / 2;
    if (chunk_indices[i] <= sample_ix) {
      ix_min = i + 1;
    } else {
      ix_max = i;
    }
  } while (ix_min < ix_max);
  if (ix_min > 0) { sample_ix -= chunk_indices[ix_min - 1]; }
  return ix_min;
}

template <int BlockDim, typename IdxT, typename DbIdxT>
__launch_bounds__(BlockDim) RAFT_KERNEL
  postprocess_neighbors_kernel(IdxT* neighbors_out,                // [n_queries, topk]
                               const uint32_t* neighbors_in,       // [n_queries, topk]
                               const DbIdxT* const* db_indices,    // [n_clusters][..]
                               const uint32_t* clusters_to_probe,  // [n_queries, n_probes]
                               const uint32_t* chunk_indices,      // [n_queries, n_probes]
                               uint32_t n_queries,
                               uint32_t n_probes,
                               uint32_t topk)
{
  static_assert(!raft::is_narrowing_v<uint32_t, IdxT>,
                "IdxT must be able to represent all values of uint32_t");
  const uint64_t i        = threadIdx.x + BlockDim * uint64_t(blockIdx.x);
  const uint32_t query_ix = i / uint64_t(topk);
  if (query_ix >= n_queries) { return; }
  const uint32_t k = i % uint64_t(topk);
  neighbors_in += query_ix * topk;
  neighbors_out += query_ix * topk;
  chunk_indices += query_ix * n_probes;
  clusters_to_probe += query_ix * n_probes;
  uint32_t data_ix        = neighbors_in[k];
  const uint32_t chunk_ix = find_chunk_ix(data_ix, n_probes, chunk_indices);
  const bool valid        = chunk_ix < n_probes;
  neighbors_out[k] = valid ? static_cast<IdxT>(db_indices[clusters_to_probe[chunk_ix]][data_ix])
                           : kOutOfBoundsRecord<IdxT>;
}

/**
 * Transform found sample indices into the corresponding database indices
 * (as stored in index.indices()).
 * The sample indices are the record indices as they appear in the database view formed by the
 * probed clusters / defined by the `chunk_indices`.
 * We assume the searched sample sizes (for a single query) fit into `uint32_t`.
 */
template <typename IdxT, typename DbIdxT>
void postprocess_neighbors(IdxT* neighbors_out,                // [n_queries, topk]
                           const uint32_t* neighbors_in,       // [n_queries, topk]
                           const DbIdxT* const* db_indices,    // [n_clusters][..]
                           const uint32_t* clusters_to_probe,  // [n_queries, n_probes]
                           const uint32_t* chunk_indices,      // [n_queries, n_probes]
                           uint32_t n_queries,
                           uint32_t n_probes,
                           uint32_t topk,
                           rmm::cuda_stream_view stream)
{
  constexpr int kPNThreads = 256;
  const int pn_blocks      = raft::div_rounding_up_unsafe<size_t>(n_queries * topk, kPNThreads);
  postprocess_neighbors_kernel<kPNThreads, IdxT>
    <<<pn_blocks, kPNThreads, 0, stream>>>(neighbors_out,
                                           neighbors_in,
                                           db_indices,
                                           clusters_to_probe,
                                           chunk_indices,
                                           n_queries,
                                           n_probes,
                                           topk);
}

/**
 * Post-process the scores depending on the metric type;
 * translate the element type if necessary.
 */
template <typename ScoreInT, typename ScoreOutT = float>
void postprocess_distances(const raft::resources& res,
                           ScoreOutT* out,      // [n_queries, topk]
                           const ScoreInT* in,  // [n_queries, topk]
                           distance::DistanceType metric,
                           uint32_t n_queries,
                           uint32_t topk,
                           float scaling_factor,
                           bool account_for_max_close)
{
  constexpr bool needs_cast = !std::is_same<ScoreInT, ScoreOutT>::value;
  const bool needs_copy     = ((void*)in) != ((void*)out);
  size_t len                = size_t(n_queries) * size_t(topk);
  auto out_view             = raft::make_device_vector_view<ScoreOutT, size_t>(out, len);
  auto in_view              = raft::make_device_vector_view<const ScoreInT, size_t>(in, len);
  switch (metric) {
    case distance::DistanceType::L2Unexpanded:
    case distance::DistanceType::L2Expanded: {
      if (scaling_factor != 1.0) {
        raft::linalg::map(
          res,
          out_view,
          raft::compose_op(raft::mul_const_op<ScoreOutT>{scaling_factor * scaling_factor},
                           raft::cast_op<ScoreOutT>{}),
          raft::make_const_mdspan(in_view));
      } else if (needs_cast || needs_copy) {
        raft::linalg::map(
          res, out_view, raft::cast_op<ScoreOutT>{}, raft::make_const_mdspan(in_view));
      }
    } break;
    case distance::DistanceType::L2SqrtUnexpanded:
    case distance::DistanceType::L2SqrtExpanded: {
      if (scaling_factor != 1.0) {
        raft::linalg::map(res,
                          out_view,
                          raft::compose_op{raft::mul_const_op<ScoreOutT>{scaling_factor},
                                           raft::sqrt_op{},
                                           raft::cast_op<ScoreOutT>{}},
                          raft::make_const_mdspan(in_view));
      } else if (needs_cast) {
        raft::linalg::map(res,
                          out_view,
                          raft::compose_op{raft::sqrt_op{}, raft::cast_op<ScoreOutT>{}},
                          raft::make_const_mdspan(in_view));
      } else {
        raft::linalg::map(res, out_view, raft::sqrt_op{}, raft::make_const_mdspan(in_view));
      }
    } break;
    case distance::DistanceType::CosineExpanded:
    case distance::DistanceType::InnerProduct: {
      float factor = (account_for_max_close ? -1.0 : 1.0) * scaling_factor * scaling_factor;
      if (factor != 1.0) {
        raft::linalg::map(
          res,
          out_view,
          raft::compose_op(raft::mul_const_op<ScoreOutT>{factor}, raft::cast_op<ScoreOutT>{}),
          raft::make_const_mdspan(in_view));
      } else if (needs_cast || needs_copy) {
        raft::linalg::map(
          res, out_view, raft::cast_op<ScoreOutT>{}, raft::make_const_mdspan(in_view));
      }
    } break;
    case distance::DistanceType::BitwiseHamming: break;
    case distance::DistanceType::L1: {
      if (scaling_factor != 1.0) {
        raft::linalg::map(res,
                          out_view,
                          raft::compose_op(raft::mul_const_op<ScoreOutT>{scaling_factor},
                                           raft::cast_op<ScoreOutT>{}),
                          raft::make_const_mdspan(in_view));
      } else if (needs_cast || needs_copy) {
        raft::linalg::map(
          res, out_view, raft::cast_op<ScoreOutT>{}, raft::make_const_mdspan(in_view));
      }
    } break;
    default: RAFT_FAIL("Unexpected metric.");
  }
}

/** Update the state of the dependent index members. */
template <typename Index>
void recompute_internal_state(const raft::resources& res, Index& index)
{
  auto stream  = raft::resource::get_cuda_stream(res);
  auto tmp_res = raft::resource::get_workspace_resource_ref(res);
  rmm::device_uvector<uint32_t> sorted_sizes(index.n_lists(), stream, tmp_res);

  // Actualize the list pointers
  auto data_ptrs = index.data_ptrs();
  auto inds_ptrs = index.inds_ptrs();
  for (uint32_t label = 0; label < index.n_lists(); label++) {
    auto& list          = index.lists()[label];
    const auto data_ptr = list ? list->data_ptr() : nullptr;
    const auto inds_ptr = list ? list->indices_ptr() : nullptr;
    raft::copy(&data_ptrs(label), &data_ptr, 1, stream);
    raft::copy(&inds_ptrs(label), &inds_ptr, 1, stream);
  }

  // Sort the cluster sizes in the descending order.
  // Use helper function to avoid including cub/device/* in this header
  sort_cluster_sizes_descending(
    index.list_sizes().data_handle(), sorted_sizes.data(), index.n_lists(), stream, tmp_res);
  // copy the results to CPU
  auto sorted_sizes_host = raft::make_host_vector<uint32_t>(index.n_lists());
  raft::copy(res,
             sorted_sizes_host.view(),
             raft::make_device_vector_view(sorted_sizes.data(), index.n_lists()));
  raft::resource::sync_stream(res);

  // accumulate the sorted cluster sizes
  auto accum_sorted_sizes = index.accum_sorted_sizes();
  accum_sorted_sizes(0)   = 0;
  for (uint32_t label = 0; label < sorted_sizes_host.extent(0); label++) {
    accum_sorted_sizes(label + 1) = accum_sorted_sizes(label) + sorted_sizes_host(label);
  }
}

}  // namespace cuvs::neighbors::ivf::detail
