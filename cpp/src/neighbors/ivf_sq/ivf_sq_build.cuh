/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../core/nvtx.hpp"
#include "../ivf_common.cuh"
#include "../ivf_list.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_sq.hpp>

#include "../detail/ann_utils.cuh"
#include <cuvs/distance/distance.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/matrix/sample_rows.cuh>
#include <raft/random/rng.cuh>
#include <raft/stats/histogram.cuh>
#include <raft/util/pow2_utils.cuh>
#include <raft/util/vectorized.cuh>

#include <cub/block/block_reduce.cuh>

#include <rmm/cuda_stream_view.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace cuvs::neighbors::ivf_sq {
using namespace cuvs::spatial::knn::detail;  // NOLINT

namespace detail {

struct ColMinMaxPair {
  float min_val;
  float max_val;
};

struct ColMinMaxOp {
  __device__ __forceinline__ ColMinMaxPair operator()(const ColMinMaxPair& a,
                                                      const ColMinMaxPair& b) const
  {
    return {fminf(a.min_val, b.min_val), fmaxf(a.max_val, b.max_val)};
  }
};

/**
 * Vectorized load helper: reads VecCols contiguous elements of type T as
 * a single aligned wide load (via raft::TxN_t -> __ldg on the promoted
 * io_t) and unpacks them into floats.
 *
 * The primary benefit over scalar loads is halving (VecCols=2) or
 * quartering (VecCols=4) the number of LDG instructions issued per warp,
 * which is the dominant cost in the column-strided access pattern of
 * fused_column_minmax_kernel. VecCols=1 is the degenerate scalar
 * fallback for odd `dim`.
 *
 * Requires `p` to be aligned to sizeof(raft::IOType<T, VecCols>::Type),
 * i.e. sizeof(T) * VecCols.
 */
template <typename T, int VecCols>
__device__ __forceinline__ void load_cols_as_float(const T* p, float (&out)[VecCols])
{
  raft::TxN_t<T, VecCols> v;
  v.load(p, 0);
#pragma unroll
  for (int k = 0; k < VecCols; ++k) {
    out[k] = static_cast<float>(v.val.data[k]);
  }
}

/**
 * Fused per-column min+max in a single pass (2x less DRAM traffic than two
 * separate reductions). Each block owns VecCols contiguous columns and
 * threads read them with a single aligned wide load (float4/float2 for
 * float, uint2/uint32 for half) per row instead of VecCols scalar loads.
 * Requires dim to be a multiple of VecCols so that
 * `data + row*dim + col_base` is `sizeof(T)*VecCols` aligned for every
 * row; the host-side dispatcher picks the widest VecCols that satisfies
 * this, down to VecCols=1 (pure scalar fallback) for arbitrary dim.
 *
 * Row-loop is manually 4x-unrolled so the compiler can overlap four
 * independent read-only loads in the memory pipeline.
 */
template <int BlockSize, int VecCols, typename T>
__launch_bounds__(BlockSize) RAFT_KERNEL fused_column_minmax_kernel(const T* __restrict__ data,
                                                                    float* __restrict__ col_min,
                                                                    float* __restrict__ col_max,
                                                                    int64_t n_rows,
                                                                    uint32_t dim)
{
  using BlockReduce = cub::BlockReduce<ColMinMaxPair, BlockSize>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const uint32_t col_base = blockIdx.x * VecCols;
  // When launched with gridDim.x = dim / VecCols and dim % VecCols == 0
  // (enforced by the host-side dispatch), col_base + VecCols <= dim always.

  ColMinMaxPair agg[VecCols];
#pragma unroll
  for (int k = 0; k < VecCols; ++k) {
    agg[k] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest()};
  }

  const int64_t stride = static_cast<int64_t>(BlockSize);
  int64_t row          = static_cast<int64_t>(threadIdx.x);

  // 4x row-unrolled loop with vectorized loads: 4 * VecCols values per iter
  // are pulled into registers via 4 wide LDGs, exposing ILP across both
  // the column and row axes.
  for (; row + 3 * stride < n_rows; row += 4 * stride) {
    float r0[VecCols], r1[VecCols], r2[VecCols], r3[VecCols];
    load_cols_as_float<T, VecCols>(data + row * dim + col_base, r0);
    load_cols_as_float<T, VecCols>(data + (row + stride) * dim + col_base, r1);
    load_cols_as_float<T, VecCols>(data + (row + 2 * stride) * dim + col_base, r2);
    load_cols_as_float<T, VecCols>(data + (row + 3 * stride) * dim + col_base, r3);
#pragma unroll
    for (int k = 0; k < VecCols; ++k) {
      float mn       = fminf(fminf(r0[k], r1[k]), fminf(r2[k], r3[k]));
      float mx       = fmaxf(fmaxf(r0[k], r1[k]), fmaxf(r2[k], r3[k]));
      agg[k].min_val = fminf(agg[k].min_val, mn);
      agg[k].max_val = fmaxf(agg[k].max_val, mx);
    }
  }
  for (; row < n_rows; row += stride) {
    float r[VecCols];
    load_cols_as_float<T, VecCols>(data + row * dim + col_base, r);
#pragma unroll
    for (int k = 0; k < VecCols; ++k) {
      agg[k].min_val = fminf(agg[k].min_val, r[k]);
      agg[k].max_val = fmaxf(agg[k].max_val, r[k]);
    }
  }

  // One block-reduce per owned column. CUB requires a __syncthreads()
  // between reuses of the shared temp_storage.
  for (int k = 0; k < VecCols; ++k) {
    if (k > 0) __syncthreads();
    auto r = BlockReduce(temp_storage).Reduce(agg[k], ColMinMaxOp());
    if (threadIdx.x == 0) {
      col_min[col_base + k] = r.min_val;
      col_max[col_base + k] = r.max_val;
    }
  }
}

/**
 * Host-side dispatch that selects the widest VecCols compatible with the
 * given dim alignment (VecCols in {4, 2, 1}), and launches
 * fused_column_minmax_kernel with the corresponding grid shape.
 */
template <int BlockSize, typename T>
inline void launch_fused_column_minmax(
  const T* data, float* col_min, float* col_max, int64_t n_rows, uint32_t dim, cudaStream_t stream)
{
  if (dim % 4 == 0) {
    fused_column_minmax_kernel<BlockSize, 4, T>
      <<<dim / 4, BlockSize, 0, stream>>>(data, col_min, col_max, n_rows, dim);
  } else if (dim % 2 == 0) {
    fused_column_minmax_kernel<BlockSize, 2, T>
      <<<dim / 2, BlockSize, 0, stream>>>(data, col_min, col_max, n_rows, dim);
  } else {
    fused_column_minmax_kernel<BlockSize, 1, T>
      <<<dim, BlockSize, 0, stream>>>(data, col_min, col_max, n_rows, dim);
  }
}

/**
 * Kernel to compute residuals on the fly and encode them to uint8_t SQ codes
 * interleaved into the target list.
 *
 * Uses warp-per-vector parallelism: each warp cooperatively encodes one vector
 * so that reads from new_vectors/centers/vmin/delta are coalesced across the
 * 32 lanes. Lane 0 handles the atomic position assignment and the index write.
 *
 * The residual (cast<float>(new_vectors[i,d]) - centers[list_id,d]) is computed
 * in registers and consumed immediately, avoiding a full HBM round trip through
 * an intermediate residuals buffer.
 */
template <int BlockSize, typename T>
__launch_bounds__(BlockSize) RAFT_KERNEL encode_and_fill_kernel(const uint32_t* labels,
                                                                const T* new_vectors,
                                                                const float* centers,
                                                                const int64_t* source_ixs,
                                                                uint8_t** list_data_ptrs,
                                                                int64_t** list_index_ptrs,
                                                                uint32_t* list_sizes_ptr,
                                                                const float* vmin,
                                                                const float* delta,
                                                                int64_t n_rows,
                                                                uint32_t dim,
                                                                int64_t batch_offset)
{
  constexpr uint32_t kWarpSize      = kIndexGroupSize;
  constexpr uint32_t kWarpsPerBlock = BlockSize / kWarpSize;

  const uint32_t lane_id = threadIdx.x % kWarpSize;
  const int64_t row_id =
    int64_t(threadIdx.x / kWarpSize) + int64_t(blockIdx.x) * int64_t(kWarpsPerBlock);
  if (row_id >= n_rows) return;

  uint32_t list_id   = 0;
  uint32_t inlist_id = 0;
  if (lane_id == 0) {
    auto source_ix = source_ixs == nullptr ? row_id + batch_offset : source_ixs[row_id];
    list_id        = labels[row_id];
    inlist_id      = atomicAdd(list_sizes_ptr + list_id, 1);
    list_index_ptrs[list_id][inlist_id] = source_ix;
  }
  list_id   = __shfl_sync(0xFFFFFFFF, list_id, 0);
  inlist_id = __shfl_sync(0xFFFFFFFF, inlist_id, 0);

  using interleaved_group = raft::Pow2<kIndexGroupSize>;
  auto group_offset       = interleaved_group::roundDown(inlist_id);
  auto ingroup_id         = interleaved_group::mod(inlist_id);

  constexpr uint32_t veclen = list_spec<uint32_t, uint8_t, int64_t>::kVecLen;
  uint32_t padded_dim       = ((dim + veclen - 1) / veclen) * veclen;
  auto* list_dat   = list_data_ptrs[list_id] + static_cast<size_t>(group_offset) * padded_dim;
  const T* src     = new_vectors + row_id * dim;
  const float* ctr = centers + static_cast<size_t>(list_id) * dim;

  for (uint32_t d = lane_id; d < padded_dim; d += kWarpSize) {
    uint8_t out;
    if (d < dim) {
      float val  = utils::mapping<float>{}(src[d]) - ctr[d];
      float dv   = delta[d];
      float v    = vmin[d];
      float code = (dv > 0.0f) ? roundf((val - v) / dv) : 0.0f;
      out        = static_cast<uint8_t>(fminf(fmaxf(code, 0.0f), 255.0f));
    } else {
      out = 0;
    }
    uint32_t l                                              = (d / veclen) * veclen;
    uint32_t j                                              = d % veclen;
    list_dat[l * kIndexGroupSize + ingroup_id * veclen + j] = out;
  }
}

/** In-place variant: dataset[i] = cast<T>(cast<float>(dataset[i]) - centers[labels[i]]) */
template <typename T>
RAFT_KERNEL compute_residuals_inplace_kernel(
  T* dataset, const float* centers, const uint32_t* labels, int64_t n_rows, uint32_t dim)
{
  int64_t i = blockIdx.x;
  if (i >= n_rows) return;
  uint32_t c = labels[i];
  for (uint32_t j = threadIdx.x; j < dim; j += blockDim.x) {
    float val            = utils::mapping<float>{}(dataset[i * dim + j]);
    dataset[i * dim + j] = utils::mapping<T>{}(val - centers[c * dim + j]);
  }
}

template <typename T, typename CodeT>
void extend_inplace(raft::resources const& handle,
                    index<CodeT>* index,
                    const T* new_vectors,
                    const int64_t* new_indices,
                    int64_t n_rows)
{
  using LabelT = uint32_t;
  RAFT_EXPECTS(index != nullptr, "index cannot be empty.");
  if (n_rows == 0) return;

  auto stream  = raft::resource::get_cuda_stream(handle);
  auto n_lists = index->n_lists();
  auto dim     = index->dim();
  list_spec<uint32_t, CodeT, int64_t> list_device_spec{index->dim(),
                                                       index->conservative_memory_allocation()};
  cuvs::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "ivf_sq::extend(%zu, %u)", size_t(n_rows), dim);

  RAFT_EXPECTS(new_indices != nullptr || index->size() == 0,
               "You must pass data indices when the index is non-empty.");

  auto new_labels =
    raft::make_device_mdarray<LabelT>(handle,
                                      raft::resource::get_large_workspace_resource_ref(handle),
                                      raft::make_extents<int64_t>(n_rows));
  cuvs::cluster::kmeans::balanced_params kmeans_params;
  kmeans_params.metric     = index->metric();
  auto orig_centroids_view = raft::make_device_matrix_view<const float, int64_t>(
    index->centers().data_handle(), n_lists, dim);

  constexpr size_t kReasonableMaxBatchSize = 65536;
  size_t max_batch_size                    = std::min<size_t>(n_rows, kReasonableMaxBatchSize);

  auto copy_stream     = raft::resource::get_cuda_stream(handle);
  bool enable_prefetch = false;
  if (handle.has_resource_factory(raft::resource::resource_type::CUDA_STREAM_POOL)) {
    if (raft::resource::get_stream_pool_size(handle) >= 1) {
      enable_prefetch = true;
      copy_stream     = raft::resource::get_stream_from_stream_pool(handle);
    }
  }

  auto vec_batches =
    utils::make_batch_load_iterator<T>(handle,
                                       new_vectors,
                                       n_rows,
                                       int64_t{index->dim()},
                                       max_batch_size,
                                       copy_stream,
                                       raft::resource::get_workspace_resource_ref(handle),
                                       enable_prefetch);
  vec_batches.prefetch_next_batch();

  const bool needs_prefetch_sync = enable_prefetch && vec_batches.does_copy();

  for (const auto& batch : vec_batches) {
    auto batch_data_view =
      raft::make_device_matrix_view<const T, int64_t>(batch.data(), batch.size(), index->dim());
    auto batch_labels_view = raft::make_device_vector_view<LabelT, int64_t>(
      new_labels.data_handle() + batch.offset(), batch.size());
    cuvs::cluster::kmeans::predict(
      handle, kmeans_params, batch_data_view, orig_centroids_view, batch_labels_view);
    vec_batches.prefetch_next_batch();
    if (needs_prefetch_sync) { raft::resource::sync_stream(handle); }
  }

  auto* list_sizes_ptr    = index->list_sizes().data_handle();
  auto old_list_sizes_dev = raft::make_device_vector<uint32_t, int64_t>(handle, n_lists);
  raft::copy(old_list_sizes_dev.data_handle(), list_sizes_ptr, n_lists, stream);

  raft::stats::histogram<uint32_t, int64_t>(raft::stats::HistTypeAuto,
                                            reinterpret_cast<int32_t*>(list_sizes_ptr),
                                            int64_t(n_lists),
                                            new_labels.data_handle(),
                                            n_rows,
                                            1,
                                            stream);
  raft::linalg::add(
    list_sizes_ptr, list_sizes_ptr, old_list_sizes_dev.data_handle(), n_lists, stream);

  std::vector<uint32_t> new_list_sizes(n_lists);
  std::vector<uint32_t> old_list_sizes(n_lists);
  {
    raft::copy(old_list_sizes.data(), old_list_sizes_dev.data_handle(), n_lists, stream);
    raft::copy(new_list_sizes.data(), list_sizes_ptr, n_lists, stream);
    raft::resource::sync_stream(handle);
    auto& lists = index->lists();
    for (uint32_t label = 0; label < n_lists; label++) {
      ivf::resize_list(handle,
                       lists[label],
                       list_device_spec,
                       new_list_sizes[label],
                       old_list_sizes[label],
                       raft::Pow2<kIndexGroupSize>::roundUp(old_list_sizes[label]));
    }
  }
  ivf::detail::recompute_internal_state(handle, *index);
  raft::copy(list_sizes_ptr, old_list_sizes_dev.data_handle(), n_lists, stream);

  auto vec_indices =
    utils::make_batch_load_iterator<int64_t>(handle,
                                             new_indices,
                                             n_rows,
                                             int64_t{1},
                                             max_batch_size,
                                             stream,
                                             raft::resource::get_workspace_resource_ref(handle));
  vec_batches.reset();
  vec_batches.prefetch_next_batch();
  auto idx_batch = vec_indices.begin();

  size_t next_report_offset = 0;
  size_t d_report_offset    = n_rows * 5 / 100;

  for (const auto& batch : vec_batches) {
    int64_t bs = batch.size();

    {
      constexpr int kEncodeBlockSize   = 256;
      constexpr int kEncodeWarpsPerBlk = kEncodeBlockSize / kIndexGroupSize;
      const dim3 block_dim(kEncodeBlockSize);
      const dim3 grid_dim(raft::ceildiv<int64_t>(bs, int64_t(kEncodeWarpsPerBlk)));
      encode_and_fill_kernel<kEncodeBlockSize, T>
        <<<grid_dim, block_dim, 0, stream>>>(new_labels.data_handle() + batch.offset(),
                                             batch.data(),
                                             index->centers().data_handle(),
                                             idx_batch->data(),
                                             index->data_ptrs().data_handle(),
                                             index->inds_ptrs().data_handle(),
                                             list_sizes_ptr,
                                             index->sq_vmin().data_handle(),
                                             index->sq_delta().data_handle(),
                                             bs,
                                             dim,
                                             batch.offset());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }

    vec_batches.prefetch_next_batch();
    if (needs_prefetch_sync) { raft::resource::sync_stream(handle); }

    if (batch.offset() > next_report_offset) {
      float progress = batch.offset() * 100.0f / n_rows;
      RAFT_LOG_DEBUG("ivf_sq::extend added vectors %zu, %6.1f%% complete",
                     static_cast<size_t>(batch.offset()),
                     progress);
      next_report_offset += d_report_offset;
    }
    ++idx_batch;
  }

  auto compute_center_norms = [&]() {
    if (index->metric() == cuvs::distance::DistanceType::CosineExpanded) {
      raft::linalg::rowNorm<raft::linalg::L2Norm, true>(index->center_norms()->data_handle(),
                                                        index->centers().data_handle(),
                                                        dim,
                                                        n_lists,
                                                        stream,
                                                        raft::sqrt_op{});
    } else {
      raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
        index->center_norms()->data_handle(), index->centers().data_handle(), dim, n_lists, stream);
    }
  };

  if (!index->center_norms().has_value()) {
    index->allocate_center_norms(handle);
    if (index->center_norms().has_value()) { compute_center_norms(); }
  }
}

template <typename T, typename CodeT, typename accessor>
inline auto build(
  raft::resources const& handle,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, accessor> dataset)
  -> index<CodeT>
{
  int64_t n_rows = dataset.extent(0);
  uint32_t dim   = dataset.extent(1);
  auto stream    = raft::resource::get_cuda_stream(handle);
  cuvs::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "ivf_sq::build(%zu, %u)", size_t(n_rows), dim);
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>, "unsupported data type");
  RAFT_EXPECTS(n_rows > 0 && dim > 0, "empty dataset");
  RAFT_EXPECTS(n_rows >= params.n_lists, "number of rows can't be less than n_lists");
  RAFT_EXPECTS(params.max_train_points_per_cluster > 0, "max_train_points_per_cluster must be > 0");
  RAFT_EXPECTS(params.metric != cuvs::distance::DistanceType::CosineExpanded || dim > 1,
               "Cosine metric requires more than one dim");

  index<CodeT> idx(handle, params, dim);

  // Train k-means centroids and SQ parameters on the same training subset.
  // This mirrors IVF-PQ, which also trains its codebook on a subset of the data.
  {
    raft::random::RngState random_state{137};
    auto max_train_rows =
      static_cast<size_t>(idx.n_lists()) * static_cast<size_t>(params.max_train_points_per_cluster);
    if (max_train_rows > static_cast<size_t>(n_rows)) {
      RAFT_LOG_WARN(
        "ivf_sq::build: requested k-means training set size (%zu) exceeds "
        "dataset size (%zu); using the full dataset for training.",
        max_train_rows,
        static_cast<size_t>(n_rows));
    }
    auto n_rows_train = std::min<size_t>(static_cast<size_t>(n_rows), max_train_rows);
    auto trainset     = raft::make_device_mdarray<T>(
      handle,
      raft::resource::get_large_workspace_resource_ref(handle),
      raft::make_extents<int64_t>(static_cast<int64_t>(n_rows_train), idx.dim()));
    raft::matrix::sample_rows<T, int64_t>(handle, random_state, dataset, trainset.view());
    auto trainset_const_view = raft::make_const_mdspan(trainset.view());
    auto centers_view        = raft::make_device_matrix_view<float, int64_t>(
      idx.centers().data_handle(), idx.n_lists(), idx.dim());
    cuvs::cluster::kmeans::balanced_params kmeans_params;
    kmeans_params.n_iters = params.kmeans_n_iters;
    kmeans_params.metric  = idx.metric();
    cuvs::cluster::kmeans::fit(handle, kmeans_params, trainset_const_view, centers_view);

    // Train SQ: predict labels for the training subset, compute residuals in-place,
    // and derive per-dimension vmin/delta from them.
    {
      auto train_labels       = raft::make_device_vector<uint32_t, int64_t>(handle, n_rows_train);
      auto centers_const_view = raft::make_device_matrix_view<const float, int64_t>(
        idx.centers().data_handle(), idx.n_lists(), dim);
      cuvs::cluster::kmeans::predict(
        handle, kmeans_params, trainset_const_view, centers_const_view, train_labels.view());

      constexpr int kResidualBlockSize = 256;
      compute_residuals_inplace_kernel<T>
        <<<n_rows_train, kResidualBlockSize, 0, stream>>>(trainset.data_handle(),
                                                          idx.centers().data_handle(),
                                                          train_labels.data_handle(),
                                                          n_rows_train,
                                                          dim);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }

    // After the in-place kernel, trainset now contains residuals.
    auto& residuals = trainset;

    {
      auto vmax_buf  = raft::make_device_vector<float, uint32_t>(handle, dim);
      auto* vmin_ptr = idx.sq_vmin().data_handle();
      auto* vmax_ptr = vmax_buf.data_handle();

      constexpr int kMinMaxBlockSize = 256;
      launch_fused_column_minmax<kMinMaxBlockSize, T>(
        residuals.data_handle(), vmin_ptr, vmax_ptr, n_rows_train, dim, stream);
      RAFT_CUDA_TRY(cudaPeekAtLastError());

      // Expand the observed range by a small margin to reduce clipping on unseen data,
      // since the SQ parameters are trained on a subset rather than the full dataset.
      constexpr float kMargin = 0.05f;
      auto* delta_ptr         = idx.sq_delta().data_handle();
      raft::linalg::map_offset(
        handle, idx.sq_vmin(), [vmin_ptr, vmax_ptr, delta_ptr, kMargin] __device__(uint32_t j) {
          float range  = vmax_ptr[j] - vmin_ptr[j];
          float margin = range * kMargin;
          delta_ptr[j] = (range > 0.0f) ? (range + 2.0f * margin) / 255.0f : 1.0f;
          return vmin_ptr[j] - margin;
        });
    }
  }

  if (params.add_data_on_build) {
    detail::extend_inplace<T, CodeT>(handle, &idx, dataset.data_handle(), nullptr, n_rows);
  }

  return idx;
}

template <typename T, typename CodeT>
void extend(raft::resources const& handle,
            raft::device_matrix_view<const T, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            index<CodeT>* idx)
{
  RAFT_EXPECTS(new_vectors.extent(1) == idx->dim(),
               "new_vectors should have the same dimension as the index");
  if (new_indices.has_value()) {
    RAFT_EXPECTS(new_indices.value().extent(0) == new_vectors.extent(0),
                 "new_vectors and new_indices have different number of rows");
  }
  detail::extend_inplace(handle,
                         idx,
                         new_vectors.data_handle(),
                         new_indices.has_value() ? new_indices.value().data_handle() : nullptr,
                         new_vectors.extent(0));
}

template <typename T, typename CodeT>
void extend(raft::resources const& handle,
            raft::host_matrix_view<const T, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            index<CodeT>* idx)
{
  RAFT_EXPECTS(new_vectors.extent(1) == idx->dim(),
               "new_vectors should have the same dimension as the index");
  if (new_indices.has_value()) {
    RAFT_EXPECTS(new_indices.value().extent(0) == new_vectors.extent(0),
                 "new_vectors and new_indices have different number of rows");
  }
  detail::extend_inplace(handle,
                         idx,
                         new_vectors.data_handle(),
                         new_indices.has_value() ? new_indices.value().data_handle() : nullptr,
                         new_vectors.extent(0));
}

}  // namespace detail
}  // namespace cuvs::neighbors::ivf_sq
