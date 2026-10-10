/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <cstddef>
#include <cstdint>

/**
 * Deterministic per-cluster weighted row sums for the k-means centroid update.
 *
 * Accumulating the rows of each cluster with atomics (as raft::linalg::reduce_rows_by_key does)
 * makes the floating-point summation order, and therefore the centroids, vary from run to run.
 * Through the convergence check this can change the number of Lloyd iterations, so two identical
 * fits can return noticeably different centroids. Here the summation order is a function of the
 * input only:
 *
 *   1. Row indices are stably sorted by label, so each cluster becomes a contiguous segment.
 *   2. Each segment is cut into pieces of at most `kRowsPerPiece` rows. One thread block reduces a
 *      piece with a fixed assignment of rows to threads and a fixed-order combine.
 *   3. One thread block per cluster adds up that cluster's piece sums in the same fixed way, and
 *      adds the result to the output.
 */
namespace cuvs::cluster::kmeans::detail::sums_by_label {

constexpr int kBlockThreads = 256;
constexpr int kRowsPerPiece = 256;

/** Number of threads along the columns: smallest power of two >= n_cols (at most kBlockThreads). */
inline int column_threads(int64_t n_cols)
{
  int tc = 1;
  while (tc < n_cols && tc < kBlockThreads) {
    tc <<= 1;
  }
  return tc;
}

/**
 * Sums `values(i, j)` over i in [begin, end) for each column j in [0, n_cols), using a fixed
 * assignment of items to threads and a fixed-order combine, so the result does not depend on
 * scheduling. The result for column j is passed to `write(j, sum)` by a single thread.
 */
template <typename DataT, typename IndexT, typename ValueOp, typename WriteOp>
__device__ void fixed_order_column_sums(
  IndexT begin, IndexT end, IndexT n_cols, int col_threads, ValueOp values, WriteOp write)
{
  __shared__ DataT smem[kBlockThreads];
  const int lanes = kBlockThreads / col_threads;
  const int tc    = threadIdx.x % col_threads;
  const int lane  = threadIdx.x / col_threads;
  for (IndexT col0 = 0; col0 < n_cols; col0 += col_threads) {
    const IndexT j = col0 + tc;
    DataT acc      = DataT{0};
    if (j < n_cols) {
#pragma unroll 4
      for (IndexT i = begin + lane; i < end; i += lanes) {
        acc += values(i, j);
      }
    }
    smem[threadIdx.x] = acc;
    __syncthreads();
    if (lane == 0 && j < n_cols) {
      DataT sum = smem[tc];
      for (int l = 1; l < lanes; ++l) {
        sum += smem[l * col_threads + tc];
      }
      write(j, sum);
    }
    __syncthreads();
  }
}

template <typename LabelsIterator, typename IndexT>
RAFT_KERNEL init_sort_kernel(LabelsIterator labels, IndexT n_rows, uint32_t* keys, IndexT* rows)
{
  for (IndexT i = static_cast<IndexT>(blockIdx.x) * blockDim.x + threadIdx.x; i < n_rows;
       i += static_cast<IndexT>(blockDim.x) * gridDim.x) {
    keys[i] = static_cast<uint32_t>(labels[i]);
    rows[i] = i;
  }
}

/** Finds each cluster's segment of the sorted labels and the number of pieces it is cut into. */
template <typename IndexT>
RAFT_KERNEL segment_kernel(const uint32_t* sorted_keys,
                           IndexT n_rows,
                           IndexT n_clusters,
                           IndexT* cluster_offsets,
                           IndexT* piece_counts)
{
  auto lower_bound = [=](IndexT c) {
    IndexT lo = 0, hi = n_rows;
    while (lo < hi) {
      IndexT mid = lo + (hi - lo) / 2;
      if (static_cast<IndexT>(sorted_keys[mid]) < c) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return lo;
  };
  for (IndexT c = static_cast<IndexT>(blockIdx.x) * blockDim.x + threadIdx.x; c <= n_clusters;
       c += static_cast<IndexT>(blockDim.x) * gridDim.x) {
    IndexT begin       = lower_bound(c);
    cluster_offsets[c] = begin;
    if (c < n_clusters) {
      IndexT end      = lower_bound(c + 1);
      piece_counts[c] = raft::ceildiv<IndexT>(end - begin, kRowsPerPiece);
    } else {
      piece_counts[c] = 0;
    }
  }
}

template <typename DataT, typename IndexT>
RAFT_KERNEL piece_sums_kernel(const DataT* X,
                              IndexT n_cols,
                              const DataT* weights,
                              const IndexT* sorted_rows,
                              const IndexT* cluster_offsets,
                              const IndexT* piece_offsets,
                              IndexT n_clusters,
                              int col_threads,
                              DataT* piece_sums)
{
  const IndexT n_pieces = piece_offsets[n_clusters];
  for (IndexT piece = blockIdx.x; piece < n_pieces; piece += gridDim.x) {
    // The cluster of this piece is the last c with piece_offsets[c] <= piece.
    IndexT lo = 0, hi = n_clusters;
    while (hi - lo > 1) {
      IndexT mid = lo + (hi - lo) / 2;
      if (piece_offsets[mid] <= piece) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    const IndexT c     = lo;
    const IndexT begin = cluster_offsets[c] + (piece - piece_offsets[c]) * kRowsPerPiece;
    const IndexT end   = std::min<IndexT>(begin + kRowsPerPiece, cluster_offsets[c + 1]);
    DataT* out         = piece_sums + static_cast<int64_t>(piece) * (n_cols + 1);
    fixed_order_column_sums<DataT>(
      begin,
      end,
      n_cols,
      col_threads,
      [=](IndexT i, IndexT j) {
        const IndexT row = sorted_rows[i];
        const DataT x    = X[static_cast<int64_t>(row) * n_cols + j];
        return weights != nullptr ? x * weights[row] : x;
      },
      [=](IndexT j, DataT sum) { out[j] = sum; });
    // The sum of weights is stored after the column sums.
    fixed_order_column_sums<DataT>(
      begin,
      end,
      IndexT{1},
      1,
      [=](IndexT i, IndexT) { return weights != nullptr ? weights[sorted_rows[i]] : DataT{1}; },
      [=](IndexT, DataT sum) { out[n_cols] = sum; });
  }
}

template <typename DataT, typename IndexT>
RAFT_KERNEL cluster_sums_kernel(const DataT* piece_sums,
                                IndexT n_cols,
                                const IndexT* piece_offsets,
                                IndexT n_clusters,
                                int col_threads,
                                bool reset_sums,
                                DataT* centroid_sums,
                                DataT* weight_per_cluster)
{
  for (IndexT c = blockIdx.x; c < n_clusters; c += gridDim.x) {
    const IndexT p_begin = piece_offsets[c];
    const IndexT p_end   = piece_offsets[c + 1];
    const IndexT stride  = n_cols + 1;
    DataT* out           = centroid_sums + static_cast<int64_t>(c) * n_cols;
    fixed_order_column_sums<DataT>(
      p_begin,
      p_end,
      n_cols,
      col_threads,
      [=](IndexT p, IndexT j) { return piece_sums[static_cast<int64_t>(p) * stride + j]; },
      [=](IndexT j, DataT sum) { out[j] = reset_sums ? sum : out[j] + sum; });
    fixed_order_column_sums<DataT>(
      p_begin,
      p_end,
      IndexT{1},
      1,
      [=](IndexT p, IndexT) { return piece_sums[static_cast<int64_t>(p) * stride + n_cols]; },
      [=](IndexT, DataT sum) {
        weight_per_cluster[c] = reset_sums ? sum : weight_per_cluster[c] + sum;
      });
  }
}

inline size_t align_bytes(size_t bytes) { return raft::alignTo<size_t>(bytes, 256); }

/**
 * @brief Deterministic weighted sum of the rows of X per label, and of the weights per label.
 *
 * centroid_sums[c, :] (+)= sum_{i : labels[i] == c} weights[i] * X[i, :]
 * weight_per_cluster[c] (+)= sum_{i : labels[i] == c} weights[i]
 *
 * The result is bitwise reproducible for identical inputs on the same device.
 *
 * @param[in]    handle             RAFT resources handle
 * @param[in]    X                  Row-major input [n_rows x n_cols]
 * @param[in]    weights            Per-row weights [n_rows]
 * @param[in]    labels             Label of each row, in [0, n_clusters)
 * @param[in]    n_rows             Number of rows of X
 * @param[in]    n_cols             Number of columns of X
 * @param[in]    n_clusters         Number of labels
 * @param[inout] centroid_sums      Row-major output [n_clusters x n_cols]
 * @param[inout] weight_per_cluster Output [n_clusters]
 * @param[inout] workspace          Scratch buffer, resized as needed
 * @param[in]    reset_sums         Overwrite the outputs if true, add to them otherwise
 */
template <typename DataT, typename IndexT, typename LabelsIterator>
void weighted_sums_by_label(raft::resources const& handle,
                            const DataT* X,
                            const DataT* weights,
                            LabelsIterator labels,
                            IndexT n_rows,
                            IndexT n_cols,
                            IndexT n_clusters,
                            DataT* centroid_sums,
                            DataT* weight_per_cluster,
                            rmm::device_uvector<char>& workspace,
                            bool reset_sums)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle).get();
  if (n_clusters <= 0) { return; }

  int end_bit = 1;
  while (end_bit < 32 && (uint64_t{1} << end_bit) < static_cast<uint64_t>(n_clusters)) {
    ++end_bit;
  }
  const IndexT max_pieces = raft::ceildiv<IndexT>(n_rows, kRowsPerPiece) + n_clusters;

  // Workspace layout
  size_t sort_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(nullptr,
                                                sort_bytes,
                                                static_cast<const uint32_t*>(nullptr),
                                                static_cast<uint32_t*>(nullptr),
                                                static_cast<const IndexT*>(nullptr),
                                                static_cast<IndexT*>(nullptr),
                                                n_rows,
                                                0,
                                                end_bit,
                                                stream));
  size_t scan_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceScan::ExclusiveSum(nullptr,
                                              scan_bytes,
                                              static_cast<const IndexT*>(nullptr),
                                              static_cast<IndexT*>(nullptr),
                                              n_clusters + 1,
                                              stream));
  const size_t keys_bytes    = align_bytes(sizeof(uint32_t) * n_rows);
  const size_t rows_bytes    = align_bytes(sizeof(IndexT) * n_rows);
  const size_t offsets_bytes = align_bytes(sizeof(IndexT) * (n_clusters + 1));
  const size_t pieces_bytes =
    align_bytes(sizeof(DataT) * static_cast<size_t>(max_pieces) * (n_cols + 1));
  const size_t temp_bytes = align_bytes(std::max(sort_bytes, scan_bytes));
  workspace.resize(2 * keys_bytes + 2 * rows_bytes + 3 * offsets_bytes + pieces_bytes + temp_bytes,
                   stream);

  char* ptr = workspace.data();
  auto take = [&ptr](size_t bytes) {
    char* p = ptr;
    ptr += bytes;
    return p;
  };
  auto* keys_in         = reinterpret_cast<uint32_t*>(take(keys_bytes));
  auto* keys_out        = reinterpret_cast<uint32_t*>(take(keys_bytes));
  auto* rows_in         = reinterpret_cast<IndexT*>(take(rows_bytes));
  auto* rows_out        = reinterpret_cast<IndexT*>(take(rows_bytes));
  auto* cluster_offsets = reinterpret_cast<IndexT*>(take(offsets_bytes));
  auto* piece_counts    = reinterpret_cast<IndexT*>(take(offsets_bytes));
  auto* piece_offsets   = reinterpret_cast<IndexT*>(take(offsets_bytes));
  auto* piece_sums      = reinterpret_cast<DataT*>(take(pieces_bytes));
  void* temp            = take(temp_bytes);

  if (n_rows > 0) {
    const int grid = static_cast<int>(
      std::min<int64_t>(raft::ceildiv<int64_t>(n_rows, kBlockThreads), int64_t{65536}));
    init_sort_kernel<<<grid, kBlockThreads, 0, stream>>>(labels, n_rows, keys_in, rows_in);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(
      temp, sort_bytes, keys_in, keys_out, rows_in, rows_out, n_rows, 0, end_bit, stream));
  }
  {
    const int grid = static_cast<int>(
      std::min<int64_t>(raft::ceildiv<int64_t>(n_clusters + 1, kBlockThreads), int64_t{65536}));
    segment_kernel<<<grid, kBlockThreads, 0, stream>>>(
      keys_out, n_rows, n_clusters, cluster_offsets, piece_counts);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
  RAFT_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
    temp, scan_bytes, piece_counts, piece_offsets, n_clusters + 1, stream));

  const int col_threads = column_threads(n_cols);
  if (max_pieces > 0) {
    const int grid = static_cast<int>(std::min<int64_t>(max_pieces, int64_t{65536}));
    piece_sums_kernel<<<grid, kBlockThreads, 0, stream>>>(X,
                                                          n_cols,
                                                          weights,
                                                          rows_out,
                                                          cluster_offsets,
                                                          piece_offsets,
                                                          n_clusters,
                                                          col_threads,
                                                          piece_sums);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
  {
    const int grid = static_cast<int>(std::min<int64_t>(n_clusters, int64_t{65536}));
    cluster_sums_kernel<<<grid, kBlockThreads, 0, stream>>>(piece_sums,
                                                            n_cols,
                                                            piece_offsets,
                                                            n_clusters,
                                                            col_threads,
                                                            reset_sums,
                                                            centroid_sums,
                                                            weight_per_cluster);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

}  // namespace cuvs::cluster::kmeans::detail::sums_by_label
