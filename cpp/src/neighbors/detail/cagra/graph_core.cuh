/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "cagra_helpers.hpp"
#include "utils.hpp"

#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/reduce.cuh>
#include <raft/matrix/init.cuh>

// TODO: This shouldn't be invoking anything from spatial/knn
#include "../../../core/nvtx.hpp"
#include "../../../core/omp_wrapper.hpp"
#include "../ann_utils.cuh"

#include <cuvs/neighbors/cagra.hpp>

#include <raft/util/bitonic_sort.cuh>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/integer_utils.hpp>

#include <cuda_fp16.h>

#include <float.h>
#include <sys/time.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <climits>
#include <iostream>
#include <memory>
#include <random>

namespace cg = cooperative_groups;

namespace cuvs::neighbors::cagra::detail::graph {

// unnamed namespace to avoid multiple definition error
namespace {
inline double cur_time(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return ((double)tv.tv_sec + (double)tv.tv_usec * 1e-6);
}

template <typename T>
__device__ inline void swap(T& val1, T& val2)
{
  T val0 = val1;
  val1   = val2;
  val2   = val0;
}

template <typename K, typename V>
__device__ inline bool swap_if_needed(K& key1, K& key2, V& val1, V& val2, bool ascending)
{
  if (key1 == key2) { return false; }
  if ((key1 > key2) == ascending) {
    swap<K>(key1, key2);
    swap<V>(val1, val2);
    return true;
  }
  return false;
}

template <class DATA_T, class IdxT, int numElementsPerThread>
__global__ void kern_sort(const DATA_T* const dataset,  // [dataset_chunk_size, dataset_dim]
                          const IdxT dataset_size,
                          const uint32_t dataset_dim,
                          IdxT* const knn_graph,  // [graph_chunk_size, graph_degree]
                          const uint32_t graph_size,
                          const uint32_t graph_degree,
                          const cuvs::distance::DistanceType metric)
{
  const IdxT srcNode = (blockDim.x * blockIdx.x + threadIdx.x) / raft::WarpSize;
  if (srcNode >= graph_size) { return; }

  const uint32_t lane_id = threadIdx.x % raft::WarpSize;

  float my_keys[numElementsPerThread];
  IdxT my_vals[numElementsPerThread];

  // Compute distance from a src node to its neighbors
  for (int k = 0; k < graph_degree; k++) {
    const IdxT dstNode = knn_graph[k + static_cast<uint64_t>(graph_degree) * srcNode];
    float dist         = 0;
    float norm2_dst    = 0;
    if (metric == cuvs::distance::DistanceType::InnerProduct ||
        metric == cuvs::distance::DistanceType::CosineExpanded) {
      for (int d = lane_id; d < dataset_dim; d += raft::WarpSize) {
        auto elem_b = cuvs::spatial::knn::detail::utils::mapping<float>{}(
          dataset[d + static_cast<uint64_t>(dataset_dim) * dstNode]);
        dist -= cuvs::spatial::knn::detail::utils::mapping<float>{}(
                  dataset[d + static_cast<uint64_t>(dataset_dim) * srcNode]) *
                elem_b;

        if (metric == cuvs::distance::DistanceType::CosineExpanded) {
          norm2_dst += elem_b * elem_b;
        }
      }
    } else if (metric == cuvs::distance::DistanceType::L2Expanded) {
      // L2Expanded
      for (int d = lane_id; d < dataset_dim; d += raft::WarpSize) {
        float diff = cuvs::spatial::knn::detail::utils::mapping<float>{}(
                       dataset[d + static_cast<uint64_t>(dataset_dim) * srcNode]) -
                     cuvs::spatial::knn::detail::utils::mapping<float>{}(
                       dataset[d + static_cast<uint64_t>(dataset_dim) * dstNode]);
        dist += diff * diff;
      }
    } else if (metric == cuvs::distance::DistanceType::L1) {
      for (int d = lane_id; d < dataset_dim; d += raft::WarpSize) {
        float diff = cuvs::spatial::knn::detail::utils::mapping<float>{}(
                       dataset[d + static_cast<uint64_t>(dataset_dim) * srcNode]) -
                     cuvs::spatial::knn::detail::utils::mapping<float>{}(
                       dataset[d + static_cast<uint64_t>(dataset_dim) * dstNode]);
        dist += raft::abs(diff);
      }
    } else if (metric == cuvs::distance::DistanceType::BitwiseHamming) {
      if constexpr (std::is_integral_v<DATA_T>) {
        for (int d = lane_id; d < dataset_dim; d += raft::WarpSize) {
          dist += __popc(
            static_cast<uint32_t>(dataset[d + static_cast<uint64_t>(dataset_dim) * srcNode] ^
                                  dataset[d + static_cast<uint64_t>(dataset_dim) * dstNode]) &
            0xffu);
        }
      }
    }
    dist += __shfl_xor_sync(0xffffffff, dist, 1);
    dist += __shfl_xor_sync(0xffffffff, dist, 2);
    dist += __shfl_xor_sync(0xffffffff, dist, 4);
    dist += __shfl_xor_sync(0xffffffff, dist, 8);
    dist += __shfl_xor_sync(0xffffffff, dist, 16);

    if (metric == cuvs::distance::DistanceType::CosineExpanded) {
      norm2_dst += __shfl_xor_sync(0xffffffff, norm2_dst, 1);
      norm2_dst += __shfl_xor_sync(0xffffffff, norm2_dst, 2);
      norm2_dst += __shfl_xor_sync(0xffffffff, norm2_dst, 4);
      norm2_dst += __shfl_xor_sync(0xffffffff, norm2_dst, 8);
      norm2_dst += __shfl_xor_sync(0xffffffff, norm2_dst, 16);
      if (lane_id == (k % raft::WarpSize)) { dist /= sqrt(norm2_dst); }
    }

    if (lane_id == (k % raft::WarpSize)) {
      my_keys[k / raft::WarpSize] = dist;
      my_vals[k / raft::WarpSize] = dstNode;
    }
  }
  for (int k = graph_degree; k < raft::WarpSize * numElementsPerThread; k++) {
    if (lane_id == k % raft::WarpSize) {
      my_keys[k / raft::WarpSize] = utils::get_max_value<float>();
      my_vals[k / raft::WarpSize] = utils::get_max_value<IdxT>();
    }
  }

  // Sort by RAFT bitonic sort
  raft::util::bitonic<numElementsPerThread>(true).sort(my_keys, my_vals);

  // Update knn_graph
  for (int i = 0; i < numElementsPerThread; i++) {
    const int k = i * raft::WarpSize + lane_id;
    if (k < graph_degree) {
      knn_graph[k + (static_cast<uint64_t>(graph_degree) * srcNode)] = my_vals[i];
    }
  }
}

// `output_graph` may be either the full 2D graph ([graph_size, degree], indexed with column `k`)
// or a single pre-extracted column passed as a 1D vector ([graph_size], indexed by row only). The
// rank is inspected at compile time; `k` is always the logical column index and is used for the
// natural-degree mask regardless of the view rank.
template <typename IdxT, typename OutputView>
__global__ void kern_make_rev_graph_k(
  OutputView output_graph,                            // [graph_size, degree] or [graph_size]
  raft::device_matrix_view<IdxT, int64_t> rev_graph,  // [graph_size, degree]
  raft::device_vector_view<uint32_t, int64_t> rev_graph_count,  // [graph_size]
  uint64_t k)
{
  const uint64_t tid  = threadIdx.x + (blockDim.x * blockIdx.x);
  const uint64_t tnum = blockDim.x * gridDim.x;

  const uint64_t graph_size       = rev_graph.extent(0);
  const uint32_t rev_graph_degree = rev_graph.extent(1);

  for (uint64_t src_id = tid; src_id < graph_size; src_id += tnum) {
    IdxT dest_id;
    if constexpr (OutputView::rank() == 2) {
      dest_id = output_graph(src_id, k);
    } else {
      dest_id = output_graph(src_id);
    }
    if (dest_id >= graph_size) continue;

    const uint32_t pos = atomicAdd(&rev_graph_count(dest_id), 1);
    if (pos < rev_graph_degree) { rev_graph(dest_id, pos) = static_cast<IdxT>(src_id); }
  }
}

// KnnGraphView and OutputGraphView are mdspans with element type IdxT and
// dynamic 2D extents. They may have layout_right (raft::device_matrix_view) or
// layout_stride (the result of cuda::std::submdspan), and any accessor that is
// device-accessible (default_accessor, raft::host_device_accessor with a
// device-accessible memory_type, etc.).
//
// When VariableDegree is true, the kernel additionally writes a per-node
// "natural degree" (count of low-detour edges) into d_natural_degree.
// target_pruned_degree is the user-provided floor controlling where the natural
// detour-band boundary is detected (see comments in optimize()). When
// VariableDegree is false, d_natural_degree and target_pruned_degree are unused
// and the kernel behaves identically to its pre-feature form.
template <class IdxT,
          uint32_t num_warps,
          bool VariableDegree,
          class KnnGraphView,
          class OutputGraphView>
__global__ void kern_fused_prune(KnnGraphView knn_graph,        // [graph_chunk_size, graph_degree]
                                 OutputGraphView output_graph,  // [batch_size, output_graph_degree]
                                 const uint32_t batch_size,
                                 const uint32_t batch_id,
                                 uint32_t* const d_invalid_neighbor_list,
                                 uint64_t* const stats,
                                 raft::device_vector_view<uint32_t, int64_t> d_natural_degree,
                                 const uint32_t target_pruned_degree)
{
  // Check assumption we have at least one warp per row of the batch
  assert(blockDim.x == raft::WarpSize * num_warps);
  assert(gridDim.x * num_warps >= batch_size);

  extern __shared__ unsigned char smem_buf[];

  cg::thread_block block         = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

  const uint32_t wid     = threadIdx.x / raft::WarpSize;
  const uint32_t lane_id = threadIdx.x % raft::WarpSize;

  const uint64_t graph_size          = knn_graph.extent(0);
  const uint32_t knn_graph_degree    = knn_graph.extent(1);
  const uint32_t output_graph_degree = output_graph.extent(1);

  IdxT* const smem_indices =
    reinterpret_cast<IdxT*>(smem_buf + wid * knn_graph_degree * sizeof(IdxT));
  uint32_t* const smem_num_detour =
    reinterpret_cast<uint32_t*>(smem_buf + num_warps * knn_graph_degree * sizeof(IdxT) +
                                wid * knn_graph_degree * sizeof(uint32_t));

#ifndef NDEBUG
  uint64_t* const num_retain = stats;
  uint64_t* const num_full   = stats + 1;
#endif

  const uint32_t maxval16 = 0x0000ffff;

  const uint32_t nid_batch = blockIdx.x * num_warps + wid;
  const uint64_t nid       = static_cast<uint64_t>(nid_batch) +
                       (static_cast<uint64_t>(batch_size) * static_cast<uint64_t>(batch_id));

  if (nid >= graph_size) { return; }

  // Load this node's neighbor row into shared memory to reduce global reads
  for (uint32_t k = lane_id; k < knn_graph_degree; k += raft::WarpSize) {
    smem_num_detour[k] = 0;
    smem_indices[k]    = knn_graph(nid, k);
    if (smem_indices[k] == nid) {
      // Lower the priority of self-edge
      smem_num_detour[k] = knn_graph_degree;
    }
  }
  warp.sync();

  // count number of detours (A->D->B)
  for (uint32_t kAD = 0; kAD < knn_graph_degree - 1; kAD++) {
    const uint64_t iD = smem_indices[kAD];
    if (iD >= graph_size) { continue; }
    for (uint32_t kDB = lane_id; kDB < knn_graph_degree; kDB += raft::WarpSize) {
      const uint64_t iB_candidate = knn_graph(iD, kDB);
      for (uint32_t kAB = kAD + 1; kAB < knn_graph_degree; kAB++) {
        // if ( kDB < kAB )
        {
          const uint64_t iB = smem_indices[kAB];
          if (iB == iB_candidate) {
            atomicAdd(smem_num_detour + kAB, 1);
            break;
          }
        }
      }
    }
    warp.sync();
  }

  uint32_t num_edges_no_detour = 0;
  for (uint32_t k = lane_id; k < knn_graph_degree; k += raft::WarpSize) {
    smem_num_detour[k] = min(smem_num_detour[k], maxval16);
    if (smem_num_detour[k] == 0) { num_edges_no_detour++; }
    if (smem_indices[k] >= graph_size) { smem_num_detour[k] = maxval16; }
  }

  warp.sync();

  num_edges_no_detour = cg::reduce(warp, num_edges_no_detour, cg::plus<uint32_t>());
  num_edges_no_detour = min(num_edges_no_detour, output_graph_degree);

#ifndef NDEBUG
  if (lane_id == 0) {
    atomicAdd((unsigned long long int*)num_retain, (unsigned long long int)num_edges_no_detour);
    if (num_edges_no_detour >= output_graph_degree) {
      atomicAdd((unsigned long long int*)num_full, 1);
    }
  }
#endif

  // Detour level recorded at iteration (target_pruned_degree - 1). The natural-degree boundary is
  // the first iteration `i >= target_pruned_degree` for which warp_min_count strictly exceeds
  // this level. Initialized to maxval16 so the comparison below is well-defined before we reach
  // the target index.
  uint32_t target_detour_level = maxval16;
  // Final natural degree to write back to global memory; the default `output_graph_degree` means
  // "all edges fit under the target detour level" (i.e. no padding will be needed).
  uint32_t natural_degree = output_graph_degree;

  for (uint32_t i = 0; i < output_graph_degree; i++) {
    uint32_t local_min = maxval16;
    uint32_t local_idx = maxval16;
    for (uint32_t k = lane_id; k < knn_graph_degree; k += raft::WarpSize) {
      if (smem_num_detour[k] < local_min) {
        local_min = smem_num_detour[k];
        local_idx = k;
      }
    }

    uint32_t local_min_with_tag = (local_min << 16) | ((uint32_t)local_idx);
    uint32_t warp_min_with_tag  = cg::reduce(warp, local_min_with_tag, cg::less<uint32_t>());
    uint32_t warp_min_count     = warp_min_with_tag >> 16;
    uint32_t warp_local_idx     = warp_min_with_tag & 0xffff;

    if (warp_min_count == maxval16 || warp_local_idx == maxval16) {
      if (lane_id == 0) { atomicExch(d_invalid_neighbor_list, 1u); }
      break;
    }

    if constexpr (VariableDegree) {
      if (i + 1 == target_pruned_degree) {
        // Freeze the detour level after we've placed exactly target_pruned_degree edges.
        target_detour_level = warp_min_count;
      } else if (i >= target_pruned_degree && warp_min_count > target_detour_level &&
                 natural_degree == output_graph_degree) {
        // The detour level just rose above the target band. Record the natural degree once.
        natural_degree = i;
      }
    }

    IdxT selected_node = smem_indices[warp_local_idx];

    for (uint32_t k = lane_id; k < knn_graph_degree; k += raft::WarpSize) {
      if (smem_indices[k] == selected_node) { smem_num_detour[k] = maxval16; }
    }
    warp.sync();

    if (lane_id == 0) { output_graph(nid_batch, i) = selected_node; }
  }

  if constexpr (VariableDegree) {
    if (lane_id == 0) { d_natural_degree(nid) = natural_degree; }
  }
}

// Cheap stateless integer hash (PCG-style mixing) used for deterministic, warp-uniform thinning of
// reverse-graph edges during the merge step.
__device__ __forceinline__ uint32_t mix_hash(uint32_t x)
{
  x             = x * 747796405u + 2891336453u;
  uint32_t word = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return (word >> 22u) ^ word;
}

// Helper functions for merging the graph
template <typename T>
__device__ unsigned int warp_pos_in_array(T val, const T* array, uint64_t num)
{
  unsigned int ret       = num;
  const uint32_t lane_id = threadIdx.x % 32;
  for (uint64_t i = lane_id; i < num; i += 32) {
    if (val == array[i]) {
      ret = i;
      break;
    }
  }

  cg::thread_block block         = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
  ret                            = cg::reduce(warp, ret, cg::less<unsigned int>());
  return ret;
}

template <typename T>
__device__ void warp_shift_array_one_right(uint32_t lane_id, T* array, uint64_t num)
{
  if (num == 0) { return; }
  for (auto chunk_end = static_cast<int64_t>(num); chunk_end >= 1; chunk_end -= 31) {
    const int64_t chunk_start_lo = chunk_end - 31;
    const int64_t chunk_start    = (chunk_start_lo > 0) ? chunk_start_lo : 0;
    const int64_t k              = chunk_start + static_cast<int64_t>(lane_id);
    T val{};
    const bool read_active = (k <= chunk_end);
    if (read_active) { val = array[k]; }
    const T shifted         = raft::shfl_up(val, 1);
    const bool write_active = (lane_id > 0) && read_active;
    if (write_active) { array[k] = shifted; }
  }
}

// OutputGraphView, MstGraphView and MstNumEdgesView are 2D mdspans that may
// have layout_right or layout_stride and any device-accessible accessor; see
// the comment on kern_fused_prune for details.
//
// When VariableDegree is true, `d_natural_degree` (size graph_size) drives:
//   * the per-node `effective_degree` used to cap `num_protected_edges`
//   * the `kInvalidNeighbor` sentinel padding for slots past the final degree
//   * the write-back of the final effective degree
//   * popularity-based thinning of reverse-graph candidates (see the merge loop)
// When VariableDegree is false, the parameter is unused (and may be a default view).
template <typename IdxT,
          uint32_t num_warps,
          bool VariableDegree,
          class OutputGraphView,
          class MstGraphView,
          class MstNumEdgesView>
__global__ void kern_merge_graph(
  OutputGraphView output_graph,                                 // [batch_size, output_graph_degree]
  raft::device_matrix_view<IdxT, int64_t> rev_graph,            // [graph_size, output_graph_degree]
  raft::device_vector_view<uint32_t, int64_t> rev_graph_count,  // [graph_size]
  MstGraphView mst_graph,                                       // [batch_size, output_graph_degree]
  MstNumEdgesView mst_graph_num_edges,                          // [batch_size, 1]
  const uint32_t batch_size,
  const uint32_t batch_id,
  bool guarantee_connectivity,
  raft::device_scalar_view<uint32_t> check_num_protected_edges,
  raft::device_vector_view<uint32_t, int64_t> d_natural_degree)
{
  // Check assumption we have at least one warp per row of the batch
  assert(blockDim.x == raft::WarpSize * num_warps);
  assert(gridDim.x * num_warps >= batch_size);

  const uint64_t graph_size          = rev_graph.extent(0);
  const uint32_t output_graph_degree = output_graph.extent(1);
  const uint32_t mst_graph_degree    = mst_graph.extent(1);

  extern __shared__ unsigned char smem_buf[];

  const uint32_t wid     = threadIdx.x / raft::WarpSize;
  const uint32_t lane_id = threadIdx.x % raft::WarpSize;

  IdxT* smem_sorted_output_graph =
    reinterpret_cast<IdxT*>(smem_buf + wid * output_graph_degree * sizeof(IdxT));

  const uint32_t nid_batch = blockIdx.x * num_warps + wid;
  const uint64_t nid       = static_cast<uint64_t>(nid_batch) +
                       (static_cast<uint64_t>(batch_size) * static_cast<uint64_t>(batch_id));

  if (nid >= graph_size) { return; }

  // Per-node "in" degree before reverse-graph back-fill. In the constant-degree path this is just
  // `output_graph_degree`. In the variable-degree path it comes from the prune step.
  uint32_t effective_degree = VariableDegree ? d_natural_degree(nid) : output_graph_degree;

  const auto current_mst_graph_num_edges =
    guarantee_connectivity ? mst_graph_num_edges(nid_batch, 0) : 0;
  // If guarantee_connectivity == true, use a temporal list to merge the
  // neighbor lists of the graphs.
  if (guarantee_connectivity) {
    for (uint32_t i = lane_id; i < current_mst_graph_num_edges; i += raft::WarpSize) {
      smem_sorted_output_graph[i] = mst_graph(nid_batch, i);
    }
    __syncwarp();
    for (uint32_t pruned_j = 0, output_j = current_mst_graph_num_edges;
         (pruned_j < output_graph_degree) && (output_j < output_graph_degree);
         pruned_j++) {
      const auto v     = output_graph(nid_batch, pruned_j);
      unsigned int dup = 0;
      for (uint32_t m = lane_id; m < output_j; m += raft::WarpSize) {
        if (v == smem_sorted_output_graph[m]) {
          dup = 1;
          break;
        }
      }

      auto warp_dup = raft::ballot(dup);
      if (warp_dup == 0) {
        if (lane_id == 0) smem_sorted_output_graph[output_j] = v;
        output_j++;
      }
      __syncwarp();
    }
  }

  else {
    for (uint32_t i = lane_id; i < output_graph_degree; i += raft::WarpSize) {
      smem_sorted_output_graph[i] = output_graph(nid_batch, i);
    }
    __syncwarp();
  }

  // In the variable-degree path, `num_protected_edges` is additionally capped by the per-node
  // `effective_degree`: protecting more than the natural degree would prevent rev-graph edges
  // from being inserted (since they always go into slot `num_protected_edges`) and would mask
  // them with the prune-step contents that have not yet been validated as kept.
  const auto num_protected_edges =
    max(current_mst_graph_num_edges, min(effective_degree, output_graph_degree / 2));

  if (num_protected_edges > effective_degree) {
    check_num_protected_edges(0) = 0u;
    return;
  }
  // Variable-degree path always needs to write the -1 padding (and write-back natural_degree),
  // so we cannot early-return like the constant-degree path does when there is no room to insert.
  if constexpr (!VariableDegree) {
    if (num_protected_edges == output_graph_degree) { return; }
  }

  const auto my_in_degree = rev_graph_count(nid);
  const auto ed0          = effective_degree;
  auto kr                 = min(my_in_degree, output_graph_degree);

  while (kr) {
    kr -= 1;
    const auto rev_graph_value = rev_graph(nid, kr);
    if (rev_graph_value < graph_size) {
      if constexpr (VariableDegree) {
        const uint32_t in_degree = rev_graph_count(rev_graph_value);
        if (my_in_degree < output_graph_degree * 2 || in_degree < output_graph_degree * 2) {
          // Don't prune highways (edges between high in-degree nodes)
          // P(skipping the edge) = a / b
          const uint32_t a = output_graph_degree + ed0;
          const uint32_t b = output_graph_degree * 2;
          const uint32_t r = mix_hash(static_cast<uint32_t>(nid) * 0x9e3779b9u ^
                                      static_cast<uint32_t>(rev_graph_value));
          if ((r % b) < a) { continue; }
        }
      }
      uint64_t pos =
        warp_pos_in_array<IdxT>(rev_graph_value, smem_sorted_output_graph, output_graph_degree);
      if (pos < num_protected_edges) { continue; }
      uint64_t num_shift = pos - num_protected_edges;
      if (pos >= output_graph_degree) { num_shift = output_graph_degree - num_protected_edges - 1; }
      warp_shift_array_one_right<IdxT>(
        lane_id, smem_sorted_output_graph + num_protected_edges, num_shift);
      if (lane_id == 0) { smem_sorted_output_graph[num_protected_edges] = rev_graph_value; }
      __syncwarp();
      // A new rev-graph edge has been inserted at position `num_protected_edges`, growing the
      // node's effective degree up to (but not beyond) the full output_graph_degree. In the
      // constant-degree path effective_degree already equals output_graph_degree, so this is a
      // no-op and needs no VariableDegree guard.
      if (effective_degree < output_graph_degree) { effective_degree++; }
    }
  }

  // The write-back stays guarded: d_natural_degree is an empty view in the constant-degree path.
  if constexpr (VariableDegree) {
    if (lane_id == 0) { d_natural_degree(nid) = effective_degree; }
  }

  // Slots past the final effective degree are written with the kInvalidNeighbor sentinel that
  // downstream consumers (HNSW serialization, out-of-range check) recognize as "end of neighbor
  // list". In the constant-degree path effective_degree == output_graph_degree, so no slot is
  // padded.
  for (uint32_t i = lane_id; i < output_graph_degree; i += raft::WarpSize) {
    output_graph(nid_batch, i) =
      (i < effective_degree) ? smem_sorted_output_graph[i] : kInvalidNeighbor<IdxT>;
  }
}

template <class IdxT, class LabelT>
__device__ __host__ LabelT get_root_label(IdxT i, const LabelT* label)
{
  LabelT l = label[i];
  while (l != label[l]) {
    l = label[l];
  }
  return l;
}

template <class IdxT>
__global__ void kern_mst_opt_update_graph(IdxT* mst_graph,  // [graph_size, graph_degree]
                                          const IdxT* candidate_edges,     // [graph_size]
                                          IdxT* outgoing_num_edges,        // [graph_size]
                                          IdxT* incoming_num_edges,        // [graph_size]
                                          const IdxT* outgoing_max_edges,  // [graph_size]
                                          const IdxT* incoming_max_edges,  // [graph_size]
                                          const IdxT* label,               // [graph_size]
                                          const uint32_t graph_size,
                                          const uint32_t graph_degree,
                                          uint64_t* stats)
{
  const uint64_t i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i >= graph_size) return;

  int ret = 0;  // 0: No edge, 1: Direct edge, 2: Alternate edge, 3: Failure

  if (outgoing_num_edges[i] >= outgoing_max_edges[i]) return;
  uint64_t j = candidate_edges[i];
  if (j >= graph_size) return;
  const uint32_t ri = get_root_label(i, label);
  const uint32_t rj = get_root_label(j, label);
  if (ri == rj) return;

  // Try to add a direct edge to destination node with different label.
  if (incoming_num_edges[j] < incoming_max_edges[j]) {
    ret = 1;
    // Check to avoid duplication
    for (uint64_t kj = 0; kj < graph_degree; kj++) {
      uint64_t l = mst_graph[(graph_degree * j) + kj];
      if (l >= graph_size) continue;
      const uint32_t rl = get_root_label(l, label);
      if (ri == rl) {
        ret = 0;
        break;
      }
    }
    if (ret == 0) return;

    ret     = 0;
    auto kj = atomicAdd(incoming_num_edges + j, (IdxT)1);
    if (kj < incoming_max_edges[j]) {
      auto ki                                      = outgoing_num_edges[i];
      outgoing_num_edges[i]                        = ki + 1;
      mst_graph[(graph_degree * (i)) + ki]         = j;  // outgoing
      mst_graph[(graph_degree * (j + 1)) - 1 - kj] = i;  // incoming
      ret                                          = 1;
    }
  }
  if (ret > 0) {
    atomicAdd((unsigned long long int*)stats + ret, 1);
    return;
  }

  // Try to add an edge to an alternate node instead
  ret = 3;
  for (uint64_t kj = 0; kj < graph_degree; kj++) {
    uint64_t l = mst_graph[(graph_degree * (j + 1)) - 1 - kj];
    if (l >= graph_size) continue;
    uint32_t rl = get_root_label(l, label);
    if (ri == rl) {
      ret = 0;
      break;
    }
    if (incoming_num_edges[l] >= incoming_max_edges[l]) continue;

    // Check to avoid duplication
    for (uint64_t kl = 0; kl < graph_degree; kl++) {
      uint64_t m = mst_graph[(graph_degree * l) + kl];
      if (m >= graph_size) continue;
      uint32_t rm = get_root_label(m, label);
      if (ri == rm) {
        ret = 0;
        break;
      }
    }
    if (ret == 0) { break; }

    auto kl = atomicAdd(incoming_num_edges + l, (IdxT)1);
    if (kl < incoming_max_edges[l]) {
      auto ki                                      = outgoing_num_edges[i]++;
      mst_graph[(graph_degree * (i)) + ki]         = l;  // outgoing
      mst_graph[(graph_degree * (l + 1)) - 1 - kl] = i;  // incoming
      ret                                          = 2;
      break;
    }
  }
  if (ret > 0) { atomicAdd((unsigned long long int*)stats + ret, 1); }
}

template <class IdxT>
__global__ void kern_mst_opt_labeling(IdxT* label,            // [graph_size]
                                      const IdxT* mst_graph,  // [graph_size, graph_degree]
                                      const uint32_t graph_size,
                                      const uint32_t graph_degree,
                                      uint64_t* stats)
{
  const uint64_t i = threadIdx.x + (blockDim.x * blockIdx.x);

  __shared__ uint32_t smem_updated[1];
  if (threadIdx.x == 0) { smem_updated[0] = 0; }
  __syncthreads();

  if (i < graph_size) {
    for (uint64_t ki = 0; ki < graph_degree; ki++) {
      uint64_t j = mst_graph[(graph_degree * i) + ki];
      if (j >= graph_size) continue;

      IdxT li = label[i];
      IdxT ri = get_root_label(i, label);
      if (ri < li) { atomicMin(label + i, ri); }
      IdxT lj = label[j];
      IdxT rj = get_root_label(j, label);
      if (rj < lj) { atomicMin(label + j, rj); }
      if (ri == rj) continue;

      if (ri > rj) {
        atomicCAS(label + i, ri, rj);
      } else if (rj > ri) {
        atomicCAS(label + j, rj, ri);
      }
      smem_updated[0] = 1;
    }
  }

  __syncthreads();
  if ((threadIdx.x == 0) && (smem_updated[0] > 0)) { stats[0] = 1; }
}

template <class IdxT>
__global__ void kern_mst_opt_cluster_size(IdxT* cluster_size,  // [graph_size]
                                          const IdxT* label,   // [graph_size]
                                          const uint32_t graph_size,
                                          uint64_t* stats)
{
  const uint64_t i = threadIdx.x + (blockDim.x * blockIdx.x);

  __shared__ uint64_t smem_num_clusters[1];
  if (threadIdx.x == 0) { smem_num_clusters[0] = 0; }
  __syncthreads();

  if (i < graph_size) {
    IdxT ri = get_root_label(i, label);
    if (ri == i) {
      atomicAdd((unsigned long long int*)smem_num_clusters, 1);
    } else {
      atomicAdd(cluster_size + ri, cluster_size[i]);
      cluster_size[i] = 0;
    }
  }

  __syncthreads();
  if ((threadIdx.x == 0) && (smem_num_clusters[0] > 0)) {
    atomicAdd((unsigned long long int*)stats, (unsigned long long int)(smem_num_clusters[0]));
  }
}

template <class IdxT>
__global__ void kern_mst_opt_postprocessing(IdxT* outgoing_num_edges,  // [graph_size]
                                            IdxT* incoming_num_edges,  // [graph_size]
                                            IdxT* outgoing_max_edges,  // [graph_size]
                                            IdxT* incoming_max_edges,  // [graph_size]
                                            const IdxT* cluster_size,  // [graph_size]
                                            const uint32_t graph_size,
                                            const uint32_t graph_degree,
                                            uint64_t* stats)
{
  const uint64_t i = threadIdx.x + (blockDim.x * blockIdx.x);
  __shared__ uint64_t smem_cluster_size_min[1];
  __shared__ uint64_t smem_cluster_size_max[1];
  __shared__ uint64_t smem_total_outgoing_edges[1];
  __shared__ uint64_t smem_total_incoming_edges[1];
  if (threadIdx.x == 0) {
    smem_cluster_size_min[0]     = stats[0];
    smem_cluster_size_max[0]     = stats[1];
    smem_total_outgoing_edges[0] = 0;
    smem_total_incoming_edges[0] = 0;
  }
  __syncthreads();

  if (i < graph_size) {
    // Adjust incoming_num_edges
    if (incoming_num_edges[i] > incoming_max_edges[i]) {
      incoming_num_edges[i] = incoming_max_edges[i];
    }

    // Calculate min/max of cluster_size
    if (cluster_size[i] > 0) {
      if (smem_cluster_size_min[0] > cluster_size[i]) {
        atomicMin((unsigned long long int*)smem_cluster_size_min,
                  (unsigned long long int)(cluster_size[i]));
      }
      if (smem_cluster_size_max[0] < cluster_size[i]) {
        atomicMax((unsigned long long int*)smem_cluster_size_max,
                  (unsigned long long int)(cluster_size[i]));
      }
    }

    // Calculate total number of outgoing/incoming edges
    atomicAdd((unsigned long long int*)smem_total_outgoing_edges,
              (unsigned long long int)(outgoing_num_edges[i]));
    atomicAdd((unsigned long long int*)smem_total_incoming_edges,
              (unsigned long long int)(incoming_num_edges[i]));

    // Adjust incoming/outgoing_max_edges
    if (outgoing_num_edges[i] == outgoing_max_edges[i]) {
      if (outgoing_num_edges[i] + incoming_num_edges[i] < graph_degree) {
        outgoing_max_edges[i] += 1;
        incoming_max_edges[i] -= 1;
      }
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    atomicMin((unsigned long long int*)stats + 0,
              (unsigned long long int)(smem_cluster_size_min[0]));
    atomicMax((unsigned long long int*)stats + 1,
              (unsigned long long int)(smem_cluster_size_max[0]));
    atomicAdd((unsigned long long int*)stats + 2,
              (unsigned long long int)(smem_total_outgoing_edges[0]));
    atomicAdd((unsigned long long int*)stats + 3,
              (unsigned long long int)(smem_total_incoming_edges[0]));
  }
}

template <typename IdxT, typename AccessorOutputGraph>
void log_incoming_edges_histogram(
  raft::resources const& res,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorOutputGraph>
    output_graph)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> block_scope(
    "cagra::graph::optimize/check_edges");
  const uint64_t graph_size          = output_graph.extent(0);
  const uint64_t output_graph_degree = output_graph.extent(1);

  // copy to host if required
  IdxT* output_graph_ptr = nullptr;
  int64_t buffer_size    = AccessorOutputGraph::is_host_accessible ? graph_size : 0;
  auto host_copy_output_graph =
    raft::make_host_matrix<IdxT, int64_t>(buffer_size, output_graph_degree);
  if constexpr (AccessorOutputGraph::is_host_accessible) {
    output_graph_ptr = output_graph.data_handle();
  } else {
    raft::copy(res, host_copy_output_graph.view(), output_graph);
    output_graph_ptr = host_copy_output_graph.data_handle();
  }

  auto in_edge_count     = raft::make_host_vector<uint32_t, int64_t>(graph_size);
  auto in_edge_count_ptr = in_edge_count.data_handle();
#pragma omp parallel for
  for (uint64_t i = 0; i < graph_size; i++) {
    in_edge_count_ptr[i] = 0;
  }
#pragma omp parallel for
  for (uint64_t i = 0; i < graph_size; i++) {
    for (uint64_t k = 0; k < output_graph_degree; k++) {
      const uint64_t j = output_graph_ptr[k + (output_graph_degree * i)];
      if (j >= graph_size) continue;
#pragma omp atomic
      in_edge_count_ptr[j] += 1;
    }
  }
  auto hist     = raft::make_host_vector<uint32_t, int64_t>(output_graph_degree);
  auto hist_ptr = hist.data_handle();
  for (uint64_t k = 0; k < output_graph_degree; k++) {
    hist_ptr[k] = 0;
  }
#pragma omp parallel for
  for (uint64_t i = 0; i < graph_size; i++) {
    uint32_t count = in_edge_count_ptr[i];
    if (count >= output_graph_degree) continue;
#pragma omp atomic
    hist_ptr[count] += 1;
  }
  RAFT_LOG_DEBUG("# Histogram for number of incoming edges\n");
  uint32_t sum_hist = 0;
  for (uint64_t k = 0; k < output_graph_degree; k++) {
    sum_hist += hist_ptr[k];
    RAFT_LOG_DEBUG("# %3lu, %8u, %lf, (%8u, %lf)\n",
                   k,
                   hist_ptr[k],
                   (double)hist_ptr[k] / graph_size,
                   sum_hist,
                   (double)sum_hist / graph_size);
  }
}

template <typename IdxT, typename AccessorOutputGraph>
void check_duplicates_and_out_of_range(
  raft::resources const& res,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorOutputGraph>
    output_graph)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> block_scope(
    "cagra::graph::optimize/check_duplicates");

  const uint64_t graph_size          = output_graph.extent(0);
  const uint64_t output_graph_degree = output_graph.extent(1);

  // copy to host if required
  IdxT* output_graph_ptr = nullptr;
  int64_t buffer_size    = AccessorOutputGraph::is_host_accessible ? graph_size : 0;
  auto host_copy_output_graph =
    raft::make_host_matrix<IdxT, int64_t>(buffer_size, output_graph_degree);
  if constexpr (AccessorOutputGraph::is_host_accessible) {
    output_graph_ptr = output_graph.data_handle();
  } else {
    raft::copy(res, host_copy_output_graph.view(), output_graph);
    output_graph_ptr = host_copy_output_graph.data_handle();
  }

  uint64_t num_dup = 0;
  uint64_t num_oor = 0;
#pragma omp parallel for reduction(+ : num_dup) reduction(+ : num_oor)
  for (uint64_t i = 0; i < graph_size; i++) {
    auto my_out_graph = output_graph_ptr + (output_graph_degree * i);
    for (uint32_t j = 0; j < output_graph_degree; j++) {
      const auto neighbor_a = my_out_graph[j];

      // The variable-degree path pads unused slots with kInvalidNeighbor. Without this guard the
      // sentinel would (correctly) trigger the `>= graph_size` out-of-range check below.
      if (neighbor_a == kInvalidNeighbor<IdxT>) { continue; }

      if (neighbor_a >= graph_size) {
        num_oor++;
        continue;
      }

      for (uint32_t k = j + 1; k < output_graph_degree; k++) {
        const auto neighbor_b = my_out_graph[k];
        if (neighbor_a == neighbor_b) {
          num_dup++;
          break;
        }
      }
    }
  }
  RAFT_EXPECTS(
    num_dup == 0, "%lu duplicated node(s) are found in the generated CAGRA graph", num_dup);
  RAFT_EXPECTS(
    num_oor == 0, "%lu out-of-range index node(s) are found in the generated CAGRA graph", num_oor);
}

template <typename IdxT, typename AccessorOutputGraph, bool VariableDegree>
void merge_graph_gpu(
  raft::resources const& res,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorOutputGraph>
    output_graph,
  raft::device_matrix_view<IdxT, int64_t> d_rev_graph,
  raft::device_vector_view<uint32_t, int64_t> d_rev_graph_count,
  raft::host_matrix_view<IdxT, int64_t> mst_graph,
  raft::host_vector_view<uint32_t, int64_t> mst_graph_num_edges,
  bool guarantee_connectivity,
  raft::device_vector_view<uint32_t, int64_t> d_natural_degree)
{
  const uint64_t graph_size          = output_graph.extent(0);
  const uint64_t output_graph_degree = output_graph.extent(1);

  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> block_scope(
    "cagra::graph::optimize/combine");

  const double merge_graph_start = cur_time();

  auto d_check_num_protected_edges = raft::make_device_scalar<uint32_t>(res, 1u);

  uint32_t batch_size =
    static_cast<uint32_t>(std::min<uint64_t>(graph_size, helpers::kOptimizeBatchSize));
  const uint32_t num_batch = raft::div_rounding_up_safe<uint64_t>(graph_size, batch_size);

  namespace bli                       = cuvs::spatial::knn::detail::utils;
  auto [copy_stream, enable_prefetch] = bli::get_prefetch_stream(res);
  auto workspace_mr                   = raft::resource::get_workspace_resource_ref(res);

  bli::batch_load_iterator<
    raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorOutputGraph>>
    d_output_graph(res,
                   output_graph,
                   batch_size,
                   copy_stream,
                   workspace_mr,
                   enable_prefetch,
                   /*initialize=*/true,
                   /*host_writeback=*/true);

  bli::batch_load_iterator<raft::host_matrix_view<IdxT, int64_t>> d_mst_graph(
    res, mst_graph, batch_size, copy_stream, workspace_mr, enable_prefetch);

  bli::batch_load_iterator<raft::host_matrix_view<uint32_t, int64_t>> d_mst_graph_num_edges(
    res,
    raft::make_host_matrix_view<uint32_t, int64_t>(
      mst_graph_num_edges.data_handle(), mst_graph_num_edges.extent(0), 1l),
    batch_size,
    copy_stream,
    workspace_mr,
    enable_prefetch);

  d_output_graph.prefetch_next_batch();
  d_mst_graph.prefetch_next_batch();
  d_mst_graph_num_edges.prefetch_next_batch();

  const uint32_t num_warps = 4;
  const dim3 threads_merge(raft::WarpSize * num_warps, 1, 1);
  const dim3 blocks_merge(raft::ceildiv(batch_size, num_warps), 1, 1);
  const size_t merge_smem_size = num_warps * output_graph_degree * sizeof(IdxT);
  for (uint32_t i_batch = 0; i_batch < num_batch; i_batch++) {
    auto mst_graph_view           = (*d_mst_graph).view();
    auto mst_graph_num_edges_view = (*d_mst_graph_num_edges).view();
    auto output_view              = (*d_output_graph).view();
    kern_merge_graph<IdxT, num_warps, VariableDegree>
      <<<blocks_merge, threads_merge, merge_smem_size, raft::resource::get_cuda_stream(res)>>>(
        output_view,
        d_rev_graph,
        d_rev_graph_count,
        mst_graph_view,
        mst_graph_num_edges_view,
        batch_size,
        i_batch,
        guarantee_connectivity,
        d_check_num_protected_edges.view(),
        d_natural_degree);

    d_output_graph.prefetch_next_batch();
    d_mst_graph.prefetch_next_batch();
    d_mst_graph_num_edges.prefetch_next_batch();
    ++d_output_graph;
    ++d_mst_graph;
    ++d_mst_graph_num_edges;
  }

  uint32_t check_num_protected_edges = 1u;
  raft::copy(res,
             raft::make_host_scalar_view(&check_num_protected_edges),
             d_check_num_protected_edges.view());
  raft::resource::sync_stream(res);
  const auto merge_graph_end = cur_time();
  RAFT_EXPECTS(check_num_protected_edges,
               "Failed to merge the MST, pruned, and reverse edge graphs. "
               "Some nodes have too "
               "many MST optimization edges.");

  RAFT_LOG_DEBUG("# Time for merging graphs: %.1lf ms",
                 (merge_graph_end - merge_graph_start) * 1000.0);
}

template <typename IdxT, typename AccessorOutputGraph>
void make_reverse_graph_gpu(
  raft::resources const& res,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorOutputGraph>
    output_graph,
  raft::device_matrix_view<IdxT, int64_t> d_rev_graph,
  raft::device_vector_view<uint32_t, int64_t> d_rev_graph_count)
{
  const uint64_t graph_size          = output_graph.extent(0);
  const uint64_t output_graph_degree = output_graph.extent(1);

  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> block_scope(
    "cagra::graph::optimize/reverse");

  //
  // Make reverse graph
  //
  raft::matrix::fill(res, d_rev_graph, kInvalidNeighbor<IdxT>);
  raft::matrix::fill(res, d_rev_graph_count, uint32_t(0));

  if constexpr (AccessorOutputGraph::is_device_accessible) {
    // output graph is fully device accessible, so we need no copy to device
    dim3 threads(256, 1, 1);
    dim3 blocks(1024, 1, 1);
    for (uint64_t k = 0; k < output_graph_degree; k++) {
      kern_make_rev_graph_k<IdxT><<<blocks, threads, 0, raft::resource::get_cuda_stream(res)>>>(
        output_graph, d_rev_graph, d_rev_graph_count, k);
    }
  } else {
    // Host variant: the output graph is host-only, so we extract one column at a time into a
    // device vector and hand that to the kernel. The natural-degree mask is applied inside the
    // kernel using the real column index `k` (the kernel indexes the 1D `d_dest_nodes` by row).
    auto d_dest_nodes = raft::make_device_vector<IdxT, int64_t>(res, graph_size);
    auto dest_nodes   = raft::make_host_vector<IdxT, int64_t>(res, graph_size);

    for (uint64_t k = 0; k < output_graph_degree; k++) {
#pragma omp parallel for
      for (uint64_t i = 0; i < graph_size; i++) {
        dest_nodes(i) = output_graph(i, k);
      }
      raft::copy(res, d_dest_nodes.view(), raft::make_const_mdspan(dest_nodes.view()));

      dim3 threads(256, 1, 1);
      dim3 blocks(1024, 1, 1);
      kern_make_rev_graph_k<IdxT><<<blocks, threads, 0, raft::resource::get_cuda_stream(res)>>>(
        d_dest_nodes.view(), d_rev_graph, d_rev_graph_count, k);
      raft::resource::sync_stream(res);
      RAFT_LOG_DEBUG("# Making reverse graph on GPUs: %lu / %u    \r", k, output_graph_degree);
    }
  }
}

template <typename DataT,
          typename IdxT       = uint32_t,
          typename d_accessor = raft::host_device_accessor<cuda::std::default_accessor<DataT>,
                                                           raft::memory_type::device>,
          typename g_accessor =
            raft::host_device_accessor<cuda::std::default_accessor<IdxT>, raft::memory_type::host>>
void sort_knn_graph(
  raft::resources const& res,
  const cuvs::distance::DistanceType metric,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, d_accessor> dataset,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, g_accessor> knn_graph)
{
  RAFT_EXPECTS(dataset.extent(0) == knn_graph.extent(0),
               "dataset size is expected to have the same number of graph index size");
  RAFT_EXPECTS(
    metric == cuvs::distance::DistanceType::InnerProduct ||
      metric == cuvs::distance::DistanceType::CosineExpanded ||
      metric == cuvs::distance::DistanceType::L2Expanded ||
      metric == cuvs::distance::DistanceType::BitwiseHamming ||
      metric == cuvs::distance::DistanceType::L1,
    "Unsupported metric. Only InnerProduct, CosineExpanded, L2Expanded, BitwiseHamming and L1 are "
    "supported");
  const uint64_t dataset_size = dataset.extent(0);
  const uint64_t dataset_dim  = dataset.extent(1);
  const DataT* dataset_ptr    = dataset.data_handle();

  const IdxT graph_size             = dataset_size;
  const uint64_t input_graph_degree = knn_graph.extent(1);
  IdxT* const input_graph_ptr       = knn_graph.data_handle();

  auto large_tmp_mr = raft::resource::get_large_workspace_resource_ref(res);

  auto d_input_graph = raft::make_device_mdarray<IdxT>(
    res, large_tmp_mr, raft::make_extents<int64_t>(graph_size, input_graph_degree));

  //
  // Sorting kNN graph
  //
  const double time_sort_start = cur_time();
  RAFT_LOG_DEBUG("# Sorting kNN Graph on GPUs ");

  auto d_dataset = raft::make_device_mdarray<DataT>(
    res, large_tmp_mr, raft::make_extents<int64_t>(dataset_size, dataset_dim));
  raft::copy(res, d_dataset.view(), dataset);

  raft::copy(res, d_input_graph.view(), knn_graph);

  void (*kernel_sort)(const DataT* const,
                      const IdxT,
                      const uint32_t,
                      IdxT* const,
                      const uint32_t,
                      const uint32_t,
                      const cuvs::distance::DistanceType);
  if (input_graph_degree <= 32) {
    constexpr int numElementsPerThread = 1;
    kernel_sort                        = kern_sort<DataT, IdxT, numElementsPerThread>;
  } else if (input_graph_degree <= 64) {
    constexpr int numElementsPerThread = 2;
    kernel_sort                        = kern_sort<DataT, IdxT, numElementsPerThread>;
  } else if (input_graph_degree <= 128) {
    constexpr int numElementsPerThread = 4;
    kernel_sort                        = kern_sort<DataT, IdxT, numElementsPerThread>;
  } else if (input_graph_degree <= 256) {
    constexpr int numElementsPerThread = 8;
    kernel_sort                        = kern_sort<DataT, IdxT, numElementsPerThread>;
  } else if (input_graph_degree <= 512) {
    constexpr int numElementsPerThread = 16;
    kernel_sort                        = kern_sort<DataT, IdxT, numElementsPerThread>;
  } else if (input_graph_degree <= 1024) {
    constexpr int numElementsPerThread = 32;
    kernel_sort                        = kern_sort<DataT, IdxT, numElementsPerThread>;
  } else {
    RAFT_FAIL(
      "The degree of input knn graph is too large (%lu). "
      "It must be equal to or smaller than %d.",
      input_graph_degree,
      1024);
  }
  const auto block_size          = 256;
  const auto num_warps_per_block = block_size / raft::WarpSize;
  const auto grid_size           = (graph_size + num_warps_per_block - 1) / num_warps_per_block;

  RAFT_LOG_DEBUG(".");
  kernel_sort<<<grid_size, block_size, 0, raft::resource::get_cuda_stream(res)>>>(
    d_dataset.data_handle(),
    dataset_size,
    dataset_dim,
    d_input_graph.data_handle(),
    graph_size,
    input_graph_degree,
    metric);
  raft::resource::sync_stream(res);
  RAFT_LOG_DEBUG(".");
  raft::copy(res, knn_graph, raft::make_const_mdspan(d_input_graph.view()));
  RAFT_LOG_DEBUG("\n");

  const double time_sort_end = cur_time();
  RAFT_LOG_DEBUG("# Sorting kNN graph time: %.1lf sec\n", time_sort_end - time_sort_start);
}

template <typename IdxT = uint32_t>
void mst_opt_update_graph(IdxT* mst_graph_ptr,
                          IdxT* candidate_edges_ptr,
                          IdxT* outgoing_num_edges_ptr,
                          IdxT* incoming_num_edges_ptr,
                          IdxT* outgoing_max_edges_ptr,
                          IdxT* incoming_max_edges_ptr,
                          IdxT* label_ptr,
                          IdxT graph_size,
                          uint32_t mst_graph_degree,
                          uint64_t k,
                          int& num_direct,
                          int& num_alternate,
                          int& num_failure)
{
#pragma omp parallel for reduction(+ : num_direct, num_alternate, num_failure)
  for (uint64_t ii = 0; ii < graph_size; ii++) {
    uint64_t i = ii;
    if (k % 2 == 0) { i = graph_size - (ii + 1); }
    int ret = 0;  // 0: No edge, 1: Direct edge, 2: Alternate edge, 3: Failure

    if (outgoing_num_edges_ptr[i] >= outgoing_max_edges_ptr[i]) continue;
    uint64_t j = candidate_edges_ptr[i];
    if (j >= graph_size) continue;
    if (label_ptr[i] == label_ptr[j]) continue;

    // Try to add a direct edge to destination node with different label.
    if (incoming_num_edges_ptr[j] < incoming_max_edges_ptr[j]) {
      ret = 1;
      // Check to avoid duplication
      for (uint64_t kj = 0; kj < mst_graph_degree; kj++) {
        uint64_t l = mst_graph_ptr[(mst_graph_degree * j) + kj];
        if (l >= graph_size) continue;
        if (label_ptr[i] == label_ptr[l]) {
          ret = 0;
          break;
        }
      }
      if (ret == 0) continue;

      // Use atomic to avoid conflicts, since 'incoming_num_edges_ptr[j]'
      // can be updated by other threads.
      ret = 0;
      uint32_t kj;
#pragma omp atomic capture
      kj = incoming_num_edges_ptr[j]++;
      if (kj < incoming_max_edges_ptr[j]) {
        auto ki                                              = outgoing_num_edges_ptr[i]++;
        mst_graph_ptr[(mst_graph_degree * (i)) + ki]         = j;  // OUT
        mst_graph_ptr[(mst_graph_degree * (j + 1)) - 1 - kj] = i;  // IN
        ret                                                  = 1;
      }
    }
    if (ret == 1) {
      num_direct += 1;
      continue;
    }

    // Try to add an edge to an alternate node instead
    ret = 3;
    for (uint64_t kj = 0; kj < mst_graph_degree; kj++) {
      uint64_t l = mst_graph_ptr[(mst_graph_degree * (j + 1)) - 1 - kj];
      if (l >= graph_size) continue;
      if (label_ptr[i] == label_ptr[l]) {
        ret = 0;
        break;
      }
      if (incoming_num_edges_ptr[l] >= incoming_max_edges_ptr[l]) continue;

      // Check to avoid duplication
      for (uint64_t kl = 0; kl < mst_graph_degree; kl++) {
        uint64_t m = mst_graph_ptr[(mst_graph_degree * l) + kl];
        if (m >= graph_size) continue;
        if (label_ptr[i] == label_ptr[m]) {
          ret = 0;
          break;
        }
      }
      if (ret == 0) { break; }

      // Use atomic to avoid conflicts, since 'incoming_num_edges_ptr[l]'
      // can be updated by other threads.
      uint32_t kl;
#pragma omp atomic capture
      kl = incoming_num_edges_ptr[l]++;
      if (kl < incoming_max_edges_ptr[l]) {
        auto ki                                              = outgoing_num_edges_ptr[i]++;
        mst_graph_ptr[(mst_graph_degree * (i)) + ki]         = l;  // OUT
        mst_graph_ptr[(mst_graph_degree * (l + 1)) - 1 - kl] = i;  // IN
        ret                                                  = 2;
        break;
      }
    }
    if (ret == 2) {
      num_alternate += 1;
    } else if (ret == 3) {
      num_failure += 1;
    }
  }
}

//
// Create approximate MSTs with kNN graphs as input to guarantee connectivity of search graphs
//
// * Since there is an upper limit to the degree of a graph for search, what is created is a
//   degree-constraied MST.
// * The number of edges is not a minimum because strict MST is not required. Therefore, it is
//   an approximate MST.
// * If the input kNN graph is disconnected, random connection is added to the largest cluster.
//
template <typename IdxT, typename AccessorKnnGraph>
void mst_optimization(
  raft::resources const& res,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorKnnGraph> input_graph,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> output_graph,
  raft::host_vector_view<uint32_t, int64_t> mst_graph_num_edges,
  bool use_gpu = true)
{
  if (use_gpu) {
    RAFT_LOG_DEBUG("# MST optimization on GPU");
  } else {
    RAFT_LOG_DEBUG("# MST optimization on CPU");
  }
  const double time_mst_opt_start = cur_time();

  const IdxT graph_size              = input_graph.extent(0);
  const uint32_t input_graph_degree  = input_graph.extent(1);
  const uint32_t output_graph_degree = output_graph.extent(1);

  // Allocate temporal arrays
  const uint32_t mst_graph_degree = output_graph_degree;
  auto mst_graph              = raft::make_host_matrix<IdxT, int64_t>(graph_size, mst_graph_degree);
  auto outgoing_max_edges     = raft::make_host_vector<IdxT, int64_t>(graph_size);
  auto incoming_max_edges     = raft::make_host_vector<IdxT, int64_t>(graph_size);
  auto outgoing_num_edges     = raft::make_host_vector<IdxT, int64_t>(graph_size);
  auto incoming_num_edges     = raft::make_host_vector<IdxT, int64_t>(graph_size);
  auto label                  = raft::make_host_vector<IdxT, int64_t>(graph_size);
  auto cluster_size           = raft::make_host_vector<IdxT, int64_t>(graph_size);
  auto candidate_edges        = raft::make_host_vector<IdxT, int64_t>(graph_size);
  auto mst_graph_ptr          = mst_graph.data_handle();
  auto outgoing_max_edges_ptr = outgoing_max_edges.data_handle();
  auto incoming_max_edges_ptr = incoming_max_edges.data_handle();
  auto outgoing_num_edges_ptr = outgoing_num_edges.data_handle();
  auto incoming_num_edges_ptr = incoming_num_edges.data_handle();
  auto label_ptr              = label.data_handle();
  auto cluster_size_ptr       = cluster_size.data_handle();
  auto candidate_edges_ptr    = candidate_edges.data_handle();

  // Initialize arrays
#pragma omp parallel for
  for (uint64_t i = 0; i < graph_size; i++) {
    for (uint64_t k = 0; k < mst_graph_degree; k++) {
      // mst_graph_ptr[(mst_graph_degree * i) + k] = graph_size;
      mst_graph(i, k) = graph_size;
    }
    outgoing_max_edges_ptr[i] = 2;
    incoming_max_edges_ptr[i] = mst_graph_degree - outgoing_max_edges_ptr[i];
    outgoing_num_edges_ptr[i] = 0;
    incoming_num_edges_ptr[i] = 0;
    label_ptr[i]              = i;
    cluster_size_ptr[i]       = 1;
  }

  // Allocate arrays on GPU
  uint32_t d_graph_size = graph_size;
  if (!use_gpu) {
    // (*) If GPU is not used, arrays of size 0 are created.
    d_graph_size = 0;
  }
  auto d_mst_graph_num_edges = raft::make_device_vector<IdxT, int64_t>(res, d_graph_size);
  auto d_mst_graph = raft::make_device_matrix<IdxT, int64_t>(res, d_graph_size, mst_graph_degree);
  auto d_outgoing_max_edges      = raft::make_device_vector<IdxT, int64_t>(res, d_graph_size);
  auto d_incoming_max_edges      = raft::make_device_vector<IdxT, int64_t>(res, d_graph_size);
  auto d_outgoing_num_edges      = raft::make_device_vector<IdxT, int64_t>(res, d_graph_size);
  auto d_incoming_num_edges      = raft::make_device_vector<IdxT, int64_t>(res, d_graph_size);
  auto d_label                   = raft::make_device_vector<IdxT, int64_t>(res, d_graph_size);
  auto d_cluster_size            = raft::make_device_vector<IdxT, int64_t>(res, d_graph_size);
  auto d_candidate_edges         = raft::make_device_vector<IdxT, int64_t>(res, d_graph_size);
  auto d_mst_graph_num_edges_ptr = d_mst_graph_num_edges.data_handle();
  auto d_mst_graph_ptr           = d_mst_graph.data_handle();
  auto d_outgoing_max_edges_ptr  = d_outgoing_max_edges.data_handle();
  auto d_incoming_max_edges_ptr  = d_incoming_max_edges.data_handle();
  auto d_outgoing_num_edges_ptr  = d_outgoing_num_edges.data_handle();
  auto d_incoming_num_edges_ptr  = d_incoming_num_edges.data_handle();
  auto d_label_ptr               = d_label.data_handle();
  auto d_cluster_size_ptr        = d_cluster_size.data_handle();
  auto d_candidate_edges_ptr     = d_candidate_edges.data_handle();

  constexpr int stats_size = 4;
  auto stats               = raft::make_host_vector<uint64_t, int64_t>(stats_size);
  auto d_stats             = raft::make_device_vector<uint64_t, int64_t>(res, stats_size);
  auto stats_ptr           = stats.data_handle();
  auto d_stats_ptr         = d_stats.data_handle();

  if (use_gpu) {
    raft::copy(res, d_mst_graph.view(), raft::make_const_mdspan(mst_graph.view()));
    raft::copy(
      res, d_outgoing_num_edges.view(), raft::make_const_mdspan(outgoing_num_edges.view()));
    raft::copy(
      res, d_incoming_num_edges.view(), raft::make_const_mdspan(incoming_num_edges.view()));
    raft::copy(
      res, d_outgoing_max_edges.view(), raft::make_const_mdspan(outgoing_max_edges.view()));
    raft::copy(
      res, d_incoming_max_edges.view(), raft::make_const_mdspan(incoming_max_edges.view()));
    raft::copy(res, d_label.view(), raft::make_const_mdspan(label.view()));
    raft::copy(res, d_cluster_size.view(), raft::make_const_mdspan(cluster_size.view()));
  }

  IdxT num_clusters     = 0;
  IdxT num_clusters_pre = graph_size;
  IdxT cluster_size_min = graph_size;
  IdxT cluster_size_max = 0;
  for (uint64_t k = 0; k <= input_graph_degree; k++) {
    int num_direct    = 0;
    int num_alternate = 0;
    int num_failure   = 0;

    // 1. Prepare candidate edges
    if (k == input_graph_degree) {
      // If the number of clusters does not converge to 1, then edges are
      // made from all nodes not belonging to the main cluster to any node
      // in the main cluster.
      raft::copy(res, cluster_size.view(), raft::make_const_mdspan(d_cluster_size.view()));
      raft::copy(res, label.view(), raft::make_const_mdspan(d_label.view()));
      raft::resource::sync_stream(res);
      uint32_t main_cluster_label = graph_size;
#pragma omp parallel for reduction(min : main_cluster_label)
      for (uint64_t i = 0; i < graph_size; i++) {
        if ((cluster_size_ptr[i] == cluster_size_max) && (main_cluster_label > i)) {
          main_cluster_label = i;
        }
      }
#pragma omp parallel for
      for (uint64_t i = 0; i < graph_size; i++) {
        candidate_edges_ptr[i] = graph_size;
        if (label_ptr[i] == main_cluster_label) continue;
        uint64_t j = i;
        while (label_ptr[j] != main_cluster_label) {
          constexpr uint32_t ofst = 97;
          j                       = (j + ofst) % graph_size;
        }
        candidate_edges_ptr[i] = j;
      }
    } else {
      // Copy rank-k edges from the input knn graph to 'candidate_edges'
      if constexpr (AccessorKnnGraph::is_host_accessible) {
#pragma omp parallel for
        for (uint64_t i = 0; i < graph_size; i++) {
          candidate_edges_ptr[i] = input_graph(i, k);
        }
      } else {
        // handle device knn graph
        RAFT_CUDA_TRY(cudaMemcpy2DAsync(candidate_edges_ptr,
                                        sizeof(IdxT),
                                        &input_graph(0, k),
                                        input_graph_degree * sizeof(IdxT),
                                        1 * sizeof(IdxT),  // width
                                        graph_size,
                                        cudaMemcpyDeviceToHost,
                                        raft::resource::get_cuda_stream(res)));
        raft::resource::sync_stream(res);

        // FIXME: use submdspan and raft::copy once supported
        /*auto column_view = cuda::std::submdspan(input_graph,
                                                  cuda::std::tuple{uint64_t(0),
        uint64_t(graph_size)}, cuda::std::tuple{uint64_t(k), uint64_t(k+1)}); raft::copy(res,
        candidate_edges.view(), raft::make_const_mdspan(column_view));*/
      }
    }

    // 2. Update MST graph
    //  * Try to add candidate edges to MST graph
    if (use_gpu) {
      raft::copy(res, d_candidate_edges.view(), raft::make_const_mdspan(candidate_edges.view()));
      stats_ptr[0] = 0;
      stats_ptr[1] = num_direct;
      stats_ptr[2] = num_alternate;
      stats_ptr[3] = num_failure;
      raft::copy(res, d_stats.view(), raft::make_const_mdspan(stats.view()));

      constexpr uint64_t n_threads = 256;
      const dim3 threads(n_threads, 1, 1);
      const dim3 blocks(raft::ceildiv<uint64_t>(graph_size, n_threads), 1, 1);
      kern_mst_opt_update_graph<<<blocks, threads, 0, raft::resource::get_cuda_stream(res)>>>(
        d_mst_graph_ptr,
        d_candidate_edges_ptr,
        d_outgoing_num_edges_ptr,
        d_incoming_num_edges_ptr,
        d_outgoing_max_edges_ptr,
        d_incoming_max_edges_ptr,
        d_label_ptr,
        graph_size,
        mst_graph_degree,
        d_stats_ptr);

      raft::copy(res, stats.view(), raft::make_const_mdspan(d_stats.view()));
      raft::resource::sync_stream(res);
      num_direct    = stats_ptr[1];
      num_alternate = stats_ptr[2];
      num_failure   = stats_ptr[3];
    } else {
      mst_opt_update_graph(mst_graph_ptr,
                           candidate_edges_ptr,
                           outgoing_num_edges_ptr,
                           incoming_num_edges_ptr,
                           outgoing_max_edges_ptr,
                           incoming_max_edges_ptr,
                           label_ptr,
                           graph_size,
                           mst_graph_degree,
                           k,
                           num_direct,
                           num_alternate,
                           num_failure);
    }

    // 3. Labeling
    uint32_t flag_update = 1;
    while (flag_update) {
      flag_update = 0;
      if (use_gpu) {
        stats_ptr[0] = flag_update;
        raft::copy(res,
                   raft::make_device_vector_view(d_stats_ptr, int64_t(1)),
                   raft::make_host_vector_view(stats_ptr, int64_t(1)));

        constexpr uint64_t n_threads = 256;
        const dim3 threads(n_threads, 1, 1);
        const dim3 blocks((graph_size + n_threads - 1) / n_threads, 1, 1);
        kern_mst_opt_labeling<<<blocks, threads, 0, raft::resource::get_cuda_stream(res)>>>(
          d_label_ptr, d_mst_graph_ptr, graph_size, mst_graph_degree, d_stats_ptr);

        raft::copy(res,
                   raft::make_host_vector_view(stats_ptr, int64_t(1)),
                   raft::make_device_vector_view(d_stats_ptr, int64_t(1)));
        raft::resource::sync_stream(res);
        flag_update = stats_ptr[0];
      } else {
#pragma omp parallel for reduction(+ : flag_update)
        for (uint64_t i = 0; i < graph_size; i++) {
          for (uint64_t ki = 0; ki < mst_graph_degree; ki++) {
            uint64_t j = mst_graph_ptr[(mst_graph_degree * i) + ki];
            if (j >= graph_size) continue;
            if (label_ptr[i] > label_ptr[j]) {
              flag_update += 1;
              label_ptr[i] = label_ptr[j];
            }
          }
        }
      }
    }

    // 4. Calculate the number of clusters and the size of each cluster
    num_clusters = 0;
    if (use_gpu) {
      stats_ptr[0] = num_clusters;
      raft::copy(res,
                 raft::make_device_vector_view(d_stats_ptr, int64_t(1)),
                 raft::make_host_vector_view(stats_ptr, int64_t(1)));

      constexpr uint64_t n_threads = 256;
      const dim3 threads(n_threads, 1, 1);
      const dim3 blocks(raft::ceildiv<uint64_t>(graph_size, n_threads), 1, 1);
      kern_mst_opt_cluster_size<<<blocks, threads, 0, raft::resource::get_cuda_stream(res)>>>(
        d_cluster_size_ptr, d_label_ptr, graph_size, d_stats_ptr);

      raft::copy(res,
                 raft::make_host_vector_view(stats_ptr, int64_t(1)),
                 raft::make_device_vector_view(d_stats_ptr, int64_t(1)));
      raft::resource::sync_stream(res);
      num_clusters = stats_ptr[0];
    } else {
#pragma omp parallel for reduction(+ : num_clusters)
      for (uint64_t i = 0; i < graph_size; i++) {
        uint64_t ri = get_root_label(i, label_ptr);
        if (ri == i) {
          num_clusters += 1;
        } else {
#pragma omp atomic update
          cluster_size_ptr[ri] += cluster_size_ptr[i];
          cluster_size_ptr[i] = 0;
        }
      }
    }

    // 5. Postprocessings
    //  * Adjust incoming_num_edges
    //  * Calculate the min/max size of clusters.
    //  * Calculate the total number of outgoing/incoming edges
    //  * Increase the limit of outgoing edges as needed
    cluster_size_min              = graph_size;
    cluster_size_max              = 0;
    uint64_t total_outgoing_edges = 0;
    uint64_t total_incoming_edges = 0;
    if (use_gpu) {
      stats_ptr[0] = cluster_size_min;
      stats_ptr[1] = cluster_size_max;
      stats_ptr[2] = total_outgoing_edges;
      stats_ptr[3] = total_incoming_edges;
      raft::copy(res, d_stats.view(), raft::make_const_mdspan(stats.view()));

      constexpr uint64_t n_threads = 256;
      const dim3 threads(n_threads, 1, 1);
      const dim3 blocks((graph_size + n_threads - 1) / n_threads, 1, 1);
      kern_mst_opt_postprocessing<<<blocks, threads, 0, raft::resource::get_cuda_stream(res)>>>(
        d_outgoing_num_edges_ptr,
        d_incoming_num_edges_ptr,
        d_outgoing_max_edges_ptr,
        d_incoming_max_edges_ptr,
        d_cluster_size_ptr,
        graph_size,
        mst_graph_degree,
        d_stats_ptr);

      raft::copy(res, stats.view(), raft::make_const_mdspan(d_stats.view()));
      raft::resource::sync_stream(res);
      cluster_size_min     = stats_ptr[0];
      cluster_size_max     = stats_ptr[1];
      total_outgoing_edges = stats_ptr[2];
      total_incoming_edges = stats_ptr[3];
    } else {
#pragma omp parallel for
      for (uint64_t i = 0; i < graph_size; i++) {
        if (incoming_num_edges_ptr[i] > incoming_max_edges_ptr[i]) {
          incoming_num_edges_ptr[i] = incoming_max_edges_ptr[i];
        }
      }

#pragma omp parallel for reduction(max : cluster_size_max) reduction(min : cluster_size_min)
      for (uint64_t i = 0; i < graph_size; i++) {
        if (cluster_size_ptr[i] == 0) continue;
        cluster_size_min = min(cluster_size_min, cluster_size_ptr[i]);
        cluster_size_max = max(cluster_size_max, cluster_size_ptr[i]);
      }

#pragma omp parallel for reduction(+ : total_outgoing_edges, total_incoming_edges)
      for (uint64_t i = 0; i < graph_size; i++) {
        total_outgoing_edges += outgoing_num_edges_ptr[i];
        total_incoming_edges += incoming_num_edges_ptr[i];
      }

      bool check_num_mst_edges = true;
#pragma omp parallel for
      for (uint64_t i = 0; i < graph_size; i++) {
        if (outgoing_num_edges_ptr[i] < outgoing_max_edges_ptr[i]) continue;
        if (outgoing_num_edges_ptr[i] + incoming_num_edges_ptr[i] == mst_graph_degree) continue;
        if (outgoing_num_edges_ptr[i] + incoming_num_edges_ptr[i] > mst_graph_degree) {
          check_num_mst_edges = false;
        }
        outgoing_max_edges_ptr[i] += 1;
        incoming_max_edges_ptr[i] = mst_graph_degree - outgoing_max_edges_ptr[i];
      }
      RAFT_EXPECTS(check_num_mst_edges, "Some nodes have too many MST graph edges.");
    }

    // 6. Show stats
    if (num_clusters != num_clusters_pre) {
      std::string msg = "# k: " + std::to_string(k);
      msg += ", num_clusters: " + std::to_string(num_clusters);
      msg += ", cluster_size: " + std::to_string(cluster_size_min) + " to " +
             std::to_string(cluster_size_max);
      msg += ", total_num_edges: " + std::to_string(total_outgoing_edges) + ", " +
             std::to_string(total_incoming_edges);
      if (num_alternate + num_failure > 0) {
        msg += ", alternate: " + std::to_string(num_alternate);
        if (num_failure > 0) { msg += ", failure: " + std::to_string(num_failure); }
      }
      RAFT_LOG_DEBUG("%s", msg.c_str());
    }
    RAFT_EXPECTS(num_clusters > 0, "No clusters could not be created in MST optimization.");
    RAFT_EXPECTS(total_outgoing_edges == total_incoming_edges,
                 "The numbers of incoming and outcoming edges are mismatch.");
    if (num_clusters == 1) { break; }
    num_clusters_pre = num_clusters;
  }

  // The edges that make up the MST are stored as edges in the output graph.
  if (use_gpu) {
    raft::copy(res, mst_graph.view(), raft::make_const_mdspan(d_mst_graph.view()));
    raft::resource::sync_stream(res);
  }
#pragma omp parallel for
  for (uint64_t i = 0; i < graph_size; i++) {
    uint64_t k = 0;
    for (uint64_t kj = 0; kj < mst_graph_degree; kj++) {
      uint64_t j = mst_graph(i, kj);
      if (j >= graph_size) continue;

      // Check to avoid duplication
      auto flag_match = false;
      for (uint64_t ki = 0; ki < k; ki++) {
        if (j == output_graph(i, ki)) {
          flag_match = true;
          break;
        }
      }
      if (flag_match) continue;

      output_graph(i, k) = j;
      k += 1;
    }
    mst_graph_num_edges(i) = k;
  }

  const double time_mst_opt_end = cur_time();
  RAFT_LOG_DEBUG("# MST optimization time: %.1lf sec", time_mst_opt_end - time_mst_opt_start);
}

//
// Prune unimportant edges based on 2-hop detour counts.
//
// The edge to be retained is determined without explicitly considering distance or angle.
// Suppose the edge is the k-th edge of some node-A to node-B (A->B). Among the edges
// originating at node-A, there are k-1 edges shorter than the edge A->B. Each of these
// k-1 edges are connected to a different k-1 nodes. Among these k-1 nodes, count the
// number of nodes with edges to node-B, which is the number of 2-hop detours for the
// edge A->B. Once the number of 2-hop detours has been counted for all edges, the
// specified number of edges are picked up for each node, starting with the edge with
// the lowest number of 2-hop detours.
//
template <typename IdxT,
          typename AccessorKnnGraph,
          typename AccessorOutputGraph,
          bool VariableDegree>
void prune_graph_gpu(
  raft::resources const& res,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorKnnGraph> knn_graph,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorOutputGraph>
    output_graph,
  raft::device_vector_view<uint32_t, int64_t> d_natural_degree,
  uint32_t target_pruned_degree)
{
  const uint64_t graph_size          = output_graph.extent(0);
  const uint64_t knn_graph_degree    = knn_graph.extent(1);
  const uint64_t output_graph_degree = output_graph.extent(1);

  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> block_scope(
    "cagra::graph::optimize/prune");

  uint32_t batch_size =
    static_cast<uint32_t>(std::min<uint64_t>(graph_size, helpers::kOptimizeBatchSize));
  const uint32_t num_batch = raft::div_rounding_up_safe<uint64_t>(graph_size, batch_size);

  RAFT_LOG_DEBUG("# Pruning kNN Graph on GPUs\r");

  const double prune_start = cur_time();

  [[maybe_unused]] uint64_t num_keep = 0;
  [[maybe_unused]] uint64_t num_full = 0;
  auto dev_stats                     = raft::make_device_vector<uint64_t>(res, 2);
  auto host_stats                    = raft::make_host_vector<uint64_t>(2);
  raft::matrix::fill(res, dev_stats.view(), uint64_t(0));

  namespace bli                       = cuvs::spatial::knn::detail::utils;
  auto [copy_stream, enable_prefetch] = bli::get_prefetch_stream(res);
  auto workspace_mr                   = raft::resource::get_workspace_resource_ref(res);
  auto large_workspace_mr             = raft::resource::get_large_workspace_resource_ref(res);

  // Single-batch read-only iterator for the input graph (graph_size rows fit in one batch).
  bli::batch_load_iterator<
    raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorKnnGraph>>
    d_input_graph(res, knn_graph, graph_size, copy_stream, large_workspace_mr);
  auto input_view = (*d_input_graph).view();

  bli::batch_load_iterator<
    raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorOutputGraph>>
    d_output_graph(res,
                   output_graph,
                   batch_size,
                   copy_stream,
                   workspace_mr,
                   enable_prefetch,
                   /*initialize=*/false,
                   /*host_writeback=*/true);
  d_output_graph.prefetch_next_batch();

  auto d_invalid_neighbor_list = raft::make_device_scalar<uint32_t>(res, 0u);

  const uint32_t num_warps = 4;
  const dim3 threads_prune(raft::WarpSize * num_warps, 1, 1);
  const dim3 blocks_prune(raft::ceildiv(batch_size, num_warps), 1, 1);
  const size_t prune_smem_size = num_warps * knn_graph_degree * (sizeof(IdxT) + sizeof(uint32_t));

  for (uint32_t i_batch = 0; i_batch < num_batch; i_batch++) {
    auto output_view = (*d_output_graph).view();
    kern_fused_prune<IdxT, num_warps, VariableDegree>
      <<<blocks_prune, threads_prune, prune_smem_size, raft::resource::get_cuda_stream(res)>>>(
        input_view,
        output_view,
        batch_size,
        i_batch,
        d_invalid_neighbor_list.data_handle(),
        dev_stats.data_handle(),
        d_natural_degree,
        target_pruned_degree);

    d_output_graph.prefetch_next_batch();
    ++d_output_graph;
    RAFT_LOG_DEBUG(
      "# Pruning kNN Graph on GPUs (%.1lf %%)\r",
      (double)std::min<IdxT>((i_batch + 1) * batch_size, graph_size) / graph_size * 100);
  }
  raft::resource::sync_stream(res);
  RAFT_LOG_DEBUG("\n");

  uint32_t invalid_neighbor_list = 0;
  raft::copy(
    res, raft::make_host_scalar_view(&invalid_neighbor_list), d_invalid_neighbor_list.view());
  raft::copy(res, host_stats.view(), raft::make_const_mdspan(dev_stats.view()));
  raft::resource::sync_stream(res);

  RAFT_EXPECTS(
    invalid_neighbor_list == 0,
    "Could not generate an intermediate CAGRA graph because the initial kNN graph contains too "
    "many invalid or duplicated neighbor nodes. This error can occur, for example, if too many "
    "overflows occur during the norm computation between the dataset vectors.");

  num_keep = host_stats.data_handle()[0];
  num_full = host_stats.data_handle()[1];

  const double prune_end = cur_time();
  RAFT_LOG_DEBUG(
    "# Time for pruning on GPU: %.1lf sec, "
    "avg_no_detour_edges_per_node: %.2lf/%u, "
    "nodes_with_no_detour_at_all_edges: %.1lf%%",
    prune_end - prune_start,
    (double)num_keep / graph_size,
    output_graph_degree,
    (double)num_full / graph_size * 100);
}

}  // namespace

template <typename IdxT = uint32_t,
          typename AccessorKnnGraph =
            raft::host_device_accessor<cuda::std::default_accessor<IdxT>, raft::memory_type::host>,
          typename AccessorOutputGraph =
            raft::host_device_accessor<cuda::std::default_accessor<IdxT>, raft::memory_type::host>>
void optimize(
  raft::resources const& res_const,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorKnnGraph> knn_graph,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorOutputGraph> new_graph,
  const bool guarantee_connectivity           = true,
  const bool use_gpu_for_mst_optimization     = true,
  const double variable_graph_degree_fraction = 1.0)
{
  RAFT_LOG_DEBUG(
    "# Pruning kNN graph (size=%lu, degree=%lu)\n", knn_graph.extent(0), knn_graph.extent(1));

  // TODO(achirkin): come up with a reasonable API to initialize a non-empty stream pool.
  // raft::resource::set_cuda_stream_pool below modifies the resource, so it cannot be const.
  // The optimize() is a heavy function, so copying the resource and creating a private stream pool
  // is not a big overhead.
  raft::resources res{res_const};

  // large temporary memory for large arrays, e.g. everything >= O(graph_size)
  auto large_tmp_mr = raft::resource::get_large_workspace_resource_ref(res);
  // temporary memory for small arrays, e.g. everything <= O(batchsize * graph_degree)
  auto default_ws_mr = raft::resource::get_workspace_resource_ref(res);

  // create a stream pool if not already present
  if (!res.has_resource_factory(raft::resource::resource_type::CUDA_STREAM_POOL) ||
      raft::resource::get_stream_pool_size(res) < 1) {
    raft::resource::set_cuda_stream_pool(res, std::make_shared<rmm::cuda_stream_pool>(1));
  }

  RAFT_EXPECTS(knn_graph.extent(0) == new_graph.extent(0),
               "Each input array is expected to have the same number of rows");
  RAFT_EXPECTS(new_graph.extent(1) <= knn_graph.extent(1),
               "output graph cannot have more columns than input graph");
  // const uint64_t input_graph_degree  = knn_graph.extent(1);
  const uint64_t knn_graph_degree     = knn_graph.extent(1);
  const uint64_t output_graph_degree  = new_graph.extent(1);
  const uint64_t graph_size           = new_graph.extent(0);
  const uint64_t target_pruned_degree = std::max<uint64_t>(
    1, static_cast<uint64_t>(std::ceil(output_graph_degree * variable_graph_degree_fraction)));
  const bool variable_graph_degree = (target_pruned_degree < output_graph_degree);

  if (variable_graph_degree) {
    RAFT_LOG_INFO("# Pruning kNN graph (size=%lu, degree=%lu, target_pruned_degree=%lu)",
                  graph_size,
                  knn_graph_degree,
                  target_pruned_degree);
  }

  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "cagra::graph::optimize(%zu, %zu, %u)", graph_size, knn_graph_degree, output_graph_degree);

  // Per-node "natural" degree produced by the prune step. Only allocated when the variable-degree
  // path is enabled. The prune kernel records here the count of edges with detour level <=
  // detour level at position (target_pruned_degree - 1). The merge kernel then both reads and
  // updates this array (after rev-graph back-fill the value becomes the final effective degree).
  auto d_natural_degree = raft::make_device_mdarray<uint32_t>(
    res,
    default_ws_mr,
    raft::make_extents<int64_t>(variable_graph_degree ? static_cast<int64_t>(graph_size) : 0));

  // MST optimization
  // currently, only using GPU path for MST optimization
  int64_t mst_graph_size = guarantee_connectivity ? graph_size : 0;
  auto mst_graph =
    raft::make_host_matrix<IdxT, int64_t, raft::row_major>(mst_graph_size, output_graph_degree);
  auto mst_graph_num_edges = raft::make_host_vector<uint32_t, int64_t>(mst_graph_size);

  if (guarantee_connectivity) {
    raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> block_scope(
      "cagra::graph::optimize/check_connectivity");
    RAFT_LOG_INFO("MST optimization is used to guarantee graph connectivity.");
    mst_optimization<IdxT>(
      res, knn_graph, mst_graph.view(), mst_graph_num_edges.view(), use_gpu_for_mst_optimization);

    for (uint64_t i = 0; i < graph_size; i++) {
      if (i < 8 || i >= graph_size - 8) {
        RAFT_LOG_DEBUG("# mst_graph_num_edges[%lu]: %u\n", i, mst_graph_num_edges(i));
      }
    }
  }

  // prune graph -- will always use GPU path
  {
    if (variable_graph_degree) {
      prune_graph_gpu<IdxT, AccessorKnnGraph, AccessorOutputGraph, true>(
        res, knn_graph, new_graph, d_natural_degree.view(), target_pruned_degree);
    } else {
      prune_graph_gpu<IdxT, AccessorKnnGraph, AccessorOutputGraph, false>(
        res, knn_graph, new_graph, d_natural_degree.view(), target_pruned_degree);
    }
  }

  // reverse graph creation will always use the GPU / large workspace resource
  auto d_rev_graph = raft::make_device_mdarray<IdxT>(
    res, large_tmp_mr, raft::make_extents<int64_t>(graph_size, output_graph_degree));

  // This should use the default workspace resource for random access / atomics
  auto d_rev_graph_count = raft::make_device_mdarray<uint32_t>(
    res, default_ws_mr, raft::make_extents<int64_t>(graph_size));

  const double time_make_start = cur_time();

  make_reverse_graph_gpu<IdxT, AccessorOutputGraph>(
    res, new_graph, d_rev_graph.view(), d_rev_graph_count.view());

  raft::resource::sync_stream(res);

  const double time_make_end = cur_time();
  RAFT_LOG_DEBUG("\n# Making reverse graph time: %.1lf ms",
                 (time_make_end - time_make_start) * 1000.0);

  // merge graph -- will always use GPU path
  {
    if (variable_graph_degree) {
      merge_graph_gpu<IdxT, AccessorOutputGraph, true>(res,
                                                       new_graph,
                                                       d_rev_graph.view(),
                                                       d_rev_graph_count.view(),
                                                       mst_graph.view(),
                                                       mst_graph_num_edges.view(),
                                                       guarantee_connectivity,
                                                       d_natural_degree.view());
    } else {
      merge_graph_gpu<IdxT, AccessorOutputGraph, false>(res,
                                                        new_graph,
                                                        d_rev_graph.view(),
                                                        d_rev_graph_count.view(),
                                                        mst_graph.view(),
                                                        mst_graph_num_edges.view(),
                                                        guarantee_connectivity,
                                                        d_natural_degree.view());
    }
  }

  auto d_avg_natural = raft::make_device_scalar<double>(res, 0);
  double avg_natural = 0;
  if (variable_graph_degree) {
    // main_op: cast each degree to double (innermost op, which also drops the reduction-index
    // argument), then divide by graph_size, so the reduction accumulates the mean directly.
    auto normalize_mean = raft::compose_op{
      raft::div_const_op<double>{static_cast<double>(graph_size)}, raft::cast_op<double>{}};
    raft::linalg::reduce<raft::Apply::ALONG_ROWS>(
      res,
      raft::make_device_matrix_view<const uint32_t, int64_t, raft::row_major>(
        d_natural_degree.data_handle(), int64_t{1}, static_cast<int64_t>(graph_size)),
      raft::make_device_vector_view<double, int64_t>(d_avg_natural.data_handle(), int64_t{1}),
      0.0,
      false,
      normalize_mean);
    raft::copy(res, raft::make_host_scalar_view(&avg_natural), d_avg_natural.view());
  }

  raft::resource::sync_stream(res);

  if (variable_graph_degree) {
    RAFT_LOG_INFO(
      "# Variable graph degree: avg natural degree = %.2f / %lu", avg_natural, output_graph_degree);
  }

  // These host-side checks are expensive (O(N*D^2)) and only used as debug
  // diagnostics, so only run them when debug logging is active at runtime.
  if (raft::default_logger().should_log(rapids_logger::level_enum::debug)) {
    log_incoming_edges_histogram<IdxT, AccessorOutputGraph>(res, new_graph);

    check_duplicates_and_out_of_range<IdxT, AccessorOutputGraph>(res, new_graph);
  }
}

}  // namespace cuvs::neighbors::cagra::detail::graph
