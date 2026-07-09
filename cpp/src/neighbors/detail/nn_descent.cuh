/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ann_utils.cuh"
#include "neighbors_device_intrinsics.cuh"
#include "nn_descent_gnnd.hpp"

#include "../../core/omp_wrapper.hpp"
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/nn_descent.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/matrix/init.cuh>
#include <raft/matrix/slice.cuh>
#include <raft/util/arch.cuh>  // raft::util::arch::SM_*
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/pow2_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include <mma.h>

#include <limits>
#include <optional>
#include <queue>
#include <random>
#include <type_traits>

namespace cuvs::neighbors::nn_descent::detail {

template <typename Index_t>
struct ResultItem;

template <>
class ResultItem<int> {
 private:
  using Index_t = int;
  Index_t id_;
  DistData_t dist_;

 public:
  __host__ __device__ ResultItem()
    : id_(std::numeric_limits<Index_t>::max()), dist_(std::numeric_limits<DistData_t>::max()) {};
  __host__ __device__ ResultItem(const Index_t id_with_flag, const DistData_t dist)
    : id_(id_with_flag), dist_(dist) {};
  __host__ __device__ bool is_new() const { return id_ >= 0; }
  __host__ __device__ Index_t& id_with_flag() { return id_; }
  __host__ __device__ Index_t id() const
  {
    if (is_new()) return id_;
    return -id_ - 1;
  }
  __host__ __device__ DistData_t& dist() { return dist_; }

  __host__ __device__ void mark_old()
  {
    if (id_ >= 0) id_ = -id_ - 1;
  }

  __host__ __device__ bool operator<(const ResultItem<Index_t>& other) const
  {
    if (dist_ == other.dist_) return id() < other.id();
    return dist_ < other.dist_;
  }
  __host__ __device__ bool operator==(const ResultItem<Index_t>& other) const
  {
    return id() == other.id();
  }
  __host__ __device__ bool operator>=(const ResultItem<Index_t>& other) const
  {
    return !(*this < other);
  }
  __host__ __device__ bool operator<=(const ResultItem<Index_t>& other) const
  {
    return (*this == other) || (*this < other);
  }
  __host__ __device__ bool operator>(const ResultItem<Index_t>& other) const
  {
    return !(*this <= other);
  }
  __host__ __device__ bool operator!=(const ResultItem<Index_t>& other) const
  {
    return !(*this == other);
  }
};

using align32 = raft::Pow2<32>;

template <typename T>
int get_batch_size(const int it_now, const T nrow, const int batch_size)
{
  int it_total = raft::ceildiv(nrow, batch_size);
  return (it_now == it_total - 1) ? nrow - it_now * batch_size : batch_size;
}

// for avoiding bank conflict
template <typename T>
constexpr __host__ __device__ __forceinline__ int skew_dim(int ndim)
{
  // all "4"s are for alignment
  if constexpr (std::is_same<T, float>::value) {
    ndim = raft::ceildiv(ndim, 4) * 4;
    return ndim + (ndim % 32 == 0) * 4;
  }
}

template <typename T>
struct dtype_traits;

template <>
struct dtype_traits<float> {
  static constexpr int APAD           = 4;
  static constexpr int BPAD           = 4;
  static constexpr int TILE_COL_WIDTH = 32;
  static __device__ __forceinline__ float to_float(float v) { return v; }
};

template <>
struct dtype_traits<__half> {
  static constexpr int APAD           = 8;
  static constexpr int BPAD           = 8;
  static constexpr int TILE_COL_WIDTH = 64;
  static __device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
};

template <typename T>
concept Byte = std::is_same_v<T, uint8_t> or std::is_same_v<T, int8_t>;
template <Byte T>
struct dtype_traits<T> {
  static constexpr int APAD           = 4;
  static constexpr int BPAD           = 4;
  static constexpr int TILE_COL_WIDTH = 128;
  static __device__ __forceinline__ float to_float(T v) { return static_cast<float>(v); }
};

template <typename T>
__device__ __forceinline__ ResultItem<T> xor_swap(ResultItem<T> x, int mask, int dir)
{
  ResultItem<T> y;
  y.dist() = __shfl_xor_sync(raft::warp_full_mask(), x.dist(), mask, raft::warp_size());
  y.id_with_flag() =
    __shfl_xor_sync(raft::warp_full_mask(), x.id_with_flag(), mask, raft::warp_size());
  return x < y == dir ? y : x;
}

__device__ __forceinline__ int xor_swap(int x, int mask, int dir)
{
  int y = __shfl_xor_sync(raft::warp_full_mask(), x, mask, raft::warp_size());
  return x < y == dir ? y : x;
}

// TODO: Move to RAFT utils https://github.com/rapidsai/raft/issues/1827
__device__ __forceinline__ uint bfe(uint lane_id, uint pos)
{
  uint res;
  asm("bfe.u32 %0,%1,%2,%3;" : "=r"(res) : "r"(lane_id), "r"(pos), "r"(1));
  return res;
}

template <typename T>
__device__ __forceinline__ void warp_bitonic_sort(T* element_ptr, const int lane_id)
{
  static_assert(raft::warp_size() == 32);
  auto& element = *element_ptr;
  element       = xor_swap(element, 0x01, bfe(lane_id, 1) ^ bfe(lane_id, 0));
  element       = xor_swap(element, 0x02, bfe(lane_id, 2) ^ bfe(lane_id, 1));
  element       = xor_swap(element, 0x01, bfe(lane_id, 2) ^ bfe(lane_id, 0));
  element       = xor_swap(element, 0x04, bfe(lane_id, 3) ^ bfe(lane_id, 2));
  element       = xor_swap(element, 0x02, bfe(lane_id, 3) ^ bfe(lane_id, 1));
  element       = xor_swap(element, 0x01, bfe(lane_id, 3) ^ bfe(lane_id, 0));
  element       = xor_swap(element, 0x08, bfe(lane_id, 4) ^ bfe(lane_id, 3));
  element       = xor_swap(element, 0x04, bfe(lane_id, 4) ^ bfe(lane_id, 2));
  element       = xor_swap(element, 0x02, bfe(lane_id, 4) ^ bfe(lane_id, 1));
  element       = xor_swap(element, 0x01, bfe(lane_id, 4) ^ bfe(lane_id, 0));
  element       = xor_swap(element, 0x10, bfe(lane_id, 4));
  element       = xor_swap(element, 0x08, bfe(lane_id, 3));
  element       = xor_swap(element, 0x04, bfe(lane_id, 2));
  element       = xor_swap(element, 0x02, bfe(lane_id, 1));
  element       = xor_swap(element, 0x01, bfe(lane_id, 0));
  return;
}

constexpr int NUM_SAMPLES = 32;
// For now, the max. number of samples is 32, so the sample cache size is fixed
// to 64 (32 * 2).
constexpr int MAX_NUM_BI_SAMPLES        = 64;
constexpr int SKEWED_MAX_NUM_BI_SAMPLES = skew_dim<float>(MAX_NUM_BI_SAMPLES);
constexpr int BLOCK_SIZE                = 512;
constexpr int WMMA_M                    = 16;
constexpr int WMMA_N                    = 16;
constexpr int WMMA_K                    = 16;

template <typename Data_t>
__device__ __forceinline__ void load_vec(Data_t* vec_buffer,
                                         const Data_t* d_vec,
                                         const int load_dims,
                                         const int padding_dims,
                                         const int lane_id)
{
  if constexpr (std::is_same_v<Data_t, float> or std::is_same_v<Data_t, uint8_t> or
                std::is_same_v<Data_t, int8_t>) {
    constexpr int num_load_elems_per_warp = raft::warp_size();
    for (int step = 0; step < raft::ceildiv(padding_dims, num_load_elems_per_warp); step++) {
      int idx = step * num_load_elems_per_warp + lane_id;
      if (idx < load_dims) {
        vec_buffer[idx] = d_vec[idx];
      } else if (idx < padding_dims) {
        vec_buffer[idx] = 0.0f;
      }
    }
  }
  if constexpr (std::is_same_v<Data_t, __half>) {
    if ((size_t)d_vec % sizeof(float2) == 0 && (size_t)vec_buffer % sizeof(float2) == 0 &&
        load_dims % 4 == 0 && padding_dims % 4 == 0) {
      constexpr int num_load_elems_per_warp = raft::warp_size() * 4;
#pragma unroll
      for (int step = 0; step < raft::ceildiv(padding_dims, num_load_elems_per_warp); step++) {
        int idx_in_vec = step * num_load_elems_per_warp + lane_id * 4;
        if (idx_in_vec + 4 <= load_dims) {
          *(float2*)(vec_buffer + idx_in_vec) = *(float2*)(d_vec + idx_in_vec);
        } else if (idx_in_vec + 4 <= padding_dims) {
          *(float2*)(vec_buffer + idx_in_vec) = float2({0.0f, 0.0f});
        }
      }
    } else {
      constexpr int num_load_elems_per_warp = raft::warp_size();
      for (int step = 0; step < raft::ceildiv(padding_dims, num_load_elems_per_warp); step++) {
        int idx = step * num_load_elems_per_warp + lane_id;
        if (idx < load_dims) {
          vec_buffer[idx] = d_vec[idx];
        } else if (idx < padding_dims) {
          vec_buffer[idx] = 0.0f;
        }
      }
    }
  }
}

/** Converting load: loads Data_t from global memory into __half shared memory buffer. */
template <typename Data_t>
  requires(!std::is_same_v<Data_t, __half>)
__device__ __forceinline__ void load_vec(__half* vec_buffer,
                                         const Data_t* d_vec,
                                         const int load_dims,
                                         const int padding_dims,
                                         const int lane_id)
{
  constexpr int num_load_elems_per_warp = raft::warp_size();
  const __half half_0                   = __float2half(0.0f);
  for (int step = 0; step < raft::ceildiv(padding_dims, num_load_elems_per_warp); step++) {
    int idx = step * num_load_elems_per_warp + lane_id;
    if (idx < load_dims) {
      vec_buffer[idx] = d_vec[idx];
    } else if (idx < padding_dims) {
      vec_buffer[idx] = half_0;
    }
  }
}

/** One warp per block. Computes squared L2 norm for each row. */
template <typename Data_t>
RAFT_KERNEL compute_l2_norms_kernel(const Data_t* data, int dim, DistData_t* l2_norms)
{
  extern __shared__ char buffer[];
  __shared__ float l2_norm;
  Data_t* s_vec  = (Data_t*)buffer;
  size_t list_id = blockIdx.x;
  int lane_id    = threadIdx.x % raft::warp_size();

  load_vec(s_vec, data + static_cast<size_t>(blockIdx.x) * dim, dim, dim, lane_id);
  if (threadIdx.x == 0) { l2_norm = 0; }
  __syncthreads();

  for (int step = 0; step < raft::ceildiv(dim, raft::warp_size()); step++) {
    int idx         = step * raft::warp_size() + lane_id;
    float part_dist = 0;
    if (idx < dim) {
      part_dist = static_cast<float>(s_vec[idx]);
      part_dist = part_dist * part_dist;
    }
    __syncwarp();
    for (int offset = raft::warp_size() >> 1; offset >= 1; offset >>= 1) {
      part_dist += __shfl_down_sync(raft::warp_full_mask(), part_dist, offset);
    }
    if (lane_id == 0) { l2_norm += part_dist; }
    __syncwarp();
  }

  if (lane_id == 0) { l2_norms[list_id] = l2_norm; }
}

template <typename Data_t>
size_t compute_l2_norms_smem_size(size_t dim)
{
  return sizeof(Data_t) * raft::ceildiv(dim, static_cast<size_t>(raft::warp_size())) *
         static_cast<size_t>(raft::warp_size());
}

template <typename Data_t>
size_t configure_compute_l2_norms_kernel_smem(size_t dim)
{
  size_t smem = compute_l2_norms_smem_size<Data_t>(dim);

  int device_id = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&device_id));

  int max_smem = 0;
  RAFT_CUDA_TRY(
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id));
  RAFT_EXPECTS(smem <= static_cast<size_t>(max_smem),
               "NN_DESCENT L2 norm kernel requires %zu bytes of dynamic shared memory "
               "(dim=%zu, sizeof(Data_t)=%zu), but the device supports at most %d bytes "
               "per block via opt-in.",
               smem,
               dim,
               sizeof(Data_t),
               max_smem);

  auto kernel = compute_l2_norms_kernel<Data_t>;
  RAFT_CUDA_TRY(cudaFuncSetAttribute(
    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
  return smem;
}

template <typename Src_t, typename Dst_t>
RAFT_KERNEL convert_copy_kernel(const Src_t* src, Dst_t* dst, size_t n)
{
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) { dst[idx] = static_cast<Dst_t>(src[idx]); }
}

template <typename Index_t>
RAFT_KERNEL add_rev_edges_kernel(const Index_t* graph,
                                 Index_t* rev_graph,
                                 int num_samples,
                                 int2* list_sizes)
{
  size_t list_id = blockIdx.x;
  int2 list_size = list_sizes[list_id];

  for (int idx = threadIdx.x; idx < list_size.x; idx += blockDim.x) {
    // each node has same number (num_samples) of forward and reverse edges
    Index_t rev_list_id = graph[list_id * num_samples + idx];
    if (rev_list_id == std::numeric_limits<Index_t>::max()) {
      // sentinel value
      continue;
    }

    // there are already num_samples forward edges
    int idx_in_rev_list = atomicAdd(&list_sizes[rev_list_id].y, 1);
    if (idx_in_rev_list >= num_samples) {
      atomicExch(&list_sizes[rev_list_id].y, num_samples);
    } else {
      rev_graph[rev_list_id * num_samples + idx_in_rev_list] = list_id;
    }
  }
}

template <typename Index_t, typename ID_t = InternalID_t<Index_t>>
__device__ void insert_to_global_graph(ResultItem<Index_t> elem,
                                       size_t list_id,
                                       ID_t* graph,
                                       DistData_t* dists,
                                       int node_degree,
                                       int* locks)
{
  int tx                 = threadIdx.x;
  int lane_id            = tx % raft::warp_size();
  size_t global_idx_base = list_id * node_degree;
  if (elem.id() == list_id) return;

  const int num_segments = raft::ceildiv(node_degree, raft::warp_size());

  int loop_flag = 0;
  do {
    int segment_id = elem.id() % num_segments;
    if (lane_id == 0) {
      loop_flag = atomicCAS(&locks[list_id * num_segments + segment_id], 0, 1) == 0;
    }

    loop_flag = __shfl_sync(raft::warp_full_mask(), loop_flag, 0);

    if (loop_flag == 1) {
      ResultItem<Index_t> knn_list_frag;
      int local_idx     = segment_id * raft::warp_size() + lane_id;
      size_t global_idx = global_idx_base + local_idx;
      if (local_idx < node_degree) {
        knn_list_frag.id_with_flag() = graph[global_idx].id_with_flag();
        knn_list_frag.dist()         = dists[global_idx];
      }

      int pos_to_insert = -1;
      ResultItem<Index_t> prev_elem;

      prev_elem.id_with_flag() =
        __shfl_up_sync(raft::warp_full_mask(), knn_list_frag.id_with_flag(), 1);
      prev_elem.dist() = __shfl_up_sync(raft::warp_full_mask(), knn_list_frag.dist(), 1);

      if (lane_id == 0) {
        prev_elem = ResultItem<Index_t>{std::numeric_limits<Index_t>::min(),
                                        std::numeric_limits<DistData_t>::lowest()};
      }
      if (elem > prev_elem && elem < knn_list_frag) {
        pos_to_insert = segment_id * raft::warp_size() + lane_id;
      } else if (elem == prev_elem || elem == knn_list_frag) {
        pos_to_insert = -2;
      }
      uint mask = __ballot_sync(raft::warp_full_mask(), pos_to_insert >= 0);
      if (mask) {
        uint set_lane_id = __fns(mask, 0, 1);
        pos_to_insert    = __shfl_sync(raft::warp_full_mask(), pos_to_insert, set_lane_id);
      }

      if (pos_to_insert >= 0) {
        int local_idx = segment_id * raft::warp_size() + lane_id;
        if (local_idx > pos_to_insert) {
          local_idx++;
        } else if (local_idx == pos_to_insert) {
          graph[global_idx_base + local_idx].id_with_flag() = elem.id_with_flag();
          dists[global_idx_base + local_idx]                = elem.dist();
          local_idx++;
        }
        size_t global_pos = global_idx_base + local_idx;
        if (local_idx < (segment_id + 1) * raft::warp_size() && local_idx < node_degree) {
          graph[global_pos].id_with_flag() = knn_list_frag.id_with_flag();
          dists[global_pos]                = knn_list_frag.dist();
        }
      }
      __threadfence();
      if (loop_flag && lane_id == 0) { atomicExch(&locks[list_id * num_segments + segment_id], 0); }
    }
  } while (!loop_flag);
}

template <typename Index_t>
__device__ ResultItem<Index_t> get_min_item(const Index_t id,
                                            const int idx_in_list,
                                            const Index_t* neighbs,
                                            const DistData_t* distances,
                                            const bool find_in_row = true)
{
  int lane_id = threadIdx.x % raft::warp_size();

  static_assert(MAX_NUM_BI_SAMPLES == 64);
  int idx[MAX_NUM_BI_SAMPLES / raft::warp_size()];
  float dist[MAX_NUM_BI_SAMPLES / raft::warp_size()] = {std::numeric_limits<DistData_t>::max(),
                                                        std::numeric_limits<DistData_t>::max()};
  idx[0]                                             = lane_id;
  idx[1]                                             = raft::warp_size() + lane_id;

  if (neighbs[idx[0]] != id) {
    dist[0] = find_in_row ? distances[idx_in_list * SKEWED_MAX_NUM_BI_SAMPLES + lane_id]
                          : distances[idx_in_list + lane_id * SKEWED_MAX_NUM_BI_SAMPLES];
  }

  if (neighbs[idx[1]] != id) {
    dist[1] =
      find_in_row
        ? distances[idx_in_list * SKEWED_MAX_NUM_BI_SAMPLES + raft::warp_size() + lane_id]
        : distances[idx_in_list + (raft::warp_size() + lane_id) * SKEWED_MAX_NUM_BI_SAMPLES];
  }

  if (dist[1] < dist[0]) {
    dist[0] = dist[1];
    idx[0]  = idx[1];
  }
  __syncwarp();
  for (int offset = raft::warp_size() >> 1; offset >= 1; offset >>= 1) {
    float other_idx  = __shfl_down_sync(raft::warp_full_mask(), idx[0], offset);
    float other_dist = __shfl_down_sync(raft::warp_full_mask(), dist[0], offset);
    if (other_dist < dist[0]) {
      dist[0] = other_dist;
      idx[0]  = other_idx;
    }
  }

  ResultItem<Index_t> result;
  result.dist()         = __shfl_sync(raft::warp_full_mask(), dist[0], 0);
  result.id_with_flag() = neighbs[__shfl_sync(raft::warp_full_mask(), idx[0], 0)];
  return result;
}

template <typename T>
__device__ __forceinline__ void remove_duplicates(
  T* list_a, int list_a_size, T* list_b, int list_b_size, int& unique_counter, int execute_warp_id)
{
  static_assert(raft::warp_size() == 32);
  if (!(threadIdx.x >= execute_warp_id * raft::warp_size() &&
        threadIdx.x < execute_warp_id * raft::warp_size() + raft::warp_size())) {
    return;
  }
  int lane_id = threadIdx.x % raft::warp_size();
  T elem      = std::numeric_limits<T>::max();
  if (lane_id < list_a_size) { elem = list_a[lane_id]; }
  warp_bitonic_sort(&elem, lane_id);

  if (elem != std::numeric_limits<T>::max()) { list_a[lane_id] = elem; }

  T elem_b = std::numeric_limits<T>::max();

  if (lane_id < list_b_size) { elem_b = list_b[lane_id]; }
  __syncwarp();

  int idx_l    = 0;
  int idx_r    = list_a_size;
  bool existed = false;
  while (idx_l < idx_r) {
    int idx  = (idx_l + idx_r) / 2;
    int elem = list_a[idx];
    if (elem == elem_b) {
      existed = true;
      break;
    }
    if (elem_b > elem) {
      idx_l = idx + 1;
    } else {
      idx_r = idx;
    }
  }
  if (!existed && elem_b != std::numeric_limits<T>::max()) {
    int idx                   = atomicAdd(&unique_counter, 1);
    list_a[list_a_size + idx] = elem_b;
  }
}

template <typename Index_t, typename Data_t, typename DistEpilogue_t>
__device__ __forceinline__ void calculate_metric(float* s_distances,
                                                 Index_t* row_neighbors,
                                                 int list_row_size,
                                                 Index_t* col_neighbors,
                                                 int list_col_size,
                                                 const Data_t* data,
                                                 const int data_dim,
                                                 DistData_t* l2_norms,
                                                 cuvs::distance::DistanceType metric,
                                                 DistEpilogue_t dist_epilogue)
{
  // if we have a distance epilogue, distances need to be fully calculated instead of postprocessing
  // them.
  bool can_postprocess_dist = std::is_same_v<DistEpilogue_t, raft::identity_op>;

  for (int i = threadIdx.x; i < MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES; i += blockDim.x) {
    int row_id = i / SKEWED_MAX_NUM_BI_SAMPLES;
    int col_id = i % SKEWED_MAX_NUM_BI_SAMPLES;

    if (row_id < list_row_size && col_id < list_col_size) {
      if (metric == cuvs::distance::DistanceType::InnerProduct && can_postprocess_dist) {
        s_distances[i] = -s_distances[i];
      } else if (metric == cuvs::distance::DistanceType::CosineExpanded) {
        float norm_product = l2_norms[row_neighbors[row_id]] * l2_norms[col_neighbors[col_id]];
        s_distances[i] =
          (norm_product > 0.0f) ? (1.0f - s_distances[i] / sqrtf(norm_product)) : 0.0f;
      } else if (metric == cuvs::distance::DistanceType::BitwiseHamming) {
        s_distances[i] = 0.0;
        int n1         = row_neighbors[row_id];
        int n2         = col_neighbors[col_id];
        // TODO: https://github.com/rapidsai/cuvs/issues/1127
        const uint8_t* data_n1 = reinterpret_cast<const uint8_t*>(data) + n1 * data_dim;
        const uint8_t* data_n2 = reinterpret_cast<const uint8_t*>(data) + n2 * data_dim;
        for (int d = 0; d < data_dim; d++) {
          s_distances[i] += __popc(static_cast<uint32_t>(data_n1[d] ^ data_n2[d]) & 0xff);
        }
      } else if (metric == cuvs::distance::DistanceType::L2Expanded ||
                 metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
        s_distances[i] =
          l2_norms[row_neighbors[row_id]] + l2_norms[col_neighbors[col_id]] - 2.0 * s_distances[i];
        // for fp32 vs fp16 precision differences resulting in negative distances when distance
        // should be 0 related issue: https://github.com/rapidsai/cuvs/issues/991
        s_distances[i] = s_distances[i] < 0.0f ? 0.0f : s_distances[i];
        if (!can_postprocess_dist && metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
          s_distances[i] = sqrtf(s_distances[i]);
        }
      }
      s_distances[i] = dist_epilogue(s_distances[i], row_neighbors[row_id], col_neighbors[col_id]);
    } else {
      s_distances[i] = std::numeric_limits<float>::max();
    }
  }
}

struct DistAccumulator {
  cuvs::distance::DistanceType metric;
  __device__ __forceinline__ float operator()(float a, float b) const
  {
    if (metric == cuvs::distance::DistanceType::L1) { return raft::abs(a - b); }
    // dot product: reused by IP, cosine, and L2 (postprocessed in calculate_metric)
    return a * b;
  }
};

// launch_bounds here denote BLOCK_SIZE = 512 and MIN_BLOCKS_PER_SM = 4
// Per
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications,
// MAX_RESIDENT_THREAD_PER_SM = BLOCK_SIZE * BLOCKS_PER_SM = 2048
// For architectures 750 and 860 (890), the values for MAX_RESIDENT_THREAD_PER_SM
// is 1024 and 1536 respectively, which means the bounds don't work anymore
// SIMT kernel: scalar element-wise distance computation.
// Used for fp32 data (all metrics) and L1 distance computation for all dtypes (which cannot use
// tensor cores).
template <typename Data_t,
          typename Index_t,
          typename ID_t = InternalID_t<Index_t>,
          typename DistEpilogue_t>
RAFT_KERNEL
#ifdef __CUDA_ARCH__
// Use minBlocksPerMultiprocessor = 4 on specific arches
#if (__CUDA_ARCH__) == 700 || (__CUDA_ARCH__) == 800 || (__CUDA_ARCH__) == 900 || \
  (__CUDA_ARCH__) == 1000
__launch_bounds__(BLOCK_SIZE, 4)
#else
__launch_bounds__(BLOCK_SIZE)
#endif
#endif
  local_join_kernel_simt(const Index_t* graph_new,
                         const Index_t* rev_graph_new,
                         const int2* sizes_new,
                         const Index_t* graph_old,
                         const Index_t* rev_graph_old,
                         const int2* sizes_old,
                         const int width,
                         const Data_t* data,
                         const int data_dim,
                         ID_t* graph,
                         DistData_t* dists,
                         int graph_width,
                         int* locks,
                         DistData_t* l2_norms,
                         cuvs::distance::DistanceType metric,
                         DistEpilogue_t dist_epilogue)
{
#if (__CUDA_ARCH__ >= 700)
  __shared__ int s_list[MAX_NUM_BI_SAMPLES * 2];

  constexpr int APAD           = dtype_traits<Data_t>::APAD;
  constexpr int BPAD           = dtype_traits<Data_t>::BPAD;
  constexpr int TILE_COL_WIDTH = dtype_traits<Data_t>::TILE_COL_WIDTH;
  __shared__ Data_t s_nv[MAX_NUM_BI_SAMPLES][TILE_COL_WIDTH + APAD];
  __shared__ Data_t s_ov[MAX_NUM_BI_SAMPLES][TILE_COL_WIDTH + BPAD];
  __shared__ float s_distances[MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES];

  // s_distances: MAX_NUM_BI_SAMPLES x SKEWED_MAX_NUM_BI_SAMPLES, reuse the space of s_ov
  int* s_unique_counter = (int*)&s_ov[0][0];

  if (threadIdx.x == 0) {
    s_unique_counter[0] = 0;
    s_unique_counter[1] = 0;
  }

  Index_t* new_neighbors = s_list;
  Index_t* old_neighbors = s_list + MAX_NUM_BI_SAMPLES;

  size_t list_id      = blockIdx.x;
  int2 list_new_size2 = sizes_new[list_id];
  int list_new_size   = list_new_size2.x + list_new_size2.y;
  int2 list_old_size2 = sizes_old[list_id];
  int list_old_size   = list_old_size2.x + list_old_size2.y;

  if (!list_new_size) return;
  int tx = threadIdx.x;

  if (tx < list_new_size2.x) {
    new_neighbors[tx] = graph_new[list_id * width + tx];
  } else if (tx >= list_new_size2.x && tx < list_new_size) {
    new_neighbors[tx] = rev_graph_new[list_id * width + tx - list_new_size2.x];
  }

  if (tx < list_old_size2.x) {
    old_neighbors[tx] = graph_old[list_id * width + tx];
  } else if (tx >= list_old_size2.x && tx < list_old_size) {
    old_neighbors[tx] = rev_graph_old[list_id * width + tx - list_old_size2.x];
  }

  __syncthreads();

  remove_duplicates(new_neighbors,
                    list_new_size2.x,
                    new_neighbors + list_new_size2.x,
                    list_new_size2.y,
                    s_unique_counter[0],
                    0);

  remove_duplicates(old_neighbors,
                    list_old_size2.x,
                    old_neighbors + list_old_size2.x,
                    list_old_size2.y,
                    s_unique_counter[1],
                    1);
  __syncthreads();
  list_new_size = list_new_size2.x + s_unique_counter[0];
  list_old_size = list_old_size2.x + s_unique_counter[1];

  int warp_id             = threadIdx.x / raft::warp_size();
  int lane_id             = threadIdx.x % raft::warp_size();
  constexpr int num_warps = BLOCK_SIZE / raft::warp_size();

  DistAccumulator dist_acc(metric);

  int tid = threadIdx.x;
  for (int i = tid; i < MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES; i += blockDim.x)
    s_distances[i] = 0.0f;

  __syncthreads();

  for (int step = 0; step < raft::ceildiv(data_dim, TILE_COL_WIDTH); step++) {
    int num_load_elems = (step == raft::ceildiv(data_dim, TILE_COL_WIDTH) - 1)
                           ? data_dim - step * TILE_COL_WIDTH
                           : TILE_COL_WIDTH;
#pragma unroll
    for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; i++) {
      int idx = i * num_warps + warp_id;
      if (idx < list_new_size) {
        size_t neighbor_id = new_neighbors[idx];
        size_t idx_in_data = neighbor_id * data_dim;
        // loaded to shared memory while keeping the original dtype
        load_vec(s_nv[idx],
                 data + idx_in_data + step * TILE_COL_WIDTH,
                 num_load_elems,
                 TILE_COL_WIDTH,
                 lane_id);
      }
    }
    __syncthreads();

    // this is much faster than a warp-collaborative multiplication because MAX_NUM_BI_SAMPLES is
    // fixed and small (64)
    for (int i = threadIdx.x; i < MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES; i += blockDim.x) {
      int tmp_row = i / SKEWED_MAX_NUM_BI_SAMPLES;
      int tmp_col = i % SKEWED_MAX_NUM_BI_SAMPLES;
      if (tmp_row < list_new_size && tmp_col < list_new_size) {
        float acc = 0.0f;
        for (int d = 0; d < num_load_elems; d++) {
          // converted to float for distance computation
          float a = dtype_traits<Data_t>::to_float(s_nv[tmp_row][d]);
          float b = dtype_traits<Data_t>::to_float(s_nv[tmp_col][d]);
          acc += dist_acc(a, b);
        }
        s_distances[i] += acc;
      }
    }
    __syncthreads();
  }
  __syncthreads();

  calculate_metric(s_distances,
                   new_neighbors,
                   list_new_size,
                   new_neighbors,
                   list_new_size,
                   data,
                   data_dim,
                   l2_norms,
                   metric,
                   dist_epilogue);

  __syncthreads();

  for (int step = 0; step < raft::ceildiv(list_new_size, num_warps); step++) {
    int idx_in_list = step * num_warps + tx / raft::warp_size();
    if (idx_in_list >= list_new_size) continue;
    auto min_elem = get_min_item(s_list[idx_in_list], idx_in_list, new_neighbors, s_distances);
    if (min_elem.id() < gridDim.x) {
      insert_to_global_graph(min_elem, s_list[idx_in_list], graph, dists, graph_width, locks);
    }
  }

  if (!list_old_size) return;

  __syncthreads();

  for (int i = tid; i < MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES; i += blockDim.x)
    s_distances[i] = 0.0f;

  __syncthreads();

  for (int step = 0; step < raft::ceildiv(data_dim, TILE_COL_WIDTH); step++) {
    int num_load_elems = (step == raft::ceildiv(data_dim, TILE_COL_WIDTH) - 1)
                           ? data_dim - step * TILE_COL_WIDTH
                           : TILE_COL_WIDTH;
    if (TILE_COL_WIDTH < data_dim) {
#pragma unroll
      for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; i++) {
        int idx = i * num_warps + warp_id;
        if (idx < list_new_size) {
          size_t neighbor_id = new_neighbors[idx];
          size_t idx_in_data = neighbor_id * data_dim;
          load_vec(s_nv[idx],
                   data + idx_in_data + step * TILE_COL_WIDTH,
                   num_load_elems,
                   TILE_COL_WIDTH,
                   lane_id);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; i++) {
      int idx = i * num_warps + warp_id;
      if (idx < list_old_size) {
        size_t neighbor_id = old_neighbors[idx];
        size_t idx_in_data = neighbor_id * data_dim;
        load_vec(s_ov[idx],
                 data + idx_in_data + step * TILE_COL_WIDTH,
                 num_load_elems,
                 TILE_COL_WIDTH,
                 lane_id);
      }
    }
    __syncthreads();

    // this is much faster than a warp-collaborative multiplication because MAX_NUM_BI_SAMPLES is
    // fixed and small (64)
    for (int i = threadIdx.x; i < MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES; i += blockDim.x) {
      int tmp_row = i / SKEWED_MAX_NUM_BI_SAMPLES;
      int tmp_col = i % SKEWED_MAX_NUM_BI_SAMPLES;
      if (tmp_row < list_new_size && tmp_col < list_old_size) {
        float acc = 0.0f;
        for (int d = 0; d < num_load_elems; d++) {
          float a = dtype_traits<Data_t>::to_float(s_nv[tmp_row][d]);
          float b = dtype_traits<Data_t>::to_float(s_ov[tmp_col][d]);
          acc += dist_acc(a, b);
        }
        s_distances[i] += acc;
      }
    }
    __syncthreads();
  }
  __syncthreads();

  calculate_metric(s_distances,
                   new_neighbors,
                   list_new_size,
                   old_neighbors,
                   list_old_size,
                   data,
                   data_dim,
                   l2_norms,
                   metric,
                   dist_epilogue);

  __syncthreads();

  for (int step = 0; step < raft::ceildiv(MAX_NUM_BI_SAMPLES * 2, num_warps); step++) {
    int idx_in_list = step * num_warps + tx / raft::warp_size();
    if (idx_in_list >= list_new_size && idx_in_list < MAX_NUM_BI_SAMPLES) continue;
    if (idx_in_list >= MAX_NUM_BI_SAMPLES + list_old_size && idx_in_list < MAX_NUM_BI_SAMPLES * 2)
      continue;
    ResultItem<Index_t> min_elem{std::numeric_limits<Index_t>::max(),
                                 std::numeric_limits<DistData_t>::max()};
    if (idx_in_list < MAX_NUM_BI_SAMPLES) {
      auto temp_min_item =
        get_min_item(s_list[idx_in_list], idx_in_list, old_neighbors, s_distances);
      if (temp_min_item.dist() < min_elem.dist()) { min_elem = temp_min_item; }
    } else {
      auto temp_min_item = get_min_item(
        s_list[idx_in_list], idx_in_list - MAX_NUM_BI_SAMPLES, new_neighbors, s_distances, false);
      if (temp_min_item.dist() < min_elem.dist()) { min_elem = temp_min_item; }
    }

    if (min_elem.id() < gridDim.x) {
      insert_to_global_graph(min_elem, s_list[idx_in_list], graph, dists, graph_width, locks);
    }
  }
#endif
}

// launch_bounds here denote BLOCK_SIZE = 512 and MIN_BLOCKS_PER_SM = 4
// Per
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications,
// MAX_RESIDENT_THREAD_PER_SM = BLOCK_SIZE * BLOCKS_PER_SM = 2048
// For architectures 750 and 860 (890), the values for MAX_RESIDENT_THREAD_PER_SM
// is 1024 and 1536 respectively, which means the bounds don't work anymore
// Used for fp32 data downcast to fp16, and all types using non-L1 distance metric.
template <typename Data_t,
          typename Index_t,
          typename ID_t = InternalID_t<Index_t>,
          typename DistEpilogue_t>
RAFT_KERNEL
#ifdef __CUDA_ARCH__
// Use minBlocksPerMultiprocessor = 4 on specific arches
#if (__CUDA_ARCH__) == 700 || (__CUDA_ARCH__) == 800 || (__CUDA_ARCH__) == 900 || \
  (__CUDA_ARCH__) == 1000
__launch_bounds__(BLOCK_SIZE, 4)
#else
__launch_bounds__(BLOCK_SIZE)
#endif
#endif
  local_join_kernel_wmma(const Index_t* graph_new,
                         const Index_t* rev_graph_new,
                         const int2* sizes_new,
                         const Index_t* graph_old,
                         const Index_t* rev_graph_old,
                         const int2* sizes_old,
                         const int width,
                         const Data_t* data,
                         const int data_dim,
                         ID_t* graph,
                         DistData_t* dists,
                         int graph_width,
                         int* locks,
                         DistData_t* l2_norms,
                         cuvs::distance::DistanceType metric,
                         DistEpilogue_t dist_epilogue)
{
#if (__CUDA_ARCH__ >= 700)
  using namespace nvcuda;
  __shared__ int s_list[MAX_NUM_BI_SAMPLES * 2];

  constexpr int APAD           = 8;
  constexpr int BPAD           = 8;
  constexpr int TILE_COL_WIDTH = 128;
  __shared__ __half s_nv[MAX_NUM_BI_SAMPLES][TILE_COL_WIDTH + APAD];  // New vectors
  __shared__ __half s_ov[MAX_NUM_BI_SAMPLES][TILE_COL_WIDTH + BPAD];  // Old vectors
  static_assert(sizeof(float) * MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES <=
                sizeof(__half) * MAX_NUM_BI_SAMPLES * (TILE_COL_WIDTH + BPAD));
  // s_distances: MAX_NUM_BI_SAMPLES x SKEWED_MAX_NUM_BI_SAMPLES, reuse the space of s_ov
  float* s_distances    = (float*)&s_ov[0][0];
  int* s_unique_counter = (int*)&s_ov[0][0];

  if (threadIdx.x == 0) {
    s_unique_counter[0] = 0;
    s_unique_counter[1] = 0;
  }

  Index_t* new_neighbors = s_list;
  Index_t* old_neighbors = s_list + MAX_NUM_BI_SAMPLES;

  size_t list_id      = blockIdx.x;
  int2 list_new_size2 = sizes_new[list_id];
  int list_new_size   = list_new_size2.x + list_new_size2.y;
  int2 list_old_size2 = sizes_old[list_id];
  int list_old_size   = list_old_size2.x + list_old_size2.y;

  if (!list_new_size) return;
  int tx = threadIdx.x;

  if (tx < list_new_size2.x) {
    new_neighbors[tx] = graph_new[list_id * width + tx];
  } else if (tx >= list_new_size2.x && tx < list_new_size) {
    new_neighbors[tx] = rev_graph_new[list_id * width + tx - list_new_size2.x];
  }

  if (tx < list_old_size2.x) {
    old_neighbors[tx] = graph_old[list_id * width + tx];
  } else if (tx >= list_old_size2.x && tx < list_old_size) {
    old_neighbors[tx] = rev_graph_old[list_id * width + tx - list_old_size2.x];
  }

  __syncthreads();

  remove_duplicates(new_neighbors,
                    list_new_size2.x,
                    new_neighbors + list_new_size2.x,
                    list_new_size2.y,
                    s_unique_counter[0],
                    0);

  remove_duplicates(old_neighbors,
                    list_old_size2.x,
                    old_neighbors + list_old_size2.x,
                    list_old_size2.y,
                    s_unique_counter[1],
                    1);
  __syncthreads();
  list_new_size = list_new_size2.x + s_unique_counter[0];
  list_old_size = list_old_size2.x + s_unique_counter[1];

  int warp_id             = threadIdx.x / raft::warp_size();
  int lane_id             = threadIdx.x % raft::warp_size();
  constexpr int num_warps = BLOCK_SIZE / raft::warp_size();

  int warp_id_y = warp_id / 4;
  int warp_id_x = warp_id % 4;

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  if (metric != cuvs::distance::DistanceType::BitwiseHamming) {
    wmma::fill_fragment(c_frag, 0.0);

    for (int step = 0; step < raft::ceildiv(data_dim, TILE_COL_WIDTH); step++) {
      int num_load_elems = (step == raft::ceildiv(data_dim, TILE_COL_WIDTH) - 1)
                             ? data_dim - step * TILE_COL_WIDTH
                             : TILE_COL_WIDTH;
#pragma unroll
      for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; i++) {
        int idx = i * num_warps + warp_id;
        if (idx < list_new_size) {
          size_t neighbor_id = new_neighbors[idx];
          size_t idx_in_data = neighbor_id * data_dim;
          // converted to fp16 on-the-fly while loading
          load_vec(s_nv[idx],
                   data + idx_in_data + step * TILE_COL_WIDTH,
                   num_load_elems,
                   TILE_COL_WIDTH,
                   lane_id);
        }
      }
      __syncthreads();

      for (int i = 0; i < TILE_COL_WIDTH / WMMA_K; i++) {
        wmma::load_matrix_sync(
          a_frag, s_nv[warp_id_y * WMMA_M] + i * WMMA_K, TILE_COL_WIDTH + APAD);
        wmma::load_matrix_sync(
          b_frag, s_nv[warp_id_x * WMMA_N] + i * WMMA_K, TILE_COL_WIDTH + BPAD);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
      }
    }

    wmma::store_matrix_sync(
      s_distances + warp_id_y * WMMA_M * SKEWED_MAX_NUM_BI_SAMPLES + warp_id_x * WMMA_N,
      c_frag,
      SKEWED_MAX_NUM_BI_SAMPLES,
      wmma::mem_row_major);
  }
  __syncthreads();

  calculate_metric(s_distances,
                   new_neighbors,
                   list_new_size,
                   new_neighbors,
                   list_new_size,
                   data,
                   data_dim,
                   l2_norms,
                   metric,
                   dist_epilogue);
  __syncthreads();

  for (int step = 0; step < raft::ceildiv(list_new_size, num_warps); step++) {
    int idx_in_list = step * num_warps + tx / raft::warp_size();
    if (idx_in_list >= list_new_size) continue;
    auto min_elem = get_min_item(s_list[idx_in_list], idx_in_list, new_neighbors, s_distances);
    if (min_elem.id() < gridDim.x) {
      insert_to_global_graph(min_elem, s_list[idx_in_list], graph, dists, graph_width, locks);
    }
  }

  if (!list_old_size) return;

  __syncthreads();

  if (metric != cuvs::distance::DistanceType::BitwiseHamming) {
    wmma::fill_fragment(c_frag, 0.0);
    for (int step = 0; step < raft::ceildiv(data_dim, TILE_COL_WIDTH); step++) {
      int num_load_elems = (step == raft::ceildiv(data_dim, TILE_COL_WIDTH) - 1)
                             ? data_dim - step * TILE_COL_WIDTH
                             : TILE_COL_WIDTH;
      if (TILE_COL_WIDTH < data_dim) {
#pragma unroll
        for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; i++) {
          int idx = i * num_warps + warp_id;
          if (idx < list_new_size) {
            size_t neighbor_id = new_neighbors[idx];
            size_t idx_in_data = neighbor_id * data_dim;
            load_vec(s_nv[idx],
                     data + idx_in_data + step * TILE_COL_WIDTH,
                     num_load_elems,
                     TILE_COL_WIDTH,
                     lane_id);
          }
        }
      }
#pragma unroll
      for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; i++) {
        int idx = i * num_warps + warp_id;
        if (idx < list_old_size) {
          size_t neighbor_id = old_neighbors[idx];
          size_t idx_in_data = neighbor_id * data_dim;
          load_vec(s_ov[idx],
                   data + idx_in_data + step * TILE_COL_WIDTH,
                   num_load_elems,
                   TILE_COL_WIDTH,
                   lane_id);
        }
      }
      __syncthreads();

      for (int i = 0; i < TILE_COL_WIDTH / WMMA_K; i++) {
        wmma::load_matrix_sync(
          a_frag, s_nv[warp_id_y * WMMA_M] + i * WMMA_K, TILE_COL_WIDTH + APAD);
        wmma::load_matrix_sync(
          b_frag, s_ov[warp_id_x * WMMA_N] + i * WMMA_K, TILE_COL_WIDTH + BPAD);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
      }
    }

    wmma::store_matrix_sync(
      s_distances + warp_id_y * WMMA_M * SKEWED_MAX_NUM_BI_SAMPLES + warp_id_x * WMMA_N,
      c_frag,
      SKEWED_MAX_NUM_BI_SAMPLES,
      wmma::mem_row_major);
    __syncthreads();
  }

  calculate_metric(s_distances,
                   new_neighbors,
                   list_new_size,
                   old_neighbors,
                   list_old_size,
                   data,
                   data_dim,
                   l2_norms,
                   metric,
                   dist_epilogue);

  __syncthreads();

  for (int step = 0; step < raft::ceildiv(MAX_NUM_BI_SAMPLES * 2, num_warps); step++) {
    int idx_in_list = step * num_warps + tx / raft::warp_size();
    if (idx_in_list >= list_new_size && idx_in_list < MAX_NUM_BI_SAMPLES) continue;
    if (idx_in_list >= MAX_NUM_BI_SAMPLES + list_old_size && idx_in_list < MAX_NUM_BI_SAMPLES * 2)
      continue;
    ResultItem<Index_t> min_elem{std::numeric_limits<Index_t>::max(),
                                 std::numeric_limits<DistData_t>::max()};
    if (idx_in_list < MAX_NUM_BI_SAMPLES) {
      auto temp_min_item =
        get_min_item(s_list[idx_in_list], idx_in_list, old_neighbors, s_distances);
      if (temp_min_item.dist() < min_elem.dist()) { min_elem = temp_min_item; }
    } else {
      auto temp_min_item = get_min_item(
        s_list[idx_in_list], idx_in_list - MAX_NUM_BI_SAMPLES, new_neighbors, s_distances, false);
      if (temp_min_item.dist() < min_elem.dist()) { min_elem = temp_min_item; }
    }

    if (min_elem.id() < gridDim.x) {
      insert_to_global_graph(min_elem, s_list[idx_in_list], graph, dists, graph_width, locks);
    }
  }
#endif
}

namespace {
template <typename Index_t>
int insert_to_ordered_list(InternalID_t<Index_t>* list,
                           DistData_t* dist_list,
                           const int width,
                           const InternalID_t<Index_t> neighb_id,
                           const DistData_t dist)
{
  if (dist > dist_list[width - 1]) { return width; }

  int idx_insert      = width;
  bool position_found = false;
  for (int i = 0; i < width; i++) {
    if (list[i].id() == neighb_id.id()) { return width; }
    if (!position_found && dist_list[i] > dist) {
      idx_insert     = i;
      position_found = true;
    }
  }
  if (idx_insert == width) return idx_insert;

  memmove(list + idx_insert + 1, list + idx_insert, sizeof(*list) * (width - idx_insert - 1));
  memmove(dist_list + idx_insert + 1,
          dist_list + idx_insert,
          sizeof(*dist_list) * (width - idx_insert - 1));

  list[idx_insert]      = neighb_id;
  dist_list[idx_insert] = dist;
  return idx_insert;
};

}  // namespace

template <typename Index_t>
GnndGraph<Index_t>::GnndGraph(raft::resources const& res,
                              const size_t nrow,
                              const size_t node_degree,
                              const size_t internal_node_degree,
                              const size_t num_samples)
  : res(res),
    nrow(nrow),
    node_degree(node_degree),
    num_samples(num_samples),
    bloom_filter(nrow, internal_node_degree / segment_size, 3),
    h_dists{raft::make_host_matrix<DistData_t, size_t, raft::row_major>(nrow, node_degree)},
    h_graph_new{raft::make_pinned_matrix<Index_t, size_t, raft::row_major>(res, nrow, num_samples)},
    h_list_sizes_new{raft::make_pinned_vector<int2, size_t>(res, nrow)},
    h_graph_old{raft::make_pinned_matrix<Index_t, size_t, raft::row_major>(res, nrow, num_samples)},
    h_list_sizes_old{raft::make_pinned_vector<int2, size_t>(res, nrow)}
{
  // node_degree must be a multiple of segment_size;
  assert(node_degree % segment_size == 0);
  assert(internal_node_degree % segment_size == 0);

  num_segments = node_degree / segment_size;
  // To save the CPU memory, graph should be allocated by external function
  h_graph = nullptr;
}

// This is the only operation on the CPU that cannot be overlapped.
// So it should be as fast as possible.
template <typename Index_t>
void GnndGraph<Index_t>::sample_graph_new(InternalID_t<Index_t>* new_neighbors, const size_t width)
{
  std::fill_n(h_graph_new.data_handle(), nrow * num_samples, std::numeric_limits<Index_t>::max());
#pragma omp parallel for
  for (size_t i = 0; i < nrow; i++) {
    auto list_new                       = h_graph_new.data_handle() + i * num_samples;
    h_list_sizes_new.data_handle()[i].x = 0;
    h_list_sizes_new.data_handle()[i].y = 0;

    for (size_t j = 0; j < width; j++) {
      auto new_neighb_id = new_neighbors[i * width + j].id();
      if ((size_t)new_neighb_id >= nrow) break;
      if (bloom_filter.check(i, new_neighb_id)) { continue; }
      bloom_filter.add(i, new_neighb_id);
      new_neighbors[i * width + j].mark_old();
      list_new[h_list_sizes_new.data_handle()[i].x++] = new_neighb_id;
      if (h_list_sizes_new.data_handle()[i].x == num_samples) break;
    }
  }
}

template <typename Index_t>
void GnndGraph<Index_t>::init_random_graph()
{
  for (size_t seg_idx = 0; seg_idx < static_cast<size_t>(num_segments); seg_idx++) {
    // random sequence (range: 0~nrow)
    // segment_x stores neighbors which id % num_segments == x
    std::vector<Index_t> rand_seq((nrow + num_segments - 1) / num_segments);
    std::iota(rand_seq.begin(), rand_seq.end(), 0);
    auto gen = std::default_random_engine{seg_idx};
    std::shuffle(rand_seq.begin(), rand_seq.end(), gen);

#pragma omp parallel for
    for (size_t i = 0; i < nrow; i++) {
      size_t base_idx         = i * node_degree + seg_idx * segment_size;
      auto h_neighbor_list    = h_graph + base_idx;
      auto h_dist_list        = h_dists.data_handle() + base_idx;
      size_t idx              = base_idx;
      size_t self_in_this_seg = 0;
      for (size_t j = 0; j < static_cast<size_t>(segment_size); j++) {
        Index_t id = rand_seq[idx % rand_seq.size()] * num_segments + seg_idx;
        if ((size_t)id == i) {
          idx++;
          id               = rand_seq[idx % rand_seq.size()] * num_segments + seg_idx;
          self_in_this_seg = 1;
        }

        h_neighbor_list[j].id_with_flag() =
          j < (rand_seq.size() - self_in_this_seg) && size_t(id) < nrow
            ? id
            : std::numeric_limits<Index_t>::max();
        h_dist_list[j] = std::numeric_limits<DistData_t>::max();
        idx++;
      }
    }
  }
}

template <typename Index_t>
void GnndGraph<Index_t>::sample_graph(bool sample_new)
{
  std::fill_n(h_graph_old.data_handle(), nrow * num_samples, std::numeric_limits<Index_t>::max());
  if (sample_new) {
    std::fill_n(h_graph_new.data_handle(), nrow * num_samples, std::numeric_limits<Index_t>::max());
  }

#pragma omp parallel for
  for (size_t i = 0; i < nrow; i++) {
    h_list_sizes_old.data_handle()[i].x = 0;
    h_list_sizes_old.data_handle()[i].y = 0;
    h_list_sizes_new.data_handle()[i].x = 0;
    h_list_sizes_new.data_handle()[i].y = 0;

    auto list     = h_graph + i * node_degree;
    auto list_old = h_graph_old.data_handle() + i * num_samples;
    auto list_new = h_graph_new.data_handle() + i * num_samples;
    for (int j = 0; j < segment_size; j++) {
      for (int k = 0; k < num_segments; k++) {
        auto neighbor = list[k * segment_size + j];
        if ((size_t)neighbor.id() >= nrow) continue;
        if (!neighbor.is_new()) {
          if (h_list_sizes_old.data_handle()[i].x < num_samples) {
            list_old[h_list_sizes_old.data_handle()[i].x++] = neighbor.id();
          }
        } else if (sample_new) {
          if (h_list_sizes_new.data_handle()[i].x < num_samples) {
            list[k * segment_size + j].mark_old();
            list_new[h_list_sizes_new.data_handle()[i].x++] = neighbor.id();
          }
        }
        if (h_list_sizes_old.data_handle()[i].x == num_samples &&
            h_list_sizes_new.data_handle()[i].x == num_samples) {
          break;
        }
      }
      if (h_list_sizes_old.data_handle()[i].x == num_samples &&
          h_list_sizes_new.data_handle()[i].x == num_samples) {
        break;
      }
    }
  }
}

template <typename Index_t>
void GnndGraph<Index_t>::update_graph(const InternalID_t<Index_t>* new_neighbors,
                                      const DistData_t* new_dists,
                                      const size_t width,
                                      std::atomic<int64_t>& update_counter)
{
#pragma omp parallel for
  for (size_t i = 0; i < nrow; i++) {
    for (size_t j = 0; j < width; j++) {
      auto new_neighb_id = new_neighbors[i * width + j];
      auto new_dist      = new_dists[i * width + j];
      if (new_dist == std::numeric_limits<DistData_t>::max()) break;
      if ((size_t)new_neighb_id.id() == i) continue;
      int seg_idx    = new_neighb_id.id() % num_segments;
      auto list      = h_graph + i * node_degree + seg_idx * segment_size;
      auto dist_list = h_dists.data_handle() + i * node_degree + seg_idx * segment_size;
      int insert_pos =
        insert_to_ordered_list(list, dist_list, segment_size, new_neighb_id, new_dist);
      if (i % counter_interval == 0 && insert_pos != segment_size) { update_counter++; }
    }
  }
}

template <typename Index_t>
void GnndGraph<Index_t>::sort_lists()
{
#pragma omp parallel for
  for (size_t i = 0; i < nrow; i++) {
    std::vector<std::pair<DistData_t, Index_t>> new_list;
    for (size_t j = 0; j < node_degree; j++) {
      new_list.emplace_back(h_dists.data_handle()[i * node_degree + j],
                            h_graph[i * node_degree + j].id());
    }
    std::sort(new_list.begin(), new_list.end());
    for (size_t j = 0; j < node_degree; j++) {
      h_graph[i * node_degree + j].id_with_flag() = new_list[j].second;
      h_dists.data_handle()[i * node_degree + j]  = new_list[j].first;
    }
  }
}

template <typename Index_t>
void GnndGraph<Index_t>::clear()
{
  bloom_filter.clear();
}

template <typename Index_t>
GnndGraph<Index_t>::~GnndGraph()
{
  assert(h_graph == nullptr);
}

template <typename Data_t, typename Index_t>
GNND<Data_t, Index_t>::GNND(raft::resources const& res, const BuildConfig& build_config)
  : res(res),
    build_config_(build_config),
    graph_(res,
           build_config.max_dataset_size,
           align32::roundUp(build_config.node_degree),
           align32::roundUp(build_config.internal_node_degree ? build_config.internal_node_degree
                                                              : build_config.node_degree),
           NUM_SAMPLES),
    nrow_(build_config.max_dataset_size),
    ndim_(build_config.dataset_dim),
    l2_norms_{raft::make_device_vector<DistData_t, size_t>(res, 0)},
    graph_buffer_{
      raft::make_device_matrix<ID_t, size_t, raft::row_major>(res, nrow_, DEGREE_ON_DEVICE)},
    dists_buffer_{
      raft::make_device_matrix<DistData_t, size_t, raft::row_major>(res, nrow_, DEGREE_ON_DEVICE)},
    graph_host_buffer_{
      raft::make_pinned_matrix<ID_t, size_t, raft::row_major>(res, nrow_, DEGREE_ON_DEVICE)},
    dists_host_buffer_{
      raft::make_pinned_matrix<DistData_t, size_t, raft::row_major>(res, nrow_, DEGREE_ON_DEVICE)},
    d_locks_{raft::make_device_vector<int, size_t>(res, nrow_)},
    h_rev_graph_new_{
      raft::make_pinned_matrix<Index_t, size_t, raft::row_major>(res, nrow_, NUM_SAMPLES)},
    h_graph_old_(
      raft::make_pinned_matrix<Index_t, size_t, raft::row_major>(res, nrow_, NUM_SAMPLES)),
    h_rev_graph_old_{
      raft::make_pinned_matrix<Index_t, size_t, raft::row_major>(res, nrow_, NUM_SAMPLES)},
    d_list_sizes_new_{raft::make_device_vector<int2, size_t>(res, nrow_)},
    d_list_sizes_old_{raft::make_device_vector<int2, size_t>(res, nrow_)}
{
  static_assert(NUM_SAMPLES <= 32);

  raft::matrix::fill(res, dists_buffer_.view(), std::numeric_limits<float>::max());
  auto graph_buffer_view = raft::make_device_matrix_view<Index_t, int64_t>(
    reinterpret_cast<Index_t*>(graph_buffer_.data_handle()), nrow_, DEGREE_ON_DEVICE);
  raft::matrix::fill(res, graph_buffer_view, std::numeric_limits<Index_t>::max());
  raft::matrix::fill(res, d_locks_.view(), 0);

  if (build_config.metric == cuvs::distance::DistanceType::L2Expanded ||
      build_config.metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
      build_config.metric == cuvs::distance::DistanceType::CosineExpanded) {
    // for device memory efficiency, we do not allocate a separate array for the data
    // to normalize the data when using CosineExpanded metric. Instead, we use the l2_norms_ vector
    // and compute inside the calculate_metric kernel.
    l2_norms_ = raft::make_device_vector<DistData_t, size_t>(res, nrow_);
  }
};

template <typename Data_t, typename Index_t>
void GNND<Data_t, Index_t>::reset(raft::resources const& res)
{
  raft::matrix::fill(res, dists_buffer_.view(), std::numeric_limits<float>::max());
  auto graph_buffer_view = raft::make_device_matrix_view<Index_t, int64_t>(
    reinterpret_cast<Index_t*>(graph_buffer_.data_handle()), nrow_, DEGREE_ON_DEVICE);
  raft::matrix::fill(res, graph_buffer_view, std::numeric_limits<Index_t>::max());
  raft::matrix::fill(res, d_locks_.view(), 0);
}

template <typename Data_t, typename Index_t>
void GNND<Data_t, Index_t>::add_reverse_edges(Index_t* graph_ptr,
                                              Index_t* h_rev_graph_ptr,
                                              Index_t* d_rev_graph_ptr,
                                              int2* list_sizes,
                                              cudaStream_t stream)
{
  raft::matrix::fill(
    res,
    raft::make_device_matrix_view<Index_t, int64_t>(d_rev_graph_ptr, nrow_, DEGREE_ON_DEVICE),
    std::numeric_limits<Index_t>::max());
  add_rev_edges_kernel<<<nrow_, raft::warp_size(), 0, stream>>>(
    graph_ptr, d_rev_graph_ptr, NUM_SAMPLES, list_sizes);
  raft::copy(res,
             raft::make_host_vector_view(h_rev_graph_ptr, nrow_ * NUM_SAMPLES),
             raft::make_device_vector_view(d_rev_graph_ptr, nrow_ * NUM_SAMPLES));
}

template <typename Data_t, typename Index_t>
template <typename DistEpilogue_t>
void GNND<Data_t, Index_t>::local_join(cudaStream_t stream, DistEpilogue_t dist_epilogue)
{
  raft::matrix::fill(res, dists_buffer_.view(), std::numeric_limits<float>::max());

  // Kernel dispatch logic, based on the effective distance-computation dtype (which depends on
  // the input dtype and dist_comp_dtype):
  //   fp32 dist (only fp32 input, dist_comp_dtype == FP32 or AUTO with dim <= 16) -> SIMT: scalar
  //     element-wise distance computation in fp32.
  //   fp16 dist (everything else: fp16/int8/uint8 input, or fp32 input with dist_comp_dtype ==
  //     FP16 or AUTO with dim > 16) -> WMMA (tensor-core accelerated dot product). Non-fp16
  //     dtypes are converted to fp16 on-the-fly while loading into shared memory; for fp32 host
  //     input this conversion happens earlier at copy-in time (see d_data_half_).
  //   L1 distance for any input -> SIMT (L1 needs element-wise ops, can't use tensor cores).
  using DCT = cuvs::neighbors::nn_descent::DIST_COMP_DTYPE;
  bool use_fp16_dist =
    std::is_same_v<input_t, float> && (build_config_.dist_comp_dtype == DCT::FP16 ||
                                       (build_config_.dist_comp_dtype == DCT::AUTO && ndim_ > 16));
  bool use_simt = (std::is_same_v<input_t, float> && !use_fp16_dist) ||
                  build_config_.metric == cuvs::distance::DistanceType::L1;

  auto launch_kernel = [&](auto* typed_ptr) {
    if (use_simt) {
      local_join_kernel_simt<<<nrow_, BLOCK_SIZE, 0, stream>>>(graph_.h_graph_new.data_handle(),
                                                               h_rev_graph_new_.data_handle(),
                                                               d_list_sizes_new_.data_handle(),
                                                               h_graph_old_.data_handle(),
                                                               h_rev_graph_old_.data_handle(),
                                                               d_list_sizes_old_.data_handle(),
                                                               NUM_SAMPLES,
                                                               typed_ptr,
                                                               ndim_,
                                                               graph_buffer_.data_handle(),
                                                               dists_buffer_.data_handle(),
                                                               DEGREE_ON_DEVICE,
                                                               d_locks_.data_handle(),
                                                               l2_norms_.data_handle(),
                                                               build_config_.metric,
                                                               dist_epilogue);
    } else {
      local_join_kernel_wmma<<<nrow_, BLOCK_SIZE, 0, stream>>>(graph_.h_graph_new.data_handle(),
                                                               h_rev_graph_new_.data_handle(),
                                                               d_list_sizes_new_.data_handle(),
                                                               h_graph_old_.data_handle(),
                                                               h_rev_graph_old_.data_handle(),
                                                               d_list_sizes_old_.data_handle(),
                                                               NUM_SAMPLES,
                                                               typed_ptr,
                                                               ndim_,
                                                               graph_buffer_.data_handle(),
                                                               dists_buffer_.data_handle(),
                                                               DEGREE_ON_DEVICE,
                                                               d_locks_.data_handle(),
                                                               l2_norms_.data_handle(),
                                                               build_config_.metric,
                                                               dist_epilogue);
    }
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  };

  if (d_data_half_.has_value()) {
    // Host fp32 input was downcast to a device-side fp16 buffer because distances are computed in
    // fp16 (dist_comp_dtype == FP16, or AUTO with dim > 16).
    launch_kernel(static_cast<const half*>(d_data_ptr_));
  } else {
    // Data stored as input_t: device data used directly, or host data copied as-is.
    launch_kernel(static_cast<const input_t*>(d_data_ptr_));
  }
}

template <typename Data_t, typename Index_t>
template <typename DistEpilogue_t>
void GNND<Data_t, Index_t>::build(Data_t* data,
                                  const Index_t nrow,
                                  Index_t* output_graph,
                                  bool return_distances,
                                  DistData_t* output_distances,
                                  DistEpilogue_t dist_epilogue)
{
  using input_t = typename std::remove_const<Data_t>::type;

  if (build_config_.metric == distance::DistanceType::BitwiseHamming &&
      !(std::is_same_v<input_t, uint8_t> || std::is_same_v<input_t, int8_t>)) {
    RAFT_FAIL(
      "Data type needs to be int8 or uint8 for NN Descent to run with BitwiseHamming distance.");
  }

  cudaStream_t stream = raft::resource::get_cuda_stream(res);
  nrow_               = nrow;
  graph_.nrow         = nrow;
  graph_.bloom_filter.set_nrow(nrow);
  update_counter_ = 0;
  graph_.h_graph  = (InternalID_t<Index_t>*)output_graph;

  d_data_ptr_ = nullptr;

  cudaPointerAttributes data_ptr_attr;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&data_ptr_attr, data));
  bool data_on_device = (data_ptr_attr.type == cudaMemoryTypeDevice);

  bool needs_l2_norms = build_config_.metric == cuvs::distance::DistanceType::L2Expanded ||
                        build_config_.metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                        build_config_.metric == cuvs::distance::DistanceType::CosineExpanded;

  // For fp32 host input, downcast to a device-side fp16 buffer when distance computation will be
  // done in fp16 anyway: dispatch matches the SIMT/WMMA decision in local_join() (FP16 explicit, or
  // AUTO with dim > 16).
  using DCT = cuvs::neighbors::nn_descent::DIST_COMP_DTYPE;
  bool fp32_input_uses_fp16_dist =
    std::is_same_v<input_t, float> &&
    (build_config_.dist_comp_dtype == DCT::FP16 ||
     (build_config_.dist_comp_dtype == DCT::AUTO && build_config_.dataset_dim > 16));
  bool downcast_host_data = !data_on_device && fp32_input_uses_fp16_dist;

  if (data_on_device) {
    // When user-given data is on device, we use it directly. This can be any type (fp32, fp16,
    // int8, uint8)
    d_data_ptr_ = data;
  } else if (downcast_host_data) {
    // When user-given data is fp32 host data and distances will be computed in fp16, we allocate
    // an fp16 device buffer and downcast at copy-in time. Storing the dataset on device in fp16
    // (instead of fp32) for this path halves both the device memory footprint and the per-
    // iteration read bandwidth of the WMMA kernel.
    if (!d_data_half_.has_value()) {
      d_data_half_.emplace(raft::make_device_matrix<half, size_t, raft::row_major>(
        res, build_config_.max_dataset_size, build_config_.dataset_dim));
    }
    size_t batch_size = 100000;
    auto vec_batches  = cuvs::spatial::knn::detail::utils::make_batch_load_iterator<Data_t>(
      res,
      data,
      static_cast<int64_t>(nrow_),
      static_cast<int64_t>(build_config_.dataset_dim),
      batch_size,
      stream);
    constexpr int TPB = 256;
    size_t l2_norms_smem =
      needs_l2_norms ? configure_compute_l2_norms_kernel_smem<float>(build_config_.dataset_dim)
                     : 0;
    for (auto const& batch : vec_batches) {
      size_t n_elems    = batch.size() * build_config_.dataset_dim;
      int num_blocks    = raft::ceildiv(n_elems, static_cast<size_t>(TPB));
      size_t dst_offset = batch.offset() * build_config_.dataset_dim;
      if (needs_l2_norms) {
        // Compute l2 norms on the fp32 batches before they're downcast to fp16.
        compute_l2_norms_kernel<<<batch.size(), raft::warp_size(), l2_norms_smem, stream>>>(
          batch.data(), build_config_.dataset_dim, l2_norms_.data_handle() + batch.offset());
        RAFT_CUDA_TRY(cudaPeekAtLastError());
      }
      convert_copy_kernel<<<num_blocks, TPB, 0, stream>>>(
        batch.data(), d_data_half_.value().data_handle() + dst_offset, n_elems);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
    d_data_ptr_ = d_data_half_.value().data_handle();
  } else {
    // Other cases: user-given data is not device-accessible, but we don't need a precision
    // conversion. Allocate a device buffer in input_t and copy as-is.
    if (!d_data_direct_.has_value()) {
      d_data_direct_.emplace(raft::make_device_matrix<input_t, size_t, raft::row_major>(
        res, build_config_.max_dataset_size, build_config_.dataset_dim));
    }
    raft::copy(d_data_direct_.value().data_handle(),
               data,
               static_cast<size_t>(nrow_) * build_config_.dataset_dim,
               stream);
    d_data_ptr_ = d_data_direct_.value().data_handle();
  }

  if (needs_l2_norms && !downcast_host_data) {
    auto smem = configure_compute_l2_norms_kernel_smem<input_t>(build_config_.dataset_dim);
    compute_l2_norms_kernel<<<nrow_, raft::warp_size(), smem, stream>>>(
      static_cast<const input_t*>(d_data_ptr_), build_config_.dataset_dim, l2_norms_.data_handle());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    raft::resource::sync_stream(res);
  }

  graph_.clear();
  graph_.init_random_graph();
  graph_.sample_graph(true);

  auto update_and_sample = [&](bool update_graph) {
    if (update_graph) {
      update_counter_ = 0;
      graph_.update_graph(graph_host_buffer_.data_handle(),
                          dists_host_buffer_.data_handle(),
                          DEGREE_ON_DEVICE,
                          update_counter_);
      if (update_counter_ < build_config_.termination_threshold * nrow_ *
                              build_config_.dataset_dim / counter_interval) {
        update_counter_ = -1;
      }
    }
    graph_.sample_graph(false);
  };

  for (size_t it = 0; it < build_config_.max_iterations; it++) {
    raft::copy(res, d_list_sizes_new_.view(), graph_.h_list_sizes_new.view());
    raft::copy(res, h_graph_old_.view(), graph_.h_graph_old.view());
    raft::copy(res, d_list_sizes_old_.view(), graph_.h_list_sizes_old.view());
    raft::resource::sync_stream(res);

    std::thread update_and_sample_thread(update_and_sample, it);

    RAFT_LOG_DEBUG("# GNND iteration: %lu / %lu", it + 1, build_config_.max_iterations);

    // Reuse dists_buffer_ to save GPU memory. graph_buffer_ cannot be reused, because it
    // contains some information for local_join.
    static_assert(DEGREE_ON_DEVICE * sizeof(*(dists_buffer_.data_handle())) >=
                  NUM_SAMPLES * sizeof(*(graph_buffer_.data_handle())));
    add_reverse_edges(graph_.h_graph_new.data_handle(),
                      h_rev_graph_new_.data_handle(),
                      (Index_t*)dists_buffer_.data_handle(),
                      d_list_sizes_new_.data_handle(),
                      stream);
    add_reverse_edges(h_graph_old_.data_handle(),
                      h_rev_graph_old_.data_handle(),
                      (Index_t*)dists_buffer_.data_handle(),
                      d_list_sizes_old_.data_handle(),
                      stream);

    // Tensor operations from `mma.h` are guarded with archicteture
    // __CUDA_ARCH__ >= 700. Since RAFT supports compilation for ARCH 600,
    // we need to ensure that `local_join_kernel` (which uses tensor) operations
    // is not only not compiled, but also a runtime error is presented to the user
    auto kernel       = compute_l2_norms_kernel<input_t>;
    void* kernel_ptr  = reinterpret_cast<void*>(kernel);
    auto runtime_arch = raft::util::arch::kernel_virtual_arch(kernel_ptr);
    auto wmma_range =
      raft::util::arch::SM_range(raft::util::arch::SM_70(), raft::util::arch::SM_future());

    if (wmma_range.contains(runtime_arch)) {
      local_join(stream, dist_epilogue);
    } else {
      THROW("NN_DESCENT cannot be run for __CUDA_ARCH__ < 700");
    }

    update_and_sample_thread.join();

    if (update_counter_ == -1) { break; }
    raft::copy(res, graph_host_buffer_.view(), graph_buffer_.view());
    raft::copy(res, dists_host_buffer_.view(), dists_buffer_.view());
    raft::resource::sync_stream(res);

    graph_.sample_graph_new(graph_host_buffer_.data_handle(), DEGREE_ON_DEVICE);
  }

  graph_.update_graph(graph_host_buffer_.data_handle(),
                      dists_host_buffer_.data_handle(),
                      DEGREE_ON_DEVICE,
                      update_counter_);
  raft::resource::sync_stream(res);
  graph_.sort_lists();

  // Reuse graph_.h_dists as the buffer for shrink the lists in graph
  static_assert(sizeof(decltype(*(graph_.h_dists.data_handle()))) >= sizeof(Index_t));

  if (return_distances) {
    auto graph_h_dists = raft::make_host_matrix<DistData_t, int64_t, raft::row_major>(
      nrow_, build_config_.output_graph_degree);

// slice on host
#pragma omp parallel for
    for (size_t i = 0; i < (size_t)nrow_; i++) {
      for (size_t j = 0; j < build_config_.output_graph_degree; j++) {
        graph_h_dists(i, j) = graph_.h_dists(i, j);
      }
    }
    raft::copy(
      res,
      raft::make_device_vector_view(output_distances, nrow_ * build_config_.output_graph_degree),
      raft::make_host_vector_view(graph_h_dists.data_handle(),
                                  nrow_ * build_config_.output_graph_degree));

    auto output_dist_view = raft::make_device_matrix_view<DistData_t, int64_t, raft::row_major>(
      output_distances, nrow_, build_config_.output_graph_degree);
    // distance post-processing
    bool can_postprocess_dist = std::is_same_v<DistEpilogue_t, raft::identity_op>;
    if (build_config_.metric == cuvs::distance::DistanceType::L2SqrtExpanded &&
        can_postprocess_dist) {
      raft::linalg::map(
        res, output_dist_view, raft::sqrt_op{}, raft::make_const_mdspan(output_dist_view));
    } else if (!cuvs::distance::is_min_close(build_config_.metric) && can_postprocess_dist) {
      // revert negated innerproduct
      raft::linalg::map(res,
                        output_dist_view,
                        raft::mul_const_op<DistData_t>(-1),
                        raft::make_const_mdspan(output_dist_view));
    }
    raft::resource::sync_stream(res);
  }

  Index_t* graph_shrink_buffer = (Index_t*)graph_.h_dists.data_handle();

#pragma omp parallel for
  for (size_t i = 0; i < (size_t)nrow_; i++) {
    for (size_t j = 0; j < build_config_.node_degree; j++) {
      size_t idx = i * graph_.node_degree + j;
      int id     = graph_.h_graph[idx].id();
      if (id < static_cast<int>(nrow_)) {
        graph_shrink_buffer[i * build_config_.node_degree + j] = id;
      } else {
        graph_shrink_buffer[i * build_config_.node_degree + j] =
          cuvs::neighbors::detail::device::xorshift64(idx) % nrow_;
      }
    }
  }
  graph_.h_graph = nullptr;

#pragma omp parallel for
  for (size_t i = 0; i < (size_t)nrow_; i++) {
    for (size_t j = 0; j < build_config_.node_degree; j++) {
      output_graph[i * build_config_.node_degree + j] =
        graph_shrink_buffer[i * build_config_.node_degree + j];
    }
  }
}

template <typename T,
          typename IdxT = uint32_t,
          typename Accessor =
            raft::host_device_accessor<cuda::std::default_accessor<T>, raft::memory_type::host>>
void build(raft::resources const& res,
           const index_params& params,
           raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
           index<IdxT>& idx)
{
  size_t extended_graph_degree, graph_degree;
  auto build_config = get_build_config(res,
                                       params,
                                       static_cast<size_t>(dataset.extent(0)),
                                       static_cast<size_t>(dataset.extent(1)),
                                       idx.metric(),
                                       extended_graph_degree,
                                       graph_degree);

  auto int_graph =
    raft::make_host_matrix<int, int64_t, raft::row_major>(dataset.extent(0), extended_graph_degree);

  GNND<const T, int> nnd(res, build_config);

  if (idx.distances().has_value() || !params.return_distances) {
    nnd.build(dataset.data_handle(),
              dataset.extent(0),
              int_graph.data_handle(),
              params.return_distances,
              idx.distances()
                .value_or(raft::make_device_matrix<float, int64_t>(res, 0, 0).view())
                .data_handle());
  } else {
    RAFT_EXPECTS(!params.return_distances,
                 "Distance view not allocated. Using return_distances set to true requires "
                 "distance view to be allocated.");
  }

#pragma omp parallel for
  for (size_t i = 0; i < static_cast<size_t>(dataset.extent(0)); i++) {
    for (size_t j = 0; j < graph_degree; j++) {
      auto graph                  = idx.graph().data_handle();
      graph[i * graph_degree + j] = int_graph.data_handle()[i * extended_graph_degree + j];
    }
  }
}

template <typename T,
          typename IdxT = uint32_t,
          typename Accessor =
            raft::host_device_accessor<cuda::std::default_accessor<T>, raft::memory_type::host>>
index<IdxT> build(
  raft::resources const& res,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset)
{
  size_t intermediate_degree = params.intermediate_graph_degree;
  size_t graph_degree        = params.graph_degree;

  if (intermediate_degree < graph_degree) {
    RAFT_LOG_WARN(
      "Graph degree (%lu) cannot be larger than intermediate graph degree (%lu), reducing "
      "graph_degree.",
      graph_degree,
      intermediate_degree);
    graph_degree = intermediate_degree;
  }

  index<IdxT> idx{res,
                  dataset.extent(0),
                  static_cast<int64_t>(graph_degree),
                  params.return_distances,
                  params.metric};

  build(res, params, dataset, idx);

  return idx;
}

}  // namespace cuvs::neighbors::nn_descent::detail
