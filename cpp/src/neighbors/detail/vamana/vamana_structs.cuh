/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <float.h>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/warp_primitives.cuh>

#include <cuvs/distance/distance.hpp>

namespace cuvs::neighbors::vamana::detail {

/* @defgroup vamana_structures vamana structures
 * @{
 */

#define FULL_BITMASK 0xFFFFFFFF

// Warp stride for per-warp distance reduction (GreedySearch uses multiple warps per block).
static constexpr int VAMANA_WARP_SIZE = 32;

// vamana fp16 instantiations use CUDA's half type (alias of __half on device).
template <typename T>
inline constexpr bool is_cuda_fp16_v = std::is_same_v<std::remove_cv_t<T>, half>;

// GreedySearch promotes fp16 queries to float in shared memory for distance reuse.
template <typename T>
struct greedy_search_query_coord {
  using type = std::conditional_t<is_cuda_fp16_v<T>, float, T>;
};

// Wide vectors: store cached GreedySearch query coords as __half in smem (dim >= 512 only).
// Half dataset is excluded: it already caches query as float (promote once); fp16 smem would
// re-widen on every distance and lose the vectorized float-query vs half-neighbor path.
static constexpr int kGreedySearchFp16QuerySmemMinDim = 512;

template <typename T>
__host__ __device__ inline bool greedy_search_use_fp16_query_smem(int dim)
{
  return dim >= kGreedySearchFp16QuerySmemMinDim && !is_cuda_fp16_v<T>;
}

template <typename T>
__host__ __device__ inline int greedy_search_query_smem_elem_size(int dim)
{
  if (greedy_search_use_fp16_query_smem<T>(dim)) { return static_cast<int>(sizeof(__half)); }
  return static_cast<int>(sizeof(typename greedy_search_query_coord<T>::type));
}

// Currently supported values for graph_degree.
static const int DEGREE_SIZES[4] = {32, 64, 128, 256};

// Object used to store id,distance combination graph construction operations
template <typename IdxT, typename accT>
struct __align__(16) DistPair {
  accT dist;
  IdxT idx;
};

// Swap the values of two DistPair<SUMTYPE> objects
template <typename IdxT, typename accT>
__device__ __host__ void swap(DistPair<IdxT, accT>* a, DistPair<IdxT, accT>* b)
{
  DistPair<IdxT, accT> temp;
  temp.dist = a->dist;
  temp.idx  = a->idx;
  a->dist   = b->dist;
  a->idx    = b->idx;
  b->dist   = temp.dist;
  b->idx    = temp.idx;
}

// Structure to sort by distance
template <typename IdxT, typename accT>
struct CmpDist {
  __host__ __device__ bool operator()(const DistPair<IdxT, accT>& lhs,
                                      const DistPair<IdxT, accT>& rhs)
  {
    return lhs.dist < rhs.dist;
  }
};

// Used to sort reverse edges by destination
template <typename IdxT>
struct CmpEdge {
  __host__ __device__ bool operator()(const IdxT& lhs, const IdxT& rhs) { return lhs < rhs; }
};

/*********************************************************************
 * Object representing a Dim-dimensional point, with each coordinate
 * represented by a element of datatype T
 * Note, memory is allocated separately and coords set to offsets
 *********************************************************************/
template <typename T, typename SUMTYPE>
class Point {
 public:
  int id;
  int Dim;
  T* coords;

  __host__ __device__ Point& operator=(const Point& other)
  {
    for (int i = 0; i < Dim; i++) {
      coords[i] = other.coords[i];
    }
    id = other.id;
    return *this;
  }
};

/* L2 fallback for low dimension when ILP is not possible */
template <typename T, typename SUMTYPE>
__device__ SUMTYPE l2_SEQ(Point<T, SUMTYPE>* src_vec, Point<T, SUMTYPE>* dst_vec)
{
  SUMTYPE partial_sum = 0;

  for (int i = threadIdx.x; i < src_vec->Dim; i += blockDim.x) {
    partial_sum = fmaf((src_vec[0].coords[i] - dst_vec[0].coords[i]),
                       (src_vec[0].coords[i] - dst_vec[0].coords[i]),
                       partial_sum);
  }

  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum += __shfl_down_sync(FULL_BITMASK, partial_sum, offset);
  }
  return partial_sum;
}

/* L2 optimized with 2-way ILP for DIM >= 64 */
template <typename T, typename SUMTYPE>
__device__ SUMTYPE l2_ILP2(Point<T, SUMTYPE>* src_vec, Point<T, SUMTYPE>* dst_vec)
{
  T temp_dst[2]          = {0, 0};
  SUMTYPE partial_sum[2] = {0, 0};
  for (int i = threadIdx.x; i < src_vec->Dim; i += 2 * blockDim.x) {
    temp_dst[0] = dst_vec->coords[i];
    if (i + 32 < src_vec->Dim) temp_dst[1] = dst_vec->coords[i + 32];

    partial_sum[0] = fmaf(
      (src_vec[0].coords[i] - temp_dst[0]), (src_vec[0].coords[i] - temp_dst[0]), partial_sum[0]);
    if (i + 32 < src_vec->Dim)
      partial_sum[1] = fmaf((src_vec[0].coords[i + 32] - temp_dst[1]),
                            (src_vec[0].coords[i + 32] - temp_dst[1]),
                            partial_sum[1]);
  }
  partial_sum[0] += partial_sum[1];

  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum[0] += __shfl_down_sync(FULL_BITMASK, partial_sum[0], offset);
  }
  return partial_sum[0];
}

/* L2 optimized with 4-way ILP for optimal performance for DIM >= 128 */
template <typename T, typename SUMTYPE>
__device__ SUMTYPE l2_ILP4(Point<T, SUMTYPE>* src_vec, Point<T, SUMTYPE>* dst_vec)
{
  T temp_dst[4]          = {0, 0, 0, 0};
  SUMTYPE partial_sum[4] = {0, 0, 0, 0};
  for (int i = threadIdx.x; i < src_vec->Dim; i += 4 * blockDim.x) {
    temp_dst[0] = dst_vec->coords[i];
    if (i + 32 < src_vec->Dim) temp_dst[1] = dst_vec->coords[i + 32];
    if (i + 64 < src_vec->Dim) temp_dst[2] = dst_vec->coords[i + 64];
    if (i + 96 < src_vec->Dim) temp_dst[3] = dst_vec->coords[i + 96];

    partial_sum[0] = fmaf(
      (src_vec[0].coords[i] - temp_dst[0]), (src_vec[0].coords[i] - temp_dst[0]), partial_sum[0]);
    if (i + 32 < src_vec->Dim)
      partial_sum[1] = fmaf((src_vec[0].coords[i + 32] - temp_dst[1]),
                            (src_vec[0].coords[i + 32] - temp_dst[1]),
                            partial_sum[1]);
    if (i + 64 < src_vec->Dim)
      partial_sum[2] = fmaf((src_vec[0].coords[i + 64] - temp_dst[2]),
                            (src_vec[0].coords[i + 64] - temp_dst[2]),
                            partial_sum[2]);
    if (i + 96 < src_vec->Dim)
      partial_sum[3] = fmaf((src_vec[0].coords[i + 96] - temp_dst[3]),
                            (src_vec[0].coords[i + 96] - temp_dst[3]),
                            partial_sum[3]);
  }
  partial_sum[0] += partial_sum[1] + partial_sum[2] + partial_sum[3];

  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum[0] += __shfl_down_sync(FULL_BITMASK, partial_sum[0], offset);
  }

  return partial_sum[0];
}

/* fp16: native __hsub/__hfma throughout; single float widen at return */
__device__ __forceinline__ void l2_half_accum(__half& lane_sum, __half s, __half t)
{
  __half d = __hsub(s, t);
  lane_sum = __hfma(d, d, lane_sum);
}

/* ILP helpers: accumulate (s-t)^2 into acc; operands must already be loaded */
__device__ __forceinline__ void l2_half_fma_sq(__half& acc, __half s, __half t)
{
  __half d = __hsub(s, t);
  acc      = __hfma(d, d, acc);
}

__device__ __forceinline__ __half l2_half_shfl_down(__half val, int offset)
{
  unsigned int v = static_cast<unsigned int>(__half_as_ushort(val));
  v              = __shfl_down_sync(FULL_BITMASK, v, offset);
  return __ushort_as_half(static_cast<unsigned short>(v & 0xFFFFu));
}

__device__ __forceinline__ __half l2_half_warp_reduce_to_half(__half lane_sum)
{
  for (int offset = 16; offset > 0; offset /= 2) {
    lane_sum = __hadd(lane_sum, l2_half_shfl_down(lane_sum, offset));
  }
  return lane_sum;
}

template <typename SUMTYPE>
__device__ __forceinline__ SUMTYPE l2_half_warp_reduce(__half lane_sum)
{
  return static_cast<SUMTYPE>(__half2float(l2_half_warp_reduce_to_half(lane_sum)));
}

template <typename SUMTYPE>
__device__ SUMTYPE l2_SEQ_half(Point<__half, SUMTYPE>* src_vec, Point<__half, SUMTYPE>* dst_vec)
{
  __half lane_sum = __float2half(0.0f);

  for (int i = threadIdx.x; i < src_vec->Dim; i += blockDim.x) {
    l2_half_accum(lane_sum, src_vec[0].coords[i], dst_vec[0].coords[i]);
  }

  return l2_half_warp_reduce<SUMTYPE>(lane_sum);
}

template <typename SUMTYPE>
__device__ SUMTYPE l2_ILP2_half(Point<__half, SUMTYPE>* src_vec, Point<__half, SUMTYPE>* dst_vec)
{
  __half temp_dst[2]    = {__float2half(0.0f), __float2half(0.0f)};
  __half partial_sum[2] = {__float2half(0.0f), __float2half(0.0f)};
  for (int i = threadIdx.x; i < src_vec->Dim; i += 2 * blockDim.x) {
    temp_dst[0] = dst_vec->coords[i];
    if (i + 32 < src_vec->Dim) temp_dst[1] = dst_vec->coords[i + 32];

    l2_half_fma_sq(partial_sum[0], src_vec[0].coords[i], temp_dst[0]);
    if (i + 32 < src_vec->Dim)
      l2_half_fma_sq(partial_sum[1], src_vec[0].coords[i + 32], temp_dst[1]);
  }
  partial_sum[0] = __hadd(partial_sum[0], partial_sum[1]);

  return l2_half_warp_reduce<SUMTYPE>(partial_sum[0]);
}

template <typename SUMTYPE>
__device__ SUMTYPE l2_ILP4_half(Point<__half, SUMTYPE>* src_vec, Point<__half, SUMTYPE>* dst_vec)
{
  __half temp_dst[4] = {
    __float2half(0.0f), __float2half(0.0f), __float2half(0.0f), __float2half(0.0f)};
  __half partial_sum[4] = {
    __float2half(0.0f), __float2half(0.0f), __float2half(0.0f), __float2half(0.0f)};
  for (int i = threadIdx.x; i < src_vec->Dim; i += 4 * blockDim.x) {
    temp_dst[0] = dst_vec->coords[i];
    if (i + 32 < src_vec->Dim) temp_dst[1] = dst_vec->coords[i + 32];
    if (i + 64 < src_vec->Dim) temp_dst[2] = dst_vec->coords[i + 64];
    if (i + 96 < src_vec->Dim) temp_dst[3] = dst_vec->coords[i + 96];

    l2_half_fma_sq(partial_sum[0], src_vec[0].coords[i], temp_dst[0]);
    if (i + 32 < src_vec->Dim)
      l2_half_fma_sq(partial_sum[1], src_vec[0].coords[i + 32], temp_dst[1]);
    if (i + 64 < src_vec->Dim)
      l2_half_fma_sq(partial_sum[2], src_vec[0].coords[i + 64], temp_dst[2]);
    if (i + 96 < src_vec->Dim)
      l2_half_fma_sq(partial_sum[3], src_vec[0].coords[i + 96], temp_dst[3]);
  }
  partial_sum[0] =
    __hadd(partial_sum[0], __hadd(partial_sum[1], __hadd(partial_sum[2], partial_sum[3])));

  return l2_half_warp_reduce<SUMTYPE>(partial_sum[0]);
}

/* Selects ILP optimization level based on dimension */
template <typename T, typename SUMTYPE>
__forceinline__ __device__ SUMTYPE l2(Point<T, SUMTYPE>* src_vec, Point<T, SUMTYPE>* dst_vec)
{
  if constexpr (std::is_same_v<std::remove_cv_t<T>, __half>) {
    if (src_vec->Dim >= 128) {
      return l2_ILP4_half<SUMTYPE>(src_vec, dst_vec);
    } else if (src_vec->Dim >= 64) {
      return l2_ILP2_half<SUMTYPE>(src_vec, dst_vec);
    } else {
      return l2_SEQ_half<SUMTYPE>(src_vec, dst_vec);
    }
  } else {
    if (src_vec->Dim >= 128) {
      return l2_ILP4<T, SUMTYPE>(src_vec, dst_vec);
    } else if (src_vec->Dim >= 64) {
      return l2_ILP2<T, SUMTYPE>(src_vec, dst_vec);
    } else {
      return l2_SEQ<T, SUMTYPE>(src_vec, dst_vec);
    }
  }
}

/* Convert vectors to point structure to performance distance comparison */
template <typename T, typename SUMTYPE>
__host__ __device__ SUMTYPE l2(const T* src, const T* dest, int dim)
{
  Point<T, SUMTYPE> src_p;
  src_p.coords = const_cast<T*>(src);
  src_p.Dim    = dim;
  Point<T, SUMTYPE> dest_p;
  dest_p.coords = const_cast<T*>(dest);
  dest_p.Dim    = dim;

  return l2<T, SUMTYPE>(&src_p, &dest_p);
}

template <typename T, typename SUMTYPE>
__host__ __device__ SUMTYPE
dist(const T* src, const T* dest, int dim, cuvs::distance::DistanceType metric)
{
  SUMTYPE d = l2<T, SUMTYPE>(src, dest, dim);
  if (metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
    return static_cast<SUMTYPE>(sqrtf(static_cast<float>(d)));
  }
  return d;
}

/*
 * Warp-strided L2 / dist: each warp computes one distance using lanes 0..31 only.
 * Required when blockDim.x > 32 but only one distance per warp is desired — plain l2() shards
 * work across the whole block then reduces inside each warp only, which under-counts L2.
 */
template <typename T, typename SUMTYPE>
__device__ SUMTYPE l2_SEQ_warp(Point<T, SUMTYPE>* src_vec, Point<T, SUMTYPE>* dst_vec, int lane)
{
  SUMTYPE partial_sum = 0;
  for (int i = lane; i < src_vec->Dim; i += VAMANA_WARP_SIZE) {
    partial_sum = fmaf((src_vec[0].coords[i] - dst_vec[0].coords[i]),
                       (src_vec[0].coords[i] - dst_vec[0].coords[i]),
                       partial_sum);
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum += __shfl_down_sync(FULL_BITMASK, partial_sum, offset);
  }
  return partial_sum;
}

template <typename T, typename SUMTYPE>
__device__ SUMTYPE l2_ILP2_warp(Point<T, SUMTYPE>* src_vec, Point<T, SUMTYPE>* dst_vec, int lane)
{
  T temp_dst[2]          = {0, 0};
  SUMTYPE partial_sum[2] = {0, 0};
  for (int i = lane; i < src_vec->Dim; i += 2 * VAMANA_WARP_SIZE) {
    temp_dst[0] = dst_vec->coords[i];
    if (i + 32 < src_vec->Dim) temp_dst[1] = dst_vec->coords[i + 32];

    partial_sum[0] = fmaf(
      (src_vec[0].coords[i] - temp_dst[0]), (src_vec[0].coords[i] - temp_dst[0]), partial_sum[0]);
    if (i + 32 < src_vec->Dim)
      partial_sum[1] = fmaf((src_vec[0].coords[i + 32] - temp_dst[1]),
                            (src_vec[0].coords[i + 32] - temp_dst[1]),
                            partial_sum[1]);
  }
  partial_sum[0] += partial_sum[1];
  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum[0] += __shfl_down_sync(FULL_BITMASK, partial_sum[0], offset);
  }
  return partial_sum[0];
}

template <typename T, typename SUMTYPE>
__device__ SUMTYPE l2_ILP4_warp(Point<T, SUMTYPE>* src_vec, Point<T, SUMTYPE>* dst_vec, int lane)
{
  T temp_dst[4]          = {0, 0, 0, 0};
  SUMTYPE partial_sum[4] = {0, 0, 0, 0};
  for (int i = lane; i < src_vec->Dim; i += 4 * VAMANA_WARP_SIZE) {
    temp_dst[0] = dst_vec->coords[i];
    if (i + 32 < src_vec->Dim) temp_dst[1] = dst_vec->coords[i + 32];
    if (i + 64 < src_vec->Dim) temp_dst[2] = dst_vec->coords[i + 64];
    if (i + 96 < src_vec->Dim) temp_dst[3] = dst_vec->coords[i + 96];

    partial_sum[0] = fmaf(
      (src_vec[0].coords[i] - temp_dst[0]), (src_vec[0].coords[i] - temp_dst[0]), partial_sum[0]);
    if (i + 32 < src_vec->Dim)
      partial_sum[1] = fmaf((src_vec[0].coords[i + 32] - temp_dst[1]),
                            (src_vec[0].coords[i + 32] - temp_dst[1]),
                            partial_sum[1]);
    if (i + 64 < src_vec->Dim)
      partial_sum[2] = fmaf((src_vec[0].coords[i + 64] - temp_dst[2]),
                            (src_vec[0].coords[i + 64] - temp_dst[2]),
                            partial_sum[2]);
    if (i + 96 < src_vec->Dim)
      partial_sum[3] = fmaf((src_vec[0].coords[i + 96] - temp_dst[3]),
                            (src_vec[0].coords[i + 96] - temp_dst[3]),
                            partial_sum[3]);
  }
  partial_sum[0] += partial_sum[1] + partial_sum[2] + partial_sum[3];
  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum[0] += __shfl_down_sync(FULL_BITMASK, partial_sum[0], offset);
  }
  return partial_sum[0];
}

template <typename SUMTYPE>
__device__ SUMTYPE l2_SEQ_half_warp(Point<__half, SUMTYPE>* src_vec,
                                    Point<__half, SUMTYPE>* dst_vec,
                                    int lane)
{
  __half lane_sum = __float2half(0.0f);
  for (int i = lane; i < src_vec->Dim; i += VAMANA_WARP_SIZE) {
    l2_half_accum(lane_sum, src_vec[0].coords[i], dst_vec[0].coords[i]);
  }
  return l2_half_warp_reduce<SUMTYPE>(lane_sum);
}

template <typename SUMTYPE>
__device__ SUMTYPE l2_ILP2_half_warp(Point<__half, SUMTYPE>* src_vec,
                                     Point<__half, SUMTYPE>* dst_vec,
                                     int lane)
{
  __half temp_dst[2]    = {__float2half(0.0f), __float2half(0.0f)};
  __half partial_sum[2] = {__float2half(0.0f), __float2half(0.0f)};
  for (int i = lane; i < src_vec->Dim; i += 2 * VAMANA_WARP_SIZE) {
    temp_dst[0] = dst_vec->coords[i];
    if (i + 32 < src_vec->Dim) temp_dst[1] = dst_vec->coords[i + 32];

    l2_half_fma_sq(partial_sum[0], src_vec[0].coords[i], temp_dst[0]);
    if (i + 32 < src_vec->Dim)
      l2_half_fma_sq(partial_sum[1], src_vec[0].coords[i + 32], temp_dst[1]);
  }
  partial_sum[0] = __hadd(partial_sum[0], partial_sum[1]);
  return l2_half_warp_reduce<SUMTYPE>(partial_sum[0]);
}

template <typename SUMTYPE>
__device__ SUMTYPE l2_ILP4_half_warp(Point<__half, SUMTYPE>* src_vec,
                                     Point<__half, SUMTYPE>* dst_vec,
                                     int lane)
{
  __half temp_dst[4] = {
    __float2half(0.0f), __float2half(0.0f), __float2half(0.0f), __float2half(0.0f)};
  __half partial_sum[4] = {
    __float2half(0.0f), __float2half(0.0f), __float2half(0.0f), __float2half(0.0f)};
  for (int i = lane; i < src_vec->Dim; i += 4 * VAMANA_WARP_SIZE) {
    temp_dst[0] = dst_vec->coords[i];
    if (i + 32 < src_vec->Dim) temp_dst[1] = dst_vec->coords[i + 32];
    if (i + 64 < src_vec->Dim) temp_dst[2] = dst_vec->coords[i + 64];
    if (i + 96 < src_vec->Dim) temp_dst[3] = dst_vec->coords[i + 96];

    l2_half_fma_sq(partial_sum[0], src_vec[0].coords[i], temp_dst[0]);
    if (i + 32 < src_vec->Dim)
      l2_half_fma_sq(partial_sum[1], src_vec[0].coords[i + 32], temp_dst[1]);
    if (i + 64 < src_vec->Dim)
      l2_half_fma_sq(partial_sum[2], src_vec[0].coords[i + 64], temp_dst[2]);
    if (i + 96 < src_vec->Dim)
      l2_half_fma_sq(partial_sum[3], src_vec[0].coords[i + 96], temp_dst[3]);
  }
  partial_sum[0] =
    __hadd(partial_sum[0], __hadd(partial_sum[1], __hadd(partial_sum[2], partial_sum[3])));
  return l2_half_warp_reduce<SUMTYPE>(partial_sum[0]);
}

template <typename T, typename SUMTYPE>
__forceinline__ __device__ SUMTYPE l2_warp(Point<T, SUMTYPE>* src_vec,
                                           Point<T, SUMTYPE>* dst_vec,
                                           int lane)
{
  if constexpr (std::is_same_v<std::remove_cv_t<T>, __half>) {
    if (src_vec->Dim >= 128) {
      return l2_ILP4_half_warp<SUMTYPE>(src_vec, dst_vec, lane);
    } else if (src_vec->Dim >= 64) {
      return l2_ILP2_half_warp<SUMTYPE>(src_vec, dst_vec, lane);
    } else {
      return l2_SEQ_half_warp<SUMTYPE>(src_vec, dst_vec, lane);
    }
  } else {
    if (src_vec->Dim >= 128) {
      return l2_ILP4_warp<T, SUMTYPE>(src_vec, dst_vec, lane);
    } else if (src_vec->Dim >= 64) {
      return l2_ILP2_warp<T, SUMTYPE>(src_vec, dst_vec, lane);
    } else {
      return l2_SEQ_warp<T, SUMTYPE>(src_vec, dst_vec, lane);
    }
  }
}

template <typename T, typename SUMTYPE>
__forceinline__ __device__ SUMTYPE l2_warp(const T* src, const T* dest, int dim, int lane)
{
  Point<T, SUMTYPE> src_p;
  src_p.coords = const_cast<T*>(src);
  src_p.Dim    = dim;
  Point<T, SUMTYPE> dest_p;
  dest_p.coords = const_cast<T*>(dest);
  dest_p.Dim    = dim;

  return l2_warp<T, SUMTYPE>(&src_p, &dest_p, lane);
}

/* float query vs half dataset: vectorized half2/float2 loads (even lane*2 indices) */
__device__ __forceinline__ float2 l2_load_src2(const float* src, int i)
{
  return *reinterpret_cast<const float2*>(&src[i]);
}

__device__ __forceinline__ float2 l2_load_dst2_half(const __half* dst, int i)
{
  return __half22float2(*reinterpret_cast<const half2*>(&dst[i]));
}

template <typename SUMTYPE>
__device__ __forceinline__ void l2_fma_sq2(SUMTYPE& acc, float sx, float sy, float2 dst2)
{
  float dx = sx - dst2.x;
  float dy = sy - dst2.y;
  acc      = fmaf(dx, dx, acc);
  acc      = fmaf(dy, dy, acc);
}

// Widen any supported operand type to float for the scalar fallback path.
template <typename X>
__device__ __forceinline__ float l2_widen_to_float(X x)
{
  return static_cast<float>(x);
}
__device__ __forceinline__ float l2_widen_to_float(__half x) { return __half2float(x); }

// Scalar, alignment-agnostic warp L2 used as a fallback when `dim` is ODD. The
// vectorized float2/half2 comparators below assume each dataset row begins on a
// vector-aligned boundary, which only holds when dim is even (row stride =
// dim * sizeof(elem) is then a multiple of the vector width). For odd dim, odd
// rows are under-aligned and a float2/half2 load raises cudaErrorMisalignedAddress,
// so we widen both operands to float and accumulate one element per lane instead.
template <typename SUMTYPE, typename SrcT, typename DstT>
__device__ __forceinline__ SUMTYPE
l2_warp_scalar_widen(const SrcT* src, const DstT* dst, int dim, int lane)
{
  SUMTYPE partial_sum = 0;
  for (int i = lane; i < dim; i += VAMANA_WARP_SIZE) {
    float diff  = l2_widen_to_float(src[i]) - l2_widen_to_float(dst[i]);
    partial_sum = fmaf(diff, diff, partial_sum);
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum += __shfl_down_sync(FULL_BITMASK, partial_sum, offset);
  }
  return partial_sum;
}

template <typename SUMTYPE>
__device__ SUMTYPE l2_SEQ_warp_float_half(const float* src, const __half* dst, int dim, int lane)
{
  SUMTYPE partial_sum = 0;
  for (int i = lane * 2; i < dim; i += VAMANA_WARP_SIZE * 2) {
    float2 dst2 = l2_load_dst2_half(dst, i);
    float2 src2 = l2_load_src2(src, i);
    l2_fma_sq2(partial_sum, src2.x, src2.y, dst2);
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum += __shfl_down_sync(FULL_BITMASK, partial_sum, offset);
  }
  return partial_sum;
}

template <typename SUMTYPE>
__device__ SUMTYPE l2_ILP2_warp_float_half(const float* src, const __half* dst, int dim, int lane)
{
  SUMTYPE partial_sum[2] = {0, 0};
  for (int i = lane * 2; i < dim; i += 2 * VAMANA_WARP_SIZE * 2) {
    float2 temp_dst[2] = {{0, 0}, {0, 0}};
    temp_dst[0]        = l2_load_dst2_half(dst, i);
    if (i + 64 < dim) temp_dst[1] = l2_load_dst2_half(dst, i + 64);

    float2 src0 = l2_load_src2(src, i);
    l2_fma_sq2(partial_sum[0], src0.x, src0.y, temp_dst[0]);
    if (i + 64 < dim) {
      float2 src1 = l2_load_src2(src, i + 64);
      l2_fma_sq2(partial_sum[1], src1.x, src1.y, temp_dst[1]);
    }
  }
  partial_sum[0] += partial_sum[1];
  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum[0] += __shfl_down_sync(FULL_BITMASK, partial_sum[0], offset);
  }
  return partial_sum[0];
}

template <typename SUMTYPE>
__device__ SUMTYPE l2_ILP4_warp_float_half(const float* src, const __half* dst, int dim, int lane)
{
  SUMTYPE partial_sum[4] = {0, 0, 0, 0};
  for (int i = lane * 2; i < dim; i += 4 * VAMANA_WARP_SIZE * 2) {
    float2 temp_dst[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    temp_dst[0]        = l2_load_dst2_half(dst, i);
    if (i + 64 < dim) temp_dst[1] = l2_load_dst2_half(dst, i + 64);
    if (i + 128 < dim) temp_dst[2] = l2_load_dst2_half(dst, i + 128);
    if (i + 192 < dim) temp_dst[3] = l2_load_dst2_half(dst, i + 192);

    float2 src0 = l2_load_src2(src, i);
    l2_fma_sq2(partial_sum[0], src0.x, src0.y, temp_dst[0]);
    if (i + 64 < dim) {
      float2 src1 = l2_load_src2(src, i + 64);
      l2_fma_sq2(partial_sum[1], src1.x, src1.y, temp_dst[1]);
    }
    if (i + 128 < dim) {
      float2 src2 = l2_load_src2(src, i + 128);
      l2_fma_sq2(partial_sum[2], src2.x, src2.y, temp_dst[2]);
    }
    if (i + 192 < dim) {
      float2 src3 = l2_load_src2(src, i + 192);
      l2_fma_sq2(partial_sum[3], src3.x, src3.y, temp_dst[3]);
    }
  }
  partial_sum[0] += partial_sum[1] + partial_sum[2] + partial_sum[3];
  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum[0] += __shfl_down_sync(FULL_BITMASK, partial_sum[0], offset);
  }
  return partial_sum[0];
}

template <typename SUMTYPE>
__forceinline__ __device__ SUMTYPE
l2_warp_float_half(const float* src, const __half* dest, int dim, int lane)
{
  if (dim & 1) { return l2_warp_scalar_widen<SUMTYPE>(src, dest, dim, lane); }
  if (dim >= 128) {
    return l2_ILP4_warp_float_half<SUMTYPE>(src, dest, dim, lane);
  } else if (dim >= 64) {
    return l2_ILP2_warp_float_half<SUMTYPE>(src, dest, dim, lane);
  } else {
    return l2_SEQ_warp_float_half<SUMTYPE>(src, dest, dim, lane);
  }
}

/* half query vs float dataset: widen query coords to float at use (mirror of float_half) */
__device__ __forceinline__ float2 l2_load_src2_half(const __half* src, int i)
{
  return __half22float2(*reinterpret_cast<const half2*>(&src[i]));
}

template <typename SUMTYPE>
__device__ SUMTYPE l2_SEQ_warp_half_float(const __half* src, const float* dst, int dim, int lane)
{
  SUMTYPE partial_sum = 0;
  for (int i = lane * 2; i < dim; i += VAMANA_WARP_SIZE * 2) {
    float2 src2 = l2_load_src2_half(src, i);
    float2 dst2 = l2_load_src2(dst, i);
    l2_fma_sq2(partial_sum, src2.x, src2.y, dst2);
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum += __shfl_down_sync(FULL_BITMASK, partial_sum, offset);
  }
  return partial_sum;
}

template <typename SUMTYPE>
__device__ SUMTYPE l2_ILP2_warp_half_float(const __half* src, const float* dst, int dim, int lane)
{
  SUMTYPE partial_sum[2] = {0, 0};
  for (int i = lane * 2; i < dim; i += 2 * VAMANA_WARP_SIZE * 2) {
    float2 temp_dst[2] = {{0, 0}, {0, 0}};
    temp_dst[0]        = l2_load_src2(dst, i);
    if (i + 64 < dim) temp_dst[1] = l2_load_src2(dst, i + 64);

    float2 src0 = l2_load_src2_half(src, i);
    l2_fma_sq2(partial_sum[0], src0.x, src0.y, temp_dst[0]);
    if (i + 64 < dim) {
      float2 src1 = l2_load_src2_half(src, i + 64);
      l2_fma_sq2(partial_sum[1], src1.x, src1.y, temp_dst[1]);
    }
  }
  partial_sum[0] += partial_sum[1];
  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum[0] += __shfl_down_sync(FULL_BITMASK, partial_sum[0], offset);
  }
  return partial_sum[0];
}

template <typename SUMTYPE>
__device__ SUMTYPE l2_ILP4_warp_half_float(const __half* src, const float* dst, int dim, int lane)
{
  SUMTYPE partial_sum[4] = {0, 0, 0, 0};
  for (int i = lane * 2; i < dim; i += 4 * VAMANA_WARP_SIZE * 2) {
    float2 temp_dst[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    temp_dst[0]        = l2_load_src2(dst, i);
    if (i + 64 < dim) temp_dst[1] = l2_load_src2(dst, i + 64);
    if (i + 128 < dim) temp_dst[2] = l2_load_src2(dst, i + 128);
    if (i + 192 < dim) temp_dst[3] = l2_load_src2(dst, i + 192);

    float2 src0 = l2_load_src2_half(src, i);
    l2_fma_sq2(partial_sum[0], src0.x, src0.y, temp_dst[0]);
    if (i + 64 < dim) {
      float2 src1 = l2_load_src2_half(src, i + 64);
      l2_fma_sq2(partial_sum[1], src1.x, src1.y, temp_dst[1]);
    }
    if (i + 128 < dim) {
      float2 src2 = l2_load_src2_half(src, i + 128);
      l2_fma_sq2(partial_sum[2], src2.x, src2.y, temp_dst[2]);
    }
    if (i + 192 < dim) {
      float2 src3 = l2_load_src2_half(src, i + 192);
      l2_fma_sq2(partial_sum[3], src3.x, src3.y, temp_dst[3]);
    }
  }
  partial_sum[0] += partial_sum[1] + partial_sum[2] + partial_sum[3];
  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum[0] += __shfl_down_sync(FULL_BITMASK, partial_sum[0], offset);
  }
  return partial_sum[0];
}

template <typename SUMTYPE>
__forceinline__ __device__ SUMTYPE
l2_warp_half_float(const __half* src, const float* dest, int dim, int lane)
{
  if (dim & 1) { return l2_warp_scalar_widen<SUMTYPE>(src, dest, dim, lane); }
  if (dim >= 128) {
    return l2_ILP4_warp_half_float<SUMTYPE>(src, dest, dim, lane);
  } else if (dim >= 64) {
    return l2_ILP2_warp_half_float<SUMTYPE>(src, dest, dim, lane);
  } else {
    return l2_SEQ_warp_half_float<SUMTYPE>(src, dest, dim, lane);
  }
}

/* fp16 query smem vs half dataset: vectorized half2 loads, widen to float, float accumulate
 * (mirror of l2_warp_float_half; avoids scalar smem loads and native half FMA) */
template <typename SUMTYPE>
__device__ SUMTYPE
l2_SEQ_warp_half_smem_half(const __half* src, const __half* dst, int dim, int lane)
{
  SUMTYPE partial_sum = 0;
  for (int i = lane * 2; i < dim; i += VAMANA_WARP_SIZE * 2) {
    float2 dst2 = l2_load_dst2_half(dst, i);
    float2 src2 = l2_load_src2_half(src, i);
    l2_fma_sq2(partial_sum, src2.x, src2.y, dst2);
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum += __shfl_down_sync(FULL_BITMASK, partial_sum, offset);
  }
  return partial_sum;
}

template <typename SUMTYPE>
__device__ SUMTYPE
l2_ILP2_warp_half_smem_half(const __half* src, const __half* dst, int dim, int lane)
{
  SUMTYPE partial_sum[2] = {0, 0};
  for (int i = lane * 2; i < dim; i += 2 * VAMANA_WARP_SIZE * 2) {
    float2 temp_dst[2] = {{0, 0}, {0, 0}};
    temp_dst[0]        = l2_load_dst2_half(dst, i);
    if (i + 64 < dim) temp_dst[1] = l2_load_dst2_half(dst, i + 64);

    float2 src0 = l2_load_src2_half(src, i);
    l2_fma_sq2(partial_sum[0], src0.x, src0.y, temp_dst[0]);
    if (i + 64 < dim) {
      float2 src1 = l2_load_src2_half(src, i + 64);
      l2_fma_sq2(partial_sum[1], src1.x, src1.y, temp_dst[1]);
    }
  }
  partial_sum[0] += partial_sum[1];
  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum[0] += __shfl_down_sync(FULL_BITMASK, partial_sum[0], offset);
  }
  return partial_sum[0];
}

template <typename SUMTYPE>
__device__ SUMTYPE
l2_ILP4_warp_half_smem_half(const __half* src, const __half* dst, int dim, int lane)
{
  SUMTYPE partial_sum[4] = {0, 0, 0, 0};
  for (int i = lane * 2; i < dim; i += 4 * VAMANA_WARP_SIZE * 2) {
    float2 temp_dst[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    temp_dst[0]        = l2_load_dst2_half(dst, i);
    if (i + 64 < dim) temp_dst[1] = l2_load_dst2_half(dst, i + 64);
    if (i + 128 < dim) temp_dst[2] = l2_load_dst2_half(dst, i + 128);
    if (i + 192 < dim) temp_dst[3] = l2_load_dst2_half(dst, i + 192);

    float2 src0 = l2_load_src2_half(src, i);
    l2_fma_sq2(partial_sum[0], src0.x, src0.y, temp_dst[0]);
    if (i + 64 < dim) {
      float2 src1 = l2_load_src2_half(src, i + 64);
      l2_fma_sq2(partial_sum[1], src1.x, src1.y, temp_dst[1]);
    }
    if (i + 128 < dim) {
      float2 src2 = l2_load_src2_half(src, i + 128);
      l2_fma_sq2(partial_sum[2], src2.x, src2.y, temp_dst[2]);
    }
    if (i + 192 < dim) {
      float2 src3 = l2_load_src2_half(src, i + 192);
      l2_fma_sq2(partial_sum[3], src3.x, src3.y, temp_dst[3]);
    }
  }
  partial_sum[0] += partial_sum[1] + partial_sum[2] + partial_sum[3];
  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum[0] += __shfl_down_sync(FULL_BITMASK, partial_sum[0], offset);
  }
  return partial_sum[0];
}

template <typename SUMTYPE>
__forceinline__ __device__ SUMTYPE
l2_warp_half_smem_half(const __half* src, const __half* dest, int dim, int lane)
{
  if (dim & 1) { return l2_warp_scalar_widen<SUMTYPE>(src, dest, dim, lane); }
  if (dim >= 128) {
    return l2_ILP4_warp_half_smem_half<SUMTYPE>(src, dest, dim, lane);
  } else if (dim >= 64) {
    return l2_ILP2_warp_half_smem_half<SUMTYPE>(src, dest, dim, lane);
  } else {
    return l2_SEQ_warp_half_smem_half<SUMTYPE>(src, dest, dim, lane);
  }
}

/* fp16 query smem vs int8 (or other native) dataset: same vectorized query widen, float accumulate
 */
template <typename SUMTYPE, typename DataT>
__device__ __forceinline__ void l2_fma_sq2_half_native(SUMTYPE& acc,
                                                       float2 src2,
                                                       const DataT* dst,
                                                       int i)
{
  float dx = src2.x - static_cast<float>(dst[i]);
  float dy = src2.y - static_cast<float>(dst[i + 1]);
  acc      = fmaf(dx, dx, acc);
  acc      = fmaf(dy, dy, acc);
}

template <typename SUMTYPE, typename DataT>
__device__ SUMTYPE
l2_SEQ_warp_half_smem_native(const __half* src, const DataT* dst, int dim, int lane)
{
  SUMTYPE partial_sum = 0;
  for (int i = lane * 2; i < dim; i += VAMANA_WARP_SIZE * 2) {
    float2 src2 = l2_load_src2_half(src, i);
    l2_fma_sq2_half_native(partial_sum, src2, dst, i);
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum += __shfl_down_sync(FULL_BITMASK, partial_sum, offset);
  }
  return partial_sum;
}

template <typename SUMTYPE, typename DataT>
__device__ SUMTYPE
l2_ILP2_warp_half_smem_native(const __half* src, const DataT* dst, int dim, int lane)
{
  SUMTYPE partial_sum[2] = {0, 0};
  for (int i = lane * 2; i < dim; i += 2 * VAMANA_WARP_SIZE * 2) {
    float2 src0 = l2_load_src2_half(src, i);
    l2_fma_sq2_half_native(partial_sum[0], src0, dst, i);
    if (i + 64 < dim) {
      float2 src1 = l2_load_src2_half(src, i + 64);
      l2_fma_sq2_half_native(partial_sum[1], src1, dst, i + 64);
    }
  }
  partial_sum[0] += partial_sum[1];
  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum[0] += __shfl_down_sync(FULL_BITMASK, partial_sum[0], offset);
  }
  return partial_sum[0];
}

template <typename SUMTYPE, typename DataT>
__device__ SUMTYPE
l2_ILP4_warp_half_smem_native(const __half* src, const DataT* dst, int dim, int lane)
{
  SUMTYPE partial_sum[4] = {0, 0, 0, 0};
  for (int i = lane * 2; i < dim; i += 4 * VAMANA_WARP_SIZE * 2) {
    float2 src0 = l2_load_src2_half(src, i);
    l2_fma_sq2_half_native(partial_sum[0], src0, dst, i);
    if (i + 64 < dim) {
      float2 src1 = l2_load_src2_half(src, i + 64);
      l2_fma_sq2_half_native(partial_sum[1], src1, dst, i + 64);
    }
    if (i + 128 < dim) {
      float2 src2 = l2_load_src2_half(src, i + 128);
      l2_fma_sq2_half_native(partial_sum[2], src2, dst, i + 128);
    }
    if (i + 192 < dim) {
      float2 src3 = l2_load_src2_half(src, i + 192);
      l2_fma_sq2_half_native(partial_sum[3], src3, dst, i + 192);
    }
  }
  partial_sum[0] += partial_sum[1] + partial_sum[2] + partial_sum[3];
  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum[0] += __shfl_down_sync(FULL_BITMASK, partial_sum[0], offset);
  }
  return partial_sum[0];
}

template <typename SUMTYPE, typename DataT>
__forceinline__ __device__ SUMTYPE
l2_warp_half_smem_native(const __half* src, const DataT* dest, int dim, int lane)
{
  if (dim & 1) { return l2_warp_scalar_widen<SUMTYPE>(src, dest, dim, lane); }
  if (dim >= 128) {
    return l2_ILP4_warp_half_smem_native<SUMTYPE, DataT>(src, dest, dim, lane);
  } else if (dim >= 64) {
    return l2_ILP2_warp_half_smem_native<SUMTYPE, DataT>(src, dest, dim, lane);
  } else {
    return l2_SEQ_warp_half_smem_native<SUMTYPE, DataT>(src, dest, dim, lane);
  }
}

template <typename SUMTYPE, typename DataT>
__forceinline__ __device__ SUMTYPE dist_warp_half_query(
  const __half* src, const DataT* dest, int dim, cuvs::distance::DistanceType metric, int lane)
{
  SUMTYPE d;
  if constexpr (is_cuda_fp16_v<DataT>) {
    d = l2_warp_half_smem_half<SUMTYPE>(src, reinterpret_cast<const __half*>(dest), dim, lane);
  } else if constexpr (std::is_same_v<DataT, float>) {
    d = l2_warp_half_float<SUMTYPE>(src, dest, dim, lane);
  } else {
    d = l2_warp_half_smem_native<SUMTYPE, DataT>(src, dest, dim, lane);
  }
  if (metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
    return static_cast<SUMTYPE>(sqrtf(static_cast<float>(d)));
  }
  return d;
}

template <typename SUMTYPE>
__forceinline__ __device__ SUMTYPE dist_warp(
  const float* src, const half* dest, int dim, cuvs::distance::DistanceType metric, int lane)
{
  SUMTYPE d = l2_warp_float_half<SUMTYPE>(src, reinterpret_cast<const __half*>(dest), dim, lane);
  if (metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
    return static_cast<SUMTYPE>(sqrtf(static_cast<float>(d)));
  }
  return d;
}

template <typename T, typename SUMTYPE>
__forceinline__ __device__ SUMTYPE
dist_warp(const T* src, const T* dest, int dim, cuvs::distance::DistanceType metric, int lane)
{
  SUMTYPE d = l2_warp<T, SUMTYPE>(src, dest, dim, lane);
  if (metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
    return static_cast<SUMTYPE>(sqrtf(static_cast<float>(d)));
  }
  return d;
}

// Warp-cooperative id lookup in a GreedySearch visited list (sorted by distance, not id).
// All lanes execute the same number of ballot rounds to avoid warp deadlock.
template <typename IdxT, typename accT>
__forceinline__ __device__ bool lookup_visited_dist_warp(
  const IdxT* ids, const accT* dists, int size, IdxT target, accT& out_dist, int laneId)
{
  bool found          = false;
  accT found_dist     = static_cast<accT>(0);
  const int num_iters = (size + 31) >> 5;
  for (int k = 0; k < num_iters; ++k) {
    const int j         = (k << 5) + laneId;
    const bool hit      = (j < size) && (ids[j] == target);
    accT my_dist        = hit ? dists[j] : static_cast<accT>(0);
    const unsigned hits = raft::ballot(hit);
    if (hits != 0 && !found) {
      const int src_lane = __ffs(hits) - 1;
      found_dist         = raft::shfl(my_dist, src_lane);
      found              = true;
    }
  }
  if (found) { out_dist = found_dist; }
  return found;
}

/***************************************************************************************
 * Structure that holds information about and results of a query. Use by both
 * GreedySearch and RobustPrune, as well as reverse edge lists.
 ***************************************************************************************/
template <typename IdxT, typename accT>
struct QueryCandidates {
  IdxT* ids;
  accT* dists;
  int queryId;
  int size;
  int maxSize;

  __device__ void reset()
  {
    for (int i = threadIdx.x; i < maxSize; i += blockDim.x) {
      ids[i]   = raft::upper_bound<IdxT>();
      dists[i] = raft::upper_bound<accT>();
    }
    size = 0;
  }

  // Warp-level reset: uses laneId and stride 32, no block sync
  __device__ void reset_warp(int laneId)
  {
    for (int i = laneId; i < maxSize; i += 32) {
      ids[i]   = raft::upper_bound<IdxT>();
      dists[i] = raft::upper_bound<accT>();
    }
    if (laneId == 0) { size = 0; }
  }

  // Checks current list to see if a node as previously been visited
  __inline__ __device__ bool check_visited(IdxT target, accT dist)
  {
    __syncthreads();
    __shared__ bool found;
    found = false;
    __syncthreads();

    if (size < maxSize) {
      __syncthreads();
      for (int i = threadIdx.x; i < size; i += blockDim.x) {
        if (ids[i] == target) { found = true; }
      }
      __syncthreads();
      if (!found && threadIdx.x == 0) {
        ids[size]   = target;
        dists[size] = dist;
        size++;
      }
      __syncthreads();
    }
    return found;
  }

  // Warp-level check_visited: no __syncthreads, uses laneId and warp ballot
  __inline__ __device__ bool check_visited_warp(IdxT target, accT dist_val, int laneId)
  {
    bool my_found = false;
    for (int i = laneId; i < size; i += 32) {
      if (ids[i] == target) {
        my_found = true;
        break;
      }
    }
    unsigned mask = raft::ballot(my_found);
    bool found    = (mask != 0);
    if (!found && size < maxSize && laneId == 0) {
      ids[size]   = target;
      dists[size] = dist_val;
      size++;
    }
    return found;
  }
  // For debugging
  /*
  __inline__ __device__ void print_visited() {
    printf("queryId:%d, size:%d\n", queryId, size);
    for(int i=0; i<size; i++) {
      printf("%d (%f), ", ids[i], dists[i]);
    }
    printf("\n");
  }
  */
};

namespace {

/********************************************************************************************
 * Kernels that work on QueryCandidates objects *
 *******************************************************************************************/
// For debugging
template <typename accT, typename IdxT = uint32_t>
__global__ void print_query_results(void* query_list_ptr, int count)
{
  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);

  for (int i = 0; i < count; i++) {
    query_list[i].print_visited();
  }
}

// Initialize a list of QueryCandidates objects: assign memory to mpointers and initialize values
template <typename IdxT, typename accT>
__global__ void init_query_candidate_list(QueryCandidates<IdxT, accT>* query_list,
                                          IdxT* visited_id_ptr,
                                          accT* visited_dist_ptr,
                                          int num_queries,
                                          int maxSize,
                                          int extra_queries_in_list = 0)
{
  IdxT* ids_ptr  = static_cast<IdxT*>(visited_id_ptr);
  accT* dist_ptr = static_cast<accT*>(visited_dist_ptr);

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_queries * maxSize;
       i += blockDim.x * gridDim.x) {
    ids_ptr[i]  = raft::upper_bound<IdxT>();
    dist_ptr[i] = raft::upper_bound<accT>();
  }

  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_queries + extra_queries_in_list;
       i += blockDim.x * gridDim.x) {
    query_list[i].maxSize = maxSize;
    query_list[i].size    = 0;
    query_list[i].ids     = &ids_ptr[i * (size_t)(maxSize)];
    query_list[i].dists   = &dist_ptr[i * (size_t)(maxSize)];
  }
}

// Copy query ID values from input array
template <typename IdxT, typename accT>
__global__ void set_query_ids(void* query_list_ptr, IdxT* d_query_ids, int step_size)
{
  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < step_size; i += blockDim.x * gridDim.x) {
    query_list[i].queryId = d_query_ids[i];
    query_list[i].size    = 0;
  }
}

// Device fcn to have a threadblock copy coordinates into shared memory
template <typename T, typename accT>
__device__ void update_shared_point(
  Point<T, accT>* shared_point, const T* data_ptr, int id, int dim, int idx)
{
  shared_point->id  = id;
  shared_point->Dim = dim;
  for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
    shared_point->coords[i] = data_ptr[(size_t)(id) * (size_t)(dim) + i];
  }
}

// Device fcn to have a threadblock copy coordinates into shared memory
template <typename T, typename accT>
__device__ void update_shared_point(Point<T, accT>* shared_point,
                                    const T* data_ptr,
                                    int id,
                                    int dim)
{
  shared_point->id  = id;
  shared_point->Dim = dim;
  for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
    shared_point->coords[i] = data_ptr[(size_t)(id) * (size_t)(dim) + i];
  }
}

// Warp-level: uses laneId and stride 32 for coordinate copy
template <typename T, typename accT>
__device__ void update_shared_point_warp(
  Point<T, accT>* shared_point, const T* data_ptr, int id, int dim, int laneId)
{
  shared_point->id  = id;
  shared_point->Dim = dim;
  for (size_t i = laneId; i < dim; i += 32) {
    shared_point->coords[i] = data_ptr[(size_t)(id) * (size_t)(dim) + i];
  }
}

// Promote half dataset vector to float in shared memory (once per query)
template <typename accT>
__device__ void update_shared_point_half_to_float(Point<float, accT>* shared_point,
                                                  const half* data_ptr,
                                                  int id,
                                                  int dim)
{
  const __half* half_ptr = reinterpret_cast<const __half*>(data_ptr);
  shared_point->id       = id;
  shared_point->Dim      = dim;
  const size_t base      = (size_t)id * (size_t)dim;
  // Odd dim => odd rows start at an under-aligned address; half2 loads would
  // fault (cudaErrorMisalignedAddress). Use scalar promotion in that case.
  if ((dim & 1) != 0) {
    for (size_t i = threadIdx.x; i < (size_t)dim; i += blockDim.x) {
      shared_point->coords[i] = __half2float(half_ptr[base + i]);
    }
    return;
  }
  for (size_t i = threadIdx.x * 2; i + 1 < (size_t)dim; i += (size_t)blockDim.x * 2) {
    float2 promoted    = __half22float2(*reinterpret_cast<const half2*>(&half_ptr[base + i]));
    float2* coord_pair = reinterpret_cast<float2*>(&shared_point->coords[i]);
    *coord_pair        = promoted;
  }
}

template <typename accT>
__device__ void update_shared_point_warp_half_to_float(
  Point<float, accT>* shared_point, const half* data_ptr, int id, int dim, int laneId)
{
  const __half* half_ptr = reinterpret_cast<const __half*>(data_ptr);
  shared_point->id       = id;
  shared_point->Dim      = dim;
  const size_t base      = (size_t)id * (size_t)dim;
  // Odd dim => odd rows are under-aligned for half2 loads; promote scalar.
  if ((dim & 1) != 0) {
    for (size_t i = laneId; i < (size_t)dim; i += 32) {
      shared_point->coords[i] = __half2float(half_ptr[base + i]);
    }
    return;
  }
  for (size_t i = laneId * 2; i + 1 < (size_t)dim; i += 64) {
    float2 promoted    = __half22float2(*reinterpret_cast<const half2*>(&half_ptr[base + i]));
    float2* coord_pair = reinterpret_cast<float2*>(&shared_point->coords[i]);
    *coord_pair        = promoted;
  }
}

template <typename accT>
__device__ void update_shared_point_warp_fp16_query_smem(
  Point<__half, accT>* shared_point, const half* data_ptr, int id, int dim, int laneId)
{
  const __half* half_ptr = reinterpret_cast<const __half*>(data_ptr);
  shared_point->id       = id;
  shared_point->Dim      = dim;
  const size_t base      = (size_t)id * (size_t)dim;
  // Odd dim => odd rows are under-aligned for half2 loads; copy scalar.
  if ((dim & 1) != 0) {
    for (size_t i = laneId; i < (size_t)dim; i += 32) {
      shared_point->coords[i] = half_ptr[base + i];
    }
    return;
  }
  for (size_t i = laneId * 2; i + 1 < (size_t)dim; i += 64) {
    *reinterpret_cast<half2*>(&shared_point->coords[i]) =
      *reinterpret_cast<const half2*>(&half_ptr[base + i]);
  }
}

template <typename accT>
__device__ void update_shared_point_warp_fp16_query_smem(
  Point<__half, accT>* shared_point, const float* data_ptr, int id, int dim, int laneId)
{
  shared_point->id  = id;
  shared_point->Dim = dim;
  const size_t base = (size_t)id * (size_t)dim;
  // Odd dim => odd rows are under-aligned for float2 loads; convert scalar.
  if ((dim & 1) != 0) {
    for (size_t i = laneId; i < (size_t)dim; i += 32) {
      shared_point->coords[i] = __float2half(data_ptr[base + i]);
    }
    return;
  }
  for (size_t i = laneId * 2; i + 1 < (size_t)dim; i += 64) {
    float2 v = *reinterpret_cast<const float2*>(&data_ptr[base + i]);
    *reinterpret_cast<half2*>(&shared_point->coords[i]) = __float22half2_rn(v);
  }
}

template <typename T, typename accT>
__device__ std::enable_if_t<!is_cuda_fp16_v<T> && !std::is_same_v<T, float>, void>
update_shared_point_warp_fp16_query_smem(
  Point<__half, accT>* shared_point, const T* data_ptr, int id, int dim, int laneId)
{
  shared_point->id  = id;
  shared_point->Dim = dim;
  const size_t base = (size_t)id * (size_t)dim;
  for (size_t i = laneId; i < (size_t)dim; i += 32) {
    shared_point->coords[i] = __float2half(static_cast<float>(data_ptr[base + i]));
  }
}

// Update the graph from the results of the query list (or reverse edge list)
template <typename accT, typename IdxT = uint32_t>
__global__ void write_graph_edges_kernel(raft::device_matrix_view<IdxT, int64_t> graph,
                                         void* query_list_ptr,
                                         int degree,
                                         int num_queries)
{
  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);

  for (int i = blockIdx.x; i < num_queries; i += gridDim.x) {
    for (int j = threadIdx.x; j < query_list[i].size; j += blockDim.x) {
      graph(query_list[i].queryId, j) = query_list[i].ids[j];
    }
  }
}

// Create src and dest edge lists used to sort and create reverse edges
template <typename accT, typename IdxT = uint32_t>
__global__ void create_reverse_edge_list(void* query_list_ptr,
                                         int num_queries,
                                         int degree,
                                         IdxT* edge_src,
                                         DistPair<IdxT, accT>* edge_dest)
{
  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_queries;
       i += blockDim.x * gridDim.x) {
    int read_idx   = i * query_list[i].maxSize;
    int cand_count = query_list[i + 1].size - query_list[i].size;

    for (int j = 0; j < cand_count; j++) {
      edge_src[query_list[i].size + j]       = query_list[i].queryId;
      edge_dest[query_list[i].size + j].idx  = query_list[i].ids[j];
      edge_dest[query_list[i].size + j].dist = query_list[i].dists[j];
    }
  }
}

// Populate reverse edge QueryCandidates structure based on sorted edge list and unique indices
// values
template <typename T, typename accT, typename IdxT = uint32_t>
__global__ void populate_reverse_list_struct(QueryCandidates<IdxT, accT>* reverse_list,
                                             IdxT* edge_src,
                                             IdxT* edge_dest,
                                             int* unique_indices,
                                             int unique_dests,
                                             int total_edges,
                                             int N,
                                             int rev_start,
                                             int reverse_batch)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < reverse_batch;
       i += blockDim.x * gridDim.x) {
    reverse_list[i].queryId = edge_dest[unique_indices[i + rev_start]];
    if (rev_start + i == unique_dests - 1) {
      reverse_list[i].size = total_edges - unique_indices[i + rev_start];
    } else {
      reverse_list[i].size = unique_indices[i + rev_start + 1] - unique_indices[i + rev_start];
    }
    if (reverse_list[i].size > reverse_list[i].maxSize) {
      reverse_list[i].size = reverse_list[i].maxSize;
    }

    for (int j = 0; j < reverse_list[i].size; j++) {
      reverse_list[i].ids[j] = edge_src[unique_indices[i + rev_start] + j];
    }
    for (int j = reverse_list[i].size; j < reverse_list[i].maxSize; j++) {
      reverse_list[i].ids[j]   = raft::upper_bound<IdxT>();
      reverse_list[i].dists[j] = raft::upper_bound<accT>();
    }
  }
}

// Recompute distances of reverse list. Allows us to avoid keeping distances during sort
template <typename T,
          typename accT,
          typename IdxT = uint32_t,
          typename Accessor =
            raft::host_device_accessor<cuda::std::default_accessor<T>, raft::memory_type::host>>
__global__ void recompute_reverse_dists(
  QueryCandidates<IdxT, accT>* reverse_list,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  int unique_dests,
  cuvs::distance::DistanceType metric)
{
  int dim          = dataset.extent(1);
  const T* vec_ptr = dataset.data_handle();

  for (int i = blockIdx.x; i < unique_dests; i += gridDim.x) {
    for (int j = 0; j < reverse_list[i].size; j++) {
      reverse_list[i].dists[j] =
        dist<T, accT>(&vec_ptr[(size_t)(reverse_list[i].queryId) * (size_t)dim],
                      &vec_ptr[(size_t)(reverse_list[i].ids[j]) * (size_t)dim],
                      dim,
                      metric);
    }
  }
}

}  // namespace

/**
 * @}
 */

}  // namespace cuvs::neighbors::vamana::detail
