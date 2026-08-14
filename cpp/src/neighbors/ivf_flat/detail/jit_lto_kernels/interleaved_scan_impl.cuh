/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../../ivf_common.cuh"
#include "device_functions.cuh"

#include <cuvs/neighbors/ivf_flat.hpp>

#include <raft/matrix/detail/select_warpsort.cuh>
#include <raft/util/device_loads_stores.cuh>
#include <raft/util/integer_utils.hpp>
#include <raft/util/vectorized.cuh>

// This header contains the kernel definition and should only be included
// when compiling JIT-LTO kernel fragments (when BUILD_KERNEL is defined).

namespace cuvs::neighbors::ivf_flat::detail {

static constexpr int kThreadsPerBlock = 128;

// switch to dummy blocksort when Capacity is 0 this explicit dummy is chosen
// to support access to warpsort constants like ::queue_t::kDummy
template <int Capacity, bool Ascending, typename T, typename IdxT>
struct flat_block_sort {
  using type = raft::matrix::detail::select::warpsort::block_sort<
    raft::matrix::detail::select::warpsort::warp_sort_filtered,
    Capacity,
    Ascending,
    T,
    IdxT>;
};

template <typename T, bool Ascending, typename IdxT>
struct flat_block_sort<0, Ascending, T, IdxT>
  : ivf::detail::dummy_block_sort_t<T, IdxT, Ascending> {
  using type = ivf::detail::dummy_block_sort_t<T, IdxT, Ascending>;
};

template <int Capacity, bool Ascending, typename T, typename IdxT>
using block_sort_t = typename flat_block_sort<Capacity, Ascending, T, IdxT>::type;

/**
 * Scan clusters for nearest neighbors of the query vectors.
 * See `ivfflat_interleaved_scan` for more information.
 *
 * The clusters are stored in the interleaved index format described in ivf_flat_types.hpp.
 * For each query vector, a set of clusters is probed: the distance to each vector in the cluster is
 * calculated, and the top-k nearest neighbors are selected.
 *
 * @param compute_dist distance function
 * @param query_smem_elems number of dimensions of the query vector to fit in a shared memory of a
 * block; this number must be a multiple of `WarpSize * Veclen`.
 * @param[in] query a pointer to all queries in a row-major contiguous format [gridDim.y, dim]
 * @param[in] coarse_index a pointer to the cluster indices to search through [n_probes]
 * @param[in] list_indices index<T, IdxT>.indices
 * @param[in] list_data index<T, IdxT>.data
 * @param[in] list_sizes index<T, IdxT>.list_sizes
 * @param[in] list_offsets index<T, IdxT>.list_offsets
 * @param n_probes
 * @param k
 * @param dim
 * @param sample_filter
 * @param[out] neighbors
 * @param[out] distances
 */
template <typename T, typename AccT, typename IdxT, int Capacity, bool Ascending>
__device__ __forceinline__ void interleaved_scan_impl(const uint32_t query_smem_elems,
                                                      const T* query,
                                                      const uint32_t* coarse_index,
                                                      const T* const* list_data_ptrs,
                                                      const uint32_t* list_sizes,
                                                      const uint32_t queries_offset,
                                                      const uint32_t n_probes,
                                                      const uint32_t k,
                                                      const uint32_t max_samples,
                                                      const uint32_t* chunk_indices,
                                                      const uint32_t dim,
                                                      IdxT* const* const inds_ptrs,
                                                      uint32_t* bitset_ptr,
                                                      IdxT bitset_len,
                                                      IdxT original_nbits,
                                                      uint32_t* neighbors,
                                                      float* distances)
{
  extern __shared__ __align__(256) uint8_t interleaved_scan_kernel_smem[];
  constexpr bool kManageLocalTopK = Capacity > 0;
  // Using shared memory for the (part of the) query;
  // This allows to save on global memory bandwidth when reading index and query
  // data at the same time.
  // Its size is `query_smem_elems`.
  T* query_shared = reinterpret_cast<T*>(interleaved_scan_kernel_smem);
  // Make the query input and output point to this block's shared query
  {
    const int query_id = blockIdx.y;
    query += query_id * dim;
    if constexpr (kManageLocalTopK) {
      neighbors += query_id * k * gridDim.x + blockIdx.x * k;
      distances += query_id * k * gridDim.x + blockIdx.x * k;
    } else {
      distances += query_id * uint64_t(max_samples);
    }
    chunk_indices += (n_probes * query_id);
    coarse_index += query_id * n_probes;
  }

  // Copy a part of the query into shared memory for faster processing
  copy_vectorized(query_shared, query, std::min(dim, query_smem_elems));
  __syncthreads();

  using local_topk_t = block_sort_t<Capacity, Ascending, float, uint32_t>;
  local_topk_t queue(k);
  {
    using align_warp  = raft::Pow2<raft::WarpSize>;
    const int lane_id = align_warp::mod(threadIdx.x);

    // How many full warps needed to compute the distance (without remainder)
    const uint32_t full_warps_along_dim = align_warp::roundDown(dim);

    const uint32_t shm_assisted_dim =
      (dim > query_smem_elems) ? query_smem_elems : full_warps_along_dim;

    // Every CUDA block scans one cluster at a time.
    for (int probe_id = blockIdx.x; probe_id < n_probes; probe_id += gridDim.x) {
      const uint32_t list_id = coarse_index[probe_id];  // The id of cluster(list)

      // The number of vectors in each cluster(list); [nlist]
      const uint32_t list_length = list_sizes[list_id];

      // The number of interleaved groups to be processed
      const uint32_t num_groups =
        align_warp::div(list_length + align_warp::Mask);  // ceildiv by power of 2

      uint32_t sample_offset = 0;
      if (probe_id > 0) { sample_offset = chunk_indices[probe_id - 1]; }
      assert(list_length == chunk_indices[probe_id] - sample_offset);
      if constexpr (!kManageLocalTopK) {
        // max_samples is zero/unused in the kManageLocalTopK mode
        assert(sample_offset + list_length <= max_samples);
      }

      constexpr uint32_t kNumWarps = kThreadsPerBlock / raft::WarpSize;
      // Every warp reads WarpSize vectors and computes the distances to them.
      // Then, the distances and corresponding ids are distributed among the threads,
      // and each thread adds one (id, dist) pair to the filtering queue.
      for (uint32_t group_id = align_warp::div(threadIdx.x); group_id < num_groups;
           group_id += kNumWarps) {
        AccT dist         = 0;
        AccT norm_query   = 0;
        AccT norm_dataset = 0;
        // This is where this warp begins reading data (start position of an interleaved group)
        const T* data = list_data_ptrs[list_id] + (group_id * kIndexGroupSize) * dim;

        // This is the vector a given lane/thread handles
        const uint32_t vec_id = group_id * raft::WarpSize + lane_id;
        const bool valid      = vec_id < list_length && sample_filter<IdxT>(inds_ptrs,
                                                                       queries_offset + blockIdx.y,
                                                                       list_id,
                                                                       vec_id,
                                                                       bitset_ptr,
                                                                       bitset_len,
                                                                       original_nbits);

        // Enqueue one element per thread
        float val = local_topk_t::queue_t::kDummy;
        if (valid) {
          val = load_and_compute_dist<T, AccT>(dist,
                                               norm_query,
                                               norm_dataset,
                                               shm_assisted_dim,
                                               data,
                                               query,
                                               query_shared,
                                               dim,
                                               query_smem_elems);
        }

        if constexpr (kManageLocalTopK) {
          queue.add(val, sample_offset + vec_id);
        } else {
          if (vec_id < list_length) distances[sample_offset + vec_id] = val;
        }
      }

      // fill up unused slots for current query
      if constexpr (!kManageLocalTopK) {
        if (probe_id + 1 == n_probes) {
          for (uint32_t i = threadIdx.x + sample_offset + list_length; i < max_samples;
               i += blockDim.x) {
            distances[i] = local_topk_t::queue_t::kDummy;
          }
        }
      }
    }
  }

  // finalize and store selected neighbours
  if constexpr (kManageLocalTopK) {
    __syncthreads();
    queue.done(interleaved_scan_kernel_smem);
    queue.store(distances, neighbors, [](auto val) { return post_process(val); });
  }
}

}  // namespace cuvs::neighbors::ivf_flat::detail
