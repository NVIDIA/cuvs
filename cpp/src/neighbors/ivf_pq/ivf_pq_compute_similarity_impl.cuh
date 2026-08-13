/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../ivf_common.cuh"     // dummy_block_sort_t
#include "../sample_filter.cuh"  // none_sample_filter
#include "detail/jit_lto_kernels/block_sort.cuh"
#include "detail/jit_lto_kernels/compute_similarity_planner.hpp"
#include "detail/jit_lto_kernels/kernel_def.cuh"
#include "ivf_pq_compute_similarity.hpp"  // cuvs::neighbors::ivf_pq::detail::selected
#include "ivf_pq_fp_8bit.cuh"
#include <cuvs/distance/distance.hpp>  // cuvs::distance::DistanceType
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>               // codebook_gen
#include <raft/matrix/detail/select_warpsort.cuh>  // matrix::detail::select::warpsort::warp_sort_distributed
#include <raft/util/cuda_rt_essentials.hpp>  // RAFT_CUDA_TRY
#include <raft/util/device_atomics.cuh>      // raft::atomicMin
#include <raft/util/pow2_utils.cuh>          // raft::Pow2
#include <raft/util/vectorized.cuh>          // raft::TxN_t

#include <rmm/cuda_stream_view.hpp>  // rmm::cuda_stream_view

namespace cuvs::neighbors::ivf_pq::detail {

/**
 * Maximum value of k for the fused calculate & select in ivfpq.
 *
 * If runtime value of k is larger than this, the main search operation
 * is split into two kernels (per batch, first calculate distance, then select top-k).
 */
static constexpr int kMaxCapacity = 128;
static_assert((kMaxCapacity >= 32) && !(kMaxCapacity & (kMaxCapacity - 1)),
              "kMaxCapacity must be a power of two, not smaller than the WarpSize.");

// using weak attribute here, because it may be compiled multiple times.
auto RAFT_WEAK_FUNCTION is_local_topk_feasible(uint32_t k, uint32_t n_probes, uint32_t n_queries)
  -> bool
{
  if (k > kMaxCapacity) { return false; }            // warp_sort not possible
  if (n_queries * n_probes <= 16) { return false; }  // overall amount of work is too small
  return true;
}

/**
 * Estimate a carveout value as expected by `cudaFuncAttributePreferredSharedMemoryCarveout`
 * (which does not take into account `reservedSharedMemPerBlock`),
 * given by a desired schmem-L1 split and a per-block memory requirement in bytes.
 *
 * NB: As per the programming guide, the memory carveout setting is just a hint for the driver; it's
 * free to choose any shmem-L1 configuration it deems appropriate. For example, if you set the
 * carveout to zero, it will choose a non-zero config that will allow to run at least one active
 * block per SM.
 *
 * @param shmem_fraction
 *   a fraction representing a desired split (shmem / (shmem + L1)) [0, 1].
 * @param shmem_per_block
 *   a shared memory usage per block (dynamic + static shared memory sizes), in bytes.
 * @param dev_props
 *   device properties.
 * @return
 *   a carveout value in percents [0, 100].
 */
constexpr inline auto estimate_carveout(double shmem_fraction,
                                        size_t shmem_per_block,
                                        const cudaDeviceProp& dev_props) -> int
{
  using shmem_unit = raft::Pow2<128>;
  size_t m         = shmem_unit::roundUp(shmem_per_block);
  size_t r         = dev_props.reservedSharedMemPerBlock;
  size_t s         = dev_props.sharedMemPerMultiprocessor;
  return (size_t(100 * s * m * shmem_fraction) - (m - 1) * r) / (s * (m + r));
}

template <typename OutT>
auto get_out_type_tag()
{
  if constexpr (std::is_same_v<OutT, float>) {
    return tag_out_f{};
  } else if constexpr (std::is_same_v<OutT, half>) {
    return tag_out_h{};
  } else {
    static_assert(sizeof(OutT) == 0, "Unsupported OutT type");
  }
}

template <typename LutT>
auto get_lut_type_tag()
{
  if constexpr (std::is_same_v<LutT, float>) {
    return tag_lut_f{};
  } else if constexpr (std::is_same_v<LutT, half>) {
    return tag_lut_h{};
  } else if constexpr (std::is_same_v<LutT, fp_8bit<5u, false>>) {
    return tag_lut_fp8_unsigned{};
  } else if constexpr (std::is_same_v<LutT, fp_8bit<5u, true>>) {
    return tag_lut_fp8_signed{};
  } else {
    static_assert(sizeof(LutT) == 0, "Unsupported LutT type");
  }
}

template <typename FilterT>
auto get_filter_type_tag()
{
  using namespace cuvs::neighbors::filtering;

  // Determine the filter implementation tag
  if constexpr (std::is_same_v<FilterT, none_sample_filter>) {
    return cuvs::neighbors::detail::tag_filter_none{};
  }
  if constexpr (std::is_same_v<FilterT, bitset_filter<uint32_t, int64_t>>) {
    return cuvs::neighbors::detail::tag_filter_bitset{};
  }
}

template <typename OutT,
          typename LutT,
          bool PrecompBaseDiff,
          bool EnableSMemLut,
          uint32_t PqBits,
          int Capacity,
          typename FilterT,
          typename MetricTag,
          bool IncrementScore>
auto kernel_try_capacity(uint32_t k_max)
{
  if constexpr (Capacity > 0) {
    if (k_max == 0 || k_max > Capacity) {
      return kernel_try_capacity<OutT,
                                 LutT,
                                 PrecompBaseDiff,
                                 EnableSMemLut,
                                 PqBits,
                                 0,
                                 FilterT,
                                 MetricTag,
                                 IncrementScore>(k_max);
    }
  }
  if constexpr (Capacity > 1) {
    if (k_max * 2 <= Capacity) {
      return kernel_try_capacity<OutT,
                                 LutT,
                                 PrecompBaseDiff,
                                 EnableSMemLut,
                                 PqBits,
                                 (Capacity / 2),
                                 FilterT,
                                 MetricTag,
                                 IncrementScore>(k_max);
    }
  }

  using out_tag    = decltype(get_out_type_tag<OutT>());
  using lut_tag    = decltype(get_lut_type_tag<LutT>());
  using filter_tag = decltype(get_filter_type_tag<FilterT>());
  using precomp_base_diff_metric_tag =
    std::conditional_t<PrecompBaseDiff, MetricTag, tag_metric_none>;

  constexpr bool kManageLocalTopK = Capacity > 0;

  ComputeSimilarityPlanner planner;
  planner.add_entrypoint<out_tag, lut_tag>();
  planner.add_prepare_lut_function<lut_tag, EnableSMemLut, PqBits>();
  planner.add_store_calculated_distances_function<out_tag, kManageLocalTopK>();
  planner.add_precompute_base_diff_function<precomp_base_diff_metric_tag>();
  planner.add_create_lut_function<lut_tag, MetricTag, PrecompBaseDiff, PqBits>();
  planner.add_compute_distances_function<out_tag, lut_tag, Capacity>();
  planner.add_get_early_stop_limit_function<out_tag, MetricTag>();
  planner.add_sample_filter_function<filter_tag>();
  planner.add_get_line_width_function<PqBits>();
  planner.add_compute_score_function<out_tag, lut_tag, PqBits>();
  planner.add_increment_score_function<out_tag, IncrementScore>();
  return planner.get_launcher();
}

template <typename OutT,
          typename LutT,
          bool PrecompBaseDiff,
          bool EnableSMemLut,
          typename FilterT,
          typename MetricTag,
          bool IncrementScore>
auto get_compute_similarity_launcher(uint32_t pq_bits, uint32_t k_max)
{
  switch (pq_bits) {
    case 4:
      return kernel_try_capacity<OutT,
                                 LutT,
                                 PrecompBaseDiff,
                                 EnableSMemLut,
                                 4,
                                 kMaxCapacity,
                                 FilterT,
                                 MetricTag,
                                 IncrementScore>(k_max);
    case 5:
      return kernel_try_capacity<OutT,
                                 LutT,
                                 PrecompBaseDiff,
                                 EnableSMemLut,
                                 5,
                                 kMaxCapacity,
                                 FilterT,
                                 MetricTag,
                                 IncrementScore>(k_max);
    case 6:
      return kernel_try_capacity<OutT,
                                 LutT,
                                 PrecompBaseDiff,
                                 EnableSMemLut,
                                 6,
                                 kMaxCapacity,
                                 FilterT,
                                 MetricTag,
                                 IncrementScore>(k_max);
    case 7:
      return kernel_try_capacity<OutT,
                                 LutT,
                                 PrecompBaseDiff,
                                 EnableSMemLut,
                                 7,
                                 kMaxCapacity,
                                 FilterT,
                                 MetricTag,
                                 IncrementScore>(k_max);
    case 8:
      return kernel_try_capacity<OutT,
                                 LutT,
                                 PrecompBaseDiff,
                                 EnableSMemLut,
                                 8,
                                 kMaxCapacity,
                                 FilterT,
                                 MetricTag,
                                 IncrementScore>(k_max);
    default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
  }
}

/** Estimate the occupancy for the given kernel on the given device. */
struct occupancy_t {
  using shmem_unit = raft::Pow2<128>;

  int blocks_per_sm = 0;
  double occupancy  = 0.0;
  double shmem_use  = 1.0;

  inline occupancy_t() = default;
  inline occupancy_t(size_t smem,
                     uint32_t n_threads,
                     cudaKernel_t kernel,
                     const cudaDeviceProp& dev_props)
  {
    RAFT_CUDA_TRY(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, n_threads, smem));
    occupancy = double(blocks_per_sm * n_threads) / double(dev_props.maxThreadsPerMultiProcessor);
    shmem_use = double(shmem_unit::roundUp(smem) * blocks_per_sm) /
                double(dev_props.sharedMemPerMultiprocessor);
  }
};

template <typename OutT, typename LutT>
void compute_similarity_run(selected<OutT, LutT> s,
                            rmm::cuda_stream_view stream,
                            uint32_t dim,
                            uint32_t n_probes,
                            uint32_t pq_dim,
                            uint32_t n_queries,
                            uint32_t queries_offset,
                            codebook_gen codebook_kind,
                            uint32_t topk,
                            uint32_t max_samples,
                            const float* cluster_centers,
                            const float* pq_centers,
                            const uint8_t* const* pq_dataset,
                            const uint32_t* cluster_labels,
                            const uint32_t* _chunk_indices,
                            const float* queries,
                            const uint32_t* index_list,
                            float* query_kths,
                            const int64_t* const* inds_ptrs,
                            uint32_t* bitset_ptr,
                            int64_t bitset_len,
                            int64_t original_nbits,
                            LutT* lut_scores,
                            OutT* _out_scores,
                            uint32_t* _out_indices)
{
  s.launcher->template dispatch<compute_similarity_func_t<OutT, LutT>>(stream,
                                                                       s.grid_dim,
                                                                       s.block_dim,
                                                                       s.smem_size,
                                                                       dim,
                                                                       n_probes,
                                                                       pq_dim,
                                                                       n_queries,
                                                                       queries_offset,
                                                                       codebook_kind,
                                                                       topk,
                                                                       max_samples,
                                                                       cluster_centers,
                                                                       pq_centers,
                                                                       pq_dataset,
                                                                       cluster_labels,
                                                                       _chunk_indices,
                                                                       queries,
                                                                       index_list,
                                                                       query_kths,
                                                                       inds_ptrs,
                                                                       bitset_ptr,
                                                                       bitset_len,
                                                                       original_nbits,
                                                                       lut_scores,
                                                                       _out_scores,
                                                                       _out_indices);
  RAFT_CHECK_CUDA(stream);
}

/**
 * Use heuristics to choose an optimal instance of the search kernel.
 * It selects among a few kernel variants (with/out using shared mem for
 * lookup tables / precomputed distances) and tries to choose the block size
 * to maximize kernel occupancy.
 *
 * @param manage_local_topk
 *    whether use the fused calculate+select or just calculate the distances for each
 *    query and probed cluster.
 *
 * @param locality_hint
 *    beyond this limit do not consider increasing the number of active blocks per SM
 *    would improve locality anymore.
 */
template <typename OutT, typename LutT, typename FilterT, typename MetricTag, bool IncrementScore>
auto compute_similarity_select(const cudaDeviceProp& dev_props,
                               bool manage_local_topk,
                               int locality_hint,
                               double preferred_shmem_carveout,
                               uint32_t pq_bits,
                               uint32_t pq_dim,
                               uint32_t precomp_data_count,
                               uint32_t n_queries,
                               uint32_t n_probes,
                               uint32_t topk) -> selected<OutT, LutT>
{
  // Shared memory for storing the lookup table
  size_t lut_mem = sizeof(LutT) * (pq_dim << pq_bits);
  // Shared memory for storing pre-computed pieces to speedup the lookup table construction
  // (e.g. the distance between a cluster center and the query for L2).
  size_t bdf_mem = sizeof(float) * precomp_data_count;

  // Shared memory used by the fused top-k during cluster scanning;
  // may overlap with the precomputed distance array
  struct ltk_add_mem_t {
    size_t (*mem_required)(uint32_t);

    ltk_add_mem_t(bool manage_local_topk, uint32_t topk)
      : mem_required(pq_block_sort<kMaxCapacity, OutT, uint32_t>::get_mem_required(
          manage_local_topk ? topk : 0))
    {
    }

    [[nodiscard]] auto operator()(uint32_t n_threads) const -> size_t
    {
      return mem_required(n_threads);
    }
  } ltk_add_mem{manage_local_topk, topk};

  // Shared memory for the fused top-k component;
  // may overlap with all other uses of shared memory
  struct ltk_reduce_mem_t {
    uint32_t subwarp_size;
    uint32_t topk;
    bool manage_local_topk;
    ltk_reduce_mem_t(bool manage_local_topk, uint32_t topk)
      : manage_local_topk(manage_local_topk), topk(topk)
    {
      subwarp_size = raft::WarpSize;
      while (topk * 2 <= subwarp_size) {
        subwarp_size /= 2;
      }
    }

    [[nodiscard]] auto operator()(uint32_t n_threads) const -> size_t
    {
      return manage_local_topk ? raft::matrix::detail::select::warpsort::
                                   template calc_smem_size_for_block_wide<OutT, uint32_t>(
                                     n_threads / subwarp_size, topk)
                               : 0;
    }
  } ltk_reduce_mem{manage_local_topk, topk};

  struct total_shared_mem_t {
    ltk_add_mem_t& ltk_add_mem;
    ltk_reduce_mem_t& ltk_reduce_mem;
    size_t lut_mem;
    size_t bdf_mem;
    [[nodiscard]] auto operator()(uint32_t n_threads) const -> size_t
    {
      return std::max(ltk_reduce_mem(n_threads),
                      lut_mem + std::max(bdf_mem, ltk_add_mem(n_threads)));
    }
  };

  // Total amount of work; should be enough to occupy the GPU.
  uint32_t n_blocks = n_queries * n_probes;

  // The minimum block size we may want:
  //   1. It's a power-of-two for efficient L1 caching of pq_centers values
  //      (multiples of `1 << pq_bits`).
  //   2. It should be large enough to fully utilize an SM.
  uint32_t n_threads_min = raft::WarpSize;
  while (dev_props.maxBlocksPerMultiProcessor * int(n_threads_min) <
         dev_props.maxThreadsPerMultiProcessor) {
    n_threads_min *= 2;
  }
  // Further increase the minimum block size to make sure full device occupancy
  // (NB: this may lead to `n_threads_min` being larger than the kernel's maximum)
  while (int(n_blocks * n_threads_min) <
           dev_props.multiProcessorCount * dev_props.maxThreadsPerMultiProcessor &&
         int(n_threads_min) < dev_props.maxThreadsPerBlock) {
    n_threads_min *= 2;
  }
  // Even further, increase it to allow less blocks per SM if there not enough queries.
  // With this, we reduce the chance of different clusters being processed by two blocks
  // on the same SM and thus improve the data locality for L1 caching.
  while (int(n_queries * n_threads_min) < dev_props.maxThreadsPerMultiProcessor &&
         int(n_threads_min) < dev_props.maxThreadsPerBlock) {
    n_threads_min *= 2;
  }

  // Granularity of changing the number of threads when computing the maximum block size.
  // It's good to have it multiple of the PQ book width.
  uint32_t n_threads_gty = raft::round_up_safe<uint32_t>(1u << pq_bits, raft::WarpSize);

  /*
   Shared memory / L1 cache balance is the main limiter of this kernel.
   The more blocks per SM we launch, the more shared memory we need. Besides that, we have
   three versions of the kernel varying in performance and shmem usage.

   We try the most demanding and the fastest kernel first, trying to maximize occupancy with
   the minimum number of blocks (just one, really). Then, we tweak the `n_threads` to further
   optimize occupancy and data locality for the L1 cache.
   */
  auto topk_or_zero = manage_local_topk ? topk : 0u;
  auto conf_fast =
    get_compute_similarity_launcher<OutT, LutT, true, true, FilterT, MetricTag, IncrementScore>(
      pq_bits, topk_or_zero);
  auto conf_no_basediff =
    get_compute_similarity_launcher<OutT, LutT, false, true, FilterT, MetricTag, IncrementScore>(
      pq_bits, topk_or_zero);
  auto conf_no_smem_lut =
    get_compute_similarity_launcher<OutT, LutT, true, false, FilterT, MetricTag, IncrementScore>(
      pq_bits, topk_or_zero);
  std::array candidates{
    std::make_tuple(
      conf_fast, total_shared_mem_t{ltk_add_mem, ltk_reduce_mem, lut_mem, bdf_mem}, true),
    std::make_tuple(
      conf_no_basediff, total_shared_mem_t{ltk_add_mem, ltk_reduce_mem, lut_mem, 0}, true),
    std::make_tuple(
      conf_no_smem_lut, total_shared_mem_t{ltk_add_mem, ltk_reduce_mem, 0, bdf_mem}, false)};

  // we may allow slightly lower than 100% occupancy;
  constexpr double kTargetOccupancy = 0.75;
  // This struct is used to select the better candidate
  occupancy_t selected_perf{};
  selected<OutT, LutT> selected_config;
  for (auto [launcher, smem_size_f, lut_is_in_shmem] : candidates) {
    if (smem_size_f(raft::WarpSize) > dev_props.sharedMemPerBlockOptin) {
      // Even a single block cannot fit into an SM due to shmem requirements. Skip the candidate.
      continue;
    }

    auto kernel = launcher->get_kernel();

    // First, we set the carveout hint to the preferred value. The driver will increase this if
    // needed to run at least one block per SM. At the same time, if more blocks fit into one SM,
    // this carveout value will limit the calculated occupancy. When we're done selecting the best
    // launch configuration, we will tighten the carveout once more, based on the final memory
    // usage and occupancy.
    const int max_carveout =
      estimate_carveout(preferred_shmem_carveout, smem_size_f(raft::WarpSize), dev_props);
    RAFT_CUDA_TRY(
      cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, max_carveout));

    // Get the theoretical maximum possible number of threads per block
    cudaFuncAttributes kernel_attrs;
    RAFT_CUDA_TRY(cudaFuncGetAttributes(&kernel_attrs, kernel));
    uint32_t n_threads =
      raft::round_down_safe<uint32_t>(kernel_attrs.maxThreadsPerBlock, n_threads_gty);

    // Actual required shmem depens on the number of threads
    size_t smem_size = smem_size_f(n_threads);

    // Make sure the kernel can get enough shmem.
    cudaError_t cuda_status =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (cuda_status != cudaSuccess) {
      RAFT_EXPECTS(
        cuda_status == cudaGetLastError(),
        "Tried to reset the expected cuda error code, but it didn't match the expectation");
      // Failed to request enough shmem for the kernel. Skip the candidate.
      continue;
    }

    occupancy_t cur(smem_size, n_threads, kernel, dev_props);
    if (cur.blocks_per_sm <= 0) {
      // For some reason, we still cannot make this kernel run. Skip the candidate.
      continue;
    }

    {
      // Try to reduce the number of threads to increase occupancy and data locality
      auto n_threads_tmp = n_threads_min;
      while (n_threads_tmp * 2 < n_threads) {
        n_threads_tmp *= 2;
      }
      if (n_threads_tmp < n_threads) {
        while (n_threads_tmp >= n_threads_min) {
          auto smem_size_tmp = smem_size_f(n_threads_tmp);
          occupancy_t tmp(smem_size_tmp, n_threads_tmp, kernel, dev_props);
          bool select_it = false;
          if (lut_is_in_shmem && locality_hint >= tmp.blocks_per_sm) {
            // Normally, the smaller the block the better for L1 cache hit rate.
            // Hence, the occupancy should be "just good enough"
            select_it = tmp.occupancy >= min(kTargetOccupancy, cur.occupancy);
          } else if (lut_is_in_shmem) {
            // If we don't have enough repeating probes (locality_hint < tmp.blocks_per_sm),
            // the locality is not going to improve with increasing the number of blocks per SM.
            // Hence, the only metric here is the occupancy.
            bool improves_occupancy = tmp.occupancy > cur.occupancy;
            // Otherwise, the performance still improves with a smaller block size,
            // given there is enough work to do
            bool improves_parallelism =
              tmp.occupancy == cur.occupancy &&
              7u * tmp.blocks_per_sm * dev_props.multiProcessorCount <= n_blocks;
            select_it = improves_occupancy || improves_parallelism;
          } else {
            // If we don't use shared memory for the lookup table, increasing the number of blocks
            // is very taxing on the global memory usage.
            // In this case, the occupancy must increase a lot to make it worth the cost.
            select_it = tmp.occupancy >= min(1.0, cur.occupancy / kTargetOccupancy);
          }
          if (select_it) {
            n_threads = n_threads_tmp;
            smem_size = smem_size_tmp;
            cur       = tmp;
          }
          n_threads_tmp /= 2;
        }
      }
    }

    {
      if (selected_perf.occupancy <= 0.0  // no candidate yet
          || (selected_perf.occupancy < cur.occupancy * kTargetOccupancy &&
              selected_perf.shmem_use >= cur.shmem_use)  // much improved occupancy
      ) {
        selected_perf = cur;
        if (lut_is_in_shmem) {
          selected_config = selected<OutT, LutT>(
            std::move(launcher), dim3(n_blocks, 1, 1), dim3(n_threads, 1, 1), smem_size, size_t(0));
        } else {
          // When the global memory is used for the lookup table, we need to minimize the grid
          // size; otherwise, the kernel may quickly run out of memory.
          auto n_blocks_min =
            std::min<uint32_t>(n_blocks, cur.blocks_per_sm * dev_props.multiProcessorCount);
          selected_config = selected<OutT, LutT>(std::move(launcher),
                                                 dim3(n_blocks_min, 1, 1),
                                                 dim3(n_threads, 1, 1),
                                                 smem_size,
                                                 size_t(n_blocks_min) * size_t(pq_dim << pq_bits));
        }
        // Actual shmem/L1 split wildly rounds up the specified preferred carveout, so we set here
        // a rather conservative bar; most likely, the kernel gets more shared memory than this,
        // and the occupancy doesn't get hurt.
        auto carveout = std::min<int>(max_carveout, std::ceil(100.0 * cur.shmem_use));
        RAFT_CUDA_TRY(
          cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, carveout));
        if (cur.occupancy >= kTargetOccupancy) { break; }
      } else if (selected_perf.occupancy > 0.0) {
        // If we found a reasonable candidate on a previous iteration, and this one is not better,
        // then don't try any more candidates because they are much slower anyway.
        break;
      }
    }
  }

  RAFT_EXPECTS(selected_perf.occupancy > 0.0,
               "Couldn't determine a working kernel launch configuration.");

  return selected_config;
}

}  // namespace cuvs::neighbors::ivf_pq::detail
