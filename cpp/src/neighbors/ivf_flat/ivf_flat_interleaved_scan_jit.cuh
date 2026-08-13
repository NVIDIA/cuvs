/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../ivf_common.cuh"
#include "detail/jit_lto_kernels/interleaved_scan_planner.hpp"
#include "detail/jit_lto_kernels/kernel_def.hpp"
#include <cstdint>
#include <cuvs/detail/jit_lto/NVRTCLTOFragmentCompiler.hpp>
#include <cuvs/detail/jit_lto/common_fragments.hpp>
#include <cuvs/detail/jit_lto/ivf_flat/interleaved_scan_fragments.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <optional>
#include <string>
#include <type_traits>

#include "../detail/ann_utils.cuh"
#include <cuvs/distance/distance.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/operators.hpp>
#include <raft/util/cuda_rt_essentials.hpp>  // RAFT_CUDA_TRY
#include <raft/util/pow2_utils.cuh>

#include <rmm/cuda_stream_view.hpp>

namespace cuvs::neighbors::ivf_flat::detail {
static constexpr int kThreadsPerBlock = 128;

using namespace cuvs::spatial::knn::detail;  // NOLINT

// Constexpr mapping functions from actual types to tags
template <typename T>
constexpr auto get_data_type_tag()
{
  if constexpr (std::is_same_v<T, float>) { return cuvs::neighbors::detail::tag_f{}; }
  if constexpr (std::is_same_v<T, __half>) { return cuvs::neighbors::detail::tag_h{}; }
  if constexpr (std::is_same_v<T, int8_t>) { return cuvs::neighbors::detail::tag_i8{}; }
  if constexpr (std::is_same_v<T, uint8_t>) { return cuvs::neighbors::detail::tag_u8{}; }
}

template <typename AccT>
constexpr auto get_acc_type_tag()
{
  if constexpr (std::is_same_v<AccT, float>) { return tag_acc_f{}; }
  if constexpr (std::is_same_v<AccT, __half>) { return tag_acc_h{}; }
  if constexpr (std::is_same_v<AccT, int32_t>) { return tag_acc_i32{}; }
  if constexpr (std::is_same_v<AccT, uint32_t>) { return tag_acc_u32{}; }
}

template <typename IdxT>
constexpr auto get_idx_type_tag()
{
  if constexpr (std::is_same_v<IdxT, int64_t>) { return cuvs::neighbors::detail::tag_index_i64{}; }
}

// Convert type to string for JIT code generation (strip cv/ref so const/volatile T still matches).
template <typename>
inline constexpr bool type_name_always_false_v = false;

template <typename T>
constexpr const char* type_name()
{
  using U = std::remove_cv_t<std::remove_reference_t<T>>;

  if constexpr (std::is_same_v<U, float>) {
    return "float";
  } else if constexpr (std::is_same_v<U, __half>) {
    return "__half";
  } else if constexpr (std::is_same_v<U, half>) {
    return "__half";
  } else if constexpr (std::is_same_v<U, int8_t> || std::is_same_v<U, signed char>) {
    return "int8_t";
  } else if constexpr (std::is_same_v<U, uint8_t> || std::is_same_v<U, unsigned char>) {
    return "uint8_t";
  } else if constexpr (std::is_same_v<U, int32_t> ||
                       (std::is_same_v<U, long> && sizeof(long) == sizeof(int32_t))) {
    return "int32_t";
  } else if constexpr (std::is_same_v<U, uint32_t> || std::is_same_v<U, unsigned int> ||
                       (std::is_same_v<U, unsigned long> && sizeof(unsigned long) == 4)) {
    return "uint32_t";
  } else if constexpr (std::is_same_v<U, int64_t>) {
    return "int64_t";
  } else {
    static_assert(type_name_always_false_v<U>, "Unsupported type to create UDF");
    return "";
  }
}

template <typename FilterT>
constexpr auto get_filter_type_tag()
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

/**
 *  Configure the gridDim.x to maximize GPU occupancy, but reduce the output size
 */
inline uint32_t configure_launch_x(uint32_t numQueries,
                                   uint32_t n_probes,
                                   int32_t sMemSize,
                                   cudaKernel_t func)
{
  int dev_id;
  RAFT_CUDA_TRY(cudaGetDevice(&dev_id));
  int num_sms;
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));
  int num_blocks_per_sm = 0;
  RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks_per_sm, func, kThreadsPerBlock, sMemSize));

  size_t min_grid_size = num_sms * num_blocks_per_sm;
  size_t min_grid_x    = raft::ceildiv<size_t>(min_grid_size, numQueries);
  return min_grid_x > n_probes ? n_probes : static_cast<uint32_t>(min_grid_x);
}

template <int Capacity,
          int Veclen,
          bool Ascending,
          bool ComputeNorm,
          typename T,
          typename AccT,
          typename IdxT,
          typename IvfSampleFilterTag,
          typename MetricTag,
          typename PostLambdaTag>
void launch_kernel(const index<T, IdxT>& index,
                   const T* queries,
                   const uint32_t* coarse_index,
                   const uint32_t num_queries,
                   const uint32_t queries_offset,
                   const uint32_t n_probes,
                   const uint32_t k,
                   const uint32_t max_samples,
                   const uint32_t* chunk_indices,
                   IdxT* const* const inds_ptrs,
                   cuda::std::optional<uint32_t*> bitset_ptr,
                   cuda::std::optional<IdxT> bitset_len,
                   cuda::std::optional<IdxT> original_nbits,
                   uint32_t* neighbors,
                   float* distances,
                   uint32_t& grid_dim_x,
                   rmm::cuda_stream_view stream,
                   const std::optional<std::string>& metric_udf)
{
  RAFT_EXPECTS(Veclen == index.veclen(),
               "Configured Veclen does not match the index interleaving pattern.");

  using DataTag = decltype(get_data_type_tag<T>());
  using AccTag  = decltype(get_acc_type_tag<AccT>());
  using IdxTag  = decltype(get_idx_type_tag<IdxT>());

  InterleavedScanPlanner kernel_planner;
  kernel_planner.add_entrypoint<DataTag, AccTag, IdxTag, Capacity, Ascending>();

  if constexpr (std::is_same_v<MetricTag, tag_metric_custom_udf>) {
    RAFT_EXPECTS(metric_udf.has_value(), "CustomUDF search requires metric_udf");
    std::string metric_udf_code = metric_udf.value();
    metric_udf_code +=
      experimental::udf::instantiate_udf(type_name<T>(), type_name<AccT>(), Veclen);
    auto udf_fragment = nvrtc_compiler().compile(metric_udf_code, metric_udf_code);
    kernel_planner.add_metric_udf_fragment(std::move(udf_fragment));
  } else {
    kernel_planner.add_metric_device_function<DataTag, AccTag, MetricTag, Veclen>();
  }

  kernel_planner.add_load_and_compute_dist_function<DataTag, AccTag, ComputeNorm, Veclen>();
  kernel_planner.add_filter_device_function<IdxTag, IvfSampleFilterTag>();
  kernel_planner.add_post_lambda_device_function<PostLambdaTag>();
  auto kernel_launcher = kernel_planner.get_launcher();

  const int max_query_smem  = 16384;
  uint32_t query_smem_elems = std::min<int>(
    max_query_smem / sizeof(T), raft::Pow2<Veclen * raft::WarpSize>::roundUp(index.dim()));
  size_t smem_size = query_smem_elems * sizeof(T);

  if constexpr (Capacity > 0) {
    constexpr int kSubwarpSize = std::min<int>(Capacity, raft::WarpSize);
    auto block_merge_mem =
      raft::matrix::detail::select::warpsort::calc_smem_size_for_block_wide<float, IdxT>(
        kThreadsPerBlock / kSubwarpSize, k);
    smem_size += std::max<int>(smem_size, block_merge_mem);
  }

  // power-of-two less than cuda limit (for better addr alignment)
  constexpr uint32_t kMaxGridY = 32768;

  if (grid_dim_x == 0) {
    grid_dim_x = configure_launch_x(
      std::min(kMaxGridY, num_queries), n_probes, smem_size, kernel_launcher->get_kernel());
    return;
  }

  // Pass individual filter parameters like CAGRA does
  // The kernel will construct filter_data struct internally when needed

  for (uint32_t query_offset = 0; query_offset < num_queries; query_offset += kMaxGridY) {
    uint32_t grid_dim_y = std::min<uint32_t>(kMaxGridY, num_queries - query_offset);
    dim3 grid_dim(grid_dim_x, grid_dim_y, 1);
    dim3 block_dim(kThreadsPerBlock);
    RAFT_LOG_TRACE(
      "Launching the ivf-flat interleaved_scan_kernel (%d, %d, 1) x (%d, 1, 1), n_probes = %d, "
      "smem_size = %d",
      grid_dim.x,
      grid_dim.y,
      block_dim.x,
      n_probes,
      smem_size);
    kernel_launcher->dispatch<interleaved_scan_func_t<T, IdxT>>(
      stream,
      grid_dim,
      block_dim,
      smem_size,
      query_smem_elems,
      queries,
      coarse_index,
      const_cast<const T* const*>(index.data_ptrs().data_handle()),
      index.list_sizes().data_handle(),
      static_cast<uint32_t>(queries_offset + query_offset),
      n_probes,
      k,
      max_samples,
      chunk_indices,
      index.dim(),
      // sample_filter,
      inds_ptrs,
      bitset_ptr.value_or(nullptr),
      bitset_len.value_or(0),
      original_nbits.value_or(0),
      neighbors,
      distances);
    queries += grid_dim_y * index.dim();
    if constexpr (Capacity > 0) {
      neighbors += grid_dim_y * grid_dim_x * k;
      distances += grid_dim_y * grid_dim_x * k;
    } else {
      distances += grid_dim_y * max_samples;
    }
    chunk_indices += grid_dim_y * n_probes;
    coarse_index += grid_dim_y * n_probes;
  }
}

/** Select the distance computation function and forward the rest of the arguments. */
template <int Capacity,
          int Veclen,
          bool Ascending,
          typename T,
          typename AccT,
          typename IdxT,
          typename IvfSampleFilterTag,
          typename... Args>
void launch_with_fixed_consts(cuvs::distance::DistanceType metric, Args&&... args)
{
  switch (metric) {
    case cuvs::distance::DistanceType::L2Expanded:
    case cuvs::distance::DistanceType::L2Unexpanded:
      return launch_kernel<Capacity,
                           Veclen,
                           Ascending,
                           false,
                           T,
                           AccT,
                           IdxT,
                           IvfSampleFilterTag,
                           tag_metric_euclidean,
                           tag_post_process_identity>(std::forward<Args>(args)...);
    case cuvs::distance::DistanceType::L2SqrtExpanded:
    case cuvs::distance::DistanceType::L2SqrtUnexpanded:
      return launch_kernel<Capacity,
                           Veclen,
                           Ascending,
                           false,
                           T,
                           AccT,
                           IdxT,
                           IvfSampleFilterTag,
                           tag_metric_euclidean,
                           tag_post_process_sqrt>(std::forward<Args>(args)...);
    case cuvs::distance::DistanceType::InnerProduct:
      return launch_kernel<Capacity,
                           Veclen,
                           Ascending,
                           false,
                           T,
                           AccT,
                           IdxT,
                           IvfSampleFilterTag,
                           tag_metric_inner_product,
                           tag_post_process_identity>(std::forward<Args>(args)...);
    case cuvs::distance::DistanceType::CosineExpanded:
      // NB: "Ascending" is reversed because the post-processing step is done after that sort
      return launch_kernel<Capacity,
                           Veclen,
                           !Ascending,
                           true,
                           T,
                           AccT,
                           IdxT,
                           IvfSampleFilterTag,
                           tag_metric_inner_product,
                           tag_post_process_compose>(
        std::forward<Args>(args)...);  // NB: update the description of `knn::ivf_flat::build` when
                                       // adding here a new metric.
    case cuvs::distance::DistanceType::CustomUDF:
      return launch_kernel<Capacity,
                           Veclen,
                           Ascending,
                           false,
                           T,
                           AccT,
                           IdxT,
                           IvfSampleFilterTag,
                           tag_metric_custom_udf,
                           tag_post_process_identity>(std::forward<Args>(args)...);
    default: RAFT_FAIL("The chosen distance metric is not supported (%d)", int(metric));
  }
}

/**
 * Lift the `capacity` and `veclen` parameters to the template level,
 * forward the rest of the arguments unmodified to `launch_interleaved_scan_kernel`.
 */
template <typename T,
          typename AccT,
          typename IdxT,
          typename IvfSampleFilterTag,
          int Capacity = raft::matrix::detail::select::warpsort::kMaxCapacity,
          int Veclen   = std::max<int>(1, 16 / sizeof(T))>
struct select_interleaved_scan_kernel {
  /**
   * Recursively reduce the `Capacity` and `Veclen` parameters until they match the
   * corresponding runtime arguments.
   * By default, this recursive process starts with maximum possible values of the
   * two parameters and ends with both values equal to 1.
   */
  template <typename... Args>
  static inline void run(int k_max, int veclen, bool select_min, Args&&... args)
  {
    if constexpr (Capacity > 0) {
      if (k_max == 0 || k_max > Capacity) {
        return select_interleaved_scan_kernel<T, AccT, IdxT, IvfSampleFilterTag, 0, Veclen>::run(
          k_max, veclen, select_min, std::forward<Args>(args)...);
      }
    }
    if constexpr (Capacity > 1) {
      if (k_max * 2 <= Capacity) {
        return select_interleaved_scan_kernel<T,
                                              AccT,
                                              IdxT,
                                              IvfSampleFilterTag,
                                              Capacity / 2,
                                              Veclen>::run(k_max,
                                                           veclen,
                                                           select_min,
                                                           std::forward<Args>(args)...);
      }
    }
    if constexpr (Veclen > 1) {
      if (veclen % Veclen != 0) {
        return select_interleaved_scan_kernel<T, AccT, IdxT, IvfSampleFilterTag, Capacity, 1>::run(
          k_max, 1, select_min, std::forward<Args>(args)...);
      }
    }
    // NB: this is the limitation of the warpsort structures that use a huge number of
    //     registers (used in the main kernel here).
    RAFT_EXPECTS(Capacity == 0 || k_max == Capacity,
                 "Capacity must be either 0 or a power-of-two not bigger than the maximum "
                 "allowed size matrix::detail::select::warpsort::kMaxCapacity (%d).",
                 raft::matrix::detail::select::warpsort::kMaxCapacity);
    RAFT_EXPECTS(
      veclen == Veclen,
      "Veclen must be power-of-two not bigger than the maximum allowed size for this data type.");
    if (select_min) {
      launch_with_fixed_consts<Capacity, Veclen, true, T, AccT, IdxT, IvfSampleFilterTag>(
        std::forward<Args>(args)...);
    } else {
      launch_with_fixed_consts<Capacity, Veclen, false, T, AccT, IdxT, IvfSampleFilterTag>(
        std::forward<Args>(args)...);
    }
  }
};

/**
 * @brief Configure and launch an appropriate template instance of the interleaved scan kernel.
 *
 * @tparam T value type
 * @tparam AccT accumulated type
 * @tparam IdxT type of the indices
 *
 * @param index previously built ivf-flat index
 * @param[in] queries device pointer to the query vectors [batch_size, dim]
 * @param[in] coarse_query_results device pointer to the cluster (list) ids [batch_size, n_probes]
 * @param n_queries batch size
 * @param[in] queries_offset
 *   An offset of the current query batch. It is used for feeding sample_filter with the
 *   correct query index.
 * @param k number of nearest neighbors.
 *            NB: the maximum value of `k` is limited statically by `kMaxCapacity`.
 * @param[out] neighbors device pointer to the result indices for each query and cluster
 * [batch_size, grid_dim_x, k]
 * @param[out] distances device pointer to the result distances for each query and cluster
 * [batch_size, grid_dim_x, k]
 * @param[inout] grid_dim_x number of blocks launched across all n_probes clusters;
 *               (one block processes one or more probes, hence: 1 <= grid_dim_x <= n_probes)
 * @param stream
 * @param sample_filter
 *   A filter that selects samples for a given query. Use an instance of none_sample_filter to
 *   provide a green light for every sample.
 */
template <typename T, typename AccT, typename IdxT, typename IvfSampleFilterT>
void ivfflat_interleaved_scan(const index<T, IdxT>& index,
                              const T* queries,
                              const uint32_t* coarse_query_results,
                              const uint32_t n_queries,
                              const uint32_t queries_offset,
                              cuvs::distance::DistanceType metric,
                              const uint32_t n_probes,
                              const uint32_t k,
                              const uint32_t max_samples,
                              const uint32_t* chunk_indices,
                              bool select_min,
                              IvfSampleFilterT sample_filter,
                              uint32_t* neighbors,
                              float* distances,
                              uint32_t& grid_dim_x,
                              rmm::cuda_stream_view stream,
                              const std::optional<std::string>& metric_udf)
{
  const uint32_t n_probes_clamped = std::min(n_probes, index.n_lists());
  const int capacity              = raft::bound_by_power_of_two(k);

  cuda::std::optional<uint32_t*> bitset_ptr;
  cuda::std::optional<IdxT> bitset_len;
  cuda::std::optional<IdxT> original_nbits;

  if constexpr (std::is_same_v<IvfSampleFilterT,
                               cuvs::neighbors::filtering::bitset_filter<uint32_t, IdxT>>) {
    bitset_ptr     = sample_filter.view().data();
    bitset_len     = sample_filter.view().size();
    original_nbits = sample_filter.view().get_original_nbits();
  }
  select_interleaved_scan_kernel<T, AccT, IdxT, decltype(get_filter_type_tag<IvfSampleFilterT>())>::
    run(capacity,
        index.veclen(),
        select_min,
        metric,
        index,
        queries,
        coarse_query_results,
        n_queries,
        queries_offset,
        n_probes_clamped,
        k,
        max_samples,
        chunk_indices,
        index.inds_ptrs().data_handle(),
        bitset_ptr,
        bitset_len,
        original_nbits,
        neighbors,
        distances,
        grid_dim_x,
        stream,
        metric_udf);
}

}  // namespace cuvs::neighbors::ivf_flat::detail
