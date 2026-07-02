/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../core/nvtx.hpp"
#include "../detail/ann_utils.cuh"
#include "../ivf_common.cuh"
#include "../sample_filter.cuh"
#include "detail/jit_lto_kernels/kernel_def.hpp"
#include "detail/jit_lto_kernels/scan_planner.hpp"
#include <cuvs/detail/jit_lto/common_fragments.hpp>
#include <cuvs/detail/jit_lto/ivf_sq/scan_fragments.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_sq.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/selection/select_k.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/detail/select_warpsort.cuh>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/integer_utils.hpp>

#include <rmm/resource_ref.hpp>

#include <thrust/fill.h>

#include <optional>

namespace cuvs::neighbors::ivf_sq::detail {

using namespace cuvs::spatial::knn::detail;  // NOLINT

// Maximum fused top-k capacity we instantiate for the scan kernel.
// Must match the highest Capacity case in ivf_sq_scan's switch.
static constexpr int kMaxSqScanCapacity = 256;
static_assert(kMaxSqScanCapacity <= raft::matrix::detail::select::warpsort::kMaxCapacity,
              "kMaxSqScanCapacity must not exceed the warpsort library's maximum supported "
              "capacity; reduce kMaxSqScanCapacity or update the warpsort dependency.");

auto RAFT_WEAK_FUNCTION is_local_topk_feasible(uint32_t k) -> bool
{
  return k <= kMaxSqScanCapacity;
}

// Compute shared-memory size for the scan kernel (3 x dim floats), plus the
// block_sort merge buffer when fused top-k is enabled. Must match the smem
// layout used by ivf_sq_scan_impl.
inline size_t sq_scan_smem_size(uint32_t dim) { return 3 * dim * sizeof(float); }

template <int Capacity>
size_t sq_scan_total_smem(uint32_t dim, uint32_t k)
{
  size_t scan_smem = sq_scan_smem_size(dim);
  if constexpr (Capacity > 0) {
    constexpr int kSubwarpSize = std::min<int>(Capacity, raft::WarpSize);
    int num_subwarps           = kSqScanThreads / kSubwarpSize;
    size_t merge_smem =
      raft::matrix::detail::select::warpsort::calc_smem_size_for_block_wide<float, uint32_t>(
        num_subwarps, k);
    return std::max(scan_smem, merge_smem);
  }
  return scan_smem;
}

// Map the runtime distance metric onto the corresponding compile-time tag and
// invoke `f` (a generic lambda) with the tag type. Used by the JIT-LTO launch
// dispatch to forward the metric to the planner without bloating the static
// library with the kernel body.
template <typename F>
void dispatch_metric_tag(cuvs::distance::DistanceType metric, F&& f)
{
  switch (metric) {
    case cuvs::distance::DistanceType::L2Expanded:
    case cuvs::distance::DistanceType::L2SqrtExpanded: f(tag_metric_l2{}); return;
    case cuvs::distance::DistanceType::InnerProduct: f(tag_metric_ip{}); return;
    case cuvs::distance::DistanceType::CosineExpanded: f(tag_metric_cosine{}); return;
    default: RAFT_FAIL("Unsupported metric type for IVF-SQ scan.");
  }
}

template <typename FilterT>
constexpr auto get_filter_type_tag()
{
  using namespace cuvs::neighbors::filtering;
  if constexpr (std::is_same_v<FilterT, none_sample_filter>) {
    return cuvs::neighbors::detail::tag_filter_none{};
  } else if constexpr (std::is_same_v<FilterT, bitset_filter<uint32_t, int64_t>>) {
    return cuvs::neighbors::detail::tag_filter_bitset{};
  }
}

// Choose grid.x to saturate the GPU. Mirrors the previous static
// configure_grid_dim_x but takes the JIT-linked cudaKernel_t.
inline uint32_t configure_grid_dim_x(
  uint32_t n_queries, uint32_t n_probes, int smem_size, int block_size, cudaKernel_t kernel)
{
  int dev_id;
  RAFT_CUDA_TRY(cudaGetDevice(&dev_id));
  int num_sms;
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));
  int num_blocks_per_sm = 0;
  RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks_per_sm, kernel, block_size, smem_size));

  size_t min_grid_size = size_t(num_sms) * num_blocks_per_sm;
  size_t min_grid_x    = raft::ceildiv<size_t>(min_grid_size, n_queries);
  return std::min<uint32_t>(n_probes, static_cast<uint32_t>(min_grid_x));
}

// ---------------------------------------------------------------------------
// JIT-LTO scan launcher
//
// The scan kernel `ivf_sq_scan_impl` lives as a fatbin fragment under
// `detail/jit_lto_kernels/`. At runtime we ask the planner for the linked
// kernel matching (Capacity, MetricTag, FilterTag), set the dynamic shared
// memory attribute, compute grid_dim_x via cudaOccupancy*, and dispatch.
//
// Grid layout matches the original static kernel:
//   kManageLocalTopK (Capacity > 0):
//     grid (grid_dim_x, n_queries) - each block loops over probes
//   otherwise (Capacity == 0):
//     grid (n_probes, n_queries) - one block per (query, probe)
// ---------------------------------------------------------------------------
template <int Capacity,
          typename MetricTag,
          typename CodeT,
          typename IvfSampleFilterTag,
          typename IvfSampleFilterT>
void launch_kernel(const index<CodeT>& idx,
                   const float* queries_float,
                   const float* query_norms,
                   uint32_t n_queries,
                   uint32_t n_probes,
                   uint32_t k,
                   uint32_t max_samples,
                   const uint32_t* coarse_indices,
                   const uint32_t* chunk_indices,
                   float* out_distances,
                   uint32_t* out_indices,
                   uint32_t& grid_dim_x,
                   rmm::cuda_stream_view stream,
                   IvfSampleFilterT sample_filter)
{
  static_assert(std::is_same_v<CodeT, uint8_t>, "IVF-SQ JIT-LTO scan only supports CodeT=uint8_t");

  constexpr bool kManageLocalTopK = (Capacity > 0);
  constexpr int kThreads          = kSqScanThreads;
  uint32_t dim                    = idx.dim();

  // The scan entrypoint is templated only on Capacity. The warpsort runs
  // ascending for all metrics; metric specialization is composed in at link
  // time via four device-function fragments (setup_invariant_smem,
  // setup_per_probe_smem, accumulate_distance, finalize_distance).
  IvfSqScanPlanner kernel_planner;
  kernel_planner.add_entrypoint<Capacity>();
  kernel_planner.add_setup_invariant_smem_function<MetricTag>();
  kernel_planner.add_setup_per_probe_smem_function<MetricTag>();
  kernel_planner.add_accumulate_distance_function<MetricTag>();
  kernel_planner.add_finalize_distance_function<MetricTag>();
  kernel_planner.add_filter_device_function<IvfSampleFilterTag>();
  auto kernel_launcher = kernel_planner.get_launcher();

  // Extract bitset arguments from the user's filter object so they can be
  // passed to the JIT-linked kernel as plain pointers/scalars; the JIT'd
  // sample_filter fragment reconstructs the bitset_view inside the kernel.
  int64_t* const* inds_ptrs_dev = idx.inds_ptrs().data_handle();
  uint32_t* bitset_ptr          = nullptr;
  int64_t bitset_len            = 0;
  int64_t original_nbits        = 0;
  if constexpr (std::is_same_v<IvfSampleFilterT,
                               cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>>) {
    bitset_ptr     = sample_filter.view().data();
    bitset_len     = sample_filter.view().size();
    original_nbits = sample_filter.view().get_original_nbits();
  }

  size_t smem = sq_scan_total_smem<Capacity>(dim, k);

  {
    int dev_id;
    RAFT_CUDA_TRY(cudaGetDevice(&dev_id));
    int max_smem;
    RAFT_CUDA_TRY(
      cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_id));
    RAFT_EXPECTS(smem <= size_t(max_smem),
                 "IVF-SQ scan kernel requires %zu bytes of shared memory (dim=%u, k=%u), "
                 "but the device supports at most %d bytes per block.",
                 smem,
                 dim,
                 k,
                 max_smem);
  }

  {
    int dev_id;
    RAFT_CUDA_TRY(cudaGetDevice(&dev_id));
    RAFT_CUDA_TRY(cudaKernelSetAttributeForDevice(kernel_launcher->get_kernel(),
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  static_cast<int>(smem),
                                                  dev_id));
  }

  constexpr uint32_t kMaxGridY = 65535;

  if constexpr (kManageLocalTopK) {
    if (grid_dim_x == 0) {
      grid_dim_x = configure_grid_dim_x(
        std::min(kMaxGridY, n_queries), n_probes, smem, kThreads, kernel_launcher->get_kernel());
      return;
    }
  }

  dim3 block(kThreads);

  for (uint32_t query_offset = 0; query_offset < n_queries; query_offset += kMaxGridY) {
    uint32_t batch = std::min(kMaxGridY, n_queries - query_offset);
    dim3 grid      = kManageLocalTopK ? dim3(grid_dim_x, batch) : dim3(n_probes, batch);

    const float* q_ptr  = queries_float + uint64_t(query_offset) * dim;
    const float* qn_ptr = query_norms ? query_norms + query_offset : query_norms;
    const uint32_t* ci  = coarse_indices + uint64_t(query_offset) * n_probes;
    const uint32_t* ch  = chunk_indices + uint64_t(query_offset) * n_probes;
    float* od           = out_distances;
    uint32_t* oi        = out_indices;
    if constexpr (kManageLocalTopK) {
      od += uint64_t(query_offset) * grid_dim_x * k;
      oi += uint64_t(query_offset) * grid_dim_x * k;
    } else {
      od += uint64_t(query_offset) * max_samples;
      oi += uint64_t(query_offset) * max_samples;
    }

    const uint8_t* const* data_ptrs = idx.data_ptrs().data_handle();
    const uint32_t* list_sizes      = idx.list_sizes().data_handle();
    const float* centers            = idx.centers().data_handle();
    const float* sq_vmin            = idx.sq_vmin().data_handle();
    const float* sq_delta           = idx.sq_delta().data_handle();

    kernel_launcher->dispatch<ivf_sq_scan_func_t<int64_t>>(stream,
                                                           grid,
                                                           block,
                                                           smem,
                                                           data_ptrs,
                                                           list_sizes,
                                                           ci,
                                                           q_ptr,
                                                           centers,
                                                           sq_vmin,
                                                           sq_delta,
                                                           qn_ptr,
                                                           n_probes,
                                                           dim,
                                                           k,
                                                           max_samples,
                                                           ch,
                                                           od,
                                                           oi,
                                                           inds_ptrs_dev,
                                                           bitset_ptr,
                                                           bitset_len,
                                                           original_nbits);
  }
}

// ---------------------------------------------------------------------------
// ivf_sq_scan: top-level scan dispatch with Capacity selection
// ---------------------------------------------------------------------------
template <typename CodeT, typename IvfSampleFilterT>
void ivf_sq_scan(raft::resources const& handle,
                 const index<CodeT>& idx,
                 const float* queries_float,
                 const float* query_norms,
                 uint32_t n_queries,
                 uint32_t n_probes,
                 uint32_t k,
                 uint32_t max_samples,
                 const uint32_t* coarse_indices,
                 const uint32_t* chunk_indices,
                 float* out_distances,
                 uint32_t* out_indices,
                 IvfSampleFilterT sample_filter,
                 uint32_t& grid_dim_x,
                 rmm::cuda_stream_view stream)
{
  // Determine the fused top-k capacity (0 = disabled / fallback to materialization)
  int capacity = is_local_topk_feasible(k) ? raft::bound_by_power_of_two(int(k)) : 0;

  // Snap to the nearest supported compile-time Capacity value (must be a
  // power of two). Values up to 32 share one instantiation; 64, 128 and 256
  // each get their own.  Beyond kMaxSqScanCapacity we fall back to the
  // non-fused path (Capacity == 0).
  if (capacity > 0 && capacity < 32) {
    capacity = 32;
  } else if (capacity > kMaxSqScanCapacity) {
    capacity = 0;
  }

  using IvfSampleFilterTag = decltype(get_filter_type_tag<IvfSampleFilterT>());

  auto fwd = [&](auto cap_tag) {
    constexpr int kCap = decltype(cap_tag)::value;
    dispatch_metric_tag(idx.metric(), [&](auto metric_tag) {
      using MetricTag = decltype(metric_tag);
      launch_kernel<kCap, MetricTag, CodeT, IvfSampleFilterTag, IvfSampleFilterT>(idx,
                                                                                  queries_float,
                                                                                  query_norms,
                                                                                  n_queries,
                                                                                  n_probes,
                                                                                  k,
                                                                                  max_samples,
                                                                                  coarse_indices,
                                                                                  chunk_indices,
                                                                                  out_distances,
                                                                                  out_indices,
                                                                                  grid_dim_x,
                                                                                  stream,
                                                                                  sample_filter);
    });
  };

  switch (capacity) {
    case 0: fwd(std::integral_constant<int, 0>{}); break;
    case 32: fwd(std::integral_constant<int, 32>{}); break;
    case 64: fwd(std::integral_constant<int, 64>{}); break;
    case 128: fwd(std::integral_constant<int, 128>{}); break;
    case 256: fwd(std::integral_constant<int, 256>{}); break;
    default: RAFT_FAIL("Unexpected capacity value %d", capacity);
  }
}

// ---------------------------------------------------------------------------
// search_impl — host-side search logic
// ---------------------------------------------------------------------------
template <typename T, typename CodeT, typename IvfSampleFilterT>
void search_impl(raft::resources const& handle,
                 const index<CodeT>& index,
                 const T* queries,
                 uint32_t n_queries,
                 uint32_t k,
                 uint32_t n_probes,
                 int64_t* neighbors,
                 float* distances,
                 rmm::device_async_resource_ref search_mr,
                 IvfSampleFilterT sample_filter)
{
  auto stream = raft::resource::get_cuda_stream(handle);
  auto dim    = index.dim();

  // The scan kernel emits min-close distances for every metric (IP scores are
  // negated by finalize_distance and undone by postprocess_distances below),
  // so all scan-related top-k selections use select_min = true. Only the
  // coarse search, which runs on the raw GEMM output, follows the metric's
  // native direction.
  const bool coarse_select_min = cuvs::distance::is_min_close(index.metric());

  std::size_t n_queries_probes = std::size_t(n_queries) * std::size_t(n_probes);

  bool needs_query_norms = index.metric() != cuvs::distance::DistanceType::InnerProduct;
  std::optional<raft::device_vector<float>> query_norm_dev{std::nullopt};
  if (needs_query_norms) {
    query_norm_dev.emplace(
      raft::make_device_mdarray<float>(handle, search_mr, raft::make_extents<uint32_t>(n_queries)));
  }
  const float* qn_ptr = query_norm_dev ? query_norm_dev->data_handle() : nullptr;
  rmm::device_uvector<float> distance_buffer_dev(n_queries * index.n_lists(), stream, search_mr);
  rmm::device_uvector<float> coarse_distances_dev(n_queries_probes, stream, search_mr);
  rmm::device_uvector<uint32_t> coarse_indices_dev(n_queries_probes, stream, search_mr);

  size_t float_query_size;
  if constexpr (std::is_same_v<T, float>) {
    float_query_size = 0;
  } else {
    float_query_size = n_queries * dim;
  }
  rmm::device_uvector<float> converted_queries_dev(float_query_size, stream, search_mr);
  float* converted_queries_ptr = converted_queries_dev.data();

  if constexpr (std::is_same_v<T, float>) {
    converted_queries_ptr = const_cast<float*>(queries);
  } else {
    raft::linalg::unaryOp(
      converted_queries_ptr, queries, n_queries * dim, utils::mapping<float>{}, stream);
  }

  auto distance_buffer_dev_view = raft::make_device_matrix_view<float, int64_t>(
    distance_buffer_dev.data(), n_queries, index.n_lists());

  RAFT_EXPECTS(index.metric() == cuvs::distance::DistanceType::InnerProduct ||
                 index.center_norms().has_value(),
               "Center norms are required for search with L2 or Cosine metric. "
               "Rebuild the index with add_data_on_build=true or call extend() first.");

  float alpha = 1.0f;
  float beta  = 0.0f;
  switch (index.metric()) {
    case cuvs::distance::DistanceType::L2Expanded:
    case cuvs::distance::DistanceType::L2SqrtExpanded: {
      alpha = -2.0f;
      beta  = 1.0f;
      raft::linalg::rowNorm<raft::linalg::L2Norm, true>(query_norm_dev->data_handle(),
                                                        converted_queries_ptr,
                                                        static_cast<int64_t>(dim),
                                                        static_cast<int64_t>(n_queries),
                                                        stream);
      utils::outer_add(query_norm_dev->data_handle(),
                       (int64_t)n_queries,
                       index.center_norms()->data_handle(),
                       (int64_t)index.n_lists(),
                       distance_buffer_dev.data(),
                       stream);
      break;
    }
    case cuvs::distance::DistanceType::CosineExpanded: {
      raft::linalg::rowNorm<raft::linalg::L2Norm, true>(query_norm_dev->data_handle(),
                                                        converted_queries_ptr,
                                                        static_cast<int64_t>(dim),
                                                        static_cast<int64_t>(n_queries),
                                                        stream,
                                                        raft::sqrt_op{});
      alpha = -1.0f;
      beta  = 0.0f;
      break;
    }
    case cuvs::distance::DistanceType::InnerProduct: {
      alpha = 1.0f;
      beta  = 0.0f;
      break;
    }
    default: RAFT_FAIL("Unsupported metric type for IVF-SQ search.");
  }

  raft::linalg::gemm(handle,
                     true,
                     false,
                     index.n_lists(),
                     n_queries,
                     dim,
                     &alpha,
                     index.centers().data_handle(),
                     dim,
                     converted_queries_ptr,
                     dim,
                     &beta,
                     distance_buffer_dev.data(),
                     index.n_lists(),
                     stream);

  if (index.metric() == cuvs::distance::DistanceType::CosineExpanded) {
    auto n_lists_local          = index.n_lists();
    const auto* q_norm_ptr      = query_norm_dev->data_handle();
    const auto* center_norm_ptr = index.center_norms()->data_handle();
    raft::linalg::map_offset(
      handle,
      distance_buffer_dev_view,
      [=] __device__(const int64_t idx, const float dist) {
        const auto query   = idx / n_lists_local;
        const auto cluster = idx % n_lists_local;
        float denom        = q_norm_ptr[query] * center_norm_ptr[cluster];
        return (denom > 0.0f) ? dist / denom : 0.0f;
      },
      raft::make_const_mdspan(distance_buffer_dev_view));
  }

  cuvs::selection::select_k(
    handle,
    raft::make_const_mdspan(distance_buffer_dev_view),
    std::nullopt,
    raft::make_device_matrix_view<float, int64_t>(coarse_distances_dev.data(), n_queries, n_probes),
    raft::make_device_matrix_view<uint32_t, int64_t>(
      coarse_indices_dev.data(), n_queries, n_probes),
    coarse_select_min);

  rmm::device_uvector<uint32_t> num_samples(n_queries, stream, search_mr);
  rmm::device_uvector<uint32_t> chunk_index(n_queries_probes, stream, search_mr);

  ivf::detail::calc_chunk_indices::configure(n_probes, n_queries)(index.list_sizes().data_handle(),
                                                                  coarse_indices_dev.data(),
                                                                  chunk_index.data(),
                                                                  num_samples.data(),
                                                                  stream);

  bool manage_local_topk = is_local_topk_feasible(k);

  // Determine grid_dim_x for the fused path
  uint32_t grid_dim_x = 0;
  if (manage_local_topk) {
    // Query the occupancy to compute optimal grid_dim_x (does not launch)
    ivf_sq_scan(handle,
                index,
                converted_queries_ptr,
                qn_ptr,
                n_queries,
                n_probes,
                k,
                0,
                coarse_indices_dev.data(),
                chunk_index.data(),
                nullptr,
                nullptr,
                sample_filter,
                grid_dim_x,
                stream);
    if (grid_dim_x == 0) {
      manage_local_topk = false;
      RAFT_LOG_WARN(
        "IVF-SQ fused top-k kernel has zero occupancy (dim=%u, k=%u); "
        "falling back to the non-fused scan path.",
        index.dim(),
        k);
    }
  }

  // The scan kernel writes sample-local uint32 indices; postprocess_neighbors
  // lifts them to int64_t database indices via inds_ptrs.
  rmm::device_uvector<uint32_t> neighbors_uint32(std::size_t(n_queries) * k, stream, search_mr);
  uint32_t* neighbors_uint32_ptr = neighbors_uint32.data();

  if (manage_local_topk) {
    // --- Fused top-k path ---
    auto target_size = std::size_t(n_queries) * grid_dim_x * k;
    rmm::device_uvector<float> distances_tmp(0, stream, search_mr);
    rmm::device_uvector<uint32_t> indices_tmp(0, stream, search_mr);

    float* dist_out_ptr   = nullptr;
    uint32_t* idx_out_ptr = nullptr;

    if (grid_dim_x > 1) {
      distances_tmp.resize(target_size, stream);
      indices_tmp.resize(target_size, stream);
      dist_out_ptr = distances_tmp.data();
      idx_out_ptr  = indices_tmp.data();
    } else {
      dist_out_ptr = distances;
      idx_out_ptr  = neighbors_uint32_ptr;
    }

    ivf_sq_scan(handle,
                index,
                converted_queries_ptr,
                qn_ptr,
                n_queries,
                n_probes,
                k,
                0,
                coarse_indices_dev.data(),
                chunk_index.data(),
                dist_out_ptr,
                idx_out_ptr,
                sample_filter,
                grid_dim_x,
                stream);

    // Merge across blocks if needed
    if (grid_dim_x > 1) {
      auto cols = uint32_t(grid_dim_x) * k;
      cuvs::selection::select_k(
        handle,
        raft::make_device_matrix_view<const float, int64_t>(distances_tmp.data(), n_queries, cols),
        raft::make_device_matrix_view<const uint32_t, int64_t>(indices_tmp.data(), n_queries, cols),
        raft::make_device_matrix_view<float, int64_t>(distances, n_queries, k),
        raft::make_device_matrix_view<uint32_t, int64_t>(neighbors_uint32_ptr, n_queries, k),
        /*select_min=*/true);
    }
  } else {
    // --- Fallback: materialize all distances ---
    int64_t ms = std::max<int64_t>(index.accum_sorted_sizes()(n_probes), k);
    RAFT_EXPECTS(ms <= int64_t(std::numeric_limits<uint32_t>::max()),
                 "The maximum sample size is too big.");
    uint32_t max_samples = static_cast<uint32_t>(ms);

    rmm::device_uvector<float> all_distances(
      std::size_t(n_queries) * max_samples, stream, search_mr);
    rmm::device_uvector<uint32_t> all_indices(
      std::size_t(n_queries) * max_samples, stream, search_mr);

    thrust::fill_n(raft::resource::get_thrust_policy(handle),
                   all_distances.data(),
                   std::size_t(n_queries) * max_samples,
                   std::numeric_limits<float>::max());
    thrust::fill_n(raft::resource::get_thrust_policy(handle),
                   all_indices.data(),
                   std::size_t(n_queries) * max_samples,
                   uint32_t(0xFFFFFFFF));

    // grid_dim_x is unused for the non-fused path; set to n_probes so each
    // block in the (n_probes, n_queries) grid processes exactly one probe
    uint32_t gdx = n_probes;
    ivf_sq_scan(handle,
                index,
                converted_queries_ptr,
                qn_ptr,
                n_queries,
                n_probes,
                k,
                max_samples,
                coarse_indices_dev.data(),
                chunk_index.data(),
                all_distances.data(),
                all_indices.data(),
                sample_filter,
                gdx,
                stream);

    auto num_samples_view =
      raft::make_device_vector_view<const uint32_t>(num_samples.data(), n_queries);

    cuvs::selection::select_k(
      handle,
      raft::make_device_matrix_view<const float, int64_t>(
        all_distances.data(), n_queries, max_samples),
      raft::make_device_matrix_view<const uint32_t, int64_t>(
        all_indices.data(), n_queries, max_samples),
      raft::make_device_matrix_view<float, int64_t>(distances, n_queries, k),
      raft::make_device_matrix_view<uint32_t, int64_t>(neighbors_uint32_ptr, n_queries, k),
      /*select_min=*/true,
      false,
      cuvs::selection::SelectAlgo::kAuto,
      num_samples_view);
  }

  // For IP, the kernel returned negated scores so the warpsort could run
  // ascending; postprocess_distances multiplies by -1 to restore the public
  // contract (positive inner products). Cosine distance is already min-close
  // so it must not be flipped (account_for_max_close = false for cosine).
  const bool account_for_max_close = index.metric() != cuvs::distance::DistanceType::CosineExpanded;
  ivf::detail::postprocess_distances(
    handle, distances, distances, index.metric(), n_queries, k, 1.0, account_for_max_close);

  ivf::detail::postprocess_neighbors(neighbors,
                                     neighbors_uint32_ptr,
                                     index.inds_ptrs().data_handle(),
                                     coarse_indices_dev.data(),
                                     chunk_index.data(),
                                     n_queries,
                                     n_probes,
                                     k,
                                     stream);
}

template <typename T,
          typename CodeT,
          typename IvfSampleFilterT = cuvs::neighbors::filtering::none_sample_filter>
inline void search_with_filtering(raft::resources const& handle,
                                  const search_params& params,
                                  const index<CodeT>& index,
                                  const T* queries,
                                  uint32_t n_queries,
                                  uint32_t k,
                                  int64_t* neighbors,
                                  float* distances,
                                  IvfSampleFilterT sample_filter = IvfSampleFilterT())
{
  cuvs::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "ivf_sq::search(k = %u, n_queries = %u, dim = %zu)", k, n_queries, index.dim());

  RAFT_EXPECTS(params.n_probes > 0,
               "n_probes (number of clusters to probe in the search) must be positive.");
  auto n_probes = std::min<uint32_t>(params.n_probes, index.n_lists());
  if (n_probes < params.n_probes) {
    RAFT_LOG_WARN(
      "n_probes (%u) is larger than the number of lists in the index (%u), clamping to %u.",
      params.n_probes,
      index.n_lists(),
      n_probes);
  }

  bool manage_local_topk = is_local_topk_feasible(k);

  // Always compute max_samples: even on the fused path, search_impl may fall
  // back to the materialize-all branch when the scan kernel reports zero
  // occupancy, in which case the workspace must already be sized for the
  // (much larger) materialize-path requirement.
  int64_t ms = std::max<int64_t>(index.accum_sorted_sizes()(n_probes), k);
  RAFT_EXPECTS(ms <= int64_t(std::numeric_limits<uint32_t>::max()),
               "The maximum sample size is too big.");
  uint32_t max_samples = static_cast<uint32_t>(ms);

  constexpr uint64_t kExpectedWsSize = 1024ull * 1024 * 1024;
  uint64_t max_ws_size =
    std::min<uint64_t>(raft::resource::get_workspace_free_bytes(handle), kExpectedWsSize);

  uint64_t converted_query_floats = std::is_same_v<T, float> ? 0 : index.dim();

  // Materialize-path estimate: used directly for the non-fused path, and as
  // a conservative upper bound for the fused path so the fallback branch in
  // search_impl never overshoots the workspace.
  uint64_t ws_materialize = sizeof(float) * (uint64_t(index.n_lists()) + n_probes + 1 +
                                             max_samples + converted_query_floats) +
                            sizeof(uint32_t) * (uint64_t(n_probes) * 2 + 1 + max_samples + k);

  uint64_t ws_per_query;
  if (manage_local_topk) {
    // Fused path: only small per-query buffers for coarse search + chunk indices.
    // The scan output is at most grid_dim_x * k per query (we conservatively
    // assume grid_dim_x <= n_probes for the workspace estimate).
    uint64_t fused_out = uint64_t(n_probes) * k;
    uint64_t ws_fused  = sizeof(float) * (uint64_t(index.n_lists()) + n_probes + 1 + fused_out +
                                         converted_query_floats) +
                        sizeof(uint32_t) * (uint64_t(n_probes) * 2 + 1 + fused_out + k);
    // Take the conservative max so search_impl can safely fall back to the
    // materialize path if the fused-path occupancy probe returns zero.
    ws_per_query = std::max(ws_fused, ws_materialize);
  } else {
    ws_per_query = ws_materialize;
  }

  const uint32_t max_queries =
    std::min<uint32_t>(n_queries, std::max<uint64_t>(1, max_ws_size / ws_per_query));

  for (uint32_t offset_q = 0; offset_q < n_queries; offset_q += max_queries) {
    uint32_t queries_batch = std::min(max_queries, n_queries - offset_q);

    search_impl<T, CodeT, IvfSampleFilterT>(handle,
                                            index,
                                            queries + std::size_t(offset_q) * index.dim(),
                                            queries_batch,
                                            k,
                                            n_probes,
                                            neighbors + std::size_t(offset_q) * k,
                                            distances + std::size_t(offset_q) * k,
                                            raft::resource::get_workspace_resource_ref(handle),
                                            sample_filter);
  }
}

template <typename T, typename CodeT, typename IvfSampleFilterT>
void search_with_filtering(raft::resources const& handle,
                           const search_params& params,
                           const index<CodeT>& index,
                           raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
                           raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
                           raft::device_matrix_view<float, int64_t, raft::row_major> distances,
                           IvfSampleFilterT sample_filter = IvfSampleFilterT())
{
  RAFT_EXPECTS(
    queries.extent(0) == neighbors.extent(0) && queries.extent(0) == distances.extent(0),
    "Number of rows in output neighbors and distances matrices must equal the number of queries.");
  RAFT_EXPECTS(neighbors.extent(1) == distances.extent(1),
               "Number of columns in output neighbors and distances matrices must be equal");
  RAFT_EXPECTS(queries.extent(1) == index.dim(),
               "Number of query dimensions should equal number of dimensions in the index.");

  search_with_filtering(handle,
                        params,
                        index,
                        queries.data_handle(),
                        static_cast<std::uint32_t>(queries.extent(0)),
                        static_cast<std::uint32_t>(neighbors.extent(1)),
                        neighbors.data_handle(),
                        distances.data_handle(),
                        sample_filter);
}

template <typename T, typename CodeT>
void search(raft::resources const& handle,
            const search_params& params,
            const index<CodeT>& idx,
            raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter_ref)
{
  try {
    auto& sample_filter =
      dynamic_cast<const cuvs::neighbors::filtering::none_sample_filter&>(sample_filter_ref);
    return search_with_filtering(handle, params, idx, queries, neighbors, distances, sample_filter);
  } catch (const std::bad_cast&) {
  }

  try {
    auto& sample_filter =
      dynamic_cast<const cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>&>(
        sample_filter_ref);
    return search_with_filtering(handle, params, idx, queries, neighbors, distances, sample_filter);
  } catch (const std::bad_cast&) {
    RAFT_FAIL("Unsupported sample filter type");
  }
}

}  // namespace cuvs::neighbors::ivf_sq::detail
