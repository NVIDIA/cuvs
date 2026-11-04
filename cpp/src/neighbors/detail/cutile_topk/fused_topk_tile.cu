/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fused_topk_merge.cuh"
#include "fused_topk_register_merge.cuh"
#include "fused_topk_subgroup_merge.cuh"
#include "fused_topk_tile.hpp"

#include "fused_topk_pk16_planner.hpp"
#include "fused_topk_pk16_stages_planner.hpp"
#include "fused_topk_pk4_planner.hpp"
#include "fused_topk_pk8_longscan_planner.hpp"
#include "fused_topk_pk8_planner.hpp"
#include "fused_topk_planner.hpp"
#include "fused_topk_stage_planner.hpp"

#include <cuvs/core/export.hpp>
#include <cuvs/detail/jit_lto/cutile_module.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cuda_utils.cuh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace cuvs::neighbors::detail {
namespace {

bool is_16_byte_aligned(const void* ptr)
{
  return ptr == nullptr || reinterpret_cast<std::uintptr_t>(ptr) % 16 == 0;
}

bool byte_ranges_overlap(const void* lhs, size_t lhs_bytes, const void* rhs, size_t rhs_bytes)
{
  if (lhs == nullptr || rhs == nullptr || lhs_bytes == 0 || rhs_bytes == 0) { return false; }
  const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs);
  const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs);
  return lhs_begin <= rhs_begin ? rhs_begin - lhs_begin < lhs_bytes
                                : lhs_begin - rhs_begin < rhs_bytes;
}

template <typename IdxT>
size_t checked_tensor_bytes(IdxT rows, IdxT cols, size_t element_size)
{
  const auto rows_u       = static_cast<uint64_t>(rows);
  const auto cols_u       = static_cast<uint64_t>(cols);
  constexpr auto max_size = std::numeric_limits<size_t>::max();
  if (cols_u != 0 && rows_u > max_size / cols_u) { return max_size; }
  const auto elements = rows_u * cols_u;
  if (element_size != 0 && elements > max_size / element_size) { return max_size; }
  return static_cast<size_t>(elements) * element_size;
}

struct Pk8LargebatchStages {
  std::shared_ptr<rtcx::algorithm_launcher> selection;
  std::shared_ptr<rtcx::algorithm_launcher> refinement;
  cuvs::detail::jit_lto::CutileTileConfig config{};
  explicit operator bool() const { return selection && refinement; }
};

template <typename DataT>
Pk8LargebatchStages resolve_pk8_largebatch(
  int query_rows, int corpus_rows, int dimensions, int splits, cuvs::distance::DistanceType metric)
{
  if (!fused_topk_pk8_largebatch_eligible(query_rows, corpus_rows, dimensions, splits, metric)) {
    return {};
  }
  cuvs::detail::jit_lto::CutileRuntimeCapabilities capabilities{};
  if (!cuvs::detail::jit_lto::query_current_cutile_runtime_capabilities(capabilities) ||
      capabilities.cc_major != 12 || capabilities.cc_minor != 0) {
    return {};
  }
  Pk8LargebatchStages stages;
  if (fused_topk_pk8_largebatch_query_reuse<DataT>(query_rows, dimensions)) {
    using Select = FusedTopkStagePlanner<8, DataT, fused_topk_stage::select_largeq>;
    Select select;
    stages.selection = select.try_get_launcher();
    stages.config    = {Select::tile_m, Select::tile_n, Select::tile_k};
  } else {
    using Select = FusedTopkStagePlanner<8, DataT, fused_topk_stage::select_largecorpus>;
    Select select;
    stages.selection = select.try_get_launcher();
    stages.config    = {Select::tile_m, Select::tile_n, Select::tile_k};
  }
  FusedTopkStagePlanner<8, DataT, fused_topk_stage::refine> refine;
  stages.refinement = refine.try_get_launcher();
  return stages;
}

template <int PhysicalK, typename DataT, typename AbiTag, typename OutputIdxT>
bool launch_fused_topk(OutputIdxT* neighbors,
                       float* distances,
                       const DataT* queries,
                       const DataT* corpus,
                       int query_rows,
                       int corpus_rows,
                       int dimensions,
                       int num_splits,
                       cuvs::distance::DistanceType metric,
                       cudaStream_t stream,
                       int packed_output_width = 0)
{
  static_assert(PhysicalK == 2 || PhysicalK == 4 || PhysicalK == 8 || PhysicalK == 16);
  static_assert(PhysicalK == 8 || std::is_same_v<OutputIdxT, int>);
  if (packed_output_width != 0 &&
      (PhysicalK != 8 || num_splits != 1 || packed_output_width < 5 || packed_output_width > 8)) {
    return false;
  }
  using Planner = std::conditional_t<
    PhysicalK == 2,
    FusedTopkPk2Planner<DataT, AbiTag>,
    std::conditional_t<PhysicalK == 4,
                       FusedTopkPk4Planner<DataT, AbiTag>,
                       std::conditional_t<PhysicalK == 8,
                                          FusedTopkPk8Planner<DataT, AbiTag, OutputIdxT>,
                                          FusedTopkPk16Planner<DataT, AbiTag>>>>;
  std::shared_ptr<rtcx::algorithm_launcher> launcher;
  std::shared_ptr<rtcx::algorithm_launcher> refine_launcher;
  cuvs::detail::jit_lto::CutileTileConfig tile_cfg{};
  if constexpr (PhysicalK == 8 && std::is_same_v<OutputIdxT, int> &&
                std::is_same_v<AbiTag, cuvs::distance::detail::cutile_abi_strict>) {
    if (auto stages =
          resolve_pk8_largebatch<DataT>(query_rows, corpus_rows, dimensions, num_splits, metric)) {
      launcher        = std::move(stages.selection);
      refine_launcher = std::move(stages.refinement);
      tile_cfg        = stages.config;
    }
  }
  if constexpr (PhysicalK <= 8 && std::is_same_v<OutputIdxT, int> &&
                std::is_same_v<AbiTag, cuvs::distance::detail::cutile_abi_strict>) {
    // Preserve corpus splits and other architecture/ABI paths. Half PK2 uses
    // one broad low-D fused region, never a per-shape benchmark lookup table.
    const bool staged_eligible = [&] {
      if constexpr (PhysicalK == 8) {
        return fused_topk_pk8_staged_refinement_eligible(dimensions, num_splits, corpus_rows);
      } else {
        return fused_topk_separate_refinement_eligible<PhysicalK, DataT>(dimensions, num_splits);
      }
    }();
    if (!launcher && staged_eligible) {
      cuvs::detail::jit_lto::CutileRuntimeCapabilities capabilities{};
      if (cuvs::detail::jit_lto::query_current_cutile_runtime_capabilities(capabilities) &&
          capabilities.cc_major == 12 && capabilities.cc_minor == 0) {
        using Select = FusedTopkStagePlanner<PhysicalK, DataT, fused_topk_stage::select>;
        using Refine = FusedTopkStagePlanner<PhysicalK, DataT, fused_topk_stage::refine>;
        std::shared_ptr<rtcx::algorithm_launcher> selection;
        if constexpr (PhysicalK == 4 && std::is_same_v<DataT, half>) {
          if (num_splits == 1) {
            FusedTopkStagePlanner<4, half, fused_topk_stage::witness> witness;
            selection = witness.try_get_launcher();
          } else {
            Select select;
            selection = select.try_get_launcher();
          }
        } else if constexpr (PhysicalK == 8) {
          // L2/sqrt-L2 share candidate ranking; all other metrics retain the
          // generic image. Resolve the optional specialization before enqueue.
          if (metric == cuvs::distance::DistanceType::L2Expanded ||
              metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
            FusedTopkStagePlanner<8, DataT, fused_topk_stage::select_l2> l2_select;
            selection = l2_select.try_get_launcher();
          }
          if (!selection) {
            Select select;
            selection = select.try_get_launcher();
          }
        } else {
          Select select;
          selection = select.try_get_launcher();
        }
        Refine refine;
        auto refinement = refine.try_get_launcher();
        // Resolve both handles before any output writes. An unavailable stage
        // falls back to the original fused kernel, never to a partial result.
        if (selection && refinement) {
          launcher        = std::move(selection);
          refine_launcher = std::move(refinement);
          tile_cfg        = {Select::tile_m, Select::tile_n, Select::tile_k};
        }
      }
    }
  }
  if (!launcher) {
    Planner planner;
    planner.add_entrypoint();
    launcher = planner.try_get_launcher();
    if (!launcher) { return false; }
    tile_cfg = planner.tile_config();
  }
  int refine_tile_m = 0;
  std::shared_ptr<rtcx::algorithm_launcher> refiner;
  if constexpr (PhysicalK == 16) {
    cuvs::detail::jit_lto::CutileRuntimeCapabilities capabilities{};
    if (cuvs::detail::jit_lto::query_current_cutile_runtime_capabilities(capabilities) &&
        capabilities.cc_major == 12 && capabilities.cc_minor == 0) {
      FusedTopkPk16StagePlanner<DataT, AbiTag, 0> select_planner;
      FusedTopkPk16StagePlanner<DataT, AbiTag, 1> refine_planner;
      select_planner.add_entrypoint();
      refine_planner.add_entrypoint();
      auto selector = select_planner.try_get_launcher();
      refiner       = refine_planner.try_get_launcher();
      // Resolve BOTH stages before any enqueue: fallback must never run after
      // a partial selection launch. Other architectures keep the fused path.
      if (selector == nullptr || refiner == nullptr) { return false; }
      tile_cfg              = select_planner.tile_config();
      const auto refine_cfg = refine_planner.tile_config();
      // Both stages share corpus partitions, but their feature-reduction tiles
      // are independent: refinement reads the original vectors, not GEMM tiles.
      if (tile_cfg.tile_m <= 0 || refine_cfg.tile_m <= 0 || tile_cfg.tile_n <= 0 ||
          tile_cfg.tile_n != refine_cfg.tile_n || tile_cfg.tile_k <= 0 || refine_cfg.tile_k <= 0) {
        return false;
      }
      if constexpr ((std::is_same_v<DataT, half> || std::is_same_v<DataT, float>) &&
                    std::is_same_v<AbiTag, cuvs::distance::detail::cutile_abi_strict>) {
        if (fused_topk_pk16_use_witness<DataT>(corpus_rows, num_splits, tile_cfg.tile_n, true)) {
          FusedTopkPk16StagePlanner<DataT, AbiTag, 2> witness_planner;
          witness_planner.add_entrypoint();
          auto witness = witness_planner.try_get_launcher();
          if (witness != nullptr) {
            const auto witness_cfg = witness_planner.tile_config();
            if (witness_cfg.tile_m != tile_cfg.tile_m || witness_cfg.tile_n != tile_cfg.tile_n ||
                witness_cfg.tile_k != tile_cfg.tile_k) {
              return false;
            }
            selector = std::move(witness);
          }
          // Unavailable optional image retains the ordinary selector. All image
          // resolution precedes enqueue; fallback cannot duplicate partial work.
        }
      }
      refine_tile_m = refine_cfg.tile_m;
      launcher      = std::move(selector);
    }
  }
  const int num_corpus_tiles = fused_topk_pk2_ceil_div(corpus_rows, tile_cfg.tile_n);
  if (num_splits <= 0 || num_splits > num_corpus_tiles) { return false; }
  if constexpr (PhysicalK == 8) {
    if (!refine_launcher &&
        fused_topk_pk8_use_longscan(
          corpus_rows, num_splits, tile_cfg.tile_m, tile_cfg.tile_n, tile_cfg.tile_k)) {
      FusedTopkPk8LongscanPlanner<DataT, AbiTag, OutputIdxT> guarded;
      guarded.add_entrypoint();
      // Resolve before any enqueue. Unsupported architectures keep the original image.
      if (auto candidate = guarded.try_get_launcher(); candidate != nullptr) {
        const auto guard_cfg = guarded.tile_config();
        if (guard_cfg.tile_m == tile_cfg.tile_m && guard_cfg.tile_n == tile_cfg.tile_n &&
            guard_cfg.tile_k == tile_cfg.tile_k) {
          launcher = std::move(candidate);
        }
      }
    }
  }

  int metric_code;
  switch (metric) {
    case cuvs::distance::DistanceType::InnerProduct:
    case cuvs::distance::DistanceType::L2Expanded:
    case cuvs::distance::DistanceType::L2SqrtExpanded:
    case cuvs::distance::DistanceType::CosineExpanded:
      metric_code = static_cast<int>(metric);
      break;
    default: return false;
  }

  void* query_ptr    = const_cast<DataT*>(queries);
  void* corpus_ptr   = const_cast<DataT*>(corpus);
  void* index_ptr    = neighbors;
  void* distance_ptr = distances;

  int shape_queries[2]    = {query_rows, dimensions};
  int stride_queries[2]   = {dimensions, 1};
  int shape_corpus[2]     = {corpus_rows, dimensions};
  int stride_corpus[2]    = {dimensions, 1};
  const int output_width  = packed_output_width != 0 ? packed_output_width : num_splits * PhysicalK;
  int shape_indices[2]    = {query_rows, output_width};
  int stride_indices[2]   = {output_width, 1};
  int shape_distances[2]  = {query_rows, output_width};
  int stride_distances[2] = {output_width, 1};

  dim3 block(1, 1, 1);

  using fused_topk_cutile_kernel_t = void(void*,
                                          int,
                                          int,
                                          int,
                                          int,
                                          void*,
                                          int,
                                          int,
                                          int,
                                          int,
                                          void*,
                                          int,
                                          int,
                                          int,
                                          int,
                                          void*,
                                          int,
                                          int,
                                          int,
                                          int,
                                          int,
                                          int,
                                          int,
                                          int,
                                          int);
  const auto dispatch = [&](const std::shared_ptr<rtcx::algorithm_launcher>& stage, int tile_m) {
    dim3 grid((static_cast<uint64_t>(query_rows) + tile_m - 1) / tile_m, num_splits, 1);
    stage->template dispatch<fused_topk_cutile_kernel_t>(stream,
                                                         grid,
                                                         block,
                                                         0,
                                                         query_ptr,
                                                         shape_queries[0],
                                                         shape_queries[1],
                                                         stride_queries[0],
                                                         stride_queries[1],
                                                         corpus_ptr,
                                                         shape_corpus[0],
                                                         shape_corpus[1],
                                                         stride_corpus[0],
                                                         stride_corpus[1],
                                                         index_ptr,
                                                         shape_indices[0],
                                                         shape_indices[1],
                                                         stride_indices[0],
                                                         stride_indices[1],
                                                         distance_ptr,
                                                         shape_distances[0],
                                                         shape_distances[1],
                                                         stride_distances[0],
                                                         stride_distances[1],
                                                         query_rows,
                                                         corpus_rows,
                                                         dimensions,
                                                         metric_code,
                                                         num_splits);
    RAFT_CUDA_TRY(cudaGetLastError());
  };
  dispatch(launcher, tile_cfg.tile_m);
  if (refiner != nullptr) { dispatch(refiner, refine_tile_m); }
  if constexpr (PhysicalK <= 8) {
    if (refine_launcher) {
      // Each stage has its own query grid. The full physical-list stride
      // remains num_splits*PK, including PK8 TM32 selection/TM4 refinement.
      dispatch(refine_launcher,
               FusedTopkStagePlanner<PhysicalK, DataT, fused_topk_stage::refine>::tile_m);
    }
  }
  return true;
}

struct FusedTopkWorkspaceView {
  int* indices;
  float* distances;
};

template <int PhysicalK>
FusedTopkWorkspaceView fused_topk_workspace_view(void* workspace, int query_rows, int num_splits)
{
  auto* bytes       = static_cast<std::byte*>(workspace);
  const auto count  = static_cast<size_t>(query_rows) * static_cast<size_t>(num_splits) * PhysicalK;
  const auto offset = fused_topk_align_workspace(count * sizeof(int));
  return {reinterpret_cast<int*>(bytes), reinterpret_cast<float*>(bytes + offset)};
}

template <int PhysicalK, typename IdxT>
__global__ void merge_fused_topk_splits(IdxT* neighbors,
                                        float* distances,
                                        const int* partial_indices,
                                        const float* partial_distances,
                                        int query_rows,
                                        int logical_k,
                                        int num_splits,
                                        int metric_code)
{
  constexpr float inf        = std::numeric_limits<float>::infinity();
  constexpr int invalid_high = std::numeric_limits<int>::max();
  const bool inner_product =
    metric_code == static_cast<int>(cuvs::distance::DistanceType::InnerProduct);
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; row < query_rows;
       row += stride) {
    float best_scores[PhysicalK];
    int best_indices[PhysicalK];
#pragma unroll
    for (int slot = 0; slot < PhysicalK; ++slot) {
      best_scores[slot]  = inf;
      best_indices[slot] = invalid_high;
    }
    const int width = num_splits * PhysicalK;
    for (int column = 0; column < width; ++column) {
      const int64_t offset  = row * width + column;
      int candidate_index   = partial_indices[offset];
      const float value     = partial_distances[offset];
      float candidate_score = inner_product ? -value : value;
      if (candidate_index < 0) {
        candidate_index = invalid_high;
        candidate_score = inf;
      }

#pragma unroll
      for (int slot = 0; slot < PhysicalK; ++slot) {
        const bool take =
          candidate_score < best_scores[slot] ||
          (candidate_score == best_scores[slot] && candidate_index < best_indices[slot]);
        if (take) {
          const float displaced_score = best_scores[slot];
          const int displaced_index   = best_indices[slot];
          best_scores[slot]           = candidate_score;
          best_indices[slot]          = candidate_index;
          candidate_score             = displaced_score;
          candidate_index             = displaced_index;
        }
      }
    }

    const int64_t output_offset = row * logical_k;
    for (int column = 0; column < logical_k; ++column) {
      const int candidate_index         = best_indices[column];
      neighbors[output_offset + column] = candidate_index == invalid_high
                                            ? static_cast<IdxT>(-1)
                                            : static_cast<IdxT>(candidate_index);
      distances[output_offset + column] =
        candidate_index == invalid_high
          ? inf
          : (inner_product ? -best_scores[column] : best_scores[column]);
    }
  }
}

template <int PhysicalK, typename IdxT, typename DataT, typename AbiTag>
bool launch_and_convert(IdxT* neighbors,
                        float* distances,
                        const DataT* queries,
                        const DataT* corpus,
                        int query_rows,
                        int corpus_rows,
                        int dimensions,
                        int logical_k,
                        int num_splits,
                        cuvs::distance::DistanceType metric,
                        cudaStream_t stream,
                        void* workspace)
{
  if (num_splits == 1 && logical_k == PhysicalK) {
    if constexpr (std::is_same_v<IdxT, int>) {
      return launch_fused_topk<PhysicalK, DataT, AbiTag>(neighbors,
                                                         distances,
                                                         queries,
                                                         corpus,
                                                         query_rows,
                                                         corpus_rows,
                                                         dimensions,
                                                         num_splits,
                                                         metric,
                                                         stream);
    } else {
      auto* temporary_indices = static_cast<int*>(workspace);
      if (!launch_fused_topk<PhysicalK, DataT, AbiTag>(temporary_indices,
                                                       distances,
                                                       queries,
                                                       corpus,
                                                       query_rows,
                                                       corpus_rows,
                                                       dimensions,
                                                       num_splits,
                                                       metric,
                                                       stream)) {
        return false;
      }
      raft::linalg::unaryOp(neighbors,
                            temporary_indices,
                            static_cast<size_t>(query_rows) * PhysicalK,
                            raft::cast_op<int64_t>{},
                            stream);
      return true;
    }
  }

  const auto partial = fused_topk_workspace_view<PhysicalK>(workspace, query_rows, num_splits);
  if (!launch_fused_topk<PhysicalK, DataT, AbiTag>(partial.indices,
                                                   partial.distances,
                                                   queries,
                                                   corpus,
                                                   query_rows,
                                                   corpus_rows,
                                                   dimensions,
                                                   num_splits,
                                                   metric,
                                                   stream)) {
    return false;
  }
  if constexpr (PhysicalK == 2 || PhysicalK == 4) {
    // Capacity-only merge of sorted lists; no benchmark shape allowlist.
    if (num_splits >= 2 && num_splits <= 128) {
      const bool inner = metric == cuvs::distance::DistanceType::InnerProduct;
      if (num_splits == 2) {
        const int blocks = fused_topk_merge_block_count(query_rows);
        merge_fused_topk_register_heads<PhysicalK, 2><<<blocks, 256, 0, stream>>>(neighbors,
                                                                                  distances,
                                                                                  partial.indices,
                                                                                  partial.distances,
                                                                                  query_rows,
                                                                                  logical_k,
                                                                                  num_splits,
                                                                                  inner);
      } else if (num_splits <= 4) {
        const int blocks = fused_topk_merge_block_count(query_rows);
        merge_fused_topk_register_heads<PhysicalK, 4><<<blocks, 256, 0, stream>>>(neighbors,
                                                                                  distances,
                                                                                  partial.indices,
                                                                                  partial.distances,
                                                                                  query_rows,
                                                                                  logical_k,
                                                                                  num_splits,
                                                                                  inner);
      } else if (num_splits <= 8) {
        const int blocks =
          static_cast<int>(std::min<int64_t>((int64_t(query_rows) + 31) / 32, 65535));
        merge_fused_topk_subgroups<PhysicalK, 8><<<blocks, 256, 0, stream>>>(neighbors,
                                                                             distances,
                                                                             partial.indices,
                                                                             partial.distances,
                                                                             query_rows,
                                                                             logical_k,
                                                                             num_splits,
                                                                             inner);
      } else if (num_splits <= 16) {
        const int blocks =
          static_cast<int>(std::min<int64_t>((int64_t(query_rows) + 15) / 16, 65535));
        merge_fused_topk_subgroups<PhysicalK, 16><<<blocks, 256, 0, stream>>>(neighbors,
                                                                              distances,
                                                                              partial.indices,
                                                                              partial.distances,
                                                                              query_rows,
                                                                              logical_k,
                                                                              num_splits,
                                                                              inner);
      } else {
        const int blocks = std::min(query_rows, 65535);
        merge_fused_topk_sorted_splits<PhysicalK><<<blocks, 32, 0, stream>>>(neighbors,
                                                                             distances,
                                                                             partial.indices,
                                                                             partial.distances,
                                                                             query_rows,
                                                                             logical_k,
                                                                             num_splits,
                                                                             inner);
      }
      RAFT_CUDA_TRY(cudaGetLastError());
      return true;
    }
  }
  if constexpr (PhysicalK == 8) {
    // The producer's refined split lists are sorted. Keep the established
    // single-split path and the serial fallback beyond the warp's capacity.
    if (num_splits >= 2 && num_splits <= 128) {
      const int blocks = std::min(query_rows, 65535);
      merge_fused_topk_sorted_splits<8>
        <<<blocks, 32, 0, stream>>>(neighbors,
                                    distances,
                                    partial.indices,
                                    partial.distances,
                                    query_rows,
                                    logical_k,
                                    num_splits,
                                    metric == cuvs::distance::DistanceType::InnerProduct);
      RAFT_CUDA_TRY(cudaGetLastError());
      return true;
    }
  }
  if constexpr (PhysicalK == 16) {
    // Capacity-based sorted-list merge, not a benchmark shape allowlist.
    // Single-split and more than 128 splits retain the legacy path.
    if (num_splits >= 2 && num_splits <= 128) {
      const int blocks = std::min(query_rows, 65535);
      merge_fused_topk_sorted_splits<PhysicalK>
        <<<blocks, 32, 0, stream>>>(neighbors,
                                    distances,
                                    partial.indices,
                                    partial.distances,
                                    query_rows,
                                    logical_k,
                                    num_splits,
                                    metric == cuvs::distance::DistanceType::InnerProduct);
      RAFT_CUDA_TRY(cudaGetLastError());
      return true;
    }
  }
  constexpr int threads = 256;
  const int blocks      = fused_topk_merge_block_count(query_rows);
  merge_fused_topk_splits<PhysicalK><<<blocks, threads, 0, stream>>>(neighbors,
                                                                     distances,
                                                                     partial.indices,
                                                                     partial.distances,
                                                                     query_rows,
                                                                     logical_k,
                                                                     num_splits,
                                                                     static_cast<int>(metric));
  RAFT_CUDA_TRY(cudaGetLastError());
  return true;
}

template <int PhysicalK, typename IdxT, typename DataT>
bool valid_fused_topk_buffers(IdxT* neighbors,
                              float* distances,
                              const DataT* queries,
                              const DataT* corpus,
                              IdxT query_rows,
                              IdxT corpus_rows,
                              IdxT dimensions,
                              int logical_k,
                              int num_splits,
                              void* workspace)
{
  if (neighbors == nullptr || distances == nullptr || queries == nullptr || corpus == nullptr) {
    return false;
  }
  if (!is_16_byte_aligned(neighbors) || !is_16_byte_aligned(distances) ||
      !is_16_byte_aligned(queries) || !is_16_byte_aligned(corpus)) {
    return false;
  }
  const auto workspace_bytes =
    fused_topk_workspace_bytes<PhysicalK>(query_rows, num_splits, logical_k);
  if (workspace_bytes != 0 && (workspace == nullptr || !is_16_byte_aligned(workspace))) {
    return false;
  }

  const auto query_bytes  = checked_tensor_bytes(query_rows, dimensions, sizeof(DataT));
  const auto corpus_bytes = checked_tensor_bytes(corpus_rows, dimensions, sizeof(DataT));
  const auto index_bytes =
    checked_tensor_bytes(query_rows, static_cast<IdxT>(logical_k), sizeof(IdxT));
  const auto distance_bytes =
    checked_tensor_bytes(query_rows, static_cast<IdxT>(logical_k), sizeof(float));
  if (byte_ranges_overlap(neighbors, index_bytes, queries, query_bytes) ||
      byte_ranges_overlap(neighbors, index_bytes, corpus, corpus_bytes) ||
      byte_ranges_overlap(distances, distance_bytes, queries, query_bytes) ||
      byte_ranges_overlap(distances, distance_bytes, corpus, corpus_bytes) ||
      byte_ranges_overlap(neighbors, index_bytes, distances, distance_bytes)) {
    return false;
  }
  if (workspace_bytes != 0 &&
      (byte_ranges_overlap(workspace, workspace_bytes, queries, query_bytes) ||
       byte_ranges_overlap(workspace, workspace_bytes, corpus, corpus_bytes) ||
       byte_ranges_overlap(workspace, workspace_bytes, neighbors, index_bytes) ||
       byte_ranges_overlap(workspace, workspace_bytes, distances, distance_bytes))) {
    return false;
  }
  return true;
}

}  // namespace

bool is_fused_topk_pk2_auto_dispatch_available()
{
  if (!cuvs::detail::jit_lto::cutile_launch_available_on_current_device()) { return false; }
  cuvs::detail::jit_lto::CutileRuntimeCapabilities capabilities{};
  return cuvs::detail::jit_lto::query_current_cutile_runtime_capabilities(capabilities) &&
         capabilities.cc_major == 12 && capabilities.cc_minor == 0;
}

bool is_fused_topk_pk4_auto_dispatch_available()
{
  return is_fused_topk_pk2_auto_dispatch_available();
}

bool is_fused_topk_pk8_auto_dispatch_available()
{
  return is_fused_topk_pk2_auto_dispatch_available();
}

bool is_fused_topk_pk16_auto_dispatch_available()
{
  return is_fused_topk_pk2_auto_dispatch_available();
}

template <typename IdxT, typename DataT>
bool try_fused_topk_pk2_tile(IdxT* neighbors,
                             float* distances,
                             const DataT* queries,
                             const DataT* corpus,
                             IdxT query_rows,
                             IdxT corpus_rows,
                             IdxT dimensions,
                             int num_splits,
                             cuvs::distance::DistanceType metric,
                             cudaStream_t stream,
                             void* workspace)
{
  static_assert(std::is_same_v<IdxT, int> || std::is_same_v<IdxT, int64_t>);
  static_assert(std::is_same_v<DataT, float> || std::is_same_v<DataT, half>);
  if constexpr (std::is_same_v<DataT, half>) {
    if (!is_fused_topk_pk2_auto_dispatch_available()) { return false; }
  }
  if (!cuvs::detail::jit_lto::cutile_launch_available_on_current_device()) { return false; }
  if (!is_fused_topk_metric(metric)) { return false; }
  constexpr int64_t max_i32 = std::numeric_limits<int>::max();
  if (query_rows <= 0 || corpus_rows <= 0 || dimensions <= 0 || num_splits <= 0 ||
      num_splits > 65535 || query_rows > max_i32 || corpus_rows > max_i32 || dimensions > max_i32) {
    return false;
  }
  if (!valid_fused_topk_buffers<2>(neighbors,
                                   distances,
                                   queries,
                                   corpus,
                                   query_rows,
                                   corpus_rows,
                                   dimensions,
                                   2,
                                   num_splits,
                                   workspace)) {
    return false;
  }

  const int m       = static_cast<int>(query_rows);
  const int n       = static_cast<int>(corpus_rows);
  const int d       = static_cast<int>(dimensions);
  const bool strict = d % (16 / sizeof(DataT)) == 0;
  if (strict) {
    return launch_and_convert<2, IdxT, DataT, cuvs::distance::detail::cutile_abi_strict>(
      neighbors, distances, queries, corpus, m, n, d, 2, num_splits, metric, stream, workspace);
  }
  return launch_and_convert<2, IdxT, DataT, cuvs::distance::detail::cutile_abi_relaxed>(
    neighbors, distances, queries, corpus, m, n, d, 2, num_splits, metric, stream, workspace);
}

template <typename IdxT, typename DataT>
bool try_fused_topk_pk4_tile(IdxT* neighbors,
                             float* distances,
                             const DataT* queries,
                             const DataT* corpus,
                             IdxT query_rows,
                             IdxT corpus_rows,
                             IdxT dimensions,
                             IdxT logical_k,
                             int num_splits,
                             cuvs::distance::DistanceType metric,
                             cudaStream_t stream,
                             void* workspace)
{
  static_assert(std::is_same_v<IdxT, int> || std::is_same_v<IdxT, int64_t>);
  static_assert(std::is_same_v<DataT, float> || std::is_same_v<DataT, half>);
  if constexpr (std::is_same_v<DataT, half>) {
    if (!is_fused_topk_pk2_auto_dispatch_available()) { return false; }
  }
  if (!cuvs::detail::jit_lto::cutile_launch_available_on_current_device()) { return false; }
  if (!is_fused_topk_metric(metric) || (logical_k != 3 && logical_k != 4)) { return false; }
  constexpr int64_t max_i32 = std::numeric_limits<int>::max();
  if (query_rows <= 0 || corpus_rows <= 0 || dimensions <= 0 || num_splits <= 0 ||
      num_splits > 65535 || query_rows > max_i32 || corpus_rows > max_i32 || dimensions > max_i32) {
    return false;
  }
  const int k = static_cast<int>(logical_k);
  if (!valid_fused_topk_buffers<4>(neighbors,
                                   distances,
                                   queries,
                                   corpus,
                                   query_rows,
                                   corpus_rows,
                                   dimensions,
                                   k,
                                   num_splits,
                                   workspace)) {
    return false;
  }

  const int m       = static_cast<int>(query_rows);
  const int n       = static_cast<int>(corpus_rows);
  const int d       = static_cast<int>(dimensions);
  const bool strict = d % (16 / sizeof(DataT)) == 0;
  if (strict) {
    return launch_and_convert<4, IdxT, DataT, cuvs::distance::detail::cutile_abi_strict>(
      neighbors, distances, queries, corpus, m, n, d, k, num_splits, metric, stream, workspace);
  }
  return launch_and_convert<4, IdxT, DataT, cuvs::distance::detail::cutile_abi_relaxed>(
    neighbors, distances, queries, corpus, m, n, d, k, num_splits, metric, stream, workspace);
}

template <typename IdxT, typename DataT>
bool try_fused_topk_pk8_tile(IdxT* neighbors,
                             float* distances,
                             const DataT* queries,
                             const DataT* corpus,
                             IdxT query_rows,
                             IdxT corpus_rows,
                             IdxT dimensions,
                             IdxT logical_k,
                             int num_splits,
                             cuvs::distance::DistanceType metric,
                             cudaStream_t stream,
                             void* workspace)
{
  static_assert(std::is_same_v<IdxT, int> || std::is_same_v<IdxT, int64_t>);
  static_assert(std::is_same_v<DataT, float> || std::is_same_v<DataT, half>);
  if constexpr (std::is_same_v<DataT, half>) {
    if (!is_fused_topk_pk2_auto_dispatch_available()) { return false; }
  }
  if (!cuvs::detail::jit_lto::cutile_launch_available_on_current_device()) { return false; }
  if (!is_fused_topk_metric(metric) || logical_k < 5 || logical_k > 8) { return false; }
  constexpr int64_t max_i32 = std::numeric_limits<int>::max();
  if (query_rows <= 0 || corpus_rows <= 0 || dimensions <= 0 || num_splits <= 0 ||
      num_splits > 65535 || query_rows > max_i32 || corpus_rows > max_i32 || dimensions > max_i32) {
    return false;
  }
  const int k = static_cast<int>(logical_k);
  if (!valid_fused_topk_buffers<8>(neighbors,
                                   distances,
                                   queries,
                                   corpus,
                                   query_rows,
                                   corpus_rows,
                                   dimensions,
                                   k,
                                   num_splits,
                                   workspace)) {
    return false;
  }

  const int m       = static_cast<int>(query_rows);
  const int n       = static_cast<int>(corpus_rows);
  const int d       = static_cast<int>(dimensions);
  const bool strict = d % (16 / sizeof(DataT)) == 0;
  if (strict) {
    return launch_and_convert<8, IdxT, DataT, cuvs::distance::detail::cutile_abi_strict>(
      neighbors, distances, queries, corpus, m, n, d, k, num_splits, metric, stream, workspace);
  }
  return launch_and_convert<8, IdxT, DataT, cuvs::distance::detail::cutile_abi_relaxed>(
    neighbors, distances, queries, corpus, m, n, d, k, num_splits, metric, stream, workspace);
}

template <typename IdxT, typename DataT>
bool try_fused_topk_pk16_tile(IdxT* neighbors,
                              float* distances,
                              const DataT* queries,
                              const DataT* corpus,
                              IdxT query_rows,
                              IdxT corpus_rows,
                              IdxT dimensions,
                              IdxT logical_k,
                              int num_splits,
                              cuvs::distance::DistanceType metric,
                              cudaStream_t stream,
                              void* workspace)
{
  static_assert(std::is_same_v<IdxT, int> || std::is_same_v<IdxT, int64_t>);
  static_assert(std::is_same_v<DataT, float> || std::is_same_v<DataT, half>);
  if constexpr (std::is_same_v<DataT, half>) {
    if (!is_fused_topk_pk2_auto_dispatch_available()) { return false; }
  }
  if (!cuvs::detail::jit_lto::cutile_launch_available_on_current_device()) { return false; }
  if (!is_fused_topk_metric(metric) || logical_k < 9 || logical_k > 16) { return false; }
  constexpr int64_t max_i32 = std::numeric_limits<int>::max();
  if (query_rows <= 0 || corpus_rows <= 0 || dimensions <= 0 || num_splits <= 0 ||
      num_splits > 65535 || query_rows > max_i32 || corpus_rows > max_i32 || dimensions > max_i32) {
    return false;
  }
  const int k = static_cast<int>(logical_k);
  if (!valid_fused_topk_buffers<16>(neighbors,
                                    distances,
                                    queries,
                                    corpus,
                                    query_rows,
                                    corpus_rows,
                                    dimensions,
                                    k,
                                    num_splits,
                                    workspace)) {
    return false;
  }

  const int m       = static_cast<int>(query_rows);
  const int n       = static_cast<int>(corpus_rows);
  const int d       = static_cast<int>(dimensions);
  const bool strict = d % (16 / sizeof(DataT)) == 0;
  if (strict) {
    return launch_and_convert<16, IdxT, DataT, cuvs::distance::detail::cutile_abi_strict>(
      neighbors, distances, queries, corpus, m, n, d, k, num_splits, metric, stream, workspace);
  }
  return launch_and_convert<16, IdxT, DataT, cuvs::distance::detail::cutile_abi_relaxed>(
    neighbors, distances, queries, corpus, m, n, d, k, num_splits, metric, stream, workspace);
}

#define CUVS_INST_TRY_FUSED_TOPK_PK2(IdxT, DataT)                                              \
  template CUVS_EXPORT bool try_fused_topk_pk2_tile<IdxT, DataT>(IdxT*,                        \
                                                                 float*,                       \
                                                                 const DataT*,                 \
                                                                 const DataT*,                 \
                                                                 IdxT,                         \
                                                                 IdxT,                         \
                                                                 IdxT,                         \
                                                                 int,                          \
                                                                 cuvs::distance::DistanceType, \
                                                                 cudaStream_t,                 \
                                                                 void*)

CUVS_INST_TRY_FUSED_TOPK_PK2(int, float);
CUVS_INST_TRY_FUSED_TOPK_PK2(int64_t, float);
CUVS_INST_TRY_FUSED_TOPK_PK2(int, half);
CUVS_INST_TRY_FUSED_TOPK_PK2(int64_t, half);

#define CUVS_INST_TRY_FUSED_TOPK_PK4(IdxT, DataT)                                              \
  template CUVS_EXPORT bool try_fused_topk_pk4_tile<IdxT, DataT>(IdxT*,                        \
                                                                 float*,                       \
                                                                 const DataT*,                 \
                                                                 const DataT*,                 \
                                                                 IdxT,                         \
                                                                 IdxT,                         \
                                                                 IdxT,                         \
                                                                 IdxT,                         \
                                                                 int,                          \
                                                                 cuvs::distance::DistanceType, \
                                                                 cudaStream_t,                 \
                                                                 void*)

CUVS_INST_TRY_FUSED_TOPK_PK4(int, float);
CUVS_INST_TRY_FUSED_TOPK_PK4(int64_t, float);
CUVS_INST_TRY_FUSED_TOPK_PK4(int, half);
CUVS_INST_TRY_FUSED_TOPK_PK4(int64_t, half);

#define CUVS_INST_TRY_FUSED_TOPK_PK8(IdxT, DataT)                                              \
  template CUVS_EXPORT bool try_fused_topk_pk8_tile<IdxT, DataT>(IdxT*,                        \
                                                                 float*,                       \
                                                                 const DataT*,                 \
                                                                 const DataT*,                 \
                                                                 IdxT,                         \
                                                                 IdxT,                         \
                                                                 IdxT,                         \
                                                                 IdxT,                         \
                                                                 int,                          \
                                                                 cuvs::distance::DistanceType, \
                                                                 cudaStream_t,                 \
                                                                 void*)

CUVS_INST_TRY_FUSED_TOPK_PK8(int, float);
CUVS_INST_TRY_FUSED_TOPK_PK8(int64_t, float);
CUVS_INST_TRY_FUSED_TOPK_PK8(int, half);
CUVS_INST_TRY_FUSED_TOPK_PK8(int64_t, half);

#define CUVS_INST_TRY_FUSED_TOPK_PK16(IdxT, DataT)                                              \
  template CUVS_EXPORT bool try_fused_topk_pk16_tile<IdxT, DataT>(IdxT*,                        \
                                                                  float*,                       \
                                                                  const DataT*,                 \
                                                                  const DataT*,                 \
                                                                  IdxT,                         \
                                                                  IdxT,                         \
                                                                  IdxT,                         \
                                                                  IdxT,                         \
                                                                  int,                          \
                                                                  cuvs::distance::DistanceType, \
                                                                  cudaStream_t,                 \
                                                                  void*)

CUVS_INST_TRY_FUSED_TOPK_PK16(int, float);
CUVS_INST_TRY_FUSED_TOPK_PK16(int64_t, float);
CUVS_INST_TRY_FUSED_TOPK_PK16(int, half);
CUVS_INST_TRY_FUSED_TOPK_PK16(int64_t, half);

template <typename IdxT, typename DataT>
bool try_fused_topk_pk8_single_tile(IdxT* neighbors,
                                    float* distances,
                                    const DataT* queries,
                                    const DataT* corpus,
                                    IdxT query_rows,
                                    IdxT corpus_rows,
                                    IdxT dimensions,
                                    IdxT logical_k,
                                    cuvs::distance::DistanceType metric,
                                    cudaStream_t stream)
{
  static_assert(std::is_same_v<IdxT, int> || std::is_same_v<IdxT, int64_t>);
  static_assert(std::is_same_v<DataT, float> || std::is_same_v<DataT, half>);
  if (!is_fused_topk_pk8_auto_dispatch_available() ||
      !cuvs::detail::jit_lto::cutile_launch_available_on_current_device() ||
      !is_fused_topk_pk8_candidate<DataT>(
        query_rows, corpus_rows, dimensions, logical_k, metric, false) ||
      !fused_topk_public_buffers_supported(
        neighbors, distances, queries, corpus, query_rows, corpus_rows, dimensions, logical_k)) {
    return false;
  }
  // Every physical candidate is refined and sorted before the descriptor's
  // logical width bounds-mask discards the surplus. No scratch is required.
  const bool strict = dimensions % (16 / sizeof(DataT)) == 0;
  // The staged route needs full physical-width scratch before logical-prefix
  // compaction. Decline only when BOTH optional images resolve, before writes;
  // unavailable images retain this original no-workspace fused implementation.
  if (strict && resolve_pk8_largebatch<DataT>(static_cast<int>(query_rows),
                                              static_cast<int>(corpus_rows),
                                              static_cast<int>(dimensions),
                                              1,
                                              metric)) {
    return false;
  }
  if (strict) {
    return launch_fused_topk<8, DataT, cuvs::distance::detail::cutile_abi_strict>(
      neighbors,
      distances,
      queries,
      corpus,
      static_cast<int>(query_rows),
      static_cast<int>(corpus_rows),
      static_cast<int>(dimensions),
      1,
      metric,
      stream,
      static_cast<int>(logical_k));
  }
  return launch_fused_topk<8, DataT, cuvs::distance::detail::cutile_abi_relaxed>(
    neighbors,
    distances,
    queries,
    corpus,
    static_cast<int>(query_rows),
    static_cast<int>(corpus_rows),
    static_cast<int>(dimensions),
    1,
    metric,
    stream,
    static_cast<int>(logical_k));
}

#define CUVS_INST_PK8_SINGLE(IdxT, DataT)                                \
  template CUVS_EXPORT bool try_fused_topk_pk8_single_tile<IdxT, DataT>( \
    IdxT*,                                                               \
    float*,                                                              \
    const DataT*,                                                        \
    const DataT*,                                                        \
    IdxT,                                                                \
    IdxT,                                                                \
    IdxT,                                                                \
    IdxT,                                                                \
    cuvs::distance::DistanceType,                                        \
    cudaStream_t)
CUVS_INST_PK8_SINGLE(int, float);
CUVS_INST_PK8_SINGLE(int64_t, float);
CUVS_INST_PK8_SINGLE(int, half);
CUVS_INST_PK8_SINGLE(int64_t, half);
#undef CUVS_INST_PK8_SINGLE

}  // namespace cuvs::neighbors::detail
