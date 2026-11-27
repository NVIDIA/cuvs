/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fused_filter_tile.hpp"

#include "../../../../core/nvtx.hpp"

#include <cuvs/detail/jit_lto/TileAlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/common_fragments.hpp>
#include <cuvs/detail/jit_lto/cutile_arch_tags.hpp>
#include <cuvs/detail/jit_lto/pruned_top_1_nn/fused_filter_fragments.hpp>

#include <raft/core/error.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/kernel_launch.hpp>

#include <cuda_fp16.h>

#include <limits>

namespace cuvs::distance::detail::pruned_nn {

namespace {

/** Must match kernel_symbol() in fused_filter_kernel.py. */
constexpr const char* kFusedFilterEntrypoint = "pruned_1nn_fused_filter_h";

struct FusedFilterTilePlanner : cuvs::detail::jit_lto::TileAlgorithmPlanner {
  using DataTag = cuvs::neighbors::detail::tag_h;

  inline static cuvs::detail::jit_lto::TileLauncherCache launcher_cache{};

  FusedFilterTilePlanner() : TileAlgorithmPlanner(kFusedFilterEntrypoint, launcher_cache)
  {
    using namespace cuvs::detail::jit_lto;
    add_static_fragment<fragment_tag_fused_filter_cubin<DataTag, cutile_arch_8_0>>();
    add_static_fragment<fragment_tag_fused_filter_cubin<DataTag, cutile_arch_8_6>>();
    add_static_fragment<fragment_tag_fused_filter_cubin<DataTag, cutile_arch_9_0>>();
    add_static_fragment<fragment_tag_fused_filter_cubin<DataTag, cutile_arch_10_0>>();
#if CUVS_CUTILE_SM103_ENABLED
    add_static_fragment<fragment_tag_fused_filter_cubin<DataTag, cutile_arch_10_3>>();
#endif
#if CUVS_CUTILE_SM107_ENABLED
    add_static_fragment<fragment_tag_fused_filter_cubin<DataTag, cutile_arch_10_7>>();
#endif
    add_static_fragment<fragment_tag_fused_filter_cubin<DataTag, cutile_arch_12_0>>();
  }
};

}  // namespace

auto fused_filter_tile_config() -> std::optional<fused_filter_tile>
{
  FusedFilterTilePlanner planner;
  if (planner.try_get_launcher() == nullptr) { return std::nullopt; }
  const auto cfg = planner.tile_config();
  return fused_filter_tile{cfg.tile_m, cfg.tile_n, cfg.tile_k};
}

void launch_fused_filter(const raft::resources& handle,
                         const half* rows,
                         int n_rows,
                         const half* centroids,
                         int n_centroids,
                         int dim,
                         const float* row_norms,
                         const float* centroid_norms,
                         const long long* best,
                         float ratio,
                         int partial_dim,
                         uint32_t* bitmap,
                         int bitmap_stride)
{
  common::nvtx::range<common::nvtx::domain::cuvs> range("pruned_1nn::fused_filter");
  FusedFilterTilePlanner planner;
  auto launcher = planner.try_get_launcher();
  RAFT_EXPECTS(launcher != nullptr, "Pruned 1-NN fused filter: no cuTile cubin for this device");
  const auto tile = planner.tile_config();

  constexpr int row_pitch_divisor = 16 / sizeof(half);
  RAFT_EXPECTS(n_rows > 0 && n_centroids > 0, "Pruned 1-NN fused filter: empty input");
  RAFT_EXPECTS(dim % row_pitch_divisor == 0,
               "Pruned 1-NN fused filter: the row pitch must be a multiple of 16 bytes");
  RAFT_EXPECTS(partial_dim > 0 && partial_dim <= dim && partial_dim % tile.tile_k == 0,
               "Pruned 1-NN fused filter: partial_dim must be a positive multiple of %d, at most "
               "dim",
               tile.tile_k);
  for (const void* ptr : {static_cast<const void*>(rows),
                          static_cast<const void*>(centroids),
                          static_cast<const void*>(row_norms),
                          static_cast<const void*>(centroid_norms),
                          static_cast<const void*>(best),
                          static_cast<const void*>(bitmap)}) {
    RAFT_EXPECTS(reinterpret_cast<uintptr_t>(ptr) % 16 == 0,
                 "Pruned 1-NN fused filter: buffers must be 16-byte aligned");
  }
  constexpr int kBitsPerWord = 32;
  RAFT_EXPECTS(tile.tile_n % kBitsPerWord == 0,
               "Pruned 1-NN filter tiles must contain whole bitmap words");
  const int64_t centroid_tiles = (int64_t{n_centroids} + tile.tile_n - 1) / tile.tile_n;
  RAFT_EXPECTS(bitmap_stride >= centroid_tiles * (tile.tile_n / kBitsPerWord),
               "Pruned 1-NN fused filter: bitmap row stride is too small");
  RAFT_EXPECTS(int64_t{n_rows} * bitmap_stride <= std::numeric_limits<int>::max(),
               "Pruned 1-NN fused filter: bitmap is too large for 32-bit indexing");

  using fused_filter_kernel_t = void(void*,
                                     int,
                                     int,
                                     int,
                                     int,  // rows
                                     void*,
                                     int,
                                     int,
                                     int,
                                     int,  // centroids
                                     void*,
                                     int,
                                     int,  // row norms
                                     void*,
                                     int,
                                     int,  // centroid norms
                                     void*,
                                     int,
                                     int,  // winners
                                     void*,
                                     int,
                                     int,
                                     int,
                                     int,  // bitmap
                                     int,  // n_centroids
                                     float,
                                     int);
  void* rows_ptr              = const_cast<half*>(rows);
  void* centroids_ptr         = const_cast<half*>(centroids);
  void* row_norms_ptr         = const_cast<float*>(row_norms);
  void* centroid_norms_ptr    = const_cast<float*>(centroid_norms);
  void* best_ptr              = const_cast<long long*>(best);
  void* bitmap_ptr            = bitmap;
  const int unit              = 1;

  dim3 grid((n_rows + tile.tile_m - 1) / tile.tile_m, 1, 1);
  dim3 block(1, 1, 1);
  // Each array as (pointer, shape..., stride...), then the scalars; the tile sizes are compiled in.
  raft::launch_kernel(handle,
                      grid,
                      block,
                      raft::kernel_ref<fused_filter_kernel_t>{launcher->get_kernel()},
                      rows_ptr,
                      int{n_rows},
                      int{dim},
                      int{dim},
                      int{unit},
                      centroids_ptr,
                      int{n_centroids},
                      int{dim},
                      int{dim},
                      int{unit},
                      row_norms_ptr,
                      int{n_rows},
                      int{unit},
                      centroid_norms_ptr,
                      int{n_centroids},
                      int{unit},
                      best_ptr,
                      int{n_rows},
                      int{unit},
                      bitmap_ptr,
                      int{n_rows},
                      int{bitmap_stride},
                      int{bitmap_stride},
                      int{unit},
                      int{n_centroids},
                      float{ratio},
                      int{partial_dim});
}

}  // namespace cuvs::distance::detail::pruned_nn
