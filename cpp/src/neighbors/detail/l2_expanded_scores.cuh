/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/cublas_macros.hpp>
#include <raft/core/resource/cublaslt_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/custom_resource.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/detail/cublaslt_wrappers.hpp>
#include <raft/util/cache.hpp>
#include <raft/util/cuda_data_type.hpp>

#include <cublasLt.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>

namespace cuvs::neighbors::detail {

/** The largest power of two (at most 256) dividing the address: what cuBLASLt may assume of it. */
inline auto l2_scores_alignment(const void* ptr) -> uint32_t
{
  auto const address = reinterpret_cast<uintptr_t>(ptr);
  uint32_t alignment = 256;
  while (alignment > 1 && address % alignment != 0) {
    alignment /= 2;
  }
  return alignment;
}

struct l2_scores_key {
  uint64_t m;
  uint64_t n;
  uint64_t d;
  uint32_t search_alignment;
  uint32_t index_alignment;
  uint32_t scores_alignment;

  auto operator==(const l2_scores_key&) const -> bool = default;
};

struct l2_scores_key_hash {
  auto operator()(const l2_scores_key& x) const noexcept -> std::size_t
  {
    return x.m * x.n * x.d + x.search_alignment + x.index_alignment * 512 +
           x.scores_alignment * 512 * 512;
  }
};

/** A matmul with the bias epilogue, and the algorithm cuBLASLt chose for it (if any). */
struct l2_scores_plan {
  raft::linalg::detail::cublastlt_matmul_desc desc;
  raft::linalg::detail::cublastlt_matrix_layout index_layout;
  raft::linalg::detail::cublastlt_matrix_layout search_layout;
  raft::linalg::detail::cublastlt_matrix_layout scores_layout;
  cublasLtMatmulHeuristicResult_t heuristics{};
  bool supported = false;
};

template <typename T>
struct l2_scores_cache {
  raft::cache::
    lru<l2_scores_key, l2_scores_key_hash, std::equal_to<>, std::shared_ptr<l2_scores_plan>>
      value{};
};

template <typename T>
auto make_l2_scores_plan(raft::resources const& handle, const l2_scores_key& key) -> l2_scores_plan
{
  namespace lt = raft::linalg::detail;
  // Column-major, the [m x n] row-major scores are [n x m] = index [n x d] * search^T: the index
  // rows are the rows of the result, so the bias epilogue adds one norm per index row.
  l2_scores_plan plan{
    lt::cublastlt_matmul_desc::for_matmul<float, T, T, float>(/*transpose_a=*/true,
                                                              /*transpose_b=*/false),
    lt::cublastlt_matrix_layout{raft::get_cuda_data_type<T>(), key.d, key.n, key.d},
    lt::cublastlt_matrix_layout{raft::get_cuda_data_type<T>(), key.d, key.m, key.d},
    lt::cublastlt_matrix_layout{CUDA_R_32F, key.n, key.m, key.n}};
  const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
  RAFT_CUBLAS_TRY(cublasLtMatmulDescSetAttribute(
    plan.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
  const cudaDataType_t bias_type = CUDA_R_32F;
  RAFT_CUBLAS_TRY(cublasLtMatmulDescSetAttribute(
    plan.desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type)));

  // Tiles start at arbitrary rows, so the operands are only as aligned as their addresses say.
  lt::cublastlt_matmul_preference preference;
  auto const set_alignment = [&](cublasLtMatmulPreferenceAttributes_t attribute, uint32_t bytes) {
    RAFT_CUBLAS_TRY(
      cublasLtMatmulPreferenceSetAttribute(preference, attribute, &bytes, sizeof(bytes)));
  };
  set_alignment(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, key.index_alignment);
  set_alignment(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, key.search_alignment);
  set_alignment(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, key.scores_alignment);
  set_alignment(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, key.scores_alignment);

  // The FP32 matmul in RAFT steers cuBLASLt 13.6.0 off a faulty algorithm; this one does the same.
  const lt::matmul_key_t args{key.n, key.m, key.d, key.d, key.d, key.n, true, false};
  bool const workaround =
    std::is_same_v<T, float> && lt::needs_cublaslt_13_6_workaround(args, cublasLtGetVersion());
  auto const heuristic_index =
    workaround ? lt::cublastlt_matrix_layout{raft::get_cuda_data_type<T>(),
                                             key.d,
                                             key.n,
                                             lt::get_cublaslt_13_6_heuristic_args(args).lda}
               : lt::cublastlt_matrix_layout{raft::get_cuda_data_type<T>(), key.d, key.n, key.d};
  int algo_count = 0;
  RAFT_CUBLAS_TRY(cublasLtMatmulAlgoGetHeuristic(raft::resource::get_cublaslt_handle(handle),
                                                 plan.desc,
                                                 heuristic_index,
                                                 plan.search_layout,
                                                 plan.scores_layout,
                                                 plan.scores_layout,
                                                 preference,
                                                 1,
                                                 &plan.heuristics,
                                                 &algo_count));
  constexpr int faulty_algorithm = 68;
  plan.supported =
    algo_count > 0 &&
    !(workaround && lt::get_cublaslt_algorithm_id(plan.heuristics) == faulty_algorithm);
  return plan;
}

/**
 * @brief Scores a tile for expanded L2 ranking in one cuBLASLt matmul:
 * scores(i, j) = |index_j|^2 - 2 search_i . index_j, row-major [m x n].
 *
 * alpha = -2 scales the dot products and the bias epilogue adds the index norms, so no pass over
 * the tile turns dot products into distances. |search_i|^2 is the same along a row and does not
 * change its ranking; the caller adds it to the rows it selects. Inputs are multiplied as given
 * (FP16 on the tensor cores, FP32 on CUDA cores; no TF32) and accumulated in FP32.
 *
 * @return false, leaving @p scores untouched, if cuBLASLt has no algorithm for this matmul.
 */
template <typename T>
auto l2_expanded_scores(raft::resources const& handle,
                        const T* search,
                        const T* index,
                        uint64_t m,
                        uint64_t n,
                        uint64_t d,
                        const float* index_norms,
                        float* scores) -> bool
{
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>);
  if (raft::resource::get_dry_run_flag(handle)) { return true; }
  const l2_scores_key key{
    m, n, d, l2_scores_alignment(search), l2_scores_alignment(index), l2_scores_alignment(scores)};
  auto& cache = raft::resource::get_custom_resource<l2_scores_cache<T>>(handle)->value;
  std::shared_ptr<l2_scores_plan> plan;
  if (!cache.get(key, &plan)) {
    plan = std::make_shared<l2_scores_plan>(make_l2_scores_plan<T>(handle, key));
    cache.set(key, plan);
  }
  if (!plan->supported) { return false; }

  RAFT_CUBLAS_TRY(cublasLtMatmulDescSetAttribute(
    plan->desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &index_norms, sizeof(index_norms)));
  const float alpha = -2.0f;
  const float beta  = 0.0f;
  RAFT_CUBLAS_TRY(cublasLtMatmul(raft::resource::get_cublaslt_handle(handle),
                                 plan->desc,
                                 &alpha,
                                 index,
                                 plan->index_layout,
                                 search,
                                 plan->search_layout,
                                 &beta,
                                 scores,
                                 plan->scores_layout,
                                 scores,
                                 plan->scores_layout,
                                 &plan->heuristics.algo,
                                 nullptr,
                                 0,
                                 raft::resource::get_cuda_stream(handle).get()));
  return true;
}

}  // namespace cuvs::neighbors::detail
