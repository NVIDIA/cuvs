/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../detail/ann_utils.cuh"  // cuvs::spatial::knn::detail::utils::mapping

#include <cuvs/distance/distance.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/sample_rows.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cstdint>

namespace cuvs::neighbors::ivf_pq::detail {

/** Reduce the maximum squared L2 norm over a set of row-major vectors of element type DataT. */
template <typename DataT>
__global__ void kern_max_squared_norm(const DataT* __restrict__ data,
                                      int64_t n_rows,
                                      int64_t dim,
                                      float* __restrict__ out_max_sq)
{
  for (int64_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n_rows;
       row += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const DataT* v = data + row * dim;
    float sq       = 0.0f;
    for (int64_t d = 0; d < dim; d++) {
      // internally, IVF-PQ distance computations will map the input data type (e.g. FP16) to float before
      // doing arithmetic, so we need to apply the same mapping here to get a correct estimate of the squared norms
      // instead of using static_cast<float>(v[d])
      float e = cuvs::spatial::knn::detail::utils::mapping<float>{}(v[d]);
      sq += e * e;
    }
    // - There is no atomicMax for floats, so we embrace the bitwise representation monoticity
    //   between float and int. This is valid when values are non-negative, which is the case 
    //   for squared norms.
    // - Choose global atomic instead of shared memory tree reduction for simplicity, assuming
    //   low contention.
    atomicMax(reinterpret_cast<int*>(out_max_sq), __float_as_int(sq));
  }
}

/**
 * Estimate max_i ||x_i||^2 over the dataset by sampling a fraction of its rows.
 *
 * NOTE: sampling yields a *lower-bound* estimate of the true max norm, so a too-small fraction can
 * miss outlier vectors. Increase `sample_fraction` (up to 1.0 for an exact, no-false-negative scan)
 * if you observe overflow slipping through.
 */
template <typename DataT, typename Accessor>
float estimate_max_squared_norm(
  raft::resources const& res,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset)
{
  auto stream          = raft::resource::get_cuda_stream(res);
  const int64_t n_rows = dataset.extent(0);
  const int64_t dim    = dataset.extent(1);

  // Determine sample size based on a smooth saturation equation. The equation satisfies:
  // - n_sample is always less than or equal to n_rows
  // - n_sample saturates to kSaturation when n_rows is inf
  // - n_sample increases fast for small n_rows and slow to saturation for large n_rows
  // Idea: we greedily sample most of the dataset when it is small-sized, and cap it to kSaturation
  // when dataset size grows very large.
  constexpr int64_t kSaturation = 100000;
  int64_t n_sample = (n_rows * kSaturation + (n_rows + kSaturation - 1)) / (n_rows + kSaturation);

  // Sample from the dataset
  auto mr = raft::resource::get_workspace_resource_ref(res);
  auto sample =
    raft::make_device_mdarray<DataT>(res, mr, raft::make_extents<int64_t>(n_sample, dim));
  raft::matrix::sample_rows<DataT, int64_t>(
    res, raft::random::RngState{137}, dataset, sample.view());

  auto d_max_sq            = raft::make_device_scalar<float>(res, 0.0f);
  constexpr int block_size = 256;
  const int grid_size      = static_cast<int>((n_sample + block_size - 1) / block_size);
  kern_max_squared_norm<DataT><<<grid_size, block_size, 0, stream>>>(
    sample.data_handle(), n_sample, dim, d_max_sq.data_handle());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  float h_max_sq = 0.0f;
  raft::update_host(&h_max_sq, d_max_sq.data_handle(), 1, stream);
  raft::resource::sync_stream(res);
  return h_max_sq;
}

}  // namespace cuvs::neighbors::ivf_pq::detail

namespace cuvs::neighbors::ivf_pq::helpers {

/**
 * @brief Estimate whether FP16 is likely insufficient for IVF-PQ's full-magnitude distance
 * computations on this dataset (i.e. `internal_distance_dtype` and `coarse_search_dtype`).
 *
 * We bound the largest achievable score from the dataset's vector norms. With R = max_i ||x_i||
 * (estimated from a random sample of the dataset):
 *   - L2Expanded:     ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y> <= (||x|| + ||y||)^2 <= 4 * R^2
 *   - InnerProduct:   |<x, y>|    <= ||x|| * ||y||                                  <=     R^2
 *   - CosineExpanded: data is L2-normalized, so |score| <= 1 and overflow is impossible.
 */
template <typename DataT, typename Accessor>
bool estimate_fp16_overflow(
  raft::resources const& res,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  cuvs::distance::DistanceType metric)
{
  // Cosine similarity scores does normalization itself, so overflow won't happen
  if (metric == cuvs::distance::DistanceType::CosineExpanded) { return false; }

  // FP16 largest finite value, with a defensive margin to also avoid precision loss near the limit.
  constexpr float kFp16Max              = 65504.0f;
  constexpr float kFp16DefensiveMargin  = 0.25f;
  const float overflow_detect_threshold = kFp16DefensiveMargin * kFp16Max;

  const float max_vector_sq_norm =
    cuvs::neighbors::ivf_pq::detail::estimate_max_squared_norm(res, dataset);

  const float max_distance_sq_norm = metric == cuvs::distance::DistanceType::L2Expanded
                                       ? 4.0f * max_vector_sq_norm
                                       : max_vector_sq_norm;

  return max_distance_sq_norm > overflow_detect_threshold;
}

}  // namespace cuvs::neighbors::ivf_pq::helpers
