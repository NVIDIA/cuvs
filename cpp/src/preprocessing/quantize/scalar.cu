/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/cuda_fp16.hpp>
#include <cuvs/core/dataset/sq.hpp>

#include "./detail/scalar.cuh"

#include <cuvs/preprocessing/quantize/scalar.hpp>

namespace cuvs::preprocessing::quantize::scalar {

quantizer::quantizer(double min, double max) : value{detail::checked_bounds(min, max)} {}

quantizer::quantizer(raft::resources const& res,
                     raft::device_vector_view<const double, int64_t> mins,
                     raft::device_vector_view<const double, int64_t> maxs)
{
  RAFT_EXPECTS(mins.extent(0) > 0 && mins.extent(0) == maxs.extent(0),
               "Bounds must be nonempty and have equal lengths");
  auto minima = raft::make_device_vector<double, int64_t>(res, mins.extent(0));
  auto maxima = raft::make_device_vector<double, int64_t>(res, maxs.extent(0));
  raft::copy(res, minima.view(), mins);
  raft::copy(res, maxima.view(), maxs);
  value = detail::checked_bounds(res, std::move(minima), std::move(maxima));
}

#define CUVS_INST_QUANTIZATION(T, QuantI)                                                       \
  auto train(raft::resources const& res,                                                        \
             const params params,                                                               \
             raft::device_matrix_view<const T, int64_t> dataset) -> quantizer                   \
  {                                                                                             \
    return detail::train<T>(res, params, dataset);                                              \
  }                                                                                             \
  auto train(raft::resources const& res,                                                        \
             const params params,                                                               \
             raft::host_matrix_view<const T, int64_t> dataset) -> quantizer                     \
  {                                                                                             \
    return detail::train<T>(res, params, dataset);                                              \
  }                                                                                             \
  double compute_max_error(raft::resources const& res,                                          \
                           const quantizer& quantizer,                                          \
                           raft::device_matrix_view<const T, int64_t> dataset)                  \
  {                                                                                             \
    return detail::max_error<T, QuantI>(res, quantizer.view(), dataset, 0);                     \
  }                                                                                             \
  double compute_max_error(raft::resources const& res,                                          \
                           const quantizer& quantizer,                                          \
                           raft::host_matrix_view<const T, int64_t> dataset,                    \
                           int64_t device_buffer_samples)                                       \
  {                                                                                             \
    return detail::max_error<T, QuantI>(res, quantizer.view(), dataset, device_buffer_samples); \
  }                                                                                             \
  void transform(raft::resources const& res,                                                    \
                 const quantizer& quantizer,                                                    \
                 raft::device_matrix_view<const T, int64_t> dataset,                            \
                 raft::device_matrix_view<QuantI, int64_t> out)                                 \
  {                                                                                             \
    detail::apply_quantizer<T, QuantI>(res, quantizer.view(), dataset, out);                    \
  }                                                                                             \
  void transform(raft::resources const& res,                                                    \
                 const quantizer& quantizer,                                                    \
                 raft::host_matrix_view<const T, int64_t> dataset,                              \
                 raft::host_matrix_view<QuantI, int64_t> out)                                   \
  {                                                                                             \
    detail::apply_quantizer<T, QuantI>(res, quantizer.view(), dataset, out);                    \
  }                                                                                             \
  void inverse_transform(raft::resources const& res,                                            \
                         const quantizer& quantizer,                                            \
                         raft::device_matrix_view<const QuantI, int64_t> dataset,               \
                         raft::device_matrix_view<T, int64_t> out)                              \
  {                                                                                             \
    detail::apply_quantizer<T, QuantI>(res, quantizer.view(), dataset, out);                    \
  }                                                                                             \
  void inverse_transform(raft::resources const& res,                                            \
                         const quantizer& quantizer,                                            \
                         raft::host_matrix_view<const QuantI, int64_t> dataset,                 \
                         raft::host_matrix_view<T, int64_t> out)                                \
  {                                                                                             \
    detail::apply_quantizer<T, QuantI>(res, quantizer.view(), dataset, out);                    \
  }                                                                                             \
  auto make_sq_dataset(raft::resources const& res,                                              \
                       params params,                                                           \
                       raft::host_matrix_view<const T, int64_t> dataset,                        \
                       bool compute_max_error) -> device_sq_dataset<T, int64_t, T>              \
  {                                                                                             \
    using result_type = device_sq_dataset<T, int64_t, T>;                                       \
    auto codes =                                                                                \
      raft::make_device_matrix<int8_t, int64_t>(res, dataset.extent(0), dataset.extent(1));     \
    auto quantizer = detail::train<T>(res, params, dataset);                                    \
    /* The encoding pass also measures the error when requested. */                             \
    auto error = detail::max_error<T, QuantI>(res,                                              \
                                              quantizer.view(),                                 \
                                              dataset,                                          \
                                              params.device_buffer_samples,                     \
                                              compute_max_error,                                \
                                              codes.view());                                    \
    return result_type{std::move(codes),                                                        \
                       typename result_type::manifest_type{std::move(quantizer), error}};       \
  }

CUVS_INST_QUANTIZATION(double, int8_t);
CUVS_INST_QUANTIZATION(float, int8_t);
CUVS_INST_QUANTIZATION(half, int8_t);

#undef CUVS_INST_QUANTIZATION

}  // namespace cuvs::preprocessing::quantize::scalar
