/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./detail/scalar.cuh"

#include <cuvs/preprocessing/quantize/scalar.hpp>

namespace cuvs::preprocessing::quantize::scalar {

template <typename T>
quantizer<T>::quantizer(raft::resources const& res,
                        raft::device_vector_view<const T, int64_t> mins,
                        raft::device_vector_view<const T, int64_t> maxs)
{
  RAFT_EXPECTS(mins.extent(0) > 0 && mins.extent(0) == maxs.extent(0),
               "Bounds must be nonempty and have equal lengths");
  auto minima = raft::make_device_vector<T, int64_t>(res, mins.extent(0));
  auto maxima = raft::make_device_vector<T, int64_t>(res, maxs.extent(0));
  raft::copy(res, minima.view(), mins);
  raft::copy(res, maxima.view(), maxs);
  value = detail::train(res, std::move(minima), std::move(maxima));
}

#define CUVS_INST_QUANTIZATION(T, QuantI)                                         \
  auto train(raft::resources const& res,                                          \
             const params params,                                                 \
             raft::device_matrix_view<const T, int64_t> dataset) -> quantizer<T>  \
  {                                                                               \
    return detail::train(res, params, dataset);                                   \
  }                                                                               \
  auto train(raft::resources const& res,                                          \
             const params params,                                                 \
             raft::host_matrix_view<const T, int64_t> dataset) -> quantizer<T>    \
  {                                                                               \
    return detail::train(res, params, dataset);                                   \
  }                                                                               \
  void transform(raft::resources const& res,                                      \
                 quantizer_const_view<T> quantizer,                               \
                 raft::device_matrix_view<const T, int64_t> dataset,              \
                 raft::device_matrix_view<QuantI, int64_t> out)                   \
  {                                                                               \
    detail::transform(res, quantizer, dataset, out);                              \
  }                                                                               \
  void transform(raft::resources const& res,                                      \
                 quantizer_const_view<T> quantizer,                               \
                 raft::host_matrix_view<const T, int64_t> dataset,                \
                 raft::host_matrix_view<QuantI, int64_t> out)                     \
  {                                                                               \
    detail::transform(res, quantizer, dataset, out);                              \
  }                                                                               \
  void inverse_transform(raft::resources const& res,                              \
                         quantizer_const_view<T> quantizer,                       \
                         raft::device_matrix_view<const QuantI, int64_t> dataset, \
                         raft::device_matrix_view<T, int64_t> out)                \
  {                                                                               \
    detail::inverse_transform(res, quantizer, dataset, out);                      \
  }                                                                               \
  void inverse_transform(raft::resources const& res,                              \
                         quantizer_const_view<T> quantizer,                       \
                         raft::host_matrix_view<const QuantI, int64_t> dataset,   \
                         raft::host_matrix_view<T, int64_t> out)                  \
  {                                                                               \
    detail::inverse_transform(res, quantizer, dataset, out);                      \
  }                                                                               \
  template struct quantizer<T>;

CUVS_INST_QUANTIZATION(double, int8_t);
CUVS_INST_QUANTIZATION(float, int8_t);
CUVS_INST_QUANTIZATION(half, int8_t);

#undef CUVS_INST_QUANTIZATION

}  // namespace cuvs::preprocessing::quantize::scalar
