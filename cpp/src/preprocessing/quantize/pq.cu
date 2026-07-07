/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./detail/pq.cuh"

#include <cuvs/preprocessing/quantize/pq.hpp>

namespace cuvs::preprocessing::quantize::pq {

#define CUVS_INST_QUANTIZATION(T, QuantI)                                                   \
  auto build(raft::resources const& res,                                                    \
             const params params,                                                           \
             raft::device_matrix_view<const T, int64_t> dataset) -> quantizer<T>            \
  {                                                                                         \
    return detail::build<T, T>(res, params, dataset);                                       \
  }                                                                                         \
  auto build(raft::resources const& res,                                                    \
             const params params,                                                           \
             raft::host_matrix_view<const T, int64_t> dataset) -> quantizer<T>              \
  {                                                                                         \
    return detail::build<T, T>(res, params, dataset);                                       \
  }                                                                                         \
  auto build(                                                                               \
    raft::resources const& res,                                                             \
    const params params,                                                                    \
    raft::device_matrix_view<const T, uint32_t, raft::row_major> pq_centers,                \
    std::optional<raft::device_matrix_view<const T, uint32_t, raft::row_major>> vq_centers) \
    -> quantizer<T>                                                                         \
  {                                                                                         \
    return detail::build_view<T>(res, params, pq_centers, vq_centers);                      \
  }                                                                                         \
  void transform(raft::resources const& res,                                                \
                 const quantizer<T>& quantizer,                                             \
                 raft::device_matrix_view<const T, int64_t> dataset,                        \
                 raft::device_matrix_view<QuantI, int64_t> codes_out,                       \
                 std::optional<raft::device_vector_view<uint32_t, int64_t>> vq_labels)      \
  {                                                                                         \
    detail::transform(res, quantizer, dataset, codes_out, vq_labels);                       \
  }                                                                                         \
  void transform(raft::resources const& res,                                                \
                 const quantizer<T>& quantizer,                                             \
                 raft::host_matrix_view<const T, int64_t> dataset,                          \
                 raft::device_matrix_view<QuantI, int64_t> codes_out,                       \
                 std::optional<raft::device_vector_view<uint32_t, int64_t>> vq_labels)      \
  {                                                                                         \
    detail::transform(res, quantizer, dataset, codes_out, vq_labels);                       \
  }                                                                                         \
  void inverse_transform(                                                                   \
    raft::resources const& res,                                                             \
    const quantizer<T>& quantizer,                                                          \
    raft::device_matrix_view<const QuantI, int64_t> pq_codes,                               \
    raft::device_matrix_view<T, int64_t> out,                                               \
    std::optional<raft::device_vector_view<const uint32_t, int64_t>> vq_labels)             \
  {                                                                                         \
    detail::inverse_transform(res, quantizer, pq_codes, out, vq_labels);                    \
  }

CUVS_INST_QUANTIZATION(float, uint8_t);

#undef CUVS_INST_QUANTIZATION

#define CUVS_INST_VPQ_BUILD(T)                                                               \
  auto vpq_build(const raft::resources& res,                                                 \
                 const cuvs::neighbors::vpq_params& params,                                  \
                 const raft::host_matrix_view<const T, int64_t, raft::row_major>& dataset)   \
  {                                                                                          \
    return detail::vpq_build_half<decltype(dataset)>(res, params, dataset);                  \
  }                                                                                          \
  auto vpq_build(const raft::resources& res,                                                 \
                 const cuvs::neighbors::vpq_params& params,                                  \
                 const raft::device_matrix_view<const T, int64_t, raft::row_major>& dataset) \
  {                                                                                          \
    return detail::vpq_build_half<decltype(dataset)>(res, params, dataset);                  \
  }

CUVS_INST_VPQ_BUILD(float);
CUVS_INST_VPQ_BUILD(half);
CUVS_INST_VPQ_BUILD(int8_t);
CUVS_INST_VPQ_BUILD(uint8_t);

#undef CUVS_INST_VPQ_BUILD

template class vpq_codebooks<half>;
template class vpq_codebooks<float>;
template class vpq_dataset<half, int64_t>;
template class vpq_dataset<float, int64_t>;

}  // namespace cuvs::preprocessing::quantize::pq
