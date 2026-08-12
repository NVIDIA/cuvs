/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/knn_merge_parts.cuh"

#include <cuvs/neighbors/knn_merge_parts.hpp>

namespace cuvs::neighbors {
namespace {
template <typename T, typename IdxT>
void _knn_merge_parts(raft::resources const& res,
                      raft::device_matrix_view<const T, int64_t> inK,
                      raft::device_matrix_view<const IdxT, int64_t> inV,
                      raft::device_matrix_view<T, int64_t> outK,
                      raft::device_matrix_view<IdxT, int64_t> outV,
                      raft::device_vector_view<IdxT, int64_t> translations)
{
  auto parts = translations.extent(0);
  auto rows  = outK.extent(0);
  auto k     = outK.extent(1);

  detail::knn_merge_parts(inK.data_handle(),
                          inV.data_handle(),
                          outK.data_handle(),
                          outV.data_handle(),
                          rows,
                          parts,
                          k,
                          raft::resource::get_cuda_stream(res),
                          translations.data_handle());
}
}  // namespace

void knn_merge_parts(raft::resources const& res,
                     raft::device_matrix_view<const float, int64_t> inK,
                     raft::device_matrix_view<const int64_t, int64_t> inV,
                     raft::device_matrix_view<float, int64_t> outK,
                     raft::device_matrix_view<int64_t, int64_t> outV,
                     raft::device_vector_view<int64_t, int64_t> translations)
{
  _knn_merge_parts(res, inK, inV, outK, outV, translations);
}
void knn_merge_parts(raft::resources const& res,
                     raft::device_matrix_view<const float, int64_t> inK,
                     raft::device_matrix_view<const uint32_t, int64_t> inV,
                     raft::device_matrix_view<float, int64_t> outK,
                     raft::device_matrix_view<uint32_t, int64_t> outV,
                     raft::device_vector_view<uint32_t, int64_t> translations)
{
  _knn_merge_parts(res, inK, inV, outK, outV, translations);
}
void knn_merge_parts(raft::resources const& res,
                     raft::device_matrix_view<const float, int64_t> inK,
                     raft::device_matrix_view<const int32_t, int64_t> inV,
                     raft::device_matrix_view<float, int64_t> outK,
                     raft::device_matrix_view<int32_t, int64_t> outV,
                     raft::device_vector_view<int32_t, int64_t> translations)
{
  _knn_merge_parts(res, inK, inV, outK, outV, translations);
}
}  // namespace cuvs::neighbors
