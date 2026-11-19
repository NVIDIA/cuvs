/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <dlpack/dlpack.h>

#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/refine.h>
#include <cuvs/neighbors/refine.hpp>

#include "../core/exceptions.hpp"
#include "../core/interop.hpp"

namespace {

template <typename T>
void _refine(bool on_device,
             cuvsResources_t res,
             DLManagedTensor* dataset_tensor,
             DLManagedTensor* queries_tensor,
             DLManagedTensor* candidates_tensor,
             cuvsDistanceType metric,
             DLManagedTensor* indices_tensor,
             DLManagedTensor* distances_tensor)
{
  RAFT_EXPECTS(res != 0, "res must not be NULL");
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  if (on_device) {
    using queries_type    = raft::device_matrix_view<const T, int64_t, raft::row_major>;
    using candidates_type = raft::device_matrix_view<const int64_t, int64_t, raft::row_major>;
    using indices_type    = raft::device_matrix_view<int64_t, int64_t, raft::row_major>;
    using distances_type  = raft::device_matrix_view<float, int64_t, raft::row_major>;
    auto dataset          = cuvs::core::from_dlpack<queries_type>(dataset_tensor);
    auto queries          = cuvs::core::from_dlpack<queries_type>(queries_tensor);
    auto candidates       = cuvs::core::from_dlpack<candidates_type>(candidates_tensor);
    auto indices          = cuvs::core::from_dlpack<indices_type>(indices_tensor);
    auto distances        = cuvs::core::from_dlpack<distances_type>(distances_tensor);
    cuvs::neighbors::refine(*res_ptr, dataset, queries, candidates, indices, distances, static_cast<cuvs::distance::DistanceType>((int)metric));
  } else {
    using queries_type    = raft::host_matrix_view<const T, int64_t, raft::row_major>;
    using candidates_type = raft::host_matrix_view<const int64_t, int64_t, raft::row_major>;
    using indices_type    = raft::host_matrix_view<int64_t, int64_t, raft::row_major>;
    using distances_type  = raft::host_matrix_view<float, int64_t, raft::row_major>;
    auto dataset          = cuvs::core::from_dlpack<queries_type>(dataset_tensor);
    auto queries          = cuvs::core::from_dlpack<queries_type>(queries_tensor);
    auto candidates       = cuvs::core::from_dlpack<candidates_type>(candidates_tensor);
    auto indices          = cuvs::core::from_dlpack<indices_type>(indices_tensor);
    auto distances        = cuvs::core::from_dlpack<distances_type>(distances_tensor);
    cuvs::neighbors::refine(*res_ptr, dataset, queries, candidates, indices, distances, static_cast<cuvs::distance::DistanceType>((int)metric));
  }
}
}  // namespace

extern "C" cuvsError_t cuvsRefine(cuvsResources_t res,
                                  DLManagedTensor* dataset_tensor,
                                  DLManagedTensor* queries_tensor,
                                  DLManagedTensor* candidates_tensor,
                                  cuvsDistanceType metric,
                                  DLManagedTensor* indices_tensor,
                                  DLManagedTensor* distances_tensor)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset    = dataset_tensor->dl_tensor;
    auto queries    = queries_tensor->dl_tensor;
    auto candidates = candidates_tensor->dl_tensor;
    auto indices    = indices_tensor->dl_tensor;
    auto distances  = distances_tensor->dl_tensor;

    // all matrices must either be on host or on device, can't mix and match
    bool on_device = cuvs::core::is_dlpack_device_compatible(dataset);
    if (on_device != cuvs::core::is_dlpack_device_compatible(queries) ||
        on_device != cuvs::core::is_dlpack_device_compatible(candidates) ||
        on_device != cuvs::core::is_dlpack_device_compatible(indices) ||
        on_device != cuvs::core::is_dlpack_device_compatible(distances)) {
      RAFT_FAIL("Tensors must either all be on device memory, or all on host memory");
    }

    RAFT_EXPECTS(candidates.dtype.code == kDLInt && candidates.dtype.bits == 64,
                 "candidates should be of type int64_t");
    RAFT_EXPECTS(indices.dtype.code == kDLInt && indices.dtype.bits == 64,
                 "indices should be of type int64_t");
    RAFT_EXPECTS(distances.dtype.code == kDLFloat && distances.dtype.bits == 32,
                 "distances should be of type float32");

    RAFT_EXPECTS(queries.dtype.code == dataset.dtype.code,
                 "type mismatch between dataset and queries");

    if (queries.dtype.code == kDLFloat && queries.dtype.bits == 32) {
      _refine<float>(on_device,
                     res,
                     dataset_tensor,
                     queries_tensor,
                     candidates_tensor,
                     metric,
                     indices_tensor,
                     distances_tensor);
    } else if (queries.dtype.code == kDLFloat && queries.dtype.bits == 16) {
      _refine<half>(on_device,
                    res,
                    dataset_tensor,
                    queries_tensor,
                    candidates_tensor,
                    metric,
                    indices_tensor,
                    distances_tensor);
    } else if (queries.dtype.code == kDLInt && queries.dtype.bits == 8) {
      _refine<int8_t>(on_device,
                      res,
                      dataset_tensor,
                      queries_tensor,
                      candidates_tensor,
                      metric,
                      indices_tensor,
                      distances_tensor);
    } else if (queries.dtype.code == kDLUInt && queries.dtype.bits == 8) {
      _refine<uint8_t>(on_device,
                       res,
                       dataset_tensor,
                       queries_tensor,
                       candidates_tensor,
                       metric,
                       indices_tensor,
                       distances_tensor);
    } else {
      RAFT_FAIL("Unsupported queries DLtensor dtype: %d and bits: %d",
                queries.dtype.code,
                queries.dtype.bits);
    }
  });
}
