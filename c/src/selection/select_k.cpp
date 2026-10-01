/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/c_api.h>
#include "../core/exceptions.hpp"
#include "../core/interop.hpp"
#include <cuvs/selection/select_k.h>   // C declaration with CUVS_EXPORT (default visibility)
#include <cuvs/selection/select_k.hpp>
#include <dlpack/dlpack.h>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resources.hpp>

extern "C" cuvsError_t cuvsSelectK(cuvsResources_t res,
                                   DLManagedTensor* in_val,
                                   DLManagedTensor* out_val,
                                   DLManagedTensor* out_idx)
{
  return cuvs::core::translate_exceptions([=] {
    auto* res_ptr = reinterpret_cast<raft::resources*>(res);

    RAFT_EXPECTS(in_val != nullptr && out_val != nullptr && out_idx != nullptr,
                 "Tensors passed to cuvsSelectK must not be null");
    RAFT_EXPECTS(in_val->dl_tensor.byte_offset == 0 && out_val->dl_tensor.byte_offset == 0 &&
                   out_idx->dl_tensor.byte_offset == 0,
                 "Tensors passed to cuvsSelectK must have a zero byte_offset");

    // from_dlpack validates ndim, dtype, device, and row-major contiguity against each mdspan type.
    using in_view_type  = raft::device_matrix_view<const float, int64_t, raft::row_major>;
    using val_view_type = raft::device_matrix_view<float, int64_t, raft::row_major>;
    using idx_view_type = raft::device_matrix_view<int64_t, int64_t, raft::row_major>;

    auto in_view      = cuvs::core::from_dlpack<in_view_type>(in_val);
    auto out_val_view = cuvs::core::from_dlpack<val_view_type>(out_val);
    auto out_idx_view = cuvs::core::from_dlpack<idx_view_type>(out_idx);

    RAFT_EXPECTS(in_view.extent(0) == 1 && out_val_view.extent(0) == 1 &&
                   out_idx_view.extent(0) == 1,
                 "cuvsSelectK only supports a single row (batch_size == 1)");
    RAFT_EXPECTS(out_val_view.extent(1) == out_idx_view.extent(1),
                 "out_val and out_idx passed to cuvsSelectK must have the same length");
    RAFT_EXPECTS(out_val_view.extent(1) <= in_view.extent(1),
                 "k passed to cuvsSelectK must not exceed the input length");

    cuvs::selection::select_k(
      *res_ptr,
      in_view,
      std::nullopt,  // implicit positions [0, n) as in_idx
      out_val_view,
      out_idx_view,
      true);  // select_min = true (smallest distance = nearest neighbor)
  });
}
