
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <dlpack/dlpack.h>

#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <cuvs/distance/pairwise_distance.h>
#include <cuvs/distance/distance.hpp>

#include "../core/exceptions.hpp"
#include "../core/interop.hpp"

namespace {

template <typename T, typename DistT, typename LayoutT = raft::row_major>
void _pairwise_distance(cuvsResources_t res,
                        DLManagedTensor* x_tensor,
                        DLManagedTensor* y_tensor,
                        DLManagedTensor* distances_tensor,
                        cuvsDistanceType metric,
                        float metric_arg)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  using mdspan_type           = raft::device_matrix_view<T const, int64_t, LayoutT>;
  using distances_mdspan_type = raft::device_matrix_view<DistT, int64_t, LayoutT>;

  auto x_mds         = cuvs::core::from_dlpack<mdspan_type>(x_tensor);
  auto y_mds         = cuvs::core::from_dlpack<mdspan_type>(y_tensor);
  auto distances_mds = cuvs::core::from_dlpack<distances_mdspan_type>(distances_tensor);
  auto metric_type   = static_cast<cuvs::distance::DistanceType>(metric);

  cuvs::distance::pairwise_distance(*res_ptr, x_mds, y_mds, distances_mds, metric_type, metric_arg);
}
}  // namespace

extern "C" cuvsError_t cuvsPairwiseDistance(cuvsResources_t res,
                                            DLManagedTensor* x_tensor,
                                            DLManagedTensor* y_tensor,
                                            DLManagedTensor* distances_tensor,
                                            cuvsDistanceType metric,
                                            float metric_arg)
{
  return cuvs::core::translate_exceptions([=] {
    auto x_dt    = x_tensor->dl_tensor.dtype;
    auto y_dt    = x_tensor->dl_tensor.dtype;
    auto dist_dt = x_tensor->dl_tensor.dtype;

    if ((x_dt.code != kDLFloat) || (y_dt.code != kDLFloat) || (dist_dt.code != kDLFloat)) {
      RAFT_FAIL("Inputs to cuvsPairwiseDistance must all be floating point tensors");
    }

    if ((x_dt.bits != y_dt.bits) || (x_dt.bits != dist_dt.bits)) {
      RAFT_FAIL("Inputs to cuvsPairwiseDistance must all have the same dtype");
    }

    bool x_row_major;
    if (cuvs::core::is_c_contiguous(x_tensor)) {
      x_row_major = true;
    } else if (cuvs::core::is_f_contiguous(x_tensor)) {
      x_row_major = false;
    } else {
      RAFT_FAIL("X input to cuvsPairwiseDistance must be contiguous (non-strided)");
    }

    bool y_row_major;
    if (cuvs::core::is_c_contiguous(y_tensor)) {
      y_row_major = true;
    } else if (cuvs::core::is_f_contiguous(y_tensor)) {
      y_row_major = false;
    } else {
      RAFT_FAIL("Y input to cuvsPairwiseDistance must be contiguous (non-strided)");
    }

    bool distances_row_major;
    if (cuvs::core::is_c_contiguous(distances_tensor)) {
      distances_row_major = true;
    } else if (cuvs::core::is_f_contiguous(distances_tensor)) {
      distances_row_major = false;
    } else {
      RAFT_FAIL("distances input to cuvsPairwiseDistance must be contiguous (non-strided)");
    }

    if ((x_row_major != y_row_major) || (x_row_major != distances_row_major)) {
      RAFT_FAIL(
        "Inputs to cuvsPairwiseDistance must all have the same layout (row-major or col-major)");
    }

    if (x_row_major) {
      if (x_dt.bits == 32) {
        _pairwise_distance<float, float>(
          res, x_tensor, y_tensor, distances_tensor, metric, metric_arg);
      } else if (x_dt.bits == 16) {
        _pairwise_distance<half, float>(
          res, x_tensor, y_tensor, distances_tensor, metric, metric_arg);
      } else if (x_dt.bits == 64) {
        _pairwise_distance<double, double>(
          res, x_tensor, y_tensor, distances_tensor, metric, metric_arg);
      } else {
        RAFT_FAIL("Unsupported DLtensor dtype: %d and bits: %d", x_dt.code, x_dt.bits);
      }
    } else {
      if (x_dt.bits == 32) {
        _pairwise_distance<float, float, raft::col_major>(
          res, x_tensor, y_tensor, distances_tensor, metric, metric_arg);
      } else if (x_dt.bits == 16) {
        _pairwise_distance<half, float, raft::col_major>(
          res, x_tensor, y_tensor, distances_tensor, metric, metric_arg);
      } else if (x_dt.bits == 64) {
        _pairwise_distance<double, double, raft::col_major>(
          res, x_tensor, y_tensor, distances_tensor, metric, metric_arg);
      } else {
        RAFT_FAIL("Unsupported DLtensor dtype: %d and bits: %d", x_dt.code, x_dt.bits);
      }
    }
  });
}
