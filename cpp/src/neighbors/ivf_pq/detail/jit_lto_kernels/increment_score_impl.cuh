/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cuvs::neighbors::ivf_pq::detail {

template <typename OutT, bool Increment>
__device__ OutT increment_score_impl(OutT score)
{
  if constexpr (Increment) {
    return score + OutT(1);
  } else {
    return score;
  }
}

}  // namespace cuvs::neighbors::ivf_pq::detail
