/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/util/cuda_dev_essentials.cuh>

namespace cuvs::neighbors::detail {

/** Jaccard distance is the number of differing bits divided by the union size. */
template <typename DistanceT>
RAFT_DEVICE_INLINE_FUNCTION DistanceT bitwise_jaccard_distance(DistanceT differing,
                                                               DistanceT set_union)
{
  return set_union > 0 ? differing / set_union : DistanceT(0);
}

}  // namespace cuvs::neighbors::detail
