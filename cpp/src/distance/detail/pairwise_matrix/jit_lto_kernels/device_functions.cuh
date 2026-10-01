/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cuvs::distance::detail {

template <typename OpT, typename DataT, typename AccT, typename IdxT>
extern __device__ void compute_distance(OpT distance_op, AccT& acc, DataT x, DataT y);

template <typename Policy, typename OpT, typename AccT, typename IdxT>
extern __device__ void compute_distance_epilog(OpT distance_op,
                                               AccT acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                                               AccT* regxn,
                                               AccT* regyn,
                                               IdxT grid_stride_x,
                                               IdxT grid_stride_y);

}  // namespace cuvs::distance::detail
