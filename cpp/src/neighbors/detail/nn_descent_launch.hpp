/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "nn_descent_gnnd.hpp"
#include <cuvs/core/export.hpp>
#include <raft/util/arch.cuh>

namespace cuvs::neighbors::nn_descent::detail {
template <typename T>
decltype(raft::util::arch::kernel_virtual_arch(nullptr)) bbq_join_virtual_arch();
template <typename T>
CUVS_EXPORT decltype(raft::util::arch::kernel_virtual_arch(nullptr)) norm_kernel_virtual_arch();
template <typename Data_t>
CUVS_EXPORT void launch_compute_l2_norms_kernel(dim3 grid,
                                                dim3 block,
                                                size_t shared,
                                                cudaStream_t stream,
                                                const Data_t* data,
                                                int dim,
                                                DistData_t* l2_norms);
template <typename Src_t, typename Dst_t>
CUVS_EXPORT void launch_convert_copy_kernel(dim3 grid,
                                            dim3 block,
                                            size_t shared,
                                            cudaStream_t stream,
                                            const Src_t* src,
                                            Dst_t* dst,
                                            size_t n);

extern template class GnndGraph<int>;
}  // namespace cuvs::neighbors::nn_descent::detail
