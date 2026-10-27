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

template <typename Index_t>
void launch_add_rev_edges(size_t nrow,
                          cudaStream_t stream,
                          const Index_t* graph,
                          Index_t* rev_graph,
                          int num_samples,
                          int2* list_sizes);

template <typename Data_t, typename Index_t, typename DistEpilogue_t>
void launch_local_join_simt(dim3 grid,
                            dim3 block,
                            size_t shared,
                            cudaStream_t stream,
                            const Index_t* graph_new,
                            const Index_t* rev_graph_new,
                            const int2* sizes_new,
                            const Index_t* graph_old,
                            const Index_t* rev_graph_old,
                            const int2* sizes_old,
                            int width,
                            const Data_t* data,
                            int data_dim,
                            InternalID_t<Index_t>* graph,
                            DistData_t* dists,
                            int* locks,
                            DistData_t* l2_norms,
                            cuvs::distance::DistanceType metric,
                            DistEpilogue_t dist_epilogue);

template <typename Data_t, typename Index_t, typename DistEpilogue_t>
void launch_local_join_wmma(dim3 grid,
                            dim3 block,
                            size_t shared,
                            cudaStream_t stream,
                            const Index_t* graph_new,
                            const Index_t* rev_graph_new,
                            const int2* sizes_new,
                            const Index_t* graph_old,
                            const Index_t* rev_graph_old,
                            const int2* sizes_old,
                            int width,
                            const Data_t* data,
                            int data_dim,
                            InternalID_t<Index_t>* graph,
                            DistData_t* dists,
                            int* locks,
                            DistData_t* l2_norms,
                            cuvs::distance::DistanceType metric,
                            DistEpilogue_t dist_epilogue);

void* norm_kernel_pointer(const float*);

void* norm_kernel_pointer(const __half*);

void* norm_kernel_pointer(const int8_t*);

void* norm_kernel_pointer(const uint8_t*);

void* bbq_kernel_pointer(const float*);

void* bbq_kernel_pointer(const __half*);

void* bbq_kernel_pointer(const int8_t*);

void* bbq_kernel_pointer(const uint8_t*);

extern template class GnndGraph<int>;
}  // namespace cuvs::neighbors::nn_descent::detail
