/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace cuvs::neighbors::ivf_flat::detail {

template <typename T, typename IdxT, typename LabelT, typename SourceIdxT, bool gather_src>
void launch_build_index(dim3 grid,
                        dim3 block,
                        cudaStream_t stream,
                        const LabelT* labels,
                        const T* source_vecs,
                        const SourceIdxT* source_ixs,
                        T** list_data_ptrs,
                        IdxT** list_index_ptrs,
                        uint32_t* list_sizes_ptr,
                        IdxT n_rows,
                        uint32_t dim,
                        uint32_t veclen,
                        IdxT batch_offset = 0);

}  // namespace cuvs::neighbors::ivf_flat::detail
