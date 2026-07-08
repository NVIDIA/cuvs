/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../gpu_index/searcher_gpu_common.cuh"

namespace cuvs::neighbors::ivf_rabitq::detail {

using compute_inner_products_with_lut_func_t = void(const ComputeInnerProductsKernelParams);

}  // namespace cuvs::neighbors::ivf_rabitq::detail
