/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Imported from the standalone RaBitQ GPU quantizer. Behaviour unchanged;
 * only include paths, error macros and the enclosing namespace differ.
 */

//
// Device-side helpers for the RaBitQ rescale-factor search, plus a host entry
// point that draws the constant scaling factor used in the fast-quantize path.
//
// Extracted from quantizer_gpu_fast.cu — only the pieces needed by the
// standalone quantizer.
//

#ifndef RABITQ_GPU_RESCALE_SEARCH_CUH
#define RABITQ_GPU_RESCALE_SEARCH_CUH

#include <cstddef>
#include <cstdint>

namespace cuvs::preprocessing::quantize::rabitq::detail {

/// Warp-cooperative rescale-factor search. Declared here so the standalone
/// quantizer kernel can call it via `extern __device__` linkage.
///
/// s_xp_norm  : shared-memory array of |x|/||x|| values, length D.
/// D          : working dimension (padded).
/// EX_BITS    : number of extended bits (total_bits - 1).
/// reuse_space: scratch array in shared memory, at least BlockSize + 2*nWarps
///              floats (BlockSize/32 == nWarps).
/// BlockSize  : blockDim.x of the calling kernel.
///
/// Returns the selected rescale factor `t` (broadcast via the return value
/// from thread 0's path; all threads read it from shared memory through the
/// kernel's own __syncthreads pattern).
__device__ float compute_best_rescale_parallel(
        float* s_xp_norm,
        int D,
        int EX_BITS,
        float* reuse_space,
        int BlockSize,
        int coarse_samples = 64,
        int fine_samples = 64);

/// Host: estimate the constant scaling factor used by the fast-quantize path.
/// Averages `kConstNum` rescale factors computed on random Gaussian vectors.
float get_const_scaling_factor(size_t dim, size_t ex_bits,
                                           uint64_t seed = 12345ULL,
                                           int coarse_samples = 64,
                                           int fine_samples = 64);

}  // namespace cuvs::preprocessing::quantize::rabitq::detail

#endif // RABITQ_GPU_RESCALE_SEARCH_CUH
