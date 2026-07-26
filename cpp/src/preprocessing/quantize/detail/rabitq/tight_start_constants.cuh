/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Imported from the standalone RaBitQ GPU quantizer (unchanged apart from the
 * enclosing namespace).
 */

//
// Per-TU __constant__ table used by the RaBitQ rescale-factor search.
// Defined `static` so each translation unit holding a kernel that reads it
// gets its own initialized copy — avoids needing -rdc=true for cross-TU
// __constant__ linkage.
//

#ifndef RABITQ_GPU_TIGHT_START_CONSTANTS_CUH
#define RABITQ_GPU_TIGHT_START_CONSTANTS_CUH

namespace cuvs::preprocessing::quantize::rabitq::detail {

static __device__ __constant__ float d_kTightStart_opt[9] = {
    0.0f,
    0.15f,
    0.20f,
    0.52f,
    0.59f,
    0.71f,
    0.75f,
    0.77f,
    0.81f,
};

}  // namespace cuvs::preprocessing::quantize::rabitq::detail

#endif // RABITQ_GPU_TIGHT_START_CONSTANTS_CUH
