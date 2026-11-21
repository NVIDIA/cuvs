/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cuvs_scann_wrapper.h"

namespace cuvs::bench {
template class cuvs_scann<float, int64_t>;
template class cuvs_scann<half, int64_t>;
template class cuvs_scann<uint8_t, int64_t>;
template class cuvs_scann<int8_t, int64_t>;
}  // namespace cuvs::bench
