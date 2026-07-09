/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cuvs_ivf_rabitq_wrapper.h"

namespace cuvs::bench {
template class cuvs_ivf_rabitq<float, int64_t>;
}  // namespace cuvs::bench
