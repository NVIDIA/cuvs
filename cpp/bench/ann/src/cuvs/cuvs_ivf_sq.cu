/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cuvs_ivf_sq_wrapper.h"

namespace cuvs::bench {
template class cuvs_ivf_sq<float>;
template class cuvs_ivf_sq<half>;
}  // namespace cuvs::bench
