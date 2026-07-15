/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>

namespace cuvs::neighbors::cagra::helpers {

/** The batch size for the CAGRA optimize stage. */
constexpr static size_t kOptimizeBatchSize = 256 * 1024;

}  // namespace cuvs::neighbors::cagra::helpers
