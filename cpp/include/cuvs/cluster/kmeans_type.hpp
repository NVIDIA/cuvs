/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/core/export.hpp>

namespace CUVS_EXPORT cuvs {
namespace cluster {
namespace kmeans {

/**
 * @brief Type of k-means algorithm.
 */
enum class kmeans_type { KMeans = 0, KMeansBalanced = 1 };

}  // namespace kmeans
}  // namespace cluster
}  // namespace CUVS_EXPORT cuvs
