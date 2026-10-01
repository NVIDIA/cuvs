/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuvs/neighbors/ivf_sq.h>
#include <cuvs/neighbors/ivf_sq.hpp>

namespace cuvs::neighbors::ivf_sq {
/// Converts a cuvsIvfSqIndexParams struct (c) to a ivf_sq::index_params (C++) struct
void convert_c_index_params(cuvsIvfSqIndexParams params,
                            cuvs::neighbors::ivf_sq::index_params* out);
void convert_c_search_params(cuvsIvfSqSearchParams params,
                             cuvs::neighbors::ivf_sq::search_params* out);
}  // namespace cuvs::neighbors::ivf_sq
