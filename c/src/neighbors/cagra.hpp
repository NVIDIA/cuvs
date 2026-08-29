/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::cagra {
/// Converts a cuvsCagraIndexParams struct (c) to a cagra::index_params (C++) struct
void convert_c_index_params(cuvsCagraIndexParams params,
                            int64_t n_rows,
                            int64_t dim,
                            cuvs::neighbors::cagra::index_params* out);

/// Converts C search params to C++
void convert_c_search_params(cuvsCagraSearchParams params,
                             cuvs::neighbors::cagra::search_params* out);

/** Resolves `cuvsCagraIndex::addr` to `cagra::index*`; nullptr if the handle is empty. */
void* cagra_c_api_index_ptr(cuvsCagraIndex const* idx);
}  // namespace cuvs::neighbors::cagra
