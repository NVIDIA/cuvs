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

/** The dataset layout of the `cagra::index` type stored behind a C API handle. */
enum class c_api_dataset_layout : uint8_t {
  device_padded,
  device_standard,
  device_vpq,
  device_bbq,
  host_padded,
  host_standard
};

/** Resolves `cuvsCagraIndex::addr` to `cagra::index*`; nullptr if the handle is empty. */
void* cagra_c_api_index_ptr(cuvsCagraIndex const* idx);

/**
 * The dataset layout of the index behind the handle. The pointer returned by
 * `cagra_c_api_index_ptr` must be cast to the index type of this layout.
 */
c_api_dataset_layout cagra_c_api_index_layout(cuvsCagraIndex const* idx);

namespace detail {
template <typename T, typename IdxT, cuvs::neighbors::ann_dataset_view DatasetViewT>
int64_t merged_dataset_size(
  raft::resources const& res,
  std::vector<cuvs::neighbors::cagra::index<T, IdxT, DatasetViewT>*> const& indices,
  cuvs::neighbors::filtering::base_filter const& row_filter);
}  // namespace detail
}  // namespace cuvs::neighbors::cagra
