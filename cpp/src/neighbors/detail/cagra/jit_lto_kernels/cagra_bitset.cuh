/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * reserved. SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../../sample_filter.cuh"  // bitset_filter, none_sample_filter
#include "../../sample_filter_data.cuh"

#include <cstdint>
#include <type_traits>

// The single-partition sample-filter payload (cagra_sample_filter, is_bitset_filter,
// extract_cagra_sample_filter) lives in cagra_filter_payload.{hpp,cuh}. This header retains only
// the CAGRA-side alias for the bitset payload passed by value into the (multi-partition) kernels.
namespace cuvs::neighbors::cagra::detail {

template <typename SourceIndexT>
using cagra_bitset = cuvs::neighbors::detail::bitset_filter_data_t<SourceIndexT>;

}  // namespace cuvs::neighbors::cagra::detail
