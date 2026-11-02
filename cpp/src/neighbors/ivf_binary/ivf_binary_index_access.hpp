/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/ivf_binary.hpp>

namespace cuvs::neighbors::ivf_binary::detail {

/** Internal access to the mutable state of the index. */
struct index_access {
  static auto lists(index<int64_t>& idx) -> ivf_flat::index<uint8_t, int64_t>&
  {
    return idx.lists_;
  }
  static auto lists(const index<int64_t>& idx) -> const ivf_flat::index<uint8_t, int64_t>&
  {
    return idx.lists_;
  }
  static auto centers(index<int64_t>& idx)
    -> raft::device_matrix<uint8_t, int64_t, raft::row_major>&
  {
    return idx.centers_;
  }
  static auto center_counts(index<int64_t>& idx)
    -> raft::device_matrix<uint32_t, int64_t, raft::row_major>&
  {
    return idx.center_counts_;
  }
  static auto quantizer(index<int64_t>& idx) -> preprocessing::quantize::binary::quantizer<float>&
  {
    return idx.quantizer_;
  }
};

}  // namespace cuvs::neighbors::ivf_binary::detail
