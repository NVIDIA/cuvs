/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cuvs::cluster::kmeans::detail::superkmeans {

template <typename DataTag, typename ArchTag>
struct fragment_tag_fused_filter_cubin {
  static constexpr int cc_major = ArchTag::cc_major;
  static constexpr int cc_minor = ArchTag::cc_minor;
};

}  // namespace cuvs::cluster::kmeans::detail::superkmeans
