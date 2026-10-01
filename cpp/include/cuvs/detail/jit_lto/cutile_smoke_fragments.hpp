/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace cuvs::detail::jit_lto {

template <typename ArchTag>
struct fragment_tag_cutile_smoke_add_cubin {
  static constexpr int cc_major = ArchTag::cc_major;
  static constexpr int cc_minor = ArchTag::cc_minor;
};

}  // namespace cuvs::detail::jit_lto
