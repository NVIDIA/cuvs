/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/detail/jit_lto/FragmentEntry.hpp>

bool FatbinFragmentEntry::add_to(nvJitLinkHandle& handle) const
{
  auto result = nvJitLinkAddData(handle, NVJITLINK_INPUT_ANY, get_data(), get_length(), get_key());

  check_nvjitlink_result(handle, result);
  return true;
}
