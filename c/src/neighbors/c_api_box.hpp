/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cuvs::neighbors::c_api::detail {

struct owner_record {
  void* owner;
  void (*destroy_owner)(void*);
};

template <typename OwnerT>
static void destroy_typed_owner(void* owner)
{
  delete reinterpret_cast<OwnerT*>(owner);
}

template <typename OwnerT>
static auto make_owner_record(OwnerT* owner) -> owner_record
{
  return owner_record{owner, &destroy_typed_owner<OwnerT>};
}

[[maybe_unused]] static void destroy_owner_record(owner_record rec)
{
  if (rec.destroy_owner != nullptr) { rec.destroy_owner(rec.owner); }
}

}  // namespace cuvs::neighbors::c_api::detail
