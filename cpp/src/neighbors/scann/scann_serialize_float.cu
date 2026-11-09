/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/scann_serialize.cuh"
#include <cuvs/neighbors/scann.hpp>

namespace cuvs::neighbors::scann {

/**
 * @defgroup ScaNN graph serialize/derserialize
 * @{
 */

#define CUVS_INST_SCANN_SERIALIZE(DTYPE, IdxT)                                           \
  void serialize(raft::resources const& handle,                                          \
                 const std::string& file_prefix,                                         \
                 const cuvs::neighbors::scann::index<DTYPE, IdxT>& index_)              \
  {                                                                                      \
    cuvs::neighbors::scann::detail::serialize<DTYPE, IdxT>(                              \
      handle, file_prefix, index_);                                                      \
  };

CUVS_INST_SCANN_SERIALIZE(float, int64_t);

/** @} */  // end group scann

}  // namespace cuvs::neighbors::scann
