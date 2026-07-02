/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/cagra/cagra_serialize.cuh"

namespace cuvs::neighbors::cagra {

#define CUVS_INST_CAGRA_SERIALIZE(DTYPE)                                                          \
  void serialize(raft::resources const& handle,                                                   \
                 const std::string& filename,                                                     \
                 const cuvs::neighbors::cagra::device_padded_index<DTYPE, uint32_t>& index,       \
                 bool include_dataset)                                                            \
  {                                                                                               \
    cuvs::neighbors::cagra::detail::serialize<DTYPE, uint32_t>(                                   \
      handle, filename, index, include_dataset);                                                  \
  };                                                                                              \
                                                                                                  \
  void deserialize(                                                                               \
    raft::resources const& handle,                                                                \
    const std::string& filename,                                                                  \
    cuvs::neighbors::cagra::device_padded_index<DTYPE, uint32_t>* index,                          \
    std::unique_ptr<cuvs::neighbors::device_padded_dataset<DTYPE, int64_t>>* out_dataset)         \
  {                                                                                               \
    cuvs::neighbors::cagra::detail::deserialize<DTYPE, uint32_t>(                                 \
      handle, filename, index, out_dataset);                                                      \
  };                                                                                              \
  void serialize(raft::resources const& handle,                                                   \
                 std::ostream& os,                                                                \
                 const cuvs::neighbors::cagra::device_padded_index<DTYPE, uint32_t>& index,       \
                 bool include_dataset)                                                            \
  {                                                                                               \
    cuvs::neighbors::cagra::detail::serialize<DTYPE, uint32_t>(                                   \
      handle, os, index, include_dataset);                                                        \
  }                                                                                               \
                                                                                                  \
  void deserialize(                                                                               \
    raft::resources const& handle,                                                                \
    std::istream& is,                                                                             \
    cuvs::neighbors::cagra::device_padded_index<DTYPE, uint32_t>* index,                          \
    std::unique_ptr<cuvs::neighbors::device_padded_dataset<DTYPE, int64_t>>* out_dataset)         \
  {                                                                                               \
    cuvs::neighbors::cagra::detail::deserialize<DTYPE, uint32_t>(handle, is, index, out_dataset); \
  }                                                                                               \
                                                                                                  \
  void serialize(raft::resources const& handle,                                                   \
                 const std::string& filename,                                                     \
                 const cuvs::neighbors::cagra::device_standard_index<DTYPE, uint32_t>& index,     \
                 bool include_dataset)                                                            \
  {                                                                                               \
    cuvs::neighbors::cagra::detail::serialize<DTYPE, uint32_t>(                                   \
      handle, filename, index, include_dataset);                                                  \
  };                                                                                              \
                                                                                                  \
  void deserialize(                                                                               \
    raft::resources const& handle,                                                                \
    const std::string& filename,                                                                  \
    cuvs::neighbors::cagra::device_standard_index<DTYPE, uint32_t>* index,                        \
    std::unique_ptr<cuvs::neighbors::device_standard_dataset<DTYPE, int64_t>>* out_dataset)       \
  {                                                                                               \
    cuvs::neighbors::cagra::detail::deserialize<DTYPE, uint32_t>(                                 \
      handle, filename, index, out_dataset);                                                      \
  };                                                                                              \
  void serialize(raft::resources const& handle,                                                   \
                 std::ostream& os,                                                                \
                 const cuvs::neighbors::cagra::device_standard_index<DTYPE, uint32_t>& index,     \
                 bool include_dataset)                                                            \
  {                                                                                               \
    cuvs::neighbors::cagra::detail::serialize<DTYPE, uint32_t>(                                   \
      handle, os, index, include_dataset);                                                        \
  }                                                                                               \
                                                                                                  \
  void deserialize(                                                                               \
    raft::resources const& handle,                                                                \
    std::istream& is,                                                                             \
    cuvs::neighbors::cagra::device_standard_index<DTYPE, uint32_t>* index,                        \
    std::unique_ptr<cuvs::neighbors::device_standard_dataset<DTYPE, int64_t>>* out_dataset)       \
  {                                                                                               \
    cuvs::neighbors::cagra::detail::deserialize<DTYPE, uint32_t>(handle, is, index, out_dataset); \
  }                                                                                               \
                                                                                                  \
  void serialize_to_hnswlib(                                                                      \
    raft::resources const& handle,                                                                \
    std::ostream& os,                                                                             \
    const cuvs::neighbors::cagra::device_padded_index<DTYPE, uint32_t>& index,                    \
    std::optional<raft::host_matrix_view<const DTYPE, int64_t, raft::row_major>> dataset)         \
  {                                                                                               \
    cuvs::neighbors::cagra::detail::serialize_to_hnswlib<DTYPE, uint32_t>(                        \
      handle, os, index, dataset);                                                                \
  }                                                                                               \
                                                                                                  \
  void serialize_to_hnswlib(                                                                      \
    raft::resources const& handle,                                                                \
    const std::string& filename,                                                                  \
    const cuvs::neighbors::cagra::device_padded_index<DTYPE, uint32_t>& index,                    \
    std::optional<raft::host_matrix_view<const DTYPE, int64_t, raft::row_major>> dataset)         \
  {                                                                                               \
    cuvs::neighbors::cagra::detail::serialize_to_hnswlib<DTYPE, uint32_t>(                        \
      handle, filename, index, dataset);                                                          \
  }                                                                                               \
                                                                                                  \
  void serialize_to_hnswlib(                                                                      \
    raft::resources const& handle,                                                                \
    std::ostream& os,                                                                             \
    const cuvs::neighbors::cagra::host_padded_index<DTYPE, uint32_t>& index,                      \
    std::optional<raft::host_matrix_view<const DTYPE, int64_t, raft::row_major>> dataset)         \
  {                                                                                               \
    cuvs::neighbors::cagra::detail::serialize_to_hnswlib<DTYPE, uint32_t>(                        \
      handle, os, index, dataset);                                                                \
  }                                                                                               \
                                                                                                  \
  void serialize_to_hnswlib(                                                                      \
    raft::resources const& handle,                                                                \
    const std::string& filename,                                                                  \
    const cuvs::neighbors::cagra::host_padded_index<DTYPE, uint32_t>& index,                      \
    std::optional<raft::host_matrix_view<const DTYPE, int64_t, raft::row_major>> dataset)         \
  {                                                                                               \
    cuvs::neighbors::cagra::detail::serialize_to_hnswlib<DTYPE, uint32_t>(                        \
      handle, filename, index, dataset);                                                          \
  }                                                                                               \
                                                                                                  \
  void serialize_to_hnswlib(                                                                      \
    raft::resources const& handle,                                                                \
    std::ostream& os,                                                                             \
    const cuvs::neighbors::cagra::device_standard_index<DTYPE, uint32_t>& index,                  \
    std::optional<raft::host_matrix_view<const DTYPE, int64_t, raft::row_major>> dataset)         \
  {                                                                                               \
    cuvs::neighbors::cagra::detail::serialize_to_hnswlib<DTYPE, uint32_t>(                        \
      handle, os, index, dataset);                                                                \
  }                                                                                               \
                                                                                                  \
  void serialize_to_hnswlib(                                                                      \
    raft::resources const& handle,                                                                \
    const std::string& filename,                                                                  \
    const cuvs::neighbors::cagra::device_standard_index<DTYPE, uint32_t>& index,                  \
    std::optional<raft::host_matrix_view<const DTYPE, int64_t, raft::row_major>> dataset)         \
  {                                                                                               \
    cuvs::neighbors::cagra::detail::serialize_to_hnswlib<DTYPE, uint32_t>(                        \
      handle, filename, index, dataset);                                                          \
  }                                                                                               \
                                                                                                  \
  void serialize_to_hnswlib(                                                                      \
    raft::resources const& handle,                                                                \
    std::ostream& os,                                                                             \
    const cuvs::neighbors::cagra::host_standard_index<DTYPE, uint32_t>& index,                    \
    std::optional<raft::host_matrix_view<const DTYPE, int64_t, raft::row_major>> dataset)         \
  {                                                                                               \
    cuvs::neighbors::cagra::detail::serialize_to_hnswlib<DTYPE, uint32_t>(                        \
      handle, os, index, dataset);                                                                \
  }                                                                                               \
                                                                                                  \
  void serialize_to_hnswlib(                                                                      \
    raft::resources const& handle,                                                                \
    const std::string& filename,                                                                  \
    const cuvs::neighbors::cagra::host_standard_index<DTYPE, uint32_t>& index,                    \
    std::optional<raft::host_matrix_view<const DTYPE, int64_t, raft::row_major>> dataset)         \
  {                                                                                               \
    cuvs::neighbors::cagra::detail::serialize_to_hnswlib<DTYPE, uint32_t>(                        \
      handle, filename, index, dataset);                                                          \
  }

}  // namespace cuvs::neighbors::cagra
