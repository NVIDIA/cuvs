/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../util/serialize_validation.hpp"
#include "../ivf_common.cuh"
#include "../ivf_list.cuh"
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_sq.hpp>

#include <raft/core/detail/mdspan_numpy_serializer.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/serialize.hpp>
#include <raft/util/pow2_utils.cuh>

#include <fstream>

namespace cuvs::neighbors::ivf_sq::detail {

constexpr int serialization_version = 1;

template <typename CodeT>
void serialize(raft::resources const& handle, std::ostream& os, const index<CodeT>& index_)
{
  RAFT_LOG_DEBUG(
    "Saving IVF-SQ index, size %zu, dim %u", static_cast<size_t>(index_.size()), index_.dim());

  std::string dtype_string = raft::detail::numpy_serializer::get_numpy_dtype<CodeT>().to_string();
  dtype_string.resize(4);
  os << dtype_string;

  serialize_scalar(handle, os, serialization_version);
  serialize_scalar(handle, os, index_.size());
  serialize_scalar(handle, os, index_.dim());
  serialize_scalar(handle, os, index_.n_lists());
  serialize_scalar(handle, os, index_.metric());
  serialize_scalar(handle, os, index_.conservative_memory_allocation());
  serialize_mdspan(handle, os, index_.centers());

  if (index_.center_norms()) {
    bool has_norms = true;
    serialize_scalar(handle, os, has_norms);
    serialize_mdspan(handle, os, *index_.center_norms());
  } else {
    bool has_norms = false;
    serialize_scalar(handle, os, has_norms);
  }

  serialize_mdspan(handle, os, index_.sq_vmin());
  serialize_mdspan(handle, os, index_.sq_delta());

  auto sizes_host = raft::make_host_vector<uint32_t, uint32_t>(index_.list_sizes().extent(0));
  raft::copy(sizes_host.data_handle(),
             index_.list_sizes().data_handle(),
             sizes_host.size(),
             raft::resource::get_cuda_stream(handle));
  raft::resource::sync_stream(handle);
  serialize_mdspan(handle, os, sizes_host.view());

  list_spec<uint32_t, CodeT, int64_t> list_store_spec{index_.dim(), true};
  for (uint32_t label = 0; label < index_.n_lists(); label++) {
    ivf::serialize_list(handle,
                        os,
                        index_.lists()[label],
                        list_store_spec,
                        raft::Pow2<kIndexGroupSize>::roundUp(sizes_host(label)));
  }
  raft::resource::sync_stream(handle);
}

template <typename CodeT>
void serialize(raft::resources const& handle,
               const std::string& filename,
               const index<CodeT>& index_)
{
  std::ofstream of(filename, std::ios::out | std::ios::binary);
  if (!of) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }
  detail::serialize(handle, of, index_);
  of.close();
  if (!of) { RAFT_FAIL("Error writing output %s", filename.c_str()); }
}

template <typename CodeT>
auto deserialize(raft::resources const& handle, std::istream& is) -> index<CodeT>
{
  char dtype_string[4];
  RAFT_EXPECTS(is.read(dtype_string, 4), "ivf_sq::deserialize: failed to read dtype prefix");
  RAFT_EXPECTS(cuvs::util::validate_serialized_dtype<CodeT>(dtype_string, sizeof(dtype_string)),
               "ivf_sq::deserialize: serialized dtype prefix does not match requested type");

  auto ver = raft::deserialize_scalar<int>(handle, is);
  if (ver != serialization_version) {
    RAFT_FAIL("serialization version mismatch, expected %d, got %d ", serialization_version, ver);
  }
  auto n_rows  = raft::deserialize_scalar<int64_t>(handle, is);
  auto dim     = raft::deserialize_scalar<uint32_t>(handle, is);
  auto n_lists = raft::deserialize_scalar<uint32_t>(handle, is);
  auto metric  = raft::deserialize_scalar<cuvs::distance::DistanceType>(handle, is);
  bool cma     = raft::deserialize_scalar<bool>(handle, is);

  RAFT_EXPECTS(cuvs::util::is_valid_distance_type(metric),
               "ivf_sq::deserialize: invalid metric value %d",
               static_cast<int>(metric));
  RAFT_EXPECTS(n_lists <= cuvs::util::kMaxIvfNLists,
               "ivf_sq::deserialize: n_lists=%u exceeds maximum %u",
               n_lists,
               cuvs::util::kMaxIvfNLists);
  RAFT_EXPECTS(cuvs::util::is_mul_no_overflow(
                 static_cast<std::size_t>(n_lists), static_cast<std::size_t>(dim), sizeof(float)),
               "ivf_sq::deserialize: integer overflow in n_lists*dim*sizeof(float) "
               "(n_lists=%u, dim=%u)",
               n_lists,
               dim);
  RAFT_EXPECTS(cuvs::util::is_mul_no_overflow(static_cast<std::size_t>(dim), sizeof(float)),
               "ivf_sq::deserialize: integer overflow in dim*sizeof(float) (dim=%u)",
               dim);

  index<CodeT> index_ = index<CodeT>(handle, metric, n_lists, dim, cma);

  deserialize_mdspan(handle, is, index_.centers());

  bool has_norms = raft::deserialize_scalar<bool>(handle, is);
  if (has_norms) {
    index_.allocate_center_norms(handle);
    if (!index_.center_norms()) {
      RAFT_FAIL("Error inconsistent center norms");
    } else {
      auto center_norms = index_.center_norms().value();
      deserialize_mdspan(handle, is, center_norms);
    }
  }

  deserialize_mdspan(handle, is, index_.sq_vmin());
  deserialize_mdspan(handle, is, index_.sq_delta());

  deserialize_mdspan(handle, is, index_.list_sizes());

  list_spec<uint32_t, CodeT, int64_t> list_device_spec{index_.dim(), cma};
  list_spec<uint32_t, CodeT, int64_t> list_store_spec{index_.dim(), true};
  for (uint32_t label = 0; label < index_.n_lists(); label++) {
    ivf::deserialize_list(handle, is, index_.lists()[label], list_store_spec, list_device_spec);
  }
  raft::resource::sync_stream(handle);

  ivf::detail::recompute_internal_state(handle, index_);

  return index_;
}

template <typename CodeT>
auto deserialize(raft::resources const& handle, const std::string& filename) -> index<CodeT>
{
  std::ifstream is(filename, std::ios::in | std::ios::binary);
  if (!is) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }
  auto index = detail::deserialize<CodeT>(handle, is);
  is.close();
  return index;
}

}  // namespace cuvs::neighbors::ivf_sq::detail

#define CUVS_INST_IVF_SQ_SERIALIZE(CodeT)                                           \
  void serialize(raft::resources const& handle,                                     \
                 const std::string& filename,                                       \
                 const cuvs::neighbors::ivf_sq::index<CodeT>& index)                \
  {                                                                                 \
    cuvs::neighbors::ivf_sq::detail::serialize(handle, filename, index);            \
  }                                                                                 \
                                                                                    \
  void deserialize(raft::resources const& handle,                                   \
                   const std::string& filename,                                     \
                   cuvs::neighbors::ivf_sq::index<CodeT>* index)                    \
  {                                                                                 \
    *index = cuvs::neighbors::ivf_sq::detail::deserialize<CodeT>(handle, filename); \
  }
