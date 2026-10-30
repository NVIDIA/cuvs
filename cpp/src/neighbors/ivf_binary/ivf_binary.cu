/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../util/kvikio_serialize.hpp"
#include "../../util/serialize_validation.hpp"
#include "../ivf_flat/ivf_flat_serialize.cuh"
#include "ivf_binary_build.cuh"
#include "ivf_binary_search.cuh"

#include <cuvs/neighbors/ivf_binary.hpp>

#include <raft/core/serialize.hpp>

namespace cuvs::neighbors::ivf_binary {
namespace {

// Serialization version of the index: the header, the quantizer and the centers, followed by
// the inverted lists in the IVF-Flat format.
constexpr int kSerializationVersion = 1;

void serialize_impl(raft::resources const& handle, std::ostream& os, const index<int64_t>& idx)
{
  raft::serialize_scalar(handle, os, kSerializationVersion);
  raft::serialize_scalar(handle, os, idx.dim());
  raft::serialize_scalar(handle, os, idx.n_lists());
  raft::serialize_scalar(handle, os, idx.adaptive_centers());
  raft::serialize_scalar(handle, os, idx.conservative_memory_allocation());
  // The zero threshold is implied by the dimension and has no stored thresholds
  const auto* threshold = std::get_if<1>(&idx.quantizer().value);
  raft::serialize_scalar(handle, os, int64_t{threshold != nullptr ? threshold->extent(0) : 0});
  if (threshold != nullptr) { cuvs::util::detail::serialize_mdspan(handle, os, threshold->view()); }
  cuvs::util::detail::serialize_mdspan(handle, os, idx.centers());
  if (idx.adaptive_centers()) {
    cuvs::util::detail::serialize_mdspan(handle, os, idx.center_counts());
  }
  ivf_flat::detail::serialize(handle, os, detail::index_access::lists(idx));
}

template <typename Input>
auto deserialize_impl(raft::resources const& handle, Input& input) -> index<int64_t>
{
  auto& is = cuvs::util::detail::input_stream(input);

  auto version = raft::deserialize_scalar<int>(handle, is);
  RAFT_EXPECTS(version == kSerializationVersion,
               "ivf_binary::deserialize: serialization version mismatch, expected %d, got %d",
               kSerializationVersion,
               version);
  auto dim         = raft::deserialize_scalar<uint32_t>(handle, is);
  auto n_lists     = raft::deserialize_scalar<uint32_t>(handle, is);
  bool adaptive    = raft::deserialize_scalar<bool>(handle, is);
  bool cma         = raft::deserialize_scalar<bool>(handle, is);
  auto n_threshold = raft::deserialize_scalar<int64_t>(handle, is);

  RAFT_EXPECTS(dim > 0 && n_lists > 0 && n_lists <= cuvs::util::kMaxIvfNLists,
               "ivf_binary::deserialize: invalid dimension %u or number of lists %u",
               dim,
               n_lists);
  RAFT_EXPECTS(n_threshold == 0 || n_threshold == static_cast<int64_t>(dim),
               "ivf_binary::deserialize: invalid threshold size %ld",
               static_cast<long>(n_threshold));

  index_params params;
  params.n_lists                        = n_lists;
  params.adaptive_centers               = adaptive;
  params.conservative_memory_allocation = cma;
  index<int64_t> idx(handle, params, dim);

  // The index is created with the zero-threshold quantizer of the given dimension
  if (n_threshold > 0) {
    auto thresholds = raft::make_device_vector<float, int64_t>(handle, n_threshold);
    cuvs::util::detail::deserialize_mdspan(handle, input, thresholds.view());
    detail::index_access::quantizer(idx) =
      cuvs::preprocessing::quantize::binary::quantizer<float>(std::move(thresholds));
  }
  cuvs::util::detail::deserialize_mdspan(handle, input, detail::index_access::centers(idx).view());
  if (adaptive) {
    cuvs::util::detail::deserialize_mdspan(
      handle, input, detail::index_access::center_counts(idx).view());
  }

  auto lists = ivf_flat::detail::deserialize_impl<uint8_t, int64_t>(handle, input);
  RAFT_EXPECTS(lists.n_lists() == n_lists && lists.dim() == idx.storage_dim() &&
                 lists.metric() == cuvs::distance::DistanceType::BitwiseHamming,
               "ivf_binary::deserialize: the lists do not match the index header");
  detail::index_access::lists(idx) = std::move(lists);
  return idx;
}

}  // namespace

auto build(raft::resources const& handle,
           const index_params& index_params,
           device_binary_dataset_view<float, int64_t> dataset) -> index<int64_t>
{
  return detail::build(handle, index_params, dataset);
}

void extend(raft::resources const& handle,
            device_binary_dataset_view<float, int64_t> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            index<int64_t>* index)
{
  detail::extend(handle, new_vectors, new_indices, index);
}

void search(raft::resources const& handle,
            const search_params& params,
            const index<int64_t>& index,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter)
{
  detail::search(handle, params, index, queries, neighbors, distances, sample_filter);
}

void serialize(raft::resources const& handle, std::ostream& os, const index<int64_t>& index)
{
  serialize_impl(handle, os, index);
}

void serialize(raft::resources const& handle,
               const std::string& filename,
               const index<int64_t>& index)
{
  cuvs::util::kvikio_ofstream of(filename);
  serialize_impl(handle, of, index);
  of.close();
  if (!of) { RAFT_FAIL("Error writing output %s", filename.c_str()); }
}

void deserialize(raft::resources const& handle, std::istream& is, index<int64_t>* index)
{
  *index = deserialize_impl(handle, is);
}

void deserialize(raft::resources const& handle, const std::string& filename, index<int64_t>* index)
{
  cuvs::util::kvikio_file_reader reader(filename);
  *index = deserialize_impl(handle, reader);
}

}  // namespace cuvs::neighbors::ivf_binary
