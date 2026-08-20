/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../util/kvikio_serialize.hpp"
#include "../util/serialize_validation.hpp"

#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/copy.cuh>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/numpy_serializer.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>

#include <fstream>

namespace cuvs::neighbors::brute_force {

int constexpr serialization_version = 0;

template <typename T, typename DistT, typename Output>
void serialize(raft::resources const& handle,
               Output& os,
               const index<T, DistT>& index,
               bool include_dataset = true)
{
  RAFT_LOG_DEBUG(
    "Saving brute force index, size %zu, dim %u", static_cast<size_t>(index.size()), index.dim());

  auto dtype_string = raft::numpy_serializer::get_numpy_dtype<T>().to_string();
  dtype_string.resize(4);
  os << dtype_string;

  raft::serialize_scalar(handle, os, serialization_version);
  raft::serialize_scalar(handle, os, index.size());
  raft::serialize_scalar(handle, os, index.dim());
  raft::serialize_scalar(handle, os, index.metric());
  raft::serialize_scalar(handle, os, index.metric_arg());
  raft::serialize_scalar(handle, os, include_dataset);
  if (include_dataset) { cuvs::util::detail::serialize_mdspan(handle, os, index.dataset()); }
  auto has_norms = index.has_norms();
  raft::serialize_scalar(handle, os, has_norms);
  if (has_norms) { cuvs::util::detail::serialize_mdspan(handle, os, index.norms()); }
  raft::resource::sync_stream(handle);
}

void serialize(raft::resources const& handle,
               const std::string& filename,
               const index<half, float>& index,
               bool include_dataset)
{
  cuvs::util::kvikio_ofstream os(filename);
  RAFT_EXPECTS(os, "Cannot open file %s", filename.c_str());
  serialize<half, float>(handle, os, index, include_dataset);
  os.close();
  RAFT_EXPECTS(os, "Error writing output %s", filename.c_str());
}

void serialize(raft::resources const& handle,
               const std::string& filename,
               const index<float, float>& index,
               bool include_dataset)
{
  cuvs::util::kvikio_ofstream os(filename);
  RAFT_EXPECTS(os, "Cannot open file %s", filename.c_str());
  serialize<float, float>(handle, os, index, include_dataset);
  os.close();
  RAFT_EXPECTS(os, "Error writing output %s", filename.c_str());
}

void serialize(raft::resources const& handle,
               std::ostream& os,
               const index<half, float>& index,
               bool include_dataset)
{
  serialize<half, float>(handle, os, index, include_dataset);
}

void serialize(raft::resources const& handle,
               std::ostream& os,
               const index<float, float>& index,
               bool include_dataset)
{
  serialize<float, float>(handle, os, index, include_dataset);
}

template <typename T, typename DistT, typename Input>
auto deserialize(raft::resources const& handle, Input& input)
{
  auto& is = cuvs::util::detail::input_stream(input);

  char dtype_string[4];
  RAFT_EXPECTS(is.read(dtype_string, 4), "brute_force::deserialize: failed to read dtype prefix");
  RAFT_EXPECTS(cuvs::util::validate_serialized_dtype<T>(dtype_string, sizeof(dtype_string)),
               "brute_force::deserialize: serialized dtype prefix does not match requested type");

  auto ver = raft::deserialize_scalar<int>(handle, is);
  if (ver != serialization_version) {
    RAFT_FAIL("serialization version mismatch, expected %d, got %d ", serialization_version, ver);
  }
  constexpr std::size_t kMax = static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
  auto rows_raw              = raft::deserialize_scalar<size_t>(handle, is);
  auto dim_raw               = raft::deserialize_scalar<size_t>(handle, is);
  RAFT_EXPECTS(
    rows_raw <= kMax, "brute_force::deserialize: rows=%zu does not fit in int64_t", rows_raw);
  RAFT_EXPECTS(
    dim_raw <= kMax, "brute_force::deserialize: dim=%zu does not fit in int64_t", dim_raw);
  auto rows   = static_cast<std::int64_t>(rows_raw);
  auto dim    = static_cast<std::int64_t>(dim_raw);
  auto metric = raft::deserialize_scalar<cuvs::distance::DistanceType>(handle, is);
  RAFT_EXPECTS(cuvs::util::is_valid_distance_type(metric),
               "brute_force::deserialize: invalid metric value %d",
               static_cast<int>(metric));
  auto metric_arg = raft::deserialize_scalar<DistT>(handle, is);

  auto dataset_storage = raft::make_device_matrix<T>(handle, std::int64_t{}, std::int64_t{});
  auto include_dataset = raft::deserialize_scalar<bool>(handle, is);
  if (include_dataset) {
    RAFT_EXPECTS(cuvs::util::is_mul_no_overflow(
                   static_cast<std::size_t>(rows), static_cast<std::size_t>(dim), sizeof(T)),
                 "brute_force::deserialize: integer overflow in rows*dim*sizeof(T) "
                 "(rows=%lld, dim=%lld, sizeof(T)=%zu)",
                 static_cast<long long>(rows),
                 static_cast<long long>(dim),
                 sizeof(T));
    dataset_storage = raft::make_device_matrix<T>(handle, rows, dim);
    cuvs::util::detail::deserialize_mdspan(handle, input, dataset_storage.view());
  }

  auto has_norms = raft::deserialize_scalar<bool>(handle, is);
  auto norms_storage =
    has_norms ? std::optional{raft::make_device_vector<DistT, std::int64_t>(handle, rows)}
              : std::optional<raft::device_vector<DistT, std::int64_t>>{};
  if (has_norms) { cuvs::util::detail::deserialize_mdspan(handle, input, norms_storage->view()); }

  auto result = index<T, DistT>(
    handle, std::move(dataset_storage), std::move(norms_storage), metric, metric_arg);
  raft::resource::sync_stream(handle);

  return result;
}

void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::brute_force::index<half, float>* index)
{
  cuvs::util::kvikio_file_reader reader(filename);
  *index = deserialize<half, float>(handle, reader);
}

void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::brute_force::index<float, float>* index)
{
  cuvs::util::kvikio_file_reader reader(filename);
  *index = deserialize<float, float>(handle, reader);
}

void deserialize(raft::resources const& handle,
                 std::istream& is,
                 cuvs::neighbors::brute_force::index<half, float>* index)
{
  *index = deserialize<half, float>(handle, is);
}

void deserialize(raft::resources const& handle,
                 std::istream& is,
                 cuvs::neighbors::brute_force::index<float, float>* index)
{
  *index = deserialize<float, float>(handle, is);
}

}  // namespace cuvs::neighbors::brute_force
