/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/dataset_view_concepts.hpp>

#include <raft/core/host_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/util/cudart_utils.hpp>

#include <raft/core/logger.hpp>

#include <cuda_fp16.h>

#include <fstream>
#include <memory>

namespace cuvs::neighbors::detail {

using dataset_instance_tag                              = uint32_t;
constexpr dataset_instance_tag kSerializeEmptyDataset   = 1;
constexpr dataset_instance_tag kSerializeStridedDataset = 2;
constexpr dataset_instance_tag kSerializeVPQDataset     = 3;

template <typename DataT, typename IdxT, typename ViewT>
  requires cuvs::neighbors::is_dense_row_major_device_dataset_view_v<ViewT>
void serialize(const raft::resources& res, std::ostream& os, ViewT const& dataset)
{
  auto n_rows = dataset.n_rows();
  auto dim    = dataset.dim();
  auto stride = dataset.stride();
  raft::serialize_scalar(res, os, n_rows);
  raft::serialize_scalar(res, os, dim);
  raft::serialize_scalar(res, os, stride);
  auto src = dataset.view();
  auto dst = raft::make_host_matrix<DataT, IdxT>(n_rows, dim);
  raft::copy_matrix(dst.data_handle(),
                    dim,
                    src.data_handle(),
                    stride,
                    dim,
                    n_rows,
                    raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  raft::serialize_mdspan(res, os, dst.view());
}

/** Write CAGRA index dataset blob (tag + element dtype + strided payload). */
template <typename DataT, typename IdxT, typename ViewT>
  requires cuvs::neighbors::is_dense_row_major_device_dataset_view_v<ViewT>
void serialize_cagra_padded_dataset(const raft::resources& res,
                                    std::ostream& os,
                                    ViewT const& dataset)
{
  raft::serialize_scalar(res, os, kSerializeStridedDataset);
  if constexpr (std::is_same_v<DataT, float>) {
    raft::serialize_scalar(res, os, CUDA_R_32F);
  } else if constexpr (std::is_same_v<DataT, half>) {
    raft::serialize_scalar(res, os, CUDA_R_16F);
  } else if constexpr (std::is_same_v<DataT, int8_t>) {
    raft::serialize_scalar(res, os, CUDA_R_8I);
  } else if constexpr (std::is_same_v<DataT, uint8_t>) {
    raft::serialize_scalar(res, os, CUDA_R_8U);
  } else {
    static_assert(!std::is_same_v<DataT, DataT>, "unsupported element type for CAGRA serialize");
  }
  serialize<DataT, IdxT>(res, os, dataset);
}

template <typename IdxT>
auto deserialize_empty(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<device_empty_dataset<IdxT>>
{
  auto suggested_dim = raft::deserialize_scalar<uint32_t>(res, is);
  return std::make_unique<device_empty_dataset<IdxT>>(suggested_dim);
}

template <typename DataT, typename IdxT>
auto deserialize_padded(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<device_padded_dataset<DataT, IdxT>>
{
  auto n_rows = raft::deserialize_scalar<IdxT>(res, is);
  auto dim    = raft::deserialize_scalar<uint32_t>(res, is);
  auto stride = raft::deserialize_scalar<uint32_t>(res, is);
  RAFT_EXPECTS(dim <= stride,
               "deserialize_padded: logical dim (%u) must not exceed row stride (%u).",
               static_cast<unsigned>(dim),
               static_cast<unsigned>(stride));
  auto host_array = raft::make_host_matrix<DataT, IdxT>(n_rows, dim);
  raft::deserialize_mdspan(res, is, host_array.view());
  return cuvs::neighbors::make_device_padded_dataset(res, host_array.view());
}

template <typename DataT, typename IdxT>
auto deserialize_standard(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<device_standard_dataset<DataT, IdxT>>
{
  auto n_rows = raft::deserialize_scalar<IdxT>(res, is);
  auto dim    = raft::deserialize_scalar<uint32_t>(res, is);
  auto stride = raft::deserialize_scalar<uint32_t>(res, is);
  RAFT_EXPECTS(dim <= stride,
               "deserialize_standard: logical dim (%u) must not exceed row stride (%u).",
               static_cast<unsigned>(dim),
               static_cast<unsigned>(stride));
  auto host_array = raft::make_host_matrix<DataT, IdxT>(n_rows, dim);
  raft::deserialize_mdspan(res, is, host_array.view());
  auto device_array = raft::make_device_matrix<DataT, IdxT>(res, static_cast<IdxT>(n_rows), stride);
  if (stride > dim) {
    RAFT_CUDA_TRY(cudaMemsetAsync(device_array.data_handle(),
                                  0,
                                  device_array.size() * sizeof(DataT),
                                  raft::resource::get_cuda_stream(res)));
  }
  raft::copy_matrix(device_array.data_handle(),
                    stride,
                    host_array.data_handle(),
                    dim,
                    dim,
                    static_cast<IdxT>(n_rows),
                    raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  return std::make_unique<device_standard_dataset<DataT, IdxT>>(std::move(device_array), dim);
}

template <typename DataT, typename IdxT>
auto deserialize_vpq(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<device_vpq_dataset<DataT, IdxT>>
{
  auto n_rows             = raft::deserialize_scalar<IdxT>(res, is);
  auto dim                = raft::deserialize_scalar<uint32_t>(res, is);
  auto vq_n_centers       = raft::deserialize_scalar<uint32_t>(res, is);
  auto pq_n_centers       = raft::deserialize_scalar<uint32_t>(res, is);
  auto pq_len             = raft::deserialize_scalar<uint32_t>(res, is);
  auto encoded_row_length = raft::deserialize_scalar<uint32_t>(res, is);

  auto vq_code_book =
    raft::make_device_matrix<DataT, uint32_t, raft::row_major>(res, vq_n_centers, dim);
  auto pq_code_book =
    raft::make_device_matrix<DataT, uint32_t, raft::row_major>(res, pq_n_centers, pq_len);
  auto data =
    raft::make_device_matrix<uint8_t, IdxT, raft::row_major>(res, n_rows, encoded_row_length);

  raft::deserialize_mdspan(res, is, vq_code_book.view());
  raft::deserialize_mdspan(res, is, pq_code_book.view());
  raft::deserialize_mdspan(res, is, data.view());

  return std::make_unique<device_vpq_dataset<DataT, IdxT>>(
    std::move(vq_code_book), std::move(pq_code_book), std::move(data));
}

template <typename DataT, typename IdxT, typename OwningDatasetT>
auto deserialize_dense_dataset(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<OwningDatasetT>
{
  const auto tag = raft::deserialize_scalar<dataset_instance_tag>(res, is);
  RAFT_EXPECTS(tag == kSerializeStridedDataset,
               "deserialize_dataset: expected strided tag, got %u",
               static_cast<unsigned>(tag));
  const auto dtype                        = raft::deserialize_scalar<cudaDataType_t>(res, is);
  constexpr cudaDataType_t expected_dtype = std::is_same_v<DataT, float>    ? CUDA_R_32F
                                            : std::is_same_v<DataT, half>   ? CUDA_R_16F
                                            : std::is_same_v<DataT, int8_t> ? CUDA_R_8I
                                                                            : CUDA_R_8U;  // uint8_t
  RAFT_EXPECTS(dtype == expected_dtype,
               "deserialize_dataset: serialized dtype (%d) does not match expected (%d)",
               static_cast<int>(dtype),
               static_cast<int>(expected_dtype));
  if constexpr (std::is_same_v<OwningDatasetT, device_padded_dataset<DataT, IdxT>>) {
    return deserialize_padded<DataT, IdxT>(res, is);
  } else if constexpr (std::is_same_v<OwningDatasetT, device_standard_dataset<DataT, IdxT>>) {
    return deserialize_standard<DataT, IdxT>(res, is);
  } else {
    static_assert(!std::is_same_v<OwningDatasetT, OwningDatasetT>,
                  "deserialize_dense_dataset: unsupported owning dataset type");
  }
}

// Reads tag + dtype prefix, validates they match DataT, and returns a concrete
// device_padded_dataset. When a new dataset kind is supported, add a matching overload of
// deserialize_dataset here rather than extending this one — overload dispatch replaces the old
// type-erased variant routing.
template <typename DataT, typename IdxT>
auto deserialize_dataset(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<device_padded_dataset<DataT, IdxT>>
{
  return deserialize_dense_dataset<DataT, IdxT, device_padded_dataset<DataT, IdxT>>(res, is);
}

template <typename DataT, typename IdxT>
auto deserialize_standard_dataset(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<device_standard_dataset<DataT, IdxT>>
{
  return deserialize_dense_dataset<DataT, IdxT, device_standard_dataset<DataT, IdxT>>(res, is);
}

}  // namespace cuvs::neighbors::detail
