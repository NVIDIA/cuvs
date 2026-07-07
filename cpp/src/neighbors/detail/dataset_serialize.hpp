/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../preprocessing/quantize/detail/vpq_dataset_impl.hpp"
#include <cuvs/neighbors/common.hpp>
#include <cuvs/preprocessing/quantize/vpq_dataset.hpp>

#include <raft/core/host_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>
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

template <typename IdxT>
void serialize(const raft::resources& res, std::ostream& os, const empty_dataset<IdxT>& dataset)
{
  raft::serialize_scalar(res, os, dataset.suggested_dim);
}

template <typename DataT, typename IdxT>
void serialize(const raft::resources& res,
               std::ostream& os,
               const strided_dataset<DataT, IdxT>& dataset)
{
  auto n_rows = dataset.n_rows();
  auto dim    = dataset.dim();
  auto stride = dataset.stride();
  raft::serialize_scalar(res, os, n_rows);
  raft::serialize_scalar(res, os, dim);
  raft::serialize_scalar(res, os, stride);
  // Remove padding before saving the dataset
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

template <typename MathT, typename IdxT>
void serialize(const raft::resources& res,
               std::ostream& os,
               const cuvs::preprocessing::quantize::pq::vpq_dataset<MathT, IdxT>& dataset)
{
  raft::serialize_scalar(res, os, dataset.n_rows());
  raft::serialize_scalar(res, os, dataset.dim());
  raft::serialize_scalar(res, os, dataset.vq_n_centers());
  raft::serialize_scalar(res, os, dataset.pq_n_centers());
  raft::serialize_scalar(res, os, dataset.pq_len());
  raft::serialize_scalar(res, os, dataset.encoded_row_length());
  auto vq_view = dataset.vq_code_book().value_or(
    raft::make_device_matrix_view<const MathT, uint32_t, raft::row_major>(nullptr, 0, 0));
  raft::serialize_mdspan(res, os, vq_view);
  raft::serialize_mdspan(res, os, dataset.pq_code_book());
  raft::serialize_mdspan(res, os, dataset.data.view());
}

template <typename IdxT>
void serialize(const raft::resources& res, std::ostream& os, const dataset<IdxT>& dataset)
{
  if (auto x = dynamic_cast<const empty_dataset<IdxT>*>(&dataset); x != nullptr) {
    raft::serialize_scalar(res, os, kSerializeEmptyDataset);
    return serialize(res, os, *x);
  }
  if (auto x = dynamic_cast<const strided_dataset<float, IdxT>*>(&dataset); x != nullptr) {
    raft::serialize_scalar(res, os, kSerializeStridedDataset);
    raft::serialize_scalar(res, os, CUDA_R_32F);
    return serialize(res, os, *x);
  }
  if (auto x = dynamic_cast<const strided_dataset<half, IdxT>*>(&dataset); x != nullptr) {
    raft::serialize_scalar(res, os, kSerializeStridedDataset);
    raft::serialize_scalar(res, os, CUDA_R_16F);
    return serialize(res, os, *x);
  }
  if (auto x = dynamic_cast<const strided_dataset<int8_t, IdxT>*>(&dataset); x != nullptr) {
    raft::serialize_scalar(res, os, kSerializeStridedDataset);
    raft::serialize_scalar(res, os, CUDA_R_8I);
    return serialize(res, os, *x);
  }
  if (auto x = dynamic_cast<const strided_dataset<uint8_t, IdxT>*>(&dataset); x != nullptr) {
    raft::serialize_scalar(res, os, kSerializeStridedDataset);
    raft::serialize_scalar(res, os, CUDA_R_8U);
    return serialize(res, os, *x);
  }
  if (auto x =
        dynamic_cast<const cuvs::preprocessing::quantize::pq::vpq_dataset<float, IdxT>*>(&dataset);
      x != nullptr) {
    raft::serialize_scalar(res, os, kSerializeVPQDataset);
    raft::serialize_scalar(res, os, CUDA_R_32F);
    return serialize(res, os, *x);
  }
  if (auto x =
        dynamic_cast<const cuvs::preprocessing::quantize::pq::vpq_dataset<half, IdxT>*>(&dataset);
      x != nullptr) {
    raft::serialize_scalar(res, os, kSerializeVPQDataset);
    raft::serialize_scalar(res, os, CUDA_R_16F);
    return serialize(res, os, *x);
  }
  RAFT_FAIL("unsupported dataset type.");
}

template <typename IdxT>
auto deserialize_empty(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<empty_dataset<IdxT>>
{
  auto suggested_dim = raft::deserialize_scalar<uint32_t>(res, is);
  return std::make_unique<empty_dataset<IdxT>>(suggested_dim);
}

template <typename DataT, typename IdxT>
auto deserialize_strided(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<strided_dataset<DataT, IdxT>>
{
  auto n_rows     = raft::deserialize_scalar<IdxT>(res, is);
  auto dim        = raft::deserialize_scalar<uint32_t>(res, is);
  auto stride     = raft::deserialize_scalar<uint32_t>(res, is);
  auto host_array = raft::make_host_matrix<DataT, IdxT>(n_rows, dim);
  raft::deserialize_mdspan(res, is, host_array.view());
  return make_strided_dataset(res, std::move(host_array), stride);
}

template <typename MathT, typename IdxT>
auto deserialize_vpq(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<cuvs::preprocessing::quantize::pq::vpq_dataset<MathT, IdxT>>
{
  auto n_rows             = raft::deserialize_scalar<IdxT>(res, is);
  auto dim                = raft::deserialize_scalar<uint32_t>(res, is);
  auto vq_n_centers       = raft::deserialize_scalar<uint32_t>(res, is);
  auto pq_n_centers       = raft::deserialize_scalar<uint32_t>(res, is);
  auto pq_len             = raft::deserialize_scalar<uint32_t>(res, is);
  auto encoded_row_length = raft::deserialize_scalar<uint32_t>(res, is);

  auto vq_code_book =
    raft::make_device_matrix<MathT, uint32_t, raft::row_major>(res, vq_n_centers, dim);
  auto pq_code_book =
    raft::make_device_matrix<MathT, uint32_t, raft::row_major>(res, pq_n_centers, pq_len);
  auto data =
    raft::make_device_matrix<uint8_t, IdxT, raft::row_major>(res, n_rows, encoded_row_length);

  raft::deserialize_mdspan(res, is, vq_code_book.view());
  raft::deserialize_mdspan(res, is, pq_code_book.view());
  raft::deserialize_mdspan(res, is, data.view());

  std::optional<raft::device_matrix<MathT, uint32_t, raft::row_major>> vq_code_book_opt;
  if (vq_n_centers > 0) { vq_code_book_opt = std::move(vq_code_book); }
  return std::make_unique<cuvs::preprocessing::quantize::pq::vpq_dataset<MathT, IdxT>>(
    cuvs::preprocessing::quantize::pq::vpq_codebooks<MathT>{
      std::make_unique<cuvs::preprocessing::quantize::pq::vpq_codebooks_owning<MathT>>(
        std::move(pq_code_book), std::move(vq_code_book_opt))},
    std::move(data));
}

template <typename IdxT>
auto deserialize_dataset(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<dataset<IdxT>>
{
  const auto tag = raft::deserialize_scalar<dataset_instance_tag>(res, is);
  switch (tag) {
    case kSerializeEmptyDataset: return deserialize_empty<IdxT>(res, is);
    case kSerializeStridedDataset: {
      const auto dtype = raft::deserialize_scalar<cudaDataType_t>(res, is);
      switch (dtype) {
        case CUDA_R_32F: return deserialize_strided<float, IdxT>(res, is);
        case CUDA_R_16F: return deserialize_strided<half, IdxT>(res, is);
        case CUDA_R_8I: return deserialize_strided<int8_t, IdxT>(res, is);
        case CUDA_R_8U: return deserialize_strided<uint8_t, IdxT>(res, is);
        default:
          RAFT_FAIL("Failed to deserialize dataset: unsupported strided dataset element type %d.",
                    static_cast<int>(dtype));
      }
    }
    case kSerializeVPQDataset: {
      const auto dtype = raft::deserialize_scalar<cudaDataType_t>(res, is);
      switch (dtype) {
        case CUDA_R_32F: return deserialize_vpq<float, IdxT>(res, is);
        case CUDA_R_16F: return deserialize_vpq<half, IdxT>(res, is);
        default:
          RAFT_FAIL("Failed to deserialize dataset: unsupported VPQ dtype %d.",
                    static_cast<int>(dtype));
      }
    }
    default:
      RAFT_FAIL("Failed to deserialize dataset: unknown instance tag %u.",
                static_cast<unsigned>(tag));
  }
}

}  // namespace cuvs::neighbors::detail
