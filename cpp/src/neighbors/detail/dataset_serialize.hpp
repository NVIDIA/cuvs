/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/neighbors/common.hpp>
#include <cuvs/util/file_io.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/numpy_serializer.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/util/cudart_utils.hpp>

#include <raft/core/logger.hpp>

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <type_traits>

#include "../../util/kvikio_serialize.hpp"

namespace cuvs::neighbors::detail {

using dataset_instance_tag                              = uint32_t;
constexpr dataset_instance_tag kSerializeEmptyDataset   = 1;
constexpr dataset_instance_tag kSerializeStridedDataset = 2;
constexpr dataset_instance_tag kSerializeVPQDataset     = 3;

template <typename DataT>
void write_dense_bytes(std::ostream& os, DataT const* data, std::size_t elements)
{
  constexpr auto max_elements_per_write =
    static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(DataT);
  while (elements > 0) {
    auto const chunk = std::min(elements, max_elements_per_write);
    os.write(reinterpret_cast<char const*>(data),
             static_cast<std::streamsize>(chunk * sizeof(DataT)));
    RAFT_EXPECTS(os.good(), "serialize_dense_dataset: failed to write dataset payload");
    data += chunk;
    elements -= chunk;
  }
}

template <typename DataT>
void read_dense_bytes(std::istream& is, DataT* data, std::size_t elements)
{
  constexpr auto max_elements_per_read =
    static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(DataT);
  while (elements > 0) {
    auto const chunk = std::min(elements, max_elements_per_read);
    is.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(chunk * sizeof(DataT)));
    RAFT_EXPECTS(is.good(), "deserialize_dense_dataset: invalid or truncated dataset payload");
    data += chunk;
    elements -= chunk;
  }
}

template <typename DataT, typename IdxT>
auto dense_matrix_elements(IdxT n_rows, uint32_t dim, char const* context) -> std::size_t
{
  if constexpr (std::is_signed_v<IdxT>) {
    RAFT_EXPECTS(n_rows >= 0, "%s: row count must not be negative", context);
  }
  auto const rows = static_cast<std::size_t>(n_rows);
  RAFT_EXPECTS(rows == 0 || dim <= std::numeric_limits<std::size_t>::max() / rows,
               "%s: element count overflow",
               context);
  auto const elements = rows * static_cast<std::size_t>(dim);
  RAFT_EXPECTS(elements <= std::numeric_limits<std::size_t>::max() / sizeof(DataT),
               "%s: byte count overflow",
               context);
  return elements;
}

template <typename DataT, typename IdxT, typename ViewT>
  requires cuvs::neighbors::is_dense_row_major_dataset_view_v<ViewT>
void serialize(const raft::resources& res, std::ostream& os, ViewT const& dataset)
{
  auto n_rows = dataset.n_rows();
  auto dim    = dataset.dim();
  auto stride = dataset.stride();
  raft::serialize_scalar(res, os, n_rows);
  raft::serialize_scalar(res, os, dim);
  raft::serialize_scalar(res, os, stride);
  auto src            = dataset.view();
  auto const elements = dense_matrix_elements<DataT>(n_rows, dim, "serialize_dense_dataset");
  raft::numpy_serializer::write_header(os,
                                       {raft::numpy_serializer::get_numpy_dtype<DataT>(),
                                        false,
                                        {static_cast<raft::numpy_serializer::ndarray_len_t>(n_rows),
                                         static_cast<raft::numpy_serializer::ndarray_len_t>(dim)}});
  if (elements == 0) { return; }

  if constexpr (cuvs::neighbors::is_device_dataset_view_v<ViewT>) {
    if (auto* kvikio_stream = dynamic_cast<cuvs::util::kvikio_ofstream*>(&os);
        kvikio_stream != nullptr) {
      auto const row_bytes = static_cast<std::size_t>(dim) * sizeof(DataT);
      if (stride == dim) {
        raft::resource::sync_stream(res);
        kvikio_stream->write_device(src.data_handle(), elements * sizeof(DataT));
        return;
      }

      auto const batch_rows = static_cast<IdxT>(std::min<std::size_t>(
        static_cast<std::size_t>(n_rows),
        std::max<std::size_t>(1, cuvs::util::detail::kDeviceSerializationBatchBytes / row_bytes)));
      auto packed           = raft::make_device_matrix<DataT, IdxT>(res, batch_rows, dim);
      auto const stream     = raft::resource::get_cuda_stream(res);
      for (IdxT first_row = 0; first_row < n_rows; first_row += batch_rows) {
        auto const rows = std::min<IdxT>(batch_rows, n_rows - first_row);
        raft::copy_matrix(packed.data_handle(),
                          dim,
                          src.data_handle() + first_row * stride,
                          stride,
                          dim,
                          rows,
                          stream);
        raft::resource::sync_stream(res);
        kvikio_stream->write_device(packed.data_handle(),
                                    static_cast<std::size_t>(rows) * row_bytes);
      }
      return;
    }

    auto staging = raft::make_host_matrix<DataT, IdxT>(n_rows, dim);
    raft::copy_matrix(staging.data_handle(),
                      dim,
                      src.data_handle(),
                      stride,
                      dim,
                      n_rows,
                      raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    write_dense_bytes(os, staging.data_handle(), elements);
  } else {
    if (stride == dim) {
      write_dense_bytes(os, src.data_handle(), elements);
    } else {
      for (IdxT row = 0; row < n_rows; ++row) {
        write_dense_bytes(os, src.data_handle() + row * stride, dim);
      }
    }
  }
}

/** Write CAGRA index dataset blob (tag + element dtype + strided payload). */
template <typename DataT, typename IdxT, typename ViewT>
  requires cuvs::neighbors::is_dense_row_major_dataset_view_v<ViewT>
void serialize_cagra_dense_dataset(const raft::resources& res,
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

/** Read and validate shared dense wire metadata and the tight `[n_rows x dim]` NumPy header. */
template <typename IdxT>
struct dense_payload_metadata {
  IdxT n_rows;
  uint32_t dim;
  uint32_t stride;
  std::size_t elements;
};

class device_deserialize_event {
 public:
  device_deserialize_event()
  {
    RAFT_CUDA_TRY(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
  }

  ~device_deserialize_event() { RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(event_)); }

  device_deserialize_event(const device_deserialize_event&)            = delete;
  device_deserialize_event& operator=(const device_deserialize_event&) = delete;

  void record(cudaStream_t stream) { RAFT_CUDA_TRY(cudaEventRecord(event_, stream)); }
  void wait() { RAFT_CUDA_TRY(cudaEventSynchronize(event_)); }

 private:
  cudaEvent_t event_{};
};

template <typename DataT, typename IdxT>
auto deserialize_dense_payload_metadata(raft::resources const& res, std::istream& is)
  -> dense_payload_metadata<IdxT>
{
  auto n_rows = raft::deserialize_scalar<IdxT>(res, is);
  auto dim    = raft::deserialize_scalar<uint32_t>(res, is);
  auto stride = raft::deserialize_scalar<uint32_t>(res, is);
  RAFT_EXPECTS(dim <= stride,
               "deserialize_dense_payload: logical dim (%u) must not exceed row stride (%u).",
               static_cast<unsigned>(dim),
               static_cast<unsigned>(stride));
  auto const elements = dense_matrix_elements<DataT>(n_rows, dim, "deserialize_dense_payload");
  static_cast<void>(dense_matrix_elements<DataT>(n_rows, stride, "deserialize_dense_storage"));

  auto const header         = raft::numpy_serializer::read_header(is);
  auto const expected_dtype = raft::numpy_serializer::get_numpy_dtype<DataT>();
  RAFT_EXPECTS(header.dtype == expected_dtype,
               "deserialize_dense_payload: expected dtype %s but got %s",
               expected_dtype.to_string().c_str(),
               header.dtype.to_string().c_str());
  RAFT_EXPECTS(!header.fortran_order, "deserialize_dense_payload: expected row-major payload");
  RAFT_EXPECTS(header.shape.size() == 2,
               "deserialize_dense_payload: expected rank 2 but got %zu",
               header.shape.size());
  RAFT_EXPECTS(header.shape[0] == static_cast<raft::numpy_serializer::ndarray_len_t>(n_rows) &&
                 header.shape[1] == static_cast<raft::numpy_serializer::ndarray_len_t>(dim),
               "deserialize_dense_payload: payload shape does not match serialized dimensions");

  return {n_rows, dim, stride, elements};
}

template <typename DataT, typename IdxT>
void skip_dense_payload(raft::resources const& res, std::istream& is)
{
  auto const metadata = deserialize_dense_payload_metadata<DataT, IdxT>(res, is);
  RAFT_EXPECTS(metadata.elements <= std::numeric_limits<std::size_t>::max() / sizeof(DataT),
               "skip_dense_payload: byte count overflow");
  auto remaining = metadata.elements * sizeof(DataT);

  using pos_type              = std::istream::pos_type;
  using off_type              = std::istream::off_type;
  auto* buffer                = is.rdbuf();
  auto const invalid_position = pos_type{off_type{-1}};
  auto const current          = buffer->pubseekoff(0, std::ios_base::cur, std::ios_base::in);
  if (current != invalid_position &&
      remaining <= static_cast<std::size_t>(std::numeric_limits<off_type>::max())) {
    auto const end = buffer->pubseekoff(0, std::ios_base::end, std::ios_base::in);
    if (end != invalid_position) {
      auto const available = end - current;
      RAFT_EXPECTS(available >= 0 && static_cast<std::size_t>(available) >= remaining,
                   "skip_dense_payload: truncated payload");
      auto const next =
        buffer->pubseekpos(current + static_cast<off_type>(remaining), std::ios_base::in);
      RAFT_EXPECTS(next != invalid_position, "skip_dense_payload: failed to seek past payload");
      return;
    }
    RAFT_EXPECTS(buffer->pubseekpos(current, std::ios_base::in) != invalid_position,
                 "skip_dense_payload: failed to restore stream position");
  }

  std::array<char, 64 * 1024> discard_buffer{};
  while (remaining > 0) {
    auto const chunk = std::min<std::size_t>(remaining, discard_buffer.size());
    is.read(discard_buffer.data(), static_cast<std::streamsize>(chunk));
    RAFT_EXPECTS(static_cast<std::size_t>(is.gcount()) == chunk,
                 "skip_dense_payload: truncated payload");
    remaining -= chunk;
  }
}

template <typename DataT, typename IdxT, typename OwningDatasetT>
auto deserialize_device_dense(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<OwningDatasetT>
{
  auto const metadata = deserialize_dense_payload_metadata<DataT, IdxT>(res, is);
  auto staging        = raft::make_host_matrix<DataT, IdxT>(metadata.n_rows, metadata.dim);
  read_dense_bytes(is, staging.data_handle(), metadata.elements);

  auto storage     = raft::make_device_matrix<DataT, IdxT>(res, metadata.n_rows, metadata.stride);
  auto stream      = raft::resource::get_cuda_stream(res);
  bool work_queued = false;
  if (metadata.stride > metadata.dim && metadata.n_rows > 0) {
    RAFT_CUDA_TRY(cudaMemset2DAsync(storage.data_handle() + metadata.dim,
                                    metadata.stride * sizeof(DataT),
                                    0,
                                    (metadata.stride - metadata.dim) * sizeof(DataT),
                                    metadata.n_rows,
                                    stream));
    work_queued = true;
  }
  if (metadata.elements > 0) {
    raft::copy_matrix(storage.data_handle(),
                      metadata.stride,
                      staging.data_handle(),
                      metadata.dim,
                      metadata.dim,
                      metadata.n_rows,
                      stream);
    work_queued = true;
  }
  if (work_queued) { raft::resource::sync_stream(res); }
  return std::make_unique<OwningDatasetT>(std::move(storage), metadata.dim);
}

template <typename DataT, typename IdxT, typename OwningDatasetT>
auto deserialize_device_dense(raft::resources const& res, cuvs::util::kvikio_file_reader& reader)
  -> std::unique_ptr<OwningDatasetT>
{
  auto const metadata = deserialize_dense_payload_metadata<DataT, IdxT>(res, reader.stream());
  auto storage      = raft::make_device_matrix<DataT, IdxT>(res, metadata.n_rows, metadata.stride);
  auto const stream = raft::resource::get_cuda_stream(res);

  if (metadata.stride > metadata.dim && metadata.n_rows > 0) {
    RAFT_CUDA_TRY(cudaMemset2DAsync(storage.data_handle() + metadata.dim,
                                    metadata.stride * sizeof(DataT),
                                    0,
                                    (metadata.stride - metadata.dim) * sizeof(DataT),
                                    metadata.n_rows,
                                    stream));
  }
  raft::resource::sync_stream(res);

  if (metadata.elements == 0) {
    return std::make_unique<OwningDatasetT>(std::move(storage), metadata.dim);
  }

  auto const row_bytes = static_cast<std::size_t>(metadata.dim) * sizeof(DataT);
  if (metadata.stride == metadata.dim) {
    reader.read_device(storage.data_handle(), metadata.elements * sizeof(DataT));
  } else {
    auto const batch_rows = static_cast<IdxT>(std::min<std::size_t>(
      static_cast<std::size_t>(metadata.n_rows),
      std::max<std::size_t>(1, cuvs::util::detail::kDeviceSerializationBatchBytes / row_bytes)));
    auto packed0          = raft::make_device_matrix<DataT, IdxT>(res, batch_rows, metadata.dim);
    if (batch_rows == metadata.n_rows) {
      raft::resource::sync_stream(res);
      reader.read_device(packed0.data_handle(), metadata.elements * sizeof(DataT));
      raft::copy_matrix(storage.data_handle(),
                        metadata.stride,
                        packed0.data_handle(),
                        metadata.dim,
                        metadata.dim,
                        metadata.n_rows,
                        stream);
      raft::resource::sync_stream(res);
      return std::make_unique<OwningDatasetT>(std::move(storage), metadata.dim);
    }

    auto packed1 = raft::make_device_matrix<DataT, IdxT>(res, batch_rows, metadata.dim);
    raft::resource::sync_stream(res);

    // KvikIO reads block the host but run outside the RAFT stream. Alternate buffers so the next
    // read can overlap the previous batch's device-to-device scatter.
    std::array<DataT*, 2> packed_buffers{packed0.data_handle(), packed1.data_handle()};
    std::array<device_deserialize_event, 2> copy_complete;
    std::array<bool, 2> copy_in_flight{};
    std::size_t batch_index = 0;
    try {
      for (IdxT first_row = 0; first_row < metadata.n_rows;
           first_row += batch_rows, ++batch_index) {
        const std::size_t slot = batch_index % packed_buffers.size();
        if (copy_in_flight[slot]) { copy_complete[slot].wait(); }

        auto const rows = std::min<IdxT>(batch_rows, metadata.n_rows - first_row);
        reader.read_device(packed_buffers[slot], static_cast<std::size_t>(rows) * row_bytes);
        raft::copy_matrix(storage.data_handle() + first_row * metadata.stride,
                          metadata.stride,
                          packed_buffers[slot],
                          metadata.dim,
                          metadata.dim,
                          rows,
                          stream);
        copy_complete[slot].record(stream);
        copy_in_flight[slot] = true;
      }
      raft::resource::sync_stream(res);
    } catch (...) {
      RAFT_CUDA_TRY_NO_THROW(cudaStreamSynchronize(stream));
      throw;
    }
  }
  return std::make_unique<OwningDatasetT>(std::move(storage), metadata.dim);
}

template <typename DataT, typename IdxT, typename OwningDatasetT>
auto deserialize_host_dense(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<OwningDatasetT>
{
  auto const metadata = deserialize_dense_payload_metadata<DataT, IdxT>(res, is);
  auto storage        = raft::make_host_matrix<DataT, IdxT>(metadata.n_rows, metadata.stride);
  if (metadata.stride == metadata.dim) {
    read_dense_bytes(is, storage.data_handle(), metadata.elements);
  } else {
    for (IdxT row = 0; row < metadata.n_rows; ++row) {
      auto* row_data = storage.data_handle() + row * metadata.stride;
      read_dense_bytes(is, row_data, metadata.dim);
      std::memset(row_data + metadata.dim, 0, (metadata.stride - metadata.dim) * sizeof(DataT));
    }
  }
  return std::make_unique<OwningDatasetT>(std::move(storage), metadata.dim);
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

template <typename DataT, typename IdxT, typename OwningDatasetT, typename Input>
auto deserialize_dense_dataset(raft::resources const& res, Input& input)
  -> std::unique_ptr<OwningDatasetT>
{
  auto& is       = cuvs::util::detail::input_stream(input);
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
    return deserialize_device_dense<DataT, IdxT, OwningDatasetT>(res, input);
  } else if constexpr (std::is_same_v<OwningDatasetT, device_standard_dataset<DataT, IdxT>>) {
    return deserialize_device_dense<DataT, IdxT, OwningDatasetT>(res, input);
  } else if constexpr (std::is_same_v<OwningDatasetT, host_padded_dataset<DataT, IdxT>>) {
    return deserialize_host_dense<DataT, IdxT, OwningDatasetT>(res, is);
  } else if constexpr (std::is_same_v<OwningDatasetT, host_standard_dataset<DataT, IdxT>>) {
    return deserialize_host_dense<DataT, IdxT, OwningDatasetT>(res, is);
  } else {
    static_assert(!std::is_same_v<OwningDatasetT, OwningDatasetT>,
                  "deserialize_dense_dataset: unsupported owning dataset type");
  }
}

template <typename DataT, typename IdxT>
void skip_dense_dataset(raft::resources const& res, std::istream& is)
{
  const auto tag = raft::deserialize_scalar<dataset_instance_tag>(res, is);
  RAFT_EXPECTS(tag == kSerializeStridedDataset,
               "skip_dense_dataset: expected strided tag, got %u",
               static_cast<unsigned>(tag));
  const auto dtype                        = raft::deserialize_scalar<cudaDataType_t>(res, is);
  constexpr cudaDataType_t expected_dtype = std::is_same_v<DataT, float>    ? CUDA_R_32F
                                            : std::is_same_v<DataT, half>   ? CUDA_R_16F
                                            : std::is_same_v<DataT, int8_t> ? CUDA_R_8I
                                                                            : CUDA_R_8U;
  RAFT_EXPECTS(dtype == expected_dtype,
               "skip_dense_dataset: serialized dtype (%d) does not match expected (%d)",
               static_cast<int>(dtype),
               static_cast<int>(expected_dtype));
  skip_dense_payload<DataT, IdxT>(res, is);
}

// Reads tag + dtype prefix, validates they match DataT, and returns the requested concrete
// dense owner. When a new dataset kind is supported, add a matching overload of
// deserialize_dataset here rather than extending this one — overload dispatch replaces the old
// type-erased variant routing.
template <typename DataT, typename IdxT, typename Input>
auto deserialize_padded_dataset(raft::resources const& res, Input& input)
  -> std::unique_ptr<device_padded_dataset<DataT, IdxT>>
{
  return deserialize_dense_dataset<DataT, IdxT, device_padded_dataset<DataT, IdxT>>(res, input);
}

template <typename DataT, typename IdxT, typename Input>
auto deserialize_standard_dataset(raft::resources const& res, Input& input)
  -> std::unique_ptr<device_standard_dataset<DataT, IdxT>>
{
  return deserialize_dense_dataset<DataT, IdxT, device_standard_dataset<DataT, IdxT>>(res, input);
}

template <typename DataT, typename IdxT, typename Input>
auto deserialize_host_padded_dataset(raft::resources const& res, Input& input)
  -> std::unique_ptr<host_padded_dataset<DataT, IdxT>>
{
  return deserialize_dense_dataset<DataT, IdxT, host_padded_dataset<DataT, IdxT>>(res, input);
}

template <typename DataT, typename IdxT, typename Input>
auto deserialize_host_standard_dataset(raft::resources const& res, Input& input)
  -> std::unique_ptr<host_standard_dataset<DataT, IdxT>>
{
  return deserialize_dense_dataset<DataT, IdxT, host_standard_dataset<DataT, IdxT>>(res, input);
}

}  // namespace cuvs::neighbors::detail
