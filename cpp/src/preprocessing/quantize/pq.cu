/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../neighbors/detail/dataset_serialize.hpp"
#include "../../util/serialize_validation.hpp"
#include "./detail/pq.cuh"

#include <cuvs/preprocessing/quantize/pq.hpp>

#include <raft/core/numpy_serializer.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/util/cudart_utils.hpp>

#include <fstream>
#include <memory>
#include <string>

namespace cuvs::preprocessing::quantize::pq {

#define CUVS_INST_QUANTIZATION(T, QuantI)                                              \
  auto build(raft::resources const& res,                                               \
             const params params,                                                      \
             raft::device_matrix_view<const T, int64_t> dataset) -> quantizer<T>       \
  {                                                                                    \
    return detail::build<T, T>(res, params, dataset);                                  \
  }                                                                                    \
  auto build(raft::resources const& res,                                               \
             const params params,                                                      \
             raft::host_matrix_view<const T, int64_t> dataset) -> quantizer<T>         \
  {                                                                                    \
    return detail::build<T, T>(res, params, dataset);                                  \
  }                                                                                    \
  void transform(raft::resources const& res,                                           \
                 const quantizer<T>& quantizer,                                        \
                 raft::device_matrix_view<const T, int64_t> dataset,                   \
                 raft::device_matrix_view<QuantI, int64_t> codes_out,                  \
                 std::optional<raft::device_vector_view<uint32_t, int64_t>> vq_labels) \
  {                                                                                    \
    detail::transform(res, quantizer, dataset, codes_out, vq_labels);                  \
  }                                                                                    \
  void transform(raft::resources const& res,                                           \
                 const quantizer<T>& quantizer,                                        \
                 raft::host_matrix_view<const T, int64_t> dataset,                     \
                 raft::device_matrix_view<QuantI, int64_t> codes_out,                  \
                 std::optional<raft::device_vector_view<uint32_t, int64_t>> vq_labels) \
  {                                                                                    \
    detail::transform(res, quantizer, dataset, codes_out, vq_labels);                  \
  }                                                                                    \
  void inverse_transform(                                                              \
    raft::resources const& res,                                                        \
    const quantizer<T>& quantizer,                                                     \
    raft::device_matrix_view<const QuantI, int64_t> pq_codes,                          \
    raft::device_matrix_view<T, int64_t> out,                                          \
    std::optional<raft::device_vector_view<const uint32_t, int64_t>> vq_labels)        \
  {                                                                                    \
    detail::inverse_transform(res, quantizer, pq_codes, out, vq_labels);               \
  }

CUVS_INST_QUANTIZATION(float, uint8_t);

#undef CUVS_INST_QUANTIZATION

#define CUVS_INST_VPQ_BUILD(T)                                                               \
  auto vpq_build(const raft::resources& res,                                                 \
                 const cuvs::neighbors::vpq_params& params,                                  \
                 const raft::host_matrix_view<const T, int64_t, raft::row_major>& dataset)   \
  {                                                                                          \
    return detail::vpq_build_half<decltype(dataset)>(res, params, dataset);                  \
  }                                                                                          \
  auto vpq_build(const raft::resources& res,                                                 \
                 const cuvs::neighbors::vpq_params& params,                                  \
                 const raft::device_matrix_view<const T, int64_t, raft::row_major>& dataset) \
  {                                                                                          \
    return detail::vpq_build_half<decltype(dataset)>(res, params, dataset);                  \
  }

CUVS_INST_VPQ_BUILD(float);
CUVS_INST_VPQ_BUILD(half);
CUVS_INST_VPQ_BUILD(int8_t);
CUVS_INST_VPQ_BUILD(uint8_t);

#undef CUVS_INST_VPQ_BUILD

void serialize(raft::resources const& res,
               std::ostream& os,
               const cuvs::neighbors::device_vpq_dataset<half, int64_t>& dataset)
{
  // Same file preamble as cagra::serialize. The nested blob carries only a kind tag and dtype,
  // matching serialize_cagra_dense_dataset, because a nested blob relies on its enclosing file for
  // the version; a standalone .vpq has no enclosing file, so the version is written here.
  std::string dtype_string = raft::numpy_serializer::get_numpy_dtype<half>().to_string();
  dtype_string.resize(4);
  os << dtype_string;
  raft::serialize_scalar(res, os, vpq_serialization_version);
  ::cuvs::neighbors::detail::serialize_vpq_dataset<half, int64_t>(res, os, dataset);
}

void serialize(raft::resources const& res,
               const std::string& filename,
               const cuvs::neighbors::device_vpq_dataset<half, int64_t>& dataset)
{
  std::ofstream os(filename, std::ios::out | std::ios::binary | std::ios::trunc);
  RAFT_EXPECTS(os.good(), "pq::serialize: cannot open %s for writing", filename.c_str());
  serialize(res, os, dataset);
}

void deserialize(raft::resources const& res,
                 std::istream& is,
                 std::unique_ptr<cuvs::neighbors::device_vpq_dataset<half, int64_t>>* out_dataset)
{
  RAFT_EXPECTS(out_dataset != nullptr, "pq::deserialize: out_dataset must not be null");
  char dtype_string[4];
  RAFT_EXPECTS(is.read(dtype_string, 4), "pq::deserialize: failed to read the dtype prefix");
  RAFT_EXPECTS(cuvs::util::validate_serialized_dtype<half>(dtype_string, sizeof(dtype_string)),
               "pq::deserialize: dtype prefix does not match a VPQ dataset with half codebooks");
  auto const version = raft::deserialize_scalar<int>(res, is);
  RAFT_EXPECTS(version == vpq_serialization_version,
               "pq::deserialize: serialization version mismatch, expected %d, got %d",
               vpq_serialization_version,
               version);
  *out_dataset = ::cuvs::neighbors::detail::deserialize_vpq_dataset<half, int64_t>(res, is);
}

void deserialize(raft::resources const& res,
                 const std::string& filename,
                 std::unique_ptr<cuvs::neighbors::device_vpq_dataset<half, int64_t>>* out_dataset)
{
  std::ifstream is(filename, std::ios::in | std::ios::binary);
  RAFT_EXPECTS(is.good(), "pq::deserialize: cannot open %s for reading", filename.c_str());
  deserialize(res, is, out_dataset);
}

namespace detail {

template <typename T>
auto train_from_rows(raft::resources const& res,
                     cuvs::neighbors::vpq_params const& params,
                     T const* src_ptr,
                     int64_t n_rows,
                     int64_t dim,
                     int64_t stride) -> cuvs::neighbors::device_vpq_dataset<half, int64_t>
{
  cudaPointerAttributes ptr_attrs;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&ptr_attrs, src_ptr));
  auto const* device_ptr = reinterpret_cast<T const*>(ptr_attrs.devicePointer);
  if (device_ptr == nullptr) {
    // A host mdspan makes training subsample the rows and encoding stream them in bounded batches,
    // so the dense dataset is never staged on the device.
    RAFT_EXPECTS(stride == dim, "make_vpq_dataset: host input must be tightly packed");
    auto row_view = raft::make_host_matrix_view<const T, int64_t>(src_ptr, n_rows, dim);
    return detail::vpq_build_half(res, params, row_view);
  }
  if (stride != dim) {
    auto dense = raft::make_device_matrix<T, int64_t>(res, n_rows, dim);
    raft::copy_matrix(dense.data_handle(),
                      dim,
                      device_ptr,
                      stride,
                      dim,
                      n_rows,
                      raft::resource::get_cuda_stream(res));
    auto dense_view =
      raft::make_device_matrix_view<const T, int64_t>(dense.data_handle(), n_rows, dim);
    return detail::vpq_build_half(res, params, dense_view);
  }
  auto row_view = raft::make_device_matrix_view<const T, int64_t>(device_ptr, n_rows, dim);
  return detail::vpq_build_half(res, params, row_view);
}

auto vpq_train_from_rows(raft::resources const& res,
                         cuvs::neighbors::vpq_params const& params,
                         void const* src_ptr,
                         cudaDataType_t dtype,
                         int64_t n_rows,
                         int64_t dim,
                         int64_t stride) -> cuvs::neighbors::device_vpq_dataset<half, int64_t>
{
  switch (dtype) {
    case CUDA_R_32F:
      return train_from_rows(res, params, static_cast<float const*>(src_ptr), n_rows, dim, stride);
    case CUDA_R_16F:
      return train_from_rows(res, params, static_cast<half const*>(src_ptr), n_rows, dim, stride);
    case CUDA_R_8I:
      return train_from_rows(res, params, static_cast<int8_t const*>(src_ptr), n_rows, dim, stride);
    case CUDA_R_8U:
      return train_from_rows(
        res, params, static_cast<uint8_t const*>(src_ptr), n_rows, dim, stride);
    default:
      RAFT_FAIL("make_vpq_dataset: unsupported dataset element type %d", static_cast<int>(dtype));
  }
}

}  // namespace detail

}  // namespace cuvs::preprocessing::quantize::pq
