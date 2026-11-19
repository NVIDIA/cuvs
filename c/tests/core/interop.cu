/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_utils.cuh"
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

#include <raft/core/host_mdarray.hpp>

#include "../../src/core/interop.hpp"

#include <cuvs/core/c_api.h>
#include <gtest/gtest.h>
#include <sys/types.h>

namespace cuvs::core {

TEST(Interop, FromDLPack)
{
  raft::resources res;
  auto data = raft::make_host_vector<float>(res, 2);
  data(0)   = 5;
  data(1)   = 10;

  auto device    = DLDevice{kDLCPU};
  auto data_type = DLDataType{kDLFloat, 4 * 8, 1};
  auto shape     = std::vector<int64_t>{2};

  auto tensor         = DLTensor{data.data_handle(), device, 1, data_type, shape.data()};
  auto managed_tensor = DLManagedTensor{tensor};

  using mdspan_type = raft::host_mdspan<float const, raft::vector_extent<int64_t>>;
  auto out          = from_dlpack<mdspan_type>(&managed_tensor);

  ASSERT_EQ(out.rank(), data.rank());
  ASSERT_EQ(out.extent(0), data.extent(0));
  ASSERT_EQ(out(0), data(0));
  ASSERT_EQ(out(1), data(1));
}

TEST(Interop, FromDLPackStrides)
{
  raft::resources res;
  auto data = raft::make_host_matrix<float>(res, 3, 2);
  data(0, 0)   = 1;
  data(1, 0)   = 2;
  data(2, 0)   = 3;
  data(0, 1)   = 4;
  data(1, 1)   = 5;
  data(2, 1)   = 6;

  auto device    = DLDevice{kDLCPU};
  auto data_type = DLDataType{kDLFloat, 4 * 8, 1};
  auto shape     = std::vector<int64_t>{3 ,2};

  auto tensor         = DLTensor{data.data_handle(), device, 2, data_type, shape.data()};
  auto managed_tensor = DLManagedTensor{tensor};

  // converting a 2D dltensor to a 1D mspan should fail
  using vector_mdspan_type = raft::host_mdspan<float const, raft::vector_extent<int64_t>>;
  ASSERT_THROW(from_dlpack<vector_mdspan_type>(&managed_tensor), raft::logic_error);

  // No stride information in the dltensor indicates row major
  using mdspan_type = raft::host_matrix_view<float const, int64_t, raft::row_major>;
  auto out = from_dlpack<mdspan_type>(&managed_tensor);
  ASSERT_EQ(out.rank(), data.rank());
  ASSERT_EQ(out.extent(0), data.extent(0));
  ASSERT_EQ(out.extent(1), data.extent(1));
  for (int64_t row = 0; row < data.extent(0); row++) {
    for (int64_t col = 0; col < data.extent(1); col++) {
      ASSERT_EQ(out(row, col), data(row, col));
    }
  }

  // asking for a col-major mdspan should fail if no strides are present
  using colmajor_mdspan_type = raft::host_matrix_view<float const, int64_t, raft::col_major>;
  ASSERT_THROW(from_dlpack<colmajor_mdspan_type>(&managed_tensor), raft::logic_error);

  // Setting strides equal to row major should also work
  auto strides = std::vector<int64_t>{2, 1};
  managed_tensor.dl_tensor.strides = strides.data();
  auto out_strided = from_dlpack<mdspan_type>(&managed_tensor);
  ASSERT_EQ(out_strided.rank(), data.rank());

  // Setting strides indicating col-major should be able to convert to a col-major
  // mdspan
  auto strides_colmajor = std::vector<int64_t>{1, 3};
  managed_tensor.dl_tensor.strides = strides_colmajor.data();
  auto out_colmajor = from_dlpack<colmajor_mdspan_type>(&managed_tensor);
  ASSERT_EQ(out_colmajor.rank(), data.rank());

  // But shouldn't be able to convert to a row-major
  ASSERT_THROW(from_dlpack<mdspan_type>(&managed_tensor), raft::logic_error);
}

TEST(Interop, ValidateDLPackRejectsMalformedTensors)
{
  float data[6]{};
  auto device    = DLDevice{kDLCPU};
  auto data_type = DLDataType{kDLFloat, 4 * 8, 1};
  auto shape     = std::vector<int64_t>{3, 2};

  auto valid = DLManagedTensor{DLTensor{data, device, 2, data_type, shape.data()}};
  ASSERT_NO_THROW(validate_dlpack_tensor(&valid));
  ASSERT_NO_THROW(validate_dlpack_tensor(&valid, 2));
  ASSERT_THROW(validate_dlpack_tensor(&valid, 1), raft::logic_error);
  ASSERT_THROW(validate_dlpack_tensor(nullptr), raft::logic_error);

  auto negative_ndim = valid;
  negative_ndim.dl_tensor.ndim = -1;
  ASSERT_THROW(validate_dlpack_tensor(&negative_ndim), raft::logic_error);

  auto null_shape = valid;
  null_shape.dl_tensor.shape = nullptr;
  ASSERT_THROW(validate_dlpack_tensor(&null_shape), raft::logic_error);

  auto negative_shape = std::vector<int64_t>{-3, 2};
  auto negative_extent = valid;
  negative_extent.dl_tensor.shape = negative_shape.data();
  ASSERT_THROW(validate_dlpack_tensor(&negative_extent), raft::logic_error);

  auto huge_shape = std::vector<int64_t>{int64_t{1} << 40, int64_t{1} << 40};
  auto overflowing = valid;
  overflowing.dl_tensor.shape = huge_shape.data();
  ASSERT_THROW(validate_dlpack_tensor(&overflowing), raft::logic_error);

  auto null_data = valid;
  null_data.dl_tensor.data = nullptr;
  ASSERT_THROW(validate_dlpack_tensor(&null_data), raft::logic_error);

  auto empty_shape = std::vector<int64_t>{0, 2};
  auto empty_null_data = null_data;
  empty_null_data.dl_tensor.shape = empty_shape.data();
  ASSERT_NO_THROW(validate_dlpack_tensor(&empty_null_data));

  using mdspan_type = raft::host_matrix_view<float const, int64_t, raft::row_major>;
  ASSERT_THROW(from_dlpack<mdspan_type>(&negative_extent), raft::logic_error);
  ASSERT_THROW(from_dlpack<mdspan_type>(&null_shape), raft::logic_error);
  ASSERT_THROW(from_dlpack<mdspan_type>(&null_data), raft::logic_error);
}

TEST(Interop, MatrixSliceRowsBounds)
{
  float data[6]{};
  auto shape = std::vector<int64_t>{3, 2};
  auto src   = DLManagedTensor{
    DLTensor{data, DLDevice{kDLCPU}, 2, DLDataType{kDLFloat, 4 * 8, 1}, shape.data()}};

  cuvsResources_t res;
  ASSERT_EQ(cuvsResourcesCreate(&res), CUVS_SUCCESS);

  DLManagedTensor dst{};
  ASSERT_EQ(cuvsMatrixSliceRows(res, &src, 1, 3, &dst), CUVS_SUCCESS);
  ASSERT_EQ(dst.dl_tensor.shape[0], 2);
  ASSERT_EQ(dst.dl_tensor.data, static_cast<void*>(data + 2));
  dst.deleter(&dst);

  DLManagedTensor past_end{};
  ASSERT_EQ(cuvsMatrixSliceRows(res, &src, 2, 4, &past_end), CUVS_ERROR);
  DLManagedTensor negative_start{};
  ASSERT_EQ(cuvsMatrixSliceRows(res, &src, -1, 1, &negative_start), CUVS_ERROR);

  ASSERT_EQ(cuvsResourcesDestroy(res), CUVS_SUCCESS);
}

}  // namespace cuvs::core
