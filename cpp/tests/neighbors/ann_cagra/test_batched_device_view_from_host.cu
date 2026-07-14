/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/managed_mdarray.hpp>
#include <raft/core/pinned_mdarray.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/resources.hpp>
#include <raft/matrix/init.cuh>
#include <raft/util/cudart_utils.hpp>

#include "../../../src/neighbors/detail/cagra/utils.hpp"

#include <array>
#include <cstdint>
#include <tuple>
#include <vector>

namespace cuvs::neighbors::cagra {

using IdxT = uint32_t;

struct BatchConfig {
  bool initialize;
  bool host_writeback;
};

struct DimsConfig {
  int64_t n_rows;
  int64_t n_cols;
  uint64_t batch_size;
};

class BatchedDeviceViewFromHostTest : public ::testing::Test {
 protected:
  void SetUp() override { raft::resource::sync_stream(res); }

  /**
   * Run batched_device_view_from_host over host data, copy device views back,
   * and verify against the input.
   */
  template <typename InputMatrixView>
  void run_and_verify_batched(InputMatrixView input_view,
                              uint64_t batch_size,
                              bool host_writeback,
                              bool initialize)
  {
    int64_t n_rows = input_view.extent(0);
    int64_t n_cols = input_view.extent(1);

    std::vector<IdxT> readback(n_rows * n_cols);

    int64_t total_processed = 0;

    {
      cagra::detail::batched_device_view_from_host<IdxT, int64_t> batched(
        res,
        raft::make_host_matrix_view<IdxT, int64_t>(input_view.data_handle(), n_rows, n_cols),
        batch_size,
        host_writeback,
        initialize);
      while (true) {
        auto dev_view = batched.next_view();
        if (dev_view.extent(0) == 0) break;

        if (initialize) {
          raft::copy(readback.data() + total_processed * n_cols,
                     dev_view.data_handle(),
                     dev_view.extent(0) * dev_view.extent(1),
                     raft::resource::get_cuda_stream(res));
        }
        if (host_writeback) { raft::matrix::fill(res, dev_view, IdxT(17)); }
        total_processed += dev_view.extent(0);
      }
    }
    raft::resource::sync_stream(res);

    EXPECT_EQ(total_processed, n_rows);
    if (initialize) {
      for (int64_t i = 0; i < n_rows * n_cols; ++i) {
        EXPECT_EQ(readback[i], IdxT(13)) << "Mismatch (initialize) at index " << i;
      }
    }
    if (host_writeback) {
      auto readback_view =
        raft::make_host_matrix_view<IdxT, int64_t>(readback.data(), n_rows, n_cols);
      raft::copy(res, readback_view, input_view);
      raft::resource::sync_stream(res);
      for (int64_t i = 0; i < n_rows * n_cols; ++i) {
        EXPECT_EQ(readback[i], IdxT(17)) << "Mismatch (host_writeback) at index " << i;
      }
    }
  }

  raft::resources res;
};

TEST_F(BatchedDeviceViewFromHostTest, EmptyView)
{
  auto host_empty = raft::make_host_matrix<IdxT, int64_t>(0, 8);
  auto host_view  = host_empty.view();
  cagra::detail::batched_device_view_from_host<IdxT, int64_t> batched(
    res, host_view, /*batch_size=*/128, /*host_writeback=*/false, /*initialize=*/true);

  auto view = batched.next_view();
  EXPECT_EQ(view.extent(0), 0);
  EXPECT_EQ(view.extent(1), 8);
  EXPECT_EQ(view.data_handle(), nullptr);
}

using BatchDimsParam = std::tuple<BatchConfig, DimsConfig>;

class BatchedDeviceViewFromHostParameterizedTest
  : public BatchedDeviceViewFromHostTest,
    public ::testing::WithParamInterface<BatchDimsParam> {};

TEST_P(BatchedDeviceViewFromHostParameterizedTest, VectorHostData)
{
  auto [batch_config, dims_config]  = GetParam();
  auto [initialize, host_writeback] = batch_config;
  auto [n_rows, n_cols, batch_size] = dims_config;

  std::vector<IdxT> host_data(n_rows * n_cols);
  auto host_view = raft::make_host_matrix_view<IdxT, int64_t>(host_data.data(), n_rows, n_cols);

  std::fill(host_view.data_handle(), host_view.data_handle() + n_rows * n_cols, IdxT(13));

  run_and_verify_batched(host_view, batch_size, host_writeback, initialize);
}

TEST_P(BatchedDeviceViewFromHostParameterizedTest, PinnedMemory)
{
  auto [batch_config, dims_config]  = GetParam();
  auto [initialize, host_writeback] = batch_config;
  auto [n_rows, n_cols, batch_size] = dims_config;

  auto host_matrix = raft::make_pinned_matrix<IdxT, int64_t>(res, n_rows, n_cols);
  auto host_view   = host_matrix.view();

  std::fill(host_view.data_handle(), host_view.data_handle() + n_rows * n_cols, IdxT(13));

  run_and_verify_batched(host_view, batch_size, host_writeback, initialize);
}

TEST_P(BatchedDeviceViewFromHostParameterizedTest, ManagedMemory)
{
  auto [batch_config, dims_config]  = GetParam();
  auto [initialize, host_writeback] = batch_config;
  auto [n_rows, n_cols, batch_size] = dims_config;

  auto host_matrix = raft::make_managed_matrix<IdxT, int64_t>(res, n_rows, n_cols);
  auto host_view   = host_matrix.view();

  std::fill(host_view.data_handle(), host_view.data_handle() + n_rows * n_cols, IdxT(13));

  run_and_verify_batched(host_view, batch_size, host_writeback, initialize);
}

TEST_P(BatchedDeviceViewFromHostParameterizedTest, DeviceMemory)
{
  auto [batch_config, dims_config]  = GetParam();
  auto [initialize, host_writeback] = batch_config;
  auto [n_rows, n_cols, batch_size] = dims_config;

  auto host_matrix = raft::make_device_matrix<IdxT, int64_t>(res, n_rows, n_cols);
  auto host_view   = host_matrix.view();

  raft::matrix::fill(res, host_view, IdxT(13));

  run_and_verify_batched(host_view, batch_size, host_writeback, initialize);
}

static const std::array<BatchConfig, 3> kBatchConfigs = {{
  {/*initialize=*/true, /*host_writeback=*/false},
  {/*initialize=*/false, /*host_writeback=*/true},
  {/*initialize=*/true, /*host_writeback=*/true},
}};

static const std::array<DimsConfig, 4> kDimsConfigs = {{
  {/*n_rows=*/64, /*n_cols=*/32, /*batch_size=*/256},  // rows less than batch size, single batch
  {/*n_rows=*/64, /*n_cols=*/32, /*batch_size=*/64},   // single batch
  {/*n_rows=*/256, /*n_cols=*/32, /*batch_size=*/32},  // multiple batches
  {/*n_rows=*/500,
   /*n_cols=*/32,
   /*batch_size=*/128},  // multiple batches, partial batch in the end
}};

INSTANTIATE_TEST_SUITE_P(BatchConfigs,
                         BatchedDeviceViewFromHostParameterizedTest,
                         ::testing::Combine(::testing::ValuesIn(kBatchConfigs),
                                            ::testing::ValuesIn(kDimsConfigs)));

}  // namespace cuvs::neighbors::cagra
