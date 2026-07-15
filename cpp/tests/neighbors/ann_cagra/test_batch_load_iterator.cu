/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

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
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/init.cuh>
#include <raft/util/cudart_utils.hpp>

#include "../../../src/neighbors/detail/ann_utils.cuh"

#include <array>
#include <cstdint>
#include <tuple>
#include <vector>

namespace cuvs::neighbors::cagra {

namespace bli = cuvs::spatial::knn::detail::utils;

using IdxT = uint32_t;
using DeviceAccessor =
  raft::host_device_accessor<cuda::std::default_accessor<IdxT>, raft::memory_type::device>;
using HostAccessor =
  raft::host_device_accessor<cuda::std::default_accessor<IdxT>, raft::memory_type::host>;
using PinnedAccessor =
  raft::host_device_accessor<cuda::std::default_accessor<IdxT>, raft::memory_type::pinned>;
using ManagedAccessor =
  raft::host_device_accessor<cuda::std::default_accessor<IdxT>, raft::memory_type::managed>;

struct BatchConfig {
  bool initialize;
  bool host_writeback;
};

struct DimsConfig {
  int64_t n_rows;
  int64_t n_cols;
  uint64_t batch_size;
};

class BatchLoadIteratorTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    // Provide a stream pool so get_prefetch_stream(res) returns a non-main stream and
    // the iterator exercises its real pipelined path.
    raft::resource::set_cuda_stream_pool(res, std::make_shared<rmm::cuda_stream_pool>(1));
    raft::resource::sync_stream(res);
  }

  /**
   * Run batch_load_iterator over input_view, copy device views back, and verify against the
   * input. Mirrors the old batched_device_view test: writeback fills with IdxT(17), readback
   * (when initialize==true) verifies pre-fill of IdxT(13).
   */
  template <typename AccessorInputView>
  void run_and_verify_batched(
    raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorInputView> input_view,
    uint64_t batch_size,
    bool host_writeback,
    bool initialize)
  {
    using mdspan_t = decltype(input_view);

    int64_t n_rows = input_view.extent(0);
    int64_t n_cols = input_view.extent(1);

    std::vector<IdxT> readback(n_rows * n_cols);

    int64_t total_processed = 0;

    auto [copy_stream, enable_prefetch] = bli::get_prefetch_stream(res);
    auto workspace_mr                   = raft::resource::get_workspace_resource_ref(res);

    {
      bli::batch_load_iterator<mdspan_t> iter(res,
                                              input_view,
                                              batch_size,
                                              copy_stream,
                                              workspace_mr,
                                              enable_prefetch,
                                              initialize,
                                              host_writeback);
      iter.prefetch_next_batch();

      for (auto& batch : iter) {
        if (batch.size() == 0) break;

        auto dev_view = batch.view();

        if (initialize) {
          raft::copy(readback.data() + total_processed * n_cols,
                     dev_view.data_handle(),
                     dev_view.extent(0) * dev_view.extent(1),
                     raft::resource::get_cuda_stream(res));
        }
        if (host_writeback) {
          // The passthrough strategy returns `cuda::std::submdspan(input_view, ...)`, which
          // may be `layout_stride` and/or carry a non-default accessor (e.g. PinnedAccessor),
          // neither of which `raft::matrix::fill`'s `device_matrix_view` overload accepts.
          // Re-wrap as a plain device_matrix_view since the slice is always exhaustive
          // (a contiguous row range of a row-major input). This re-wrap should eventually be
          // removed once raft::matrix::fill grows a more generic mdspan overload.
          raft::matrix::fill(res,
                             raft::make_device_matrix_view<IdxT, int64_t>(
                               dev_view.data_handle(), dev_view.extent(0), dev_view.extent(1)),
                             IdxT(17));
        }
        total_processed += dev_view.extent(0);

        iter.prefetch_next_batch();
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

TEST_F(BatchLoadIteratorTest, EmptyViewFromHost)
{
  auto host_empty = raft::make_host_matrix<IdxT, int64_t>(0, 8);
  auto host_view  = host_empty.view();

  auto [copy_stream, enable_prefetch] = bli::get_prefetch_stream(res);
  auto workspace_mr                   = raft::resource::get_workspace_resource_ref(res);

  bli::batch_load_iterator<raft::host_matrix_view<IdxT, int64_t>> iter(
    res, host_view, /*batch_size=*/128, copy_stream, workspace_mr, enable_prefetch);
  EXPECT_TRUE(iter.begin() == iter.end());
}

TEST_F(BatchLoadIteratorTest, EmptyViewFromDevice)
{
  auto device_empty = raft::make_device_matrix<IdxT, int64_t>(res, 0, 8);
  auto device_view  = device_empty.view();

  auto [copy_stream, enable_prefetch] = bli::get_prefetch_stream(res);
  auto workspace_mr                   = raft::resource::get_workspace_resource_ref(res);

  bli::batch_load_iterator<raft::device_matrix_view<IdxT, int64_t>> iter(
    res, device_view, /*batch_size=*/128, copy_stream, workspace_mr, enable_prefetch);
  EXPECT_TRUE(iter.begin() == iter.end());
}

using BatchDimsParam = std::tuple<BatchConfig, DimsConfig>;

class BatchLoadIteratorParameterizedTest : public BatchLoadIteratorTest,
                                           public ::testing::WithParamInterface<BatchDimsParam> {};

TEST_P(BatchLoadIteratorParameterizedTest, VectorHostData)
{
  auto [batch_config, dims_config]  = GetParam();
  auto [initialize, host_writeback] = batch_config;
  auto [n_rows, n_cols, batch_size] = dims_config;

  std::vector<IdxT> host_data(n_rows * n_cols);
  auto host_view = raft::make_host_matrix_view<IdxT, int64_t>(host_data.data(), n_rows, n_cols);

  std::fill(host_view.data_handle(), host_view.data_handle() + n_rows * n_cols, IdxT(13));

  run_and_verify_batched(host_view, batch_size, host_writeback, initialize);
}

TEST_P(BatchLoadIteratorParameterizedTest, PinnedMemory)
{
  auto [batch_config, dims_config]  = GetParam();
  auto [initialize, host_writeback] = batch_config;
  auto [n_rows, n_cols, batch_size] = dims_config;

  auto pinned_matrix = raft::make_pinned_matrix<IdxT, int64_t>(res, n_rows, n_cols);
  auto pinned_view =
    raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, PinnedAccessor>(
      pinned_matrix.data_handle(), n_rows, n_cols);
  std::fill(pinned_view.data_handle(), pinned_view.data_handle() + n_rows * n_cols, IdxT(13));

  run_and_verify_batched(pinned_view, batch_size, host_writeback, initialize);
}

TEST_P(BatchLoadIteratorParameterizedTest, PinnedMemoryForcedToHost)
{
  auto [batch_config, dims_config]  = GetParam();
  auto [initialize, host_writeback] = batch_config;
  auto [n_rows, n_cols, batch_size] = dims_config;

  auto pinned_matrix = raft::make_pinned_matrix<IdxT, int64_t>(res, n_rows, n_cols);

  auto pinned_view =
    raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, HostAccessor>(
      pinned_matrix.data_handle(), n_rows, n_cols);

  std::fill(pinned_view.data_handle(), pinned_view.data_handle() + n_rows * n_cols, IdxT(13));
  run_and_verify_batched(pinned_view, batch_size, host_writeback, initialize);
}

TEST_P(BatchLoadIteratorParameterizedTest, ManagedMemory)
{
  auto [batch_config, dims_config]  = GetParam();
  auto [initialize, host_writeback] = batch_config;
  auto [n_rows, n_cols, batch_size] = dims_config;

  auto managed_matrix = raft::make_managed_matrix<IdxT, int64_t>(res, n_rows, n_cols);
  auto managed_view   = managed_matrix.view();

  std::fill(managed_view.data_handle(), managed_view.data_handle() + n_rows * n_cols, IdxT(13));

  run_and_verify_batched(managed_view, batch_size, host_writeback, initialize);
}

TEST_P(BatchLoadIteratorParameterizedTest, ManagedMemoryForcedToHost)
{
  auto [batch_config, dims_config]  = GetParam();
  auto [initialize, host_writeback] = batch_config;
  auto [n_rows, n_cols, batch_size] = dims_config;

  auto managed_matrix = raft::make_managed_matrix<IdxT, int64_t>(res, n_rows, n_cols);

  auto managed_view =
    raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, HostAccessor>(
      managed_matrix.data_handle(), n_rows, n_cols);

  std::fill(managed_view.data_handle(), managed_view.data_handle() + n_rows * n_cols, IdxT(13));

  run_and_verify_batched(managed_view, batch_size, host_writeback, initialize);
}

TEST_P(BatchLoadIteratorParameterizedTest, DeviceMemory)
{
  auto [batch_config, dims_config]  = GetParam();
  auto [initialize, host_writeback] = batch_config;
  auto [n_rows, n_cols, batch_size] = dims_config;

  auto device_matrix = raft::make_device_matrix<IdxT, int64_t>(res, n_rows, n_cols);
  auto device_view   = device_matrix.view();

  raft::matrix::fill(res, device_view, IdxT(13));

  run_and_verify_batched(device_view, batch_size, host_writeback, initialize);
}

/**
 * Drive the runtime-dispatched wrapper. Verifies that:
 *   * a host pointer dispatches to the copy_device branch (does_copy() == true), and
 *   * a device pointer dispatches to passthrough (does_copy() == false),
 * and that initialize-only iteration yields the expected pre-fill on each batch.
 */
TEST_F(BatchLoadIteratorTest, MakeBatchLoadIteratorHostPtr)
{
  const int64_t n_rows         = 256;
  const int64_t n_cols         = 32;
  const size_t batch_size_rows = 64;

  std::vector<IdxT> host_data(n_rows * n_cols, IdxT(13));
  auto [copy_stream, enable_prefetch] = bli::get_prefetch_stream(res);

  auto iter = bli::make_batch_load_iterator<IdxT>(res,
                                                  host_data.data(),
                                                  n_rows,
                                                  n_cols,
                                                  batch_size_rows,
                                                  copy_stream,
                                                  raft::resource::get_workspace_resource_ref(res),
                                                  enable_prefetch);
  EXPECT_TRUE(iter.does_copy());

  std::vector<IdxT> readback(n_rows * n_cols, IdxT(0));
  int64_t total = 0;
  iter.prefetch_next_batch();
  for (auto const& batch : iter) {
    if (batch.size() == 0) break;
    raft::copy(readback.data() + total * n_cols,
               batch.data(),
               batch.size() * n_cols,
               raft::resource::get_cuda_stream(res));
    total += batch.size();
    iter.prefetch_next_batch();
  }
  raft::resource::sync_stream(res);
  EXPECT_EQ(total, n_rows);
  for (int64_t i = 0; i < n_rows * n_cols; ++i) {
    EXPECT_EQ(readback[i], IdxT(13)) << "Mismatch at index " << i;
  }
}

TEST_F(BatchLoadIteratorTest, MakeBatchLoadIteratorDevicePtr)
{
  const int64_t n_rows         = 256;
  const int64_t n_cols         = 32;
  const size_t batch_size_rows = 64;

  auto device_matrix = raft::make_device_matrix<IdxT, int64_t>(res, n_rows, n_cols);
  raft::matrix::fill(res, device_matrix.view(), IdxT(13));

  auto [copy_stream, enable_prefetch] = bli::get_prefetch_stream(res);
  auto iter                           = bli::make_batch_load_iterator<IdxT>(res,
                                                  device_matrix.data_handle(),
                                                  n_rows,
                                                  n_cols,
                                                  batch_size_rows,
                                                  copy_stream,
                                                  raft::resource::get_workspace_resource_ref(res),
                                                  enable_prefetch);
  EXPECT_FALSE(iter.does_copy());

  std::vector<IdxT> readback(n_rows * n_cols, IdxT(0));
  int64_t total = 0;
  iter.prefetch_next_batch();
  for (auto const& batch : iter) {
    if (batch.size() == 0) break;
    raft::copy(readback.data() + total * n_cols,
               batch.data(),
               batch.size() * n_cols,
               raft::resource::get_cuda_stream(res));
    total += batch.size();
    iter.prefetch_next_batch();
  }
  raft::resource::sync_stream(res);
  EXPECT_EQ(total, n_rows);
  for (int64_t i = 0; i < n_rows * n_cols; ++i) {
    EXPECT_EQ(readback[i], IdxT(13)) << "Mismatch at index " << i;
  }
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
                         BatchLoadIteratorParameterizedTest,
                         ::testing::Combine(::testing::ValuesIn(kBatchConfigs),
                                            ::testing::ValuesIn(kDimsConfigs)));

}  // namespace cuvs::neighbors::cagra
