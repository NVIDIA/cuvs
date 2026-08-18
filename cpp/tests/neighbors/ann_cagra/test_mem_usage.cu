/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Memory-usage estimators for the CAGRA build.
 *
 * These predict how much host and device memory a build needs without running it, so that callers
 * such as hnsw::build and the ACE partition heuristic can choose between an in-memory and an
 * out-of-core build. An estimate cannot be checked against itself, so each test here compares one
 * against something derived independently: the VPQ footprint against a dataset that was actually
 * compressed, and the iterative estimate against the buffers that provably coexist during the build
 * and against the peak that raft::memory_stats_resources measures while a real build runs.
 */

#include <gtest/gtest.h>

#include "../cagra_padded_build_helpers.cuh"

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/preprocessing/quantize/pq.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/memory_stats_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/make_blobs.cuh>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace cuvs::neighbors::cagra {

namespace {

using vpq_dataset_t = cuvs::neighbors::device_vpq_dataset<half, int64_t>;

/** Size of the graph index type the CAGRA build uses internally. */
constexpr size_t kIndexSize = sizeof(uint32_t);

auto extents_of(int64_t n_rows, int64_t dim) { return raft::make_extents<int64_t>(n_rows, dim); }

/** Device bytes actually owned by a compressed dataset: both codebooks plus the encoded rows. */
auto compressed_bytes(const vpq_dataset_t& v) -> size_t
{
  return static_cast<size_t>(v.vq_code_book.extent(0)) * v.vq_code_book.extent(1) * sizeof(half) +
         static_cast<size_t>(v.pq_code_book.extent(0)) * v.pq_code_book.extent(1) * sizeof(half) +
         static_cast<size_t>(v.data.extent(0)) * v.data.extent(1);
}

auto make_clustered(const raft::resources& res, int64_t n_rows, int64_t dim)
  -> raft::device_matrix<float, int64_t>
{
  auto dataset = raft::make_device_matrix<float, int64_t>(res, n_rows, dim);
  auto labels  = raft::make_device_vector<int64_t, int64_t>(res, n_rows);
  raft::random::make_blobs<float, int64_t, raft::row_major>(res,
                                                           dataset.view(),
                                                           labels.view(),
                                                           5,             // clusters
                                                           std::nullopt,  // random centers
                                                           std::nullopt,  // scalar std
                                                           1.0F,          // cluster std
                                                           true,          // shuffle
                                                           -10.0F,        // center box min
                                                           10.0F,         // center box max
                                                           1234ULL);
  raft::resource::sync_stream(res);
  return dataset;
}

auto iterative_params(size_t graph_degree, size_t intermediate_graph_degree) -> index_params
{
  index_params params;
  params.metric                    = cuvs::distance::DistanceType::L2Expanded;
  params.graph_degree              = graph_degree;
  params.intermediate_graph_degree = intermediate_graph_degree;
  params.graph_build_params        = graph_build_params::iterative_search_params();
  return params;
}

auto device_estimate(const raft::resources& res,
                     int64_t n_rows,
                     int64_t dim,
                     const index_params& params,
                     std::optional<cuvs::neighbors::vpq_params> compression = std::nullopt) -> size_t
{
  return helpers::cagra_build_mem_usage(
           res, extents_of(n_rows, dim), CUDA_R_32F, params, compression)
    .second;
}

}  // namespace

// ---------------------------------------------------------------------------
// vpq_dataset_size: predicted footprint vs a dataset that was really compressed
// ---------------------------------------------------------------------------

TEST(CagraMemUsage, VpqDatasetSizeMatchesCompressedDataset)
{
  raft::resources res;
  constexpr int64_t n_rows = 512;
  constexpr int64_t dim    = 32;

  cuvs::neighbors::vpq_params params;
  params.pq_bits        = 8;
  params.pq_dim         = 8;  // dim must be divisible by pq_dim
  params.vq_n_centers   = 32;
  params.kmeans_n_iters = 2;

  auto dataset    = make_clustered(res, n_rows, dim);
  auto compressed = cuvs::preprocessing::quantize::pq::make_vpq_dataset(
    res, params, raft::make_const_mdspan(dataset.view()));
  raft::resource::sync_stream(res);

  EXPECT_EQ(helpers::vpq_dataset_size(extents_of(n_rows, dim), params),
            compressed_bytes(compressed));
}

TEST(CagraMemUsage, VpqDatasetSizeResolvesUnsetParams)
{
  raft::resources res;
  constexpr int64_t n_rows = 1024;
  constexpr int64_t dim    = 32;

  // pq_dim and vq_n_centers left at 0, so both the estimator and the build have to derive them.
  cuvs::neighbors::vpq_params params;
  params.pq_bits        = 8;
  params.kmeans_n_iters = 2;

  auto dataset    = make_clustered(res, n_rows, dim);
  auto compressed = cuvs::preprocessing::quantize::pq::make_vpq_dataset(
    res, params, raft::make_const_mdspan(dataset.view()));
  raft::resource::sync_stream(res);

  EXPECT_EQ(helpers::vpq_dataset_size(extents_of(n_rows, dim), params),
            compressed_bytes(compressed));
}

// ---------------------------------------------------------------------------
// optimize_workspace_size: device-resident graphs need no staging buffers
// ---------------------------------------------------------------------------

TEST(CagraMemUsage, OptimizeWorkspaceSkipsStagingForDeviceResidentGraphs)
{
  constexpr size_t n_rows              = 1000000;
  constexpr size_t graph_degree        = 32;
  constexpr size_t intermediate_degree = 256;

  auto [host_staged, dev_staged, host_fixed_staged, dev_fixed_staged] =
    helpers::optimize_workspace_size(
      n_rows, graph_degree, intermediate_degree, kIndexSize, false, false);
  auto [host_resident, dev_resident, host_fixed_resident, dev_fixed_resident] =
    helpers::optimize_workspace_size(
      n_rows, graph_degree, intermediate_degree, kIndexSize, false, true);

  // The flag only removes device-side staging buffers, so the host estimate must not move.
  EXPECT_EQ(host_resident, host_staged);
  EXPECT_EQ(host_fixed_resident, host_fixed_staged);

  // At this shape the prune stage dominates the device total, and the d_input_graph staging copy
  // it no longer needs is the largest single term the flag drops.
  EXPECT_LT(dev_resident, dev_staged);
  EXPECT_LE(dev_fixed_resident, dev_fixed_staged);
}

TEST(CagraMemUsage, OptimizeWorkspaceDefaultsToStagedGraphs)
{
  constexpr size_t n_rows = 4096;
  EXPECT_EQ(helpers::optimize_workspace_size(n_rows, 32, 64, kIndexSize),
            helpers::optimize_workspace_size(n_rows, 32, 64, kIndexSize, false, false));
}

// ---------------------------------------------------------------------------
// cagra_build_mem_usage: the iterative graph build
// ---------------------------------------------------------------------------

TEST(CagraMemUsage, IterativeEstimateCoversCoexistingBuffers)
{
  raft::resources res;
  constexpr int64_t n_rows             = 1000000;
  constexpr int64_t dim                = 128;  // 16-byte aligned for float, so never padded
  constexpr size_t graph_degree        = 64;
  constexpr size_t intermediate_degree = 128;

  const size_t dataset_bytes = static_cast<size_t>(n_rows) * dim * sizeof(float);
  const size_t graph_bytes   = static_cast<size_t>(n_rows) * graph_degree * kIndexSize;
  const size_t knn_bytes = static_cast<size_t>(n_rows) * (intermediate_degree + 1) * kIndexSize;

  const auto estimated =
    device_estimate(res, n_rows, dim, iterative_params(graph_degree, intermediate_degree));

  // The resident dataset, the graph carried over from the previous iteration and the kNN graph the
  // final search fills are all live at the same moment, so the estimate cannot be below their sum.
  EXPECT_GE(estimated, dataset_bytes + graph_bytes + knn_bytes);
}

TEST(CagraMemUsage, IterativeEstimateGrowsWithDatasetAndDegree)
{
  raft::resources res;
  constexpr int64_t dim = 128;
  const auto params     = iterative_params(64, 128);

  EXPECT_GT(device_estimate(res, 2000000, dim, params),
            device_estimate(res, 1000000, dim, params));
  EXPECT_GT(device_estimate(res, 1000000, 256, params),
            device_estimate(res, 1000000, dim, params));
  EXPECT_GT(device_estimate(res, 1000000, dim, iterative_params(128, 256)),
            device_estimate(res, 1000000, dim, params));
}

TEST(CagraMemUsage, IterativeEstimateIsSmallerForCompressedDataset)
{
  raft::resources res;
  constexpr int64_t n_rows = 1000000;
  constexpr int64_t dim    = 128;
  const auto params        = iterative_params(64, 128);

  cuvs::neighbors::vpq_params compression;
  compression.pq_bits      = 8;
  compression.pq_dim       = 32;
  compression.vq_n_centers = 1024;

  // CAGRA-Q keeps codes instead of dense rows, so the resident dataset term shrinks by an order of
  // magnitude even though the per-chunk reconstruction scratch is new.
  EXPECT_LT(device_estimate(res, n_rows, dim, params, compression),
            device_estimate(res, n_rows, dim, params));
}

TEST(CagraMemUsage, IterativeEstimateAcceptsEqualDegrees)
{
  raft::resources res;
  // graph_degree == intermediate_graph_degree once broke the iterative build (issue #1818). The
  // estimator has to accept it rather than trip the degree check inside optimize_workspace_size.
  EXPECT_GT(device_estimate(res, 100000, 64, iterative_params(16, 16)), 0U);
}

TEST(CagraMemUsage, IterativeEstimateBoundsMeasuredPeak)
{
  raft::resources res;
  constexpr int64_t n_rows = 20000;
  constexpr int64_t dim    = 64;
  const auto params        = iterative_params(32, 64);

  const auto estimated = device_estimate(res, n_rows, dim, params);

  size_t measured = 0;
  {
    // The tracking handle replaces the global device resource, so it has to outlive everything
    // allocated below; declaration order here gives it exactly that.
    raft::memory_stats_resources tracked{res};
    auto dataset = make_clustered(tracked, n_rows, dim);
    cuvs::neighbors::test::padded_device_matrix_for_cagra<float> padded(
      tracked, raft::make_const_mdspan(dataset.view()));
    auto built = cagra::build(tracked, params, padded.view);
    raft::resource::sync_stream(tracked);
    ASSERT_GT(built.size(), 0);

    const auto peak = tracked.get_bytes_peak();
    measured        = peak.device_global + peak.device_workspace + peak.device_large_workspace +
               peak.device_managed;
  }

  // One-sided on purpose: hnsw::build uses this estimate to decide whether an in-memory build
  // fits, so under-predicting is the failure that matters. No upper bound is asserted because the
  // estimate also carries the whole workspace allowance of the resources handle.
  EXPECT_GE(estimated, measured);
}

}  // namespace cuvs::neighbors::cagra
