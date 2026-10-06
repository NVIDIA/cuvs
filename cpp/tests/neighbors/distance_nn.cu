/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "distance_nn_helper.cuh"

#include "../../src/distance/top_1_nn.cuh"
#include "../../src/distance/unfused_distance_nn.cuh"

#include <cuda/stream>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/init.cuh>

#include <limits>

namespace cuvs::neighbors {

enum class ImplType { fused, unfused };

template <typename IdxT>
struct NNInputs {
  IdxT m;
  IdxT n;
  IdxT k;
  DistanceType metric;
  bool sqrt;
  uint64_t rng_seed;
  double tol;
  cuvs::distance::detail::Top1nnBackend backend = cuvs::distance::detail::Top1nnBackend::Cutlass;
  cuvs::distance::detail::Top1nnTuning tuning{};
};

__global__ void fill_int8(int8_t* buff, int len, int seed_offset)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // Fill the buffer with pseudo-random int8_t values using a simple LCG
  for (int i = tid; i < len; i += blockDim.x * gridDim.x) {
    int hash = (i + seed_offset) * 1103515245 + 12345;
    buff[i]  = static_cast<int8_t>((hash >> 16) & 0xFF);
  }
}

template <typename DataT, typename AccT, typename IdxT, ImplType impl>
class NNTest : public ::testing::TestWithParam<NNInputs<IdxT>> {
 public:
  using OutT = raft::KeyValuePair<IdxT, AccT>;
  NNTest()
    : params_{::testing::TestWithParam<NNInputs<IdxT>>::GetParam()},
      m{params_.m},
      n{params_.n},
      k{params_.k},
      metric{params_.metric},
      sqrt{params_.sqrt},
      backend{params_.backend},
      tuning{params_.tuning},
      stream{raft::resource::get_cuda_stream(handle)},
      x{raft::make_device_matrix<DataT, IdxT>(handle, m, k)},
      y{raft::make_device_matrix<DataT, IdxT>(handle, n, k)},
      x_norm{raft::make_device_vector<AccT, IdxT>(handle, m)},
      y_norm{raft::make_device_vector<AccT, IdxT>(handle, n)},
      out{raft::make_device_vector<OutT, IdxT>(handle, m)},
      ref_out{raft::make_device_vector<OutT, IdxT>(handle, m)},
      ref_dist{raft::make_device_vector<AccT, IdxT>(handle, m)},
      selected_dist{raft::make_device_vector<AccT, IdxT>(handle, m)},
      cutile_idx{raft::make_device_vector<IdxT, IdxT>(handle, m)},
      cutile_dist{raft::make_device_vector<AccT, IdxT>(handle, m)}
  {
  }

 protected:
  void SetUp() override
  {
    raft::random::RngState rng{params_.rng_seed};
    if constexpr (std::is_same_v<DataT, int8_t>) {
      fill_int8<<<1000, 256, 0, stream.get()>>>(x.data_handle(), m * k, 0);
      RAFT_CUDA_TRY(cudaGetLastError());
      fill_int8<<<1000, 256, 0, stream.get()>>>(y.data_handle(), n * k, m * k);
      RAFT_CUDA_TRY(cudaGetLastError());
    } else {
      raft::random::uniform(handle, rng, x.data_handle(), m * k, DataT(-1.0), DataT(1.0));
      raft::random::uniform(handle, rng, y.data_handle(), n * k, DataT(-1.0), DataT(1.0));
    }

    // Pre-compute norms
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
      x_norm.data_handle(), x.data_handle(), k, m, stream.get());
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
      y_norm.data_handle(), y.data_handle(), k, n, stream.get());

    // CosineExpanded expects ||x|| not ||x||^2
    if (metric == DistanceType::CosineExpanded) {
      raft::linalg::unaryOp(
        x_norm.data_handle(), x_norm.data_handle(), m, raft::sqrt_op{}, stream.get());
      raft::linalg::unaryOp(
        y_norm.data_handle(), y_norm.data_handle(), n, raft::sqrt_op{}, stream.get());
    }

    if constexpr (impl == ImplType::fused) {
      if (backend == cuvs::distance::detail::Top1nnBackend::Cutile &&
          !cuvs::distance::detail::is_top_1_nn_backend_available(
            backend, x.data_handle(), y.data_handle(), m, n, k, metric)) {
        backend_unavailable = true;
      } else {
        plan = cuvs::distance::probe_top_1_nn(
          handle, x.data_handle(), y.data_handle(), m, n, k, tuning, metric, backend);
        backend_unavailable = !plan.available;
        workspace_size      = plan.workspace_bytes;
      }
    } else if constexpr (impl == ImplType::unfused) {
      workspace_size = m * n * sizeof(AccT);
    }

    // Reset buffer
    if constexpr (std::is_same_v<OutT, raft::KeyValuePair<IdxT, AccT>>) {
      // OutT is a RAFT KeyValuePair
      raft::matrix::fill(
        handle, raft::make_device_matrix_view(out.data_handle(), m, IdxT{1}), OutT{0, 0});
    } else {
      // OutT is a scalar type
      raft::matrix::fill(
        handle, raft::make_device_matrix_view(out.data_handle(), m, IdxT{1}), OutT{0});
    }
    raft::resource::sync_stream(handle, stream);
  }

  void compute_1nn()
  {
    raft::device_vector<char, IdxT> workspace =
      raft::make_device_vector<char, IdxT>(handle, workspace_size);

    ref_nn<DataT, AccT, OutT, IdxT>(
      handle, ref_out.data_handle(), x.data_handle(), y.data_handle(), m, n, k, sqrt, metric);

    if constexpr (impl == ImplType::fused) {
      if (backend_unavailable) { GTEST_SKIP() << "Requested top_1_nn path is unavailable"; }
      auto run_top_1_nn = [&](auto output) {
        cuvs::distance::top_1_nn<DataT, IdxT>(handle,
                                              output,
                                              x.data_handle(),
                                              y.data_handle(),
                                              x_norm.data_handle(),
                                              y_norm.data_handle(),
                                              m,
                                              n,
                                              k,
                                              tuning,
                                              (void*)workspace.data_handle(),
                                              workspace_size,
                                              sqrt,
                                              true,
                                              true,
                                              metric,
                                              0.0,
                                              plan);
      };
      if (plan.output_layout == cuvs::distance::detail::Top1nnOutputLayout::Separate) {
        if constexpr (std::is_same_v<DataT, float> || std::is_same_v<DataT, half>) {
          run_top_1_nn(cuvs::distance::Top1nnOutput<IdxT, AccT>{cutile_idx.data_handle(),
                                                                cutile_dist.data_handle()});
        } else {
          RAFT_FAIL("cuTile top_1_nn test requires FP16 or FP32 data");
        }
      } else if constexpr (std::is_same_v<DataT, float>) {
        run_top_1_nn(out.data_handle());
      } else {
        RAFT_FAIL("Legacy fused top_1_nn test requires FP32 data");
      }
    } else if constexpr (impl == ImplType::unfused) {
      cuvs::distance::unfusedDistanceNNMinReduce<DataT, AccT, OutT, IdxT>(
        handle,
        out.data_handle(),
        x.data_handle(),
        y.data_handle(),
        x_norm.data_handle(),
        y_norm.data_handle(),
        m,
        n,
        k,
        (AccT*)workspace.data_handle(),
        sqrt,
        true,
        true,
        metric,
        0.0);
    }
  }

  void compare()
  {
    if constexpr (impl == ImplType::fused) {
      if (plan.output_layout == cuvs::distance::detail::Top1nnOutputLayout::Separate) {
        // cuTile MMA arithmetic can produce a different index for nearly tied candidates.
        // Validate that the returned index selects a candidate within the same numerical tolerance
        // of the true optimum.
        raft::linalg::unaryOp(
          ref_dist.data_handle(), ref_out.data_handle(), m, raft::value_op{}, stream.get());
        ref_nn_selected<DataT, AccT, IdxT>(handle,
                                           selected_dist.data_handle(),
                                           cutile_idx.data_handle(),
                                           x.data_handle(),
                                           y.data_handle(),
                                           m,
                                           n,
                                           k,
                                           sqrt,
                                           metric);
        ASSERT_TRUE(cuvs::devArrMatch(ref_dist.data_handle(),
                                      selected_dist.data_handle(),
                                      m,
                                      cuvs::CompareApproxNoScaling<AccT>{AccT(params_.tol)},
                                      stream.get()));
        ASSERT_TRUE(cuvs::devArrMatch(ref_dist.data_handle(),
                                      cutile_dist.data_handle(),
                                      m,
                                      cuvs::CompareApproxNoScaling<AccT>{AccT(params_.tol)},
                                      stream.get()));
        return;
      }
      vector_compare(handle, ref_out.data_handle(), out.data_handle(), m, summary);
    } else {
      vector_compare(handle, ref_out.data_handle(), out.data_handle(), m, summary);
    }
    ASSERT_TRUE(summary.max_diff < params_.tol) << summary;
  }

 private:
  raft::resources handle;
  cuda::stream_ref stream;
  NNInputs<IdxT> params_;
  ComparisonSummary summary;
  IdxT m;
  IdxT n;
  IdxT k;
  DistanceType metric;
  bool sqrt;
  cuvs::distance::detail::Top1nnBackend backend;
  cuvs::distance::detail::Top1nnTuning tuning;
  raft::device_matrix<DataT, IdxT> x;
  raft::device_matrix<DataT, IdxT> y;
  raft::device_vector<AccT, IdxT> x_norm;
  raft::device_vector<AccT, IdxT> y_norm;
  raft::device_vector<OutT, IdxT> out;
  raft::device_vector<OutT, IdxT> ref_out;
  raft::device_vector<AccT, IdxT> ref_dist;
  raft::device_vector<AccT, IdxT> selected_dist;
  cuvs::distance::detail::Top1nnPlan<IdxT> plan{};
  bool backend_unavailable{};
  raft::device_vector<IdxT, IdxT> cutile_idx;
  raft::device_vector<AccT, IdxT> cutile_dist;
  size_t workspace_size;
};

template <typename IdxT>
const std::vector<NNInputs<IdxT>> input_fp32 = {
  {4096, 4096, 64, DistanceType::L2Expanded, false, uint64_t(31415926), 0.1},
  {16384, 4096, 64, DistanceType::L2Expanded, false, uint64_t(31415926), 0.1},
  {4096, 4096, 128, DistanceType::L2Expanded, true, uint64_t(31415926), 0.1},
  {4096, 16384, 128, DistanceType::L2Expanded, true, uint64_t(31415926), 0.1},
  {4096, 4096, 64, DistanceType::L2SqrtExpanded, false, uint64_t(31415926), 0.1},
  {4096, 16384, 128, DistanceType::L2SqrtExpanded, false, uint64_t(31415926), 0.1},
  {4096, 4096, 64, DistanceType::CosineExpanded, false, uint64_t(31415926), 0.1},
  {8192, 4096, 64, DistanceType::CosineExpanded, false, uint64_t(31415926), 0.1},
  // Fused implementation for cosine distance ignores the sqrt parameter, therefore
  // commenting the following two tests
  // {4096, 4096, 128, DistanceType::CosineExpanded, true, uint64_t(31415926), 0.1},
  // {4096, 8192, 128, DistanceType::CosineExpanded, true, uint64_t(31415926), 0.1},
};

template <typename IdxT>
const std::vector<NNInputs<IdxT>> input_fp32_fused = [] {
  auto inputs = input_fp32<IdxT>;
  for (auto input : input_fp32<IdxT>) {
    input.backend = cuvs::distance::detail::Top1nnBackend::Auto;
    inputs.push_back(input);
  }
  for (auto input : input_fp32<IdxT>) {
    input.backend = cuvs::distance::detail::Top1nnBackend::Stable;
    inputs.push_back(input);
  }
  inputs.push_back({30,
                    20,
                    10,
                    DistanceType::L2Expanded,
                    false,
                    uint64_t(31415926),
                    0.1,
                    cuvs::distance::detail::Top1nnBackend::Auto});
  inputs.push_back({30,
                    20,
                    10,
                    DistanceType::L2Expanded,
                    false,
                    uint64_t(31415926),
                    0.1,
                    cuvs::distance::detail::Top1nnBackend::Stable});
  for (auto input : input_fp32<IdxT>) {
    input.backend = cuvs::distance::detail::Top1nnBackend::Unfused;
    inputs.push_back(input);
  }
#if CUVS_CUTILE_ENABLED
  for (auto input : input_fp32<IdxT>) {
    input.backend = cuvs::distance::detail::Top1nnBackend::Cutile;
    inputs.push_back(input);
  }
  // Non-vector-aligned k selects the relaxed ABI; InnerProduct exercises its argmax path.
  inputs.push_back({257,
                    263,
                    65,
                    DistanceType::InnerProduct,
                    false,
                    uint64_t(31415926),
                    0.1,
                    cuvs::distance::detail::Top1nnBackend::Cutile});
#endif
  return inputs;
}();

// Test fused implementation with single-precision
typedef NNTest<float, float, int32_t, ImplType::fused> NNTest_fp32_fused;
TEST_P(NNTest_fp32_fused, test)
{
  this->compute_1nn();
  this->compare();
}

INSTANTIATE_TEST_CASE_P(NNTest, NNTest_fp32_fused, ::testing::ValuesIn(input_fp32_fused<int>));

TEST(Top1nnAutoSelection, LegacyHeuristic)
{
  using Backend = cuvs::distance::detail::Top1nnBackend;
  EXPECT_EQ(cuvs::distance::detail::legacy_top_1_nn_backend(7, 32, 32, DistanceType::L2Expanded),
            Backend::Unfused);
  EXPECT_EQ(cuvs::distance::detail::legacy_top_1_nn_backend(8, 32, 32, DistanceType::L2Expanded),
            Backend::Cutlass);
  EXPECT_EQ(
    cuvs::distance::detail::legacy_top_1_nn_backend(9, 4096, 32, DistanceType::CosineExpanded),
    Backend::Cutlass);
  EXPECT_EQ(cuvs::distance::detail::legacy_top_1_nn_backend(9, 32, 32, DistanceType::L2Expanded),
            Backend::Unfused);
  EXPECT_EQ(
    cuvs::distance::detail::legacy_top_1_nn_backend(10, 8192, 8192, DistanceType::L2Expanded),
    Backend::Unfused);
}

TEST(Top1nnStableSelection, ExcludesCutile)
{
  using Backend = cuvs::distance::detail::Top1nnBackend;
  raft::resources handle;
  constexpr int m       = 32;
  constexpr int n       = 32;
  constexpr int k       = 4;
  auto x                = raft::make_device_matrix<float, int>(handle, m, k);
  auto y                = raft::make_device_matrix<float, int>(handle, n, k);
  const auto plan       = cuvs::distance::probe_top_1_nn(handle,
                                                   x.data_handle(),
                                                   y.data_handle(),
                                                   m,
                                                   n,
                                                   k,
                                                   cuvs::distance::detail::Top1nnTuning{},
                                                   DistanceType::L2Expanded,
                                                   Backend::Stable);
  const auto properties = raft::resource::get_device_properties(handle);
  EXPECT_TRUE(plan.available);
  EXPECT_EQ(plan.requested_backend, Backend::Stable);
  EXPECT_EQ(plan.backend,
            cuvs::distance::detail::legacy_top_1_nn_backend(
              properties.major, m, n, DistanceType::L2Expanded));
  EXPECT_NE(plan.backend, Backend::Cutile);
}

#if CUVS_CUTILE_ENABLED
TEST(Top1nnWorkspace, CutileInt64IndexConversionContract)
{
  using Backend = cuvs::distance::detail::Top1nnBackend;
  cuvs::distance::detail::Top1nnTuning tuning{};
  constexpr int64_t m       = 257;
  constexpr int64_t n       = 31;
  constexpr int64_t k       = 16;
  const auto workspace_size = [&](
                                int64_t rows, int64_t candidates, int64_t features, bool indices) {
    return cuvs::distance::top_1_nn_workspace_size<float, int64_t>(
      rows, candidates, features, tuning, Backend::Cutile, indices);
  };

  EXPECT_EQ(workspace_size(m, n, k, true), static_cast<std::size_t>(m) * sizeof(int));
  EXPECT_EQ(workspace_size(m, n, k, false), 0);

  constexpr int64_t batched_m = cuvs::distance::detail::fused_1nn_cutile_max_batch_m<float> + 17;
  EXPECT_EQ(workspace_size(batched_m, n, k, true),
            static_cast<std::size_t>(cuvs::distance::detail::fused_1nn_cutile_max_batch_m<float>) *
              sizeof(int));

  constexpr int64_t too_large = static_cast<int64_t>(std::numeric_limits<int>::max()) + 1;
  EXPECT_ANY_THROW(workspace_size(m, too_large, k, true));
  EXPECT_ANY_THROW(workspace_size(m, n, too_large, true));
}
#endif

TEST(Top1nnPlan, RejectsMismatchedLaunch)
{
  raft::resources handle;
  constexpr int m = 2;
  constexpr int n = 2;
  constexpr int k = 2;
  auto x          = raft::make_device_matrix<float, int>(handle, m, k + 1);
  auto y          = raft::make_device_matrix<float, int>(handle, n, k + 1);
  auto x_norm     = raft::make_device_vector<float, int>(handle, m);
  auto y_norm     = raft::make_device_vector<float, int>(handle, n);
  auto output     = raft::make_device_vector<raft::KeyValuePair<int, float>, int>(handle, m);
  cuvs::distance::detail::Top1nnTuning tuning{};
  const auto plan = cuvs::distance::probe_top_1_nn(handle,
                                                   x.data_handle(),
                                                   y.data_handle(),
                                                   m,
                                                   n,
                                                   k,
                                                   tuning,
                                                   DistanceType::L2Expanded,
                                                   cuvs::distance::detail::Top1nnBackend::Unfused);
  auto workspace  = raft::make_device_vector<char, int>(handle, plan.workspace_bytes);
  EXPECT_ANY_THROW((cuvs::distance::top_1_nn<float, int>(handle,
                                                         output.data_handle(),
                                                         x.data_handle(),
                                                         y.data_handle(),
                                                         x_norm.data_handle(),
                                                         y_norm.data_handle(),
                                                         m,
                                                         n,
                                                         k + 1,
                                                         tuning,
                                                         workspace.data_handle(),
                                                         workspace.size(),
                                                         false,
                                                         true,
                                                         true,
                                                         DistanceType::L2Expanded,
                                                         0.0f,
                                                         plan)));
}

TEST(Top1nnPlan, ExplicitUnavailableBackendIsStrict)
{
  raft::resources handle;
  auto x = raft::make_device_matrix<half, int>(handle, 2, 2);
  auto y = raft::make_device_matrix<half, int>(handle, 2, 2);
  EXPECT_ANY_THROW(cuvs::distance::probe_top_1_nn(handle,
                                                  x.data_handle(),
                                                  y.data_handle(),
                                                  2,
                                                  2,
                                                  2,
                                                  cuvs::distance::detail::Top1nnTuning{},
                                                  DistanceType::L2Expanded,
                                                  cuvs::distance::detail::Top1nnBackend::Unfused));
}

TEST(Top1nnResultView, BindsCallerAllocatedOutputs)
{
  using Kvp = raft::KeyValuePair<int, float>;
  cuvs::distance::detail::Top1nnPlan<int> plan{};
  plan.output_layout    = cuvs::distance::detail::Top1nnOutputLayout::KeyValuePair;
  plan.output_alignment = alignof(Kvp);
  plan.output_bytes     = 2 * sizeof(Kvp);
  plan.m                = 2;

  Kvp key_values[2];
  auto kvp_result = cuvs::distance::make_top_1_nn_result_view<float>(plan, 2, key_values);
  kvp_result.visit_native([](int*, float*) { FAIL() << "KVP result visited as separate output"; },
                          [&](Kvp* output) { EXPECT_EQ(output, key_values); });

  int indices[2];
  float distances[2];
  plan.output_layout    = cuvs::distance::detail::Top1nnOutputLayout::Separate;
  plan.output_alignment = alignof(float);
  plan.distance_offset  = 2 * sizeof(int);
  plan.output_bytes     = plan.distance_offset + 2 * sizeof(float);
  auto separate_result  = cuvs::distance::make_top_1_nn_result_view<float>(
    plan, 2, cuvs::distance::Top1nnOutput<int, float>{indices, distances});
  separate_result.visit_native(
    [&](int* output_indices, float* output_distances) {
      EXPECT_EQ(output_indices, indices);
      EXPECT_EQ(output_distances, distances);
    },
    [](Kvp*) { FAIL() << "Separate result visited as KVP output"; });

  EXPECT_ANY_THROW(cuvs::distance::make_top_1_nn_result_view<float>(plan, 2, key_values));
}

TEST(Top1nnResultView, BindsAlignedCallerStorage)
{
  constexpr int size                    = 2;
  constexpr std::size_t alignment       = 16;
  constexpr std::size_t distance_offset = alignment;
  constexpr std::size_t output_bytes    = distance_offset + size * sizeof(float);
  alignas(alignment) char storage[output_bytes];

  cuvs::distance::detail::Top1nnPlan<int> plan{};
  plan.output_layout    = cuvs::distance::detail::Top1nnOutputLayout::Separate;
  plan.output_alignment = alignment;
  plan.distance_offset  = distance_offset;
  plan.output_bytes     = output_bytes;
  plan.m                = size;

  auto result =
    cuvs::distance::bind_top_1_nn_result_view<float>(plan, size, storage, sizeof(storage));
  result.visit_native(
    [&](int* indices, float* distances) {
      EXPECT_EQ(static_cast<void*>(indices), static_cast<void*>(storage));
      EXPECT_EQ(static_cast<void*>(distances), static_cast<void*>(storage + distance_offset));
    },
    [](raft::KeyValuePair<int, float>*) { FAIL() << "Separate storage bound as KVP output"; });

  EXPECT_ANY_THROW(
    cuvs::distance::bind_top_1_nn_result_view<float>(plan, size, storage, output_bytes - 1));
}

#if CUVS_CUTILE_ENABLED
const std::vector<NNInputs<int64_t>> input_fp32_cutile_i64 = [] {
  auto input    = input_fp32<int64_t>.front();
  input.backend = cuvs::distance::detail::Top1nnBackend::Cutile;
  return std::vector<NNInputs<int64_t>>{input};
}();

using NNTest_fp32_fused_i64 = NNTest<float, float, int64_t, ImplType::fused>;
TEST_P(NNTest_fp32_fused_i64, test)
{
  this->compute_1nn();
  this->compare();
}

INSTANTIATE_TEST_CASE_P(NNTest, NNTest_fp32_fused_i64, ::testing::ValuesIn(input_fp32_cutile_i64));
#endif

// Test unfused implementation with single-precision
typedef NNTest<float, float, int32_t, ImplType::unfused> NNTest_fp32_unfused;
TEST_P(NNTest_fp32_unfused, test)
{
  this->compute_1nn();
  this->compare();
}

INSTANTIATE_TEST_CASE_P(NNTest, NNTest_fp32_unfused, ::testing::ValuesIn(input_fp32<int>));

template <typename IdxT>
const std::vector<NNInputs<IdxT>> input_fp16 = {
  {4096, 4096, 64, DistanceType::L2Expanded, false, uint64_t(31415926), 0.1},
  {4096, 16384, 128, DistanceType::L2Expanded, true, uint64_t(31415926), 0.1},
  {4096, 4096, 64, DistanceType::L2SqrtExpanded, false, uint64_t(31415926), 0.1},
  {4096, 16384, 128, DistanceType::L2SqrtExpanded, false, uint64_t(31415926), 0.1},
  {4096, 4096, 64, DistanceType::CosineExpanded, false, uint64_t(31415926), 0.1},
  {4096, 16384, 128, DistanceType::CosineExpanded, true, uint64_t(31415926), 0.1},
};

#if CUVS_CUTILE_ENABLED
template <typename IdxT>
// k=64 and k=65 select the strict and relaxed FP16 ABI variants, respectively.
const std::vector<NNInputs<IdxT>> input_fp16_cutile = {
  {257,
   263,
   64,
   DistanceType::L2Expanded,
   false,
   uint64_t(31415926),
   0.1,
   cuvs::distance::detail::Top1nnBackend::Cutile},
  {257,
   263,
   65,
   DistanceType::CosineExpanded,
   false,
   uint64_t(31415926),
   0.1,
   cuvs::distance::detail::Top1nnBackend::Cutile},
  {257,
   263,
   64,
   DistanceType::L2Expanded,
   false,
   uint64_t(31415926),
   0.1,
   cuvs::distance::detail::Top1nnBackend::Auto},
  {257,
   263,
   65,
   DistanceType::CosineExpanded,
   false,
   uint64_t(31415926),
   0.1,
   cuvs::distance::detail::Top1nnBackend::Auto},
};

using NNTest_fp16_fused = NNTest<half, float, int32_t, ImplType::fused>;
TEST_P(NNTest_fp16_fused, test)
{
  this->compute_1nn();
  this->compare();
}

INSTANTIATE_TEST_CASE_P(NNTest, NNTest_fp16_fused, ::testing::ValuesIn(input_fp16_cutile<int>));
#endif

// Test unfused implementation with fp16, int8
// Legacy fused implementation has no support for fp16, int8
typedef NNTest<half, float, int32_t, ImplType::unfused> NNTest_fp16_unfused;
TEST_P(NNTest_fp16_unfused, test)
{
  this->compute_1nn();
  this->compare();
}

INSTANTIATE_TEST_CASE_P(NNTest, NNTest_fp16_unfused, ::testing::ValuesIn(input_fp16<int>));

template <typename IdxT>
const std::vector<NNInputs<IdxT>> input_int8 = {
  {4096, 4096, 64, DistanceType::L2Expanded, false, uint64_t(31415926), 0.1},
  {4096, 16384, 128, DistanceType::L2Expanded, true, uint64_t(31415926), 0.1},
  {4096, 4096, 64, DistanceType::L2SqrtExpanded, false, uint64_t(31415926), 0.1},
  {4096, 16384, 128, DistanceType::L2SqrtExpanded, false, uint64_t(31415926), 0.1},
  {4096, 4096, 64, DistanceType::CosineExpanded, false, uint64_t(31415926), 0.1},
  {4096, 16384, 128, DistanceType::CosineExpanded, true, uint64_t(31415926), 0.1},
};

// DataT = int8_t, AccT = int32_t
typedef NNTest<int8_t, int32_t, int32_t, ImplType::unfused> NNTest_int8_unfused;
TEST_P(NNTest_int8_unfused, test)
{
  this->compute_1nn();
  this->compare();
}

INSTANTIATE_TEST_CASE_P(NNTest, NNTest_int8_unfused, ::testing::ValuesIn(input_int8<int>));

// DataT = int8_t, AccT = float
typedef NNTest<int8_t, float, int32_t, ImplType::unfused> NNTest_int8_unfused2;
TEST_P(NNTest_int8_unfused2, test)
{
  this->compute_1nn();
  this->compare();
}

INSTANTIATE_TEST_CASE_P(NNTest, NNTest_int8_unfused2, ::testing::ValuesIn(input_int8<int>));
}  // namespace cuvs::neighbors
