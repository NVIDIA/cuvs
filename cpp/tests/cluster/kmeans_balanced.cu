/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../src/cluster/detail/kmeans_balanced.cuh"
#include "../test_utils.h"

#include <cuvs/cluster/kmeans.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/random/rng.cuh>
#include <raft/stats/adjusted_rand_index.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <thrust/fill.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

/* This test takes advantage of the fact that make_blobs generates balanced clusters.
 * It doesn't currently test whether the algorithm can make balanced clusters with an imbalanced
 * dataset.
 */

namespace cuvs {

template <typename MathT, typename IdxT>
struct KmeansBalancedInputs {
  IdxT n_rows;
  IdxT n_cols;
  IdxT n_clusters;
  cuvs::cluster::kmeans::balanced_params kb_params;
  MathT tol;
};

template <typename MathT, typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const KmeansBalancedInputs<MathT, IdxT>& p)
{
  os << "{ " << p.n_rows << ", " << p.n_cols << ", " << p.n_clusters << ", " << p.kb_params.n_iters
     << static_cast<int>(p.kb_params.metric) << '}';
  return os;
}

template <typename DataT,
          typename MathT,
          typename LabelT,
          typename IdxT,
          typename MappingOpT,
          bool SeparateFitPredict>
class KmeansBalancedTest : public ::testing::TestWithParam<KmeansBalancedInputs<MathT, IdxT>> {
 protected:
  KmeansBalancedTest()
    : stream(raft::resource::get_cuda_stream(handle).get()),
      d_labels(0, stream),
      d_labels_ref(0, stream),
      d_centroids(0, stream)
  {
  }

  void basicTest()
  {
    MappingOpT op{};

    auto p = ::testing::TestWithParam<KmeansBalancedInputs<MathT, IdxT>>::GetParam();

    auto X           = raft::make_device_matrix<DataT, IdxT>(handle, p.n_rows, p.n_cols);
    auto blob_labels = raft::make_device_vector<IdxT, IdxT>(handle, p.n_rows);

    MathT* blobs_ptr;
    rmm::device_uvector<MathT> blobs(0, stream);
    if constexpr (!std::is_same_v<DataT, MathT>) {
      blobs.resize(p.n_rows * p.n_cols, stream);
      blobs_ptr = blobs.data();
    } else {
      blobs_ptr = X.data_handle();
    }

    raft::random::make_blobs<MathT, IdxT>(blobs_ptr,
                                          blob_labels.data_handle(),
                                          p.n_rows,
                                          p.n_cols,
                                          p.n_clusters,
                                          stream,
                                          true,
                                          nullptr,
                                          nullptr,
                                          MathT{0.1},
                                          true,
                                          MathT{-1},
                                          MathT{1},
                                          (uint64_t)1234);

    // Convert blobs dataset to DataT if necessary
    if constexpr (!std::is_same_v<DataT, MathT>) {
      raft::linalg::unaryOp(
        X.data_handle(), blobs.data(), p.n_rows * p.n_cols, op.reverse_op, stream);
      // Free the MathT staging copy now; keeping it alive during clustering more than
      // quadruples the peak device memory for the int8 cases.
      blobs.resize(0, stream);
      blobs.shrink_to_fit(stream);
    }

    d_labels.resize(p.n_rows, stream);
    d_labels_ref.resize(p.n_rows, stream);
    d_centroids.resize(p.n_clusters * p.n_cols, stream);

    raft::linalg::unaryOp(
      d_labels_ref.data(), blob_labels.data_handle(), p.n_rows, raft::cast_op<LabelT>(), stream);

    auto X_view =
      raft::make_device_matrix_view<const DataT, IdxT>(X.data_handle(), X.extent(0), X.extent(1));
    auto d_centroids_view =
      raft::make_device_matrix_view<MathT, IdxT>(d_centroids.data(), p.n_clusters, p.n_cols);
    auto d_labels_view = raft::make_device_vector_view<LabelT, IdxT>(d_labels.data(), p.n_rows);

    if constexpr (SeparateFitPredict) {
      cuvs::cluster::kmeans::fit(handle, p.kb_params, X_view, d_centroids_view);
      cuvs::cluster::kmeans::predict(
        handle, p.kb_params, X_view, raft::make_const_mdspan(d_centroids_view), d_labels_view);
    } else {
      cuvs::cluster::kmeans::fit_predict(
        handle, p.kb_params, X_view, d_centroids_view, d_labels_view);
    }

    raft::resource::sync_stream(handle, stream);

    score = raft::stats::adjusted_rand_index(d_labels_ref.data(),
                                             d_labels.data(),
                                             p.n_rows,
                                             raft::resource::get_cuda_stream(handle).get());

    if (score < 1.0) {
      std::stringstream ss;
      ss << "Expected: " << raft::arr2Str(d_labels_ref.data(), 25, "d_labels_ref", stream);
      std::cout << (ss.str().c_str()) << '\n';
      ss.str(std::string());
      ss << "Actual: " << raft::arr2Str(d_labels.data(), 25, "d_labels", stream);
      std::cout << (ss.str().c_str()) << '\n';
      std::cout << "Score = " << score << '\n';
    }
  }

  void SetUp() override { basicTest(); }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;
  rmm::device_uvector<LabelT> d_labels;
  rmm::device_uvector<LabelT> d_labels_ref;
  rmm::device_uvector<MathT> d_centroids;
  double score;
};

template <typename MathT, typename IdxT>
std::vector<KmeansBalancedInputs<MathT, IdxT>> get_kmeans_balanced_inputs()
{
  std::vector<KmeansBalancedInputs<MathT, IdxT>> out;
  KmeansBalancedInputs<MathT, IdxT> p;
  p.kb_params.n_iters = 20;
  p.kb_params.metric  = cuvs::distance::DistanceType::L2Expanded;
  p.tol               = MathT{0.0001};
  std::vector<std::tuple<size_t, size_t, size_t>> row_cols_k = {
    {1000, 32, 5},
    {1000, 100, 20},
    {10000, 32, 10},
    {10000, 100, 50},
    {10000, 500, 100},
    {1000000, 128, 10},
    {10000000, 128, 10},
  };
  for (auto& rck : row_cols_k) {
    p.n_rows     = static_cast<IdxT>(std::get<0>(rck));
    p.n_cols     = static_cast<IdxT>(std::get<1>(rck));
    p.n_clusters = static_cast<IdxT>(std::get<2>(rck));
    out.push_back(p);
  }
  return out;
}

template <typename MathT, typename IdxT>
std::vector<KmeansBalancedInputs<MathT, IdxT>> get_kmeans_balanced_cosine_inputs()
{
  std::vector<KmeansBalancedInputs<MathT, IdxT>> out;
  KmeansBalancedInputs<MathT, IdxT> p;
  p.kb_params.n_iters = 20;
  p.kb_params.metric  = cuvs::distance::DistanceType::CosineExpanded;
  p.tol               = MathT{0.0001};
  std::vector<std::tuple<size_t, size_t, size_t>> row_cols_k = {
    {1000, 32, 5},
    {1000, 100, 20},
    {10000, 32, 10},
    {10000, 100, 50},
  };
  for (auto& rck : row_cols_k) {
    p.n_rows     = static_cast<IdxT>(std::get<0>(rck));
    p.n_cols     = static_cast<IdxT>(std::get<1>(rck));
    p.n_clusters = static_cast<IdxT>(std::get<2>(rck));
    out.push_back(p);
  }
  return out;
}

const auto inputsf_i32 = get_kmeans_balanced_inputs<float, int>();
// const auto inputsd_i32 = get_kmeans_balanced_inputs<double, int>();
const auto inputsf_i64 = get_kmeans_balanced_inputs<float, int64_t>();
// const auto inputsd_i64 = get_kmeans_balanced_inputs<double, int64_t>();
const auto inputsf_cosine_i32 = get_kmeans_balanced_cosine_inputs<float, int>();
const std::vector<KmeansBalancedInputs<float, int64_t>> inputsh_i64 = [] {
  KmeansBalancedInputs<float, int64_t> l2{};
  l2.n_rows               = 1000;
  l2.n_cols               = 32;
  l2.n_clusters           = 5;
  l2.kb_params.n_iters    = 20;
  l2.kb_params.metric     = cuvs::distance::DistanceType::L2Expanded;
  l2.tol                  = 0.001f;
  auto cosine             = l2;
  cosine.kb_params.metric = cuvs::distance::DistanceType::CosineExpanded;
  return std::vector<KmeansBalancedInputs<float, int64_t>>{l2, cosine};
}();

struct half_to_float {
  raft::cast_op<half> reverse_op{};
  RAFT_INLINE_FUNCTION float operator()(half value) const { return static_cast<float>(value); }
};

#define KB_TEST(test_type, test_name, test_inputs)         \
  typedef RAFT_DEPAREN(test_type) test_name;               \
  TEST_P(test_name, Result) { ASSERT_TRUE(score == 1.0); } \
  INSTANTIATE_TEST_CASE_P(KmeansBalancedTests, test_name, ::testing::ValuesIn(test_inputs))

/*
 * First set of tests: no conversion
 */

KB_TEST((KmeansBalancedTest<float, float, uint32_t, int, raft::identity_op, false>),
        KmeansBalancedTestFFU32I32,
        inputsf_i32);
KB_TEST((KmeansBalancedTest<float, float, int, int, raft::identity_op, true>),
        KmeansBalancedTestFFI32I32_SEP,
        inputsf_i32);
// KB_TEST((KmeansBalancedTest<double, double, uint32_t, int, raft::identity_op>),
//         KmeansBalancedTestDDU32I32,
//         inputsd_i32);
KB_TEST((KmeansBalancedTest<float, float, uint32_t, int64_t, raft::identity_op, false>),
        KmeansBalancedTestFFU32I64,
        inputsf_i64);
KB_TEST((KmeansBalancedTest<float, float, int, int64_t, raft::identity_op, true>),
        KmeansBalancedTestFFI32I64_SEP,
        inputsf_i64);
// KB_TEST((KmeansBalancedTest<double, double, uint32_t, int64_t, raft::identity_op>),
//         KmeansBalancedTestDDU32I64,
//         inputsd_i64);
// KB_TEST((KmeansBalancedTest<float, float, int, int, raft::identity_op>),
//         KmeansBalancedTestFFI32I32,
//         inputsf_i32);
// KB_TEST((KmeansBalancedTest<float, float, int, int64_t, raft::identity_op>),
//         KmeansBalancedTestFFI32I64,
//         inputsf_i64);
// KB_TEST((KmeansBalancedTest<float, float, int64_t, int, raft::identity_op>),
//         KmeansBalancedTestFFI64I32,
//         inputsf_i32);
// KB_TEST((KmeansBalancedTest<float, float, int64_t, int64_t, raft::identity_op>),
//         KmeansBalancedTestFFI64I64,
//         inputsf_i64);
KB_TEST((KmeansBalancedTest<float, float, uint32_t, int, raft::identity_op, false>),
        KmeansBalancedTestCosineFFU32I32,
        inputsf_cosine_i32);

/*
 * Second set of tests: integer dataset with conversion
 */

template <typename DataT, typename MathT>
struct i2f_scaler {
  // Note: with a scaling factor of 42, and generating blobs with centers between -1 and 1 with a
  // standard deviation of 0.1, it's statistically very unlikely that we'd overflow
  const raft::compose_op<raft::div_const_op<MathT>, raft::cast_op<MathT>> op{
    raft::div_const_op<MathT>{42}, raft::cast_op<MathT>{}};
  const raft::compose_op<raft::cast_op<DataT>, raft::mul_const_op<MathT>> reverse_op{
    raft::cast_op<DataT>{}, raft::mul_const_op<MathT>{42}};

  RAFT_INLINE_FUNCTION auto operator()(const DataT& x) const { return op(x); };
};

KB_TEST((KmeansBalancedTest<half, float, uint32_t, int64_t, half_to_float, true>),
        KmeansBalancedTestHFU32I64_SEP,
        inputsh_i64);

KB_TEST((KmeansBalancedTest<int8_t, float, uint32_t, int, i2f_scaler<int8_t, float>, false>),
        KmeansBalancedTestFI8U32I32,
        inputsf_i32);
KB_TEST((KmeansBalancedTest<int8_t, float, int, int, i2f_scaler<int8_t, float>, true>),
        KmeansBalancedTestFI8I32I32_SEP,
        inputsf_i32);

/*
 * The balancing step must pick the same donor points for identical inputs: they may depend neither
 * on the order in which the GPU runs the warps nor on how many fits ran before in the process.
 * Most rows are packed in a few clusters and many clusters are empty, so that many receivers need a
 * donor.
 */
void check_adjust_centers_reproducible(cuvs::cluster::kmeans::balanced_donor_selection donor)
{
  raft::resources handle;
  auto stream           = raft::resource::get_cuda_stream(handle);
  const int64_t n_rows  = 10007;
  const int64_t dim     = 8;
  const int64_t n_lists = 200;
  auto dataset          = raft::make_device_matrix<float, int64_t>(handle, n_rows, dim);
  raft::random::RngState rng(1234ULL);
  raft::random::uniform(handle, rng, dataset.data_handle(), n_rows * dim, -1.0f, 1.0f);
  std::vector<float> h_centers(n_lists * dim);
  for (size_t i = 0; i < h_centers.size(); i++) {
    h_centers[i] = static_cast<float>(i % 97) / 97.0f;
  }

  // 95% of the rows in clusters [0, 10), the rest spread over [10, 60); clusters [60, 200) empty.
  std::vector<uint32_t> h_labels(n_rows);
  std::vector<uint32_t> h_sizes(n_lists, 0);
  for (int64_t i = 0; i < n_rows; i++) {
    h_labels[i] = i < n_rows * 95 / 100 ? i % 10 : 10 + i % 50;
    h_sizes[h_labels[i]]++;
  }
  auto labels = raft::make_device_vector<uint32_t, int64_t>(handle, n_rows);
  auto sizes  = raft::make_device_vector<uint32_t, int64_t>(handle, n_lists);
  raft::update_device(labels.data_handle(), h_labels.data(), n_rows, stream);
  raft::update_device(sizes.data_handle(), h_sizes.data(), n_lists, stream);

  const bool random     = donor == cuvs::cluster::kmeans::balanced_donor_selection::Random;
  const float lower_tol = random ? 0.25f : 0.7f;
  const float upper_tol = random ? 4.0f : 1.5f;
  const float offset    = random ? 0.05f : 0.2f;

  std::vector<std::vector<float>> results;
  for (int run = 0; run < 3; run++) {
    auto centers = raft::make_device_matrix<float, int64_t>(handle, n_lists, dim);
    raft::update_device(centers.data_handle(), h_centers.data(), h_centers.size(), stream);
    bool adjusted =
      cuvs::cluster::kmeans::detail::adjust_centers(handle,
                                                    centers.data_handle(),
                                                    n_lists,
                                                    dim,
                                                    dataset.data_handle(),
                                                    n_rows,
                                                    labels.data_handle(),
                                                    sizes.data_handle(),
                                                    lower_tol,
                                                    upper_tol,
                                                    offset,
                                                    donor,
                                                    /*balancing_step=*/3u,
                                                    raft::identity_op{},
                                                    rmm::mr::get_current_device_resource_ref());
    ASSERT_TRUE(adjusted);
    std::vector<float> h(centers.size());
    raft::update_host(h.data(), centers.data_handle(), h.size(), stream);
    raft::resource::sync_stream(handle);
    results.push_back(std::move(h));
  }
  // Both strategies move (at least) the first 60 empty clusters.
  for (int64_t c = 60; c < 120; c++) {
    ASSERT_NE(0, std::memcmp(&results[0][c * dim], &h_centers[c * dim], dim * sizeof(float)))
      << "cluster " << c << " was not moved";
  }
  for (int run = 1; run < 3; run++) {
    ASSERT_EQ(
      0, std::memcmp(results[0].data(), results[run].data(), results[0].size() * sizeof(float)))
      << "adjust_centers moved the centers differently for identical inputs (call " << run << ")";
  }
}

TEST(KmeansBalancedAdjustCenters, ReproducibleSizeSorted)
{
  check_adjust_centers_reproducible(cuvs::cluster::kmeans::balanced_donor_selection::SizeSorted);
}

TEST(KmeansBalancedAdjustCenters, ReproducibleRandomDonor)
{
  check_adjust_centers_reproducible(cuvs::cluster::kmeans::balanced_donor_selection::Random);
}

}  // namespace cuvs
