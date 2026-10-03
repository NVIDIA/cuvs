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
#include <raft/stats/adjusted_rand_index.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/fill.h>

#include <gtest/gtest.h>

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

// Packed input must behave exactly like explicitly expanding each bit to {-1, +1}.
TEST(KmeansBalancedBinary, DecodeAcrossByteAndRowBoundaries)
{
  const std::vector<uint8_t> packed{0x81, 0x56, 0xfe, 0x23, 0x00, 0xff};
  const cuvs::spatial::knn::detail::utils::bitwise_decode_op<float, int64_t> decode(packed.data());
  for (int64_t i = 0; i < 48; ++i) {
    EXPECT_EQ(decode(i), ((packed[i / 8] >> (i % 8)) & 1) ? 1.0f : -1.0f);
  }
}

TEST(KmeansBalancedBinary, MinibatchBudgetAlwaysMakesProgress)
{
  raft::resources handle;
  auto [batch, bytes_per_row] = cuvs::cluster::kmeans::detail::calc_minibatch_size<float, int64_t>(
    handle, 1, 128, int64_t{1} << 29, cuvs::distance::DistanceType::L2Expanded, true);
  EXPECT_EQ(batch, 1);
  EXPECT_GE(bytes_per_row, sizeof(float) * (size_t{1} << 29));
}

TEST(KmeansBalancedBinary, PackedDonorsPreserveBothBalancingStrategies)
{
  raft::resources handle;
  auto stream          = raft::resource::get_cuda_stream(handle);
  auto mr              = raft::resource::get_workspace_resource_ref(handle);
  constexpr int n_rows = 8, n_clusters = 3, packed_dim = 3, expanded_dim = packed_dim * 8;
  const std::vector<uint8_t> packed(n_rows * packed_dim, 0xa5);
  const std::vector<int> labels(n_rows, 1);
  const std::vector<uint32_t> counts{0, n_rows, 0};
  std::vector<float> initial(n_clusters * expanded_dim, -3.0f);
  for (int j = 0; j < expanded_dim; ++j) {
    initial[expanded_dim + j] = ((0xa5 >> (j % 8)) & 1) ? 1.0f : -1.0f;
  }
  auto X     = raft::make_device_matrix<uint8_t, int>(handle, n_rows, packed_dim);
  auto C     = raft::make_device_matrix<float, int>(handle, n_clusters, expanded_dim);
  auto L     = raft::make_device_vector<int, int>(handle, n_rows);
  auto sizes = raft::make_device_vector<uint32_t, int>(handle, n_clusters);
  raft::update_device(X.data_handle(), packed.data(), packed.size(), stream);
  raft::update_device(L.data_handle(), labels.data(), labels.size(), stream);
  raft::update_device(sizes.data_handle(), counts.data(), counts.size(), stream);
  using strategy = cuvs::cluster::kmeans::balanced_donor_selection;
  for (auto donor_selection : {strategy::SizeSorted, strategy::Random}) {
    raft::update_device(C.data_handle(), initial.data(), initial.size(), stream);
    auto decoded =
      cuvs::cluster::kmeans::detail::make_bitwise_expanded_iterator<float, int>(X.data_handle());
    ASSERT_TRUE(cuvs::cluster::kmeans::detail::adjust_centers(handle,
                                                              C.data_handle(),
                                                              n_clusters,
                                                              expanded_dim,
                                                              decoded,
                                                              n_rows,
                                                              L.data_handle(),
                                                              sizes.data_handle(),
                                                              0.333f,
                                                              3.0f,
                                                              0.01f,
                                                              donor_selection,
                                                              raft::identity_op{},
                                                              mr));
    std::vector<float> actual(initial.size());
    raft::update_host(actual.data(), C.data_handle(), actual.size(), stream);
    raft::resource::sync_stream(handle);
    for (int j = 0; j < expanded_dim; ++j) {
      EXPECT_EQ(actual[j], initial[expanded_dim + j]);
      EXPECT_EQ(actual[expanded_dim + j], initial[expanded_dim + j]);
      EXPECT_EQ(actual[2 * expanded_dim + j],
                donor_selection == strategy::Random ? initial[expanded_dim + j] : -3.0f);
    }
  }
}

TEST(KmeansBalancedBinary, PredictMatchesExpandedInput)
{
  raft::resources handle;
  auto stream                  = raft::resource::get_cuda_stream(handle);
  constexpr int64_t n_rows     = 97;
  constexpr int64_t n_clusters = 7;
  for (int64_t packed_dim : {int64_t{1}, int64_t{3}, int64_t{192}}) {
    SCOPED_TRACE(packed_dim);
    const int64_t dim = packed_dim * 8;
    std::vector<uint8_t> packed(n_rows * packed_dim);
    std::vector<float> expanded(n_rows * dim);
    std::vector<float> centers(n_clusters * dim);
    for (size_t i = 0; i < packed.size(); ++i) {
      packed[i] = uint8_t((i * 73 + i / 7 + 19) % 256);
    }
    for (size_t i = 0; i < expanded.size(); ++i) {
      expanded[i] = ((packed[i / 8] >> (i % 8)) & 1) ? 1.0f : -1.0f;
    }
    for (size_t i = 0; i < centers.size(); ++i) {
      centers[i] = (int((i * 19 + i / dim * 5) % 31) - 15) / 16.0f;
    }
    auto X          = raft::make_device_matrix<uint8_t, int64_t>(handle, n_rows, packed_dim);
    auto X_expanded = raft::make_device_matrix<float, int64_t>(handle, n_rows, dim);
    auto C          = raft::make_device_matrix<float, int64_t>(handle, n_clusters, dim);
    auto labels     = raft::make_device_vector<uint32_t, int64_t>(handle, n_rows);
    auto expected   = raft::make_device_vector<uint32_t, int64_t>(handle, n_rows);
    raft::update_device(X.data_handle(), packed.data(), packed.size(), stream);
    raft::update_device(X_expanded.data_handle(), expanded.data(), expanded.size(), stream);
    raft::update_device(C.data_handle(), centers.data(), centers.size(), stream);
    for (auto metric : {cuvs::distance::DistanceType::L2Expanded,
                        cuvs::distance::DistanceType::L2SqrtExpanded,
                        cuvs::distance::DistanceType::CosineExpanded,
                        cuvs::distance::DistanceType::InnerProduct}) {
      SCOPED_TRACE(int(metric));
      cuvs::cluster::kmeans::balanced_params params;
      params.metric           = metric;
      params.is_packed_binary = true;
      cuvs::cluster::kmeans::predict(handle,
                                     params,
                                     raft::make_const_mdspan(X.view()),
                                     raft::make_const_mdspan(C.view()),
                                     labels.view());
      params.is_packed_binary = false;
      cuvs::cluster::kmeans::predict(handle,
                                     params,
                                     raft::make_const_mdspan(X_expanded.view()),
                                     raft::make_const_mdspan(C.view()),
                                     expected.view());
      std::vector<uint32_t> actual_labels(n_rows), expected_labels(n_rows);
      raft::update_host(actual_labels.data(), labels.data_handle(), n_rows, stream);
      raft::update_host(expected_labels.data(), expected.data_handle(), n_rows, stream);
      raft::resource::sync_stream(handle);
      EXPECT_EQ(actual_labels, expected_labels);
    }
  }
}

TEST(KmeansBalancedBinary, HierarchicalCentersPreserveEveryExpandedCoordinate)
{
  raft::resources handle;
  auto stream              = raft::resource::get_cuda_stream(handle);
  constexpr int64_t n_rows = 4096, packed_dim = 32, dim = 256, n_clusters = 256;
  std::vector<uint8_t> packed(n_rows * packed_dim);
  for (int64_t row = 0; row < n_rows; ++row) {
    // Initial mesocluster labels select the low four bits of the prototype ID.
    // Within each mesocluster, fine-cluster labels select its high four bits.
    // Every true centroid therefore has an exact {-1,+1} CPU oracle, with no
    // dependence on centroid ordering or on private library symbols.
    for (int64_t byte = 0; byte < packed_dim; ++byte) {
      packed[row * packed_dim + byte] = (row & (int64_t{1} << (byte / 4))) ? 0xff : 0;
    }
  }
  auto X      = raft::make_device_matrix<uint8_t, int64_t>(handle, n_rows, packed_dim);
  auto C      = raft::make_device_matrix<float, int64_t>(handle, n_clusters, dim);
  auto labels = raft::make_device_vector<uint32_t, int64_t>(handle, n_rows);
  raft::update_device(X.data_handle(), packed.data(), packed.size(), stream);
  raft::matrix::fill(handle, C.view(), 123.0f);
  cuvs::cluster::kmeans::balanced_params params;
  params.is_packed_binary = true;
  params.n_iters          = 1;
  cuvs::cluster::kmeans::fit(handle, params, raft::make_const_mdspan(X.view()), C.view());
  cuvs::cluster::kmeans::predict(handle,
                                 params,
                                 raft::make_const_mdspan(X.view()),
                                 raft::make_const_mdspan(C.view()),
                                 labels.view());
  std::vector<float> centers(n_clusters * dim);
  std::vector<uint32_t> actual_labels(n_rows);
  raft::update_host(centers.data(), C.data_handle(), centers.size(), stream);
  raft::update_host(actual_labels.data(), labels.data_handle(), actual_labels.size(), stream);
  raft::resource::sync_stream(handle);
  std::vector<bool> seen(n_clusters, false);
  for (int64_t row = 0; row < n_rows; ++row) {
    ASSERT_LT(actual_labels[row], n_clusters);
    seen[actual_labels[row]] = true;
    for (int64_t bit = 0; bit < dim; ++bit) {
      const auto byte = packed[row * packed_dim + bit / 8];
      ASSERT_EQ(centers[actual_labels[row] * dim + bit], ((byte >> (bit % 8)) & 1) ? 1.0f : -1.0f)
        << "row=" << row << ", bit=" << bit;
    }
  }
  EXPECT_EQ(std::count(seen.begin(), seen.end(), true), n_clusters);
}

TEST(KmeansBalancedBinary, FitAndPredictRecoverPackedClusters)
{
  raft::resources handle;
  auto stream              = raft::resource::get_cuda_stream(handle);
  constexpr int64_t n_rows = 128, packed_dim = 3, dim = 24, n_clusters = 4;
  std::vector<uint8_t> packed(n_rows * packed_dim);
  for (int64_t row = 0; row < n_rows; ++row) {
    packed[row * packed_dim]     = (row & 1) ? 0xff : 0;
    packed[row * packed_dim + 1] = (row & 2) ? 0xff : 0;
    packed[row * packed_dim + 2] = (row & 2) ? 0xa5 : 0x5a;
  }
  auto X      = raft::make_device_matrix<uint8_t, int64_t>(handle, n_rows, packed_dim);
  auto C      = raft::make_device_matrix<float, int64_t>(handle, n_clusters, dim);
  auto labels = raft::make_device_vector<uint32_t, int64_t>(handle, n_rows);
  raft::update_device(X.data_handle(), packed.data(), packed.size(), stream);
  cuvs::cluster::kmeans::balanced_params params;
  params.is_packed_binary = true;
  params.n_iters          = 2;
  cuvs::cluster::kmeans::fit(handle, params, raft::make_const_mdspan(X.view()), C.view());
  cuvs::cluster::kmeans::predict(handle,
                                 params,
                                 raft::make_const_mdspan(X.view()),
                                 raft::make_const_mdspan(C.view()),
                                 labels.view());
  std::vector<uint32_t> actual_labels(n_rows);
  std::vector<float> centers(n_clusters * dim);
  raft::update_host(actual_labels.data(), labels.data_handle(), n_rows, stream);
  raft::update_host(centers.data(), C.data_handle(), centers.size(), stream);
  raft::resource::sync_stream(handle);
  for (int64_t row = 0; row < n_rows; ++row) {
    ASSERT_LT(actual_labels[row], n_clusters);
    for (int64_t bit = 0; bit < dim; ++bit) {
      const auto byte = packed[row * packed_dim + bit / 8];
      EXPECT_EQ(centers[actual_labels[row] * dim + bit], ((byte >> (bit % 8)) & 1) ? 1.0f : -1.0f);
    }
  }
}

TEST(KmeansBalancedBinary, RejectsInvalidPackedTypeDimensionsAndOverflow)
{
  raft::resources handle;
  cuvs::cluster::kmeans::balanced_params params;
  params.is_packed_binary  = true;
  const auto X             = raft::make_device_matrix_view<const uint8_t, int64_t>(nullptr, 16, 3);
  const auto wrong_centers = raft::make_device_matrix_view<float, int64_t>(nullptr, 2, 3);
  EXPECT_THROW(cuvs::cluster::kmeans::fit(handle, params, X, wrong_centers), raft::logic_error);
  const auto float_X          = raft::make_device_matrix_view<const float, int64_t>(nullptr, 16, 3);
  const auto expanded_centers = raft::make_device_matrix_view<float, int64_t>(nullptr, 2, 24);
  EXPECT_THROW(cuvs::cluster::kmeans::fit(handle, params, float_X, expanded_centers),
               raft::logic_error);
  constexpr int64_t max_index = std::numeric_limits<int64_t>::max();
  const auto overflowing_dim =
    raft::make_device_matrix_view<const uint8_t, int64_t>(nullptr, 1, max_index / 8 + 1);
  EXPECT_THROW(cuvs::cluster::kmeans::fit(handle, params, overflowing_dim, expanded_centers),
               raft::logic_error);
  const auto overflowing_rows =
    raft::make_device_matrix_view<const uint8_t, int64_t>(nullptr, max_index / 24 + 1, 3);
  EXPECT_THROW(cuvs::cluster::kmeans::fit(handle, params, overflowing_rows, expanded_centers),
               raft::logic_error);
}

TEST(KmeansBalancedBinary, IncrementalCentersMatchSinglePass)
{
  raft::resources handle;
  auto stream              = raft::resource::get_cuda_stream(handle);
  constexpr int64_t n_rows = 9, packed_dim = 3, dim = 24, n_clusters = 3;
  std::vector<uint8_t> packed(n_rows * packed_dim);
  std::vector<uint32_t> labels(n_rows);
  for (size_t i = 0; i < packed.size(); ++i) {
    packed[i] = uint8_t(i * 73 + 19);
  }
  for (int64_t row = 0; row < n_rows; ++row) {
    labels[row] = row % n_clusters;
  }
  auto X     = raft::make_device_matrix<uint8_t, int64_t>(handle, n_rows, packed_dim);
  auto L     = raft::make_device_vector<uint32_t, int64_t>(handle, n_rows);
  auto C     = raft::make_device_matrix<float, int64_t>(handle, n_clusters, dim);
  auto sizes = raft::make_device_vector<uint32_t, int64_t>(handle, n_clusters);
  raft::update_device(X.data_handle(), packed.data(), packed.size(), stream);
  raft::update_device(L.data_handle(), labels.data(), labels.size(), stream);
  auto update = [&](int64_t offset, int64_t rows, bool reset) {
    cuvs::cluster::kmeans::detail::calc_centers_and_sizes(
      handle,
      C.data_handle(),
      sizes.data_handle(),
      n_clusters,
      packed_dim,
      X.data_handle() + offset * packed_dim,
      rows,
      L.data_handle() + offset,
      reset,
      true,
      raft::identity_op{},
      raft::resource::get_workspace_resource_ref(handle));
  };
  update(0, 4, true);
  update(4, 5, false);
  std::vector<float> actual(n_clusters * dim);
  std::vector<uint32_t> actual_sizes(n_clusters);
  raft::update_host(actual.data(), C.data_handle(), actual.size(), stream);
  raft::update_host(actual_sizes.data(), sizes.data_handle(), actual_sizes.size(), stream);
  raft::resource::sync_stream(handle);
  for (int64_t cluster = 0; cluster < n_clusters; ++cluster) {
    EXPECT_EQ(actual_sizes[cluster], 3);
    for (int64_t bit = 0; bit < dim; ++bit) {
      float expected = 0;
      for (int64_t row = cluster; row < n_rows; row += n_clusters) {
        const auto byte = packed[row * packed_dim + bit / 8];
        expected += ((byte >> (bit % 8)) & 1) ? 1.0f : -1.0f;
      }
      EXPECT_NEAR(actual[cluster * dim + bit], expected / 3.0f, 1e-6f);
    }
  }
}

TEST(KmeansBalancedBinary, NumericUint8PredictionRemainsNumeric)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);
  const std::vector<uint8_t> input{0, 32, 64, 127, 128, 129, 192, 224, 255};
  const std::vector<float> centers{0, 0.5f, 1};
  auto X      = raft::make_device_matrix<uint8_t, int64_t>(handle, 9, 1);
  auto C      = raft::make_device_matrix<float, int64_t>(handle, 3, 1);
  auto labels = raft::make_device_vector<uint32_t, int64_t>(handle, 9);
  raft::update_device(X.data_handle(), input.data(), input.size(), stream);
  raft::update_device(C.data_handle(), centers.data(), centers.size(), stream);
  cuvs::cluster::kmeans::balanced_params params;
  params.is_packed_binary = false;
  cuvs::cluster::kmeans::predict(handle,
                                 params,
                                 raft::make_const_mdspan(X.view()),
                                 raft::make_const_mdspan(C.view()),
                                 labels.view());
  std::vector<uint32_t> actual(9);
  raft::update_host(actual.data(), labels.data_handle(), actual.size(), stream);
  raft::resource::sync_stream(handle);
  for (size_t row = 0; row < input.size(); ++row) {
    ASSERT_LT(actual[row], centers.size());
    const float value    = input[row] / 256.0f;
    const float distance = std::abs(value - centers[actual[row]]);
    for (float center : centers) {
      EXPECT_LE(distance, std::abs(value - center));
    }
  }
}

}  // namespace cuvs
