/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cuvs/neighbors/ivf_flat.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <rmm/device_uvector.hpp>

#include "../ann_utils.cuh"

#include <array>
#include <chrono>
#include <concepts>
#include <vector>

namespace cuvs::neighbors::ivf_flat {

// ============================================================================
// Define custom metrics using the UDF macro
// ============================================================================

// Custom L2 (squared Euclidean) metric - should match built-in L2
CUVS_METRIC(custom_l2, { acc += squared_diff(x, y); })

// L∞ (Chebyshev): per dimension acc = max(acc, |x - y|); acc starts at 0 in the scan kernel.
// For float (Veclen == 1) each invocation sees one coordinate pair.
CUVS_METRIC(chebyshev_linf, {
  auto d = abs_diff(x, y);
  acc    = (d > acc) ? d : acc;
})

// ============================================================================
// Test data traits for different types
// ============================================================================

template <typename T>
concept udf_test_fp_element = std::same_as<T, float>;

template <typename T>
concept udf_test_int_byte_element = std::same_as<T, int8_t> || std::same_as<T, uint8_t>;

template <typename T>
struct TestDataTraits;

template <udf_test_fp_element T>
struct TestDataTraits<T> {
  static constexpr int64_t dim         = 4;
  static constexpr int64_t num_db_vecs = 8;

  static std::vector<T> database()
  {
    // 4-dimensional float dataset
    // Vectors arranged for easy distance verification:
    //   db[0] = [0, 0, 0, 0]  - origin
    //   db[1] = [1, 0, 0, 0]  - unit along x
    //   db[2] = [0, 1, 0, 0]  - unit along y
    //   db[3] = [0, 0, 1, 0]  - unit along z
    //   db[4] = [1, 1, 0, 0]  - diagonal in xy
    //   db[5] = [2, 0, 0, 0]  - 2 units along x
    //   db[6] = [1, 1, 1, 1]  - all ones
    //   db[7] = [3, 4, 0, 0]  - for 3-4-5 triangle
    return {
      0.0, 0.0, 0.0, 0.0,  // db[0]: origin
      1.0, 0.0, 0.0, 0.0,  // db[1]: L2 dist rom origin = 1
      0.0, 1.0, 0.0, 0.0,  // db[2]: L2 dist rom origin = 1
      0.0, 0.0, 1.0, 0.0,  // db[3]: L2 dist rom origin = 1
      1.0, 1.0, 0.0, 0.0,  // db[4]: L2 dist rom origin = 2
      2.0, 0.0, 0.0, 0.0,  // db[5]: L2 dist rom origin = 4
      1.0, 1.0, 1.0, 1.0,  // db[6]: L2 dist rom origin = 4
      3.0, 4.0, 0.0, 0.0,  // db[7]: L2 dist rom origin = 25
    };
  }

  static std::vector<T> queries()
  {
    // query[0] = origin - nearest is db[0] (dist=0)
    // query[1] = [1,0,0,0] - nearest is db[1] (dist=0)
    return {
      0.0f,
      0.0f,
      0.0f,
      0.0f,  // query[0]: origin
      1.0f,
      0.0f,
      0.0f,
      0.0f,  // query[1]: same as db[1]
    };
  }
};

template <udf_test_int_byte_element T>
struct TestDataTraits<T> {
  static constexpr int64_t dim         = 16;
  static constexpr int64_t num_db_vecs = 8;

  static std::vector<T> database()
  {
    // 16-dimensional int8/uint8 dataset to test vectorized SIMD intrinsics
    return {
      // db[0]: all zeros
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      // db[1]: unit in first dim
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      // db[2]: unit in second dim
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      // db[3]: all ones - L2 dist from zeros = 16
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      // db[4]: first 12 dims are 2 - L2 dist from zeros = 48
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      0,
      0,
      0,
      0,
      // db[5]: all twos - L2 dist from zeros = 64
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      // db[6]: alternating 1,0 - L2 dist from zeros = 8
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      // db[7]: alternating 0,1 - L2 dist from zeros = 8
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
    };
  }

  static std::vector<T> queries()
  {
    // query[0] = all zeros - nearest is db[0] (dist=0)
    // query[1] = all ones - nearest is db[3] (dist=0)
    return {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // query[0]
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // query[1]
    };
  }
};

// ============================================================================
// Templated test fixture
// ============================================================================

template <typename T>
class IvfFlatUdfTest : public ::testing::Test {
 protected:
  using Traits = TestDataTraits<T>;

  void SetUp() override
  {
    database_    = Traits::database();
    queries_     = Traits::queries();
    num_db_vecs_ = Traits::num_db_vecs;
    num_queries_ = 2;
    dim_         = Traits::dim;
    k_           = 4;
    n_lists_     = 2;
    n_probes_    = 2;
  }

  raft::resources handle_;
  std::vector<T> database_;
  std::vector<T> queries_;
  int64_t num_db_vecs_;
  int64_t num_queries_;
  int64_t dim_;
  int64_t k_;
  uint32_t n_lists_;
  uint32_t n_probes_;
};

using TestTypes = ::testing::Types<float, int8_t, uint8_t>;
TYPED_TEST_SUITE(IvfFlatUdfTest, TestTypes);

// ============================================================================
// Test: UDF L2 metric matches built-in L2 and produces correct distances
// ============================================================================

TYPED_TEST(IvfFlatUdfTest, CustomL2MatchesBuiltIn)
{
  using T      = TypeParam;
  using Traits = TestDataTraits<T>;

  auto stream = raft::resource::get_cuda_stream(this->handle_);

  // Copy data to device
  rmm::device_uvector<T> d_database(this->num_db_vecs_ * this->dim_, stream);
  rmm::device_uvector<T> d_queries(this->num_queries_ * this->dim_, stream);
  raft::copy(d_database.data(), this->database_.data(), this->database_.size(), stream);
  raft::copy(d_queries.data(), this->queries_.data(), this->queries_.size(), stream);

  auto database_view = raft::make_device_matrix_view<const T, int64_t>(
    d_database.data(), this->num_db_vecs_, this->dim_);
  auto queries_view = raft::make_device_matrix_view<const T, int64_t>(
    d_queries.data(), this->num_queries_, this->dim_);

  // Build index with L2 metric
  ivf_flat::index_params index_params;
  index_params.n_lists = this->n_lists_;
  index_params.metric  = cuvs::distance::DistanceType::L2Expanded;

  auto idx = ivf_flat::build(this->handle_, index_params, database_view);

  // Allocate output buffers
  rmm::device_uvector<int64_t> d_indices_builtin(this->num_queries_ * this->k_, stream);
  rmm::device_uvector<float> d_distances_builtin(this->num_queries_ * this->k_, stream);
  rmm::device_uvector<int64_t> d_indices_udf(this->num_queries_ * this->k_, stream);
  rmm::device_uvector<float> d_distances_udf(this->num_queries_ * this->k_, stream);

  auto indices_builtin_view = raft::make_device_matrix_view<int64_t, int64_t>(
    d_indices_builtin.data(), this->num_queries_, this->k_);
  auto distances_builtin_view = raft::make_device_matrix_view<float, int64_t>(
    d_distances_builtin.data(), this->num_queries_, this->k_);
  auto indices_udf_view = raft::make_device_matrix_view<int64_t, int64_t>(
    d_indices_udf.data(), this->num_queries_, this->k_);
  auto distances_udf_view = raft::make_device_matrix_view<float, int64_t>(
    d_distances_udf.data(), this->num_queries_, this->k_);

  // Search with built-in metric
  ivf_flat::search_params search_params_builtin;
  search_params_builtin.n_probes = this->n_probes_;

  ivf_flat::search(this->handle_,
                   search_params_builtin,
                   idx,
                   queries_view,
                   indices_builtin_view,
                   distances_builtin_view);

  // Search with custom UDF metric (twice: first run pays JIT/link; second should be faster)
  ivf_flat::search_params search_params_udf;
  search_params_udf.n_probes   = this->n_probes_;
  search_params_udf.metric_udf = custom_l2_udf();

  std::array<double, 2> udf_ms{};
  for (int pass = 0; pass < 2; ++pass) {
    raft::resource::sync_stream(this->handle_);
    auto const t0 = std::chrono::steady_clock::now();
    ivf_flat::search(
      this->handle_, search_params_udf, idx, queries_view, indices_udf_view, distances_udf_view);
    raft::resource::sync_stream(this->handle_);
    auto const t1 = std::chrono::steady_clock::now();
    udf_ms[static_cast<std::size_t>(pass)] =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  EXPECT_LT(udf_ms[1], udf_ms[0]) << "cached UDF path should beat first search (first_ms="
                                  << udf_ms[0] << ", second_ms=" << udf_ms[1] << ")";

  // Copy results to host
  std::vector<int64_t> h_indices_builtin(this->num_queries_ * this->k_);
  std::vector<float> h_distances_builtin(this->num_queries_ * this->k_);
  std::vector<int64_t> h_indices_udf(this->num_queries_ * this->k_);
  std::vector<float> h_distances_udf(this->num_queries_ * this->k_);

  raft::copy(
    h_indices_builtin.data(), d_indices_builtin.data(), this->num_queries_ * this->k_, stream);
  raft::copy(
    h_distances_builtin.data(), d_distances_builtin.data(), this->num_queries_ * this->k_, stream);
  raft::copy(h_indices_udf.data(), d_indices_udf.data(), this->num_queries_ * this->k_, stream);
  raft::copy(h_distances_udf.data(), d_distances_udf.data(), this->num_queries_ * this->k_, stream);
  raft::resource::sync_stream(this->handle_);

  // UDF vs built-in: same neighbors/distances (per-query recall against built-in top-k).
  ASSERT_TRUE(eval_neighbours(h_indices_builtin,
                              h_indices_udf,
                              h_distances_builtin,
                              h_distances_udf,
                              static_cast<size_t>(this->num_queries_),
                              static_cast<size_t>(this->k_),
                              1e-5,
                              1.0));
}

/**
 * Build the index with native L2, search with a different metric (Chebyshev UDF), and compare to
 * exhaustive top-k from naive_knn (DistanceType::Linf). With n_probes == n_lists every cluster is
 * probed so every database vector is scored — equivalent to brute force for correctness.
 */
TEST(IvfFlatUdfChebyshev, ChebyshevMatchesNaiveKnnWhenProbingAllLists)
{
  using Traits = TestDataTraits<float>;
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  std::vector<float> const h_db = Traits::database();
  std::vector<float> const h_q  = Traits::queries();
  int64_t const n_db            = Traits::num_db_vecs;
  int64_t const n_queries       = 2;
  int64_t const dim             = Traits::dim;
  int const k                   = 4;
  uint32_t const n_lists        = 4;
  uint32_t const n_probes       = n_lists;

  rmm::device_uvector<float> d_db(static_cast<size_t>(n_db * dim), stream);
  rmm::device_uvector<float> d_q(static_cast<size_t>(n_queries * dim), stream);
  raft::copy(d_db.data(), h_db.data(), h_db.size(), stream);
  raft::copy(d_q.data(), h_q.data(), h_q.size(), stream);

  auto db_view = raft::make_device_matrix_view<const float, int64_t>(d_db.data(), n_db, dim);
  auto q_view  = raft::make_device_matrix_view<const float, int64_t>(d_q.data(), n_queries, dim);

  ivf_flat::index_params ip;
  ip.n_lists = n_lists;
  ip.metric  = cuvs::distance::DistanceType::L2Expanded;
  auto idx   = ivf_flat::build(handle, ip, db_view);

  rmm::device_uvector<int64_t> d_idx(static_cast<size_t>(n_queries * k), stream);
  rmm::device_uvector<float> d_dist(static_cast<size_t>(n_queries * k), stream);
  auto out_idx  = raft::make_device_matrix_view<int64_t, int64_t>(d_idx.data(), n_queries, k);
  auto out_dist = raft::make_device_matrix_view<float, int64_t>(d_dist.data(), n_queries, k);

  ivf_flat::search_params sp;
  sp.n_probes   = n_probes;
  sp.metric_udf = chebyshev_linf_udf();
  ivf_flat::search(handle, sp, idx, q_view, out_idx, out_dist);

  std::vector<int64_t> gpu_idx(static_cast<size_t>(n_queries * k));
  std::vector<float> gpu_dist(static_cast<size_t>(n_queries * k));
  raft::copy(gpu_idx.data(), d_idx.data(), gpu_idx.size(), stream);
  raft::copy(gpu_dist.data(), d_dist.data(), gpu_dist.size(), stream);

  rmm::device_uvector<float> d_ref_dist(static_cast<size_t>(n_queries * k), stream);
  rmm::device_uvector<int64_t> d_ref_idx(static_cast<size_t>(n_queries * k), stream);
  cuvs::neighbors::naive_knn<float, float, int64_t>(handle,
                                                    d_ref_dist.data(),
                                                    d_ref_idx.data(),
                                                    d_q.data(),
                                                    d_db.data(),
                                                    static_cast<size_t>(n_queries),
                                                    static_cast<size_t>(n_db),
                                                    static_cast<size_t>(dim),
                                                    static_cast<uint32_t>(k),
                                                    cuvs::distance::DistanceType::Linf);

  std::vector<int64_t> ref_idx(static_cast<size_t>(n_queries * k));
  std::vector<float> ref_dist(static_cast<size_t>(n_queries * k));
  raft::copy(ref_idx.data(), d_ref_idx.data(), ref_idx.size(), stream);
  raft::copy(ref_dist.data(), d_ref_dist.data(), ref_dist.size(), stream);
  raft::resource::sync_stream(handle);

  // Full probe (n_probes == n_lists): expect agreement with exhaustive naive_knn (Linf).
  double const min_recall = 1.0;
  double const eps        = 1e-4;
  ASSERT_TRUE(eval_neighbours(ref_idx,
                              gpu_idx,
                              ref_dist,
                              gpu_dist,
                              static_cast<size_t>(n_queries),
                              static_cast<size_t>(k),
                              eps,
                              min_recall));
}

/**
 * Invalid UDF source must fail NVRTC compilation; search should surface that as an exception,
 * not return garbage neighbors.
 */
TEST(IvfFlatUdfInvalidSource, SearchThrowsWhenMetricUdfDoesNotCompile)
{
  using Traits = TestDataTraits<float>;
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  std::vector<float> const h_db = Traits::database();
  std::vector<float> const h_q  = Traits::queries();
  int64_t const n_db            = Traits::num_db_vecs;
  int64_t const n_queries       = 2;
  int64_t const dim             = Traits::dim;
  int const k                   = 4;
  uint32_t const n_lists        = 4;

  rmm::device_uvector<float> d_db(static_cast<size_t>(n_db * dim), stream);
  rmm::device_uvector<float> d_q(static_cast<size_t>(n_queries * dim), stream);
  raft::copy(d_db.data(), h_db.data(), h_db.size(), stream);
  raft::copy(d_q.data(), h_q.data(), h_q.size(), stream);

  auto db_view = raft::make_device_matrix_view<const float, int64_t>(d_db.data(), n_db, dim);
  auto q_view  = raft::make_device_matrix_view<const float, int64_t>(d_q.data(), n_queries, dim);

  ivf_flat::index_params ip;
  ip.n_lists = n_lists;
  ip.metric  = cuvs::distance::DistanceType::L2Expanded;
  auto idx   = ivf_flat::build(handle, ip, db_view);

  rmm::device_uvector<int64_t> d_idx(static_cast<size_t>(n_queries * k), stream);
  rmm::device_uvector<float> d_dist(static_cast<size_t>(n_queries * k), stream);
  auto out_idx  = raft::make_device_matrix_view<int64_t, int64_t>(d_idx.data(), n_queries, k);
  auto out_dist = raft::make_device_matrix_view<float, int64_t>(d_dist.data(), n_queries, k);

  ivf_flat::search_params sp;
  sp.n_probes = 2;
  // Not valid CUDA / device code — NVRTC compile must fail (see NVRTCLTOFragmentCompiler::compile).
  sp.metric_udf = std::string{R"__(
__device__ void not_even_close_to_valid( { { {
)__"};

  EXPECT_THROW(ivf_flat::search(handle, sp, idx, q_view, out_idx, out_dist), raft::logic_error);
}

}  // namespace cuvs::neighbors::ivf_flat
