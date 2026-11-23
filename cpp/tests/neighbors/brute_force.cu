/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "./knn_utils.cuh"
#include <cuvs/core/dataset/sq.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/selection/select_k.hpp>

#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/copy.cuh>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/init.cuh>

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace cuvs::neighbors::brute_force {

template <typename T>
struct KNNInputs {
  std::vector<std::vector<T>> input;
  int k;
  std::vector<int> labels;
};

template <typename IdxT>
RAFT_KERNEL build_actual_output(
  int* output, int n_rows, int k, const int* idx_labels, const IdxT* indices)
{
  int element = threadIdx.x + blockDim.x * blockIdx.x;
  if (element >= n_rows * k) return;

  output[element] = idx_labels[indices[element]];
}

RAFT_KERNEL build_expected_output(int* output, int n_rows, int k, const int* labels)
{
  int row = threadIdx.x + blockDim.x * blockIdx.x;
  if (row >= n_rows) return;

  int cur_label = labels[row];
  for (int i = 0; i < k; i++) {
    output[row * k + i] = cur_label;
  }
}

template <typename T, typename DistT, typename IdxT>
class KNNTest : public ::testing::TestWithParam<KNNInputs<T>> {
 public:
  KNNTest()
    : params_(::testing::TestWithParam<KNNInputs<T>>::GetParam()),
      stream(raft::resource::get_cuda_stream(handle).get()),
      actual_labels_(0, stream),
      expected_labels_(0, stream),
      input_(0, stream),
      search_data_(0, stream),
      indices_(0, stream),
      distances_(0, stream),
      search_labels_(0, stream)
  {
  }

 protected:
  void testBruteForce()
  {
    raft::print_device_vector("Input array: ", input_.data(), rows_ * cols_, std::cout);
    std::cout << "K: " << k_ << std::endl;
    raft::print_device_vector("Labels array: ", search_labels_.data(), rows_, std::cout);

    auto index = raft::make_device_matrix_view<const T, IdxT, raft::row_major>(
      (const T*)(input_.data()), rows_, cols_);
    auto search = raft::make_device_matrix_view<const T, IdxT, raft::row_major>(
      (const T*)(search_data_.data()), rows_, cols_);
    auto indices =
      raft::make_device_matrix_view<IdxT, IdxT, raft::row_major>(indices_.data(), rows_, k_);
    auto distances =
      raft::make_device_matrix_view<DistT, IdxT, raft::row_major>(distances_.data(), rows_, k_);

    cuvs::neighbors::brute_force::index_params index_params;
    index_params.metric = cuvs::distance::DistanceType::L2Unexpanded;

    auto idx = cuvs::neighbors::brute_force::build(handle, index_params, index);
    cuvs::neighbors::brute_force::search_params search_params;
    cuvs::neighbors::brute_force::search(handle,
                                         search_params,
                                         idx,
                                         search,
                                         indices,
                                         distances,
                                         cuvs::neighbors::filtering::none_sample_filter{});

    build_actual_output<<<raft::ceildiv(rows_ * k_, 32), 32, 0, stream>>>(
      actual_labels_.data(), rows_, k_, search_labels_.data(), indices_.data());

    build_expected_output<<<raft::ceildiv(rows_ * k_, 32), 32, 0, stream>>>(
      expected_labels_.data(), rows_, k_, search_labels_.data());

    ASSERT_TRUE(devArrMatch(
      expected_labels_.data(), actual_labels_.data(), rows_ * k_, cuvs::Compare<int>(), stream));
  }

  void SetUp() override
  {
    rows_ = params_.input.size();
    cols_ = params_.input[0].size();
    k_    = params_.k;

    actual_labels_.resize(rows_ * k_, stream);
    expected_labels_.resize(rows_ * k_, stream);
    input_.resize(rows_ * cols_, stream);
    search_data_.resize(rows_ * cols_, stream);
    indices_.resize(rows_ * k_, stream);
    distances_.resize(rows_ * k_, stream);
    search_labels_.resize(rows_, stream);

    RAFT_CUDA_TRY(
      cudaMemsetAsync(actual_labels_.data(), 0, actual_labels_.size() * sizeof(int), stream));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(expected_labels_.data(), 0, expected_labels_.size() * sizeof(int), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(input_.data(), 0, input_.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(search_data_.data(), 0, search_data_.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(indices_.data(), 0, indices_.size() * sizeof(IdxT), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(distances_.data(), 0, distances_.size() * sizeof(DistT), stream));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(search_labels_.data(), 0, search_labels_.size() * sizeof(int), stream));

    std::vector<T> row_major_input;
    for (std::size_t i = 0; i < params_.input.size(); ++i) {
      for (std::size_t j = 0; j < params_.input[i].size(); ++j) {
        row_major_input.push_back(params_.input[i][j]);
      }
    }
    rmm::device_buffer input_d =
      rmm::device_buffer(row_major_input.data(), row_major_input.size() * sizeof(T), stream);
    T* input_ptr = static_cast<T*>(input_d.data());

    rmm::device_buffer labels_d =
      rmm::device_buffer(params_.labels.data(), params_.labels.size() * sizeof(int), stream);
    int* labels_ptr = static_cast<int*>(labels_d.data());

    raft::copy(input_.data(), input_ptr, rows_ * cols_, stream);
    raft::copy(search_data_.data(), input_ptr, rows_ * cols_, stream);
    raft::copy(search_labels_.data(), labels_ptr, rows_, stream);
    raft::resource::sync_stream(handle, stream);
  }

 private:
  raft::resources handle;
  cudaStream_t stream;

  KNNInputs<T> params_;
  int rows_;
  int cols_;
  rmm::device_uvector<T> input_;
  rmm::device_uvector<T> search_data_;
  rmm::device_uvector<IdxT> indices_;
  rmm::device_uvector<DistT> distances_;
  int k_;

  rmm::device_uvector<int> search_labels_;
  rmm::device_uvector<int> actual_labels_;
  rmm::device_uvector<int> expected_labels_;
};

template <typename T>
const std::vector<KNNInputs<T>> inputs = {
  // 2D
  {{
     {2.7810836, 2.550537003},
     {1.465489372, 2.362125076},
     {3.396561688, 4.400293529},
     {1.38807019, 1.850220317},
     {3.06407232, 3.005305973},
     {7.627531214, 2.759262235},
     {5.332441248, 2.088626775},
     {6.922596716, 1.77106367},
     {8.675418651, -0.242068655},
     {7.673756466, 3.508563011},
   },
   2,
   {0, 0, 0, 0, 0, 1, 1, 1, 1, 1}}};

typedef KNNTest<float, float, int64_t> KNNTest_float_int64_t;
TEST_P(KNNTest_float_int64_t, BruteForce) { this->testBruteForce(); }

typedef KNNTest<half, float, int64_t> KNNTest_half_int64_t;
TEST_P(KNNTest_half_int64_t, BruteForce) { this->testBruteForce(); }

INSTANTIATE_TEST_CASE_P(KNNTest, KNNTest_float_int64_t, ::testing::ValuesIn(inputs<float>));
INSTANTIATE_TEST_CASE_P(KNNTest, KNNTest_half_int64_t, ::testing::ValuesIn(inputs<half>));

TEST(BruteForceEqualNorms, PreservesNeighborOrdering)
{
  raft::resources handle;
  auto stream                  = raft::resource::get_cuda_stream(handle);
  constexpr int64_t n_database = 128;
  constexpr int64_t n_queries  = 1;
  constexpr int64_t dim        = 2;
  constexpr double angle_step  = 2e-3;

  std::vector<float> h_database(n_database * dim);
  for (int64_t i = 0; i < n_database; ++i) {
    auto rank               = (i * 73) % n_database;
    double angle            = static_cast<double>(rank) * angle_step;
    h_database[i * dim]     = static_cast<float>(std::cos(angle));
    h_database[i * dim + 1] = static_cast<float>(std::sin(angle));
  }
  std::vector<float> h_queries{1.0f, 0.0f};

  struct reference_neighbor {
    double squared_distance;
    int64_t index;
  };
  std::vector<reference_neighbor> reference(n_database);
  for (int64_t i = 0; i < n_database; ++i) {
    double dx    = static_cast<double>(h_queries[0]) - static_cast<double>(h_database[i * dim]);
    double dy    = static_cast<double>(h_queries[1]) - static_cast<double>(h_database[i * dim + 1]);
    reference[i] = {.squared_distance = dx * dx + dy * dy, .index = i};
  }
  std::ranges::sort(
    reference, [](const reference_neighbor& lhs, const reference_neighbor& rhs) -> bool {
      return lhs.squared_distance < rhs.squared_distance ||
             (lhs.squared_distance == rhs.squared_distance && lhs.index < rhs.index);
    });

  auto database = raft::make_device_matrix<float, int64_t>(handle, n_database, dim);
  auto queries  = raft::make_device_matrix<float, int64_t>(handle, n_queries, dim);
  raft::update_device(database.data_handle(), h_database.data(), h_database.size(), stream);
  raft::update_device(queries.data_handle(), h_queries.data(), h_queries.size(), stream);

  for (auto metric :
       {cuvs::distance::DistanceType::L2Expanded, cuvs::distance::DistanceType::L2SqrtExpanded}) {
    index_params index_params;
    index_params.metric = metric;
    auto index          = build(handle, index_params, raft::make_const_mdspan(database.view()));

    for (int64_t k : {int64_t{15}, int64_t{65}}) {
      auto neighbors = raft::make_device_matrix<int64_t, int64_t>(handle, n_queries, k);
      auto distances = raft::make_device_matrix<float, int64_t>(handle, n_queries, k);
      search(handle,
             search_params{},
             index,
             raft::make_const_mdspan(queries.view()),
             neighbors.view(),
             distances.view(),
             filtering::none_sample_filter{});

      std::vector<int64_t> h_neighbors(k);
      std::vector<float> h_distances(k);
      raft::update_host(h_neighbors.data(), neighbors.data_handle(), k, stream);
      raft::update_host(h_distances.data(), distances.data_handle(), k, stream);
      raft::resource::sync_stream(handle, stream);

      for (int64_t i = 0; i < k; ++i) {
        auto expected_distance = metric == cuvs::distance::DistanceType::L2Expanded
                                   ? reference[i].squared_distance
                                   : std::sqrt(reference[i].squared_distance);
        EXPECT_EQ(h_neighbors[i], reference[i].index)
          << "metric=" << static_cast<int>(metric) << " k=" << k << " rank=" << i;
        EXPECT_NEAR(h_distances[i],
                    expected_distance,
                    metric == cuvs::distance::DistanceType::L2Expanded ? 5e-7 : 5e-5)
          << "metric=" << static_cast<int>(metric) << " k=" << k << " rank=" << i;
      }
    }
  }
}

// Also test with larger random inputs, including col-major inputs
struct RandomKNNInputs {
  int num_queries;
  int num_db_vecs;
  int dim;
  int k;
  cuvs::distance::DistanceType metric;
  bool row_major;
  bool host_dataset;
};

std::ostream& operator<<(std::ostream& os, const RandomKNNInputs& input)
{
  return os << "num_queries:" << input.num_queries << " num_vecs:" << input.num_db_vecs
            << " dim:" << input.dim << " k:" << input.k
            << " metric:" << cuvs::neighbors::print_metric{input.metric}
            << " row_major:" << input.row_major << " host_dataset:" << input.host_dataset;
}

template <typename T, typename DistT = T>
class RandomBruteForceKNNTest : public ::testing::TestWithParam<RandomKNNInputs> {
 public:
  RandomBruteForceKNNTest()
    : stream_(raft::resource::get_cuda_stream(handle_).get()),
      params_(::testing::TestWithParam<RandomKNNInputs>::GetParam()),
      database(params_.num_db_vecs * params_.dim, stream_),
      search_queries(params_.num_queries * params_.dim, stream_),
      cuvs_indices_(params_.num_queries * params_.k, stream_),
      cuvs_distances_(params_.num_queries * params_.k, stream_),
      ref_indices_(params_.num_queries * params_.k, stream_),
      ref_distances_(params_.num_queries * params_.k, stream_)
  {
    raft::matrix::fill(
      handle_,
      raft::make_device_matrix_view(database.data(), params_.num_db_vecs, params_.dim),
      T{0.0});
    raft::matrix::fill(
      handle_,
      raft::make_device_matrix_view(search_queries.data(), params_.num_queries, params_.dim),
      T{0.0});
    raft::matrix::fill(
      handle_,
      raft::make_device_matrix_view(cuvs_distances_.data(), params_.num_queries, params_.k),
      DistT{0.0});
    raft::matrix::fill(
      handle_,
      raft::make_device_matrix_view(ref_distances_.data(), params_.num_queries, params_.k),
      DistT{0.0});
  }

 protected:
  // Computes the expanded L2 / cosine distances between the queries A and the database B on the
  // host, writing a row-major [num_queries, num_db_vecs] matrix to d_vals.
  void cpu_distance(const T* d_A,
                    const T* d_B,
                    DistT* d_vals,
                    bool is_row_major_A,
                    bool is_row_major_B,
                    cudaStream_t stream)
  {
    int64_t const n_queries = params_.num_queries;
    int64_t const n_db      = params_.num_db_vecs;
    int64_t const dim       = params_.dim;

    std::vector<T> h_A(n_queries * dim);
    std::vector<T> h_B(n_db * dim);
    cudaMemcpyAsync(h_A.data(), d_A, h_A.size() * sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_B.data(), d_B, h_B.size() * sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    auto to_dist = [](T v) -> DistT {
      if constexpr (sizeof(T) == 2) {
        return __half2float(v);
      } else {
        return v;
      }
    };

    // Convert the inputs once, storing A as [n_queries, dim] and B as [dim, n_db]. The inner loop
    // below then runs over contiguous database vectors and vectorizes, while each dot product and
    // norm is still accumulated over the dimensions in order.
    std::vector<DistT> A(n_queries * dim);
    std::vector<DistT> B(dim * n_db);
    for (int64_t i = 0; i < n_queries; ++i) {
      for (int64_t l = 0; l < dim; ++l) {
        A[i * dim + l] = to_dist(h_A[is_row_major_A ? i * dim + l : l * n_queries + i]);
      }
    }
    for (int64_t j = 0; j < n_db; ++j) {
      for (int64_t l = 0; l < dim; ++l) {
        B[l * n_db + j] = to_dist(h_B[is_row_major_B ? j * dim + l : l * n_db + j]);
      }
    }

    std::vector<DistT> norms_A(n_queries, DistT{0});
    std::vector<DistT> norms_B(n_db, DistT{0});
    for (int64_t i = 0; i < n_queries; ++i) {
      for (int64_t l = 0; l < dim; ++l) {
        norms_A[i] += A[i * dim + l] * A[i * dim + l];
      }
    }
    for (int64_t l = 0; l < dim; ++l) {
      for (int64_t j = 0; j < n_db; ++j) {
        norms_B[j] += B[l * n_db + j] * B[l * n_db + j];
      }
    }

    std::vector<DistT> h_vals(n_queries * n_db);
    // Process the database in blocks so that the block of B being reused by every query stays
    // in cache.
    constexpr int64_t kBlock = 512;
    for (int64_t j0 = 0; j0 < n_db; j0 += kBlock) {
      int64_t const nj = std::min(kBlock, n_db - j0);
      for (int64_t i = 0; i < n_queries; ++i) {
        DistT* __restrict__ out = h_vals.data() + i * n_db + j0;
        std::fill(out, out + nj, DistT{0});
        for (int64_t l = 0; l < dim; ++l) {
          DistT const a                 = A[i * dim + l];
          DistT const* __restrict__ b_l = B.data() + l * n_db + j0;
          for (int64_t jj = 0; jj < nj; ++jj) {
            out[jj] += a * b_l[jj];
          }
        }
        for (int64_t jj = 0; jj < nj; ++jj) {
          DistT const norm_A = norms_A[i];
          DistT const norm_B = norms_B[j0 + jj];
          if (params_.metric == cuvs::distance::DistanceType::L2Expanded) {
            out[jj] = DistT(-2.0) * out[jj] + norm_A + norm_B;
          } else if (params_.metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
            out[jj] = std::sqrt(DistT(-2.0) * out[jj] + norm_A + norm_B);
          } else if (params_.metric == cuvs::distance::DistanceType::CosineExpanded) {
            out[jj] = DistT(1.0) - out[jj] / std::sqrt(norm_A * norm_B);
          }
        }
      }
    }
    cudaMemcpyAsync(
      d_vals, h_vals.data(), h_vals.size() * sizeof(DistT), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
  }

  void testBruteForce()
  {
    DistT metric_arg = 3.0;

    // calculate the naive knn, by calculating the full pairwise distances and doing a k-select
    rmm::device_uvector<DistT> temp_distances(num_db_vecs * num_queries, stream_);
    rmm::device_uvector<char> workspace(0, stream_);

    auto temp_dist = temp_distances.data();
    rmm::device_uvector<DistT> temp_row_major_dist(num_db_vecs * num_queries, stream_);

    // For the complex post processes in these algorithms, we use CPU logic to make the baseline.
    if (metric == cuvs::distance::DistanceType::L2Expanded ||
        metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
        metric == cuvs::distance::DistanceType::CosineExpanded) {
      cpu_distance(search_queries.data(),
                   database.data(),
                   temp_distances.data(),
                   params_.row_major,
                   params_.row_major,
                   stream_);
    } else {
      if (params_.row_major) {
        distance::pairwise_distance(handle_,
                                    raft::make_device_matrix_view<const T, int64_t>(
                                      search_queries.data(), params_.num_queries, params_.dim),
                                    raft::make_device_matrix_view<const T, int64_t>(
                                      database.data(), params_.num_db_vecs, params_.dim),
                                    raft::make_device_matrix_view<DistT, int64_t>(
                                      temp_distances.data(), num_queries, num_db_vecs),
                                    metric,
                                    metric_arg);

      } else {
        distance::pairwise_distance(
          handle_,
          raft::make_device_matrix_view<const T, int64_t, raft::col_major>(
            search_queries.data(), params_.num_queries, params_.dim),
          raft::make_device_matrix_view<const T, int64_t, raft::col_major>(
            database.data(), params_.num_db_vecs, params_.dim),
          raft::make_device_matrix_view<DistT, int64_t, raft::col_major>(
            temp_distances.data(), num_queries, num_db_vecs),
          metric,
          metric_arg);

        // the pairwise_distance call assumes that the inputs and outputs are all either row-major
        // or col-major - meaning we have to transpose the output back for col-major queries
        // for comparison
        raft::linalg::transpose(
          handle_, temp_dist, temp_row_major_dist.data(), num_queries, num_db_vecs, stream_);
        temp_dist = temp_row_major_dist.data();
      }
    }

    cuvs::selection::select_k(
      handle_,
      raft::make_device_matrix_view<const DistT, int64_t>(temp_dist, num_queries, num_db_vecs),
      std::nullopt,
      raft::make_device_matrix_view(ref_distances_.data(), params_.num_queries, params_.k),
      raft::make_device_matrix_view(ref_indices_.data(), params_.num_queries, params_.k),
      cuvs::distance::is_min_close(metric),
      true);

    auto indices = raft::make_device_matrix_view<int64_t, int64_t, raft::row_major>(
      cuvs_indices_.data(), params_.num_queries, params_.k);
    auto distances = raft::make_device_matrix_view<DistT, int64_t, raft::row_major>(
      cuvs_distances_.data(), params_.num_queries, params_.k);

    cuvs::neighbors::brute_force::index_params index_params;
    index_params.metric     = metric;
    index_params.metric_arg = metric_arg;

    cuvs::neighbors::brute_force::search_params search_params;

    if (params_.host_dataset) {
      // test building from a dataset in host memory
      auto host_database =
        raft::make_host_matrix<T, int64_t, raft::row_major>(params_.num_db_vecs, params_.dim);
      raft::copy(
        host_database.data_handle(), database.data(), params_.num_db_vecs * params_.dim, stream_);

      auto idx = cuvs::neighbors::brute_force::build(
        handle_, index_params, raft::make_const_mdspan(host_database.view()));

      cuvs::neighbors::brute_force::search(
        handle_,
        search_params,
        idx,
        raft::make_device_matrix_view<const T, int64_t>(
          search_queries.data(), params_.num_queries, params_.dim),
        indices,
        distances,
        cuvs::neighbors::filtering::none_sample_filter{});
    } else {
      if (params_.row_major) {
        auto idx =
          cuvs::neighbors::brute_force::build(handle_,
                                              index_params,
                                              raft::make_device_matrix_view<const T, int64_t>(
                                                database.data(), params_.num_db_vecs, params_.dim));

        cuvs::neighbors::brute_force::search(
          handle_,
          search_params,
          idx,
          raft::make_device_matrix_view<const T, int64_t>(
            search_queries.data(), params_.num_queries, params_.dim),
          indices,
          distances,
          cuvs::neighbors::filtering::none_sample_filter{});
      } else {
        auto idx = cuvs::neighbors::brute_force::build(
          handle_,
          index_params,
          raft::make_device_matrix_view<const T, int64_t, raft::col_major>(
            database.data(), params_.num_db_vecs, params_.dim));

        cuvs::neighbors::brute_force::search(
          handle_,
          search_params,
          idx,
          raft::make_device_matrix_view<const T, int64_t, raft::col_major>(
            search_queries.data(), params_.num_queries, params_.dim),
          indices,
          distances,
          cuvs::neighbors::filtering::none_sample_filter{});
      }
    }

    ASSERT_TRUE(cuvs::neighbors::devArrMatchKnnPair(ref_indices_.data(),
                                                    cuvs_indices_.data(),
                                                    ref_distances_.data(),
                                                    cuvs_distances_.data(),
                                                    num_queries,
                                                    k_,
                                                    DistT(0.001),
                                                    stream_,
                                                    true));
  }

  void SetUp() override
  {
    num_queries = params_.num_queries;
    num_db_vecs = params_.num_db_vecs;
    dim         = params_.dim;
    k_          = params_.k;
    metric      = params_.metric;

    unsigned long long int seed = 1234ULL;
    raft::random::RngState r(seed);

    // JensenShannon distance requires positive values
    T min_val = metric == cuvs::distance::DistanceType::JensenShannon ? T(0.0) : T(-1.0);
    uniform(handle_, r, database.data(), num_db_vecs * dim, min_val, T(1.0));
    uniform(handle_, r, search_queries.data(), num_queries * dim, min_val, T(1.0));
  }

 private:
  raft::resources handle_;
  cudaStream_t stream_ = 0;
  RandomKNNInputs params_;
  int num_queries;
  int num_db_vecs;
  int dim;
  rmm::device_uvector<T> database;
  rmm::device_uvector<T> search_queries;
  rmm::device_uvector<int64_t> cuvs_indices_;
  rmm::device_uvector<DistT> cuvs_distances_;
  rmm::device_uvector<int64_t> ref_indices_;
  rmm::device_uvector<DistT> ref_distances_;
  int k_;
  cuvs::distance::DistanceType metric;
};

const std::vector<RandomKNNInputs> random_inputs = {
  // test each distance metric on a small-ish input, with row-major inputs
  {100, 256, 2, 65, cuvs::distance::DistanceType::L2Expanded, true, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::L2Unexpanded, true, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::L2SqrtExpanded, true, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::L2SqrtUnexpanded, true, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::L1, true, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::Linf, true, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::InnerProduct, true, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::CorrelationExpanded, true, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::CosineExpanded, true, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::LpUnexpanded, true, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::JensenShannon, true, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::L2SqrtExpanded, true, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::Canberra, true, false},
  // test each distance metric with col-major inputs
  {256, 512, 16, 7, cuvs::distance::DistanceType::L2Expanded, false, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::L2Unexpanded, false, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::L2SqrtExpanded, false, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::L2SqrtUnexpanded, false, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::L1, false, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::Linf, false, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::InnerProduct, false, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::CorrelationExpanded, false, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::CosineExpanded, false, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::LpUnexpanded, false, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::JensenShannon, false, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::L2SqrtExpanded, false, false},
  {256, 512, 16, 8, cuvs::distance::DistanceType::Canberra, false, false},
  // larger tests on different sized data / k values
  {10000, 40000, 32, 30, cuvs::distance::DistanceType::L2Expanded, false, false},
  {345, 1023, 16, 128, cuvs::distance::DistanceType::CosineExpanded, true, false},
  {789, 20516, 64, 256, cuvs::distance::DistanceType::L2SqrtExpanded, false, false},
  {1000, 200000, 128, 128, cuvs::distance::DistanceType::L2Expanded, true, false},
  {1000, 200000, 128, 128, cuvs::distance::DistanceType::L2Expanded, false, false},
  {1000, 5000, 128, 128, cuvs::distance::DistanceType::LpUnexpanded, true, false},
  {1000, 5000, 128, 128, cuvs::distance::DistanceType::L2SqrtExpanded, false, false},
  {1000, 5000, 128, 128, cuvs::distance::DistanceType::InnerProduct, false, false},
  // test with datasets on host memory
  {256, 512, 16, 8, cuvs::distance::DistanceType::L2Expanded, true, true},
  {256, 512, 32, 16, cuvs::distance::DistanceType::L2Unexpanded, true, true},
  {256, 512, 8, 8, cuvs::distance::DistanceType::L2SqrtExpanded, true, true},
  {256, 128, 32, 8, cuvs::distance::DistanceType::L2SqrtUnexpanded, true, true},
  {256, 512, 16, 8, cuvs::distance::DistanceType::L1, true, true},
  {256, 512, 16, 8, cuvs::distance::DistanceType::Linf, true, true},
  {256, 512, 16, 8, cuvs::distance::DistanceType::InnerProduct, true, true},
  {256, 512, 16, 7, cuvs::distance::DistanceType::L2Expanded, true, true}};

typedef RandomBruteForceKNNTest<float, float> RandomBruteForceKNNTestF;
TEST_P(RandomBruteForceKNNTestF, BruteForce) { this->testBruteForce(); }

typedef RandomBruteForceKNNTest<half, float> RandomBruteForceKNNTestH;
TEST_P(RandomBruteForceKNNTestH, BruteForce) { this->testBruteForce(); }

INSTANTIATE_TEST_CASE_P(RandomBruteForceKNNTest,
                        RandomBruteForceKNNTestF,
                        ::testing::ValuesIn(random_inputs));

INSTANTIATE_TEST_CASE_P(RandomBruteForceKNNTest,
                        RandomBruteForceKNNTestH,
                        ::testing::ValuesIn(random_inputs));

struct SQBruteForceInputs {
  cuvs::distance::DistanceType metric;
  cuvs::preprocessing::quantize::scalar::quantization_mode range;
};

template <typename T>
class SQBruteForceTest : public ::testing::TestWithParam<SQBruteForceInputs> {
 protected:
  void testSearch()
  {
    namespace scalar  = cuvs::preprocessing::quantize::scalar;
    auto const params = this->GetParam();
    raft::resources res;
    auto const stream           = raft::resource::get_cuda_stream(res);
    constexpr int64_t n_rows    = 4;
    constexpr int64_t n_queries = 2;
    constexpr int64_t dim       = 4;
    constexpr int64_t k         = 3;
    std::vector<T> const host_dataset{
      -300, 10, -16, 60, 208, 18, 12, 140, -100, 24, 20, 200, 44, 30, 4, 100};
    // Queries outside the corpus bounds must use the corpus quantizer.
    std::vector<T> const host_queries{-320, 12, -20, 74, 250, 4, 10, 240};
    auto dataset = raft::make_device_matrix<T, int64_t>(res, n_rows, dim);
    auto queries = raft::make_device_matrix<T, int64_t>(res, n_queries, dim);
    raft::update_device(
      dataset.data_handle(), host_dataset.data(), host_dataset.size(), stream.get());
    raft::update_device(
      queries.data_handle(), host_queries.data(), host_queries.size(), stream.get());

    auto sq = scalar::make_sq_dataset(res,
                                      scalar::params{.quantile = 1.0f, .range = params.range},
                                      raft::make_const_mdspan(dataset.view()));
    index_params index_params;
    index_params.metric = params.metric;
    if (params.range == scalar::quantization_mode::PER_DIMENSION) {
      EXPECT_THROW(build(res, index_params, sq.view()), raft::logic_error);
      return;
    }
    auto index     = build(res, index_params, sq.view());
    auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, n_queries, k);
    auto distances = raft::make_device_matrix<float, int64_t>(res, n_queries, k);
    search(res,
           search_params{},
           index,
           raft::make_const_mdspan(queries.view()),
           neighbors.view(),
           distances.view());

    auto query_codes = raft::make_device_matrix<int8_t, int64_t>(res, n_queries, dim);
    // Match the affine reconstruction used by search without rounding it back to the input type.
    auto const global = std::get<typename scalar::quantizer_view<const T>::global_type>(
      sq.const_manifest_view().value);
    scalar::quantizer<T> query_quantizer{global.min, global.max};
    scalar::quantizer<double> decode_quantizer{static_cast<double>(global.min),
                                               static_cast<double>(global.max)};
    auto decoded_dataset = raft::make_device_matrix<double, int64_t>(res, n_rows, dim);
    auto decoded_queries = raft::make_device_matrix<double, int64_t>(res, n_queries, dim);
    scalar::transform(
      res, query_quantizer, raft::make_const_mdspan(queries.view()), query_codes.view());
    scalar::inverse_transform(res, decode_quantizer, sq.data_const_view(), decoded_dataset.view());
    scalar::inverse_transform(
      res, decode_quantizer, raft::make_const_mdspan(query_codes.view()), decoded_queries.view());
    std::vector<double> corpus(n_rows * dim);
    std::vector<double> query_values(n_queries * dim);
    std::vector<int64_t> actual_neighbors(n_queries * k);
    std::vector<float> actual_distances(n_queries * k);
    raft::update_host(corpus.data(), decoded_dataset.data_handle(), corpus.size(), stream.get());
    raft::update_host(
      query_values.data(), decoded_queries.data_handle(), query_values.size(), stream.get());
    raft::update_host(
      actual_neighbors.data(), neighbors.data_handle(), actual_neighbors.size(), stream.get());
    raft::update_host(
      actual_distances.data(), distances.data_handle(), actual_distances.size(), stream.get());
    raft::resource::sync_stream(res);

    using cuvs::distance::DistanceType;
    for (int64_t query = 0; query < n_queries; ++query) {
      SCOPED_TRACE(query);
      std::vector<std::pair<double, int64_t>> expected;
      for (int64_t row = 0; row < n_rows; ++row) {
        double dot{}, query_norm{}, dataset_norm{}, squared_distance{};
        for (int64_t j = 0; j < dim; ++j) {
          double const x = static_cast<double>(corpus[row * dim + j]);
          double const y = static_cast<double>(query_values[query * dim + j]);
          dot += x * y;
          query_norm += y * y;
          dataset_norm += x * x;
          squared_distance += (x - y) * (x - y);
        }
        double distance = squared_distance;
        if (params.metric == DistanceType::InnerProduct) {
          distance = dot;
        } else if (params.metric == DistanceType::CosineExpanded) {
          auto const denominator = std::sqrt(query_norm * dataset_norm);
          distance               = denominator > 0.0 ? 1.0 - dot / denominator : 1.0;
        } else if (params.metric == DistanceType::L2SqrtExpanded ||
                   params.metric == DistanceType::L2SqrtUnexpanded) {
          distance = std::sqrt(distance);
        }
        expected.emplace_back(distance, row);
      }
      std::sort(expected.begin(), expected.end(), [params](auto const& a, auto const& b) {
        return params.metric == DistanceType::InnerProduct ? a.first > b.first : a.first < b.first;
      });
      for (int64_t i = 0; i < k; ++i) {
        EXPECT_EQ(actual_neighbors[query * k + i], expected[i].second);
        EXPECT_NEAR(actual_distances[query * k + i],
                    expected[i].first,
                    std::max(1e-5, std::abs(expected[i].first) * 1e-5));
      }
    }
  }
};

using SQBruteForceTestF = SQBruteForceTest<float>;
using SQBruteForceTestH = SQBruteForceTest<half>;
using SQBruteForceTestD = SQBruteForceTest<double>;
TEST_P(SQBruteForceTestF, SupportedMetrics) { this->testSearch(); }
TEST_P(SQBruteForceTestH, SupportedMetrics) { this->testSearch(); }
TEST_P(SQBruteForceTestD, SupportedMetrics) { this->testSearch(); }

TEST(BruteForceSQ, SearchShapeValidation)
{
  namespace scalar = cuvs::preprocessing::quantize::scalar;
  raft::resources res;
  auto const stream = raft::resource::get_cuda_stream(res);
  index_params index_params;
  index_params.metric = cuvs::distance::DistanceType::L2Unexpanded;

  auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, 1, 1);
  auto distances = raft::make_device_matrix<float, int64_t>(res, 1, 1);
  auto codes     = raft::make_device_matrix<int8_t, int64_t>(res, 512, 4);
  raft::matrix::fill(res, codes.view(), int8_t{0});
  cuvs::device_sq_dataset<float, int64_t, float> sq{std::move(codes),
                                                    scalar::quantizer<float>{-128.0f, 127.0f}};
  auto index           = build(res, index_params, sq.view());
  auto empty_queries   = raft::make_device_matrix<float, int64_t>(res, 0, 4);
  auto empty_neighbors = raft::make_device_matrix<int64_t, int64_t>(res, 0, 1);
  auto empty_distances = raft::make_device_matrix<float, int64_t>(res, 0, 1);

  for (int64_t dim : {int64_t{0}, int64_t{131072}, int64_t{33026}}) {
    auto invalid_codes = raft::make_device_matrix<int8_t, int64_t>(res, 1, dim);
    cuvs::device_sq_dataset<float, int64_t, float> invalid_sq{
      std::move(invalid_codes), scalar::quantizer<float>{-128.0f, 127.0f}};
    EXPECT_THROW(build(res, index_params, invalid_sq.view()), raft::logic_error);
  }
  EXPECT_NO_THROW(search(res,
                         search_params{},
                         index,
                         raft::make_const_mdspan(empty_queries.view()),
                         empty_neighbors.view(),
                         empty_distances.view()));

  // Aligned dimensions up to the INT32 dot-product limit are valid.
  for (int64_t dim : {int64_t{33028}, int64_t{131068}}) {
    auto valid_codes = raft::make_device_matrix<int8_t, int64_t>(res, 1, dim);
    auto queries     = raft::make_device_matrix<float, int64_t>(res, 1, dim);
    raft::matrix::fill(res, valid_codes.view(), int8_t{0});
    raft::matrix::fill(res, queries.view(), 0.0f);
    cuvs::device_sq_dataset<float, int64_t, float> valid_sq{
      std::move(valid_codes), scalar::quantizer<float>{-128.0f, 127.0f}};
    auto valid_index = build(res, index_params, valid_sq.view());
    search(res,
           search_params{},
           valid_index,
           raft::make_const_mdspan(queries.view()),
           neighbors.view(),
           distances.view());
    int64_t actual_neighbor;
    float actual_distance;
    raft::update_host(&actual_neighbor, neighbors.data_handle(), 1, stream.get());
    raft::update_host(&actual_distance, distances.data_handle(), 1, stream.get());
    raft::resource::sync_stream(res);
    EXPECT_EQ(actual_neighbor, 0);
    EXPECT_FLOAT_EQ(actual_distance, 0.0f);
  }
}

TEST(BruteForceSQ, TopKAndAlignment)
{
  namespace scalar = cuvs::preprocessing::quantize::scalar;
  raft::resources res;
  auto const stream = raft::resource::get_cuda_stream(res);
  index_params index_params;
  index_params.metric      = cuvs::distance::DistanceType::L2Unexpanded;
  constexpr int64_t n_rows = 65;

  for (int64_t dim : {int64_t{3}, int64_t{4}}) {
    std::vector<int8_t> host_codes(n_rows * dim, 0);
    for (int64_t row = 0; row < n_rows; ++row) {
      host_codes[row * dim] = static_cast<int8_t>(row);
    }
    auto codes = raft::make_device_matrix<int8_t, int64_t>(res, n_rows, dim);
    raft::update_device(codes.data_handle(), host_codes.data(), host_codes.size(), stream.get());
    cuvs::device_sq_dataset<float, int64_t, float> sq{std::move(codes),
                                                      scalar::quantizer<float>{-128.0f, 127.0f}};
    if (dim == 3) {
      EXPECT_THROW(build(res, index_params, sq.view()), raft::logic_error);
      continue;
    }
    auto index   = build(res, index_params, sq.view());
    auto queries = raft::make_device_matrix<float, int64_t>(res, 1, dim);
    raft::matrix::fill(res, queries.view(), 0.0f);

    for (int64_t k : {int64_t{64}, int64_t{65}}) {
      auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, 1, k);
      auto distances = raft::make_device_matrix<float, int64_t>(res, 1, k);
      search(res,
             search_params{},
             index,
             raft::make_const_mdspan(queries.view()),
             neighbors.view(),
             distances.view());

      std::vector<int64_t> actual_neighbors(k);
      std::vector<float> actual_distances(k);
      raft::update_host(
        actual_neighbors.data(), neighbors.data_handle(), actual_neighbors.size(), stream.get());
      raft::update_host(
        actual_distances.data(), distances.data_handle(), actual_distances.size(), stream.get());
      raft::resource::sync_stream(res);
      for (int64_t i = 0; i < k; ++i) {
        EXPECT_EQ(actual_neighbors[i], i);
        EXPECT_FLOAT_EQ(actual_distances[i], static_cast<float>(i * i));
      }
    }
  }
}

TEST(BruteForceSQ, TiledCandidates)
{
  namespace scalar = cuvs::preprocessing::quantize::scalar;
  raft::resources res;
  auto const stream           = raft::resource::get_cuda_stream(res);
  constexpr int64_t n_queries = 1025;
  constexpr int64_t dim       = 4;
  constexpr int64_t k         = 3;
  index_params index_params;
  index_params.metric = cuvs::distance::DistanceType::L2Unexpanded;

  std::vector<float> host_queries(n_queries * dim, 0.0f);
  host_queries[(n_queries - 1) * dim] = 4.0f;
  auto queries   = raft::make_device_matrix<float, int64_t>(res, n_queries, dim);
  auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, n_queries, k);
  auto distances = raft::make_device_matrix<float, int64_t>(res, n_queries, k);
  raft::update_device(
    queries.data_handle(), host_queries.data(), host_queries.size(), stream.get());

  // Cross a query-tile boundary with both one and multiple dataset tiles.
  for (int64_t n_rows : {int64_t{65}, int64_t{131073}}) {
    SCOPED_TRACE(n_rows);
    std::vector<int8_t> host_codes(n_rows * dim, 127);
    for (int64_t row : {int64_t{0}, int64_t{5}, n_rows - 1}) {
      std::fill_n(host_codes.begin() + row * dim, dim, int8_t{0});
    }
    host_codes[0]                  = 2;
    host_codes[5 * dim]            = 3;
    host_codes[(n_rows - 1) * dim] = 1;
    auto codes                     = raft::make_device_matrix<int8_t, int64_t>(res, n_rows, dim);
    raft::update_device(codes.data_handle(), host_codes.data(), host_codes.size(), stream.get());
    cuvs::device_sq_dataset<float, int64_t, float> sq{std::move(codes),
                                                      scalar::quantizer<float>{-128.0f, 127.0f}};
    auto index = build(res, index_params, sq.view());
    search(res,
           search_params{},
           index,
           raft::make_const_mdspan(queries.view()),
           neighbors.view(),
           distances.view());

    std::vector<int64_t> actual_neighbors(n_queries * k);
    std::vector<float> actual_distances(n_queries * k);
    raft::update_host(
      actual_neighbors.data(), neighbors.data_handle(), actual_neighbors.size(), stream.get());
    raft::update_host(
      actual_distances.data(), distances.data_handle(), actual_distances.size(), stream.get());
    raft::resource::sync_stream(res);
    for (int64_t query = 0; query < n_queries; ++query) {
      bool const last_query = query == n_queries - 1;
      EXPECT_EQ(actual_neighbors[query * k], last_query ? 5 : n_rows - 1);
      EXPECT_EQ(actual_neighbors[query * k + 1], 0);
      EXPECT_EQ(actual_neighbors[query * k + 2], last_query ? n_rows - 1 : 5);
      for (int64_t i = 0; i < k; ++i) {
        EXPECT_FLOAT_EQ(actual_distances[query * k + i], static_cast<float>((i + 1) * (i + 1)));
      }
    }
  }
}

TEST(BruteForceSQ, CosineNearZero)
{
  namespace scalar = cuvs::preprocessing::quantize::scalar;
  raft::resources res;
  auto const stream = raft::resource::get_cuda_stream(res);
  std::vector<int8_t> const host_codes{-128, -128, -128, -128, 127, 127, 127, -128};
  auto codes = raft::make_device_matrix<int8_t, int64_t>(res, 2, 4);
  raft::update_device(codes.data_handle(), host_codes.data(), host_codes.size(), stream.get());
  cuvs::device_sq_dataset<float, int64_t, float> sq{std::move(codes),
                                                    scalar::quantizer<float>{1e-4f, 1.0f}};
  index_params params;
  params.metric = cuvs::distance::DistanceType::CosineExpanded;
  auto index    = build(res, params, sq.view());
  auto queries  = raft::make_device_matrix<float, int64_t>(res, 1, 4);
  raft::matrix::fill(res, queries.view(), 1e-4f);
  auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, 1, 1);
  auto distances = raft::make_device_matrix<float, int64_t>(res, 1, 1);
  search(res,
         search_params{},
         index,
         raft::make_const_mdspan(queries.view()),
         neighbors.view(),
         distances.view());

  int64_t actual_neighbor;
  float actual_distance;
  raft::update_host(&actual_neighbor, neighbors.data_handle(), 1, stream.get());
  raft::update_host(&actual_distance, distances.data_handle(), 1, stream.get());
  raft::resource::sync_stream(res);
  EXPECT_EQ(actual_neighbor, 0);
  EXPECT_NEAR(actual_distance, 0.0f, 1e-5f);
}

TEST(BruteForceSQ, NearZeroDouble)
{
  namespace scalar = cuvs::preprocessing::quantize::scalar;
  for (auto metric :
       {cuvs::distance::DistanceType::CosineExpanded, cuvs::distance::DistanceType::InnerProduct}) {
    SCOPED_TRACE(static_cast<int>(metric));
    raft::resources res;
    auto stream          = raft::resource::get_cuda_stream(res);
    double const epsilon = std::ldexp(1.0, -40);
    std::vector<int8_t> const host_codes{-128, -128, -128, -128, -127, -128, -128, -128};
    auto codes = raft::make_device_matrix<int8_t, int64_t>(res, 2, 4);
    raft::update_device(codes.data_handle(), host_codes.data(), host_codes.size(), stream.get());
    cuvs::device_sq_dataset<double, int64_t, double> sq{
      std::move(codes), scalar::quantizer<double>{epsilon, 255 + epsilon}};
    index_params params;
    params.metric = metric;
    auto index    = build(res, params, sq.view());
    auto queries  = raft::make_device_matrix<double, int64_t>(res, 1, 4);
    raft::matrix::fill(res, queries.view(), epsilon);
    auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, 1, 1);
    auto distances = raft::make_device_matrix<float, int64_t>(res, 1, 1);
    search(res,
           search_params{},
           index,
           raft::make_const_mdspan(queries.view()),
           neighbors.view(),
           distances.view());
    int64_t actual_neighbor;
    float actual_distance;
    raft::update_host(&actual_neighbor, neighbors.data_handle(), 1, stream.get());
    raft::update_host(&actual_distance, distances.data_handle(), 1, stream.get());
    raft::resource::sync_stream(res);
    if (metric == cuvs::distance::DistanceType::InnerProduct) {
      EXPECT_EQ(actual_neighbor, 1);
      EXPECT_FLOAT_EQ(actual_distance, static_cast<float>(epsilon + 4 * epsilon * epsilon));
    } else {
      EXPECT_EQ(actual_neighbor, 0);
      EXPECT_NEAR(actual_distance, 0.0f, 1e-6f);
    }
  }
}

TEST(BruteForceSQ, CosineZeroQuery)
{
  namespace scalar = cuvs::preprocessing::quantize::scalar;
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  std::vector<float> const values{-1, -1, -1, -1, 2, 2, 2, 2, 254, 254, 254, 254};
  auto dataset = raft::make_device_matrix<float, int64_t>(res, 3, 4);
  raft::update_device(dataset.data_handle(), values.data(), values.size(), stream.get());
  auto sq = scalar::make_sq_dataset(
    res, scalar::params{.quantile = 1.0f}, raft::make_const_mdspan(dataset.view()));
  index_params params;
  params.metric = cuvs::distance::DistanceType::CosineExpanded;
  auto index    = build(res, params, sq.view());
  auto queries  = raft::make_device_matrix<float, int64_t>(res, 1, 4);
  raft::matrix::fill(res, queries.view(), 0.0f);
  auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, 1, 3);
  auto distances = raft::make_device_matrix<float, int64_t>(res, 1, 3);
  search(res,
         search_params{},
         index,
         raft::make_const_mdspan(queries.view()),
         neighbors.view(),
         distances.view());
  std::vector<float> actual(3);
  raft::update_host(actual.data(), distances.data_handle(), actual.size(), stream.get());
  raft::resource::sync_stream(res);
  EXPECT_EQ(actual, (std::vector<float>{1, 1, 1}));
}

TEST(BruteForceSQ, CosineExtremeDoubleMagnitudes)
{
  namespace scalar = cuvs::preprocessing::quantize::scalar;
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  std::vector<int8_t> const host_codes{1, 0, 0, 0, 0, 1, 0, 0};
  for (double magnitude : {std::ldexp(1.0, -665), std::ldexp(1.0, 665)}) {
    SCOPED_TRACE(magnitude);
    auto codes = raft::make_device_matrix<int8_t, int64_t>(res, 2, 4);
    raft::update_device(codes.data_handle(), host_codes.data(), host_codes.size(), stream.get());
    scalar::quantizer<double> quantizer{-128 * magnitude, 127 * magnitude};
    cuvs::device_sq_dataset<double, int64_t, double> sq{std::move(codes), std::move(quantizer)};
    index_params params;
    params.metric = cuvs::distance::DistanceType::CosineExpanded;
    auto index    = build(res, params, sq.view());
    std::vector<double> const host_queries{magnitude, 0, 0, 0, 0, magnitude, 0, 0};
    auto queries = raft::make_device_matrix<double, int64_t>(res, 2, 4);
    raft::update_device(
      queries.data_handle(), host_queries.data(), host_queries.size(), stream.get());
    auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, 2, 2);
    auto distances = raft::make_device_matrix<float, int64_t>(res, 2, 2);
    search(res,
           search_params{},
           index,
           raft::make_const_mdspan(queries.view()),
           neighbors.view(),
           distances.view());
    std::vector<int64_t> actual_neighbors(4);
    std::vector<float> actual_distances(4);
    raft::update_host(actual_neighbors.data(), neighbors.data_handle(), 4, stream.get());
    raft::update_host(actual_distances.data(), distances.data_handle(), 4, stream.get());
    raft::resource::sync_stream(res);
    EXPECT_EQ(actual_neighbors, (std::vector<int64_t>{0, 1, 1, 0}));
    EXPECT_EQ(actual_distances, (std::vector<float>{0, 1, 0, 1}));
  }
}

TEST(BruteForceSQ, InnerProductExtremeDoubleMagnitudes)
{
  namespace scalar = cuvs::preprocessing::quantize::scalar;
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  std::vector<int8_t> const host_codes{127, 127, 127, 127, -128, -128, -128, -128};
  auto codes = raft::make_device_matrix<int8_t, int64_t>(res, 2, 4);
  raft::update_device(codes.data_handle(), host_codes.data(), host_codes.size(), stream.get());
  cuvs::device_sq_dataset<double, int64_t, double> sq{std::move(codes),
                                                      scalar::quantizer<double>{-1e308, 1e308}};
  index_params params;
  params.metric = cuvs::distance::DistanceType::InnerProduct;
  auto index    = build(res, params, sq.view());
  auto queries  = raft::make_device_matrix<double, int64_t>(res, 1, 4);
  raft::matrix::fill(res, queries.view(), 1e308);
  auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, 1, 2);
  auto distances = raft::make_device_matrix<float, int64_t>(res, 1, 2);
  search(res,
         search_params{},
         index,
         raft::make_const_mdspan(queries.view()),
         neighbors.view(),
         distances.view());
  std::vector<int64_t> actual_neighbors(2);
  std::vector<float> actual_distances(2);
  raft::update_host(actual_neighbors.data(), neighbors.data_handle(), 2, stream.get());
  raft::update_host(actual_distances.data(), distances.data_handle(), 2, stream.get());
  raft::resource::sync_stream(res);
  EXPECT_EQ(actual_neighbors, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(actual_distances,
            (std::vector<float>{std::numeric_limits<float>::infinity(),
                                -std::numeric_limits<float>::infinity()}));
}

TEST(BruteForceSQ, L2LargeNorms)
{
  namespace scalar = cuvs::preprocessing::quantize::scalar;
  raft::resources res;
  auto const stream        = raft::resource::get_cuda_stream(res);
  constexpr int64_t dim    = 4096;
  constexpr int64_t n_rows = 3;
  std::vector<int8_t> host_codes(n_rows * dim, 127);
  host_codes[0]       = 126;
  host_codes[2 * dim] = 125;
  auto codes          = raft::make_device_matrix<int8_t, int64_t>(res, n_rows, dim);
  raft::update_device(codes.data_handle(), host_codes.data(), host_codes.size(), stream.get());
  cuvs::device_sq_dataset<float, int64_t, float> sq{std::move(codes),
                                                    scalar::quantizer<float>{-128.0f, 127.0f}};
  index_params params;
  params.metric = cuvs::distance::DistanceType::L2Expanded;
  auto index    = build(res, params, sq.view());
  auto queries  = raft::make_device_matrix<float, int64_t>(res, 1, dim);
  raft::matrix::fill(res, queries.view(), 127.0f);
  auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, 1, n_rows);
  auto distances = raft::make_device_matrix<float, int64_t>(res, 1, n_rows);
  search(res,
         search_params{},
         index,
         raft::make_const_mdspan(queries.view()),
         neighbors.view(),
         distances.view());

  std::vector<int64_t> actual_neighbors(n_rows);
  std::vector<float> actual_distances(n_rows);
  raft::update_host(
    actual_neighbors.data(), neighbors.data_handle(), actual_neighbors.size(), stream.get());
  raft::update_host(
    actual_distances.data(), distances.data_handle(), actual_distances.size(), stream.get());
  raft::resource::sync_stream(res);
  EXPECT_EQ(actual_neighbors, (std::vector<int64_t>{1, 0, 2}));
  EXPECT_EQ(actual_distances, (std::vector<float>{0.0f, 1.0f, 4.0f}));
}

TEST(BruteForceSQ, InnerProductPadding)
{
  namespace scalar = cuvs::preprocessing::quantize::scalar;
  raft::resources res;
  auto const stream = raft::resource::get_cuda_stream(res);
  std::vector<float> const host_dataset{
    -1e20f, -1e20f, -1e20f, -1e20f, -5e19f, -5e19f, -5e19f, -5e19f, 1e20f, 1e20f, 1e20f, 1e20f};
  auto dataset = raft::make_device_matrix<float, int64_t>(res, 3, 4);
  raft::update_device(
    dataset.data_handle(), host_dataset.data(), host_dataset.size(), stream.get());
  auto sq = scalar::make_sq_dataset(
    res, scalar::params{.quantile = 1.0f}, raft::make_const_mdspan(dataset.view()));
  index_params params;
  params.metric = cuvs::distance::DistanceType::InnerProduct;
  auto index    = build(res, params, sq.view());
  auto queries  = raft::make_device_matrix<float, int64_t>(res, 1, 4);
  raft::matrix::fill(res, queries.view(), 1e20f);
  auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, 1, 3);
  auto distances = raft::make_device_matrix<float, int64_t>(res, 1, 3);
  search(res,
         search_params{},
         index,
         raft::make_const_mdspan(queries.view()),
         neighbors.view(),
         distances.view());

  std::vector<int64_t> actual_neighbors(3);
  std::vector<float> actual_distances(3);
  raft::update_host(
    actual_neighbors.data(), neighbors.data_handle(), actual_neighbors.size(), stream.get());
  raft::update_host(
    actual_distances.data(), distances.data_handle(), actual_distances.size(), stream.get());
  raft::resource::sync_stream(res);
  EXPECT_EQ(actual_neighbors.front(), 2);
  EXPECT_TRUE(std::isinf(actual_distances.front()));
  EXPECT_GT(actual_distances.front(), 0.0f);
  for (size_t i = 1; i < actual_distances.size(); ++i) {
    EXPECT_TRUE(std::isinf(actual_distances[i]));
    EXPECT_LT(actual_distances[i], 0.0f);
  }
  std::sort(actual_neighbors.begin(), actual_neighbors.end());
  EXPECT_EQ(actual_neighbors, (std::vector<int64_t>{0, 1, 2}));
}

using cuvs::distance::DistanceType;
using cuvs::preprocessing::quantize::scalar::quantization_mode;
const std::vector<SQBruteForceInputs> sq_inputs{
  {DistanceType::L2Expanded, quantization_mode::GLOBAL},
  {DistanceType::L2Unexpanded, quantization_mode::GLOBAL},
  {DistanceType::L2SqrtExpanded, quantization_mode::GLOBAL},
  {DistanceType::L2SqrtUnexpanded, quantization_mode::GLOBAL},
  {DistanceType::InnerProduct, quantization_mode::GLOBAL},
  {DistanceType::CosineExpanded, quantization_mode::GLOBAL},
  {DistanceType::L2Expanded, quantization_mode::PER_DIMENSION}};
const std::vector<SQBruteForceInputs> sq_half_inputs{
  {DistanceType::L2Expanded, quantization_mode::GLOBAL}};
const std::vector<SQBruteForceInputs> sq_double_inputs{
  {DistanceType::InnerProduct, quantization_mode::GLOBAL}};

INSTANTIATE_TEST_CASE_P(BruteForceSQ, SQBruteForceTestF, ::testing::ValuesIn(sq_inputs));
INSTANTIATE_TEST_CASE_P(BruteForceSQ, SQBruteForceTestH, ::testing::ValuesIn(sq_half_inputs));
INSTANTIATE_TEST_CASE_P(BruteForceSQ, SQBruteForceTestD, ::testing::ValuesIn(sq_double_inputs));

}  // namespace cuvs::neighbors::brute_force
