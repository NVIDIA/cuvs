/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "./knn_utils.cuh"
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/selection/select_k.hpp>

#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/init.cuh>

#include <cuda_fp16.h>

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
  void cpu_distance(const T* d_A,
                    const T* d_B,
                    DistT* d_vals,
                    bool is_row_major_A,
                    bool is_row_major_B,
                    bool is_row_major_C,
                    cudaStream_t stream,
                    DistT alpha = 1.0,
                    DistT beta  = 0.0)
  {
    size_t size_A    = params_.num_queries * params_.dim * sizeof(T);
    size_t size_B    = params_.num_db_vecs * params_.dim * sizeof(T);
    size_t size_vals = params_.num_queries * params_.num_db_vecs * sizeof(DistT);

    T* h_A        = static_cast<T*>(malloc(size_A));
    T* h_B        = static_cast<T*>(malloc(size_B));
    DistT* h_vals = static_cast<DistT*>(malloc(size_vals));

    cudaMemcpyAsync(h_A, d_A, size_A, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_B, d_B, size_B, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_vals, d_vals, size_vals, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    bool trans_a = is_row_major_A;
    bool trans_b = is_row_major_B;
    bool trans_c = is_row_major_C;

    for (int64_t i = 0; i < params_.num_queries; ++i) {
      for (int64_t j = 0; j < params_.num_db_vecs; ++j) {
        DistT sum     = 0;
        DistT norms_A = 0;
        DistT norms_B = 0;

        for (int64_t l = 0; l < params_.dim; ++l) {
          int64_t a_index = trans_a ? i * params_.dim + l : l * params_.num_queries + i;
          int64_t b_index = trans_b ? j * params_.dim + l : l * params_.num_db_vecs + j;
          DistT A_v;
          DistT B_v;
          if constexpr (sizeof(T) == 2) {
            A_v = __half2float(h_A[a_index]);
            B_v = __half2float(h_B[b_index]);
          } else {
            A_v = h_A[a_index];
            B_v = h_B[b_index];
          }

          sum += A_v * B_v;

          norms_A += A_v * A_v;
          norms_B += B_v * B_v;
        }

        int64_t c_index = trans_c ? i * params_.num_db_vecs + j : j * params_.num_queries + i;

        h_vals[c_index] = alpha * sum + beta * h_vals[c_index];
        if (params_.metric == cuvs::distance::DistanceType::L2Expanded) {
          h_vals[c_index] = DistT(-2.0) * h_vals[c_index] + norms_A + norms_B;
        } else if (params_.metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
          h_vals[c_index] = std::sqrt(DistT(-2.0) * h_vals[c_index] + norms_A + norms_B);
        } else if (params_.metric == cuvs::distance::DistanceType::CosineExpanded) {
          h_vals[c_index] = DistT(1.0) - h_vals[c_index] / std::sqrt(norms_A * norms_B);
        }
      }
    }
    cudaMemcpyAsync(d_vals, h_vals, size_vals, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    free(h_A);
    free(h_B);
    free(h_vals);
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
                   true,
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

}  // namespace cuvs::neighbors::brute_force
