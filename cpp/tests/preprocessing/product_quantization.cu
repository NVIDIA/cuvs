/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include <chrono>
#include <cuvs/preprocessing/quantize/pq.hpp>
#include <cuvs/stats/trustworthiness_score.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/gather.cuh>
#include <raft/matrix/init.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/mr/managed_memory_resource.hpp>

namespace cuvs::preprocessing::quantize::pq {

template <typename T>
struct ProductQuantizationInputs {
  int n_samples;                                      // Number of samples in the dataset
  int n_features;                                     // Number of features in the dataset
  uint32_t pq_bits;                                   // PQ bits
  uint32_t pq_dim;                                    // PQ dimension
  cuvs::cluster::kmeans::kmeans_type pq_kmeans_type;  // PQ kmeans type
  uint32_t n_vq_centers;                              // Number of VQ centers
  bool use_vq;                                        // Whether to use VQ
  bool use_subspaces;                                 // Whether to use subspaces
  bool host_dataset;                                  // Whether to use host dataset
  uint64_t seed;                                      // Random seed
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const ProductQuantizationInputs<T>& inputs)
{
  return os << "n_samples:" << inputs.n_samples << " n_features:" << inputs.n_features
            << " pq_bits:" << inputs.pq_bits << " pq_dim:" << inputs.pq_dim
            << " pq_kmeans_type:" << (int)inputs.pq_kmeans_type
            << " n_vq_centers:" << inputs.n_vq_centers << " use_vq:" << inputs.use_vq
            << " host_dataset:" << inputs.host_dataset << " seed:" << inputs.seed;
}

template <typename T>
void compare_vectors_l2(const raft::resources& res,
                        T a,
                        T b,
                        double compression_ratio,
                        double eps,
                        bool print_data = false)
{
  auto n_rows = a.extent(0);
  auto dim    = a.extent(1);
  rmm::mr::managed_memory_resource managed_memory;
  auto dist =
    raft::make_device_mdarray<double>(res, managed_memory, raft::make_extents<uint32_t>(n_rows));
  raft::linalg::map_offset(res, dist.view(), [a, b, dim] __device__(uint32_t i) {
    double d = 0.0f;
    for (uint32_t j = 0; j < dim; j++) {
      double t = static_cast<float>(a(i, j)) - static_cast<float>(b(i, j));
      d += t * t;
    }
    return sqrt(d / double(dim));
  });
  raft::resource::sync_stream(res);
  double mean_dist = 0.0;
  for (uint32_t i = 0; i < n_rows; i++) {
    double d = dist(i);
    if ((print_data && i < 5) || (dim > 5 && (d > 1.2 * eps * std::pow(2.0, compression_ratio)))) {
      auto dim_print = std::min<uint32_t>(dim, 10);
      raft::print_vector("original", a.data_handle() + i * dim, dim_print, std::cout);
      raft::print_vector("reconstructed", b.data_handle() + i * dim, dim_print, std::cout);
      std::cout << "dist: " << d << std::endl;
    }
    mean_dist += d;
    // The theoretical estimate of the error is hard to come up with,
    // the estimate below is based on experimentation + curse of dimensionality
    if (dim > 5) {
      ASSERT_LE(d, 1.2 * eps * std::pow(2.0, compression_ratio))
        << " (ix = " << i << ", eps = " << eps << ", compression_ratio = " << compression_ratio
        << ")";
    }
  }
  mean_dist /= n_rows;
  if (print_data) {
    std::cout << "mean_dist: " << mean_dist << ", compression_ratio = " << compression_ratio
              << std::endl;
  }
}

template <typename T>
class ProductQuantizationTest : public ::testing::TestWithParam<ProductQuantizationInputs<T>> {
 public:
  ProductQuantizationTest()
    : params_(::testing::TestWithParam<ProductQuantizationInputs<T>>::GetParam()),
      stream(raft::resource::get_cuda_stream(handle)),
      dataset_(raft::make_device_matrix<T, int64_t, raft::row_major>(
        handle, params_.n_samples, params_.n_features)),
      dataset_host_(
        raft::make_host_matrix<T, int64_t, raft::row_major>(params_.n_samples, params_.n_features)),
      n_samples_(params_.n_samples),
      n_features_(params_.n_features)
  {
  }

 protected:
  void SetUp() override
  {
    if constexpr (std::is_same_v<T, half>) {
      raft::random::RngState r(params_.seed);
      raft::random::uniform(
        handle, r, dataset_.data_handle(), n_samples_ * n_features_, T(0.1), T(2.0));
    } else {
      auto labels_ = raft::make_device_vector<int64_t, int64_t>(handle, n_samples_);
      raft::random::make_blobs<T, int64_t, raft::row_major>(
        handle,
        dataset_.view(),
        labels_.view(),
        5,                      // Number of clusters
        std::nullopt,           // Generate random centers
        std::nullopt,           // Use scalar std
        static_cast<T>(1.0),    // Cluster std
        true,                   // Shuffle
        static_cast<T>(-10.0),  // Center box min
        static_cast<T>(10.0),   // Center box max
        params_.seed);          // Random seed
    }
    raft::copy(
      dataset_host_.data_handle(), dataset_.data_handle(), n_samples_ * n_features_, stream);
  }

  void TearDown() override {}

  void check_reconstruction(const cuvs::preprocessing::quantize::pq::quantizer<T>& quantizer,
                            raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
                            std::optional<raft::device_vector_view<uint32_t, int64_t>> vq_labels,
                            double compression_ratio,
                            uint32_t n_take)
  {
    auto dim = n_features_;

    if (n_take == 0) { return; }

    n_take = std::min<uint32_t>(n_take, codes.extent(0));

    auto rec_data  = raft::make_device_matrix<T, int64_t>(handle, n_take, dim);
    auto orig_data = raft::make_device_matrix_view<T, int64_t>(dataset_.data_handle(), n_take, dim);

    std::optional<raft::device_vector_view<const uint32_t, int64_t>> vq_labels_view = std::nullopt;
    if (vq_labels) { vq_labels_view = raft::make_const_mdspan(vq_labels.value()); }
    inverse_transform(handle,
                      quantizer,
                      raft::device_matrix_view<const uint8_t, int64_t>(
                        codes.data_handle(), n_take, codes.extent(1)),
                      rec_data.view(),
                      vq_labels_view);

    compare_vectors_l2(handle, orig_data, rec_data.view(), compression_ratio, 0.04, false);
  }

  void testProductQuantizationFromDataset()
  {
    using LabelT                = uint32_t;
    using kmeans_params_variant = cuvs::preprocessing::quantize::pq::kmeans_params_variant;
    kmeans_params_variant kmeans_params =
      (params_.pq_kmeans_type == cuvs::cluster::kmeans::kmeans_type::KMeansBalanced)
        ? kmeans_params_variant(cuvs::cluster::kmeans::balanced_params{.n_iters = 25})
        : kmeans_params_variant(
            cuvs::cluster::kmeans::params{.n_clusters = 1 << params_.pq_bits,
                                          .init = cuvs::cluster::kmeans::params::InitMethod::Random,
                                          .max_iter = 25});
    config_ = cuvs::preprocessing::quantize::pq::params(params_.pq_bits,
                                                        params_.pq_dim,
                                                        params_.use_subspaces,
                                                        params_.use_vq,
                                                        params_.n_vq_centers,
                                                        kmeans_params,
                                                        256,
                                                        1024);
    raft::resource::sync_stream(handle);

    if ((n_samples_ < (1 << params_.pq_bits)) || (n_features_ % params_.pq_dim != 0)) {
      EXPECT_THROW(build(handle, config_, raft::make_const_mdspan(dataset_host_.view())),
                   raft::logic_error);
      return;
    }

    auto pq = params_.host_dataset
                ? build(handle, config_, raft::make_const_mdspan(dataset_host_.view()))
                : build(handle, config_, raft::make_const_mdspan(dataset_.view()));

    auto n_encoded_cols = get_quantized_dim(pq.params_quantizer);
    auto d_quantized_output =
      raft::make_device_matrix<uint8_t, int64_t>(handle, n_samples_, n_encoded_cols);
    auto d_vq_labels = raft::make_device_vector<uint32_t, int64_t>(handle, 0);
    std::optional<raft::device_vector_view<uint32_t, int64_t>> d_vq_labels_view = std::nullopt;
    if (params_.use_vq) {
      d_vq_labels      = raft::make_device_vector<uint32_t, int64_t>(handle, n_samples_);
      d_vq_labels_view = d_vq_labels.view();
    }

    if (params_.host_dataset) {
      transform(handle,
                pq,
                raft::make_const_mdspan(dataset_host_.view()),
                d_quantized_output.view(),
                d_vq_labels_view);
    } else {
      transform(handle,
                pq,
                raft::make_const_mdspan(dataset_.view()),
                d_quantized_output.view(),
                d_vq_labels_view);
    }

    // 1. Verify that the quantized output is not all zeros
    {
      auto h_quantized_output =
        raft::make_host_matrix<uint8_t, int, raft::col_major>(n_samples_, n_encoded_cols);
      raft::update_host(h_quantized_output.data_handle(),
                        d_quantized_output.data_handle(),
                        n_samples_ * n_encoded_cols,
                        stream);
      raft::resource::sync_stream(handle, stream);

      bool all_zeros       = true;
      auto n_vecs_to_check = std::min(n_samples_, 50);

      for (int i = 0; (i < n_vecs_to_check * h_quantized_output.extent(1)) && all_zeros; i++) {
        if (h_quantized_output.data_handle()[i] != 0) { all_zeros = false; }
      }
      ASSERT_FALSE(all_zeros) << "Quantized output contains all zeros";
    }

    // 2. Verify that the quantized output is consistent with the input
    double compression_ratio =
      static_cast<double>(n_features_ * 8) /
      static_cast<double>(pq.params_quantizer.pq_dim * pq.params_quantizer.pq_bits);

    check_reconstruction(pq, d_quantized_output.view(), d_vq_labels_view, compression_ratio, 500);
  }

 private:
  raft::resources handle;
  cudaStream_t stream;

  ProductQuantizationInputs<T> params_;
  int n_samples_;
  int n_features_;

  raft::device_matrix<T, int64_t, raft::row_major> dataset_;
  raft::host_matrix<T, int64_t, raft::row_major> dataset_host_;
  params config_;
};

TEST(ProductQuantizationTestF, Parameters)
{
  raft::resources handle;
  int pq_bits        = 8;
  int pq_dim         = 0;
  bool use_subspaces = true;
  bool use_vq        = false;
  int n_vq_centers   = 0;
  auto dataset_host  = raft::make_host_matrix<float, int64_t>(100, 64);
  // Test classical kmeans parameters
  using kmeans_params_variant         = cuvs::preprocessing::quantize::pq::kmeans_params_variant;
  kmeans_params_variant kmeans_params = kmeans_params_variant(
    cuvs::cluster::kmeans::params{.n_clusters = 1 << pq_bits,
                                  .init       = cuvs::cluster::kmeans::params::InitMethod::Random,
                                  .max_iter   = 75});
  auto pq_params = cuvs::preprocessing::quantize::pq::params(
    pq_bits, pq_dim, use_subspaces, use_vq, n_vq_centers, kmeans_params);
  EXPECT_EQ(pq_params.pq_bits, pq_bits);
  EXPECT_EQ(std::get<cuvs::cluster::kmeans::params>(pq_params.kmeans_params).init,
            cuvs::cluster::kmeans::params::InitMethod::Random);
  EXPECT_EQ(std::get<cuvs::cluster::kmeans::params>(pq_params.kmeans_params).max_iter, 75);

  // Test balanced kmeans parameters
  pq_params = cuvs::preprocessing::quantize::pq::params(
    pq_bits,
    pq_dim,
    use_subspaces,
    use_vq,
    n_vq_centers,
    kmeans_params_variant(cuvs::cluster::kmeans::balanced_params{.n_iters = 76}));
  EXPECT_EQ(std::get<cuvs::cluster::kmeans::balanced_params>(pq_params.kmeans_params).n_iters, 76);

  // Test simplified constructor
  pq_params =
    cuvs::preprocessing::quantize::pq::params(pq_bits,
                                              pq_dim,
                                              use_subspaces,
                                              use_vq,
                                              n_vq_centers,
                                              77,
                                              cuvs::cluster::kmeans::kmeans_type::KMeansBalanced);
  EXPECT_EQ(std::get<cuvs::cluster::kmeans::balanced_params>(pq_params.kmeans_params).n_iters, 77);

  // Test invalid initialization (Array kmeans init)
  pq_params.kmeans_params = kmeans_params_variant(
    cuvs::cluster::kmeans::params{.n_clusters = 1 << pq_bits,
                                  .init       = cuvs::cluster::kmeans::params::InitMethod::Array,
                                  .max_iter   = 75});
  EXPECT_THROW(build(handle, pq_params, raft::make_const_mdspan(dataset_host.view())),
               raft::logic_error);

  // Test invalid initialization (metric)
  auto my_balanced_params   = cuvs::cluster::kmeans::balanced_params{};
  my_balanced_params.metric = cuvs::distance::DistanceType::JaccardExpanded;
  pq_params.kmeans_params   = kmeans_params_variant(my_balanced_params);
  EXPECT_THROW(build(handle, pq_params, raft::make_const_mdspan(dataset_host.view())),
               raft::logic_error);
}

// Define test cases with different parameters
template <typename T>
const std::vector<ProductQuantizationInputs<T>> inputs = {
  // Extreme cases
  {1, 64, 4, 8, cuvs::cluster::kmeans::kmeans_type::KMeansBalanced, 0, true, false, false, 42ULL},
  {512, 1, 8, 1, cuvs::cluster::kmeans::kmeans_type::KMeansBalanced, 0, true, true, false, 42ULL},
  {4096,
   1024,
   10,
   4,
   cuvs::cluster::kmeans::kmeans_type::KMeansBalanced,
   0,
   false,
   false,
   false,
   42ULL},
  {20, 2, 4, 1, cuvs::cluster::kmeans::kmeans_type::KMeansBalanced, 0, false, true, false, 42ULL},
  {200, 8, 7, 2, cuvs::cluster::kmeans::kmeans_type::KMeans, 2, false, true, false, 42ULL},
  {299,
   3000,
   8,
   64,
   cuvs::cluster::kmeans::kmeans_type::KMeansBalanced,
   0,
   false,
   true,
   false,
   42ULL},
  // Small dataset
  {100,
   64,
   4,
   8,
   cuvs::cluster::kmeans::kmeans_type::KMeansBalanced,
   0,
   false,
   false,
   false,
   42ULL},

  // Small dataset with bigger dims
  {100,
   90,
   6,
   10,
   cuvs::cluster::kmeans::kmeans_type::KMeansBalanced,
   0,
   false,
   false,
   true,
   42ULL},
  {300, 128, 7, 32, cuvs::cluster::kmeans::kmeans_type::KMeansBalanced, 0, true, true, true, 42ULL},

  // Medium dataset
  {500, 40, 5, 8, cuvs::cluster::kmeans::kmeans_type::KMeans, 0, false, false, false, 42ULL},
  {500, 60, 6, 6, cuvs::cluster::kmeans::kmeans_type::KMeansBalanced, 4, true, true, true, 42ULL},
  {500, 128, 5, 8, cuvs::cluster::kmeans::kmeans_type::KMeans, 0, true, false, false, 42ULL},

  // Larger dataset
  {1000, 320, 8, 64, cuvs::cluster::kmeans::kmeans_type::KMeans, 0, true, true, false, 42ULL},
  {1000,
   384,
   8,
   64,
   cuvs::cluster::kmeans::kmeans_type::KMeansBalanced,
   0,
   true,
   true,
   false,
   42ULL},
  {1000,
   40,
   4,
   10,
   cuvs::cluster::kmeans::kmeans_type::KMeansBalanced,
   0,
   false,
   false,
   false,
   42ULL},
  {3000, 1024, 4, 32, cuvs::cluster::kmeans::kmeans_type::KMeans, 0, false, false, false, 42ULL},
  {1000,
   2048,
   4,
   128,
   cuvs::cluster::kmeans::kmeans_type::KMeansBalanced,
   0,
   true,
   true,
   false,
   42ULL},

  // Benchmark datasets
  {50000,
   1024,
   8,
   128,
   cuvs::cluster::kmeans::kmeans_type::KMeansBalanced,
   0,
   false,
   true,
   false,
   42ULL},
  {50000,
   2048,
   8,
   128,
   cuvs::cluster::kmeans::kmeans_type::KMeansBalanced,
   10,
   true,
   true,
   true,
   42ULL}};

typedef ProductQuantizationTest<float> ProductQuantizationTestF;
TEST_P(ProductQuantizationTestF, Result) { this->testProductQuantizationFromDataset(); }

INSTANTIATE_TEST_CASE_P(ProductQuantizationTests,
                        ProductQuantizationTestF,
                        ::testing::ValuesIn(inputs<float>));

}  // namespace cuvs::preprocessing::quantize::pq
