/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include <cuvs/preprocessing/spectral_embedding.hpp>
#include <cuvs/stats/trustworthiness_score.hpp>  // Add trustworthiness header
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/util/cudart_utils.hpp>

namespace cuvs::preprocessing::spectral_embedding {

template <typename T>
struct SpectralEmbeddingInputs {
  int n_samples;        // Number of samples in the dataset
  int n_features;       // Number of features in the dataset
  int n_clusters;       // Number of clusters to generate
  int n_components;     // Number of components in the embedding
  int n_neighbors;      // Number of neighbors for KNN
  T cluster_std;        // Standard deviation of clusters
  bool norm_laplacian;  // Whether to use normalized Laplacian
  bool drop_first;      // Whether to drop the first eigenvector
  uint64_t seed;        // Random seed
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const SpectralEmbeddingInputs<T>& inputs)
{
  return os << "n_samples:" << inputs.n_samples << " n_features:" << inputs.n_features
            << " n_clusters:" << inputs.n_clusters << " n_components:" << inputs.n_components
            << " n_neighbors:" << inputs.n_neighbors << " cluster_std:" << inputs.cluster_std
            << " norm_laplacian:" << inputs.norm_laplacian << " drop_first:" << inputs.drop_first
            << " seed:" << inputs.seed;
}

template <typename T>
class SpectralEmbeddingTest : public ::testing::TestWithParam<SpectralEmbeddingInputs<T>> {
 public:
  SpectralEmbeddingTest()
    : params_(::testing::TestWithParam<SpectralEmbeddingInputs<T>>::GetParam()),
      stream(raft::resource::get_cuda_stream(handle).get()),
      input_(raft::make_device_matrix<T, int, raft::row_major>(
        handle, params_.n_samples, params_.n_features)),
      labels_(raft::make_device_vector<int, int, raft::row_major>(handle, params_.n_samples)),
      embedding_(raft::make_device_matrix<T, int, raft::col_major>(
        handle, params_.n_samples, params_.n_components))
  {
  }

 protected:
  void SetUp() override
  {
    n_samples_    = params_.n_samples;
    n_features_   = params_.n_features;
    n_clusters_   = params_.n_clusters;
    n_components_ = params_.n_components;

    // Generate clusters using make_blobs
    raft::random::make_blobs<T, int, raft::row_major>(
      handle,
      input_.view(),
      labels_.view(),
      n_clusters_,                          // Number of clusters
      std::nullopt,                         // Generate random centers
      std::nullopt,                         // Use scalar std
      static_cast<T>(params_.cluster_std),  // Cluster std
      true,                                 // Shuffle
      static_cast<T>(-10.0),                // Center box min
      static_cast<T>(10.0),                 // Center box max
      params_.seed);                        // Random seed

    // Output embedding matrix
    embedding_ =
      raft::make_device_matrix<T, int, raft::col_major>(handle, n_samples_, n_components_);

    // Configure spectral embedding
    config_.n_neighbors    = params_.n_neighbors;
    config_.norm_laplacian = params_.norm_laplacian;
    config_.drop_first     = params_.drop_first;
    config_.seed           = params_.seed;

    config_.n_components = params_.drop_first ? n_components_ + 1 : n_components_;
  }

  void TearDown() override {}

  void testSpectralEmbedding()
  {
    // Call the spectral embedding function
    transform(handle, config_, input_.view(), embedding_.view());

    // Basic sanity checks on the output embedding

    // 1. Check embedding dimensions - should match the input settings
    ASSERT_EQ(embedding_.extent(0), n_samples_);
    int expected_components = n_components_;
    ASSERT_EQ(embedding_.extent(1), expected_components);

    // 2. Verify that the embedding is not all zeros or NaNs
    auto h_embedding =
      raft::make_host_matrix<T, int, raft::col_major>(embedding_.extent(0), embedding_.extent(1));
    raft::update_host(h_embedding.data_handle(),
                      embedding_.data_handle(),
                      embedding_.extent(0) * embedding_.extent(1),
                      stream);
    raft::resource::sync_stream(handle, stream);

    bool all_zeros = true;
    bool has_nan   = false;

    for (int i = 0; i < h_embedding.extent(0) * h_embedding.extent(1); i++) {
      if (h_embedding.data_handle()[i] != 0) { all_zeros = false; }
      if (std::isnan(h_embedding.data_handle()[i])) {
        has_nan = true;
        break;
      }
    }

    ASSERT_FALSE(all_zeros) << "Embedding contains all zeros";
    ASSERT_FALSE(has_nan) << "Embedding contains NaN values";

    // 3. Compute trustworthiness score to evaluate the quality of embedding
    // Create views with int64_t indexing for trustworthiness_score function
    auto input_view_i64 = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
      input_.data_handle(), input_.extent(0), input_.extent(1));

    // Need to convert embedding to row-major format for trustworthiness calculation
    auto embedding_row_major = raft::make_device_matrix<float, int, raft::row_major>(
      handle, embedding_.extent(0), embedding_.extent(1));

    // Copy and transpose from col-major to row-major
    raft::copy(embedding_row_major.data_handle(),
               embedding_.data_handle(),
               embedding_.extent(0) * embedding_.extent(1),
               stream);

    auto embedding_view_i64 = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
      embedding_row_major.data_handle(),
      embedding_row_major.extent(0),
      embedding_row_major.extent(1));

    // Calculate trustworthiness score
    double trust_score = cuvs::stats::trustworthiness_score(
      handle,
      input_view_i64,
      embedding_view_i64,
      params_.n_neighbors,  // use same number of neighbors as the embedding
      cuvs::distance::DistanceType::L2SqrtUnexpanded);

    // Print the trustworthiness score (for reporting purposes)
    std::cout << "Trustworthiness score: " << trust_score << std::endl;

    // Check that trustworthiness score is reasonable (should be between 0 and 1, with higher being
    // better)
    ASSERT_GE(trust_score, 0.0) << "Trustworthiness score should be non-negative";
    ASSERT_LE(trust_score, 1.0) << "Trustworthiness score should be <= 1.0";

    // For good embeddings, we expect a reasonably high trustworthiness score
    // This threshold can be adjusted based on empirical results
    ASSERT_GT(trust_score, 0.7)
      << "Trustworthiness score is too low, indicating poor embedding quality";
  }

 private:
  raft::resources handle;
  cudaStream_t stream;

  SpectralEmbeddingInputs<T> params_;
  int n_samples_;
  int n_features_;
  int n_clusters_;
  int n_components_;

  raft::device_matrix<T, int, raft::row_major> input_;
  raft::device_vector<int, int, raft::row_major> labels_;
  raft::device_matrix<T, int, raft::col_major> embedding_;

  params config_;
};

// Define test cases with different parameters
template <typename T>
const std::vector<SpectralEmbeddingInputs<T>> inputs = {
  // Small dataset with 2 components
  {100, 10, 3, 2, 10, 1.0f, true, false, 42ULL},

  // Medium dataset with 3 components
  {500, 20, 5, 3, 15, 1.0f, true, false, 42ULL},

  // Small dataset with varying cluster standard deviation
  {100, 10, 3, 2, 10, 0.5f, true, false, 42ULL},

  // Test with different Laplacian normalization
  {100, 10, 3, 2, 10, 1.0f, false, false, 42ULL},

  // Test with dropping first eigenvector
  {100, 10, 3, 2, 10, 1.0f, true, true, 42ULL},

  // Larger dataset
  {10000, 20, 8, 2, 12, 1.0f, true, true, 42ULL}};

typedef SpectralEmbeddingTest<float> SpectralEmbeddingTestF;
TEST_P(SpectralEmbeddingTestF, Result) { this->testSpectralEmbedding(); }

INSTANTIATE_TEST_CASE_P(SpectralEmbeddingTests,
                        SpectralEmbeddingTestF,
                        ::testing::ValuesIn(inputs<float>));

}  // namespace cuvs::preprocessing::spectral_embedding
