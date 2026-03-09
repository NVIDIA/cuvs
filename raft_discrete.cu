/*
 * Demonstration of raft::random::discrete API usage
 *
 * The discrete API samples indices from a discrete probability distribution
 * defined by weights. Each index i is sampled with probability proportional
 * to weights[i].
 *
 * Compile with:
 *   nvcc -o raft_discrete raft_discrete.cu -I<raft_include_path> --expt-extended-lambda
 */

#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>

#include <rmm/device_uvector.hpp>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int main()
{
  // Create RAFT handle (manages CUDA resources)
  raft::handle_t handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  // Read weights from file (one weight per line)
  std::vector<float> h_weights;
  const char* weights_filename_env = std::getenv("WEIGHTS_FILE");
  std::string weights_filename = weights_filename_env ? weights_filename_env : "weight_tracker_phi_00001.txt";
  std::ifstream weights_file(weights_filename);
  if (!weights_file.is_open()) {
    std::cerr << "Error: Could not open " << weights_filename << std::endl;
    return 1;
  }
  std::string line;
  while (std::getline(weights_file, line)) {
    if (!line.empty()) {
      h_weights.push_back(std::stof(line));
    }
  }


  // Number of categories (weights)
  const int n_categories = static_cast<int>(h_weights.size());
  // Number of samples to draw
  const int n_samples = 9;
  rmm::device_uvector<float> d_weights(n_categories, stream);
  raft::copy(d_weights.data(), h_weights.data(), n_categories, stream);

  // Create output array for sampled indices
  rmm::device_uvector<int> d_indices(n_samples, stream);

  // Create random number generator with a unique seed each run
  auto seed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) ^
              static_cast<uint64_t>(std::random_device{}());
  const char* use_pc_env = std::getenv("USE_PC");
  bool use_pc = use_pc_env && std::string(use_pc_env) == "1";
  raft::random::RngState rng(seed, use_pc ? raft::random::GenPC : raft::random::GenPhilox);

  // Create mdspan views for the API
  auto indices_view = raft::make_device_vector_view<int, int>(d_indices.data(), n_samples);
  auto weights_view =
    raft::make_device_vector_view<const float, int>(d_weights.data(), n_categories);

  // Sample indices according to the weight distribution
  // Each index i will be sampled with probability weights[i] / sum(weights)

  // Count occurrences of each index
  std::vector<int> counts(n_categories, 0);
  std::vector<int> h_indices(n_samples);

  // Number of iterations to run
  const int n_iterations = 1000;

  for (int iter = 0; iter < n_iterations; ++iter) {
    raft::random::discrete(handle, rng, indices_view, weights_view);

    // Copy results back to host
    raft::copy(h_indices.data(), d_indices.data(), n_samples, stream);
    raft::resource::sync_stream(handle, stream);

    // Update counts
    for (int i = 0; i < n_samples; ++i) {
      counts[h_indices[i]]++;
    }
  }

  // Print results
  // std::cout << "Weights: ";
  // for (int i = 0; i < n_categories && i < 10; ++i) {
  //   std::cout << h_weights[i] << " ";
  // }
  // std::cout << std::endl;

  //std::cout << "Total samples: " << n_iterations * n_samples << std::endl;

  // Normalize counts array
  double counts_sum = 0.0;
  for (int i = 0; i < n_categories; ++i) {
    counts_sum += counts[i];
  }
  std::vector<double> normalized_counts(n_categories);
  for (int i = 0; i < n_categories; ++i) {
    normalized_counts[i] = static_cast<double>(counts[i]) / counts_sum;
  }

  // Normalize weights array
  double weights_sum = 0.0;
  for (int i = 0; i < n_categories; ++i) {
    weights_sum += h_weights[i];
  }
  std::vector<double> normalized_weights(n_categories);
  for (int i = 0; i < n_categories; ++i) {
    normalized_weights[i] = static_cast<double>(h_weights[i]) / weights_sum;
  }

  // Calculate sum of squared differences
  double sum_sq_diff = 0.0;
  for (int i = 0; i < n_categories; ++i) {
    double diff = normalized_counts[i] - normalized_weights[i];
    sum_sq_diff += diff * diff;
  }

  std::cout << weights_filename  << " " << "Sum of squared differences: " << sum_sq_diff << std::endl;

  return 0;
}
