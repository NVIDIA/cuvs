/*
 * Minimal multi-GPU KMeans example using cuVS.
 *
 *   nvcc -std=c++17 --extended-lambda --expt-relaxed-constexpr \
 *     -Xcompiler -Wno-deprecated-declarations -Xcompiler -fopenmp \
 *     -o repro_kmeans_mg repro_kmeans_mg.cu \
 *     -I$CONDA_PREFIX/include \
 *     -I$CONDA_PREFIX/include/rapids \
 *     -I$CONDA_PREFIX/include/rapids/libcudacxx \
 *     -L$CONDA_PREFIX/lib \
 *     -lcuvs -lrmm -lnccl -lucp -lucs -lucxx -lgomp \
 *     -DRAFT_SYSTEM_LITTLE_ENDIAN=1 -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE \
 *     -Xlinker -rpath=$CONDA_PREFIX/lib \
 *     -gencode arch=compute_XX,code=sm_XX
 */

#include <cuvs/cluster/kmeans.hpp>

#include <raft/comms/std_comms.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <nccl.h>
#include <omp.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <nvtx3/nvtx3.hpp>

// ----- helpers --------------------------------------------------------------
#define NCCLCHECK(cmd)                                                             \
  do {                                                                             \
    ncclResult_t res = cmd;                                                        \
    if (res != ncclSuccess) {                                                      \
      std::fprintf(                                                                \
        stderr, "NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
      std::exit(EXIT_FAILURE);                                                     \
    }                                                                              \
  } while (0)

#define CUDACHECK(cmd)                                                             \
  do {                                                                             \
    cudaError_t e = cmd;                                                           \
    if (e != cudaSuccess) {                                                        \
      std::fprintf(                                                                \
        stderr, "CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      std::exit(EXIT_FAILURE);                                                     \
    }                                                                              \
  } while (0)

constexpr int64_t N_CLUSTERS      = 1000;
constexpr int     N_BENCHMARK_ITERS = 1;


int main(int argc, char* argv[])
{
  // Parse command line arguments
  int64_t n_samples_total = 10000000;  // Default value
  int64_t n_features = 256;            // Default value
  
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--samples" || arg == "-s") {
      if (i + 1 < argc) {
        n_samples_total = std::stoll(argv[++i]);
      } else {
        std::cerr << "--samples requires a value" << std::endl;
        return 1;
      }
    } else if (arg == "--features" || arg == "-f") {
      if (i + 1 < argc) {
        n_features = std::stoll(argv[++i]);
      } else {
        std::cerr << "--features requires a value" << std::endl;
        return 1;
      }
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "  -s, --samples N    Set total number of samples (default: 10000000)" << std::endl;
      std::cout << "  -f, --features N   Set number of features (default: 256)" << std::endl;
      std::cout << "  -h, --help         Show this help message" << std::endl;
      return 0;
    }
  }
  
  // Detect available GPUs
  int n_devices = 0;
  CUDACHECK(cudaGetDeviceCount(&n_devices));
  if (n_devices < 1) {
    std::cerr << "No CUDA devices found." << std::endl;
    return 1;
  }
  std::cout << "Found " << n_devices << " GPU(s), running multi-GPU KMeans."
            << std::endl;

  // Create NCCL communicators (one per device)
  std::vector<int> dev_list(n_devices);
  for (int i = 0; i < n_devices; ++i) dev_list[i] = i;

  std::vector<ncclComm_t> nccl_comms(n_devices);
  NCCLCHECK(ncclCommInitAll(nccl_comms.data(), n_devices, dev_list.data()));

  // ------------------------------------------------------------------
  // Generate shared cluster centers on the host (BEFORE the parallel region)
  // so that every rank's data is clustered around the SAME centers.
  // Use seed 42 to match `make_blobs(random_state=42, centers=1000)`.
  // ------------------------------------------------------------------
  std::vector<float> h_centers(N_CLUSTERS * n_features);
  std::mt19937 gen(42);
  {
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& v : h_centers) v = dist(gen);
  }

  // Pre-generate per-rank seeds (mimics Dask's per-partition random seeds)
  std::vector<uint64_t> rank_seeds(n_devices);
  for (auto& s : rank_seeds) s = gen();

  // Launch one OpenMP thread per GPU (OPG model)
#pragma omp parallel for num_threads(n_devices)
  for (int rank = 0; rank < n_devices; ++rank) {
    // 1. Bind this thread to the correct GPU
    CUDACHECK(cudaSetDevice(rank));

    // 2. Create a raft handle and inject the NCCL communicator.
    //    This is the *only* extra step compared to single-GPU usage.
    raft::resources handle;
    raft::comms::build_comms_nccl_only(&handle, nccl_comms[rank], n_devices, rank);

    auto stream = raft::resource::get_cuda_stream(handle);

    // 3. Compute this rank's shard of the dataset.
    //    Each rank gets a contiguous, non-overlapping slice of the rows.
    int64_t n_samples_per_rank = n_samples_total / n_devices;
    int64_t leftover           = n_samples_total % n_devices;
    int64_t n_local            = n_samples_per_rank + (rank < leftover ? 1 : 0);

    // Copy the shared cluster centers to this GPU
    rmm::device_uvector<float> d_blob_centers(N_CLUSTERS * n_features, stream);
    CUDACHECK(cudaMemcpyAsync(d_blob_centers.data(),
                              h_centers.data(),
                              h_centers.size() * sizeof(float),
                              cudaMemcpyHostToDevice,
                              stream));

    // Generate synthetic clustered data directly on device.
    // Each rank uses a different seed so the data points are distinct,
    // but all ranks share the same cluster centers.
    rmm::device_uvector<float>   d_X(n_local * n_features, stream);
    rmm::device_uvector<int64_t> d_labels_true(n_local, stream);

    raft::random::make_blobs<float, int64_t>(d_X.data(),
                                             d_labels_true.data(),
                                             n_local,
                                             n_features,
                                             N_CLUSTERS,
                                             stream,
                                             /*row_major=*/true,
                                             /*centers=*/d_blob_centers.data(),
                                             /*cluster_std=*/nullptr,
                                             /*cluster_std_scalar=*/0.5f,
                                             /*shuffle=*/true,
                                             /*center_box_min=*/-10.0f,
                                             /*center_box_max=*/10.0f,
                                             /*seed=*/rank_seeds[rank]);

    // 4. Prepare output buffers
    rmm::device_uvector<float>   d_centroids(N_CLUSTERS * n_features, stream);
    rmm::device_uvector<int64_t> d_labels(n_local, stream);

    // Ensure data generation is complete before timing
    raft::resource::sync_stream(handle);

    if (rank == 0) {
      std::cout << "Data generation complete. "
                << n_samples_total << " total samples, "
                << n_local << " per rank, "
                << n_features << " features, "
                << N_CLUSTERS << " clusters." << std::endl;
    }

    // 5. Benchmark loop — run both RNG types to compare
    struct RngConfig {
      raft::random::GeneratorType type;
      const char* name;
    };
    RngConfig rng_configs[] = {
      {raft::random::GeneratorType::GenPC,     "GenPC"},
      {raft::random::GeneratorType::GenPhilox, "GenPhilox"},
    };

    for (const auto& rng_cfg : rng_configs) {
      // Create NVTX range for this RNG configuration
      nvtx3::scoped_range range(rng_cfg.name);
      
      if (rank == 0) {
        std::printf("\n=== RNG: %s ===\n", rng_cfg.name);
      }

      int64_t total_kmeans_iters = 0;
      for (int iter = 0; iter < N_BENCHMARK_ITERS; ++iter) {
        float   inertia      = 0;
        float   pred_inertia = 0;
        int64_t n_iter       = 0;

        cuvs::cluster::kmeans::params params;
        params.n_clusters          = static_cast<int>(N_CLUSTERS);
        params.max_iter            = 300;
        params.tol                 = 1e-4;
        params.rng_state.seed      = 42;
        params.rng_state.type      = rng_cfg.type;
        params.oversampling_factor = 2.0;
        params.n_init              = 1;

        auto X_view = raft::make_device_matrix_view<const float, int64_t>(
          d_X.data(), n_local, n_features);
        auto centroids_view = raft::make_device_matrix_view<float, int64_t>(
          d_centroids.data(), N_CLUSTERS, n_features);
        auto centroids_const_view = raft::make_device_matrix_view<const float, int64_t>(
          d_centroids.data(), N_CLUSTERS, n_features);
        auto labels_view = raft::make_device_vector_view<int64_t, int64_t>(
          d_labels.data(), n_local);

        auto t0 = std::chrono::high_resolution_clock::now();
        
        {
          nvtx3::scoped_range fit_range("kmeans_fit");
          cuvs::cluster::kmeans::fit(handle,
                                     params,
                                     X_view,
                                     std::nullopt,
                                     centroids_view,
                                     raft::make_host_scalar_view<float>(&inertia),
                                     raft::make_host_scalar_view<int64_t>(&n_iter));
          raft::resource::sync_stream(handle);
        }

        auto t1 = std::chrono::high_resolution_clock::now();

        {
          nvtx3::scoped_range predict_range("kmeans_predict");
          cuvs::cluster::kmeans::predict(handle,
                                         params,
                                         X_view,
                                         std::nullopt,
                                         centroids_const_view,
                                         labels_view,
                                         true,
                                         raft::make_host_scalar_view<float>(&pred_inertia));
          raft::resource::sync_stream(handle);
        }

        auto t2 = std::chrono::high_resolution_clock::now();

        double fit_s   = std::chrono::duration<double>(t1 - t0).count();
        double total_s = std::chrono::duration<double>(t2 - t0).count();

        total_kmeans_iters += n_iter;

        if (rank == 0) {
          std::printf("%d) Time taken by fit %.2fs and predict %.2fs (iters: %ld)\n",
                      iter, fit_s, total_s, static_cast<long>(n_iter));
        }
      }

      if (rank == 0) {
        double avg_iters = static_cast<double>(total_kmeans_iters) / N_BENCHMARK_ITERS;
        std::printf("Average KMeans iterations per run (%s): %.1f\n", rng_cfg.name, avg_iters);
      }
    }
  }

  // Clean up NCCL
  for (int i = 0; i < n_devices; ++i) {
    ncclCommDestroy(nccl_comms[i]);
  }

  std::cout << "Done." << std::endl;
  return 0;
}
