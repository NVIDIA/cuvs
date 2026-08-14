/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <cstdlib>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>

#include <cuvs/neighbors/vamana.hpp>

#include <rmm/mr/pool_memory_resource.hpp>

#include "common.cuh"

template <typename T>
void vamana_build_and_write(raft::device_resources const& dev_resources,
                            raft::device_matrix_view<const T, int64_t> dataset,
                            std::string out_fname,
                            int degree,
                            int visited_size,
                            float max_fraction,
                            float iters,
                            std::string codebook_prefix)
{
  using namespace cuvs::neighbors;

  // use default index parameters
  vamana::index_params index_params;
  index_params.max_fraction = max_fraction;
  index_params.visited_size = visited_size;
  index_params.graph_degree = degree;
  index_params.vamana_iters = iters;

  if (codebook_prefix != "") {
    index_params.codebooks = vamana::deserialize_codebooks(codebook_prefix, dataset.extent(1));
  }

  std::cout << "Building Vamana index (search graph)" << std::endl;

  auto start = std::chrono::system_clock::now();
  auto index = vamana::build(dev_resources, index_params, dataset);
  auto end   = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;

  std::cout << "Vamana index has " << index.size() << " vectors" << std::endl;
  std::cout << "Vamana graph has degree " << index.graph_degree() << ", graph size ["
            << index.graph().extent(0) << ", " << index.graph().extent(1) << "]" << std::endl;

  std::cout << "Time to build index: " << elapsed_seconds.count() << "s\n";

  if (codebook_prefix != "") {
    // Output index to file (disk sector-aligned format)
    serialize(dev_resources, out_fname, index, false, true);
  } else {
    // Output index to file (in-memory format)
    serialize(dev_resources, out_fname, index);
  }
}

void usage()
{
  printf(
    "Usage: ./vamana_example <data filename> <output filename> <datatype> "
    "<graph degree> <visited_size> <max_fraction> <iterations> <(optional) "
    "codebook prefix>\n");

  printf("Input file expected to be binary file of fp32 vectors.\n");
  printf("Datatype of input dataset (int8 or float)\n");
  printf("Graph degree sizes supported: 32, 64, 128, 256\n");
  printf("Visited_size must be > degree and a power of 2.\n");
  printf("max_fraction > 0 and <= 1. Typical values are 0.06 or 0.1.\n");
  printf("Default iterations = 1.0, increase for better quality graph.\n");
  printf(
    "Optional path prefix to pq pivots and rotation matrix files. Expects pq pivots file at "
    "${codebook_prefix}_pq_pivots.bin and rotation matrix file at "
    "${codebook_prefix}_pq_pivots.bin_rotation_matrix.bin.\n");
  exit(1);
}

int main(int argc, char* argv[])
{
  raft::device_resources dev_resources;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use
  // the same pool.
  rmm::mr::pool_memory_resource pool_mr(rmm::mr::get_current_device_resource_ref(),
                                        1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(pool_mr);

  // Alternatively, one could define a pool allocator for temporary arrays (used
  // within RAFT algorithms). In that case only the internal arrays would use
  // the pool, any other allocation uses the default RMM memory resource. Here
  // is how to change the workspace memory resource to a pool with 2 GiB upper
  // limit. raft::resource::set_workspace_to_pool_resource(dev_resources, 2 *
  // 1024 * 1024 * 1024ull);

  if (argc != 8 && argc != 9) usage();

  std::string data_fname      = (std::string)(argv[1]);  // Input filename
  std::string out_fname       = (std::string)argv[2];    // Output index filename
  std::string dtype           = (std::string)argv[3];
  int degree                  = atoi(argv[4]);
  int max_visited             = atoi(argv[5]);
  float max_fraction          = atof(argv[6]);
  float iters                 = atof(argv[7]);
  std::string codebook_prefix = "";
  if (argc >= 9)
    codebook_prefix = (std::string)argv[8];  // Path prefix to pq pivots and rotation matrix files

  if (dtype == "int8") {
    // Read in binary dataset file
    auto dataset = read_bin_dataset<int8_t, int64_t>(dev_resources, data_fname, INT_MAX);

    // Simple build example to create graph and write to a file
    vamana_build_and_write<int8_t>(dev_resources,
                                   raft::make_const_mdspan(dataset.view()),
                                   out_fname,
                                   degree,
                                   max_visited,
                                   max_fraction,
                                   iters,
                                   codebook_prefix);
  } else if (dtype == "float") {
    // Read in binary dataset file
    auto dataset = read_bin_dataset<float, int64_t>(dev_resources, data_fname, INT_MAX);

    // Simple build example to create graph and write to a file
    vamana_build_and_write<float>(dev_resources,
                                  raft::make_const_mdspan(dataset.view()),
                                  out_fname,
                                  degree,
                                  max_visited,
                                  max_fraction,
                                  iters,
                                  codebook_prefix);
  } else {
    usage();
  }
}
