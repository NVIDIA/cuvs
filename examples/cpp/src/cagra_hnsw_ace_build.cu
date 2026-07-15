/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <filesystem>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/memory_tracking_resources.hpp>
#include <raft/core/resources.hpp>
#include <string>

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/hnsw.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/util/host_memory.hpp>

#include <rmm/mr/pool_memory_resource.hpp>

#include <cstdio>
#include <cstdlib>  // for exit
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

struct app_args {
  std::string base_path;
  std::string stats_path;
  int M               = 24;
  int ef_construction = 200;
};

app_args parse_args(int argc, char** argv)
{
  app_args args;
  bool has_base_path = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--path" && i + 1 < argc) {
      args.base_path = argv[++i];
      has_base_path  = true;
    } else if (a == "--stats-path" && i + 1 < argc) {
      args.stats_path = argv[++i];
    } else if (a == "--m" && i + 1 < argc) {
      args.M = std::stoi(argv[++i]);
    } else if (a == "--efc" && i + 1 < argc) {
      args.ef_construction = std::stoi(argv[++i]);
    } else {
      std::cerr << "Usage: " << argv[0]
                << " --path <file> [--stats-path <file>] [--m M] [--efc EF_CONSTRUCTION]\n";
      std::exit(EXIT_FAILURE);
    }
  }
  if (!has_base_path) {
    std::cerr << "Error: --path is required\n";
    std::exit(EXIT_FAILURE);
  }
  if (!std::filesystem::exists(args.base_path)) {
    std::cerr << "Error: file not found: " << args.base_path << "\n";
    std::exit(EXIT_FAILURE);
  }
  if (args.stats_path.empty()) {
    auto dataset    = std::filesystem::path(args.base_path).parent_path().filename().string();
    args.stats_path = "allocations-" + dataset + ".csv";
  }
  return args;
}

std::string detect_dtype(const std::string& filename)
{
  if (filename.size() > 6 && filename.compare(filename.size() - 6, 6, "f16bin") == 0) {
    return "half";
  } else if (filename.size() > 9 && filename.compare(filename.size() - 9, 9, "fp16.fbin") == 0) {
    return "half";
  } else if (filename.size() > 4 && filename.compare(filename.size() - 4, 4, "fbin") == 0) {
    return "float";
  } else if (filename.size() > 5 && filename.compare(filename.size() - 5, 5, "u8bin") == 0) {
    return "uint8";
  } else if (filename.size() > 5 && filename.compare(filename.size() - 5, 5, "i8bin") == 0) {
    return "int8";
  }
  std::cerr << "Cannot determine data type from extension: " << filename << "\n";
  std::exit(EXIT_FAILURE);
}

template <typename T>
auto hnsw_build_example(raft::resources const& res, const app_args& args) -> int
{
  using namespace cuvs::neighbors;

  int fd = open(args.base_path.c_str(), O_RDONLY);
  if (fd == -1) {
    perror("Error opening file");
    return EXIT_FAILURE;
  }
  uint32_t shape[2];
  ssize_t bytesRead = read(fd, shape, 8);
  if (bytesRead != 8) {
    perror("Error reading shape");
    close(fd);
    return EXIT_FAILURE;
  }
  size_t data_size = shape[0] * static_cast<size_t>(shape[1]);
  std::cout << "Dataset size " << data_size << std::endl;
  size_t header_size   = sizeof(shape);
  size_t file_size     = data_size * sizeof(T) + header_size;
  uint8_t* dataset_ptr = (uint8_t*)mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
  std::cout << "shape [" << shape[0] << ", " << shape[1] << "]" << std::endl;
  if (dataset_ptr == MAP_FAILED) {
    perror("Error mmapping the file");
    close(fd);
    return EXIT_FAILURE;
  }
  uint32_t n_rows        = shape[0];
  auto dataset_host_view = raft::make_host_matrix_view<const T, int64_t, raft::row_major>(
    reinterpret_cast<const T*>(dataset_ptr + header_size), n_rows, shape[1]);

  hnsw::index_params params;
  params.M               = args.M;
  params.ef_construction = args.ef_construction;
  params.hierarchy       = cuvs::neighbors::hnsw::HnswHierarchy::GPU;

  auto hnsw_index = hnsw::build(res, params, dataset_host_view);

  std::string tmp_template =
    (std::filesystem::temp_directory_path() / "hnsw_index_XXXXXX").string();
  if (mkdtemp(tmp_template.data()) == nullptr) {
    perror("Error creating temp directory");
    munmap(dataset_ptr, file_size);
    close(fd);
    return EXIT_FAILURE;
  }
  std::string hnsw_index_path = tmp_template + "/hnsw_index.bin";
  cuvs::neighbors::hnsw::serialize(res, hnsw_index_path, *hnsw_index);
  std::cout << "HNSW index file location: " << hnsw_index_path << std::endl;

  std::filesystem::remove_all(tmp_template);

  munmap(dataset_ptr, file_size);
  close(fd);
  return 0;
}

int main(int argc, char** argv)
{
  auto args = parse_args(argc, argv);

  raft::resources res;

  // // Set pool memory resource with 1 GiB initial pool size. All allocations use the same pool.
  // rmm::mr::pool_memory_resource pool_mr(rmm::mr::get_current_device_resource_ref(), 1024 * 1024 *
  // 1024ull); rmm::mr::set_current_device_resource(pool_mr);

  // Alternatively, one could define a pool allocator for temporary arrays (used within RAFT
  // algorithms). In that case only the internal arrays would use the pool, any other allocation
  // uses the default RMM memory resource. Here is how to change the workspace memory resource to
  // a pool with 2 GiB upper limit.
  // raft::resource::set_workspace_to_pool_resource(res, 2 * 1024 * 1024 * 1024ull);

  // Manually configure pool to allocate up to its limit right away to avoid fragmentation.
  constexpr std::size_t kWorkspaceLimit = 2ull * 1024 * 1024 * 1024;  // 2 GiB
  rmm::mr::pool_memory_resource pool_mr(
    rmm::mr::get_current_device_resource_ref(), kWorkspaceLimit, kWorkspaceLimit);
  raft::resource::set_workspace_resource(
    res, raft::mr::device_resource{std::move(pool_mr)}, kWorkspaceLimit);

  raft::memory_tracking_resources tracked(res, args.stats_path, std::chrono::milliseconds(1));

  auto dtype = detect_dtype(args.base_path);
  int result = 0;
  if (dtype == "float") {
    result = hnsw_build_example<float>(tracked, args);
  } else if (dtype == "half") {
    result = hnsw_build_example<half>(tracked, args);
  } else if (dtype == "uint8") {
    result = hnsw_build_example<uint8_t>(tracked, args);
  } else if (dtype == "int8") {
    result = hnsw_build_example<int8_t>(tracked, args);
  }
  return result;
}
