/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/flowann_build.hpp>
#include <cuvs/neighbors/flowann_serialize.hpp>

#include <raft/core/device_resources.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace flowann = cuvs::neighbors::cagra::experimental::flowann;
using clock_type  = std::chrono::steady_clock;

namespace {

struct options {
  std::string base_path;
  std::string output_path;
  std::string grouping_graph_path;
  std::uint16_t grouping_bits  = 0;
  std::uint32_t grouping_count = 0;
  std::uint64_t device_graph_budget_bytes{};
  std::uint32_t graph_degree              = 32;
  std::uint32_t intermediate_graph_degree = 96;
  std::uint32_t vpq_dim                   = 192;
  std::uint32_t vpq_bits                  = 8;
  std::uint32_t vq_centers                = 10'000;
  std::uint32_t num_seeds                 = 0;
  std::uint64_t seed_training_rows        = 0;
  std::uint64_t seed                      = 0x9e3779b97f4a7c15ULL;
  bool guarantee_connectivity             = false;
  bool legacy_reference_build             = false;
};

class mapped_fbin {
 public:
  explicit mapped_fbin(std::string const& path)
  {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) { throw std::runtime_error("cannot open base file: " + path); }
    struct stat info{};
    if (fstat(fd_, &info) != 0 || info.st_size < 8) {
      throw std::runtime_error("cannot stat or invalid base file: " + path);
    }
    bytes_   = static_cast<std::size_t>(info.st_size);
    mapping_ = mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapping_ == MAP_FAILED) {
      mapping_ = nullptr;
      throw std::runtime_error("cannot mmap base file: " + path);
    }
    std::memcpy(&rows_, mapping_, sizeof(rows_));
    std::memcpy(&dim_, static_cast<std::byte const*>(mapping_) + sizeof(rows_), sizeof(dim_));
    auto const expected =
      std::uint64_t{8} + static_cast<std::uint64_t>(rows_) * dim_ * sizeof(float);
    if (rows_ == 0 || dim_ == 0 || expected != bytes_) {
      throw std::runtime_error("base file is not a tightly packed float fbin: " + path);
    }
  }

  mapped_fbin(mapped_fbin const&)                    = delete;
  auto operator=(mapped_fbin const&) -> mapped_fbin& = delete;

  ~mapped_fbin()
  {
    if (mapping_ != nullptr) { munmap(mapping_, bytes_); }
    if (fd_ >= 0) { close(fd_); }
  }

  [[nodiscard]] auto rows() const noexcept -> std::uint32_t { return rows_; }
  [[nodiscard]] auto dim() const noexcept -> std::uint32_t { return dim_; }
  [[nodiscard]] auto data() const noexcept -> float const*
  {
    return reinterpret_cast<float const*>(static_cast<std::byte const*>(mapping_) + 8);
  }

 private:
  int fd_{-1};
  void* mapping_{};
  std::size_t bytes_{};
  std::uint32_t rows_{};
  std::uint32_t dim_{};
};

template <typename T>
auto parse_number(char const* text, char const* option) -> T
{
  char* end = nullptr;
  if constexpr (std::is_floating_point_v<T>) {
    auto value = std::strtod(text, &end);
    if (end == text || *end != '\0') { throw std::invalid_argument(option); }
    return static_cast<T>(value);
  } else {
    errno      = 0;
    auto value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || text[0] == '-' || errno == ERANGE ||
        value > std::numeric_limits<T>::max()) {
      throw std::invalid_argument(option);
    }
    return static_cast<T>(value);
  }
}

void print_usage(char const* program)
{
  std::printf(
    "Usage: %s --base BASE.fbin --output INDEX --device-graph-budget-bytes N [options]\n"
    "  --grouping-graph GRAPH.ibin       partition an existing graph; output raw uint32 labels\n"
    "  --grouping-bits N                 0 = automatic 4-bit-aligned width, capped at 24 "
    "(default)\n"
    "  --grouping-count N                0 = automatic (default)\n"
    "  --graph-degree N                 default 32\n"
    "  --intermediate-graph-degree N    default 96\n"
    "  --vpq-dim N                      default 192\n"
    "  --vpq-bits N                     default 8\n"
    "  --vq-centers N                   default 10000\n"
    "  --num-seeds N                    default 0\n"
    "  --seed-training-rows N           default 0 (automatic)\n"
    "  --seed N                         deterministic sample rotation\n"
    "  --legacy-reference-build          use explicit legacy-reference IVF-PQ build parameters\n"
    "  --guarantee-connectivity         enable CAGRA MST connectivity optimization\n",
    program);
}

auto parse_options(int argc, char** argv) -> options
{
  options result;
  for (int i = 1; i < argc; ++i) {
    auto const option = std::string_view(argv[i]);
    auto value        = [&]() -> char const* {
      if (++i >= argc) { throw std::invalid_argument("missing value for " + std::string(option)); }
      return argv[i];
    };
    if (option == "--base") {
      result.base_path = value();
    } else if (option == "--grouping-graph") {
      result.grouping_graph_path = value();
    } else if (option == "--grouping-bits") {
      result.grouping_bits = parse_number<std::uint16_t>(value(), option.data());
    } else if (option == "--grouping-count") {
      result.grouping_count = parse_number<std::uint32_t>(value(), option.data());
    } else if (option == "--output") {
      result.output_path = value();
    } else if (option == "--device-graph-budget-bytes") {
      result.device_graph_budget_bytes = parse_number<std::uint64_t>(value(), option.data());
    } else if (option == "--graph-degree") {
      result.graph_degree = parse_number<std::uint32_t>(value(), option.data());
    } else if (option == "--intermediate-graph-degree") {
      result.intermediate_graph_degree = parse_number<std::uint32_t>(value(), option.data());
    } else if (option == "--vpq-dim") {
      result.vpq_dim = parse_number<std::uint32_t>(value(), option.data());
    } else if (option == "--vpq-bits") {
      result.vpq_bits = parse_number<std::uint32_t>(value(), option.data());
    } else if (option == "--vq-centers") {
      result.vq_centers = parse_number<std::uint32_t>(value(), option.data());
    } else if (option == "--num-seeds") {
      result.num_seeds = parse_number<std::uint32_t>(value(), option.data());
    } else if (option == "--seed-training-rows") {
      result.seed_training_rows = parse_number<std::uint64_t>(value(), option.data());
    } else if (option == "--seed") {
      result.seed = parse_number<std::uint64_t>(value(), option.data());
    } else if (option == "--legacy-reference-build") {
      result.legacy_reference_build = true;
    } else if (option == "--guarantee-connectivity") {
      result.guarantee_connectivity = true;
    } else if (option == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  if (result.base_path.empty() || result.output_path.empty()) {
    throw std::invalid_argument("--base and --output are required");
  }
  return result;
}

auto seconds(clock_type::time_point begin, clock_type::time_point end) -> double
{
  return std::chrono::duration<double>(end - begin).count();
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    auto const args = parse_options(argc, argv);
    auto base       = mapped_fbin(args.base_path);
    auto matrix     = raft::make_host_matrix_view<const float, int64_t>(
      base.data(), static_cast<int64_t>(base.rows()), static_cast<int64_t>(base.dim()));
    auto dataset = cuvs::neighbors::make_host_standard_dataset_view(matrix);

    flowann::index_params params;
    params.grouping.n_bits                        = args.grouping_bits;
    params.grouping.n_groups                      = args.grouping_count;
    params.device_graph_budget_bytes              = args.device_graph_budget_bytes;
    params.cagra_params.metric                    = cuvs::distance::DistanceType::L2Expanded;
    params.cagra_params.graph_degree              = args.graph_degree;
    params.cagra_params.intermediate_graph_degree = args.intermediate_graph_degree;
    params.cagra_params.guarantee_connectivity    = args.guarantee_connectivity;
    auto graph_build = cuvs::neighbors::cagra::graph_build_params::ivf_pq_params(
      raft::matrix_extent<int64_t>(base.rows(), base.dim()), params.cagra_params.metric);
    if (args.legacy_reference_build) {
      // Match the old graph-building benchmark explicitly, independently of search VPQ.
      graph_build                     = cuvs::neighbors::cagra::graph_build_params::ivf_pq_params{};
      graph_build.build_params.metric = params.cagra_params.metric;
      graph_build.build_params.n_lists                  = 1024;
      graph_build.build_params.kmeans_n_iters           = 20;
      graph_build.build_params.kmeans_trainset_fraction = 0.1;
      graph_build.build_params.pq_dim                   = 192;
      graph_build.build_params.pq_bits                  = 8;
      graph_build.build_params.codebook_kind = cuvs::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE;
      graph_build.build_params.force_random_rotation        = false;
      graph_build.build_params.max_train_points_per_pq_code = 256;
      graph_build.search_params.n_probes                    = 20;
      graph_build.search_params.lut_dtype                   = CUDA_R_32F;
      graph_build.search_params.internal_distance_dtype     = CUDA_R_32F;
      graph_build.search_params.coarse_search_dtype         = CUDA_R_32F;
      // Both profiles request at least intermediate_degree + 1 initial candidates.
      graph_build.refinement_rate = 1.0f;
    }
    params.cagra_params.graph_build_params = graph_build;
    std::printf(
      "GRAPH_BUILD profile=%s n_lists=%u n_probes=%u pq_dim=%u pq_bits=%u "
      "kmeans_n_iters=%u train_fraction=%.9f refinement_rate=%.3f "
      "lut_dtype=%d internal_dtype=%d coarse_dtype=%d\n",
      args.legacy_reference_build ? "legacy-reference" : "dataset-default",
      graph_build.build_params.n_lists,
      graph_build.search_params.n_probes,
      graph_build.build_params.pq_dim,
      graph_build.build_params.pq_bits,
      graph_build.build_params.kmeans_n_iters,
      graph_build.build_params.kmeans_trainset_fraction,
      graph_build.refinement_rate,
      static_cast<int>(graph_build.search_params.lut_dtype),
      static_cast<int>(graph_build.search_params.internal_distance_dtype),
      static_cast<int>(graph_build.search_params.coarse_search_dtype));
    std::fflush(stdout);
    params.num_seeds          = args.num_seeds;
    params.seed_training_rows = args.seed_training_rows;
    params.seed               = args.seed;

    cuvs::neighbors::vpq_params vpq;
    vpq.pq_dim       = args.vpq_dim;
    vpq.pq_bits      = args.vpq_bits;
    vpq.vq_n_centers = args.vq_centers;

    std::printf(
      "CONFIG rows=%u dim=%u graph_degree=%u intermediate_degree=%u graph_budget_bytes=%llu "
      "vpq_dim=%u vpq_bits=%u vq_centers=%u seeds=%u seed_training_rows=%llu "
      "guarantee_connectivity=%s\n",
      base.rows(),
      base.dim(),
      args.graph_degree,
      args.intermediate_graph_degree,
      static_cast<unsigned long long>(args.device_graph_budget_bytes),
      args.vpq_dim,
      args.vpq_bits,
      args.vq_centers,
      args.num_seeds,
      static_cast<unsigned long long>(args.seed_training_rows),
      args.guarantee_connectivity ? "true" : "false");

    raft::device_resources resources;
    if (!args.grouping_graph_path.empty()) {
      auto graph = mapped_fbin(args.grouping_graph_path);
      if (graph.rows() != base.rows()) { throw std::invalid_argument("graph/base row mismatch"); }
      auto graph_view = raft::make_host_matrix_view<const std::uint32_t, int64_t>(
        reinterpret_cast<std::uint32_t const*>(graph.data()), graph.rows(), graph.dim());
      auto const begin = clock_type::now();
      auto grouped     = flowann::group_graph(resources, params, graph_view, dataset);
      auto const end   = clock_type::now();
      auto* output     = std::fopen(args.output_path.c_str(), "wb");
      if (!output) { throw std::runtime_error("cannot open grouping output"); }
      auto const written = std::fwrite(
        grouped.labels.data_handle(), sizeof(std::uint32_t), grouped.labels.size(), output);
      auto const closed = std::fclose(output);
      if (written != grouped.labels.size() || closed != 0) {
        throw std::runtime_error("cannot write grouping output");
      }
      std::printf("GROUPING_RESULT seconds=%.6f n_bits=%u n_groups=%u rows=%u\n",
                  seconds(begin, end),
                  grouped.n_bits,
                  grouped.n_groups,
                  base.rows());
      return 0;
    }
    auto const build_begin = clock_type::now();
    auto built             = flowann::build(resources, params, vpq, dataset);
    auto const build_end   = clock_type::now();
    flowann::serialize(resources, args.output_path, built.index);
    auto const serialize_end = clock_type::now();
    std::printf(
      "BUILD_RESULT build_seconds=%.6f serialize_seconds=%.6f total_seconds=%.6f "
      "resident_degree=%u cross_degree=%u n_bits=%u resident_bytes=%zu cross_bytes=%zu\n",
      seconds(build_begin, build_end),
      seconds(build_end, serialize_end),
      seconds(build_begin, serialize_end),
      built.index.resident_degree(),
      built.index.cross_degree(),
      built.index.n_bits(),
      built.index.inner_graph().size() * sizeof(std::uint8_t),
      built.index.cross_graph().size() * sizeof(std::uint32_t));
    return 0;
  } catch (std::exception const& error) {
    std::fprintf(stderr, "FLOWANN_BUILD_BENCHMARK error: %s\n", error.what());
    return 1;
  }
}
