/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/flowann.hpp>
#include <cuvs/neighbors/flowann_serialize.hpp>
#include <cuvs/neighbors/refine.hpp>

#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuda_runtime.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flowann = cuvs::neighbors::cagra::experimental::flowann;
using clock_type  = std::chrono::steady_clock;

namespace {

struct benchmark_options {
  std::string index_path;
  std::string legacy_graph_path;
  std::string legacy_dataset_path;
  std::string base_path;
  std::string queries_path;
  std::string ground_truth_path;
  std::string seeds_path;

  std::uint32_t gpu_id              = 0;
  std::uint32_t batch_size          = 1;
  std::uint32_t query_count         = 0;
  std::uint32_t top_k               = 10;
  std::uint32_t candidate_count     = 32;
  std::uint32_t team_size           = 0;
  std::uint32_t thread_block_size   = 0;
  std::uint32_t itopk_size          = 64;
  std::uint32_t search_width        = 1;
  std::uint32_t min_iterations      = 0;
  std::uint32_t max_iterations      = 0;
  std::uint32_t num_queues          = 1;
  std::uint32_t empty_pause         = 64;
  std::uint32_t warmup_rounds       = 1;
  std::uint32_t measured_rounds     = 3;
  std::uint32_t cpu_threads         = 1;
  std::uint32_t sync_drop_threshold = 51;
  std::uint64_t pool_size_mib       = 2048;
  float sync_window_scale           = 10.0f;
  std::optional<std::uint32_t> num_seeds;

  cuvs::neighbors::cagra::search_algo algorithm     = cuvs::neighbors::cagra::search_algo::AUTO;
  cuvs::neighbors::cagra::internal_dtype smem_dtype = cuvs::neighbors::cagra::internal_dtype::F16;
  bool use_search_session                           = true;
  bool rerank                                       = true;
  bool collect_statistics                           = false;
  bool include_query_transfer                       = false;
};

template <typename T>
struct host_matrix_file {
  std::uint32_t rows{};
  std::uint32_t cols{};
  std::vector<T> values;
};

template <typename T>
class mapped_matrix_file {
 public:
  explicit mapped_matrix_file(std::string const& path)
  {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) { throw std::runtime_error("cannot open " + path); }
    struct stat info{};
    if (fstat(fd_, &info) != 0 || info.st_size < 8) {
      throw std::runtime_error("cannot stat or invalid matrix file " + path);
    }
    bytes_   = static_cast<std::size_t>(info.st_size);
    mapping_ = mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapping_ == MAP_FAILED) {
      mapping_ = nullptr;
      throw std::runtime_error("cannot mmap " + path);
    }
    std::memcpy(&rows_, mapping_, sizeof(rows_));
    std::memcpy(&cols_, static_cast<std::byte const*>(mapping_) + sizeof(rows_), sizeof(cols_));
    auto const elements = static_cast<std::uint64_t>(rows_) * cols_;
    auto const expected = std::uint64_t{8} + elements * sizeof(T);
    if (rows_ == 0 || cols_ == 0 || expected != bytes_) {
      throw std::runtime_error("invalid tightly packed matrix file " + path);
    }
  }

  mapped_matrix_file(mapped_matrix_file const&)                    = delete;
  auto operator=(mapped_matrix_file const&) -> mapped_matrix_file& = delete;

  ~mapped_matrix_file()
  {
    if (mapping_ != nullptr) { munmap(mapping_, bytes_); }
    if (fd_ >= 0) { close(fd_); }
  }

  [[nodiscard]] auto rows() const noexcept -> std::uint32_t { return rows_; }
  [[nodiscard]] auto cols() const noexcept -> std::uint32_t { return cols_; }
  [[nodiscard]] auto data() const noexcept -> T const*
  {
    return reinterpret_cast<T const*>(static_cast<std::byte const*>(mapping_) + 8);
  }

 private:
  int fd_{-1};
  void* mapping_{};
  std::size_t bytes_{};
  std::uint32_t rows_{};
  std::uint32_t cols_{};
};

struct round_result {
  double search_ms{};
  double rerank_prep_ms{};
  double rerank_ms{};
  double total_ms{};
  double recall = std::numeric_limits<double>::quiet_NaN();
  double qps{};
  double p50_ms{};
  double p95_ms{};
  double p99_ms{};
};

class scoped_device_resource {
 public:
  template <typename Resource>
  explicit scoped_device_resource(Resource& resource)
    : previous_(rmm::mr::set_current_device_resource(resource))
  {
  }

  ~scoped_device_resource() { rmm::mr::set_current_device_resource(std::move(previous_)); }

  scoped_device_resource(scoped_device_resource const&)                    = delete;
  auto operator=(scoped_device_resource const&) -> scoped_device_resource& = delete;

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> previous_;
};

void print_usage(char const* program)
{
  std::printf(
    "FlowANN VPQ search benchmark\n\n"
    "Usage:\n"
    "  %s --index INDEX --queries QUERIES [options]\n"
    "  %s --legacy-graph GRAPH --legacy-dataset DATASET --queries QUERIES [options]\n\n"
    "Index and input paths:\n"
    "  --index PATH              Self-contained index or legacy combined v5 split.bin\n"
    "  --legacy-graph PATH       Legacy split graph file\n"
    "  --legacy-dataset PATH     Legacy VPQ-F16 dataset-only file\n"
    "  --queries PATH            Query matrix in fbin format\n"
    "  --ground-truth PATH       Optional ground truth matrix in ibin format\n"
    "  --base PATH               Base vectors in fbin format, required for CPU rerank\n"
    "  --seeds PATH              Optional raw uint32 medoid seed file\n\n"
    "Search parameters:\n"
    "  --gpu-id N                CUDA device (default: 0)\n"
    "  --batch-size N            Queries per search call (default: 1)\n"
    "  --query-count N           Prefix to benchmark, 0 means all (default: 0)\n"
    "  --top-k N                 Recall and final result width (default: 10)\n"
    "  --candidate-count N       FlowANN candidates before rerank (default: 32)\n"
    "  --algorithm NAME          auto, single_cta, or multi_cta (default: auto)\n"
    "  --team-size N             Distance team size, 0 means auto (default: 0)\n"
    "  --thread-block-size N     CUDA threads per block, 0 means auto (default: 0)\n"
    "  --itopk-size N            Internal top-k size (default: 64)\n"
    "  --search-width N          Search width (default: 1)\n"
    "  --min-iterations N        Minimum graph iterations (default: 0)\n"
    "  --max-iterations N        Maximum graph iterations, 0 means auto (default: 0)\n"
    "  --num-seeds N             Number of loaded seeds to use; default is all loaded seeds\n"
    "  --smem-dtype NAME         f16 or e5m2 (default: f16)\n"
    "  --sync-window-scale X     Deferred request window scale (default: 10)\n"
    "  --sync-drop-threshold N   Forced synchronization threshold (default: 51)\n\n"
    "Queue, session, and measurement parameters:\n"
    "  --num-queues N            CPU-serviced queue count (default: 1)\n"
    "  --empty-pause N           Pause instructions after an empty poll (default: 64)\n"
    "  --cpu-threads N           OpenMP threads used by CPU rerank (default: 1)\n"
    "  --warmup-rounds N         Complete warmup passes (default: 1)\n"
    "  --measured-rounds N       Complete measured passes (default: 3)\n"
    "  --pool-size-mib N         Managed-memory RMM pool, 0 disables it (default: 2048)\n"
    "  --queue-statistics        Collect and print queue counters\n"
    "  --no-session              Disable the default persistent search_session\n"
    "  --no-rerank               Return FlowANN candidates without CPU exact rerank\n"
    "  --include-query-transfer  Include query upload in search and total latency\n"
    "  --help                    Show this message\n\n"
    "fbin/ibin files contain uint32 rows and columns followed by row-major values.\n"
    "The selected query count must be divisible by the batch size.\n",
    program,
    program);
}

template <typename T>
auto parse_number(std::string_view text, std::string_view option) -> T
{
  T value{};
  auto const* begin      = text.data();
  auto const* end        = begin + text.size();
  auto [position, error] = std::from_chars(begin, end, value);
  if (error != std::errc{} || position != end) {
    throw std::invalid_argument("invalid value for " + std::string(option) + ": " +
                                std::string(text));
  }
  return value;
}

template <>
auto parse_number<float>(std::string_view text, std::string_view option) -> float
{
  std::string value_text{text};
  char* end        = nullptr;
  auto const value = std::strtof(value_text.c_str(), &end);
  if (end == value_text.c_str() || end != value_text.c_str() + value_text.size() ||
      !std::isfinite(value)) {
    throw std::invalid_argument("invalid value for " + std::string(option) + ": " + value_text);
  }
  return value;
}

auto parse_algorithm(std::string_view value) -> cuvs::neighbors::cagra::search_algo
{
  if (value == "auto") { return cuvs::neighbors::cagra::search_algo::AUTO; }
  if (value == "single_cta") { return cuvs::neighbors::cagra::search_algo::SINGLE_CTA; }
  if (value == "multi_cta") { return cuvs::neighbors::cagra::search_algo::MULTI_CTA; }
  throw std::invalid_argument("--algorithm must be auto, single_cta, or multi_cta");
}

auto parse_smem_dtype(std::string_view value) -> cuvs::neighbors::cagra::internal_dtype
{
  if (value == "f16") { return cuvs::neighbors::cagra::internal_dtype::F16; }
  if (value == "e5m2") { return cuvs::neighbors::cagra::internal_dtype::E5M2; }
  throw std::invalid_argument("--smem-dtype must be f16 or e5m2");
}

auto parse_options(int argc, char** argv) -> std::optional<benchmark_options>
{
  benchmark_options options;
  auto require_value = [&](int& position, std::string_view option) -> std::string_view {
    if (++position >= argc) {
      throw std::invalid_argument(std::string(option) + " requires a value");
    }
    return argv[position];
  };

  for (int i = 1; i < argc; ++i) {
    std::string_view option = argv[i];
    if (option == "--help" || option == "-h") {
      print_usage(argv[0]);
      return std::nullopt;
    } else if (option == "--index") {
      options.index_path = require_value(i, option);
    } else if (option == "--legacy-graph") {
      options.legacy_graph_path = require_value(i, option);
    } else if (option == "--legacy-dataset") {
      options.legacy_dataset_path = require_value(i, option);
    } else if (option == "--base") {
      options.base_path = require_value(i, option);
    } else if (option == "--queries") {
      options.queries_path = require_value(i, option);
    } else if (option == "--ground-truth") {
      options.ground_truth_path = require_value(i, option);
    } else if (option == "--seeds") {
      options.seeds_path = require_value(i, option);
    } else if (option == "--gpu-id") {
      options.gpu_id = parse_number<std::uint32_t>(require_value(i, option), option);
    } else if (option == "--batch-size") {
      options.batch_size = parse_number<std::uint32_t>(require_value(i, option), option);
    } else if (option == "--query-count") {
      options.query_count = parse_number<std::uint32_t>(require_value(i, option), option);
    } else if (option == "--top-k") {
      options.top_k = parse_number<std::uint32_t>(require_value(i, option), option);
    } else if (option == "--candidate-count") {
      options.candidate_count = parse_number<std::uint32_t>(require_value(i, option), option);
    } else if (option == "--algorithm") {
      options.algorithm = parse_algorithm(require_value(i, option));
    } else if (option == "--team-size") {
      options.team_size = parse_number<std::uint32_t>(require_value(i, option), option);
    } else if (option == "--thread-block-size") {
      options.thread_block_size = parse_number<std::uint32_t>(require_value(i, option), option);
    } else if (option == "--itopk-size") {
      options.itopk_size = parse_number<std::uint32_t>(require_value(i, option), option);
    } else if (option == "--search-width") {
      options.search_width = parse_number<std::uint32_t>(require_value(i, option), option);
    } else if (option == "--min-iterations") {
      options.min_iterations = parse_number<std::uint32_t>(require_value(i, option), option);
    } else if (option == "--max-iterations") {
      options.max_iterations = parse_number<std::uint32_t>(require_value(i, option), option);
    } else if (option == "--num-seeds") {
      options.num_seeds = parse_number<std::uint32_t>(require_value(i, option), option);
    } else if (option == "--smem-dtype") {
      options.smem_dtype = parse_smem_dtype(require_value(i, option));
    } else if (option == "--sync-window-scale") {
      options.sync_window_scale = parse_number<float>(require_value(i, option), option);
    } else if (option == "--sync-drop-threshold") {
      options.sync_drop_threshold = parse_number<std::uint32_t>(require_value(i, option), option);
    } else if (option == "--num-queues") {
      options.num_queues = parse_number<std::uint32_t>(require_value(i, option), option);
    } else if (option == "--empty-pause") {
      options.empty_pause = parse_number<std::uint32_t>(require_value(i, option), option);
    } else if (option == "--cpu-threads") {
      options.cpu_threads = parse_number<std::uint32_t>(require_value(i, option), option);
    } else if (option == "--warmup-rounds") {
      options.warmup_rounds = parse_number<std::uint32_t>(require_value(i, option), option);
    } else if (option == "--measured-rounds") {
      options.measured_rounds = parse_number<std::uint32_t>(require_value(i, option), option);
    } else if (option == "--pool-size-mib") {
      options.pool_size_mib = parse_number<std::uint64_t>(require_value(i, option), option);
    } else if (option == "--queue-statistics") {
      options.collect_statistics = true;
    } else if (option == "--no-session") {
      options.use_search_session = false;
    } else if (option == "--no-rerank") {
      options.rerank = false;
    } else if (option == "--include-query-transfer") {
      options.include_query_transfer = true;
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }

  auto const has_single_file_index = !options.index_path.empty();
  auto const has_legacy_index =
    !options.legacy_graph_path.empty() || !options.legacy_dataset_path.empty();
  if (has_single_file_index == has_legacy_index) {
    throw std::invalid_argument(
      "select exactly one index mode: --index, or --legacy-graph plus --legacy-dataset");
  }
  if (has_legacy_index &&
      (options.legacy_graph_path.empty() || options.legacy_dataset_path.empty())) {
    throw std::invalid_argument("legacy index mode requires both files");
  }
  if (options.queries_path.empty()) { throw std::invalid_argument("--queries is required"); }
  if (options.rerank && options.base_path.empty()) {
    throw std::invalid_argument("--base is required unless --no-rerank is used");
  }
  if (options.batch_size == 0 || options.top_k == 0 || options.candidate_count == 0 ||
      options.itopk_size == 0 || options.search_width == 0 || options.num_queues == 0 ||
      options.cpu_threads == 0 || options.measured_rounds == 0) {
    throw std::invalid_argument(
      "batch, result, queue, thread, and measured-round values must be positive");
  }
  if (options.candidate_count < options.top_k) {
    throw std::invalid_argument("--candidate-count must be at least --top-k");
  }
  if (options.candidate_count > options.itopk_size) {
    throw std::invalid_argument("--candidate-count must not exceed --itopk-size");
  }
  if (options.sync_window_scale <= 0.0f) {
    throw std::invalid_argument("--sync-window-scale must be positive");
  }
  if (options.cpu_threads > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("--cpu-threads exceeds the OpenMP limit");
  }
  if (options.pool_size_mib > std::numeric_limits<std::uint64_t>::max() / (1024ULL * 1024ULL)) {
    throw std::invalid_argument("--pool-size-mib is too large");
  }
  return options;
}

template <typename T>
auto load_matrix_file(std::string const& path, std::uint32_t row_limit = 0) -> host_matrix_file<T>
{
  std::ifstream input(path, std::ios::binary);
  if (!input) { throw std::runtime_error("cannot open " + path); }

  host_matrix_file<T> result;
  input.read(reinterpret_cast<char*>(&result.rows), sizeof(result.rows));
  input.read(reinterpret_cast<char*>(&result.cols), sizeof(result.cols));
  if (!input || result.rows == 0 || result.cols == 0) {
    throw std::runtime_error("invalid matrix header in " + path);
  }
  if (row_limit != 0) { result.rows = std::min(result.rows, row_limit); }

  auto const count = static_cast<std::size_t>(result.rows) * result.cols;
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::runtime_error("matrix size overflows in " + path);
  }
  result.values.resize(count);
  input.read(reinterpret_cast<char*>(result.values.data()),
             static_cast<std::streamsize>(count * sizeof(T)));
  if (!input) { throw std::runtime_error("truncated matrix in " + path); }
  return result;
}

auto load_u32_vector(std::string const& path) -> std::vector<std::uint32_t>
{
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) { throw std::runtime_error("cannot open " + path); }
  auto const bytes = input.tellg();
  if (bytes <= 0 || bytes % sizeof(std::uint32_t) != 0) {
    throw std::runtime_error("invalid raw uint32 file " + path);
  }
  std::vector<std::uint32_t> values(static_cast<std::size_t>(bytes) / sizeof(std::uint32_t));
  input.seekg(0);
  input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
  if (!input) { throw std::runtime_error("truncated raw uint32 file " + path); }
  return values;
}

auto milliseconds(clock_type::time_point begin, clock_type::time_point end) -> double
{
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

auto percentile(std::vector<double> values, double fraction) -> double
{
  if (values.empty()) { return 0.0; }
  std::sort(values.begin(), values.end());
  auto const rank =
    static_cast<std::size_t>(std::ceil(fraction * static_cast<double>(values.size())));
  return values[std::min(values.size() - 1, std::max<std::size_t>(1, rank) - 1)];
}

auto median(std::vector<double> values) -> double
{
  if (values.empty()) { return 0.0; }
  std::sort(values.begin(), values.end());
  auto const middle = values.size() / 2;
  return values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2.0 : values[middle];
}

auto recall_at_k(std::vector<std::uint32_t> const& actual,
                 host_matrix_file<std::uint32_t> const& truth,
                 std::uint32_t top_k) -> double
{
  std::uint64_t matches  = 0;
  auto const query_count = actual.size() / top_k;
  for (std::size_t query = 0; query < query_count; ++query) {
    for (std::uint32_t i = 0; i < top_k; ++i) {
      auto const id = actual[query * top_k + i];
      for (std::uint32_t j = 0; j < top_k; ++j) {
        if (id == truth.values[query * truth.cols + j]) {
          ++matches;
          break;
        }
      }
    }
  }
  return static_cast<double>(matches) / static_cast<double>(query_count * top_k);
}

auto algorithm_name(cuvs::neighbors::cagra::search_algo algorithm) -> char const*
{
  switch (algorithm) {
    case cuvs::neighbors::cagra::search_algo::SINGLE_CTA: return "single_cta";
    case cuvs::neighbors::cagra::search_algo::MULTI_CTA: return "multi_cta";
    case cuvs::neighbors::cagra::search_algo::AUTO: return "auto";
    default: return "unsupported";
  }
}

auto run_round(raft::device_resources& resources,
               flowann::vpq_f16_index<float>& index,
               flowann::search_context& context,
               benchmark_options const& options,
               mapped_matrix_file<float> const* base,
               host_matrix_file<float> const& queries,
               std::optional<host_matrix_file<std::uint32_t>> const& truth) -> round_result
{
  auto const batch_count = queries.rows / options.batch_size;
  flowann::search_params params{};
  params.algo                = options.algorithm;
  params.team_size           = options.team_size;
  params.thread_block_size   = options.thread_block_size;
  params.search_width        = options.search_width;
  params.itopk_size          = options.itopk_size;
  params.min_iterations      = options.min_iterations;
  params.max_iterations      = options.max_iterations;
  params.max_queries         = options.batch_size;
  params.num_seeds           = options.num_seeds.value_or(0);
  params.smem_dtype          = options.smem_dtype;
  params.sync_window_scale   = options.sync_window_scale;
  params.sync_drop_threshold = options.sync_drop_threshold;

  auto query_device =
    raft::make_device_matrix<float, int64_t>(resources, options.batch_size, queries.cols);
  auto candidates_device = raft::make_device_matrix<std::uint32_t, int64_t>(
    resources, options.batch_size, options.candidate_count);
  auto candidate_distances_device = raft::make_device_matrix<float, int64_t>(
    resources, options.batch_size, options.candidate_count);

  std::vector<std::uint32_t> candidates_host(static_cast<std::size_t>(options.batch_size) *
                                             options.candidate_count);
  std::vector<std::uint32_t> final_indices(static_cast<std::size_t>(options.batch_size) *
                                           options.top_k);
  std::vector<float> final_distances(static_cast<std::size_t>(options.batch_size) * options.top_k);
  std::vector<std::uint32_t> all_results(static_cast<std::size_t>(queries.rows) * options.top_k);
  std::vector<double> batch_latencies;
  batch_latencies.reserve(batch_count);

  double search_total      = 0.0;
  double rerank_prep_total = 0.0;
  double rerank_total      = 0.0;
  double total             = 0.0;
  auto const stream        = raft::resource::get_cuda_stream(resources);

  for (std::uint32_t batch = 0; batch < batch_count; ++batch) {
    auto const query_offset  = static_cast<std::size_t>(batch) * options.batch_size * queries.cols;
    auto const before_upload = clock_type::now();
    raft::copy(query_device.data_handle(),
               queries.values.data() + query_offset,
               static_cast<std::size_t>(options.batch_size) * queries.cols,
               stream);
    raft::resource::sync_stream(resources);

    auto const begin = options.include_query_transfer ? before_upload : clock_type::now();
    flowann::search(resources,
                    params,
                    index,
                    context,
                    raft::make_const_mdspan(query_device.view()),
                    candidates_device.view(),
                    candidate_distances_device.view());
    auto const after_search = clock_type::now();

    raft::copy(
      candidates_host.data(), candidates_device.data_handle(), candidates_host.size(), stream);
    raft::resource::sync_stream(resources);
    auto const row_limit = base != nullptr ? base->rows() : index.size();
    for (auto& candidate : candidates_host) {
      if (candidate >= row_limit) { candidate = 0; }
    }
    auto const after_prep = clock_type::now();

    if (options.rerank) {
      auto const dataset_view = raft::make_host_matrix_view<const float, int64_t, raft::row_major>(
        base->data(), base->rows(), base->cols());
      auto const query_view = raft::make_host_matrix_view<const float, int64_t, raft::row_major>(
        queries.values.data() + query_offset, options.batch_size, queries.cols);
      auto const candidate_view =
        raft::make_host_matrix_view<const std::uint32_t, int64_t, raft::row_major>(
          candidates_host.data(), options.batch_size, options.candidate_count);
      auto const index_view = raft::make_host_matrix_view<std::uint32_t, int64_t, raft::row_major>(
        final_indices.data(), options.batch_size, options.top_k);
      auto const distance_view = raft::make_host_matrix_view<float, int64_t, raft::row_major>(
        final_distances.data(), options.batch_size, options.top_k);
      cuvs::neighbors::refine(resources,
                              dataset_view,
                              query_view,
                              candidate_view,
                              index_view,
                              distance_view,
                              cuvs::distance::DistanceType::L2Expanded);
    } else {
      for (std::uint32_t query = 0; query < options.batch_size; ++query) {
        std::copy_n(
          candidates_host.data() + static_cast<std::size_t>(query) * options.candidate_count,
          options.top_k,
          final_indices.data() + static_cast<std::size_t>(query) * options.top_k);
      }
    }
    auto const end = clock_type::now();

    auto const result_offset = static_cast<std::size_t>(batch) * options.batch_size * options.top_k;
    std::copy(final_indices.begin(), final_indices.end(), all_results.begin() + result_offset);

    auto const search_ms = milliseconds(begin, after_search);
    auto const prep_ms   = milliseconds(after_search, after_prep);
    auto const rerank_ms = milliseconds(after_prep, end);
    auto const batch_ms  = milliseconds(begin, end);
    search_total += search_ms;
    rerank_prep_total += prep_ms;
    rerank_total += rerank_ms;
    total += batch_ms;
    batch_latencies.push_back(batch_ms);
  }

  round_result result;
  result.search_ms      = search_total / batch_count;
  result.rerank_prep_ms = rerank_prep_total / batch_count;
  result.rerank_ms      = rerank_total / batch_count;
  result.total_ms       = total / batch_count;
  result.qps            = static_cast<double>(queries.rows) / (total / 1000.0);
  result.p50_ms         = percentile(batch_latencies, 0.50);
  result.p95_ms         = percentile(batch_latencies, 0.95);
  result.p99_ms         = percentile(batch_latencies, 0.99);
  if (truth) { result.recall = recall_at_k(all_results, *truth, options.top_k); }
  return result;
}

void print_round(char const* kind,
                 std::uint32_t round,
                 std::uint32_t query_count,
                 benchmark_options const& options,
                 round_result const& result)
{
  std::printf(
    "%s round=%u batch=%u query_count=%u search_ms=%.6f rerank_prep_ms=%.6f rerank_ms=%.6f "
    "total_ms=%.6f recall=%.6f qps=%.3f p50_ms=%.6f p95_ms=%.6f p99_ms=%.6f\n",
    kind,
    round,
    options.batch_size,
    query_count,
    result.search_ms,
    result.rerank_prep_ms,
    result.rerank_ms,
    result.total_ms,
    result.recall,
    result.qps,
    result.p50_ms,
    result.p95_ms,
    result.p99_ms);
}

auto run_benchmark(benchmark_options options) -> int
{
  raft::device_resources resources;
  auto const load_index_begin = clock_type::now();
  auto bundle                 = [&]() -> flowann::vpq_f16_index_bundle<float> {
    if (!options.index_path.empty()) {
      return flowann::deserialize_vpq_f16<float>(resources, options.index_path);
    }
    return flowann::deserialize_vpq_f16<float>(
      resources, options.legacy_graph_path, options.legacy_dataset_path);
  }();

  std::vector<std::uint32_t> seeds;
  if (!options.seeds_path.empty()) {
    seeds = load_u32_vector(options.seeds_path);
    bundle.index.update_seeds(
      resources,
      raft::make_host_vector_view<const std::uint32_t, int64_t>(seeds.data(), seeds.size()));
    raft::resource::sync_stream(resources);
  }
  auto const index_seeds = bundle.index.seeds();
  auto const available_seeds =
    index_seeds.has_value() ? static_cast<std::uint32_t>(index_seeds->extent(0)) : 0u;
  if (!options.num_seeds) { options.num_seeds = available_seeds; }
  if (*options.num_seeds > available_seeds) {
    throw std::invalid_argument("--num-seeds exceeds the number of seeds stored in the index");
  }

  std::printf(
    "LOAD_INDEX seconds=%.3f rows=%u dim=%u degree=%u resident_degree=%u cross_degree=%u "
    "resident_bytes=%zu cross_bytes=%zu node_per_cacheline=%u n_bits=%u\n",
    milliseconds(load_index_begin, clock_type::now()) / 1000.0,
    bundle.index.size(),
    bundle.index.dim(),
    bundle.index.graph_degree(),
    bundle.index.resident_degree(),
    bundle.index.cross_degree(),
    bundle.index.inner_graph().size() * sizeof(std::uint8_t),
    bundle.index.cross_graph().size() * sizeof(std::uint32_t),
    bundle.index.node_per_cacheline(),
    bundle.index.n_bits());

  auto const load_input_begin = clock_type::now();
  auto queries                = load_matrix_file<float>(options.queries_path, options.query_count);
  std::unique_ptr<mapped_matrix_file<float>> base;
  if (options.rerank) { base = std::make_unique<mapped_matrix_file<float>>(options.base_path); }
  std::optional<host_matrix_file<std::uint32_t>> truth;
  if (!options.ground_truth_path.empty()) {
    truth.emplace(load_matrix_file<std::uint32_t>(options.ground_truth_path, queries.rows));
  }

  if (queries.cols != bundle.index.dim()) {
    throw std::invalid_argument("query dimension does not match the index");
  }
  if (queries.rows % options.batch_size != 0) {
    throw std::invalid_argument("selected query count is not divisible by --batch-size");
  }
  if (base && (base->rows() != bundle.index.size() || base->cols() != queries.cols)) {
    throw std::invalid_argument("base matrix shape does not match the index");
  }
  if (truth && (truth->rows != queries.rows || truth->cols < options.top_k)) {
    throw std::invalid_argument("ground truth must cover every query and at least --top-k columns");
  }

  std::printf("LOAD_INPUT seconds=%.3f queries=%u dim=%u base_rows=%u gt_width=%u seeds=%u\n",
              milliseconds(load_input_begin, clock_type::now()) / 1000.0,
              queries.rows,
              queries.cols,
              base ? base->rows() : 0,
              truth ? truth->cols : 0,
              available_seeds);
  std::printf(
    "CONFIG gpu=%u batch=%u top_k=%u candidates=%u algorithm=%s team_size=%u itopk=%u "
    "thread_block_size=%u width=%u min_iterations=%u max_iterations=%u "
    "seeds=%u queues=%u empty_pause=%u "
    "cpu_threads=%u session=%s rerank=%s queue_statistics=%s sync_window_scale=%.3f "
    "sync_drop_threshold=%u\n",
    options.gpu_id,
    options.batch_size,
    options.top_k,
    options.candidate_count,
    algorithm_name(options.algorithm),
    options.team_size,
    options.itopk_size,
    options.thread_block_size,
    options.search_width,
    options.min_iterations,
    options.max_iterations,
    *options.num_seeds,
    options.num_queues,
    options.empty_pause,
    options.cpu_threads,
    options.use_search_session ? "on" : "off",
    options.rerank ? "on" : "off",
    options.collect_statistics ? "on" : "off",
    options.sync_window_scale,
    options.sync_drop_threshold);

  flowann::queue_params queue_params{};
  queue_params.num_queues         = options.num_queues;
  queue_params.empty_pause        = options.empty_pause;
  queue_params.collect_statistics = options.collect_statistics;
  flowann::search_context context(bundle.index.cross_graph(), queue_params);
  std::optional<flowann::search_session> session;
  if (options.use_search_session) { session.emplace(context); }

  for (std::uint32_t round = 1; round <= options.warmup_rounds; ++round) {
    auto const result =
      run_round(resources, bundle.index, context, options, base.get(), queries, truth);
    print_round("WARMUP", round, queries.rows, options, result);
  }

  std::vector<double> search_values;
  std::vector<double> prep_values;
  std::vector<double> rerank_values;
  std::vector<double> total_values;
  std::vector<double> recall_values;
  std::vector<double> qps_values;
  for (std::uint32_t round = 1; round <= options.measured_rounds; ++round) {
    auto const result =
      run_round(resources, bundle.index, context, options, base.get(), queries, truth);
    print_round("MEASURE", round, queries.rows, options, result);
    search_values.push_back(result.search_ms);
    prep_values.push_back(result.rerank_prep_ms);
    rerank_values.push_back(result.rerank_ms);
    total_values.push_back(result.total_ms);
    if (truth) { recall_values.push_back(result.recall); }
    qps_values.push_back(result.qps);
  }
  session.reset();

  std::printf(
    "SUMMARY batch=%u search_median_ms=%.6f rerank_prep_median_ms=%.6f "
    "rerank_median_ms=%.6f total_median_ms=%.6f recall_median=%.6f qps_median=%.3f "
    "search_min_ms=%.6f search_max_ms=%.6f\n",
    options.batch_size,
    median(search_values),
    median(prep_values),
    median(rerank_values),
    median(total_values),
    truth ? median(recall_values) : std::numeric_limits<double>::quiet_NaN(),
    median(qps_values),
    *std::min_element(search_values.begin(), search_values.end()),
    *std::max_element(search_values.begin(), search_values.end()));

  auto const statistics = context.statistics();
  std::printf("QUEUE_STATS polls=%llu empty_polls=%llu commands=%llu\n",
              static_cast<unsigned long long>(statistics.polls),
              static_cast<unsigned long long>(statistics.empty_polls),
              static_cast<unsigned long long>(statistics.processed_commands));
  return 0;
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    auto options = parse_options(argc, argv);
    if (!options) { return 0; }

    RAFT_CUDA_TRY(cudaSetDevice(options->gpu_id));
#ifdef _OPENMP
    omp_set_num_threads(static_cast<int>(options->cpu_threads));
#else
    if (options->cpu_threads != 1) {
      throw std::invalid_argument("--cpu-threads requires an OpenMP-enabled build");
    }
#endif
    if (options->pool_size_mib == 0) { return run_benchmark(*options); }

    auto const pool_bytes = options->pool_size_mib * 1024ULL * 1024ULL;
    rmm::mr::managed_memory_resource managed_resource;
    rmm::mr::pool_memory_resource pool_resource(managed_resource, pool_bytes);
    scoped_device_resource resource_scope(pool_resource);
    return run_benchmark(*options);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "FLOWANN_BENCHMARK error: %s\n", error.what());
    std::fprintf(stderr, "Use --help for command-line documentation.\n");
    return 2;
  }
}
