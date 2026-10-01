/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/cluster/kmeans.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <argp.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

const char* argp_docs = "balanced_kmeans_example 0.1";

static struct argp_option options[] = {
  {"dataset", 'd', "PATH", 0, "Path to dataset file"},
  {"dtype", 't', "TYPE", 0, "Data type [float/half/int8/uint8]"},
  {"partitions", 'P', "INT", 0, "Number of balanced partitions"},
  {"iterations", 'I', "INT", 0, "Number of k-means iterations (default: 20)"},
  {"balance-lower-tolerance",
   'L',
   "FLOATS",
   0,
   "Comma-separated lower balance tolerances (default: 0.333)"},
  {"balance-upper-tolerance",
   'U',
   "FLOATS",
   0,
   "Comma-separated upper balance tolerances (default: 3.0)"},
  {"centroid-offset", 'O', "FLOAT", 0, "Centroid offset when splitting partitions (default: 0.01)"},
  {0}};

struct arguments {
  std::string dataset_path;
  std::string dtype;
  std::uint32_t n_partitions;
  std::uint32_t n_iters;
  std::vector<float> balance_lower_tolerances;
  std::vector<float> balance_upper_tolerances;
  float centroid_offset;
};

std::vector<float> parse_float_list(std::string const& arg)
{
  std::vector<float> values;
  std::stringstream ss(arg);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      throw std::invalid_argument("Empty value in comma-separated list: " + arg);
    }
    values.push_back(std::stof(token));
  }
  if (values.empty()) { throw std::invalid_argument("Empty comma-separated list"); }
  return values;
}

static error_t parse_opt(int key, char* arg, struct argp_state* state)
{
  struct arguments* arguments = reinterpret_cast<struct arguments*>(state->input);

  switch (key) {
    case 'd': arguments->dataset_path = arg; break;
    case 't': arguments->dtype = arg; break;
    case 'P': arguments->n_partitions = std::stoul(arg); break;
    case 'I': arguments->n_iters = std::stoul(arg); break;
    case 'L': arguments->balance_lower_tolerances = parse_float_list(arg); break;
    case 'U': arguments->balance_upper_tolerances = parse_float_list(arg); break;
    case 'O': arguments->centroid_offset = std::stof(arg); break;
    case ARGP_KEY_ARG: break;
    case ARGP_KEY_END: break;
    default: return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static struct argp argp = {options, parse_opt, nullptr, argp_docs};

namespace {

struct partition_size_stats {
  std::vector<int64_t> sorted_sizes;
  int64_t min_size;
  int64_t max_size;
  int64_t underflow_count;
  int64_t overflow_count;
  double median_size;
  double mean_size;
  double stddev_size;
  double lower_threshold;
  double upper_threshold;
};

enum dataset_file_format_t { XVECS, BIGANN, AUTO_DETECT };

template <class DataT>
struct dataset_descriptor_t {
  std::size_t dim;
  std::size_t size;

  std::unique_ptr<DataT[]> data;
  dataset_file_format_t file_format;
};

template <class DataT>
void get_dataset_info(dataset_descriptor_t<DataT>& desc,
                      std::string const& file_path,
                      dataset_file_format_t file_format = AUTO_DETECT)
{
  std::ifstream ifs(file_path, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("File not exist : " + file_path + " (`" + __func__ + "` in " +
                             __FILE__ + ")");
  }

  ifs.seekg(0, std::ios::end);
  auto const file_size_in_byte = static_cast<std::size_t>(ifs.tellg());
  ifs.seekg(0, std::ios::beg);

  std::uint32_t tmp_val[2];
  ifs.read(reinterpret_cast<char*>(tmp_val), sizeof(std::uint32_t) * 2);

  desc.file_format = file_format;
  if (desc.file_format == AUTO_DETECT) {
    if (sizeof(std::uint32_t) * 2 + sizeof(DataT) * tmp_val[0] * tmp_val[1] == file_size_in_byte) {
      desc.file_format = BIGANN;
    } else {
      desc.file_format = XVECS;
    }
  }

  if (desc.file_format == BIGANN) {
    std::fprintf(stderr, "# BIGANN type file (%s)\n", file_path.c_str());
    desc.size = tmp_val[0];
    desc.dim  = tmp_val[1];
  } else {
    std::fprintf(stderr, "# Xvec type file (%s)\n", file_path.c_str());
    desc.dim = tmp_val[0];
    auto const row_size =
      sizeof(std::uint32_t) + sizeof(DataT) * static_cast<std::size_t>(desc.dim);
    if (row_size == 0 || file_size_in_byte % row_size != 0) {
      throw std::runtime_error("Invalid Xvec file size : " + file_path);
    }
    desc.size = file_size_in_byte / row_size;
  }
}

template <class DataT>
void load_dataset(dataset_descriptor_t<DataT>& desc,
                  std::string const& file_path,
                  dataset_file_format_t file_format = AUTO_DETECT)
{
  get_dataset_info(desc, file_path, file_format);
  std::ifstream ifs(file_path, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("File not exist : " + file_path + " (`" + __func__ + "` in " +
                             __FILE__ + ")");
  }

  auto const array_size = sizeof(DataT) * desc.dim * desc.size;
  desc.data             = std::make_unique<DataT[]>(desc.dim * desc.size);

  if (desc.file_format == BIGANN) {
    ifs.seekg(sizeof(std::uint32_t) * 2, std::ios::beg);
    ifs.read(reinterpret_cast<char*>(desc.data.get()), array_size);
  } else {
    for (std::size_t i = 0; i < desc.size; i++) {
      std::uint32_t row_dim = 0;
      ifs.read(reinterpret_cast<char*>(&row_dim), sizeof(row_dim));
      if (row_dim != desc.dim) {
        throw std::runtime_error("Inconsistent Xvec dimension in : " + file_path);
      }
      ifs.read(reinterpret_cast<char*>(desc.data.get() + i * desc.dim), sizeof(DataT) * desc.dim);
    }
  }
  if (!ifs) { throw std::runtime_error("Failed to read dataset : " + file_path); }
}

template <typename LabelT>
partition_size_stats compute_partition_size_stats(
  raft::device_resources const& resources,
  int64_t n_partitions,
  raft::device_vector_view<const LabelT, int64_t> labels,
  float balance_lower_tolerance,
  float balance_upper_tolerance)
{
  auto host_labels = raft::make_host_vector<LabelT, int64_t>(labels.extent(0));
  auto stream      = raft::resource::get_cuda_stream(resources);

  raft::copy(host_labels.data_handle(), labels.data_handle(), labels.size(), stream);
  raft::resource::sync_stream(resources, stream);

  std::vector<int64_t> partition_sizes(n_partitions, 0);
  for (int64_t row = 0; row < labels.extent(0); ++row) {
    ++partition_sizes.at(static_cast<std::size_t>(host_labels(row)));
  }

  std::sort(partition_sizes.begin(), partition_sizes.end());

  auto minimum = partition_sizes.front();
  auto maximum = partition_sizes.back();
  auto median =
    n_partitions % 2 == 0
      ? (partition_sizes[n_partitions / 2 - 1] + partition_sizes[n_partitions / 2]) / 2.0
      : static_cast<double>(partition_sizes[n_partitions / 2]);
  auto mean            = static_cast<double>(labels.extent(0)) / n_partitions;
  auto lower_threshold = mean * balance_lower_tolerance;
  auto upper_threshold = mean * balance_upper_tolerance;
  auto underflow_count = static_cast<int64_t>(
    std::count_if(partition_sizes.begin(), partition_sizes.end(), [lower_threshold](int64_t size) {
      return size < lower_threshold;
    }));
  auto overflow_count = static_cast<int64_t>(
    std::count_if(partition_sizes.begin(), partition_sizes.end(), [upper_threshold](int64_t size) {
      return size > upper_threshold;
    }));
  auto variance =
    std::accumulate(partition_sizes.begin(),
                    partition_sizes.end(),
                    0.0,
                    [mean](double sum, int64_t size) { return sum + std::pow(size - mean, 2); }) /
    n_partitions;

  return {std::move(partition_sizes),
          minimum,
          maximum,
          underflow_count,
          overflow_count,
          median,
          mean,
          std::sqrt(variance),
          lower_threshold,
          upper_threshold};
}

void print_partition_size_stats(std::string const& label, partition_size_stats const& stats)
{
  std::cout << label << " partition size statistics: min=" << stats.min_size
            << ", max=" << stats.max_size << ", median=" << stats.median_size
            << ", mean=" << stats.mean_size << ", standard deviation=" << stats.stddev_size
            << ", min/mean=" << stats.min_size / stats.mean_size
            << ", max/mean=" << stats.max_size / stats.mean_size
            << ", underflow=" << stats.underflow_count << " (< " << stats.lower_threshold << ")"
            << ", overflow=" << stats.overflow_count << " (> " << stats.upper_threshold << ")"
            << '\n';
}

void print_partition_size_summary(std::string const& label, partition_size_stats const& stats)
{
  std::cout << label << " partition size statistics: min=" << stats.min_size
            << ", max=" << stats.max_size << ", median=" << stats.median_size
            << ", mean=" << stats.mean_size << ", standard deviation=" << stats.stddev_size
            << ", min/mean=" << stats.min_size / stats.mean_size
            << ", max/mean=" << stats.max_size / stats.mean_size << '\n';
}

void print_partition_size_histogram(std::string const& label,
                                    partition_size_stats const& stats,
                                    int64_t histogram_min,
                                    double histogram_upper,
                                    int64_t n_bins = 20)
{
  if (stats.sorted_sizes.empty()) { return; }

  std::vector<int64_t> bins(n_bins + 1, 0);
  auto const range = histogram_upper - histogram_min;
  if (range == 0.0) {
    bins.front() = static_cast<int64_t>(stats.sorted_sizes.size());
  } else {
    for (auto size : stats.sorted_sizes) {
      if (static_cast<double>(size) > histogram_upper) {
        bins.back()++;
      } else {
        auto bin = static_cast<int64_t>((size - histogram_min) / range * n_bins);
        bins[std::min<int64_t>(bin, n_bins - 1)]++;
      }
    }
  }

  auto const max_bin_count = *std::max_element(bins.begin(), bins.end());
  auto const bar_width     = int64_t{40};
  auto const bin_width     = range / n_bins;

  std::cout << label << " partition size histogram:\n";
  for (int64_t bin = 0; bin < n_bins; ++bin) {
    auto const lower =
      range == 0.0 ? static_cast<double>(histogram_min) : histogram_min + bin_width * bin;
    auto const upper = range == 0.0 ? histogram_upper : histogram_min + bin_width * (bin + 1);
    auto const count = bins[bin];
    auto const hashes =
      max_bin_count == 0 ? int64_t{0} : std::max<int64_t>(1, count * bar_width / max_bin_count);

    std::cout << "  [" << std::setw(8) << static_cast<int64_t>(std::floor(lower)) << ", "
              << std::setw(8) << static_cast<int64_t>(std::ceil(upper)) << "] " << std::setw(4)
              << count << " | ";
    for (int64_t i = 0; i < hashes && count != 0; ++i) {
      std::cout << '#';
    }
    std::cout << '\n';
  }

  auto const overflow_count  = bins.back();
  auto const overflow_hashes = max_bin_count == 0
                                 ? int64_t{0}
                                 : std::max<int64_t>(1, overflow_count * bar_width / max_bin_count);
  std::cout << "  (" << std::setw(8) << static_cast<int64_t>(std::ceil(histogram_upper)) << ", "
            << std::setw(8) << "inf"
            << "] " << std::setw(4) << overflow_count << " | ";
  for (int64_t i = 0; i < overflow_hashes && overflow_count != 0; ++i) {
    std::cout << '#';
  }
  std::cout << '\n';
}

void print_balance_improvement(partition_size_stats const& regular_stats,
                               partition_size_stats const& balanced_stats)
{
  auto const regular_max_ratio  = regular_stats.max_size / regular_stats.mean_size;
  auto const balanced_max_ratio = balanced_stats.max_size / balanced_stats.mean_size;
  auto const regular_stddev     = regular_stats.stddev_size;
  auto const balanced_stddev    = balanced_stats.stddev_size;

  std::cout << "Balance improvement: max/mean " << regular_max_ratio << " -> " << balanced_max_ratio
            << ", standard deviation " << regular_stddev << " -> " << balanced_stddev << '\n';
}

void print_inertia_comparison(float regular_inertia, float balanced_inertia)
{
  std::cout << "Inertia: regular=" << regular_inertia << ", balanced=" << balanced_inertia;
  if (regular_inertia != 0.0f) {
    std::cout << ", balanced/regular=" << balanced_inertia / regular_inertia;
  }
  std::cout << '\n';
}

template <typename DataT>
bool run_regular_kmeans(raft::device_resources const& resources,
                        raft::device_matrix_view<const DataT, int64_t> dataset,
                        int64_t n_partitions,
                        std::uint32_t n_iters,
                        raft::device_vector_view<int64_t, int64_t> labels,
                        float* inertia_out)
{
  if constexpr (std::is_same_v<DataT, float>) {
    cuvs::cluster::kmeans::params params;
    params.metric     = cuvs::distance::DistanceType::L2Expanded;
    params.n_clusters = static_cast<int>(n_partitions);
    params.max_iter   = static_cast<int>(n_iters);

    auto centroids = raft::make_device_matrix<float, int64_t>(
      resources, n_partitions, static_cast<int64_t>(dataset.extent(1)));

    float inertia  = 0.0f;
    int64_t n_iter = 0;
    cuvs::cluster::kmeans::fit(resources,
                               params,
                               dataset,
                               std::nullopt,
                               centroids.view(),
                               raft::make_host_scalar_view(&inertia),
                               raft::make_host_scalar_view(&n_iter));
    cuvs::cluster::kmeans::predict(resources,
                                   params,
                                   dataset,
                                   std::nullopt,
                                   raft::make_const_mdspan(centroids.view()),
                                   labels,
                                   false,
                                   raft::make_host_scalar_view(&inertia));

    if (inertia_out != nullptr) { *inertia_out = inertia; }
    return true;
  } else {
    return false;
  }
}

template <typename DataT>
void partition_dataset(std::string const& dataset_path,
                       std::uint32_t n_partitions,
                       std::uint32_t n_iters,
                       std::vector<float> const& balance_lower_tolerances,
                       std::vector<float> const& balance_upper_tolerances,
                       float centroid_offset)
{
  raft::device_resources resources;

  dataset_descriptor_t<DataT> dataset_desc;
  load_dataset<DataT>(dataset_desc, dataset_path);

  auto n_samples  = static_cast<int64_t>(dataset_desc.size);
  auto n_features = static_cast<int64_t>(dataset_desc.dim);
  if (n_partitions > dataset_desc.size) {
    throw std::invalid_argument("Number of partitions cannot exceed the number of vectors");
  }

  auto dataset = raft::make_device_matrix<DataT, int64_t>(resources, n_samples, n_features);
  auto stream  = raft::resource::get_cuda_stream(resources);
  raft::copy(dataset.data_handle(), dataset_desc.data.get(), dataset.size(), stream);
  raft::resource::sync_stream(resources, stream);
  dataset_desc.data.reset();

  std::cout << "Partitioning " << n_samples << " vectors with " << n_features << " dimensions into "
            << n_partitions << " balanced partitions\n";

  auto centroids = raft::make_device_matrix<float, int64_t>(resources, n_partitions, n_features);
  auto labels    = raft::make_device_vector<uint32_t, int64_t>(resources, n_samples);
  auto regular_labels = raft::make_device_vector<int64_t, int64_t>(resources, n_samples);
  auto dataset_view   = raft::make_const_mdspan(dataset.view());

  float regular_inertia        = 0.0f;
  auto const has_regular_stats = run_regular_kmeans<DataT>(
    resources, dataset_view, n_partitions, n_iters, regular_labels.view(), &regular_inertia);
  std::optional<partition_size_stats> regular_reference_stats;
  if (has_regular_stats) {
    regular_reference_stats =
      compute_partition_size_stats(resources,
                                   n_partitions,
                                   raft::make_const_mdspan(regular_labels.view()),
                                   balance_lower_tolerances.front(),
                                   balance_upper_tolerances.front());
    print_partition_size_summary("Regular k-means", regular_reference_stats.value());
    std::cout << "Regular k-means inertia: " << regular_inertia << '\n';
    print_partition_size_histogram(
      "Regular k-means",
      regular_reference_stats.value(),
      regular_reference_stats->min_size,
      regular_reference_stats->mean_size + 2.0 * regular_reference_stats->stddev_size);
  } else {
    std::cout << "Regular k-means comparison is only shown for float input in this example.\n";
  }

  for (auto balance_lower_tolerance : balance_lower_tolerances) {
    for (auto balance_upper_tolerance : balance_upper_tolerances) {
      std::cout << "\n# balance_lower_tolerance: " << balance_lower_tolerance << '\n'
                << "# balance_upper_tolerance: " << balance_upper_tolerance << '\n';

      cuvs::cluster::kmeans::balanced_params params;
      params.metric                  = cuvs::distance::DistanceType::L2Expanded;
      params.n_iters                 = n_iters;
      params.balance_lower_tolerance = balance_lower_tolerance;
      params.balance_upper_tolerance = balance_upper_tolerance;
      params.centroid_offset         = centroid_offset;

      float balanced_inertia = 0.0f;
      if constexpr (std::is_same_v<DataT, float>) {
        cuvs::cluster::kmeans::fit(resources,
                                   params,
                                   dataset_view,
                                   centroids.view(),
                                   raft::make_host_scalar_view(&balanced_inertia));
      } else {
        cuvs::cluster::kmeans::fit(resources, params, dataset_view, centroids.view());
      }
      cuvs::cluster::kmeans::predict(
        resources, params, dataset_view, raft::make_const_mdspan(centroids.view()), labels.view());

      auto balanced_stats = compute_partition_size_stats(resources,
                                                         n_partitions,
                                                         raft::make_const_mdspan(labels.view()),
                                                         balance_lower_tolerance,
                                                         balance_upper_tolerance);

      if (has_regular_stats) {
        auto const& regular_stats = regular_reference_stats.value();
        auto const histogram_min  = std::min(regular_stats.min_size, balanced_stats.min_size);
        auto const histogram_upper =
          std::max(regular_stats.mean_size + 2.0 * regular_stats.stddev_size,
                   balanced_stats.mean_size + 2.0 * balanced_stats.stddev_size);
        print_partition_size_stats("Balanced k-means", balanced_stats);
        print_partition_size_histogram(
          "Balanced k-means", balanced_stats, histogram_min, histogram_upper);
        print_balance_improvement(regular_stats, balanced_stats);
        print_inertia_comparison(regular_inertia, balanced_inertia);
      } else {
        print_partition_size_stats("Balanced k-means", balanced_stats);
        print_partition_size_histogram("Balanced k-means",
                                       balanced_stats,
                                       balanced_stats.min_size,
                                       balanced_stats.mean_size + 2.0 * balanced_stats.stddev_size);
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    struct arguments args = {
      "",       /* dataset_path */
      "",       /* dtype */
      0,        /* n_partitions */
      20,       /* n_iters */
      {0.333f}, /* balance_lower_tolerances */
      {3.0f},   /* balance_upper_tolerances */
      0.01f,    /* centroid_offset */
    };

    argp_parse(&argp, argc, argv, 0, 0, &args);

    std::string error_message;
    if (args.dataset_path.empty()) {
      error_message += "- Path to dataset file has not been provided (-d)\n";
    }
    if (args.dtype.empty()) { error_message += "- Data type has not been provided (-t)\n"; }
    if (args.n_partitions == 0) {
      error_message += "- Number of partitions must be larger than 0 (-P)\n";
    }
    if (args.n_iters == 0) {
      error_message += "- Number of k-means iterations must be larger than 0 (-I)\n";
    }
    for (auto balance_lower_tolerance : args.balance_lower_tolerances) {
      if (!std::isfinite(balance_lower_tolerance) || balance_lower_tolerance <= 0.0f ||
          balance_lower_tolerance >= 1.0f) {
        error_message += "- Lower balance tolerances must be in the range (0, 1) (-L)\n";
        break;
      }
    }
    for (auto balance_upper_tolerance : args.balance_upper_tolerances) {
      if (!std::isfinite(balance_upper_tolerance) || balance_upper_tolerance <= 1.0f) {
        error_message += "- Upper balance tolerances must be greater than 1 (-U)\n";
        break;
      }
    }
    if (!std::isfinite(args.centroid_offset) || args.centroid_offset <= 0.0f ||
        args.centroid_offset > 1.0f) {
      error_message += "- Centroid offset must be in the range (0, 1] (-O)\n";
    }
    if (!error_message.empty()) { throw std::invalid_argument(error_message); }

    std::cout << "# dataset_path: " << args.dataset_path << '\n'
              << "# dtype: " << args.dtype << '\n'
              << "# partitions: " << args.n_partitions << '\n'
              << "# iterations: " << args.n_iters << '\n'
              << "# balance_lower_tolerances:";
    for (auto value : args.balance_lower_tolerances) {
      std::cout << ' ' << value;
    }
    std::cout << '\n' << "# balance_upper_tolerances:";
    for (auto value : args.balance_upper_tolerances) {
      std::cout << ' ' << value;
    }
    std::cout << '\n' << "# centroid_offset: " << args.centroid_offset << '\n';

    if (args.dtype == "float") {
      partition_dataset<float>(args.dataset_path,
                               args.n_partitions,
                               args.n_iters,
                               args.balance_lower_tolerances,
                               args.balance_upper_tolerances,
                               args.centroid_offset);
    } else if (args.dtype == "half") {
      partition_dataset<half>(args.dataset_path,
                              args.n_partitions,
                              args.n_iters,
                              args.balance_lower_tolerances,
                              args.balance_upper_tolerances,
                              args.centroid_offset);
    } else if (args.dtype == "int8") {
      partition_dataset<std::int8_t>(args.dataset_path,
                                     args.n_partitions,
                                     args.n_iters,
                                     args.balance_lower_tolerances,
                                     args.balance_upper_tolerances,
                                     args.centroid_offset);
    } else if (args.dtype == "uint8") {
      partition_dataset<std::uint8_t>(args.dataset_path,
                                      args.n_partitions,
                                      args.n_iters,
                                      args.balance_lower_tolerances,
                                      args.balance_upper_tolerances,
                                      args.centroid_offset);
    } else {
      throw std::invalid_argument("Unknown data type: " + args.dtype);
    }
  } catch (std::exception const& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
