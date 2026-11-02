/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/brute_force.hpp>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/init.cuh>
#include <raft/random/rng.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/util/itertools.hpp>

#include <chrono>
#include <cuda_fp16.h>
#include <iomanip>
#include <iostream>

namespace cuvs::neighbors::brute_force {

struct print_metric {
  cuvs::distance::DistanceType value;
};

struct RandomKNNInputs {
  int num_queries;
  int num_db_vecs;
  int dim;
  int k;
  cuvs::distance::DistanceType metric;
  bool row_major;
};

inline auto operator<<(std::ostream& os, const print_metric& p) -> std::ostream&
{
  switch (p.value) {
    case cuvs::distance::DistanceType::L2Expanded: os << "L2Expanded"; break;
    case cuvs::distance::DistanceType::L2SqrtExpanded: os << "L2SqrtExpanded"; break;
    case cuvs::distance::DistanceType::CosineExpanded: os << "CosineExpanded"; break;
    case cuvs::distance::DistanceType::L1: os << "L1"; break;
    case cuvs::distance::DistanceType::L2Unexpanded: os << "L2Unexpanded"; break;
    case cuvs::distance::DistanceType::L2SqrtUnexpanded: os << "L2SqrtUnexpanded"; break;
    case cuvs::distance::DistanceType::InnerProduct: os << "InnerProduct"; break;
    case cuvs::distance::DistanceType::Linf: os << "Linf"; break;
    case cuvs::distance::DistanceType::Canberra: os << "Canberra"; break;
    case cuvs::distance::DistanceType::LpUnexpanded: os << "LpUnexpanded"; break;
    case cuvs::distance::DistanceType::CorrelationExpanded: os << "CorrelationExpanded"; break;
    case cuvs::distance::DistanceType::JaccardExpanded: os << "JaccardExpanded"; break;
    case cuvs::distance::DistanceType::HellingerExpanded: os << "HellingerExpanded"; break;
    case cuvs::distance::DistanceType::Haversine: os << "Haversine"; break;
    case cuvs::distance::DistanceType::BrayCurtis: os << "BrayCurtis"; break;
    case cuvs::distance::DistanceType::JensenShannon: os << "JensenShannon"; break;
    case cuvs::distance::DistanceType::HammingUnexpanded: os << "HammingUnexpanded"; break;
    case cuvs::distance::DistanceType::KLDivergence: os << "KLDivergence"; break;
    case cuvs::distance::DistanceType::RusselRaoExpanded: os << "RusselRaoExpanded"; break;
    case cuvs::distance::DistanceType::DiceExpanded: os << "DiceExpanded"; break;
    case cuvs::distance::DistanceType::Precomputed: os << "Precomputed"; break;
    default: RAFT_FAIL("unreachable code");
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const RandomKNNInputs& input)
{
  return os << "num_queries:" << input.num_queries << " num_vecs:" << input.num_db_vecs
            << " dim:" << input.dim << " k:" << input.k << " metric:" << print_metric{input.metric}
            << " row_major:" << input.row_major;
}

template <typename T, typename DistT = T>
class BruteForceKNNBenchmark {
 public:
  BruteForceKNNBenchmark(const RandomKNNInputs& params, const std::string& type_str)
    : stream_(raft::resource::get_cuda_stream(handle_).get()),
      params_(params),
      type_str_(type_str),
      database(params_.num_db_vecs * params_.dim, stream_),
      search_queries(params_.num_queries * params_.dim, stream_),
      cuvs_indices_(params_.num_queries * params_.k, stream_),
      cuvs_distances_(params_.num_queries * params_.k, stream_)
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
  }

  void runBenchmark()
  {
    DistT metric_arg = 3.0;
    rmm::device_uvector<char> workspace(0, stream_);

    std::chrono::duration<double, std::milli> build_dur;
    std::chrono::duration<double, std::milli> search_dur;

    auto indices = raft::make_device_matrix_view<int64_t, int64_t, raft::row_major>(
      cuvs_indices_.data(), params_.num_queries, params_.k);
    auto distances = raft::make_device_matrix_view<DistT, int64_t, raft::row_major>(
      cuvs_distances_.data(), params_.num_queries, params_.k);
    raft::resource::sync_stream(handle_, stream_);

    if (params_.row_major) {
      {
        auto idx_warm =
          cuvs::neighbors::brute_force::build(handle_,
                                              raft::make_device_matrix_view<const T, int64_t>(
                                                database.data(), params_.num_db_vecs, params_.dim),
                                              params_.metric,
                                              metric_arg);
        cuvs::neighbors::brute_force::search(
          handle_,
          idx_warm,
          raft::make_device_matrix_view<const T, int64_t>(
            search_queries.data(), params_.num_queries, params_.dim),
          indices,
          distances,
          cuvs::neighbors::filtering::none_sample_filter{});
        flush_l2_cache();
        raft::resource::sync_stream(handle_, stream_);
      }

      auto start = std::chrono::high_resolution_clock::now();
      auto idx =
        cuvs::neighbors::brute_force::build(handle_,
                                            raft::make_device_matrix_view<const T, int64_t>(
                                              database.data(), params_.num_db_vecs, params_.dim),
                                            params_.metric,
                                            metric_arg);
      raft::resource::sync_stream(handle_, stream_);
      auto end  = std::chrono::high_resolution_clock::now();
      build_dur = end - start;

      start = std::chrono::high_resolution_clock::now();
      cuvs::neighbors::brute_force::search(
        handle_,
        idx,
        raft::make_device_matrix_view<const T, int64_t>(
          search_queries.data(), params_.num_queries, params_.dim),
        indices,
        distances,
        cuvs::neighbors::filtering::none_sample_filter{});
      raft::resource::sync_stream(handle_, stream_);
      end        = std::chrono::high_resolution_clock::now();
      search_dur = end - start;

    } else {
      {
        auto idx_warm =
          cuvs::neighbors::brute_force::build(handle_,
                                              raft::make_device_matrix_view<const T, int64_t>(
                                                database.data(), params_.num_db_vecs, params_.dim),
                                              params_.metric,
                                              metric_arg);
        cuvs::neighbors::brute_force::search(
          handle_,
          idx_warm,
          raft::make_device_matrix_view<const T, int64_t, raft::col_major>(
            search_queries.data(), params_.num_queries, params_.dim),
          indices,
          distances,
          cuvs::neighbors::filtering::none_sample_filter{});
        flush_l2_cache();
        raft::resource::sync_stream(handle_, stream_);
      }

      auto start = std::chrono::high_resolution_clock::now();
      auto idx   = cuvs::neighbors::brute_force::build(
        handle_,
        raft::make_device_matrix_view<const T, int64_t, raft::col_major>(
          database.data(), params_.num_db_vecs, params_.dim),
        params_.metric,
        metric_arg);
      raft::resource::sync_stream(handle_, stream_);
      auto end  = std::chrono::high_resolution_clock::now();
      build_dur = end - start;

      start = std::chrono::high_resolution_clock::now();
      cuvs::neighbors::brute_force::search(
        handle_,
        idx,
        raft::make_device_matrix_view<const T, int64_t, raft::col_major>(
          search_queries.data(), params_.num_queries, params_.dim),
        indices,
        distances,
        cuvs::neighbors::filtering::none_sample_filter{});
      raft::resource::sync_stream(handle_, stream_);
      end        = std::chrono::high_resolution_clock::now();
      search_dur = end - start;
    }

    double total_dur  = build_dur.count() + search_dur.count();
    double throughput = static_cast<double>(params_.num_queries) / (total_dur / 1000.0);
    ;
    printResult(params_, build_dur.count(), search_dur.count(), total_dur, throughput);
  }

  void setUp()
  {
    unsigned long long int seed = 1234ULL;
    raft::random::RngState r(seed);

    // JensenShannon distance requires positive values
    T min_val = params_.metric == cuvs::distance::DistanceType::JensenShannon ? T(0.0) : T(-1.0);
    uniform(handle_, r, database.data(), params_.num_db_vecs * params_.dim, min_val, T(1.0));
    uniform(handle_, r, search_queries.data(), params_.num_queries * params_.dim, min_val, T(1.0));
  }

 private:
  void flush_l2_cache()
  {
    int l2_cache_size = 0;
    int device_id     = 0;
    RAFT_CUDA_TRY(cudaGetDevice(&device_id));
    RAFT_CUDA_TRY(cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, device_id));
    scratch_buf_ = rmm::device_buffer(l2_cache_size * 3, stream_);
    RAFT_CUDA_TRY(cudaMemsetAsync(scratch_buf_.data(), 0, scratch_buf_.size(), stream_));
  };

  void printResult(const RandomKNNInputs& params,
                   double build_time,
                   double search_time,
                   double total_time,
                   double throughput)
  {
    std::cout << std::left << std::setw(15) << type_str_ << std::setw(10) << params.num_queries
              << std::setw(10) << params.num_db_vecs << std::setw(10) << params.dim << std::setw(10)
              << params.k << std::setw(20) << print_metric{params.metric} << std::setw(15)
              << (params.row_major ? "row" : "col") << std::right << std::setw(20) << std::fixed
              << std::setprecision(3) << build_time << std::right << std::setw(20) << std::fixed
              << std::setprecision(3) << search_time << std::right << std::setw(20) << std::fixed
              << std::setprecision(3) << total_time << std::right << std::setw(20) << std::fixed
              << std::setprecision(3) << throughput << "\n";
  }
  raft::resources handle_;
  cudaStream_t stream_ = 0;
  RandomKNNInputs params_;
  rmm::device_uvector<T> database;
  rmm::device_uvector<T> search_queries;
  rmm::device_uvector<int64_t> cuvs_indices_;
  rmm::device_uvector<DistT> cuvs_distances_;
  rmm::device_buffer scratch_buf_;
  std::string type_str_;
};

static std::vector<RandomKNNInputs> getInputs()
{
  std::vector<RandomKNNInputs> param_vec;
  struct TestParams {
    int num_queries;
    int num_db_vecs;
    int dim;
    int k;
    cuvs::distance::DistanceType metric;
    bool row_major;
  };

  const std::vector<TestParams> params_group = raft::util::itertools::product<TestParams>(
    {int(10), int(100), int(1024)},
    {int(1000000)},
    {int(32), int(256), int(1024)},
    {int(128), int(1024)},
    {cuvs::distance::DistanceType::InnerProduct, cuvs::distance::DistanceType::L2SqrtExpanded},
    {true, false});

  param_vec.reserve(params_group.size());
  for (TestParams params : params_group) {
    param_vec.push_back(RandomKNNInputs({params.num_queries,
                                         params.num_db_vecs,
                                         params.dim,
                                         params.k,
                                         params.metric,
                                         params.row_major}));
  }
  return param_vec;
}

void printHeader()
{
  std::cout << std::left << std::setw(15) << "Type" << std::setw(10) << "Queries" << std::setw(10)
            << "Vectors" << std::setw(10) << "Dim" << std::setw(10) << "K" << std::setw(20)
            << "Metric" << std::setw(15) << "Layout" << std::right << std::setw(20)
            << "Build Time (ms)" << std::right << std::setw(20) << "Search Time (ms)" << std::right
            << std::setw(20) << "Total Time (ms)" << std::right << std::setw(20)
            << "Throughput (q/s)"
            << "\n";
  std::cout << std::string(165, '-') << "\n";
}

void runBenchmarkForType()
{
  auto selected_inputs = getInputs();
  for (const auto& input : selected_inputs) {
    {
      BruteForceKNNBenchmark<float, float> benchmark(input, "float");
      benchmark.setUp();
      benchmark.runBenchmark();
    }
    {
      BruteForceKNNBenchmark<half, float> benchmark(input, "half");
      benchmark.setUp();
      benchmark.runBenchmark();
    }
  }
}

}  // namespace cuvs::neighbors::brute_force

int main()
{
  cuvs::neighbors::brute_force::printHeader();
  cuvs::neighbors::brute_force::runBenchmarkForType();
  return 0;
}
