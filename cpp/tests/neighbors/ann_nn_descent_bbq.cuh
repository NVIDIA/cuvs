/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ann_nn_descent.cuh"

#include <cuvs/preprocessing/quantize/bbq.hpp>
#include <cuvs_internal/preprocessing/bbq_cpu_quantize.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>

#include <gtest/gtest.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <optional>
#include <sstream>
#include <vector>

#include <raft/core/logger.hpp>

namespace cuvs::neighbors::nn_descent {
// The host reference quantizer is shared with the ann-bench CAGRA wrapper, so these tests and
// the benchmark can never disagree about the code format.
namespace cpu_bbq = cuvs_internal::bbq;
using cuvs::preprocessing::quantize::bbq::get_bit_width;
using cuvs_internal::bbq::make_device_bbq_dataset;

struct AnnNNDescentBbqInputs : AnnNNDescentInputs {
  cuvs::preprocessing::quantize::bbq::bbq_code_layout layout;
  std::optional<cuvs::preprocessing::quantize::bbq::bbq_code_layout> second_dataset_layout;
};

inline ::std::ostream& operator<<(::std::ostream& os, const AnnNNDescentBbqInputs& p)
{
  os << "dataset shape=" << p.n_rows << "x" << p.dim << ", graph_degree=" << p.graph_degree
     << ", metric="
     << cuvs::neighbors::print_metric{static_cast<cuvs::distance::DistanceType>((int)p.metric)}
     << (p.host_dataset ? ", host" : ", device") << ", layout=" << static_cast<int>(p.layout)
     << ", second_dataset_layout="
     << (p.second_dataset_layout.has_value() ? static_cast<int>(p.second_dataset_layout.value())
                                             : -1)
     << std::endl;
  return os;
}

class AnnNNDescentBbqTest : public ::testing::TestWithParam<AnnNNDescentBbqInputs> {
 public:
  AnnNNDescentBbqTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnNNDescentBbqInputs>::GetParam()),
      database(raft::make_device_matrix<float, int64_t>(handle_, ps.n_rows, ps.dim))
  {
  }

 protected:
  void testNNDescent()
  {
    // nn-descent rejects packed_4b below sm_75.
    if (ps.layout == cuvs::preprocessing::quantize::bbq::bbq_code_layout::packed_4b &&
        cuvs::neighbors::device_compute_capability() < 75) {
      GTEST_SKIP() << "packed_4b requires int4 tensor cores (compute capability 7.5 or newer)";
    }
    if (ps.second_dataset_layout.has_value()) {
      // The document must be strictly coarser than the query; the pair's layouts are stated in
      // the spec, so validity of the layout combination is the spec's business, not inferred here.
      const uint32_t query_bits  = get_bit_width(ps.layout);
      const uint32_t second_bits = get_bit_width(ps.second_dataset_layout.value());
      if (query_bits > 4 || query_bits == second_bits || query_bits == 1) {
        GTEST_SKIP() << "Second dataset is N/A: bits=" << query_bits
                     << ", layout=" << static_cast<int>(ps.layout)
                     << " and second bits=" << second_bits;
      }
    }
    size_t queries_size = ps.n_rows * ps.graph_degree;
    std::vector<uint32_t> indices_NNDescent(queries_size);
    std::vector<float> distances_NNDescent(queries_size);
    std::vector<uint32_t> indices_naive(queries_size);
    std::vector<float> distances_naive(queries_size);

    {
      rmm::device_uvector<float> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<uint32_t> indices_naive_dev(queries_size, stream_);
      naive_knn<float, float, uint32_t>(handle_,
                                        distances_naive_dev.data(),
                                        indices_naive_dev.data(),
                                        database.data_handle(),
                                        database.data_handle(),
                                        ps.n_rows,
                                        ps.n_rows,
                                        ps.dim,
                                        ps.graph_degree,
                                        ps.metric);
      raft::update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      raft::update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      raft::resource::sync_stream(handle_);
    }

    {
      std::vector<float> host_data(static_cast<size_t>(ps.n_rows) * ps.dim);
      raft::update_host(host_data.data(), database.data_handle(), host_data.size(), stream_);
      raft::resource::sync_stream(handle_);

      auto owning_dataset =
        cuvs_internal::bbq::quantize_to_device(handle_,
                                               host_data.data(),
                                               ps.n_rows,
                                               ps.dim,
                                               ps.metric,
                                               ps.layout,
                                               ps.second_dataset_layout.value_or(ps.layout));
      auto dataset = owning_dataset.as_dataset_view();
      nn_descent::index_params index_params;
      index_params.metric                    = ps.metric;
      index_params.graph_degree              = ps.graph_degree;
      index_params.intermediate_graph_degree = 2 * ps.graph_degree;
      index_params.max_iterations            = 100;
      index_params.return_distances          = true;

      auto index = nn_descent::build(handle_, index_params, dataset);

      raft::copy(indices_NNDescent.data(), index.graph().data_handle(), queries_size, stream_);
      ASSERT_TRUE(index.distances().has_value());
      raft::copy(
        distances_NNDescent.data(), index.distances().value().data_handle(), queries_size, stream_);
      raft::resource::sync_stream(handle_);
    }

    EXPECT_TRUE(eval_neighbours(indices_naive,
                                indices_NNDescent,
                                distances_naive,
                                distances_NNDescent,
                                ps.n_rows,
                                ps.graph_degree,
                                0.001,
                                ps.min_recall));
  }

  void SetUp() override
  {
    raft::random::RngState r(1234ULL);
    raft::random::normal(handle_, r, database.data_handle(), ps.n_rows * ps.dim, 0.1f, 2.0f);
    raft::resource::sync_stream(handle_);
  }

  void TearDown() override { raft::resource::sync_stream(handle_); }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnNNDescentBbqInputs ps;
  raft::device_matrix<float, int64_t> database;
};

// Estimated recall based on bruteforce (InnerProduct): 1: 0.23, 2: 0.52, 4: 0.85, 7: 0.98, 8: 0.99.
const std::vector<AnnNNDescentBbqInputs> bbq_inputs = [] {
  using cuvs::preprocessing::quantize::bbq::bbq_code_layout;
  const std::vector<std::tuple<double, bbq_code_layout, std::optional<bbq_code_layout>>>
    code_specifications{
      // min_recall, query layout, document layout (both widths follow from the layouts)
      {0.15, bbq_code_layout::packed_1b, std::optional<bbq_code_layout>{}},
      {0.50, bbq_code_layout::transposed_2b, std::optional<bbq_code_layout>{}},
      {0.27,
       bbq_code_layout::transposed_2b,
       std::optional<bbq_code_layout>{bbq_code_layout::packed_1b}},
      {0.80, bbq_code_layout::packed_4b, std::optional<bbq_code_layout>{}},
      // Asymmetric packed_4b queries take the int4 wmma path (SelfJoin = false). packed_1b is the
      // only document layout that promotes to it -- transposed_2b never reaches this kernel. At
      // dim=256 this is also the only coverage of the phase-2 staging skip (n_tiles == 1) outside
      // a self-join.
      {0.35,
       bbq_code_layout::packed_4b,
       std::optional<bbq_code_layout>{bbq_code_layout::packed_1b}},
      // Asymmetric transposed_4b queries (1 + 4t, 2t + 4t) take the SIMT path.
      {0.35,
       bbq_code_layout::transposed_4b,
       std::optional<bbq_code_layout>{bbq_code_layout::packed_1b}},
      {0.65,
       bbq_code_layout::transposed_4b,
       std::optional<bbq_code_layout>{bbq_code_layout::transposed_2b}},
      {0.80, bbq_code_layout::packed_7b, std::optional<bbq_code_layout>{}},
      {0.80, bbq_code_layout::packed_8b, std::optional<bbq_code_layout>{}}};
  std::vector<AnnNNDescentBbqInputs> out;
  for (const auto& [min_recall, layout, second_layout] : code_specifications) {
    const auto batch = raft::util::itertools::product<AnnNNDescentBbqInputs>(
      {2000},
      {256},  // dim
      {64},   // graph_degree
      {cuvs::distance::DistanceType::L2Expanded,
       cuvs::distance::DistanceType::L2SqrtExpanded,
       cuvs::distance::DistanceType::InnerProduct,
       cuvs::distance::DistanceType::CosineExpanded},
      {false},  // host_dataset
      {min_recall},
      {layout},
      {second_layout});
    out.insert(out.end(), batch.begin(), batch.end());
  }
  return out;
}();

}  // namespace cuvs::neighbors::nn_descent
