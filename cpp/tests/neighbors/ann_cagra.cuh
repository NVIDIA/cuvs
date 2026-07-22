/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../test_utils.cuh"
#include "ann_utils.cuh"
#include "vpq_utils.cuh"
#include <raft/core/resource/cuda_stream.hpp>

#include "naive_knn.cuh"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/composite/index.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/normalize.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/itertools.hpp>

#include <rmm/device_buffer.hpp>

#include <gtest/gtest.h>

#include <thrust/sequence.h>

#include <cstddef>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace cuvs::neighbors::cagra {
namespace {

struct test_cagra_sample_filter {
  static constexpr unsigned offset = 300;
  inline _RAFT_HOST_DEVICE auto operator()(
    // query index
    const uint32_t query_ix,
    // the index of the current sample inside the current inverted list
    const uint32_t sample_ix) const
  {
    return sample_ix >= offset;
  }
};

/** Xorshift rondem number generator.
 *
 * See https://en.wikipedia.org/wiki/Xorshift#xorshift for reference.
 */
_RAFT_HOST_DEVICE inline uint64_t xorshift64(uint64_t u)
{
  u ^= u >> 12;
  u ^= u << 25;
  u ^= u >> 27;
  return u * 0x2545F4914F6CDD1DULL;
}

// For sort_knn_graph test
template <typename IdxT>
void RandomSuffle(raft::host_matrix_view<IdxT, int64_t> index)
{
  for (IdxT i = 0; i < index.extent(0); i++) {
    uint64_t rand       = i;
    IdxT* const row_ptr = index.data_handle() + i * index.extent(1);
    for (unsigned j = 0; j < index.extent(1); j++) {
      // Swap two indices at random
      rand          = xorshift64(rand);
      const auto i0 = rand % index.extent(1);
      rand          = xorshift64(rand);
      const auto i1 = rand % index.extent(1);

      const auto tmp = row_ptr[i0];
      row_ptr[i0]    = row_ptr[i1];
      row_ptr[i1]    = tmp;
    }
  }
}

template <typename DistanceT, typename DatatT, typename IdxT>
testing::AssertionResult CheckOrder(raft::host_matrix_view<IdxT, int64_t> index_test,
                                    raft::host_matrix_view<DatatT, int64_t> dataset)
{
  for (IdxT i = 0; i < index_test.extent(0); i++) {
    const DatatT* const base_vec = dataset.data_handle() + i * dataset.extent(1);
    const IdxT* const index_row  = index_test.data_handle() + i * index_test.extent(1);
    DistanceT prev_distance      = 0;
    for (unsigned j = 0; j < index_test.extent(1) - 1; j++) {
      const DatatT* const target_vec = dataset.data_handle() + index_row[j] * dataset.extent(1);
      DistanceT distance             = 0;
      for (unsigned l = 0; l < dataset.extent(1); l++) {
        const auto diff =
          static_cast<DistanceT>(target_vec[l]) - static_cast<DistanceT>(base_vec[l]);
        distance += diff * diff;
      }
      if (prev_distance > distance) {
        return testing::AssertionFailure()
               << "Wrong index order (row = " << i << ", neighbor_id = " << j
               << "). (distance[neighbor_id-1] = " << prev_distance
               << "should be larger than distance[neighbor_id] = " << distance << ")";
      }
      prev_distance = distance;
    }
  }
  return testing::AssertionSuccess();
}

template <typename T>
struct fpi_mapper {};

template <>
struct fpi_mapper<double> {
  using type                         = int64_t;
  static constexpr int kBitshiftBase = 53;
};

template <>
struct fpi_mapper<float> {
  using type                         = int32_t;
  static constexpr int kBitshiftBase = 24;
};

template <>
struct fpi_mapper<half> {
  using type                         = int16_t;
  static constexpr int kBitshiftBase = 11;
};

// Generate dataset to ensure no rounding error occurs in the norm computation of any two vectors.
// When testing the CAGRA index sorting function, rounding errors can affect the norm and alter the
// order of the index. To ensure the accuracy of the test, we utilize the dataset. The generation
// method is based on the error-free transformation (EFT) method.
template <typename T>
RAFT_KERNEL GenerateRoundingErrorFreeDataset_kernel(T* const ptr,
                                                    const uint32_t size,
                                                    const typename fpi_mapper<T>::type resolution)
{
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) { return; }

  const float u32 = *reinterpret_cast<const typename fpi_mapper<T>::type*>(ptr + tid);
  ptr[tid]        = u32 / resolution;
}

template <typename T>
void GenerateRoundingErrorFreeDataset(
  const raft::resources& handle,
  T* const ptr,
  const uint32_t n_row,
  const uint32_t dim,
  raft::random::RngState& rng,
  const bool diff_flag  // true if compute the norm between two vectors
)
{
  using mapper_type         = fpi_mapper<T>;
  using int_type            = typename mapper_type::type;
  auto cuda_stream          = raft::resource::get_cuda_stream(handle);
  const uint32_t size       = n_row * dim;
  const uint32_t block_size = 256;
  const uint32_t grid_size  = (size + block_size - 1) / block_size;

  const auto bitshift = (mapper_type::kBitshiftBase - std::log2(dim) - (diff_flag ? 1 : 0)) / 2;
  // Skip the test when `dim` is too big for type `T` to allow rounding error-free test.
  if (bitshift <= 1) { GTEST_SKIP(); }
  const int_type resolution = int_type{1} << static_cast<unsigned>(std::floor(bitshift));
  raft::random::uniformInt<int_type>(
    handle, rng, reinterpret_cast<int_type*>(ptr), size, -resolution, resolution - 1);

  GenerateRoundingErrorFreeDataset_kernel<T>
    <<<grid_size, block_size, 0, cuda_stream>>>(ptr, size, resolution);
}

template <class DataT>
void InitDataset(const raft::resources& handle,
                 DataT* const datatset_ptr,
                 std::uint32_t size,
                 std::uint32_t dim,
                 distance::DistanceType metric,
                 raft::random::RngState& r)
{
  if constexpr (std::is_same_v<DataT, float> || std::is_same_v<DataT, half>) {
    GenerateRoundingErrorFreeDataset(handle, datatset_ptr, size, dim, r, true);

    if (metric == cuvs::distance::DistanceType::InnerProduct) {
      auto dataset_view = raft::make_device_matrix_view(datatset_ptr, size, dim);
      raft::linalg::row_normalize<raft::linalg::L2Norm>(
        handle, raft::make_const_mdspan(dataset_view), dataset_view);
    }
  } else if constexpr (std::is_same_v<DataT, std::uint8_t> || std::is_same_v<DataT, std::int8_t>) {
    if constexpr (std::is_same_v<DataT, std::int8_t>) {
      raft::random::uniformInt(handle, r, datatset_ptr, size * dim, DataT(-10), DataT(10));
    } else {
      raft::random::uniformInt(handle, r, datatset_ptr, size * dim, DataT(1), DataT(20));
    }

    if (metric == cuvs::distance::DistanceType::InnerProduct) {
      // TODO (enp1s0): Change this once row_normalize supports (u)int8 matrices.
      // https://github.com/rapidsai/raft/issues/2291

      using ComputeT    = float;
      auto dataset_view = raft::make_device_matrix_view(datatset_ptr, size, dim);
      auto dev_row_norm = raft::make_device_vector<ComputeT>(handle, size);
      const auto normalized_norm =
        (std::is_same_v<DataT, std::uint8_t> ? 40 : 20) * std::sqrt(static_cast<ComputeT>(dim));

      raft::linalg::reduce<true, true>(dev_row_norm.data_handle(),
                                       datatset_ptr,
                                       dim,
                                       size,
                                       0.f,
                                       raft::resource::get_cuda_stream(handle),
                                       false,
                                       raft::sq_op(),
                                       raft::add_op(),
                                       raft::sqrt_op());
      raft::linalg::matrix_vector_op<raft::Apply::ALONG_COLUMNS>(
        handle,
        raft::make_const_mdspan(dataset_view),
        raft::make_const_mdspan(dev_row_norm.view()),
        dataset_view,
        [normalized_norm] __device__(DataT elm, ComputeT norm) {
          const ComputeT v           = elm / norm * normalized_norm;
          const ComputeT max_v_range = std::numeric_limits<DataT>::max();
          const ComputeT min_v_range = std::numeric_limits<DataT>::min();
          return static_cast<DataT>(std::min(max_v_range, std::max(min_v_range, v)));
        });
    }
  }
}

enum class graph_build_algo {
  /* Use IVF-PQ to build all-neighbors knn graph */
  IVF_PQ,
  /* Experimental, use NN-Descent to build all-neighbors knn graph */
  NN_DESCENT,
  /* Experimental, iteratively execute CAGRA's search() and optimize() */
  ITERATIVE_CAGRA_SEARCH,
  /* Choose default automatically */
  AUTO
};

}  // namespace

struct AnnCagraInputs {
  int n_queries;
  int n_rows;
  int dim;
  int k;
  int graph_degree;
  graph_build_algo build_algo;
  search_algo algo;
  int max_queries;
  int team_size;
  int itopk_size;
  int search_width;
  cuvs::distance::DistanceType metric;
  bool host_dataset;
  bool include_serialized_dataset;
  bool use_source_indices;
  // std::optional<double>
  double min_recall;  // = std::nullopt;
  std::optional<float> ivf_pq_search_refine_ratio = std::nullopt;
  std::optional<vpq_params> compression           = std::nullopt;

  std::optional<bool> non_owning_memory_buffer_flag = std::nullopt;
  cuvs::neighbors::MergeStrategy merge_strategy =
    cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_PHYSICAL;
  cuvs::neighbors::cagra::internal_dtype smem_dtype = cuvs::neighbors::cagra::internal_dtype::F16;
  /** When set, physical merge uses this overload instead of the default-params merge. */
  std::optional<cagra::merge_params> physical_merge_params = std::nullopt;
};

inline ::std::ostream& operator<<(::std::ostream& os, const AnnCagraInputs& p)
{
  const auto metric_str = [](const cuvs::distance::DistanceType dist) -> std::string {
    switch (dist) {
      case cuvs::distance::DistanceType::InnerProduct: return "InnerProduct";
      case cuvs::distance::DistanceType::L2Expanded: return "L2";
      case cuvs::distance::DistanceType::BitwiseHamming: return "BitwiseHamming";
      case cuvs::distance::DistanceType::CosineExpanded: return "Cosine";
      case cuvs::distance::DistanceType::L1: return "L1";
      default: break;
    }
    return "Unknown";
  };

  std::map<search_algo, std::string> algo_name = {
    {search_algo::SINGLE_CTA, "single-cta"},      //
    {search_algo::MULTI_CTA, "multi_cta"},        //
    {search_algo::MULTI_KERNEL, "multi_kernel"},  //
    {search_algo::AUTO, "auto"}                   //
  };
  std::vector<std::string> build_algo = {"IVF_PQ", "NN_DESCENT", "ITERATIVE_CAGRA_SEARCH", "AUTO"};
  const auto smem_dtype_str           = [](cuvs::neighbors::cagra::internal_dtype dtype) {
    switch (dtype) {
      case cuvs::neighbors::cagra::internal_dtype::F16: return "F16";
      case cuvs::neighbors::cagra::internal_dtype::E5M2: return "E5M2";
    }
    return "Unknown";
  };
  std::vector<std::string> merge_strategy = {"PHYSICAL", "LOGICAL"};
  os << "{n_queries=" << p.n_queries << ", dataset shape=" << p.n_rows << "x" << p.dim
     << ", degree=" << p.graph_degree << ", k=" << p.k << ", " << algo_name[p.algo]
     << ", max_queries=" << p.max_queries << ", itopk_size=" << p.itopk_size
     << ", search_width=" << p.search_width << ", metric=" << metric_str(p.metric) << ", "
     << (p.host_dataset ? "host" : "device") << ", build_algo=" << build_algo.at((int)p.build_algo)
     << ", merge_logic=" << merge_strategy.at((int)p.merge_strategy);
  if ((int)p.build_algo == 0 && p.ivf_pq_search_refine_ratio) {
    os << "(refine_rate=" << *p.ivf_pq_search_refine_ratio << ')';
  }
  if (p.compression.has_value()) {
    auto vpq = p.compression.value();
    os << ", pq_bits=" << vpq.pq_bits << ", pq_dim=" << vpq.pq_dim
       << ", vq_n_centers=" << vpq.vq_n_centers << ", smem_dtype=" << smem_dtype_str(p.smem_dtype);
  }
  os << '}';
  return os;
}

template <typename DistanceT, typename DataT, typename IdxT>
class AnnCagraTest : public ::testing::TestWithParam<AnnCagraInputs> {
 public:
  AnnCagraTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnCagraInputs>::GetParam()),
      database(0, stream_),
      search_queries(0, stream_)
  {
  }

 protected:
  template <typename SearchIdxT = IdxT>
  void testCagra()
  {
    // IVF_PQ graph build does not support BitwiseHamming
    if (ps.metric == cuvs::distance::DistanceType::BitwiseHamming &&
        ((!std::is_same_v<DataT, uint8_t>) || (ps.build_algo == graph_build_algo::IVF_PQ)))
      GTEST_SKIP();
    // If the dataset dimension is small and the dataset size is large, there can be a lot of
    // dataset vectors that have the same distance to the query, especially in the binary Hamming
    // distance, making it impossible to make a top-k ground truth.
    if (ps.metric == cuvs::distance::DistanceType::BitwiseHamming &&
        (ps.k * ps.dim * 8 / 5 /*(=magic number)*/ < ps.n_rows))
      GTEST_SKIP();
    if (ps.metric == cuvs::distance::DistanceType::L1 &&
        ps.build_algo != graph_build_algo::ITERATIVE_CAGRA_SEARCH)
      GTEST_SKIP();
    if (ps.metric == cuvs::distance::DistanceType::CosineExpanded) {
      if (ps.compression.has_value()) { GTEST_SKIP(); }
      if (ps.build_algo == graph_build_algo::ITERATIVE_CAGRA_SEARCH || ps.dim == 1) {
        GTEST_SKIP();
      }
    }
    // Dataset size must be larger than 33 for NN Descent
    if (ps.build_algo == graph_build_algo::NN_DESCENT &&
        ((ps.n_rows < 33) || raft::round_up_safe(ps.n_rows, 32) <= ps.graph_degree * 3))
      GTEST_SKIP();

    size_t queries_size = ps.n_queries * ps.k;
    std::vector<SearchIdxT> indices_Cagra(queries_size);
    std::vector<SearchIdxT> indices_naive(queries_size);
    std::vector<DistanceT> distances_Cagra(queries_size);
    std::vector<DistanceT> distances_naive(queries_size);

    {
      rmm::device_uvector<DistanceT> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<SearchIdxT> indices_naive_dev(queries_size, stream_);

      cuvs::neighbors::naive_knn<DistanceT, DataT, SearchIdxT>(handle_,
                                                               distances_naive_dev.data(),
                                                               indices_naive_dev.data(),
                                                               search_queries.data(),
                                                               database.data(),
                                                               ps.n_queries,
                                                               ps.n_rows,
                                                               ps.dim,
                                                               ps.k,
                                                               ps.metric);
      raft::update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      raft::update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      raft::resource::sync_stream(handle_);
    }

    {
      rmm::device_uvector<DistanceT> distances_dev(queries_size, stream_);
      rmm::device_uvector<SearchIdxT> indices_dev(queries_size, stream_);

      {
        cagra::index_params index_params;
        index_params.metric = ps.metric;  // Note: currently ony the cagra::index_params metric is
                                          // not used for knn_graph building.
        index_params.graph_degree              = ps.graph_degree;
        index_params.intermediate_graph_degree = ps.graph_degree * 2;
        switch (ps.build_algo) {
          case graph_build_algo::IVF_PQ:
            index_params.graph_build_params = graph_build_params::ivf_pq_params(
              raft::matrix_extent<int64_t>(ps.n_rows, ps.dim), index_params.metric);
            if (ps.ivf_pq_search_refine_ratio) {
              std::get<cuvs::neighbors::cagra::graph_build_params::ivf_pq_params>(
                index_params.graph_build_params)
                .refinement_rate = *ps.ivf_pq_search_refine_ratio;
            }
            break;
          case graph_build_algo::NN_DESCENT: {
            index_params.graph_build_params = graph_build_params::nn_descent_params(
              index_params.intermediate_graph_degree, index_params.metric);
            break;
          }
          case graph_build_algo::ITERATIVE_CAGRA_SEARCH: {
            index_params.graph_build_params = graph_build_params::iterative_search_params();
            break;
          }
          case graph_build_algo::AUTO:
            // do nothing
            break;
        };

        index_params.compression = ps.compression;
        cagra::search_params search_params;
        search_params.algo        = ps.algo;
        search_params.max_queries = ps.max_queries;
        search_params.team_size   = ps.team_size;
        search_params.smem_dtype  = ps.smem_dtype;

        auto database_view = raft::make_device_matrix_view<const DataT, int64_t>(
          (const DataT*)database.data(), ps.n_rows, ps.dim);

        tmp_index_file index_file;
        {
          std::optional<raft::host_matrix<DataT, int64_t>> database_host{std::nullopt};
          cagra::index<DataT, IdxT> index(handle_, index_params.metric);
          if (ps.host_dataset) {
            database_host = raft::make_host_matrix<DataT, int64_t>(ps.n_rows, ps.dim);
            raft::copy(database_host->data_handle(), database.data(), database.size(), stream_);
            auto database_host_view = raft::make_host_matrix_view<const DataT, int64_t>(
              (const DataT*)database_host->data_handle(), ps.n_rows, ps.dim);

            index = cagra::build(handle_, index_params, database_host_view);
          } else {
            index = cagra::build(handle_, index_params, database_view);
          };

          if (ps.use_source_indices) {
            auto source_indices =
              raft::make_device_vector<IdxT, int64_t>(handle_, static_cast<int64_t>(index.size()));
            raft::linalg::map_offset(handle_, source_indices.view(), raft::cast_op<IdxT>{});
            index.update_source_indices(handle_, raft::make_const_mdspan(source_indices.view()));
          }

          cagra::serialize(handle_, index_file.filename, index, ps.include_serialized_dataset);
        }

        cagra::index<DataT, IdxT> index(handle_);
        cagra::deserialize(handle_, index_file.filename, &index);

        if (!ps.include_serialized_dataset) { index.update_dataset(handle_, database_view); }

        auto search_queries_view = raft::make_device_matrix_view<const DataT, int64_t>(
          search_queries.data(), ps.n_queries, ps.dim);
        auto indices_out_view = raft::make_device_matrix_view<SearchIdxT, int64_t>(
          indices_dev.data(), ps.n_queries, ps.k);
        auto dists_out_view = raft::make_device_matrix_view<DistanceT, int64_t>(
          distances_dev.data(), ps.n_queries, ps.k);

        cagra::search(
          handle_, search_params, index, search_queries_view, indices_out_view, dists_out_view);
        raft::update_host(distances_Cagra.data(), distances_dev.data(), queries_size, stream_);
        raft::update_host(indices_Cagra.data(), indices_dev.data(), queries_size, stream_);

        raft::resource::sync_stream(handle_);

        reference_recall = 1;
        if (ps.compression.has_value()) {
          auto decoded_dataset =
            raft::make_device_matrix<DataT, int64_t>(handle_, ps.n_rows, ps.dim);

          using VpqMathT = half;
          cuvs::neighbors::decode_vpq_dataset<DataT, VpqMathT>(
            decoded_dataset.view(),
            dynamic_cast<const cuvs::neighbors::vpq_dataset<VpqMathT, int64_t>&>(index.data()),
            raft::resource::get_cuda_stream(handle_));
          auto indices_out_view = raft::make_device_matrix_view<SearchIdxT, int64_t>(
            indices_dev.data(), ps.n_queries, ps.k);
          auto dists_out_view = raft::make_device_matrix_view<DistanceT, int64_t>(
            distances_dev.data(), ps.n_queries, ps.k);

          cuvs::neighbors::naive_knn<DistanceT, DataT, SearchIdxT>(handle_,
                                                                   dists_out_view.data_handle(),
                                                                   indices_out_view.data_handle(),
                                                                   search_queries.data(),
                                                                   decoded_dataset.data_handle(),
                                                                   ps.n_queries,
                                                                   ps.n_rows,
                                                                   ps.dim,
                                                                   ps.k,
                                                                   ps.metric);
          std::vector<SearchIdxT> indices_vpq_dataset(queries_size);
          std::vector<DistanceT> distances_vpq_dataset(queries_size);
          raft::update_host(
            distances_vpq_dataset.data(), dists_out_view.data_handle(), queries_size, stream_);
          raft::update_host(
            indices_vpq_dataset.data(), indices_out_view.data_handle(), queries_size, stream_);

          reference_recall = std::get<1>(calc_recall(indices_naive,
                                                     indices_vpq_dataset,
                                                     distances_naive,
                                                     distances_vpq_dataset,
                                                     ps.n_queries,
                                                     ps.k,
                                                     0));
          printf("reference_recall = %e\n", reference_recall);
        }
      }

      // for (int i = 0; i < min(ps.n_queries, 10); i++) {
      //   //  std::cout << "query " << i << std::end;
      //   print_vector("T", indices_naive.data() + i * ps.k, ps.k, std::cout);
      //   print_vector("C", indices_Cagra.data() + i * ps.k, ps.k, std::cout);
      //   print_vector("T", distances_naive.data() + i * ps.k, ps.k, std::cout);
      //   print_vector("C", distances_Cagra.data() + i * ps.k, ps.k, std::cout);
      // }

      // Heuristic recall threshold update
      double min_recall = ps.min_recall;
      if (ps.graph_degree < 50) { min_recall *= 0.94; }
      if (ps.graph_degree < 40) { min_recall *= 0.94; }
      min_recall = min_recall * reference_recall;
      EXPECT_TRUE(eval_neighbours(indices_naive,
                                  indices_Cagra,
                                  distances_naive,
                                  distances_Cagra,
                                  ps.n_queries,
                                  ps.k,
                                  0.003,
                                  min_recall));
      if (!ps.compression.has_value()) {
        // Don't evaluate distances for CAGRA-Q for now as the error can be somewhat large
        EXPECT_TRUE(eval_distances(handle_,
                                   database.data(),
                                   search_queries.data(),
                                   indices_dev.data(),
                                   distances_dev.data(),
                                   ps.n_rows,
                                   ps.dim,
                                   ps.n_queries,
                                   ps.k,
                                   ps.metric,
                                   1.0e-4));
      }
    }
  }

  void SetUp() override
  {
    database.resize(((size_t)ps.n_rows) * ps.dim, stream_);
    search_queries.resize(ps.n_queries * ps.dim, stream_);
    raft::random::RngState r(1234ULL);
    InitDataset(handle_, database.data(), ps.n_rows, ps.dim, ps.metric, r);
    InitDataset(handle_, search_queries.data(), ps.n_queries, ps.dim, ps.metric, r);
    raft::resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
    search_queries.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnCagraInputs ps;
  rmm::device_uvector<DataT> database;
  rmm::device_uvector<DataT> search_queries;
  double reference_recall;
};

template <typename DistanceT, typename DataT, typename IdxT>
class AnnCagraAddNodesTest : public ::testing::TestWithParam<AnnCagraInputs> {
 public:
  AnnCagraAddNodesTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnCagraInputs>::GetParam()),
      database(0, stream_),
      search_queries(0, stream_)
  {
  }

 protected:
  void testCagra()
  {
    if (ps.metric == cuvs::distance::DistanceType::L1 &&
        ps.build_algo != graph_build_algo::ITERATIVE_CAGRA_SEARCH)
      GTEST_SKIP();
    if (ps.metric == cuvs::distance::DistanceType::CosineExpanded) {
      if (ps.compression.has_value()) { GTEST_SKIP(); }
      if (ps.build_algo == graph_build_algo::ITERATIVE_CAGRA_SEARCH || ps.dim == 1) {
        GTEST_SKIP();
      }
    }
    if (ps.compression != std::nullopt) GTEST_SKIP();
    // IVF_PQ graph build does not support BitwiseHamming
    if (ps.metric == cuvs::distance::DistanceType::BitwiseHamming &&
        ((!std::is_same_v<DataT, uint8_t>) || (ps.build_algo == graph_build_algo::IVF_PQ)))
      GTEST_SKIP();
    // If the dataset dimension is small and the dataset size is large, there can be a lot of
    // dataset vectors that have the same distance to the query, especially in the binary Hamming
    // distance, making it impossible to make a top-k ground truth.
    if (ps.metric == cuvs::distance::DistanceType::BitwiseHamming &&
        (ps.k * ps.dim * 8 / 5 /*(=magic number)*/ < ps.n_rows))
      GTEST_SKIP();
    // Dataset size must be larger than both 33 and the intermediate graph degree
    if (ps.build_algo == graph_build_algo::NN_DESCENT &&
        ((ps.n_rows < 33) || raft::round_up_safe(ps.n_rows, 32) <= ps.graph_degree * 3))
      GTEST_SKIP();

    size_t queries_size = ps.n_queries * ps.k;
    std::vector<IdxT> indices_Cagra(queries_size);
    std::vector<IdxT> indices_naive(queries_size);
    std::vector<DistanceT> distances_Cagra(queries_size);
    std::vector<DistanceT> distances_naive(queries_size);

    {
      rmm::device_uvector<DistanceT> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);

      cuvs::neighbors::naive_knn<DistanceT, DataT, IdxT>(handle_,
                                                         distances_naive_dev.data(),
                                                         indices_naive_dev.data(),
                                                         search_queries.data(),
                                                         database.data(),
                                                         ps.n_queries,
                                                         ps.n_rows,
                                                         ps.dim,
                                                         ps.k,
                                                         ps.metric);
      raft::update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      raft::update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      raft::resource::sync_stream(handle_);
    }

    {
      rmm::device_uvector<DistanceT> distances_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_dev(queries_size, stream_);

      {
        cagra::index_params index_params;
        index_params.metric = ps.metric;  // Note: currently ony the cagra::index_params metric is
                                          // not used for knn_graph building.
        index_params.graph_degree              = ps.graph_degree;
        index_params.intermediate_graph_degree = ps.graph_degree * 2;

        switch (ps.build_algo) {
          case graph_build_algo::IVF_PQ:
            index_params.graph_build_params = graph_build_params::ivf_pq_params(
              raft::matrix_extent<int64_t>(ps.n_rows, ps.dim), index_params.metric);
            if (ps.ivf_pq_search_refine_ratio) {
              std::get<cuvs::neighbors::cagra::graph_build_params::ivf_pq_params>(
                index_params.graph_build_params)
                .refinement_rate = *ps.ivf_pq_search_refine_ratio;
            }
            break;
          case graph_build_algo::NN_DESCENT: {
            index_params.graph_build_params = graph_build_params::nn_descent_params(
              index_params.intermediate_graph_degree, index_params.metric);
            break;
          }
          case graph_build_algo::ITERATIVE_CAGRA_SEARCH: {
            index_params.graph_build_params = graph_build_params::iterative_search_params();
            break;
          }
          case graph_build_algo::AUTO:
            // do nothing
            break;
        };

        cagra::search_params search_params;
        search_params.algo        = ps.algo;
        search_params.max_queries = ps.max_queries;
        search_params.team_size   = ps.team_size;
        search_params.itopk_size  = ps.itopk_size;

        const double initial_dataset_ratio      = 0.90;
        const std::size_t initial_database_size = ps.n_rows * initial_dataset_ratio;

        auto initial_database_view = raft::make_device_matrix_view<const DataT, int64_t>(
          (const DataT*)database.data(), initial_database_size, ps.dim);

        std::optional<raft::host_matrix<DataT, int64_t>> database_host{std::nullopt};
        cagra::index<DataT, IdxT> index(handle_);
        if (ps.host_dataset) {
          database_host = raft::make_host_matrix<DataT, int64_t>(ps.n_rows, ps.dim);
          raft::copy(
            database_host->data_handle(), database.data(), initial_database_view.size(), stream_);
          auto database_host_view = raft::make_host_matrix_view<const DataT, int64_t>(
            (const DataT*)database_host->data_handle(), initial_database_size, ps.dim);
          // NB: database_host must live no less than the index, because the index _may_be_
          //     non-onwning
          index = cagra::build(handle_, index_params, database_host_view);
        } else {
          index = cagra::build(handle_, index_params, initial_database_view);
        };

        auto additional_dataset =
          raft::make_host_matrix<DataT, int64_t>(ps.n_rows - initial_database_size, index.dim());
        raft::copy(additional_dataset.data_handle(),
                   static_cast<const DataT*>(database.data()) + initial_database_view.size(),
                   additional_dataset.size(),
                   stream_);

        auto new_dataset_buffer = raft::make_device_matrix<DataT, int64_t>(handle_, 0, 0);
        auto new_graph_buffer   = raft::make_device_matrix<IdxT, int64_t>(handle_, 0, 0);
        std::optional<raft::device_matrix_view<DataT, int64_t, raft::layout_stride>>
          new_dataset_buffer_view                                                    = std::nullopt;
        std::optional<raft::device_matrix_view<IdxT, int64_t>> new_graph_buffer_view = std::nullopt;
        if (ps.non_owning_memory_buffer_flag.has_value() &&
            ps.non_owning_memory_buffer_flag.value()) {
          const auto stride =
            dynamic_cast<const cuvs::neighbors::strided_dataset<DataT, int64_t>*>(&index.data())
              ->stride();
          new_dataset_buffer = raft::make_device_matrix<DataT, int64_t>(handle_, ps.n_rows, stride);
          new_graph_buffer =
            raft::make_device_matrix<IdxT, int64_t>(handle_, ps.n_rows, index.graph_degree());

          new_dataset_buffer_view = raft::make_device_strided_matrix_view<DataT, int64_t>(
            new_dataset_buffer.data_handle(), ps.n_rows, ps.dim, stride);
          new_graph_buffer_view = new_graph_buffer.view();
        }

        cagra::extend_params extend_params;
        cagra::extend(handle_,
                      extend_params,
                      raft::make_const_mdspan(additional_dataset.view()),
                      index,
                      new_dataset_buffer_view,
                      new_graph_buffer_view);

        auto search_queries_view = raft::make_device_matrix_view<const DataT, int64_t>(
          search_queries.data(), ps.n_queries, ps.dim);
        auto indices_out_view =
          raft::make_device_matrix_view<IdxT, int64_t>(indices_dev.data(), ps.n_queries, ps.k);
        auto dists_out_view = raft::make_device_matrix_view<DistanceT, int64_t>(
          distances_dev.data(), ps.n_queries, ps.k);

        cagra::search(
          handle_, search_params, index, search_queries_view, indices_out_view, dists_out_view);
        raft::update_host(distances_Cagra.data(), distances_dev.data(), queries_size, stream_);
        raft::update_host(indices_Cagra.data(), indices_dev.data(), queries_size, stream_);
        raft::resource::sync_stream(handle_);
      }

      // Heuristic recall threshold update
      double min_recall = ps.min_recall;
      if (ps.graph_degree < 50) { min_recall *= 0.94; }
      if (ps.graph_degree < 40) { min_recall *= 0.94; }
      EXPECT_TRUE(eval_neighbours(indices_naive,
                                  indices_Cagra,
                                  distances_naive,
                                  distances_Cagra,
                                  ps.n_queries,
                                  ps.k,
                                  0.006,
                                  min_recall));
      EXPECT_TRUE(eval_distances(handle_,
                                 database.data(),
                                 search_queries.data(),
                                 indices_dev.data(),
                                 distances_dev.data(),
                                 ps.n_rows,
                                 ps.dim,
                                 ps.n_queries,
                                 ps.k,
                                 ps.metric,
                                 1.0e-4));
    }
  }

  void SetUp() override
  {
    database.resize(((size_t)ps.n_rows) * ps.dim, stream_);
    search_queries.resize(ps.n_queries * ps.dim, stream_);
    raft::random::RngState r(1234ULL);
    InitDataset(handle_, database.data(), ps.n_rows, ps.dim, ps.metric, r);
    InitDataset(handle_, search_queries.data(), ps.n_queries, ps.dim, ps.metric, r);
    raft::resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
    search_queries.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnCagraInputs ps;
  rmm::device_uvector<DataT> database;
  rmm::device_uvector<DataT> search_queries;
};

template <typename DistanceT, typename DataT, typename IdxT>
class AnnCagraFilterTest : public ::testing::TestWithParam<AnnCagraInputs> {
 public:
  AnnCagraFilterTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnCagraInputs>::GetParam()),
      database(0, stream_),
      search_queries(0, stream_)
  {
  }

 protected:
  void testCagra()
  {
    if (ps.metric == cuvs::distance::DistanceType::L1 &&
        ps.build_algo != graph_build_algo::ITERATIVE_CAGRA_SEARCH)
      GTEST_SKIP();
    if (ps.metric == cuvs::distance::DistanceType::CosineExpanded) {
      if (ps.compression.has_value()) { GTEST_SKIP(); }
      if (ps.build_algo == graph_build_algo::ITERATIVE_CAGRA_SEARCH || ps.dim == 1) {
        GTEST_SKIP();
      }
    }
    // IVF_PQ graph build does not support BitwiseHamming
    if (ps.metric == cuvs::distance::DistanceType::BitwiseHamming &&
        ((!std::is_same_v<DataT, uint8_t>) || (ps.build_algo == graph_build_algo::IVF_PQ)))
      GTEST_SKIP();
    // If the dataset dimension is small and the dataset size is large, there can be a lot of
    // dataset vectors that have the same distance to the query, especially in the binary Hamming
    // distance, making it impossible to make a top-k ground truth.
    if (ps.metric == cuvs::distance::DistanceType::BitwiseHamming &&
        (ps.k * ps.dim * 8 / 5 /*(=magic number)*/ < ps.n_rows))
      GTEST_SKIP();
    // Dataset size must be larger than both 33 and the intermediate graph degree
    if (ps.build_algo == graph_build_algo::NN_DESCENT &&
        ((ps.n_rows < 33) || raft::round_up_safe(ps.n_rows, 32) <= ps.graph_degree * 3))
      GTEST_SKIP();

    size_t queries_size = ps.n_queries * ps.k;
    std::vector<IdxT> indices_Cagra(queries_size);
    std::vector<IdxT> indices_naive(queries_size);
    std::vector<DistanceT> distances_Cagra(queries_size);
    std::vector<DistanceT> distances_naive(queries_size);

    {
      rmm::device_uvector<DistanceT> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);
      auto* database_filtered_ptr = database.data() + test_cagra_sample_filter::offset * ps.dim;
      cuvs::neighbors::naive_knn<DistanceT, DataT, IdxT>(
        handle_,
        distances_naive_dev.data(),
        indices_naive_dev.data(),
        search_queries.data(),
        database_filtered_ptr,
        ps.n_queries,
        ps.n_rows - test_cagra_sample_filter::offset,
        ps.dim,
        ps.k,
        ps.metric);
      raft::linalg::addScalar(indices_naive_dev.data(),
                              indices_naive_dev.data(),
                              IdxT(test_cagra_sample_filter::offset),
                              queries_size,
                              stream_);
      raft::update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      raft::update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      raft::resource::sync_stream(handle_);
    }

    {
      rmm::device_uvector<DistanceT> distances_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_dev(queries_size, stream_);

      {
        cagra::index_params index_params;
        index_params.metric = ps.metric;  // Note: currently ony the cagra::index_params metric is
                                          // not used for knn_graph building.
        index_params.graph_degree              = ps.graph_degree;
        index_params.intermediate_graph_degree = ps.graph_degree * 2;

        switch (ps.build_algo) {
          case graph_build_algo::IVF_PQ:
            index_params.graph_build_params = graph_build_params::ivf_pq_params(
              raft::matrix_extent<int64_t>(ps.n_rows, ps.dim), index_params.metric);
            if (ps.ivf_pq_search_refine_ratio) {
              std::get<cuvs::neighbors::cagra::graph_build_params::ivf_pq_params>(
                index_params.graph_build_params)
                .refinement_rate = *ps.ivf_pq_search_refine_ratio;
            }
            break;
          case graph_build_algo::NN_DESCENT: {
            index_params.graph_build_params =
              graph_build_params::nn_descent_params(index_params.intermediate_graph_degree);
            break;
          }
          case graph_build_algo::ITERATIVE_CAGRA_SEARCH: {
            index_params.graph_build_params = graph_build_params::iterative_search_params();
            break;
          }
          case graph_build_algo::AUTO:
            // do nothing
            break;
        };

        index_params.compression = ps.compression;
        cagra::search_params search_params;
        search_params.algo        = ps.algo;
        search_params.max_queries = ps.max_queries;
        search_params.team_size   = ps.team_size;
        search_params.itopk_size  = ps.itopk_size;

        auto database_view = raft::make_device_matrix_view<const DataT, int64_t>(
          (const DataT*)database.data(), ps.n_rows, ps.dim);

        std::optional<raft::host_matrix<DataT, int64_t>> database_host{std::nullopt};
        cagra::index<DataT, IdxT> index(handle_);
        if (ps.host_dataset) {
          database_host = raft::make_host_matrix<DataT, int64_t>(ps.n_rows, ps.dim);
          raft::copy(database_host->data_handle(), database.data(), database.size(), stream_);
          auto database_host_view = raft::make_host_matrix_view<const DataT, int64_t>(
            (const DataT*)database_host->data_handle(), ps.n_rows, ps.dim);
          index = cagra::build(handle_, index_params, database_host_view);
        } else {
          index = cagra::build(handle_, index_params, database_view);
        }

        if (!ps.include_serialized_dataset) { index.update_dataset(handle_, database_view); }

        if (ps.use_source_indices) {
          auto source_indices =
            raft::make_device_vector<IdxT, int64_t>(handle_, static_cast<int64_t>(index.size()));
          raft::linalg::map_offset(handle_, source_indices.view(), raft::cast_op<IdxT>{});
          index.update_source_indices(std::move(source_indices));
        }

        auto search_queries_view = raft::make_device_matrix_view<const DataT, int64_t>(
          search_queries.data(), ps.n_queries, ps.dim);
        auto indices_out_view =
          raft::make_device_matrix_view<IdxT, int64_t>(indices_dev.data(), ps.n_queries, ps.k);
        auto dists_out_view = raft::make_device_matrix_view<DistanceT, int64_t>(
          distances_dev.data(), ps.n_queries, ps.k);
        auto removed_indices =
          raft::make_device_vector<int64_t, int64_t>(handle_, test_cagra_sample_filter::offset);
        thrust::sequence(
          raft::resource::get_thrust_policy(handle_),
          thrust::device_pointer_cast(removed_indices.data_handle()),
          thrust::device_pointer_cast(removed_indices.data_handle() + removed_indices.extent(0)));
        raft::resource::sync_stream(handle_);
        cuvs::core::bitset<std::uint32_t, int64_t> removed_indices_bitset(
          handle_, removed_indices.view(), ps.n_rows);
        auto bitset_filter_obj =
          cuvs::neighbors::filtering::bitset_filter(removed_indices_bitset.view());
        cagra::search(handle_,
                      search_params,
                      index,
                      search_queries_view,
                      indices_out_view,
                      dists_out_view,
                      bitset_filter_obj);
        raft::update_host(distances_Cagra.data(), distances_dev.data(), queries_size, stream_);
        raft::update_host(indices_Cagra.data(), indices_dev.data(), queries_size, stream_);
        raft::resource::sync_stream(handle_);
      }

      // Test search results for nodes marked as filtered
      bool unacceptable_node = false;
      for (int q = 0; q < ps.n_queries; q++) {
        for (int i = 0; i < ps.k; i++) {
          const auto n      = indices_Cagra[q * ps.k + i];
          unacceptable_node = unacceptable_node | !test_cagra_sample_filter()(q, n);
        }
      }
      EXPECT_FALSE(unacceptable_node);

      // Heuristic recall threshold update
      double min_recall = ps.min_recall;
      if (ps.graph_degree < 50) { min_recall *= 0.94; }
      if (ps.graph_degree < 40) { min_recall *= 0.94; }
      // TODO(mfoerster): re-enable uniqueness test
      EXPECT_TRUE(eval_neighbours(indices_naive,
                                  indices_Cagra,
                                  distances_naive,
                                  distances_Cagra,
                                  ps.n_queries,
                                  ps.k,
                                  0.003,
                                  min_recall,
                                  false));
      if (!ps.compression.has_value()) {
        // Don't evaluate distances for CAGRA-Q for now as the error can be somewhat large
        EXPECT_TRUE(eval_distances(handle_,
                                   database.data(),
                                   search_queries.data(),
                                   indices_dev.data(),
                                   distances_dev.data(),
                                   ps.n_rows,
                                   ps.dim,
                                   ps.n_queries,
                                   ps.k,
                                   ps.metric,
                                   1.0e-4));
      }
    }
  }

  void SetUp() override
  {
    database.resize(((size_t)ps.n_rows) * ps.dim, stream_);
    search_queries.resize(ps.n_queries * ps.dim, stream_);
    raft::random::RngState r(1234ULL);
    InitDataset(handle_, database.data(), ps.n_rows, ps.dim, ps.metric, r);
    InitDataset(handle_, search_queries.data(), ps.n_queries, ps.dim, ps.metric, r);
    raft::resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
    search_queries.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnCagraInputs ps;
  rmm::device_uvector<DataT> database;
  rmm::device_uvector<DataT> search_queries;
};

template <typename DistanceT, typename DataT, typename IdxT>
class AnnCagraIndexFilteredMergeTest : public ::testing::TestWithParam<AnnCagraInputs> {
 public:
  AnnCagraIndexFilteredMergeTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnCagraInputs>::GetParam()),
      database(0, stream_),
      search_queries(0, stream_)
  {
  }

 protected:
  template <typename SearchIdxT = IdxT>
  void testCagra()
  {
    if (ps.metric == cuvs::distance::DistanceType::L1 &&
        ps.build_algo != graph_build_algo::ITERATIVE_CAGRA_SEARCH)
      GTEST_SKIP();
    if (ps.metric == cuvs::distance::DistanceType::CosineExpanded) {
      if (ps.build_algo == graph_build_algo::ITERATIVE_CAGRA_SEARCH || ps.dim == 1) {
        GTEST_SKIP();
      }
    }
    if (ps.compression != std::nullopt) GTEST_SKIP();
    // IVF_PQ graph build does not support BitwiseHamming
    if (ps.metric == cuvs::distance::DistanceType::BitwiseHamming &&
        ((!std::is_same_v<DataT, uint8_t>) || (ps.build_algo == graph_build_algo::IVF_PQ)))
      GTEST_SKIP();
    // If the dataset dimension is small and the dataset size is large, there can be a lot of
    // dataset vectors that have the same distance to the query, especially in the binary Hamming
    // distance, making it impossible to make a top-k ground truth.
    if (ps.metric == cuvs::distance::DistanceType::BitwiseHamming &&
        (ps.k * ps.dim * 8 / 5 /*(=magic number)*/ < ps.n_rows))
      GTEST_SKIP();

    // Avoid splitting datasets with a size of 0
    if (ps.n_rows <= 3) GTEST_SKIP();

    // IVF_PQ requires the `n_rows >= n_lists`.
    if (ps.n_rows < 8 && ps.build_algo == graph_build_algo::IVF_PQ) GTEST_SKIP();

    // can only use physical merge for filtered merge
    if (ps.merge_strategy != cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_PHYSICAL) {
      GTEST_SKIP();
    }

    // Can't filter out more rows than are in the dataset
    if (static_cast<uint32_t>(ps.n_rows) <= test_cagra_sample_filter::offset) { GTEST_SKIP(); }

    size_t queries_size = ps.n_queries * ps.k;
    std::vector<SearchIdxT> indices_Cagra(queries_size);
    std::vector<SearchIdxT> indices_naive(queries_size);
    std::vector<DistanceT> distances_Cagra(queries_size);
    std::vector<DistanceT> distances_naive(queries_size);

    // Create a bitset filter to test out the merge
    auto removed_indices =
      raft::make_device_vector<int64_t, int64_t>(handle_, test_cagra_sample_filter::offset);
    thrust::sequence(
      raft::resource::get_thrust_policy(handle_),
      thrust::device_pointer_cast(removed_indices.data_handle()),
      thrust::device_pointer_cast(removed_indices.data_handle() + removed_indices.extent(0)));
    raft::resource::sync_stream(handle_);
    cuvs::core::bitset<std::uint32_t, int64_t> removed_indices_bitset(
      handle_, removed_indices.view(), ps.n_rows);
    auto bitset_filter_obj =
      cuvs::neighbors::filtering::bitset_filter(removed_indices_bitset.view());

    {
      rmm::device_uvector<DistanceT> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<SearchIdxT> indices_naive_dev(queries_size, stream_);

      auto* database_filtered_ptr = database.data() + test_cagra_sample_filter::offset * ps.dim;

      cuvs::neighbors::naive_knn<DistanceT, DataT, IdxT>(
        handle_,
        distances_naive_dev.data(),
        indices_naive_dev.data(),
        search_queries.data(),
        database_filtered_ptr,
        ps.n_queries,
        ps.n_rows - test_cagra_sample_filter::offset,
        ps.dim,
        ps.k,
        ps.metric);

      raft::linalg::addScalar(indices_naive_dev.data(),
                              indices_naive_dev.data(),
                              IdxT(test_cagra_sample_filter::offset),
                              queries_size,
                              stream_);

      raft::update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      raft::update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      raft::resource::sync_stream(handle_);
    }

    {
      rmm::device_uvector<DistanceT> distances_dev(queries_size, stream_);
      rmm::device_uvector<SearchIdxT> indices_dev(queries_size, stream_);

      {
        cagra::index_params index_params;
        index_params.metric = ps.metric;  // Note: currently ony the cagra::index_params metric is
                                          // not used for knn_graph building.

        switch (ps.build_algo) {
          case graph_build_algo::IVF_PQ:
            index_params.graph_build_params = graph_build_params::ivf_pq_params(
              raft::matrix_extent<int64_t>(ps.n_rows, ps.dim), index_params.metric);
            if (ps.ivf_pq_search_refine_ratio) {
              std::get<cuvs::neighbors::cagra::graph_build_params::ivf_pq_params>(
                index_params.graph_build_params)
                .refinement_rate = *ps.ivf_pq_search_refine_ratio;
            }
            break;
          case graph_build_algo::NN_DESCENT: {
            index_params.graph_build_params =
              graph_build_params::nn_descent_params(index_params.intermediate_graph_degree);
            break;
          }
          case graph_build_algo::ITERATIVE_CAGRA_SEARCH: {
            index_params.graph_build_params = graph_build_params::iterative_search_params();
            break;
          }
          case graph_build_algo::AUTO:
            // do nothing
            break;
        };

        const double split_ratio         = 0.55;
        const std::size_t database0_size = ps.n_rows * split_ratio;
        const std::size_t database1_size = ps.n_rows - database0_size;

        auto database0_view = raft::make_device_matrix_view<const DataT, int64_t>(
          (const DataT*)database.data(), database0_size, ps.dim);

        auto database1_view = raft::make_device_matrix_view<const DataT, int64_t>(
          (const DataT*)database.data() + database0_view.size(), database1_size, ps.dim);

        cagra::index<DataT, IdxT> index0(handle_, index_params.metric);
        cagra::index<DataT, IdxT> index1(handle_, index_params.metric);
        std::optional<raft::host_matrix<DataT, int64_t>> database_host{std::nullopt};
        if (ps.host_dataset) {
          database_host = raft::make_host_matrix<DataT, int64_t>(handle_, ps.n_rows, ps.dim);
          raft::copy(database_host->data_handle(), database.data(), database.size(), stream_);
          {
            auto database_host_view = raft::make_host_matrix_view<const DataT, int64_t>(
              (const DataT*)database_host->data_handle(), database0_size, ps.dim);
            index0 = cagra::build(handle_, index_params, database_host_view);
          }
          {
            auto database_host_view = raft::make_host_matrix_view<const DataT, int64_t>(
              (const DataT*)database_host->data_handle() + database0_size * ps.dim,
              database1_size,
              ps.dim);
            index1 = cagra::build(handle_, index_params, database_host_view);
          }
        } else {
          index0 = cagra::build(handle_, index_params, database0_view);
          index1 = cagra::build(handle_, index_params, database1_view);
        };

        std::vector<cuvs::neighbors::cagra::index<DataT, IdxT>*> indices;
        indices.push_back(&index0);
        indices.push_back(&index1);

        auto index =
          cuvs::neighbors::cagra::merge(handle_, index_params, indices, bitset_filter_obj);

        auto search_queries_view = raft::make_device_matrix_view<const DataT, int64_t>(
          search_queries.data(), ps.n_queries, ps.dim);
        auto indices_out_view = raft::make_device_matrix_view<SearchIdxT, int64_t>(
          indices_dev.data(), ps.n_queries, ps.k);
        auto dists_out_view = raft::make_device_matrix_view<DistanceT, int64_t>(
          distances_dev.data(), ps.n_queries, ps.k);

        cagra::search_params search_params;
        search_params.algo        = ps.algo;
        search_params.max_queries = ps.max_queries;
        search_params.team_size   = ps.team_size;
        search_params.itopk_size  = ps.itopk_size;

        cuvs::neighbors::cagra::search(
          handle_, search_params, index, search_queries_view, indices_out_view, dists_out_view);

        raft::update_host(distances_Cagra.data(), distances_dev.data(), queries_size, stream_);
        raft::update_host(indices_Cagra.data(), indices_dev.data(), queries_size, stream_);
        raft::resource::sync_stream(handle_);
      }

      double min_recall = ps.min_recall;
      EXPECT_TRUE(eval_neighbours(indices_naive,
                                  indices_Cagra,
                                  distances_naive,
                                  distances_Cagra,
                                  ps.n_queries,
                                  ps.k,
                                  0.006,
                                  min_recall));

      /* TODO: eval_distances doesn't work, potentially because of id translation mismatch
      if (!ps.compression.has_value()) {
        EXPECT_TRUE(eval_distances(handle_,
                                   database.data(),
                                   search_queries.data(),
                                   indices_dev.data(),
                                   distances_dev.data(),
                                   ps.n_rows,
                                   ps.dim,
                                   ps.n_queries,
                                   ps.k,
                                   ps.metric,
                                   1.0e-4));
      }
      */
    }
  }

  void SetUp() override
  {
    database.resize(((size_t)ps.n_rows) * ps.dim, stream_);
    search_queries.resize(ps.n_queries * ps.dim, stream_);
    raft::random::RngState r(1234ULL);
    InitDataset(handle_, database.data(), ps.n_rows, ps.dim, ps.metric, r);
    InitDataset(handle_, search_queries.data(), ps.n_queries, ps.dim, ps.metric, r);
    raft::resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
    search_queries.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnCagraInputs ps;
  rmm::device_uvector<DataT> database;
  rmm::device_uvector<DataT> search_queries;
};

template <typename DistanceT, typename DataT, typename IdxT>
class AnnCagraIndexMergeTest : public ::testing::TestWithParam<AnnCagraInputs> {
 public:
  AnnCagraIndexMergeTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnCagraInputs>::GetParam()),
      database(0, stream_),
      search_queries(0, stream_)
  {
  }

 protected:
  template <typename SearchIdxT = IdxT>
  void testCagra()
  {
    if (ps.metric == cuvs::distance::DistanceType::L1 &&
        ps.build_algo != graph_build_algo::ITERATIVE_CAGRA_SEARCH)
      GTEST_SKIP();
    if (ps.metric == cuvs::distance::DistanceType::CosineExpanded) {
      if (ps.build_algo == graph_build_algo::ITERATIVE_CAGRA_SEARCH || ps.dim == 1) {
        GTEST_SKIP();
      }
    }
    if (ps.compression != std::nullopt) GTEST_SKIP();
    // IVF_PQ graph build does not support BitwiseHamming
    if (ps.metric == cuvs::distance::DistanceType::BitwiseHamming &&
        ((!std::is_same_v<DataT, uint8_t>) || (ps.build_algo == graph_build_algo::IVF_PQ)))
      GTEST_SKIP();
    // If the dataset dimension is small and the dataset size is large, there can be a lot of
    // dataset vectors that have the same distance to the query, especially in the binary Hamming
    // distance, making it impossible to make a top-k ground truth.
    if (ps.metric == cuvs::distance::DistanceType::BitwiseHamming &&
        (ps.k * ps.dim * 8 / 5 /*(=magic number)*/ < ps.n_rows))
      GTEST_SKIP();

    const double splite_ratio        = 0.55;
    const std::size_t database0_size = ps.n_rows * splite_ratio;
    const std::size_t database1_size = ps.n_rows - database0_size;
    // Dataset size must be larger than both 33 and the intermediate graph degree
    if (ps.build_algo == graph_build_algo::NN_DESCENT &&
        ((database0_size < 33) ||
         raft::round_up_safe(database0_size, 32lu) <= ps.graph_degree * 3lu))
      GTEST_SKIP();
    if (ps.build_algo == graph_build_algo::NN_DESCENT &&
        ((database1_size < 33) ||
         raft::round_up_safe(database1_size, 32lu) <= ps.graph_degree * 3lu))
      GTEST_SKIP();

    // Avoid splitting datasets with a size of 0
    if (ps.n_rows <= 3) GTEST_SKIP();

    // IVF_PQ requires the `n_rows >= n_lists`.
    if (ps.n_rows < 8 && ps.build_algo == graph_build_algo::IVF_PQ) GTEST_SKIP();

    size_t queries_size = ps.n_queries * ps.k;
    std::vector<SearchIdxT> indices_Cagra(queries_size);
    std::vector<SearchIdxT> indices_naive(queries_size);
    std::vector<DistanceT> distances_Cagra(queries_size);
    std::vector<DistanceT> distances_naive(queries_size);

    {
      rmm::device_uvector<DistanceT> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<SearchIdxT> indices_naive_dev(queries_size, stream_);

      cuvs::neighbors::naive_knn<DistanceT, DataT, SearchIdxT>(handle_,
                                                               distances_naive_dev.data(),
                                                               indices_naive_dev.data(),
                                                               search_queries.data(),
                                                               database.data(),
                                                               ps.n_queries,
                                                               ps.n_rows,
                                                               ps.dim,
                                                               ps.k,
                                                               ps.metric);
      raft::update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      raft::update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      raft::resource::sync_stream(handle_);
    }

    {
      rmm::device_uvector<DistanceT> distances_dev(queries_size, stream_);
      rmm::device_uvector<SearchIdxT> indices_dev(queries_size, stream_);

      {
        cagra::index_params index_params;
        index_params.metric = ps.metric;  // Note: currently ony the cagra::index_params metric is
                                          // not used for knn_graph building.
        index_params.graph_degree              = ps.graph_degree;
        index_params.intermediate_graph_degree = ps.graph_degree * 2;

        switch (ps.build_algo) {
          case graph_build_algo::IVF_PQ:
            index_params.graph_build_params = graph_build_params::ivf_pq_params(
              raft::matrix_extent<int64_t>(ps.n_rows, ps.dim), index_params.metric);
            if (ps.ivf_pq_search_refine_ratio) {
              std::get<cuvs::neighbors::cagra::graph_build_params::ivf_pq_params>(
                index_params.graph_build_params)
                .refinement_rate = *ps.ivf_pq_search_refine_ratio;
            }
            break;
          case graph_build_algo::NN_DESCENT: {
            index_params.graph_build_params =
              graph_build_params::nn_descent_params(index_params.intermediate_graph_degree);
            break;
          }
          case graph_build_algo::ITERATIVE_CAGRA_SEARCH: {
            index_params.graph_build_params = graph_build_params::iterative_search_params();
            break;
          }
          case graph_build_algo::AUTO:
            // do nothing
            break;
        };

        const double split_ratio         = 0.55;
        const std::size_t database0_size = ps.n_rows * split_ratio;
        const std::size_t database1_size = ps.n_rows - database0_size;

        auto database0_view = raft::make_device_matrix_view<const DataT, int64_t>(
          (const DataT*)database.data(), database0_size, ps.dim);

        auto database1_view = raft::make_device_matrix_view<const DataT, int64_t>(
          (const DataT*)database.data() + database0_view.size(), database1_size, ps.dim);

        cagra::index<DataT, IdxT> index0(handle_, index_params.metric);
        cagra::index<DataT, IdxT> index1(handle_, index_params.metric);
        std::optional<raft::host_matrix<DataT, int64_t>> database_host{std::nullopt};
        if (ps.host_dataset) {
          database_host = raft::make_host_matrix<DataT, int64_t>(handle_, ps.n_rows, ps.dim);
          raft::copy(database_host->data_handle(), database.data(), database.size(), stream_);
          {
            auto database_host_view = raft::make_host_matrix_view<const DataT, int64_t>(
              (const DataT*)database_host->data_handle(), database0_size, ps.dim);
            index0 = cagra::build(handle_, index_params, database_host_view);
          }
          {
            auto database_host_view = raft::make_host_matrix_view<const DataT, int64_t>(
              (const DataT*)database_host->data_handle() + database0_size * ps.dim,
              database1_size,
              ps.dim);
            index1 = cagra::build(handle_, index_params, database_host_view);
          }
        } else {
          index0 = cagra::build(handle_, index_params, database0_view);
          index1 = cagra::build(handle_, index_params, database1_view);
        };

        auto search_queries_view = raft::make_device_matrix_view<const DataT, int64_t>(
          search_queries.data(), ps.n_queries, ps.dim);
        auto indices_out_view = raft::make_device_matrix_view<SearchIdxT, int64_t>(
          indices_dev.data(), ps.n_queries, ps.k);
        auto dists_out_view = raft::make_device_matrix_view<DistanceT, int64_t>(
          distances_dev.data(), ps.n_queries, ps.k);

        cagra::search_params search_params;
        search_params.algo        = ps.algo;
        search_params.max_queries = ps.max_queries;
        search_params.team_size   = ps.team_size;
        search_params.itopk_size  = ps.itopk_size;

        std::vector<cagra::index<DataT, IdxT>*> indices_to_merge{&index0, &index1};

        if (ps.merge_strategy == cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_PHYSICAL) {
          auto merged =
            ps.physical_merge_params.has_value()
              ? cagra::merge(handle_, index_params, indices_to_merge, *ps.physical_merge_params)
              : cagra::merge(handle_, index_params, indices_to_merge);
          cagra::search(
            handle_, search_params, merged, search_queries_view, indices_out_view, dists_out_view);
        } else {
          cuvs::neighbors::composite::composite_index<DataT, IdxT, SearchIdxT> composite(
            indices_to_merge);
          composite.search(
            handle_, search_params, search_queries_view, indices_out_view, dists_out_view);
        }

        raft::update_host(distances_Cagra.data(), distances_dev.data(), queries_size, stream_);
        raft::update_host(indices_Cagra.data(), indices_dev.data(), queries_size, stream_);
        raft::resource::sync_stream(handle_);
      }

      // Heuristic recall threshold update
      double min_recall = ps.min_recall;
      if (ps.graph_degree < 50) { min_recall *= 0.94; }
      if (ps.graph_degree < 40) { min_recall *= 0.94; }
      EXPECT_TRUE(eval_neighbours(indices_naive,
                                  indices_Cagra,
                                  distances_naive,
                                  distances_Cagra,
                                  ps.n_queries,
                                  ps.k,
                                  0.006,
                                  min_recall));
      EXPECT_TRUE(eval_distances(handle_,
                                 database.data(),
                                 search_queries.data(),
                                 indices_dev.data(),
                                 distances_dev.data(),
                                 ps.n_rows,
                                 ps.dim,
                                 ps.n_queries,
                                 ps.k,
                                 ps.metric,
                                 1.0e-4));
    }
  }

  void SetUp() override
  {
    database.resize(((size_t)ps.n_rows) * ps.dim, stream_);
    search_queries.resize(ps.n_queries * ps.dim, stream_);
    raft::random::RngState r(1234ULL);
    InitDataset(handle_, database.data(), ps.n_rows, ps.dim, ps.metric, r);
    InitDataset(handle_, search_queries.data(), ps.n_queries, ps.dim, ps.metric, r);
    raft::resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
    search_queries.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnCagraInputs ps;
  rmm::device_uvector<DataT> database;
  rmm::device_uvector<DataT> search_queries;
};

inline std::vector<AnnCagraInputs> generate_inputs()
{
  // TODO(tfeher): test MULTI_CTA kernel with search_width > 1 to allow multiple CTA per queries
  // Change graph dim, search algo and max_query parameter
  std::vector<AnnCagraInputs> inputs = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {1000},
    {1, 16},
    {16},                                                      // k
    {32, 47, 64},                                              // degree
    {graph_build_algo::IVF_PQ, graph_build_algo::NN_DESCENT},  // build algo.
    {search_algo::SINGLE_CTA, search_algo::MULTI_CTA, search_algo::MULTI_KERNEL},
    {0, 10},  // query size
    {0},
    {256},
    {1},
    {cuvs::distance::DistanceType::L2Expanded,
     cuvs::distance::DistanceType::InnerProduct,
     cuvs::distance::DistanceType::BitwiseHamming,
     cuvs::distance::DistanceType::CosineExpanded,
     cuvs::distance::DistanceType::L1},
    {false},
    {true},
    {true, false},
    {0.995},
    {std::optional<float>{std::nullopt}},
    {std::optional<vpq_params>{std::nullopt}},
    {std::optional<bool>{std::nullopt}},
    {cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_PHYSICAL});

  auto inputs2 = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {1000},
    {1, 16},
    {16},                            // k
    {32},                            // degree
    {graph_build_algo::NN_DESCENT},  // build algo.
    {search_algo::MULTI_CTA},
    {10},  // query size
    {0},
    {256},
    {1},
    {cuvs::distance::DistanceType::L2Expanded,
     cuvs::distance::DistanceType::InnerProduct,
     cuvs::distance::DistanceType::BitwiseHamming,
     cuvs::distance::DistanceType::CosineExpanded,
     cuvs::distance::DistanceType::L1},
    {false},
    {true},
    {false},
    {0.995},
    {std::optional<float>{std::nullopt}},
    {std::optional<vpq_params>{std::nullopt}},
    {std::optional<bool>{std::nullopt}},
    {cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_LOGICAL});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  // Additional distances tested with a single search algo.
  inputs2 = raft::util::itertools::product<AnnCagraInputs>(
    {1, 100},
    {1000},
    {8},
    {1, 16},  // k
    {32},     // degree
    {graph_build_algo::NN_DESCENT},
    {search_algo::SINGLE_CTA},
    {0},  // query size
    {0},
    {256},
    {1},
    {cuvs::distance::DistanceType::InnerProduct,
     cuvs::distance::DistanceType::BitwiseHamming,
     cuvs::distance::DistanceType::CosineExpanded,
     cuvs::distance::DistanceType::L1},
    {false},
    {true},
    {false},
    {0.995},
    {std::optional<float>{std::nullopt}},
    {std::optional<vpq_params>{std::nullopt}},
    {std::optional<bool>{std::nullopt}},
    {cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_PHYSICAL});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  // Corner cases for small datasets
  inputs2 = raft::util::itertools::product<AnnCagraInputs>(
    {2},
    {3, 6, 31, 32, 64, 101},
    {1, 10},
    {2},   // k
    {32},  // degree
    {graph_build_algo::IVF_PQ, graph_build_algo::NN_DESCENT},
    {search_algo::SINGLE_CTA, search_algo::MULTI_CTA, search_algo::MULTI_KERNEL},
    {0},  // query size
    {0},
    {256},
    {1},
    {cuvs::distance::DistanceType::L2Expanded},
    {false},
    {true},
    {true},
    {0.995},
    {std::optional<float>{std::nullopt}},
    {std::optional<vpq_params>{std::nullopt}},
    {std::optional<bool>{std::nullopt}},
    {cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_PHYSICAL,
     cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_LOGICAL});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  // Varying dim and build algo.
  inputs2 = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {1000},
    {1, 3, 5, 7, 8, 17, 64, 128, 137, 192, 256, 512, 1024},  // dim
    {16},                                                    // k
    {32},                                                    // degree
    {graph_build_algo::IVF_PQ,
     graph_build_algo::NN_DESCENT,
     graph_build_algo::ITERATIVE_CAGRA_SEARCH},
    {search_algo::AUTO},
    {10},
    {0},
    {64},
    {1},
    {cuvs::distance::DistanceType::L2Expanded,
     cuvs::distance::DistanceType::InnerProduct,
     cuvs::distance::DistanceType::BitwiseHamming,
     cuvs::distance::DistanceType::L1},
    {false},
    {true},
    {false},
    {0.995},
    {std::optional<float>{std::nullopt}},
    {std::optional<vpq_params>{std::nullopt}},
    {std::optional<bool>{std::nullopt}},
    {cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_PHYSICAL,
     cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_LOGICAL});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  // Varying team_size, graph_build_algo
  inputs2 = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {1000},
    {64},
    {16},
    {32},  // degree
    {graph_build_algo::IVF_PQ,
     graph_build_algo::NN_DESCENT,
     graph_build_algo::ITERATIVE_CAGRA_SEARCH},
    {search_algo::AUTO},
    {10},
    {0},  // team_size
    {64},
    {1},
    {cuvs::distance::DistanceType::L2Expanded,
     cuvs::distance::DistanceType::InnerProduct,
     cuvs::distance::DistanceType::BitwiseHamming,
     cuvs::distance::DistanceType::CosineExpanded,
     cuvs::distance::DistanceType::L1},
    {false},
    {false},
    {false},
    {0.995},
    {std::optional<float>{std::nullopt}},
    {std::optional<vpq_params>{std::nullopt}},
    {std::optional<bool>{std::nullopt}},
    {cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_PHYSICAL,
     cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_LOGICAL});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  // Vary team size only.
  inputs2 = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {1000},
    {64},
    {16},
    {32},  // degree
    {graph_build_algo::NN_DESCENT},
    {search_algo::AUTO},
    {10},
    {8, 16, 32},  // team_size
    {64},
    {1},
    {cuvs::distance::DistanceType::L2Expanded,
     cuvs::distance::DistanceType::InnerProduct,
     cuvs::distance::DistanceType::BitwiseHamming,
     cuvs::distance::DistanceType::CosineExpanded,
     cuvs::distance::DistanceType::L1},
    {false},
    {false},
    {false},
    {0.995},
    {std::optional<float>{std::nullopt}},
    {std::optional<vpq_params>{std::nullopt}},
    {std::optional<bool>{std::nullopt}},
    {cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_PHYSICAL});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  // Varying n_rows, host_dataset
  inputs2 = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {10000},
    {32},
    {10},
    {32},  // degree
    {graph_build_algo::AUTO},
    {search_algo::AUTO},
    {10},
    {0},  // team_size
    {64},
    {1},
    {cuvs::distance::DistanceType::L2Expanded, cuvs::distance::DistanceType::InnerProduct},
    {false, true},
    {false},
    {true},
    {0.985},
    {std::optional<float>{std::nullopt}},
    {std::optional<vpq_params>{std::nullopt}},
    {std::optional<bool>{std::nullopt}},
    {cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_PHYSICAL,
     cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_LOGICAL});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  // A few PQ configurations.
  // Varying dim, vq_n_centers
  inputs2 = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {10000},
    {64, 128, 192, 256, 512, 1024},  // dim
    {16},                            // k
    {32},                            // degree
    {graph_build_algo::IVF_PQ},
    {search_algo::AUTO},
    {10},
    {0},
    {64},
    {1},
    {cuvs::distance::DistanceType::L2Expanded},
    {false},
    {true},
    {false},
    {0.6},
    {std::optional<float>{std::nullopt}},
    {std::optional<vpq_params>{std::nullopt}},
    {std::optional<bool>{std::nullopt}},
    {cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_PHYSICAL,
     cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_LOGICAL});  // don't demand high recall
                                                                // without refinement
  for (uint32_t pq_len : {2, 4, 8}) {
    for (uint32_t vq_n_centers : {100, 1000}) {
      for (auto internal_smem_dtype : {cuvs::neighbors::cagra::internal_dtype::E5M2,
                                       cuvs::neighbors::cagra::internal_dtype::F16}) {
        for (auto input : inputs2) {
          vpq_params ps{};
          ps.pq_dim       = input.dim / pq_len;
          ps.vq_n_centers = vq_n_centers;
          input.compression.emplace(ps);
          input.smem_dtype = internal_smem_dtype;
          inputs.push_back(input);
        }
      }
    }
  }

  // Refinement options
  // Varying host_dataset, ivf_pq_search_refine_ratio
  inputs2 = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {5000},
    {32, 64},
    {16},
    {32},  // degree
    {graph_build_algo::IVF_PQ},
    {search_algo::AUTO},
    {10},
    {0},  // team_size
    {64},
    {1},
    {cuvs::distance::DistanceType::L2Expanded, cuvs::distance::DistanceType::InnerProduct},
    {false, true},
    {false},
    {true},
    {0.99},
    {1.0f, 2.0f, 3.0f},
    {std::optional<vpq_params>{std::nullopt}},
    {std::optional<bool>{std::nullopt}},
    {cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_PHYSICAL,
     cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_LOGICAL});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  // Varying dim, adding non_owning_memory_buffer_flag
  inputs2 = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {1000},
    {1, 5, 8, 64, 137, 256, 619, 1024},  // dim
    {10},
    {32},  // degree
    {graph_build_algo::IVF_PQ},
    {search_algo::AUTO},
    {10},
    {0},  // team_size
    {64},
    {1},
    {cuvs::distance::DistanceType::L2Expanded},
    {false},
    {false},
    {false},
    {0.995},
    {std::optional<float>{std::nullopt}},
    {std::optional<vpq_params>{std::nullopt}},
    {std::optional<bool>{std::nullopt}},
    {cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_PHYSICAL,
     cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_LOGICAL});
  for (auto input : inputs2) {
    input.non_owning_memory_buffer_flag = true;
    inputs.push_back(input);
  }

  return inputs;
}

inline std::vector<AnnCagraInputs> generate_addnode_inputs()
{
  // changing dim
  std::vector<AnnCagraInputs> inputs =
    raft::util::itertools::product<AnnCagraInputs>({100},
                                                   {1000},
                                                   {1, 8, 17, 64, 128, 137, 512, 1024},  // dim
                                                   {16},                                 // k
                                                   {32, 47, 64},                         // degree
                                                   {graph_build_algo::ITERATIVE_CAGRA_SEARCH},
                                                   {search_algo::AUTO},
                                                   {10},
                                                   {0},
                                                   {64},
                                                   {1},
                                                   {cuvs::distance::DistanceType::L2Expanded,
                                                    cuvs::distance::DistanceType::InnerProduct,
                                                    cuvs::distance::DistanceType::BitwiseHamming,
                                                    cuvs::distance::DistanceType::L1},
                                                   {false},
                                                   {true},
                                                   {true},
                                                   {0.995});

  // testing host and device datasets
  auto inputs2 = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {10000},
    {32},
    {10},
    {32},  // degree
    {graph_build_algo::AUTO},
    {search_algo::AUTO},
    {10},
    {0},  // team_size
    {64},
    {1},
    {cuvs::distance::DistanceType::L2Expanded,
     cuvs::distance::DistanceType::InnerProduct,
     cuvs::distance::DistanceType::CosineExpanded},
    {false, true},
    {false},
    {false},
    {0.985},
    {std::optional<float>{std::nullopt}},
    {std::optional<vpq_params>{std::nullopt}},
    {std::optional<bool>{std::nullopt}},
    {cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_PHYSICAL});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  // a few PQ configurations
  inputs2 = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {10000},
    {192, 1024},  // dim
    {16},         // k
    {32},         // degree
    {graph_build_algo::IVF_PQ},
    {search_algo::AUTO},
    {10},
    {0},
    {64},
    {1},
    {cuvs::distance::DistanceType::L2Expanded},
    {false},
    {true},
    {true},
    {0.6});                      // don't demand high recall without refinement
  for (uint32_t pq_len : {2}) {  // for now, only pq_len = 2 is supported, more options coming soon
    for (uint32_t vq_n_centers : {100}) {
      for (auto input : inputs2) {
        vpq_params ps{};
        ps.pq_dim       = input.dim / pq_len;
        ps.vq_n_centers = vq_n_centers;
        input.compression.emplace(ps);
        inputs.push_back(input);
      }
    }
  }

  return inputs;
}

inline std::vector<AnnCagraInputs> generate_filtering_inputs()
{
  // Charge graph dim, search algo
  std::vector<AnnCagraInputs> inputs = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {1000},
    {1, 8, 17, 102},
    {16},          // k
    {32, 47, 64},  // degree
    {graph_build_algo::NN_DESCENT},
    {search_algo::SINGLE_CTA, search_algo::MULTI_CTA, search_algo::MULTI_KERNEL},
    {0},  // query size
    {0},
    {256},
    {1},
    {cuvs::distance::DistanceType::L2Expanded},
    {false},
    {true},
    {false},
    {0.995});

  // Fixed dim, and changing neighbors and query size (output matrix size)
  auto inputs2 = raft::util::itertools::product<AnnCagraInputs>(
    {1, 100},
    {1000},
    {8},
    {1, 16},  // k
    {32},     // degree
    {graph_build_algo::NN_DESCENT},
    {search_algo::SINGLE_CTA, search_algo::MULTI_CTA, search_algo::MULTI_KERNEL},
    {0},  // query size
    {0},
    {256},
    {1},
    {cuvs::distance::DistanceType::L2Expanded, cuvs::distance::DistanceType::InnerProduct},
    {false},
    {true},
    {false},
    {0.995});
  inputs.insert(inputs.end(), inputs2.begin(), inputs2.end());

  // a few PQ configurations
  inputs2 = raft::util::itertools::product<AnnCagraInputs>(
    {100},
    {10000},
    {256},  // dim
    {16},   // k
    {32},   // degree
    {graph_build_algo::IVF_PQ},
    {search_algo::AUTO},
    {10},
    {0},
    {64},
    {1},
    {cuvs::distance::DistanceType::L2Expanded},
    {false},
    {true},
    {true},
    {0.6});                      // don't demand high recall without refinement
  for (uint32_t pq_len : {2}) {  // for now, only pq_len = 2 is supported, more options coming soon
    for (uint32_t vq_n_centers : {100}) {
      for (auto input : inputs2) {
        vpq_params ps{};
        ps.pq_dim       = input.dim / pq_len;
        ps.vq_n_centers = vq_n_centers;
        input.compression.emplace(ps);
        inputs.push_back(input);
      }
    }
  }

  return inputs;
}
const std::vector<AnnCagraInputs> inputs           = generate_inputs();
const std::vector<AnnCagraInputs> inputs_addnode   = generate_addnode_inputs();
const std::vector<AnnCagraInputs> inputs_filtering = generate_filtering_inputs();

}  // namespace cuvs::neighbors::cagra
