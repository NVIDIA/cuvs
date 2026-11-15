/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../test_utils.cuh"
#include "ann_utils.cuh"
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/scann.hpp>
#include <cuvs/preprocessing/quantize/pq.hpp>

#include <algorithm>
#include <cmath>
#include <cuda/stream>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/map.cuh>
#include <raft/matrix/gather.cuh>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <thrust/sequence.h>
#include <vector>

namespace cuvs::neighbors::experimental::scann {

struct scann_inputs {
  uint32_t num_db_vecs = 4096;
  uint32_t dim         = 64;

  cuvs::neighbors::experimental::scann::index_params index_params;

  scann_inputs()
  {
    index_params.n_leaves            = max(32u, min(1024u, num_db_vecs / 128u));
    index_params.kmeans_n_rows_train = num_db_vecs;
    index_params.pq_n_rows_train     = num_db_vecs;
  }
};
inline ::std::ostream& operator<<(::std::ostream& os, const scann_inputs& p)
{
  os << "dataset shape=" << p.num_db_vecs << "x" << p.dim
     << ", n_leaves=" << p.index_params.n_leaves
     << ", partitioning_eta=" << p.index_params.partitioning_eta
     << ", soar_lambda=" << p.index_params.soar_lambda << ", pq_dim=" << p.index_params.pq_dim
     << ", pq_bits=" << p.index_params.pq_bits
     << ", reordering_bf16=" << p.index_params.reordering_bf16
     << ", reordering_noise_shaping_threshold="
     << p.index_params.reordering_noise_shaping_threshold;

  return os;
}

template <typename DataT, typename IdxT>
class scann_test : public ::testing::TestWithParam<scann_inputs> {
 public:
  scann_test()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<scann_inputs>::GetParam()),
      database(0, stream_)
  {
  }

  void gen_data()
  {
    database.resize(size_t{ps.num_db_vecs} * size_t{ps.dim}, stream_);

    raft::random::RngState r(1234ULL);

    if constexpr (std::is_same<DataT, float>{}) {
      raft::random::uniform(
        handle_, r, database.data(), ps.num_db_vecs * ps.dim, DataT(0.1), DataT(2.0));
    } else {
      raft::random::uniformInt(
        handle_, r, database.data(), ps.num_db_vecs * ps.dim, DataT(1), DataT(20));
    }

    raft::resource::sync_stream(handle_);
  }

  auto build_only()
  {
    auto ipams = ps.index_params;

    auto db_view =
      raft::make_device_matrix_view<const DataT, int64_t>(database.data(), ps.num_db_vecs, ps.dim);

    return cuvs::neighbors::experimental::scann::build(handle_, ipams, db_view);
  }

  auto build_only_host_input()
  {
    auto ipams = ps.index_params;

    auto h_database = raft::make_host_matrix<DataT, int64_t>(ps.num_db_vecs, ps.dim);

    raft::copy(h_database.data_handle(), database.data(), ps.num_db_vecs * ps.dim, stream_);

    auto db_view = raft::make_host_matrix_view<const DataT, int64_t>(
      h_database.data_handle(), ps.num_db_vecs, ps.dim);

    return cuvs::neighbors::experimental::scann::build(handle_, ipams, db_view);
  }

  auto build_only_host_input_overlap()
  {
    auto ipams = ps.index_params;

    // additional stream for overlapping HtoD copy
    size_t n_streams = 2;
    raft::resource::set_cuda_stream_pool(handle_,
                                         std::make_shared<rmm::cuda_stream_pool>(n_streams));

    auto h_database = raft::make_host_matrix<DataT, int64_t>(ps.num_db_vecs, ps.dim);

    raft::copy(h_database.data_handle(), database.data(), ps.num_db_vecs * ps.dim, stream_);

    auto db_view = raft::make_host_matrix_view<const DataT, int64_t>(
      h_database.data_handle(), ps.num_db_vecs, ps.dim);

    return cuvs::neighbors::experimental::scann::build(handle_, ipams, db_view);
  }

  template <typename BuildIndex>
  void run(BuildIndex build_index)
  {
    index<DataT, IdxT> index = build_index();

    // Simple checking of dimensions of index artifacts
    auto num_subspaces   = ps.dim / ps.index_params.pq_dim;
    auto num_pq_clusters = 1 << ps.index_params.pq_bits;

    ASSERT_EQ(index.quantized_residuals().extent(0), ps.num_db_vecs);
    ASSERT_EQ(index.quantized_residuals().extent(1), num_subspaces);

    ASSERT_EQ(index.quantized_soar_residuals().extent(0), ps.num_db_vecs);
    ASSERT_EQ(index.quantized_soar_residuals().extent(1), num_subspaces);

    ASSERT_EQ(index.pq_codebook().extent(0), num_pq_clusters);
    ASSERT_EQ(index.pq_codebook().extent(1), ps.dim);

    IdxT expected_bf16_size = ps.index_params.reordering_bf16 ? ps.dim * ps.num_db_vecs : 0;

    ASSERT_EQ(index.bf16_dataset().size(), expected_bf16_size);
    check_code_validity(index, num_subspaces, num_pq_clusters);
    check_reconstruction(index, num_subspaces);
  }

  void check_code_validity(const index<DataT, IdxT>& idx, int num_subspaces, int num_pq_clusters)
  {
    auto quant_res_host =
      raft::make_host_matrix<uint8_t, IdxT>(handle_, ps.num_db_vecs, num_subspaces);
    auto quant_soar_host =
      raft::make_host_matrix<uint8_t, IdxT>(handle_, ps.num_db_vecs, num_subspaces);

    raft::copy(quant_res_host.data_handle(),
               idx.quantized_residuals().data_handle(),
               idx.quantized_residuals().size(),
               stream_);
    raft::copy(quant_soar_host.data_handle(),
               idx.quantized_soar_residuals().data_handle(),
               idx.quantized_soar_residuals().size(),
               stream_);
    raft::resource::sync_stream(handle_);

    bool all_zeros       = true;
    auto n_vecs_to_check = std::min(ps.num_db_vecs, 50u);
    for (IdxT i = 0; i < n_vecs_to_check * num_subspaces; i++) {
      if (quant_res_host.data_handle()[i] != 0) { all_zeros = false; }
      if (quant_soar_host.data_handle()[i] != 0) { all_zeros = false; }
      // Check that unpacked codes are in valid range
      if (ps.index_params.pq_bits == 4) {
        ASSERT_LT(quant_res_host.data_handle()[i], num_pq_clusters)
          << "AVQ quantized code out of range at position " << i;
        ASSERT_LT(quant_soar_host.data_handle()[i], num_pq_clusters)
          << "SOAR quantized code out of range at position " << i;
      }
    }
    ASSERT_FALSE(all_zeros) << "Quantized output contains all zeros";
  }

  void check_reconstruction(const index<DataT, IdxT>& idx, int num_subspaces)
  {
    const int64_t n_rows      = ps.num_db_vecs;
    const int64_t dim         = ps.dim;
    const int64_t sub_dim     = ps.index_params.pq_dim;
    const int64_t n_codes     = idx.pq_codebook().extent(0);
    const int64_t n_leaves    = idx.centers().extent(0);
    const int64_t n_subspaces = num_subspaces;

    ASSERT_EQ(static_cast<int64_t>(idx.pq_codebook().extent(1)), dim);
    ASSERT_EQ(static_cast<int64_t>(idx.centers().extent(1)), dim);
    ASSERT_EQ(sub_dim * n_subspaces, dim);

    // Check codes and labels for out-of-range values
    {
      auto h_codes  = raft::make_host_matrix<uint8_t, int64_t>(handle_, n_rows, n_subspaces);
      auto h_labels = raft::make_host_vector<uint32_t, int64_t>(handle_, n_rows);
      raft::copy(
        h_codes.data_handle(), idx.quantized_residuals().data_handle(), h_codes.size(), stream_);
      raft::copy(h_labels.data_handle(), idx.labels().data_handle(), h_labels.size(), stream_);
      raft::resource::sync_stream(handle_);
      for (int64_t r = 0; r < n_rows; r++) {
        ASSERT_LT(h_labels(r), n_leaves) << "Label out of range at row " << r;
        for (int64_t s = 0; s < n_subspaces; s++) {
          ASSERT_LT(h_codes(r, s), n_codes)
            << "PQ code out of range at row " << r << ", subspace " << s;
        }
      }
    }

    auto codes = raft::make_device_matrix<uint8_t, int64_t>(handle_, n_rows, n_subspaces);
    raft::copy(codes.data_handle(), idx.quantized_residuals().data_handle(), codes.size(), stream_);

    auto codes_view    = codes.view();
    auto codebook_view = idx.pq_codebook();
    auto centers_view  = idx.centers();
    auto labels_view   = idx.labels();

    // Decode x_hat[row, d] = centers[labels[row], d] + codebook[code[row, d / pq_dim], d]
    auto reconstructed = raft::make_device_matrix<float, int64_t>(handle_, n_rows, dim);
    raft::linalg::map_offset(
      handle_,
      reconstructed.view(),
      [codes_view, codebook_view, centers_view, labels_view, dim, sub_dim] __device__(size_t i) {
        int64_t row  = static_cast<int64_t>(i) / dim;
        int64_t d    = static_cast<int64_t>(i) % dim;
        int64_t code = codes_view(row, d / sub_dim);
        return centers_view(labels_view(row), d) + codebook_view(code, d);
      });
    auto reconstructed_view = reconstructed.view();

    auto database_view =
      raft::make_device_matrix_view<const DataT, int64_t>(database.data(), n_rows, dim);
    auto norms      = raft::make_device_vector<float, int64_t>(handle_, n_rows);
    auto norms_view = norms.view();

    // Per-vector relative error ||x - x_hat|| / ||x|| of the reconstructed vector
    auto errors = raft::make_device_vector<float, int64_t>(handle_, n_rows);
    raft::linalg::map_offset(
      handle_,
      errors.view(),
      [database_view, reconstructed_view, norms_view, dim] __device__(int64_t r) {
        double sq_err = 0.0;
        double norm   = 0.0;
        for (int64_t k = 0; k < dim; k++) {
          double x   = database_view(r, k);
          double err = x - reconstructed_view(r, k);
          sq_err += err * err;
          norm += x * x;
        }
        norm          = sqrtf(norm);
        norms_view(r) = norm;
        return sqrtf(sq_err) / norm;
      });

    auto errors_host = raft::make_host_vector<float, int64_t>(handle_, n_rows);
    raft::copy(errors_host.data_handle(), errors.data_handle(), n_rows, stream_);
    raft::resource::sync_stream(handle_);

    double mean_error = 0.0;
    for (int64_t r = 0; r < n_rows; r++) {
      mean_error += errors_host(r);
    }
    mean_error /= static_cast<double>(n_rows);

    const double max_allowed_mean = (ps.index_params.pq_bits == 4) ? 0.45 : 0.275;
    ASSERT_LT(mean_error, max_allowed_mean)
      << "Mean relative reconstruction error too large: " << mean_error;
  }

  void SetUp() override  // NOLINT
  {
    gen_data();
  }

  void TearDown() override  // NOLINT
  {
    cudaGetLastError();
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  cuda::stream_ref stream_;
  scann_inputs ps;                      // NOLINT
  rmm::device_uvector<DataT> database;  // NOLINT
};

/* Test cases */
using test_cases_t = std::vector<scann_inputs>;

// concatenate parameter sets for different type
template <typename T>
auto operator+(const std::vector<T>& a, const std::vector<T>& b) -> std::vector<T>
{
  std::vector<T> res = a;
  res.insert(res.end(), b.begin(), b.end());
  return res;
}

template <typename B, typename A, typename F>
auto map(const std::vector<A>& xs, F f) -> std::vector<B>
{
  std::vector<B> ys(xs.size());
  std::transform(xs.begin(), xs.end(), ys.begin(), f);
  return ys;
}

inline auto with_dims(const std::vector<uint32_t>& dims) -> test_cases_t
{
  return map<scann_inputs>(dims, [](uint32_t d) {
    scann_inputs x;
    x.dim = d;
    return x;
  });
}

inline auto defaults() -> test_cases_t { return {scann_inputs{}}; }

inline auto small_dims_all_pq_bits() -> test_cases_t
{
  auto four_bit_ts = with_dims({2, 4, 6, 8, 10, 12, 14, 16, 18});

  for (auto& ts : four_bit_ts) {
    ts.index_params.pq_dim  = 2;
    ts.index_params.pq_bits = 4;
  }

  auto eight_bit_ts = with_dims({2, 4, 6, 8, 10, 12, 14, 16, 18});

  for (auto& ts : eight_bit_ts) {
    ts.index_params.pq_dim  = 2;
    ts.index_params.pq_bits = 8;
  }

  return four_bit_ts + eight_bit_ts;
}

inline auto big_dims_all_pq_bits() -> test_cases_t
{
  auto four_bit_ts = with_dims({64, 128, 256, 512, 1024, 2048});

  for (auto& ts : four_bit_ts) {
    ts.index_params.pq_dim  = 8;
    ts.index_params.pq_bits = 4;
  }

  auto eight_bit_ts = with_dims({64, 128, 256, 512, 1024, 2048});

  for (auto& ts : eight_bit_ts) {
    ts.index_params.pq_dim  = 8;
    ts.index_params.pq_bits = 8;
  }

  return four_bit_ts + eight_bit_ts;
}

inline auto bf16() -> test_cases_t
{
  scann_inputs ts;
  ts.index_params.reordering_bf16 = true;

  return {ts};
}

inline auto bf16_avq() -> test_cases_t
{
  scann_inputs ts;
  ts.index_params.reordering_bf16                    = true;
  ts.index_params.reordering_noise_shaping_threshold = 0.2;

  return {ts};
}

inline auto avq() -> test_cases_t
{
  scann_inputs ts;
  ts.index_params.partitioning_eta = 2;

  return {ts};
}

inline auto soar() -> test_cases_t
{
  scann_inputs ts;
  ts.index_params.soar_lambda = 1.5;

  return {ts};
}

/* Test instantiations */

#define TEST_BUILD(type)                                \
  TEST_P(type, build) /* NOLINT */                      \
  {                                                     \
    this->run([this]() { return this->build_only(); }); \
  }

#define TEST_BUILD_HOST_INPUT(type)                                \
  TEST_P(type, build_host_input) /* NOLINT */                      \
  {                                                                \
    this->run([this]() { return this->build_only_host_input(); }); \
  }

#define TEST_BUILD_HOST_INPUT_OVERLAP(type)                                \
  TEST_P(type, build_host_input_overlap) /* NOLINT */                      \
  {                                                                        \
    this->run([this]() { return this->build_only_host_input_overlap(); }); \
  }

#define INSTANTIATE(type, vals) \
  INSTANTIATE_TEST_SUITE_P(ScaNN, type, ::testing::ValuesIn(vals)); /* NOLINT */

}  // namespace cuvs::neighbors::experimental::scann
