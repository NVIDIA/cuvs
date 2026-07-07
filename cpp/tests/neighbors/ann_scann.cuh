/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
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
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/linalg/add.cuh>
#include <raft/matrix/gather.cuh>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <thrust/sequence.h>

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
    cuvs::preprocessing::quantize::pq::params pq_params;
    pq_params.pq_bits       = ps.index_params.pq_bits;
    pq_params.pq_dim        = num_subspaces;
    pq_params.use_subspaces = true;
    pq_params.use_vq        = true;  // SCANN uses centroids separately

    auto pq_codebook_copy = raft::make_device_matrix<float, uint32_t, raft::row_major>(
      handle_, idx.pq_codebook().extent(0), idx.pq_codebook().extent(1));
    raft::copy(pq_codebook_copy.data_handle(),
               idx.pq_codebook().data_handle(),
               idx.pq_codebook().size(),
               stream_);

    auto vq_codebook = raft::make_device_matrix<float, uint32_t, raft::row_major>(
      handle_, idx.centers().extent(0), idx.centers().extent(1));
    raft::copy(
      vq_codebook.data_handle(), idx.centers().data_handle(), idx.centers().size(), stream_);
    auto quantizer =
      cuvs::preprocessing::quantize::pq::build(handle_,
                                               pq_params,
                                               raft::make_const_mdspan(pq_codebook_copy.view()),
                                               raft::make_const_mdspan(vq_codebook.view()));

    auto quantized_residuals_device =
      raft::make_device_matrix<uint8_t, IdxT>(handle_, ps.num_db_vecs, num_subspaces);
    raft::copy(quantized_residuals_device.data_handle(),
               idx.quantized_residuals().data_handle(),
               idx.quantized_residuals().size(),
               stream_);

    // Re-pack 4-bit codes. The 8-bit codes are already in the right format
    auto codes_dim    = cuvs::preprocessing::quantize::pq::get_quantized_dim(pq_params);
    auto packed_codes = raft::make_device_matrix<uint8_t, IdxT>(handle_, ps.num_db_vecs, codes_dim);

    if (ps.index_params.pq_bits == 4) {
      raft::linalg::map_offset(
        handle_,
        packed_codes.view(),
        [qr_view = quantized_residuals_device.view(), num_subspaces, codes_dim] __device__(
          size_t i) {
          int64_t row_idx       = i / codes_dim;
          int64_t packed_idx    = i % codes_dim;
          int64_t code_idx      = packed_idx * 2;
          int64_t code_idx_next = code_idx + 1;

          uint8_t first_code = (code_idx < num_subspaces) ? qr_view(row_idx, code_idx) : 0;
          uint8_t second_code =
            (code_idx_next < num_subspaces) ? qr_view(row_idx, code_idx_next) : 0;

          return (first_code << 4) | (second_code & 0x0F);
        });
    } else {
      raft::copy(packed_codes.data_handle(),
                 quantized_residuals_device.data_handle(),
                 packed_codes.size(),
                 stream_);
    }

    auto reconstructed_vectors =
      raft::make_device_matrix<float, IdxT>(handle_, ps.num_db_vecs, ps.dim);
    auto reconstructed_vectors_view = reconstructed_vectors.view();
    cuvs::preprocessing::quantize::pq::inverse_transform(
      handle_,
      quantizer,
      raft::make_const_mdspan(packed_codes.view()),
      reconstructed_vectors_view,
      raft::make_const_mdspan(idx.labels()));

    // Compute L2 distances for reconstruction error
    auto database_view =
      raft::make_device_matrix_view<const float, int64_t>(database.data(), ps.num_db_vecs, ps.dim);
    auto distances = raft::make_device_vector<float, IdxT>(handle_, ps.num_db_vecs);
    raft::linalg::map_offset(
      handle_,
      distances.view(),
      [database_view, reconstructed_vectors_view, dim = ps.dim] __device__(IdxT i) {
        float dist = 0.0f;
        for (uint32_t j = 0; j < dim; j++) {
          float diff = database_view(i, j) - reconstructed_vectors_view(i, j);
          dist += diff * diff;
        }
        return sqrtf(dist / static_cast<float>(dim));
      });

    float max_allowed_error = 0.95f;
    auto distances_host     = raft::make_host_vector<float, IdxT>(handle_, ps.num_db_vecs);
    raft::copy(distances_host.data_handle(), distances.data_handle(), ps.num_db_vecs, stream_);
    raft::resource::sync_stream(handle_);

    float mean_error = 0.0f;
    float max_error  = 0.0f;
    for (IdxT i = 0; i < ps.num_db_vecs; i++) {
      mean_error += distances_host(i);
      max_error = std::max(max_error, distances_host(i));
    }
    mean_error /= static_cast<float>(ps.num_db_vecs);
    ASSERT_LT(mean_error, max_allowed_error)
      << "Mean reconstruction error too large: " << mean_error;
    ASSERT_LT(max_error, max_allowed_error * 1.5f)
      << "Max reconstruction error too large: " << max_error;
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
  rmm::cuda_stream_view stream_;
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
