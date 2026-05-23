/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "knn_utils.cuh"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/brute_force.hpp>

#include <raft/core/host_mdspan.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/copy.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/random/rmat_rectangular_generator.cuh>
#include <raft/random/rng.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/util/popc.cuh>

#include <cusparse.h>
#include <gtest/gtest.h>

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace cuvs::neighbors::brute_force {

template <typename index_t>
struct PrefilteredBruteForceInputs {
  index_t n_queries;
  index_t n_dataset;
  index_t dim;
  index_t top_k;
  float sparsity;
  cuvs::distance::DistanceType metric;
  bool select_min = true;
};

template <typename T>
struct CompareApproxWithInf {
  CompareApproxWithInf(T eps_) : eps(eps_) {}
  bool operator()(const T& a, const T& b) const
  {
    if (std::isinf(a) && std::isinf(b)) return true;
    T diff  = std::abs(a - b);
    T m     = std::max(std::abs(a), std::abs(b));
    T ratio = diff > eps ? diff / m : diff;

    return (ratio <= eps);
  }

 private:
  T eps;
};

template <typename OutT, typename InT>
RAFT_KERNEL normalize_kernel(
  OutT* theta, const InT* in_vals, size_t max_scale, size_t r_scale, size_t c_scale)
{
  size_t idx = threadIdx.x;
  if (idx < max_scale) {
    auto a   = OutT(in_vals[4 * idx]);
    auto b   = OutT(in_vals[4 * idx + 1]);
    auto c   = OutT(in_vals[4 * idx + 2]);
    auto d   = OutT(in_vals[4 * idx + 3]);
    auto sum = a + b + c + d;
    a /= sum;
    b /= sum;
    c /= sum;
    d /= sum;
    theta[4 * idx]     = a;
    theta[4 * idx + 1] = b;
    theta[4 * idx + 2] = c;
    theta[4 * idx + 3] = d;
  }
}

struct float_to_half {
  __host__ __device__ __half operator()(const float x) const { return __float2half(x); }
};

template <typename OutT, typename InT>
void normalize(OutT* theta,
               const InT* in_vals,
               size_t max_scale,
               size_t r_scale,
               size_t c_scale,
               bool handle_rect,
               bool theta_array,
               cudaStream_t stream)
{
  normalize_kernel<OutT, InT><<<1, 256, 0, stream>>>(theta, in_vals, max_scale, r_scale, c_scale);
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename index_t, typename bitmap_t = uint32_t>
RAFT_KERNEL set_bitmap_kernel(
  const index_t* src, const index_t* dst, bitmap_t* bitmap, index_t n_edges, index_t n_cols)
{
  size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < n_edges) {
    index_t row      = src[idx];
    index_t col      = dst[idx];
    index_t g_idx    = row * n_cols + col;
    index_t item_idx = (g_idx) >> 5;
    uint32_t bit_idx = (g_idx) & 31;
    atomicOr(bitmap + item_idx, (uint32_t(1) << bit_idx));
  }
}

template <typename index_t, typename bitmap_t = uint32_t>
void set_bitmap(const index_t* src,
                const index_t* dst,
                bitmap_t* bitmap,
                index_t n_edges,
                index_t n_cols,
                cudaStream_t stream)
{
  int block_size = 256;
  int blocks     = raft::ceildiv<index_t>(n_edges, block_size);
  set_bitmap_kernel<index_t, bitmap_t>
    <<<blocks, block_size, 0, stream>>>(src, dst, bitmap, n_edges, n_cols);
  RAFT_CUDA_TRY(cudaGetLastError());
}

bool isCuSparseVersionGreaterThan_12_0_1()
{
  int version;
  cusparseHandle_t handle;
  cusparseCreate(&handle);
  cusparseGetVersion(handle, &version);

  int major = version / 1000;
  int minor = (version % 1000) / 100;
  int patch = version % 100;

  cusparseDestroy(handle);

  return (major > 12) || (major == 12 && minor > 0) || (major == 12 && minor == 0 && patch >= 2);
}

template <typename value_t, typename dist_t, typename index_t, typename bitmap_t = uint32_t>
class PrefilteredBruteForceOnBitmapTest
  : public ::testing::TestWithParam<PrefilteredBruteForceInputs<index_t>> {
 public:
  PrefilteredBruteForceOnBitmapTest()
    : stream(raft::resource::get_cuda_stream(handle)),
      params(::testing::TestWithParam<PrefilteredBruteForceInputs<index_t>>::GetParam()),
      filter_d(0, stream),
      dataset_d(0, stream),
      queries_d(0, stream),
      out_val_d(0, stream),
      out_val_expected_d(0, stream),
      out_idx_d(0, stream),
      out_idx_expected_d(0, stream)
  {
  }

 protected:
  index_t create_sparse_matrix_with_rmat(index_t m,
                                         index_t n,
                                         float sparsity,
                                         rmm::device_uvector<bitmap_t>& filter_d)
  {
    index_t r_scale   = (index_t)std::log2(m);
    index_t c_scale   = (index_t)std::log2(n);
    index_t n_edges   = (index_t)(m * n * 1.0f * sparsity);
    index_t max_scale = std::max(r_scale, c_scale);

    rmm::device_uvector<index_t> out_src{(unsigned long)n_edges, stream};
    rmm::device_uvector<index_t> out_dst{(unsigned long)n_edges, stream};
    rmm::device_uvector<float> theta{(unsigned long)(4 * max_scale), stream};

    raft::random::RngState state{2024ULL, raft::random::GeneratorType::GenPC};

    raft::random::uniform<float>(handle, state, theta.data(), theta.size(), 0.0f, 1.0f);
    normalize<float, float>(
      theta.data(), theta.data(), max_scale, r_scale, c_scale, r_scale != c_scale, true, stream);
    raft::random::rmat_rectangular_gen((index_t*)nullptr,
                                       out_src.data(),
                                       out_dst.data(),
                                       theta.data(),
                                       r_scale,
                                       c_scale,
                                       n_edges,
                                       stream,
                                       state);

    index_t nnz_h = 0;
    {
      auto src    = out_src.data();
      auto dst    = out_dst.data();
      auto bitmap = filter_d.data();
      rmm::device_scalar<index_t> nnz(0, stream);
      auto nnz_view = raft::make_device_scalar_view<index_t>(nnz.data());
      auto filter_view =
        raft::make_device_vector_view<const uint32_t, index_t>(filter_d.data(), filter_d.size());
      index_t size_h = m * n;
      auto size_view = raft::make_host_scalar_view<const index_t, index_t>(&size_h);

      set_bitmap(src, dst, bitmap, n_edges, n, stream);

      raft::popc(handle, filter_view, size_view, nnz_view);
      raft::copy(&nnz_h, nnz.data(), 1, stream);

      raft::resource::sync_stream(handle, stream);
    }

    return nnz_h;
  }

  void cpu_convert_to_csr(std::vector<bitmap_t>& bitmap,
                          index_t rows,
                          index_t cols,
                          std::vector<index_t>& indices,
                          std::vector<index_t>& indptr)
  {
    index_t offset_indptr   = 0;
    index_t offset_values   = 0;
    indptr[offset_indptr++] = 0;

    index_t index        = 0;
    bitmap_t element     = 0;
    index_t bit_position = 0;

    for (index_t i = 0; i < rows; ++i) {
      for (index_t j = 0; j < cols; ++j) {
        index        = i * cols + j;
        element      = bitmap[index / (8 * sizeof(bitmap_t))];
        bit_position = index % (8 * sizeof(bitmap_t));

        if (((element >> bit_position) & 1)) {
          indices[offset_values] = static_cast<index_t>(j);
          offset_values++;
        }
      }
      indptr[offset_indptr++] = static_cast<index_t>(offset_values);
    }
  }

  void cpu_sddmm(const std::vector<dist_t>& A,
                 const std::vector<dist_t>& B,
                 std::vector<dist_t>& vals,
                 const std::vector<index_t>& cols,
                 const std::vector<index_t>& row_ptrs,
                 bool is_row_major_A,
                 bool is_row_major_B,
                 dist_t alpha = 1.0,
                 dist_t beta  = 0.0)
  {
    if (params.n_queries * params.dim != static_cast<index_t>(A.size()) ||
        params.dim * params.n_dataset != static_cast<index_t>(B.size())) {
      std::cerr << "Matrix dimensions and vector size do not match!" << std::endl;
      return;
    }

    bool trans_a = is_row_major_A;
    bool trans_b = is_row_major_B;

    for (index_t i = 0; i < params.n_queries; ++i) {
      for (index_t j = row_ptrs[i]; j < row_ptrs[i + 1]; ++j) {
        dist_t sum     = 0;
        dist_t norms_A = 0;
        dist_t norms_B = 0;

        for (index_t l = 0; l < params.dim; ++l) {
          index_t a_index = trans_a ? i * params.dim + l : l * params.n_queries + i;
          index_t b_index = trans_b ? l * params.n_dataset + cols[j] : cols[j] * params.dim + l;
          dist_t A_v;
          dist_t B_v;
          if constexpr (sizeof(value_t) == 2) {
            A_v = __half2float(__float2half(A[a_index]));
            B_v = __half2float(__float2half(B[b_index]));
          } else {
            A_v = A[a_index];
            B_v = B[b_index];
          }

          sum += A_v * B_v;

          norms_A += A_v * A_v;
          norms_B += B_v * B_v;
        }
        vals[j] = alpha * sum + beta * vals[j];
        if (params.metric == cuvs::distance::DistanceType::L2Expanded) {
          vals[j] = dist_t(-2.0) * vals[j] + norms_A + norms_B;
        } else if (params.metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
          vals[j] = std::sqrt(dist_t(-2.0) * vals[j] + norms_A + norms_B);
        } else if (params.metric == cuvs::distance::DistanceType::CosineExpanded) {
          vals[j] = dist_t(1.0) - vals[j] / std::sqrt(norms_A * norms_B);
        }
      }
    }
  }

  void cpu_select_k(const std::vector<index_t>& indptr_h,
                    const std::vector<index_t>& indices_h,
                    const std::vector<dist_t>& values_h,
                    std::optional<std::vector<index_t>>& in_idx_h,
                    index_t n_queries,
                    index_t n_dataset,
                    index_t top_k,
                    std::vector<dist_t>& out_values_h,
                    std::vector<index_t>& out_indices_h,
                    bool select_min = true)
  {
    auto comp = [select_min](const std::pair<dist_t, index_t>& a,
                             const std::pair<dist_t, index_t>& b) {
      return select_min ? a.first < b.first : a.first >= b.first;
    };

    for (index_t row = 0; row < n_queries; ++row) {
      std::priority_queue<std::pair<dist_t, index_t>,
                          std::vector<std::pair<dist_t, index_t>>,
                          decltype(comp)>
        pq(comp);
      for (index_t idx = indptr_h[row]; idx < indptr_h[row + 1]; ++idx) {
        pq.push({values_h[idx], (in_idx_h.has_value()) ? (*in_idx_h)[idx] : indices_h[idx]});
        if (pq.size() > size_t(top_k)) { pq.pop(); }
      }

      std::vector<std::pair<dist_t, index_t>> row_pairs;
      while (!pq.empty()) {
        row_pairs.push_back(pq.top());
        pq.pop();
      }

      if (select_min) {
        std::sort(row_pairs.begin(), row_pairs.end(), [](const auto& a, const auto& b) {
          return a.first <= b.first;
        });
      } else {
        std::sort(row_pairs.begin(), row_pairs.end(), [](const auto& a, const auto& b) {
          return a.first >= b.first;
        });
      }
      for (index_t col = 0; col < top_k; col++) {
        if (col < index_t(row_pairs.size())) {
          out_values_h[row * top_k + col]  = row_pairs[col].first;
          out_indices_h[row * top_k + col] = row_pairs[col].second;
        }
      }
    }
  }

  void SetUp() override
  {
    if (std::is_same_v<value_t, half> && !isCuSparseVersionGreaterThan_12_0_1()) {
      GTEST_SKIP() << "Skipping all tests for half-float as cuSparse doesn't support it.";
    }
    index_t element =
      raft::ceildiv(params.n_queries * params.n_dataset, index_t(sizeof(bitmap_t) * 8));
    std::vector<bitmap_t> filter_h(element);
    filter_d.resize(element, stream);

    nnz =
      create_sparse_matrix_with_rmat(params.n_queries, params.n_dataset, params.sparsity, filter_d);

    raft::update_host(filter_h.data(), filter_d.data(), filter_d.size(), stream);
    raft::resource::sync_stream(handle, stream);

    index_t dataset_size = params.n_dataset * params.dim;
    index_t queries_size = params.n_queries * params.dim;

    std::vector<dist_t> dataset_h(dataset_size);
    std::vector<dist_t> queries_h(queries_size);

    dataset_d.resize(dataset_size, stream);
    queries_d.resize(queries_size, stream);

    auto blobs_in_val =
      raft::make_device_matrix<dist_t, index_t>(handle, 1, dataset_size + queries_size);
    auto labels = raft::make_device_vector<index_t, index_t>(handle, 1);

    if constexpr (!std::is_same_v<value_t, half>) {
      raft::random::make_blobs<value_t, index_t>(blobs_in_val.data_handle(),
                                                 labels.data_handle(),
                                                 1,
                                                 dataset_size + queries_size,
                                                 1,
                                                 stream,
                                                 false,
                                                 nullptr,
                                                 nullptr,
                                                 value_t(1.0),
                                                 false,
                                                 value_t(-1.0f),
                                                 value_t(1.0f),
                                                 uint64_t(2024));
    } else {
      raft::random::make_blobs<dist_t, index_t>(blobs_in_val.data_handle(),
                                                labels.data_handle(),
                                                1,
                                                dataset_size + queries_size,
                                                1,
                                                stream,
                                                false,
                                                nullptr,
                                                nullptr,
                                                dist_t(1.0),
                                                false,
                                                dist_t(-1.0f),
                                                dist_t(1.0f),
                                                uint64_t(2024));
    }

    raft::copy(dataset_h.data(), blobs_in_val.data_handle(), dataset_size, stream);

    if constexpr (std::is_same_v<value_t, half>) {
      raft::linalg::unaryOp(
        dataset_d.data(), blobs_in_val.data_handle(), dataset_size, float_to_half(), stream);
    } else {
      raft::copy(dataset_d.data(), blobs_in_val.data_handle(), dataset_size, stream);
    }

    raft::copy(queries_h.data(), blobs_in_val.data_handle() + dataset_size, queries_size, stream);
    if constexpr (std::is_same_v<value_t, half>) {
      raft::linalg::unaryOp(queries_d.data(),
                            blobs_in_val.data_handle() + dataset_size,
                            queries_size,
                            float_to_half(),
                            stream);
    } else {
      raft::copy(queries_d.data(), blobs_in_val.data_handle() + dataset_size, queries_size, stream);
    }

    raft::resource::sync_stream(handle);

    std::vector<dist_t> values_h(nnz);
    std::vector<index_t> indices_h(nnz);
    std::vector<index_t> indptr_h(params.n_queries + 1);

    cpu_convert_to_csr(filter_h, params.n_queries, params.n_dataset, indices_h, indptr_h);

    cpu_sddmm(queries_h, dataset_h, values_h, indices_h, indptr_h, true, false);

    bool select_min = cuvs::distance::is_min_close(params.metric);

    std::vector<dist_t> out_val_h(
      params.n_queries * params.top_k,
      select_min ? std::numeric_limits<dist_t>::infinity() : std::numeric_limits<dist_t>::lowest());
    std::vector<index_t> out_idx_h(params.n_queries * params.top_k, static_cast<index_t>(0));

    out_val_d.resize(params.n_queries * params.top_k, stream);
    out_idx_d.resize(params.n_queries * params.top_k, stream);

    raft::update_device(out_val_d.data(), out_val_h.data(), out_val_h.size(), stream);
    raft::update_device(out_idx_d.data(), out_idx_h.data(), out_idx_h.size(), stream);

    raft::resource::sync_stream(handle);

    std::optional<std::vector<index_t>> optional_indices_h = std::nullopt;
    cpu_select_k(indptr_h,
                 indices_h,
                 values_h,
                 optional_indices_h,
                 params.n_queries,
                 params.n_dataset,
                 params.top_k,
                 out_val_h,
                 out_idx_h,
                 select_min);
    out_val_expected_d.resize(params.n_queries * params.top_k, stream);
    out_idx_expected_d.resize(params.n_queries * params.top_k, stream);

    raft::update_device(out_val_expected_d.data(), out_val_h.data(), out_val_h.size(), stream);
    raft::update_device(out_idx_expected_d.data(), out_idx_h.data(), out_idx_h.size(), stream);

    raft::resource::sync_stream(handle);
  }

  void Run()
  {
    auto dataset_raw = raft::make_device_matrix_view<const value_t, index_t, raft::row_major>(
      (const value_t*)dataset_d.data(), params.n_dataset, params.dim);

    auto queries = raft::make_device_matrix_view<const value_t, index_t, raft::row_major>(
      (const value_t*)queries_d.data(), params.n_queries, params.dim);

    auto dataset = brute_force::build(handle, dataset_raw, params.metric);

    auto filter = cuvs::core::bitmap_view<bitmap_t, index_t>(
      (bitmap_t*)filter_d.data(), params.n_queries, params.n_dataset);

    auto out_val = raft::make_device_matrix_view<dist_t, index_t, raft::row_major>(
      out_val_d.data(), params.n_queries, params.top_k);
    auto out_idx = raft::make_device_matrix_view<index_t, index_t, raft::row_major>(
      out_idx_d.data(), params.n_queries, params.top_k);

    brute_force::search(handle,
                        dataset,
                        queries,
                        out_idx,
                        out_val,
                        cuvs::neighbors::filtering::bitmap_filter(filter));
    std::vector<dist_t> out_val_h(params.n_queries * params.top_k,
                                  std::numeric_limits<dist_t>::infinity());

    raft::update_host(out_val_h.data(), out_val_d.data(), out_val_h.size(), stream);
    raft::resource::sync_stream(handle);

    ASSERT_TRUE(cuvs::neighbors::devArrMatchKnnPair(out_idx_expected_d.data(),
                                                    out_idx.data_handle(),
                                                    out_val_expected_d.data(),
                                                    out_val.data_handle(),
                                                    params.n_queries,
                                                    params.top_k,
                                                    0.001f,
                                                    stream,
                                                    true));
  }

  // Same as Run(), but passes the true sparsity as a filtering_rate hint to the
  // params-taking search overload. Confirms results match auto-detection.
  void RunWithFilteringRateHint()
  {
    auto dataset_raw = raft::make_device_matrix_view<const value_t, index_t, raft::row_major>(
      (const value_t*)dataset_d.data(), params.n_dataset, params.dim);

    auto queries = raft::make_device_matrix_view<const value_t, index_t, raft::row_major>(
      (const value_t*)queries_d.data(), params.n_queries, params.dim);

    auto dataset = brute_force::build(handle, dataset_raw, params.metric);

    auto filter = cuvs::core::bitmap_view<bitmap_t, index_t>(
      (bitmap_t*)filter_d.data(), params.n_queries, params.n_dataset);

    auto out_val = raft::make_device_matrix_view<dist_t, index_t, raft::row_major>(
      out_val_d.data(), params.n_queries, params.top_k);
    auto out_idx = raft::make_device_matrix_view<index_t, index_t, raft::row_major>(
      out_idx_d.data(), params.n_queries, params.top_k);

    cuvs::neighbors::brute_force::search_params search_params;
    search_params.filtering_rate = params.sparsity;

    brute_force::search(handle,
                        search_params,
                        dataset,
                        queries,
                        out_idx,
                        out_val,
                        cuvs::neighbors::filtering::bitmap_filter(filter));

    raft::resource::sync_stream(handle);

    ASSERT_TRUE(cuvs::neighbors::devArrMatchKnnPair(out_idx_expected_d.data(),
                                                    out_idx.data_handle(),
                                                    out_val_expected_d.data(),
                                                    out_val.data_handle(),
                                                    params.n_queries,
                                                    params.top_k,
                                                    0.001f,
                                                    stream,
                                                    true));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  PrefilteredBruteForceInputs<index_t> params;

  index_t nnz;

  rmm::device_uvector<value_t> dataset_d;
  rmm::device_uvector<value_t> queries_d;
  rmm::device_uvector<bitmap_t> filter_d;

  rmm::device_uvector<dist_t> out_val_d;
  rmm::device_uvector<dist_t> out_val_expected_d;

  rmm::device_uvector<index_t> out_idx_d;
  rmm::device_uvector<index_t> out_idx_expected_d;
};

template <typename value_t, typename dist_t, typename index_t, typename bitset_t = uint32_t>
class PrefilteredBruteForceOnBitsetTest
  : public ::testing::TestWithParam<PrefilteredBruteForceInputs<index_t>> {
 public:
  PrefilteredBruteForceOnBitsetTest()
    : stream(raft::resource::get_cuda_stream(handle)),
      params(::testing::TestWithParam<PrefilteredBruteForceInputs<index_t>>::GetParam()),
      filter_d(0, stream),
      dataset_d(0, stream),
      queries_d(0, stream),
      out_val_d(0, stream),
      out_val_expected_d(0, stream),
      out_idx_d(0, stream),
      out_idx_expected_d(0, stream)
  {
  }

 protected:
  void repeat_cpu_bitset(std::vector<bitset_t>& input,
                         size_t input_bits,
                         size_t repeat,
                         std::vector<bitset_t>& output)
  {
    const size_t output_bits  = input_bits * repeat;
    const size_t output_units = (output_bits + sizeof(bitset_t) * 8 - 1) / (sizeof(bitset_t) * 8);

    std::memset(output.data(), 0, output_units * sizeof(bitset_t));

    size_t output_bit_index = 0;

    for (size_t r = 0; r < repeat; ++r) {
      for (size_t i = 0; i < input_bits; ++i) {
        size_t input_unit_index = i / (sizeof(bitset_t) * 8);
        size_t input_bit_offset = i % (sizeof(bitset_t) * 8);
        bool bit                = (input[input_unit_index] >> input_bit_offset) & 1;

        size_t output_unit_index = output_bit_index / (sizeof(bitset_t) * 8);
        size_t output_bit_offset = output_bit_index % (sizeof(bitset_t) * 8);

        output[output_unit_index] |= (static_cast<bitset_t>(bit) << output_bit_offset);

        ++output_bit_index;
      }
    }
  }

  index_t create_sparse_matrix_with_rmat(index_t m,
                                         index_t n,
                                         float sparsity,
                                         rmm::device_uvector<bitset_t>& filter_d)
  {
    index_t r_scale   = (index_t)std::log2(m);
    index_t c_scale   = (index_t)std::log2(n);
    index_t n_edges   = (index_t)(m * n * 1.0f * sparsity);
    index_t max_scale = std::max(r_scale, c_scale);

    rmm::device_uvector<index_t> out_src{(unsigned long)n_edges, stream};
    rmm::device_uvector<index_t> out_dst{(unsigned long)n_edges, stream};
    rmm::device_uvector<float> theta{(unsigned long)(4 * max_scale), stream};

    raft::random::RngState state{2024ULL, raft::random::GeneratorType::GenPC};

    raft::random::uniform<float>(handle, state, theta.data(), theta.size(), 0.0f, 1.0f);
    normalize<float, float>(
      theta.data(), theta.data(), max_scale, r_scale, c_scale, r_scale != c_scale, true, stream);
    raft::random::rmat_rectangular_gen((index_t*)nullptr,
                                       out_src.data(),
                                       out_dst.data(),
                                       theta.data(),
                                       r_scale,
                                       c_scale,
                                       n_edges,
                                       stream,
                                       state);

    index_t nnz_h = 0;
    {
      auto src    = out_src.data();
      auto dst    = out_dst.data();
      auto bitset = filter_d.data();
      rmm::device_scalar<index_t> nnz(0, stream);
      auto nnz_view = raft::make_device_scalar_view<index_t>(nnz.data());
      auto filter_view =
        raft::make_device_vector_view<const uint32_t, index_t>(filter_d.data(), filter_d.size());
      index_t size_h = m * n;
      auto size_view = raft::make_host_scalar_view<const index_t, index_t>(&size_h);

      set_bitmap(src, dst, bitset, n_edges, n, stream);

      raft::popc(handle, filter_view, size_view, nnz_view);
      raft::copy(&nnz_h, nnz.data(), 1, stream);

      raft::resource::sync_stream(handle, stream);
    }

    return nnz_h;
  }

  void cpu_convert_to_csr(std::vector<bitset_t>& bitset,
                          index_t rows,
                          index_t cols,
                          std::vector<index_t>& indices,
                          std::vector<index_t>& indptr)
  {
    index_t offset_indptr   = 0;
    index_t offset_values   = 0;
    indptr[offset_indptr++] = 0;

    index_t index        = 0;
    bitset_t element     = 0;
    index_t bit_position = 0;

    for (index_t i = 0; i < rows; ++i) {
      for (index_t j = 0; j < cols; ++j) {
        index        = i * cols + j;
        element      = bitset[index / (8 * sizeof(bitset_t))];
        bit_position = index % (8 * sizeof(bitset_t));

        if (((element >> bit_position) & 1)) {
          indices[offset_values] = static_cast<index_t>(j);
          offset_values++;
        }
      }
      indptr[offset_indptr++] = static_cast<index_t>(offset_values);
    }
  }

  void cpu_sddmm(const std::vector<dist_t>& A,
                 const std::vector<dist_t>& B,
                 std::vector<dist_t>& vals,
                 const std::vector<index_t>& cols,
                 const std::vector<index_t>& row_ptrs,
                 bool is_row_major_A,
                 bool is_row_major_B,
                 dist_t alpha = 1.0,
                 dist_t beta  = 0.0)
  {
    if (params.n_queries * params.dim != static_cast<index_t>(A.size()) ||
        params.dim * params.n_dataset != static_cast<index_t>(B.size())) {
      std::cerr << "Matrix dimensions and vector size do not match!" << std::endl;
      return;
    }

    bool trans_a = is_row_major_A;
    bool trans_b = is_row_major_B;

    for (index_t i = 0; i < params.n_queries; ++i) {
      for (index_t j = row_ptrs[i]; j < row_ptrs[i + 1]; ++j) {
        dist_t sum     = 0;
        dist_t norms_A = 0;
        dist_t norms_B = 0;

        for (index_t l = 0; l < params.dim; ++l) {
          index_t a_index = trans_a ? i * params.dim + l : l * params.n_queries + i;
          index_t b_index = trans_b ? l * params.n_dataset + cols[j] : cols[j] * params.dim + l;
          dist_t A_v;
          dist_t B_v;
          if constexpr (sizeof(value_t) == 2) {
            A_v = __half2float(__float2half(A[a_index]));
            B_v = __half2float(__float2half(B[b_index]));
          } else {
            A_v = A[a_index];
            B_v = B[b_index];
          }

          sum += A_v * B_v;

          norms_A += A_v * A_v;
          norms_B += B_v * B_v;
        }
        vals[j] = alpha * sum + beta * vals[j];
        if (params.metric == cuvs::distance::DistanceType::L2Expanded) {
          vals[j] = dist_t(-2.0) * vals[j] + norms_A + norms_B;
        } else if (params.metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
          vals[j] = std::sqrt(dist_t(-2.0) * vals[j] + norms_A + norms_B);
        } else if (params.metric == cuvs::distance::DistanceType::CosineExpanded) {
          vals[j] = dist_t(1.0) - vals[j] / std::sqrt(norms_A * norms_B);
        }
      }
    }
  }

  void cpu_select_k(const std::vector<index_t>& indptr_h,
                    const std::vector<index_t>& indices_h,
                    const std::vector<dist_t>& values_h,
                    std::optional<std::vector<index_t>>& in_idx_h,
                    index_t n_queries,
                    index_t n_dataset,
                    index_t top_k,
                    std::vector<dist_t>& out_values_h,
                    std::vector<index_t>& out_indices_h,
                    bool select_min = true)
  {
    auto comp = [select_min](const std::pair<dist_t, index_t>& a,
                             const std::pair<dist_t, index_t>& b) {
      return select_min ? a.first < b.first : a.first >= b.first;
    };

    for (index_t row = 0; row < n_queries; ++row) {
      std::priority_queue<std::pair<dist_t, index_t>,
                          std::vector<std::pair<dist_t, index_t>>,
                          decltype(comp)>
        pq(comp);
      for (index_t idx = indptr_h[row]; idx < indptr_h[row + 1]; ++idx) {
        pq.push({values_h[idx], (in_idx_h.has_value()) ? (*in_idx_h)[idx] : indices_h[idx]});
        if (pq.size() > size_t(top_k)) { pq.pop(); }
      }

      std::vector<std::pair<dist_t, index_t>> row_pairs;
      while (!pq.empty()) {
        row_pairs.push_back(pq.top());
        pq.pop();
      }

      if (select_min) {
        std::sort(row_pairs.begin(), row_pairs.end(), [](const auto& a, const auto& b) {
          return a.first <= b.first;
        });
      } else {
        std::sort(row_pairs.begin(), row_pairs.end(), [](const auto& a, const auto& b) {
          return a.first >= b.first;
        });
      }
      for (index_t col = 0; col < top_k; col++) {
        if (col < index_t(row_pairs.size())) {
          out_values_h[row * top_k + col]  = row_pairs[col].first;
          out_indices_h[row * top_k + col] = row_pairs[col].second;
        }
      }
    }
  }

  void SetUp() override
  {
    if (std::is_same_v<value_t, half> && !isCuSparseVersionGreaterThan_12_0_1()) {
      GTEST_SKIP() << "Skipping all tests for half-float as cuSparse doesn't support it.";
    }
    index_t element = raft::ceildiv(1 * params.n_dataset, index_t(sizeof(bitset_t) * 8));
    std::vector<bitset_t> filter_h(element);
    std::vector<bitset_t> filter_repeat_h(element * params.n_queries);

    filter_d.resize(element, stream);

    nnz = create_sparse_matrix_with_rmat(1, params.n_dataset, params.sparsity, filter_d);
    raft::update_host(filter_h.data(), filter_d.data(), filter_d.size(), stream);
    raft::resource::sync_stream(handle, stream);

    repeat_cpu_bitset(
      filter_h, size_t(params.n_dataset), size_t(params.n_queries), filter_repeat_h);
    nnz *= params.n_queries;

    index_t dataset_size = params.n_dataset * params.dim;
    index_t queries_size = params.n_queries * params.dim;

    std::vector<dist_t> dataset_h(dataset_size);
    std::vector<dist_t> queries_h(queries_size);

    dataset_d.resize(dataset_size, stream);
    queries_d.resize(queries_size, stream);

    auto blobs_in_val =
      raft::make_device_matrix<dist_t, index_t>(handle, 1, dataset_size + queries_size);
    auto labels = raft::make_device_vector<index_t, index_t>(handle, 1);

    if constexpr (!std::is_same_v<value_t, half>) {
      raft::random::make_blobs<value_t, index_t>(blobs_in_val.data_handle(),
                                                 labels.data_handle(),
                                                 1,
                                                 dataset_size + queries_size,
                                                 1,
                                                 stream,
                                                 false,
                                                 nullptr,
                                                 nullptr,
                                                 value_t(1.0),
                                                 false,
                                                 value_t(-1.0f),
                                                 value_t(1.0f),
                                                 uint64_t(2024));
    } else {
      raft::random::make_blobs<dist_t, index_t>(blobs_in_val.data_handle(),
                                                labels.data_handle(),
                                                1,
                                                dataset_size + queries_size,
                                                1,
                                                stream,
                                                false,
                                                nullptr,
                                                nullptr,
                                                dist_t(1.0),
                                                false,
                                                dist_t(-1.0f),
                                                dist_t(1.0f),
                                                uint64_t(2024));
    }

    raft::copy(dataset_h.data(), blobs_in_val.data_handle(), dataset_size, stream);

    if constexpr (std::is_same_v<value_t, half>) {
      raft::linalg::unaryOp(
        dataset_d.data(), blobs_in_val.data_handle(), dataset_size, float_to_half(), stream);
    } else {
      raft::copy(dataset_d.data(), blobs_in_val.data_handle(), dataset_size, stream);
    }

    raft::copy(queries_h.data(), blobs_in_val.data_handle() + dataset_size, queries_size, stream);
    if constexpr (std::is_same_v<value_t, half>) {
      raft::linalg::unaryOp(queries_d.data(),
                            blobs_in_val.data_handle() + dataset_size,
                            queries_size,
                            float_to_half(),
                            stream);
    } else {
      raft::copy(queries_d.data(), blobs_in_val.data_handle() + dataset_size, queries_size, stream);
    }

    raft::resource::sync_stream(handle);

    std::vector<dist_t> values_h(nnz);
    std::vector<index_t> indices_h(nnz);
    std::vector<index_t> indptr_h(params.n_queries + 1);

    cpu_convert_to_csr(filter_repeat_h, params.n_queries, params.n_dataset, indices_h, indptr_h);

    cpu_sddmm(queries_h, dataset_h, values_h, indices_h, indptr_h, true, false);

    bool select_min = cuvs::distance::is_min_close(params.metric);

    std::vector<dist_t> out_val_h(
      params.n_queries * params.top_k,
      select_min ? std::numeric_limits<dist_t>::infinity() : std::numeric_limits<dist_t>::lowest());
    std::vector<index_t> out_idx_h(params.n_queries * params.top_k, static_cast<index_t>(0));

    out_val_d.resize(params.n_queries * params.top_k, stream);
    out_idx_d.resize(params.n_queries * params.top_k, stream);

    raft::update_device(out_val_d.data(), out_val_h.data(), out_val_h.size(), stream);
    raft::update_device(out_idx_d.data(), out_idx_h.data(), out_idx_h.size(), stream);

    raft::resource::sync_stream(handle);

    std::optional<std::vector<index_t>> optional_indices_h = std::nullopt;
    cpu_select_k(indptr_h,
                 indices_h,
                 values_h,
                 optional_indices_h,
                 params.n_queries,
                 params.n_dataset,
                 params.top_k,
                 out_val_h,
                 out_idx_h,
                 select_min);
    out_val_expected_d.resize(params.n_queries * params.top_k, stream);
    out_idx_expected_d.resize(params.n_queries * params.top_k, stream);

    raft::update_device(out_val_expected_d.data(), out_val_h.data(), out_val_h.size(), stream);
    raft::update_device(out_idx_expected_d.data(), out_idx_h.data(), out_idx_h.size(), stream);

    raft::resource::sync_stream(handle);
  }

  void Run()
  {
    auto dataset_raw = raft::make_device_matrix_view<const value_t, index_t, raft::row_major>(
      (const value_t*)dataset_d.data(), params.n_dataset, params.dim);

    auto queries = raft::make_device_matrix_view<const value_t, index_t, raft::row_major>(
      (const value_t*)queries_d.data(), params.n_queries, params.dim);

    auto dataset = brute_force::build(handle, dataset_raw, params.metric);

    auto filter =
      cuvs::core::bitset_view<bitset_t, index_t>((bitset_t*)filter_d.data(), params.n_dataset);

    auto out_val = raft::make_device_matrix_view<dist_t, index_t, raft::row_major>(
      out_val_d.data(), params.n_queries, params.top_k);
    auto out_idx = raft::make_device_matrix_view<index_t, index_t, raft::row_major>(
      out_idx_d.data(), params.n_queries, params.top_k);

    brute_force::search(handle,
                        dataset,
                        queries,
                        out_idx,
                        out_val,
                        cuvs::neighbors::filtering::bitset_filter(filter));
    std::vector<dist_t> out_val_h(params.n_queries * params.top_k,
                                  std::numeric_limits<dist_t>::infinity());

    raft::update_host(out_val_h.data(), out_val_d.data(), out_val_h.size(), stream);
    raft::resource::sync_stream(handle);

    ASSERT_TRUE(cuvs::neighbors::devArrMatchKnnPair(out_idx_expected_d.data(),
                                                    out_idx.data_handle(),
                                                    out_val_expected_d.data(),
                                                    out_val.data_handle(),
                                                    params.n_queries,
                                                    params.top_k,
                                                    0.001f,
                                                    stream,
                                                    true));
  }

  // Same as Run(), but passes the true sparsity as a filtering_rate hint to the
  // params-taking search overload. Confirms results match auto-detection.
  void RunWithFilteringRateHint()
  {
    auto dataset_raw = raft::make_device_matrix_view<const value_t, index_t, raft::row_major>(
      (const value_t*)dataset_d.data(), params.n_dataset, params.dim);

    auto queries = raft::make_device_matrix_view<const value_t, index_t, raft::row_major>(
      (const value_t*)queries_d.data(), params.n_queries, params.dim);

    auto dataset = brute_force::build(handle, dataset_raw, params.metric);

    auto filter =
      cuvs::core::bitset_view<bitset_t, index_t>((bitset_t*)filter_d.data(), params.n_dataset);

    auto out_val = raft::make_device_matrix_view<dist_t, index_t, raft::row_major>(
      out_val_d.data(), params.n_queries, params.top_k);
    auto out_idx = raft::make_device_matrix_view<index_t, index_t, raft::row_major>(
      out_idx_d.data(), params.n_queries, params.top_k);

    cuvs::neighbors::brute_force::search_params search_params;
    search_params.filtering_rate = params.sparsity;

    brute_force::search(handle,
                        search_params,
                        dataset,
                        queries,
                        out_idx,
                        out_val,
                        cuvs::neighbors::filtering::bitset_filter(filter));

    raft::resource::sync_stream(handle);

    ASSERT_TRUE(cuvs::neighbors::devArrMatchKnnPair(out_idx_expected_d.data(),
                                                    out_idx.data_handle(),
                                                    out_val_expected_d.data(),
                                                    out_val.data_handle(),
                                                    params.n_queries,
                                                    params.top_k,
                                                    0.001f,
                                                    stream,
                                                    true));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  PrefilteredBruteForceInputs<index_t> params;

  index_t nnz;

  rmm::device_uvector<value_t> dataset_d;
  rmm::device_uvector<value_t> queries_d;
  rmm::device_uvector<bitset_t> filter_d;

  rmm::device_uvector<dist_t> out_val_d;
  rmm::device_uvector<dist_t> out_val_expected_d;

  rmm::device_uvector<index_t> out_idx_d;
  rmm::device_uvector<index_t> out_idx_expected_d;
};

using PrefilteredBruteForceTestOnBitmap_float_int64 =
  PrefilteredBruteForceOnBitmapTest<float, float, int64_t>;
TEST_P(PrefilteredBruteForceTestOnBitmap_float_int64, Result) { Run(); }
TEST_P(PrefilteredBruteForceTestOnBitmap_float_int64, ResultWithFilteringRateHint)
{
  RunWithFilteringRateHint();
}

using PrefilteredBruteForceTestOnBitmap_half_int64 =
  PrefilteredBruteForceOnBitmapTest<half, float, int64_t>;
TEST_P(PrefilteredBruteForceTestOnBitmap_half_int64, Result) { Run(); }
TEST_P(PrefilteredBruteForceTestOnBitmap_half_int64, ResultWithFilteringRateHint)
{
  RunWithFilteringRateHint();
}

using PrefilteredBruteForceTestOnBitset_float_int64 =
  PrefilteredBruteForceOnBitsetTest<float, float, int64_t>;
TEST_P(PrefilteredBruteForceTestOnBitset_float_int64, Result) { Run(); }
TEST_P(PrefilteredBruteForceTestOnBitset_float_int64, ResultWithFilteringRateHint)
{
  RunWithFilteringRateHint();
}

using PrefilteredBruteForceTestOnBitset_half_int64 =
  PrefilteredBruteForceOnBitsetTest<half, float, int64_t>;
TEST_P(PrefilteredBruteForceTestOnBitset_half_int64, Result) { Run(); }
TEST_P(PrefilteredBruteForceTestOnBitset_half_int64, ResultWithFilteringRateHint)
{
  RunWithFilteringRateHint();
}

template <typename index_t>
const std::vector<PrefilteredBruteForceInputs<index_t>> selectk_inputs = {
  {8, 131072, 255, 255, 0.01, cuvs::distance::DistanceType::L2Expanded},
  {8, 131072, 255, 255, 0.01, cuvs::distance::DistanceType::InnerProduct},
  {8, 131072, 255, 255, 0.01, cuvs::distance::DistanceType::L2SqrtExpanded},
  {8, 131072, 255, 255, 0.01, cuvs::distance::DistanceType::CosineExpanded},
  {2, 131072, 255, 255, 0.4, cuvs::distance::DistanceType::L2Expanded},

  {8, 131072, 512, 16, 0.5, cuvs::distance::DistanceType::L2Expanded},
  {16, 131072, 2052, 16, 0.2, cuvs::distance::DistanceType::L2Expanded},
  {2, 8192, 255, 16, 0.01, cuvs::distance::DistanceType::InnerProduct},
  {2, 8192, 255, 16, 0.4, cuvs::distance::DistanceType::InnerProduct},
  {16, 8192, 512, 16, 0.5, cuvs::distance::DistanceType::InnerProduct},

  {128, 8192, 2052, 16, 0.2, cuvs::distance::DistanceType::InnerProduct},
  {1024, 8192, 1, 0, 0.1, cuvs::distance::DistanceType::L2Expanded},
  {1024, 8192, 3, 0, 0.1, cuvs::distance::DistanceType::InnerProduct},
  {1024, 8192, 5, 0, 0.1, cuvs::distance::DistanceType::L2SqrtExpanded},
  {1024, 8192, 8, 0, 0.1, cuvs::distance::DistanceType::CosineExpanded},

  {1024, 8192, 1, 1, 0.1, cuvs::distance::DistanceType::L2Expanded},
  {1024, 8192, 3, 1, 0.1, cuvs::distance::DistanceType::InnerProduct},
  {1024, 8192, 5, 1, 0.1, cuvs::distance::DistanceType::L2SqrtExpanded},
  {1024, 8192, 8, 1, 0.1, cuvs::distance::DistanceType::CosineExpanded},
  {1024, 8192, 2050, 16, 0.4, cuvs::distance::DistanceType::L2Expanded},

  {1024, 8192, 2051, 16, 0.5, cuvs::distance::DistanceType::L2Expanded},
  {1024, 8192, 2052, 16, 0.2, cuvs::distance::DistanceType::L2Expanded},
  {1024, 8192, 2050, 16, 0.4, cuvs::distance::DistanceType::InnerProduct},
  {1024, 8192, 2051, 16, 0.5, cuvs::distance::DistanceType::InnerProduct},
  {1024, 8192, 2052, 16, 0.2, cuvs::distance::DistanceType::InnerProduct},

  {1024, 8192, 2050, 16, 0.4, cuvs::distance::DistanceType::L2SqrtExpanded},
  {1024, 8192, 2051, 16, 0.5, cuvs::distance::DistanceType::L2SqrtExpanded},
  {1024, 8192, 2052, 16, 0.2, cuvs::distance::DistanceType::L2SqrtExpanded},
  {1024, 8192, 2050, 16, 0.4, cuvs::distance::DistanceType::CosineExpanded},
  {1024, 8192, 2051, 16, 0.5, cuvs::distance::DistanceType::CosineExpanded},

  {1024, 8192, 2052, 16, 0.2, cuvs::distance::DistanceType::CosineExpanded},
  {1024, 8192, 1, 16, 0.5, cuvs::distance::DistanceType::L2Expanded},
  {1024, 8192, 2, 16, 0.2, cuvs::distance::DistanceType::L2Expanded},

  {1024, 8192, 3, 16, 0.4, cuvs::distance::DistanceType::InnerProduct},
  {1024, 8192, 4, 16, 0.5, cuvs::distance::DistanceType::InnerProduct},
  {1024, 8192, 5, 16, 0.2, cuvs::distance::DistanceType::L2SqrtExpanded},
  {1024, 8192, 8, 16, 0.4, cuvs::distance::DistanceType::L2SqrtExpanded},
  {1024, 8192, 5, 16, 0.5, cuvs::distance::DistanceType::CosineExpanded},
  {1024, 8192, 8, 16, 0.2, cuvs::distance::DistanceType::CosineExpanded}};

INSTANTIATE_TEST_CASE_P(PrefilteredBruteForceOnBitmapTest,
                        PrefilteredBruteForceTestOnBitmap_float_int64,
                        ::testing::ValuesIn(selectk_inputs<int64_t>));

INSTANTIATE_TEST_CASE_P(PrefilteredBruteForceOnBitmapTest,
                        PrefilteredBruteForceTestOnBitmap_half_int64,
                        ::testing::ValuesIn(selectk_inputs<int64_t>));

INSTANTIATE_TEST_CASE_P(PrefilteredBruteForceOnBitsetTest,
                        PrefilteredBruteForceTestOnBitset_float_int64,
                        ::testing::ValuesIn(selectk_inputs<int64_t>));

INSTANTIATE_TEST_CASE_P(PrefilteredBruteForceOnBitsetTest,
                        PrefilteredBruteForceTestOnBitset_half_int64,
                        ::testing::ValuesIn(selectk_inputs<int64_t>));

}  // namespace cuvs::neighbors::brute_force
