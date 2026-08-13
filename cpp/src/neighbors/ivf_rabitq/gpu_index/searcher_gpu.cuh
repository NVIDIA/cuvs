/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 4/14/25.
//

#pragma once

#include "ivf_gpu.cuh"
#include "quantizer_gpu.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cuvs::neighbors::ivf_rabitq::detail {

// block-level sorting used if topk <= kMaxTopKBlockSort; must be power of 2; increases shared mem
// usage
static constexpr int kMaxTopKBlockSort = 64;

class SearcherGPU {
 public:
  /**
   * @param d vector dimension; must be divisible by 32 and by BITS_PER_CHUNK. Pass the padded
   *          dimension (e.g. IVFGPU::get_num_padded_dim()), not the raw dataset dimension.
   */
  explicit SearcherGPU(raft::resources const& handle,
                       const float* q,
                       size_t d,
                       size_t ex_bits,
                       std::string mode                                             = "",
                       DataQuantizerGPU::FastQuantizeFactors* fast_quantize_factors = nullptr,
                       bool rabitq_quantize_flag                                    = true);

  SearcherGPU(SearcherGPU&& other) = default;

  /**
   * @brief Allocate temp search space in GPU memory
   *
   * @param num_centroids number of cluster centroids
   * @param num_queries number of queries
   */
  void AllocateSearcherSpace(size_t num_centroids, size_t num_queries);

  // Getter methods
  std::string const& get_mode() { return mode_; }
  raft::resources const& get_handle() const { return handle_; }
  rmm::cuda_stream_view get_stream() const { return stream_; }
  float* get_centroid_distances() { return centroid_distances_.data_handle(); }
  float* get_q_norms() { return q_norms_.data_handle(); }

  // Setter methods
  void set_query(const float* query) { query_ = query; }
  void set_quant_query(int16_t* quant_query)
  {
    quant_query_ = std::unique_ptr<int16_t, decltype(std::free)*>(quant_query, std::free);
  }  // quant_query must be allocated with `malloc` or similar, as opposed to `new`
  void set_unit_q(float* unit_q)
  {
    unit_q_ = std::unique_ptr<float, decltype(std::free)*>(unit_q, std::free);
  }  // unit_q must be allocated with `malloc` or similar, as opposed to `new`

  /** @brief Search <cluster_id, query_id> pairs */
  void SearchClusterQueryPairs(
    const IVFGPU& cur_ivf,
    IVFGPU::GPUClusterMeta* d_cluster_meta,
    ClusterQueryPair* d_sorted_pairs,
    size_t num_queries,
    raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
    const float* d_G_k1xSumq,
    const float* d_G_kbxSumq,
    size_t nprobe,
    size_t topk,
    raft::device_matrix_view<float, int64_t, raft::row_major> d_final_dists,
    raft::device_matrix_view<uint32_t, int64_t, raft::row_major> d_final_pids);

  void SearchClusterQueryPairsSharedMemOpt(
    const IVFGPU& cur_ivf,
    IVFGPU::GPUClusterMeta* d_cluster_meta,
    ClusterQueryPair* d_sorted_pairs,
    size_t num_queries,
    raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
    const float* d_G_k1xSumq,
    const float* d_G_kbxSumq,
    size_t nprobe,
    size_t topk,
    raft::device_matrix_view<float, int64_t, raft::row_major> d_final_dists,
    raft::device_matrix_view<uint32_t, int64_t, raft::row_major> d_final_pids);

  void SearchClusterQueryPairsQuantizeQuery(
    const IVFGPU& cur_ivf,
    IVFGPU::GPUClusterMeta* d_cluster_meta,
    ClusterQueryPair* d_sorted_pairs,
    size_t num_queries,
    raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
    const float* d_G_k1xSumq,
    const float* d_G_kbxSumq,
    size_t nprobe,
    size_t topk,
    raft::device_matrix_view<float, int64_t, raft::row_major> d_final_dists,
    raft::device_matrix_view<uint32_t, int64_t, raft::row_major> d_final_pids,
    bool use_4bit = false);

 private:
  raft::resources const& handle_;  // reusable resource handle
  rmm::cuda_stream_view stream_ =
    raft::resource::get_cuda_stream(handle_);  // CUDA stream obtained from handle_
  size_t D;                                    // number of dimension
  const float* query_ = nullptr;               // rotated query (non-owning)
  std::unique_ptr<int16_t, decltype(std::free)*> quant_query_ = {
    nullptr, std::free};  // quantized query (to 2 bytes)
  std::unique_ptr<float, decltype(std::free)*> unit_q_ = {nullptr, std::free};
  bool rabitq_quantize_flag_;
  std::string mode_;

  // batch
  raft::device_vector<float, int64_t> centroid_distances_ =
    raft::make_device_vector<float, int64_t>(
      handle_, 0);  // stores distances between centroids and queries (l2 square)
  raft::device_vector<float, int64_t> q_norms_ =
    raft::make_device_vector<float, int64_t>(handle_, 0);  // query norms

  // quantization for queries
  float best_rescaling_factor = 0.0f;
};

}  // namespace cuvs::neighbors::ivf_rabitq::detail
