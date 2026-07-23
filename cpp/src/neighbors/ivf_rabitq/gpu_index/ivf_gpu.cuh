/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 2/23/25.
//

#pragma once

#include "../defines.hpp"
#include "initializer_gpu.cuh"
#include "quantizer_gpu.cuh"
#include "rotator_gpu.cuh"

#include <cuvs/distance/distance.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

namespace cuvs::neighbors::ivf_rabitq::detail {

class SearcherGPU;

// Structure for cluster-query pairs
struct ClusterQueryPair {
  int cluster_idx;
  int query_idx;
};

class IVFGPU {
 public:
  class GPUClusterMeta {
   public:
    size_t num;          // Number of vectors in this cluster.
    size_t start_index;  // Combined offset: index of first vector in the flattened arrays.

    __host__ __device__ GPUClusterMeta(size_t num, size_t start_idx)
      : num(num), start_index(start_idx)
    {
    }

    // default constructor (useful for vector initialize)
    GPUClusterMeta() : num(0), start_index(0) {}

    // Copy constructor.
    GPUClusterMeta(const GPUClusterMeta& other) = default;

    // Copy assignment.
    GPUClusterMeta& operator=(const GPUClusterMeta& other) = default;

    // Destructor.
    ~GPUClusterMeta() = default;

    /**
     * @brief Compute pointer to the first block of quantized short data.
     * Factors are stored separately in SoA layout.
     *
     * @param parent Pointer to the IVFGPU instance that holds the base pointer.
     * @return Pointer to the first block of this cluster's short data.
     */
    __host__ uint32_t* first_block(const IVFGPU& parent) const
    {
      return parent.get_short_data_device() +
             start_index * (parent.quantizer().short_code_length());
    }

    __device__ uint32_t* first_block_gpu(uint32_t* d_short_data, size_t short_code_length) const
    {
      return d_short_data + start_index * short_code_length;
    }

    __host__ uint32_t* first_block_host(const IVFGPU& parent) const
    {
      return parent.get_short_data_host() + start_index * (parent.quantizer().short_code_length());
    }

    float* short_factor_batch(const IVFGPU& parent, size_t i) const
    {
      return parent.get_short_factors_batch_device() + (start_index + i) * 3;
    }

    // Per-vector ‖x‖² for the i-th vector in this cluster (InnerProduct metric only).
    float* vec_sqr_norm(const IVFGPU& parent, size_t i) const
    {
      return parent.get_vec_sqr_norms_device() + (start_index + i);
    }

    /**
     * @brief Get the pointer to the long code for the i-th vector in this cluster.
     *
     * @param parent Pointer to the IVFGPU instance that holds the base pointer.
     * @param i Index within the cluster (0 ≤ i < Num).
     * @param long_code_length Fixed length (in bytes) for each vector’s long code.
     * @return Pointer to the long code for the i-th vector.
     */
    __host__ __device__ uint8_t* long_code(const IVFGPU& parent,
                                           size_t i,
                                           size_t long_code_length) const
    {
      return parent.get_long_code_device() + (start_index + i) * long_code_length;
    }

    __host__ __device__ uint8_t* long_code_host(const IVFGPU& parent,
                                                size_t i,
                                                size_t long_code_length) const
    {
      return parent.get_long_code_host() + (start_index + i) * long_code_length;
    }

    /**
     * @brief Get the pointer to the extra factor for the i-th vector in this cluster.
     * Two ex_factors are stored (f_add_ex and f_rescale_ex) in SoA layout.
     *
     * @param parent Pointer to the IVFGPU instance that holds the base pointer.
     * @param i Index within the cluster.
     * @return Pointer to the extra factor for the i-th vector.
     */
    ExFactor* ex_factor(const IVFGPU& parent, size_t i) const
    {
      return parent.get_ex_factor_device() + (start_index + i) * 2;
    }

    ExFactor* ex_factor_host(const IVFGPU& parent, size_t i) const
    {
      return parent.get_ex_factor_host() + (start_index + i) * 2;
    }

    /**
     * @brief Get the pointer to the vector IDs for this cluster.
     *
     * @param parent Pointer to the IVFGPU instance that holds the base pointer.
     * @return Pointer to the first ID in this cluster.
     */
    __host__ __device__ PID* ids(const IVFGPU& parent) const
    {
      return parent.get_ids_device() + start_index;
    }

    __host__ __device__ PID* ids_host(const IVFGPU& parent) const
    {
      return parent.get_ids_host() + start_index;
    }
  };

  /**
   * @brief initialize function (no memory allocated yet)
   *
   * @param n Num of data points.
   * @param dim Dimension of data points.
   * @param k Num of centroids.
   * @param bits_per_dim totalbits = EX_BITS+1
   */
  IVFGPU(raft::resources const& handle,
         size_t n,
         size_t dim,
         size_t k,
         size_t bits_per_dim,
         cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded);
  IVFGPU(raft::resources const& handle)
    : handle_(handle), initializer(nullptr), Rota(std::make_unique<RotatorGPU>(handle_, 128))
  {
  }

  /**
   * @brief Build function
   *
   * @param device_data pointer to device data.
   * @param device_centroids pointer to centroids.
   * @param device_cluster_ids PIDs of vectors.
   */
  void construct_on_gpu(const float* device_data,
                        const float* device_centroids,
                        const PID* device_cluster_ids,
                        bool fast_quantize);

  /**
   * @brief Build function with streaming from host memory
   *
   * @param host_data pointer to host-resident data.
   * @param device_centroids pointer to centroids on device.
   * @param device_cluster_ids cluster assignments for each vector (on device).
   * @param fast_quantize whether to use fast quantization.
   * @param batch_size_vectors maximum vectors per batch (upper bound).
   */
  void construct_on_gpu_streaming(const float* host_data,
                                  const float* device_centroids,
                                  const PID* device_cluster_ids,
                                  bool fast_quantize,
                                  size_t batch_size_vectors = 100000);

  void save(const char* filename) const;

  // load_transposed only applies for new batch index
  void load_transposed(const char* filename);

  // device data getters
  __host__ __device__ uint32_t* get_short_data_device() const noexcept
  {
    return const_cast<uint32_t*>(this->short_data_.data_handle());
  }
  __host__ __device__ uint8_t* get_long_code_device() const noexcept
  {
    return const_cast<uint8_t*>(this->long_code_.data_handle());
  }
  __host__ __device__ ExFactor* get_ex_factor_device() const noexcept
  {
    return const_cast<ExFactor*>(this->ex_factor_.data_handle());
  }
  __host__ __device__ PID* get_ids_device() const noexcept
  {
    return const_cast<PID*>(this->ids_.data_handle());
  }
  __host__ __device__ float* get_short_factors_batch_device() const noexcept
  {
    return const_cast<float*>(this->short_factors_batch_.data_handle());
  }
  raft::device_vector<GPUClusterMeta, int64_t> const& get_cluster_meta() const
  {
    return cluster_meta_;
  }

  // host data getters
  raft::host_vector<GPUClusterMeta, int64_t> const& get_cluster_meta_host() const
  {
    return cluster_meta_host_;
  }
  __host__ __device__ uint32_t* get_short_data_host() const noexcept
  {
    return const_cast<uint32_t*>(short_data_host_.data_handle());
  }
  __host__ __device__ uint8_t* get_long_code_host() const noexcept
  {
    return const_cast<uint8_t*>(long_code_host_.data_handle());
  }
  __host__ __device__ ExFactor* get_ex_factor_host() const noexcept
  {
    return const_cast<ExFactor*>(ex_factor_host_.data_handle());
  }
  __host__ __device__ PID* get_ids_host() const noexcept
  {
    return const_cast<PID*>(ids_host_.data_handle());
  }

  // metadata getters
  size_t get_num_vectors() const noexcept { return this->num_vectors; }
  size_t get_num_dimensions() const noexcept { return this->num_dimensions; }
  size_t get_num_padded_dim() const noexcept { return this->num_padded_dim; }
  size_t get_num_centroids() const { return num_centroids; }
  size_t get_max_cluster_length() const noexcept { return max_cluster_length; }
  size_t get_ex_bits() const noexcept { return ex_bits; }
  cuvs::distance::DistanceType metric() const noexcept { return metric_; }
  bool is_inner_product() const noexcept
  {
    return metric_ == cuvs::distance::DistanceType::InnerProduct;
  }
  __host__ __device__ float* get_vec_sqr_norms_device() const noexcept
  {
    return const_cast<float*>(this->vec_sqr_norms_.data_handle());
  }

  // member object getters
  DataQuantizerGPU& quantizer() const { return *(this->DQ); }
  RotatorGPU& rotator() const { return *(this->Rota); }

  void BatchClusterSearch(
    raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
    size_t k,
    size_t nprobe,
    SearcherGPU& searcher,
    size_t batch_size,
    raft::device_matrix_view<float, int64_t, raft::row_major> d_final_dists,
    raft::device_matrix_view<uint32_t, int64_t, raft::row_major> d_final_pids);

  void BatchClusterSearchLUT16(
    raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
    size_t k,
    size_t nprobe,
    SearcherGPU& searcher,
    size_t batch_size,
    raft::device_matrix_view<float, int64_t, raft::row_major> d_final_dists,
    raft::device_matrix_view<uint32_t, int64_t, raft::row_major> d_final_pids);

  void BatchClusterSearchQuantizeQuery(
    raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
    size_t k,
    size_t nprobe,
    SearcherGPU& searcher,
    size_t batch_size,
    raft::device_matrix_view<float, int64_t, raft::row_major> d_final_dists,
    raft::device_matrix_view<uint32_t, int64_t, raft::row_major> d_final_pids,
    int query_bits);

 private:
  void PrepareClusterSearchInputs(
    raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
    size_t nprobe,
    SearcherGPU& searcher,
    raft::device_vector<ClusterQueryPair, int64_t>& d_sorted_pairs,
    raft::device_vector<float, int64_t>& d_G_k1xSumq,
    raft::device_vector<float, int64_t>& d_G_kbxSumq);

  /**
   * @brief function to allocate memory based on the cluster
   *
   */
  void AllocateDeviceMemory();

  // Following are inline functions to compute spaces for memory allocation

  size_t GetShortDataBytesSimple() const
  {
    //        assert(num_clusters == num_centroids);  // num of clusters
    //        size_t total_blocks = 0;
    //        for (auto s = 0; s < num_clusters; s++) {
    //            total_blocks += cluster_sizes[s];
    //        }
    return num_vectors * this->quantizer().block_bytes();
  }

  size_t GetShortDataFactorBytesBatch() const
  {
    //        assert(num_clusters == num_centroids);  // num of clusters
    //        size_t total_blocks = 0;
    //        for (auto s = 0; s < num_clusters; s++) {
    //            total_blocks += cluster_sizes[s];
    //        }
    return num_vectors * 3 * sizeof(float);
  }

  size_t GetExFactorBytes() const
  {
    return 2 * sizeof(float) * num_vectors;  // only f_add_ex and f_rescale_ex
  }

  size_t GetPIDsBytes() const { return sizeof(PID) * num_vectors; }

  // Per-vector squared-norm array; only allocated/serialized for the InnerProduct metric.
  size_t GetVecSqrNormBytes() const { return sizeof(float) * num_vectors; }

  size_t GetLongCodeBytes() const
  {
    return sizeof(uint8_t) * quantizer().long_code_length() * num_vectors;
  }
  void init_clusters(const std::vector<size_t>& cluster_sizes);

  void quantize_cluster(GPUClusterMeta& cp,
                        const float* data,
                        const float* cur_centroid,
                        float* rotated_c) const;

  void compute_centroid_norms();

  void AllocateHostMemory();

  raft::resources const& handle_;  // reusable resource handle
  rmm::cuda_stream_view stream_ =
    raft::resource::get_cuda_stream(handle_);  // CUDA stream obtained from handle_

  // Device pointers for each data array.
  raft::device_vector<uint32_t, int64_t> short_data_ =
    raft::make_device_vector<uint32_t, int64_t>(handle_, 0);  // RaBitQ code and factors.
  raft::device_vector<uint8_t, int64_t> long_code_ =
    raft::make_device_vector<uint8_t, int64_t>(handle_, 0);  // ExRaBitQ code.
  raft::device_vector<ExFactor, int64_t> ex_factor_ =
    raft::make_device_vector<ExFactor, int64_t>(handle_, 0);  // ExRaBitQ factor.
  raft::device_vector<PID, int64_t> ids_ =
    raft::make_device_vector<PID, int64_t>(handle_, 0);  // PID of vectors.
  raft::device_vector<GPUClusterMeta, int64_t> cluster_meta_ =
    raft::make_device_vector<GPUClusterMeta, int64_t>(handle_,
                                                      0);  // Device-side array of clusters.
  raft::device_vector<float, int64_t> centroid_norms_ =
    raft::make_device_vector<float, int64_t>(handle_, 0);

  // batch-data (SoA layout - factors stored separately)
  raft::device_vector<float, int64_t> short_factors_batch_ =
    raft::make_device_vector<float, int64_t>(handle_, 0);  // N * 3 float rabitq factors

  // Per-vector squared L2 norm ‖x‖² of the original (rotation-invariant) vectors, in the same
  // cluster-permuted order as the factors. Only populated for the InnerProduct metric, where the
  // reconstructed squared-L2 estimate is converted to an inner product via
  // ⟨q,x⟩ = (‖q‖² + ‖x‖² − ‖q−x‖²)/2.
  raft::device_vector<float, int64_t> vec_sqr_norms_ =
    raft::make_device_vector<float, int64_t>(handle_, 0);

  // host-side copies
  raft::host_vector<uint32_t, int64_t> short_data_host_ = raft::make_host_vector<uint32_t, int64_t>(
    0);  // TODO: CPU side, we need on factors from short_data_host_, so no need to
         // store all these codes
  raft::host_vector<uint8_t, int64_t> long_code_host_ =
    raft::make_host_vector<uint8_t, int64_t>(0);  // ExRaBitQ code.
  raft::host_vector<ExFactor, int64_t> ex_factor_host_ =
    raft::make_host_vector<ExFactor, int64_t>(0);  // ExRaBitQ factor.
  raft::host_vector<PID, int64_t> ids_host_ =
    raft::make_host_vector<PID, int64_t>(0);  // PID of vectors.
  raft::host_vector<GPUClusterMeta, int64_t> cluster_meta_host_ =
    raft::make_host_vector<GPUClusterMeta, int64_t>(0);  // Host-side copy of clusters

  // Index meta-data.
  size_t num_vectors;         // Total number of vectors.
  size_t num_dimensions;      // Dimensionality of the input vectors.
  size_t num_padded_dim;      // padded dimensions
  size_t num_centroids;       // Centroids && Clusters
  size_t max_cluster_length;  // Maximum length of clusters
  size_t ex_bits;             // Extra bits parameter for quantization.
  cuvs::distance::DistanceType metric_ =
    cuvs::distance::DistanceType::L2Expanded;  // Distance metric of the index.

  std::unique_ptr<InitializerGPU>
    initializer;  // Initializer, indicates which initializer to use (currently only FlatIVF)
  std::unique_ptr<DataQuantizerGPU> DQ;  // Data quantizer.
  std::unique_ptr<RotatorGPU> Rota;      // Data Rotator.
};

}  // namespace cuvs::neighbors::ivf_rabitq::detail
