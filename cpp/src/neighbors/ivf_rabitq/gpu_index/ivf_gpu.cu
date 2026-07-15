/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 2/23/25.
//

#include "../../../util/serialize_validation.hpp"
#include "../../detail/ann_utils.cuh"
#include "../utils/reductions.cuh"
#include "ivf_gpu.cuh"
#include "searcher_gpu.cuh"
#include <cuvs/util/file_io.hpp>
#include <raft/util/integer_utils.hpp>

#include <cuvs/selection/select_k.hpp>
#include <raft/core/cublas_macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/detail/cublaslt_wrappers.hpp>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/transpose.cuh>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/device/device_histogram.cuh>
#include <thrust/sort.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <omp.h>
#include <vector>

namespace cuvs::neighbors::ivf_rabitq::detail {

IVFGPU::IVFGPU(raft::resources const& handle, size_t n, size_t dim, size_t k, size_t bits_per_dim)
  : handle_(handle),
    num_vectors(n),
    num_dimensions(dim),
    num_padded_dim(raft::round_up_safe<size_t>(dim, 64)),
    num_centroids(k),
    ex_bits(bits_per_dim - 1),
    initializer(nullptr),
    DQ(std::make_unique<DataQuantizerGPU>(handle_, dim, bits_per_dim - 1)),
    Rota(std::make_unique<RotatorGPU>(handle_, dim))
{
}

void IVFGPU::AllocateDeviceMemory()
{
  this->initializer = std::make_unique<FlatInitializerGPU>(handle_, num_padded_dim, num_centroids);

  // Compute the sizes for each data array.
  size_t short_data_size = GetShortDataBytesSimple();
  size_t long_code_size  = GetLongCodeBytes();
  size_t ex_factor_size  = GetExFactorBytes();
  size_t pids_size       = GetPIDsBytes();

  // Allocate memory for the quantized arrays (SoA layout).
  short_data_ =
    raft::make_device_vector<uint32_t, int64_t>(handle_, short_data_size / sizeof(uint32_t));
  short_factors_batch_ = raft::make_device_vector<float, int64_t>(
    handle_, GetShortDataFactorBytesBatch() / sizeof(float));
  long_code_ =
    raft::make_device_vector<uint8_t, int64_t>(handle_, long_code_size / sizeof(uint8_t));
  ex_factor_ =
    raft::make_device_vector<ExFactor, int64_t>(handle_, ex_factor_size / sizeof(ExFactor));
  ids_ = raft::make_device_vector<PID, int64_t>(handle_, pids_size / sizeof(PID));

  // Allocate memory for the per-cluster metadata and centroids.
  cluster_meta_ = raft::make_device_vector<GPUClusterMeta, int64_t>(handle_, num_centroids);
  raft::resource::sync_stream(handle_);
}

void IVFGPU::AllocateHostMemory()
{
  // Compute the sizes for each data array.
  size_t short_data_size = GetShortDataBytesSimple();
  size_t long_code_size  = GetLongCodeBytes();
  size_t ex_factor_size  = GetExFactorBytes();
  size_t pids_size       = GetPIDsBytes();

  this->short_data_host_ =
    raft::make_host_vector<uint32_t, int64_t>(short_data_size / sizeof(uint32_t));
  this->long_code_host_ =
    raft::make_host_vector<uint8_t, int64_t>(long_code_size / sizeof(uint8_t));
  this->ex_factor_host_ =
    raft::make_host_vector<ExFactor, int64_t>(ex_factor_size / sizeof(ExFactor));
  this->ids_host_ = raft::make_host_vector<PID, int64_t>(pids_size / sizeof(PID));
}

// load transposed data for short codes
void IVFGPU::load_transposed(const char* filename)
{
  cuvs::util::kvikio_file_reader reader(filename);
  auto& input = reader.stream();

  auto read_exact = [&](void* ptr, size_t n_bytes) {
    input.read(reinterpret_cast<char*>(ptr), n_bytes);
    RAFT_EXPECTS(input.gcount() == static_cast<std::streamsize>(n_bytes),
                 "unexpected EOF reading header from: %s",
                 filename);
  };

  // Load metadata.
  read_exact(&this->num_vectors, sizeof(size_t));
  read_exact(&this->num_dimensions, sizeof(size_t));
  // Compute padded dimension.
  this->num_padded_dim = raft::round_up_safe<size_t>(this->num_dimensions, 64);
  read_exact(&this->num_centroids, sizeof(size_t));
  read_exact(&this->ex_bits, sizeof(size_t));

  // Skip legacy batch_flag field for backward compatibility
  bool legacy_batch_flag;
  read_exact(&legacy_batch_flag, sizeof(bool));

  // Validate header fields to guard against malicious or corrupted files. With these bounds,
  // downstream `num_vectors * per_vector_bytes` products in AllocateDeviceMemory/the Get*Bytes
  // helpers cannot overflow size_t (per_vector_bytes is bounded by num_padded_dim).
  RAFT_EXPECTS(num_dimensions > 0, "ivf_rabitq::deserialize: num_dimensions=0 in: %s", filename);
  RAFT_EXPECTS(ex_bits < 9,
               "ivf_rabitq::deserialize: ex_bits=%zu out of valid range [0, 9) in: %s",
               ex_bits,
               filename);
  RAFT_EXPECTS(num_centroids > 0 && num_centroids <= cuvs::util::kMaxIvfNLists,
               "ivf_rabitq::deserialize: num_centroids=%zu out of valid range (0, %u] in: %s",
               num_centroids,
               cuvs::util::kMaxIvfNLists,
               filename);
  RAFT_EXPECTS(cuvs::util::is_mul_no_overflow(num_vectors, num_padded_dim),
               "ivf_rabitq::deserialize: num_vectors=%zu * num_padded_dim=%zu overflows size_t "
               "in: %s",
               num_vectors,
               num_padded_dim,
               filename);

  // Initialize quantizer and rotator (host objects that drive GPU routines).
  this->DQ = std::make_unique<DataQuantizerGPU>(handle_, num_dimensions, ex_bits);
  read_exact(this->DQ->get_query_scaling_factor(), sizeof(DataQuantizerGPU::FastQuantizeFactors));
  this->Rota = std::make_unique<RotatorGPU>(handle_, num_dimensions);
  // Load cluster sizes.
  std::vector<size_t> cluster_sizes(num_centroids, 0);
  read_exact(cluster_sizes.data(), sizeof(size_t) * num_centroids);
  RAFT_EXPECTS(
    std::accumulate(cluster_sizes.begin(), cluster_sizes.end(), size_t(0)) == num_vectors,
    "cluster sizes do not sum to num_vectors in: %s",
    filename);

  // Load rotator from file.
  this->rotator().load(handle_, reader);
  // Allocate device memory based on the cluster sizes.
  AllocateDeviceMemory();
  // Load initializer data (e.g., centroids) from file.
  this->initializer->LoadCentroids(reader, filename);
  compute_centroid_norms();
  // Read raw arrays from file into device memory.
  auto read_into_device = [&](void* d_ptr, size_t n_bytes) { reader.read_device(d_ptr, n_bytes); };

  auto read_into_device_host = [&](void* d_ptr, void* h_ptr, size_t n_bytes) {
    reader.read_device(d_ptr, n_bytes);
    raft::copy(static_cast<uint8_t*>(h_ptr), static_cast<uint8_t*>(d_ptr), n_bytes, stream_);
    raft::resource::sync_stream(handle_);
  };

  auto read_into_device_host_transposed_short =
    [&](void* d_ptr, void* h_ptr, size_t n_bytes, size_t& max_cluster_size) {
      RAFT_EXPECTS(n_bytes % sizeof(uint32_t) == 0,
                   "short-code data size must be a multiple of uint32_t");
      const size_t bytes_per_vector = DQ->block_bytes();
      RAFT_EXPECTS(bytes_per_vector % sizeof(uint32_t) == 0,
                   "short-code vector size must be a multiple of uint32_t");
      const size_t uint32s_per_vector = bytes_per_vector / sizeof(uint32_t);
      max_cluster_size = *std::max_element(cluster_sizes.begin(), cluster_sizes.end());
      RAFT_EXPECTS(max_cluster_size <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                     uint32s_per_vector <= static_cast<size_t>(std::numeric_limits<int>::max()),
                   "short-code matrix dimensions exceed the transpose API limit");
      auto sequential =
        raft::make_device_vector<uint32_t, int64_t>(handle_, max_cluster_size * uint32s_per_vector);
      raft::resource::sync_stream(handle_);

      size_t dst_offset = 0;
      size_t bytes_read = 0;
      for (size_t cluster_id = 0; cluster_id < num_centroids; cluster_id++) {
        const size_t cluster_size = cluster_sizes[cluster_id];
        if (cluster_size == 0) continue;

        const size_t cluster_bytes = cluster_size * bytes_per_vector;
        reader.read_device(sequential.data_handle(), cluster_bytes);

        raft::linalg::transpose(handle_,
                                sequential.data_handle(),
                                static_cast<uint32_t*>(d_ptr) + dst_offset,
                                static_cast<int>(cluster_size),
                                static_cast<int>(uint32s_per_vector),
                                stream_);
        raft::resource::sync_stream(handle_);

        dst_offset += cluster_bytes / sizeof(uint32_t);
        bytes_read += cluster_bytes;
      }
      RAFT_EXPECTS(bytes_read == n_bytes,
                   "short-code payload size mismatch: expected %zu bytes, consumed %zu",
                   n_bytes,
                   bytes_read);

      raft::copy(static_cast<uint8_t*>(h_ptr), static_cast<uint8_t*>(d_ptr), n_bytes, stream_);
      raft::resource::sync_stream(handle_);
    };

  // New change: host copy of ivf.
  AllocateHostMemory();
  read_into_device_host_transposed_short(short_data_.data_handle(),
                                         short_data_host_.data_handle(),
                                         GetShortDataBytesSimple(),
                                         this->max_cluster_length);
  read_into_device(short_factors_batch_.data_handle(),
                   GetShortDataFactorBytesBatch());  // SoA layout - no copy on CPU
  read_into_device_host(
    long_code_.data_handle(), long_code_host_.data_handle(), GetLongCodeBytes());
  read_into_device_host(
    ex_factor_.data_handle(), ex_factor_host_.data_handle(), GetExFactorBytes());
  read_into_device_host(ids_.data_handle(), ids_host_.data_handle(), GetPIDsBytes());

  // Initialize cluster metadata (host side) based on the loaded cluster sizes.
  init_clusters(cluster_sizes);
}

void IVFGPU::compute_centroid_norms()
{
  centroid_norms_     = raft::make_device_vector<float, int64_t>(handle_, num_centroids);
  auto centroids_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    initializer->GetCentroid(0), num_centroids, num_padded_dim);
  raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
    handle_, centroids_view, centroid_norms_.view());
}

void IVFGPU::init_clusters(const std::vector<size_t>& cluster_sizes)
{
  // Allocate a host vector to hold cluster metadata.
  cluster_meta_host_ = raft::make_host_vector<GPUClusterMeta, int64_t>(num_centroids);

  size_t added_vectors = 0;
  for (size_t i = 0; i < num_centroids; ++i) {
    size_t num            = cluster_sizes[i];
    cluster_meta_host_(i) = {num, added_vectors};
    added_vectors += num;
  }

  // Copy the host cluster metadata to device memory.
  // cluster_meta_ must have been allocated with size: num_centroids * sizeof(GPUClusterMeta)
  raft::copy(cluster_meta_.data_handle(), cluster_meta_host_.data_handle(), num_centroids, stream_);
  raft::resource::sync_stream(handle_);
}

// for debug use

// Extract i-th code (MSB-first packing) from a bitstream.
// ex_bits in [1..7]. bytes_len is the byte-length of the code stream.
static inline uint32_t extract_code_msb(const uint8_t* bytes,
                                        size_t bytes_len,
                                        size_t i,
                                        int ex_bits)
{
  const size_t bit_index = i * ex_bits;    // absolute bit index in stream
  const size_t byte_idx  = bit_index / 8;  // byte holding the MSB of this code
  const int bit_off      = bit_index % 8;  // 0 => starts at MSB of bytes[byte_idx]

  // Build a 16-bit big-endian window: [bytes[byte_idx] | bytes[byte_idx+1]]
  const uint32_t b0  = (byte_idx < bytes_len) ? bytes[byte_idx] : 0u;
  const uint32_t b1  = (byte_idx + 1 < bytes_len) ? bytes[byte_idx + 1] : 0u;
  const uint32_t val = (b0 << 8) | b1;

  // Align MSB-first window so that the ex_bits land in the LSBs.
  // Example: ex_bits=3, bit_off=0 -> take bits 15..13 -> shift by 13.
  const int shift = 16 - ex_bits - bit_off;
  return (val >> shift) & ((1u << ex_bits) - 1u);
}

void print_first_vector(uint8_t* ex_data_, float* ex_factors_, size_t dim, int ex_bits)
{
  // 1) Number of bytes for the bit-packed codes (ceil).
  const size_t code_bytes = (dim * ex_bits + 7) / 8;

  // If your format pads to 4-byte alignment before the floats, use this instead:
  // const size_t code_bytes_padded = (code_bytes + 3) & ~size_t(3);
  // and then read floats from ex_data_ + code_bytes_padded.
  const uint8_t* ex_code = reinterpret_cast<const uint8_t*>(ex_data_);

  // 2) Safely read the floats (avoid potential unaligned pointer UB).
  const float* f_exadd     = reinterpret_cast<const float*>(ex_factors_);
  const float* f_exrescale = reinterpret_cast<const float*>(ex_factors_ + 1);

  // 3) Print first 20 codes (MSB-first, left->right within each byte).
  std::cout << "ExCode (first 20 dims): ";
  const int to_print = static_cast<int>(std::min<size_t>(20, dim));
  for (int i = 0; i < to_print; ++i) {
    uint32_t code = extract_code_msb(ex_code, code_bytes, static_cast<size_t>(i), ex_bits);
    std::cout << code << (i + 1 == to_print ? '\n' : ' ');
  }

  std::cout << "F_exadd     = " << *f_exadd << "\n";
  std::cout << "F_exrescale = " << *f_exrescale << "\n";
}

void IVFGPU::save(const char* filename) const
{
  RAFT_EXPECTS(num_centroids > 0, "IVF index has not been constructed");

  cuvs::util::kvikio_ofstream output(filename);
  RAFT_EXPECTS(output, "failed to open file for saving: %s", filename);

  auto write_exact = [&](const void* ptr, size_t n_bytes) {
    output.write(reinterpret_cast<const char*>(ptr), n_bytes);
    RAFT_EXPECTS(static_cast<bool>(output), "write failed to: %s", filename);
  };

  // Save meta data.
  write_exact(&num_vectors, sizeof(size_t));
  write_exact(&num_dimensions, sizeof(size_t));
  write_exact(&num_centroids, sizeof(size_t));
  write_exact(&ex_bits, sizeof(size_t));
  // Write legacy batch_flag=true for backward compatibility
  bool legacy_batch_flag = true;
  write_exact(&legacy_batch_flag, sizeof(bool));
  write_exact(DQ->get_query_scaling_factor(), sizeof(DataQuantizerGPU::FastQuantizeFactors));

  // Save number of vectors of each cluster.
  std::vector<GPUClusterMeta> h_cluster_meta(num_centroids);
  raft::copy(h_cluster_meta.data(), cluster_meta_.data_handle(), num_centroids, stream_);
  std::vector<size_t> cluster_sizes(num_centroids);
  raft::resource::sync_stream(handle_);
  for (size_t i = 0; i < num_centroids; i++) {
    cluster_sizes[i] = h_cluster_meta[i].num;
  }
  write_exact(cluster_sizes.data(), sizeof(size_t) * num_centroids);

  // Save rotator.
  this->rotator().save(handle_, output);

  // Save initializer data.
  this->initializer->SaveCentroids(output, filename);

  // Compute sizes for device arrays.
  size_t short_data_size = GetShortDataBytesSimple();
  size_t long_code_size  = GetLongCodeBytes();
  size_t ex_factor_size  = GetExFactorBytes();
  size_t ids_size        = GetPIDsBytes();
  // for batch data
  size_t short_factors_size = GetShortDataFactorBytesBatch();

  raft::resource::sync_stream(handle_);

  // short_data_ is held in memory as per-cluster SoA (uint32s_per_vector x cluster_size), which is
  // the layout the search kernels consume. On disk we use the per-vector-contiguous (AoS) layout
  // that load_transposed() expects, so transpose each cluster back to AoS before writing. This is
  // the exact inverse of read_into_device_host_transposed_short() and avoids a double-transpose on
  // the save->load round-trip.
  {
    const size_t bytes_per_vector = DQ->block_bytes();
    RAFT_EXPECTS(bytes_per_vector % sizeof(uint32_t) == 0,
                 "short-code vector size must be a multiple of uint32_t");
    const size_t uint32s_per_vector = bytes_per_vector / sizeof(uint32_t);
    const size_t max_cluster_size =
      cluster_sizes.empty() ? 0 : *std::max_element(cluster_sizes.begin(), cluster_sizes.end());
    RAFT_EXPECTS(max_cluster_size <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                   uint32s_per_vector <= static_cast<size_t>(std::numeric_limits<int>::max()),
                 "short-code matrix dimensions exceed the transpose API limit");
    auto sequential =
      raft::make_device_vector<uint32_t, int64_t>(handle_, max_cluster_size * uint32s_per_vector);
    raft::resource::sync_stream(handle_);

    size_t src_offset    = 0;
    size_t bytes_written = 0;
    for (size_t cluster_id = 0; cluster_id < num_centroids; cluster_id++) {
      const size_t cluster_size = cluster_sizes[cluster_id];
      if (cluster_size == 0) continue;

      const size_t cluster_bytes = cluster_size * bytes_per_vector;
      raft::linalg::transpose(handle_,
                              const_cast<uint32_t*>(short_data_.data_handle()) + src_offset,
                              sequential.data_handle(),
                              static_cast<int>(uint32s_per_vector),
                              static_cast<int>(cluster_size),
                              stream_);
      raft::resource::sync_stream(handle_);

      output.write_device(sequential.data_handle(), cluster_bytes);

      src_offset += cluster_bytes / sizeof(uint32_t);
      bytes_written += cluster_bytes;
    }
    RAFT_EXPECTS(bytes_written == short_data_size,
                 "short-code payload size mismatch on save: expected %zu bytes, wrote %zu",
                 short_data_size,
                 bytes_written);
  }
  output.write_device(short_factors_batch_.data_handle(), short_factors_size);
  output.write_device(long_code_.data_handle(), long_code_size);
  output.write_device(ex_factor_.data_handle(), ex_factor_size);
  output.write_device(ids_.data_handle(), ids_size);

  output.close();
  RAFT_EXPECTS(output, "write failed to: %s", filename);
}

/**
 * @brief Build GPUClusterMeta from counts and offsets on GPU
 */
__global__ void build_cluster_meta_kernel(IVFGPU::GPUClusterMeta* d_cluster_meta,
                                          const unsigned long long* d_counts,
                                          const size_t* d_offsets,
                                          size_t num_centroids)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_centroids) {
    d_cluster_meta[idx] = IVFGPU::GPUClusterMeta(d_counts[idx], d_offsets[idx]);
  }
}

/**
 * @brief Scatter PIDs into flat array based on cluster assignment
 * Uses atomics to determine write position within each cluster
 */
__global__ void scatter_pids_kernel(PID* d_flat_pids,
                                    const PID* d_cluster_ids,
                                    const size_t* d_offsets,
                                    size_t* d_atomic_counters,  // Initialized to d_offsets values
                                    size_t num_vectors)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_vectors) {
    PID cid = d_cluster_ids[idx];
    // Atomically get position and increment
    size_t pos       = atomicAdd((unsigned long long*)&d_atomic_counters[cid], 1ULL);
    d_flat_pids[pos] = static_cast<PID>(idx);
  }
}

// ============================================
// Modified construct function
// ============================================

void IVFGPU::construct_on_gpu(const float* device_data,
                              const float* device_centroids,
                              const PID* device_cluster_ids,
                              bool fast_quantize)
{
  DQ->fast_quantize_flag = fast_quantize;

  // pre-compute rescaling factors for search
  DQ->compute_query_scaling_factors(this->num_padded_dim);

  // compute rescaling factors for query if needed
  if (DQ->fast_quantize_flag) { DQ->compute_quantize_scaling_factors(); }

  if (DQ->fast_quantize_flag) {
    if (ex_bits == 3) {
      DQ->set_quantize_scaling_factors(DQ->get_query_scaling_factor()->const_scaling_factor_4bit);
    } else if (ex_bits == 7) {
      DQ->set_quantize_scaling_factors(DQ->get_query_scaling_factor()->const_scaling_factor_8bit);
    } else {
      DQ->compute_quantize_scaling_factors();
    }
  }

  // -------------------------
  // 2. Validate cluster IDs on GPU
  // -------------------------
  int block_size = 256;
  int num_blocks = (num_vectors + block_size - 1) / block_size;

  // -------------------------
  // 9. Allocate device memory for IVF arrays
  // -------------------------
  std::vector<size_t> counts(num_centroids);  // counts is not used here
  AllocateDeviceMemory();

  // -------------------------
  // 3. Compute histogram (cluster sizes) on GPU using CUB
  // -------------------------

  // Use CUB's DeviceHistogram
  using CounterT = unsigned long long;

  int num_levels  = num_centroids + 1;
  int lower_level = 0;
  int upper_level = num_centroids;

  rmm::device_uvector<CounterT> d_histogram(num_centroids, stream_);

  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(nullptr,
                                                    temp_storage_bytes,
                                                    device_cluster_ids,
                                                    d_histogram.data(),
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    num_vectors,
                                                    stream_));

  {
    rmm::device_buffer d_temp_storage(temp_storage_bytes, stream_);
    RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(d_temp_storage.data(),
                                                      temp_storage_bytes,
                                                      device_cluster_ids,
                                                      d_histogram.data(),
                                                      num_levels,
                                                      lower_level,
                                                      upper_level,
                                                      num_vectors,
                                                      stream_));
  }

  // -------------------------
  // 4. Compute prefix sum (offsets) on GPU using CUB
  // -------------------------
  rmm::device_uvector<size_t> d_offsets(num_centroids + 1, stream_);

  temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
    nullptr, temp_storage_bytes, d_histogram.data(), d_offsets.data(), num_centroids, stream_));

  {
    rmm::device_buffer d_temp_storage(temp_storage_bytes, stream_);
    RAFT_CUDA_TRY(cub::DeviceScan::ExclusiveSum(d_temp_storage.data(),
                                                temp_storage_bytes,
                                                d_histogram.data(),
                                                d_offsets.data(),
                                                num_centroids,
                                                stream_));
  }

  // Set the last offset element
  raft::copy(d_offsets.data() + num_centroids, &num_vectors, 1, stream_);

  // -------------------------
  // 5. Build cluster metadata on GPU
  // -------------------------
  GPUClusterMeta* d_cluster_meta_temp = cluster_meta_.data_handle();

  num_blocks = (num_centroids + block_size - 1) / block_size;
  build_cluster_meta_kernel<<<num_blocks, block_size, 0, stream_>>>(
    d_cluster_meta_temp, d_histogram.data(), d_offsets.data(), num_centroids);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // -------------------------
  // 6. Scatter PIDs to flat array on GPU
  // -------------------------
  PID* d_flat_pids = ids_.data_handle();

  // Create atomic counters initialized with offsets
  rmm::device_uvector<size_t> d_atomic_counters(num_centroids, stream_);
  raft::copy(d_atomic_counters.data(), d_offsets.data(), num_centroids, stream_);

  num_blocks = (num_vectors + block_size - 1) / block_size;
  scatter_pids_kernel<<<num_blocks, block_size, 0, stream_>>>(
    d_flat_pids, device_cluster_ids, d_offsets.data(), d_atomic_counters.data(), num_vectors);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // -------------------------
  // 7. Copy cluster metadata back to host
  // -------------------------
  std::vector<GPUClusterMeta> h_cluster_meta(num_centroids);
  raft::copy(h_cluster_meta.data(), d_cluster_meta_temp, num_centroids, stream_);
  raft::resource::sync_stream(handle_);

  // -------------------------
  // 10. Allocate and process rotated centroids
  // -------------------------
  rmm::device_uvector<float> d_rotated_centroids(num_centroids * num_padded_dim, stream_);

  // -------------------------
  // 10.5 get max cluster length and allocate temporary buffer for quantization
  // Do note that this update will disable the ability of quantization with multiple streams
  // -------------------------
  max_cluster_length = 0;
  for (const auto& meta : h_cluster_meta) {
    max_cluster_length = std::max(max_cluster_length, meta.num);
  }
  DQ->alloc_buffers(max_cluster_length);

  // Process clusters sequentially
  for (size_t i = 0; i < num_centroids; ++i) {
    const float* cur_centroid = device_centroids + i * num_dimensions;
    float* cur_rotated_c      = d_rotated_centroids.data() + i * num_padded_dim;
    GPUClusterMeta& cp        = h_cluster_meta[i];
    quantize_cluster(cp, device_data, cur_centroid, cur_rotated_c);
  }

  // Add rotated centroids
  initializer->AddVectors(d_rotated_centroids.data());

  raft::resource::sync_stream(handle_);
}

void IVFGPU::construct_on_gpu_streaming(const float* host_data,
                                        const float* device_centroids,
                                        const PID* device_cluster_ids,
                                        bool fast_quantize,
                                        size_t batch_size_vectors)
{
  DQ->fast_quantize_flag = fast_quantize;

  // pre-compute rescaling factors for search
  DQ->compute_query_scaling_factors(this->num_padded_dim);

  // compute rescaling factors for query if needed
  if (DQ->fast_quantize_flag) { DQ->compute_quantize_scaling_factors(); }

  if (DQ->fast_quantize_flag) {
    if (ex_bits == 3) {
      DQ->set_quantize_scaling_factors(DQ->get_query_scaling_factor()->const_scaling_factor_4bit);
    } else if (ex_bits == 7) {
      DQ->set_quantize_scaling_factors(DQ->get_query_scaling_factor()->const_scaling_factor_8bit);
    } else {
      DQ->compute_quantize_scaling_factors();
    }
  }

  // -------------------------
  // 1. Use cluster IDs already on device
  // -------------------------

  // -------------------------
  // 2. Allocate device memory for IVF arrays
  // -------------------------
  AllocateDeviceMemory();

  // -------------------------
  // 3. Compute histogram (cluster sizes) on GPU using CUB
  // -------------------------

  using CounterT = unsigned long long;

  int num_levels  = num_centroids + 1;
  int lower_level = 0;
  int upper_level = num_centroids;

  auto d_histogram = raft::make_device_vector<CounterT, int64_t>(handle_, num_centroids);

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(d_temp_storage,
                                                    temp_storage_bytes,
                                                    device_cluster_ids,
                                                    d_histogram.data_handle(),
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    num_vectors,
                                                    stream_));

  auto d_temp_storage_vec = raft::make_device_vector<uint8_t, int64_t>(handle_, temp_storage_bytes);
  d_temp_storage          = d_temp_storage_vec.data_handle();

  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(d_temp_storage,
                                                    temp_storage_bytes,
                                                    device_cluster_ids,
                                                    d_histogram.data_handle(),
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    num_vectors,
                                                    stream_));

  // -------------------------
  // 4. Compute prefix sum (offsets) on GPU using CUB
  // -------------------------
  auto d_offsets = raft::make_device_vector<size_t, int64_t>(handle_, num_centroids + 1);

  d_temp_storage     = nullptr;
  temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                              temp_storage_bytes,
                                              d_histogram.data_handle(),
                                              d_offsets.data_handle(),
                                              num_centroids,
                                              stream_));

  d_temp_storage_vec = raft::make_device_vector<uint8_t, int64_t>(handle_, temp_storage_bytes);
  d_temp_storage     = d_temp_storage_vec.data_handle();

  RAFT_CUDA_TRY(cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                              temp_storage_bytes,
                                              d_histogram.data_handle(),
                                              d_offsets.data_handle(),
                                              num_centroids,
                                              stream_));

  // Set the last offset element
  raft::copy(d_offsets.data_handle() + num_centroids, &num_vectors, 1, stream_);

  // -------------------------
  // 5. Build cluster metadata on GPU
  // -------------------------
  GPUClusterMeta* d_cluster_meta_temp = cluster_meta_.data_handle();

  int block_size = 256;
  int num_blocks = (num_centroids + block_size - 1) / block_size;
  build_cluster_meta_kernel<<<num_blocks, block_size, 0, stream_>>>(
    d_cluster_meta_temp, d_histogram.data_handle(), d_offsets.data_handle(), num_centroids);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // -------------------------
  // 6. Scatter PIDs to flat array on GPU
  // -------------------------
  PID* d_flat_pids = ids_.data_handle();

  // Create atomic counters initialized with offsets
  auto d_atomic_counters = raft::make_device_vector<size_t, int64_t>(handle_, num_centroids);
  raft::copy(d_atomic_counters.data_handle(), d_offsets.data_handle(), num_centroids, stream_);

  num_blocks = (num_vectors + block_size - 1) / block_size;
  scatter_pids_kernel<<<num_blocks, block_size, 0, stream_>>>(d_flat_pids,
                                                              device_cluster_ids,
                                                              d_offsets.data_handle(),
                                                              d_atomic_counters.data_handle(),
                                                              num_vectors);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // -------------------------
  // 7. Copy cluster metadata back to host for batching
  // -------------------------
  std::vector<GPUClusterMeta> h_cluster_meta(num_centroids);
  raft::copy(h_cluster_meta.data(), d_cluster_meta_temp, num_centroids, stream_);
  raft::resource::sync_stream(handle_);

  // -------------------------
  // 8. Compute max cluster length and allocate buffers
  // -------------------------
  max_cluster_length = 0;
  for (size_t i = 0; i < num_centroids; ++i) {
    max_cluster_length = std::max(max_cluster_length, h_cluster_meta[i].num);
  }
  RAFT_EXPECTS(max_cluster_length <= batch_size_vectors,
               "max cluster size (%zu) exceeds batch_size_vectors (%zu); "
               "increase batch_size_vectors so every cluster fits in one batch",
               max_cluster_length,
               batch_size_vectors);
  DQ->alloc_buffers(max_cluster_length);

  // -------------------------
  // 9. Allocate rotated centroids buffer
  // -------------------------
  auto d_rotated_centroids =
    raft::make_device_vector<float, int64_t>(handle_, num_centroids * num_padded_dim);

  // -------------------------
  // 10. Batch clusters and stream to GPU
  // -------------------------
  // Preallocate batch buffers (reused across all batch iterations)
  auto batch_pids = raft::make_host_vector<PID, int64_t>(batch_size_vectors);
  auto batch_data = raft::make_host_vector<float, int64_t>(batch_size_vectors * num_dimensions);
  auto d_batch_data =
    raft::make_device_vector<float, int64_t>(handle_, batch_size_vectors * num_dimensions);

  size_t batch_count = 0;

  // Calculate number of threads for host data gathering
  const size_t num_threads = omp_get_max_threads();

  size_t cluster_idx = 0;

  while (cluster_idx < num_centroids) {
    batch_count++;
    // Determine batch: complete clusters only
    size_t batch_start_cluster = cluster_idx;
    size_t batch_vectors       = 0;
    size_t batch_start_offset  = h_cluster_meta[cluster_idx].start_index;

    while (cluster_idx < num_centroids &&
           batch_vectors + h_cluster_meta[cluster_idx].num <= batch_size_vectors) {
      batch_vectors += h_cluster_meta[cluster_idx].num;
      cluster_idx++;
    }

    if (batch_vectors == 0) break;  // No more clusters to process

    // -------------------------
    // 11. Gather and transfer batch data to GPU using reordered PIDs
    // -------------------------

    // First, copy the reordered PIDs for this batch to host (reuse preallocated buffer)
    raft::copy(batch_pids.data_handle(), d_flat_pids + batch_start_offset, batch_vectors, stream_);
    raft::resource::sync_stream(handle_);

    // Gather data on host using the reordered PIDs (reuse preallocated buffer)
    // Use OpenMP parallel region to leverage persistent thread pool across batches
#pragma omp parallel num_threads(num_threads)
    {
      size_t tid        = omp_get_thread_num();
      size_t chunk_size = batch_vectors / num_threads;
      size_t start      = tid * chunk_size;
      size_t end        = (tid == num_threads - 1) ? batch_vectors : start + chunk_size;

      for (size_t i = start; i < end; ++i) {
        std::memcpy(&batch_data.data_handle()[i * num_dimensions],
                    &host_data[batch_pids.data_handle()[i] * num_dimensions],
                    num_dimensions * sizeof(float));
      }
    }
    // OpenMP implicit barrier ensures all threads complete before continuing

    // Transfer batch to GPU (reuse preallocated buffer)
    raft::copy(d_batch_data.data_handle(),
               batch_data.data_handle(),
               batch_vectors * num_dimensions,
               stream_);

    // -------------------------
    // 12. Quantize each cluster in the batch
    // -------------------------

    size_t batch_offset = 0;
    for (size_t c = batch_start_cluster; c < cluster_idx; ++c) {
      size_t cluster_size = h_cluster_meta[c].num;
      if (cluster_size == 0) continue;

      const float* cur_centroid = device_centroids + c * num_dimensions;
      float* cur_rotated_c      = d_rotated_centroids.data_handle() + c * num_padded_dim;
      GPUClusterMeta& cp        = h_cluster_meta[c];

      // Quantize this cluster's data (contiguous in batch, SoA layout)
      const float* cluster_data = d_batch_data.data_handle() + batch_offset * num_dimensions;

      // Quantize using batch-optimized path
      DQ->quantize_batch_opt_contiguous(cluster_data,
                                        cur_centroid,
                                        cluster_size,
                                        this->rotator(),
                                        cp.first_block(*this),
                                        cp.short_factor_batch(*this, 0),
                                        cp.long_code(*this, 0, DQ->long_code_length()),
                                        reinterpret_cast<float*>(cp.ex_factor(*this, 0)),
                                        cur_rotated_c);

      batch_offset += cluster_size;
    }
  }

  // -------------------------
  // 13. Add rotated centroids
  // -------------------------
  initializer->AddVectors(d_rotated_centroids.data_handle());

  raft::resource::sync_stream(handle_);
}

void IVFGPU::quantize_cluster(GPUClusterMeta& cp,
                              const float* d_data,      // device pointer
                              const float* d_centroid,  // device pointer
                              float* d_rotated_c) const
{  // device pointer for rotated centroid
  size_t num = cp.num;
  // Copy the IDs for this cluster into device memory.
  // Note: cp.ids(this) returns ids_ + cp.start_index.
  const PID* idp = cp.ids(*this);

  // Call the GPU quantization function (SoA layout).
  DQ->quantize_batch_opt(d_data,
                         d_centroid,
                         idp,
                         num,
                         this->rotator(),
                         cp.first_block(*this),
                         cp.short_factor_batch(*this, 0),
                         cp.long_code(*this, 0, DQ->long_code_length()),
                         reinterpret_cast<float*>(cp.ex_factor(*this, 0)),
                         d_rotated_c);
}

// Optimized kernel to prepare keys and values from d_raft_idx
__global__ void prepare_keys_values(
  const int* d_raft_idx, int* d_cluster_keys, int* d_query_values, int batch_size, int nprobe)
{
  int tid         = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pairs = batch_size * nprobe;

  if (tid < total_pairs) {
    // Coalesced memory read
    d_cluster_keys[tid] = d_raft_idx[tid];
    d_query_values[tid] = tid / nprobe;  // query index
  }
}

// Optimized kernel to combine sorted keys/values into pairs
__global__ void combine_to_pairs(const int* d_sorted_clusters,
                                 const int* d_sorted_queries,
                                 ClusterQueryPair* d_sorted_pairs,
                                 int total_pairs)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < total_pairs) {
    // Coalesced writes to pair structure
    d_sorted_pairs[tid].cluster_idx = d_sorted_clusters[tid];
    d_sorted_pairs[tid].query_idx   = d_sorted_queries[tid];
  }
}

// Sorting cluster_query pairs using cub sort
void sort_cluster_query_pairs(raft::resources const& handle,
                              int* d_raft_idx,                   // Input: cluster indices
                              ClusterQueryPair* d_sorted_pairs,  // Output: sorted pairs
                              int batch_size,
                              int nprobe)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  int total_pairs     = batch_size * nprobe;

  // Allocate temporary arrays for sorting
  auto d_cluster_keys    = raft::make_device_vector<int, int64_t>(handle, total_pairs);
  auto d_query_values    = raft::make_device_vector<int, int64_t>(handle, total_pairs);
  auto d_sorted_clusters = raft::make_device_vector<int, int64_t>(handle, total_pairs);
  auto d_sorted_queries  = raft::make_device_vector<int, int64_t>(handle, total_pairs);

  // Prepare data for sorting
  int threads_per_block = 256;
  int num_blocks        = (total_pairs + threads_per_block - 1) / threads_per_block;

  prepare_keys_values<<<num_blocks, threads_per_block, 0, stream>>>(
    d_raft_idx, d_cluster_keys.data_handle(), d_query_values.data_handle(), batch_size, nprobe);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Determine temporary storage requirements for CUB
  // ---- sort by cluster id
  size_t temp_storage_bytes = 0;

  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(nullptr,
                                                temp_storage_bytes,
                                                d_cluster_keys.data_handle(),
                                                d_sorted_clusters.data_handle(),
                                                d_query_values.data_handle(),
                                                d_sorted_queries.data_handle(),
                                                total_pairs,
                                                0,
                                                sizeof(int) * 8,
                                                stream));

  rmm::device_buffer d_temp_storage(temp_storage_bytes, stream);

  // Perform radix sort
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                                temp_storage_bytes,
                                                d_cluster_keys.data_handle(),
                                                d_sorted_clusters.data_handle(),
                                                d_query_values.data_handle(),
                                                d_sorted_queries.data_handle(),
                                                total_pairs,
                                                0,
                                                sizeof(int) * 8,
                                                stream));

  // Combine sorted results into pairs
  combine_to_pairs<<<num_blocks, threads_per_block, 0, stream>>>(
    d_sorted_clusters.data_handle(), d_sorted_queries.data_handle(), d_sorted_pairs, total_pairs);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

// Generic warp-level kernel: Multiple queries per block, one query per warp
template <typename T>
__global__ void computeQueryFactorsWarpKernel(const T* d_query,    // num_queries x D matrix
                                              float* d_G_k1xSumq,  // output: num_queries array
                                              float* d_G_kbxSumq,  // output: num_queries array
                                              size_t num_queries,
                                              size_t D,        // dimension of each query
                                              size_t ex_bits)  // extra bits parameter
{
  // Calculate which query this warp is processing
  const int warp_id         = threadIdx.x / 32;
  const int lane_id         = threadIdx.x % 32;
  const int warps_per_block = blockDim.x / 32;
  const int query_idx       = blockIdx.x * warps_per_block + warp_id;

  // Early exit if beyond valid queries
  if (query_idx >= num_queries) return;

  // Compute constants
  float c_1 = -static_cast<float>((1 << 1) - 1) / 2.0f;  // -0.5f
  float c_b = -static_cast<float>((1 << (ex_bits + 1)) - 1) / 2.0f;

  // Pointer to the current query
  const T* query = d_query + query_idx * D;

  // Each thread in the warp accumulates multiple elements
  T sum = 0;

  // Coalesced memory access: warp reads consecutive elements
  for (int i = lane_id; i < D; i += 32) {
    sum += query[i];
  }

  // Warp-level reduction using shuffle instructions
  sum = warpReduceSum(sum);

  // Lane 0 writes the result
  if (lane_id == 0) {
    d_G_k1xSumq[query_idx] = sum * c_1;
    d_G_kbxSumq[query_idx] = sum * c_b;
  }
}

// Wrapper function
template <typename T>
void computeQueryFactors(raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
                         raft::device_vector_view<float, int64_t> d_G_k1xSumq,
                         raft::device_vector_view<float, int64_t> d_G_kbxSumq,
                         size_t ex_bits,
                         cudaStream_t stream)
{
  const size_t num_queries = queries.extent(0);
  const size_t D           = queries.extent(1);

  // Use 256 threads per block (8 warps per block)
  const int threads_per_block = 256;
  const int warps_per_block   = threads_per_block / 32;
  const int blocks            = (num_queries + warps_per_block - 1) / warps_per_block;

  computeQueryFactorsWarpKernel<T>
    <<<blocks, threads_per_block, 0, stream>>>(queries.data_handle(),
                                               d_G_k1xSumq.data_handle(),
                                               d_G_kbxSumq.data_handle(),
                                               num_queries,
                                               D,
                                               ex_bits);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

// normal way to first sort (cluster, query) pairs, then use a CTA to do the search
void IVFGPU::PrepareClusterSearchInputs(
  raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
  size_t nprobe,
  SearcherGPU& searcher,
  raft::device_vector<ClusterQueryPair, int64_t>& d_sorted_pairs,
  raft::device_vector<float, int64_t>& d_G_k1xSumq,
  raft::device_vector<float, int64_t>& d_G_kbxSumq)
{
  raft::resources const& searcher_handle = searcher.get_handle();
  rmm::cuda_stream_view searcher_stream  = searcher.get_stream();
  const size_t batch_size                = queries.extent(0);

  // Compute ||q - c||^2 = -2 * q . c + ||q||^2 + ||c||^2 into centroid_distances:
  //  (1) query norms (squared L2, no sqrt - default fin_op on L2Norm returns sum of squares;
  //      centroid norms were precomputed in compute_centroid_norms() and cached in centroid_norms_)
  //  (2) outer_add writes (q_norm + c_norm) into centroid_distances, overwriting the prior
  //      uninitialized contents of the freshly-allocated SearcherGPU scratch buffer
  //  (3) matmul with beta=1 accumulates -2 * Q * C^T on top
  raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
    searcher_handle,
    queries,
    raft::make_device_vector_view<float, int64_t>(searcher.get_q_norms(), batch_size));
  cuvs::spatial::knn::detail::utils::outer_add(searcher.get_q_norms(),
                                               batch_size,
                                               centroid_norms_.data_handle(),
                                               num_centroids,
                                               searcher.get_centroid_distances(),
                                               searcher_stream);
  const float alpha = -2.f, beta = 1.f;
  raft::linalg::detail::matmul</* DevicePointerMode = */ false>(searcher_handle,
                                                                /* trans_a = */ true,
                                                                /* trans_b = */ false,
                                                                num_centroids,
                                                                batch_size,
                                                                num_padded_dim,
                                                                &alpha,
                                                                initializer->GetCentroid(0),
                                                                num_padded_dim,
                                                                queries.data_handle(),
                                                                num_padded_dim,
                                                                &beta,
                                                                searcher.get_centroid_distances(),
                                                                num_centroids);

  // Step 4: select top-nprobe clusters per query
  auto d_raft_vals = raft::make_device_matrix<float, int64_t>(searcher_handle, batch_size, nprobe);
  auto d_raft_idx  = raft::make_device_matrix<int, int64_t>(searcher_handle, batch_size, nprobe);
  auto in_view     = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    searcher.get_centroid_distances(), batch_size, num_centroids);
  cuvs::selection::select_k(searcher_handle,
                            in_view,
                            std::nullopt,
                            d_raft_vals.view(),
                            d_raft_idx.view(),
                            /*select_min=*/true,
                            /*sorted=*/true);

  // Step 5: sort (cluster, query) pairs
  d_sorted_pairs =
    raft::make_device_vector<ClusterQueryPair, int64_t>(searcher_handle, batch_size * nprobe);
  sort_cluster_query_pairs(
    searcher_handle, d_raft_idx.data_handle(), d_sorted_pairs.data_handle(), batch_size, nprobe);

  // Step 6: compute query factors
  d_G_k1xSumq = raft::make_device_vector<float, int64_t>(searcher_handle, batch_size);
  d_G_kbxSumq = raft::make_device_vector<float, int64_t>(searcher_handle, batch_size);
  computeQueryFactors<float>(
    queries, d_G_k1xSumq.view(), d_G_kbxSumq.view(), ex_bits, searcher_stream);

  // Sync here to ensure all outputs are visible before SearcherGPU reads them.
  raft::resource::sync_stream(searcher_handle);
}

void IVFGPU::BatchClusterSearch(
  raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
  size_t k,
  size_t nprobe,
  SearcherGPU& searcher,
  size_t batch_size,
  raft::device_matrix_view<float, int64_t, raft::row_major> d_final_dists,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> d_final_pids)
{
  auto d_sorted_pairs =
    raft::make_device_vector<ClusterQueryPair, int64_t>(searcher.get_handle(), 0);
  auto d_G_k1xSumq = raft::make_device_vector<float, int64_t>(searcher.get_handle(), 0);
  auto d_G_kbxSumq = raft::make_device_vector<float, int64_t>(searcher.get_handle(), 0);
  PrepareClusterSearchInputs(queries, nprobe, searcher, d_sorted_pairs, d_G_k1xSumq, d_G_kbxSumq);
  searcher.SearchClusterQueryPairs(*this,
                                   cluster_meta_.data_handle(),
                                   d_sorted_pairs.data_handle(),
                                   batch_size,
                                   queries,
                                   d_G_k1xSumq.data_handle(),
                                   d_G_kbxSumq.data_handle(),
                                   nprobe,
                                   k,
                                   d_final_dists,
                                   d_final_pids);
}

void IVFGPU::BatchClusterSearchLUT16(
  raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
  size_t k,
  size_t nprobe,
  SearcherGPU& searcher,
  size_t batch_size,
  raft::device_matrix_view<float, int64_t, raft::row_major> d_final_dists,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> d_final_pids)
{
  auto d_sorted_pairs =
    raft::make_device_vector<ClusterQueryPair, int64_t>(searcher.get_handle(), 0);
  auto d_G_k1xSumq = raft::make_device_vector<float, int64_t>(searcher.get_handle(), 0);
  auto d_G_kbxSumq = raft::make_device_vector<float, int64_t>(searcher.get_handle(), 0);
  PrepareClusterSearchInputs(queries, nprobe, searcher, d_sorted_pairs, d_G_k1xSumq, d_G_kbxSumq);
  searcher.SearchClusterQueryPairsSharedMemOpt(*this,
                                               cluster_meta_.data_handle(),
                                               d_sorted_pairs.data_handle(),
                                               batch_size,
                                               queries,
                                               d_G_k1xSumq.data_handle(),
                                               d_G_kbxSumq.data_handle(),
                                               nprobe,
                                               k,
                                               d_final_dists,
                                               d_final_pids);
}

void IVFGPU::BatchClusterSearchQuantizeQuery(
  raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
  size_t k,
  size_t nprobe,
  SearcherGPU& searcher,
  size_t batch_size,
  raft::device_matrix_view<float, int64_t, raft::row_major> d_final_dists,
  raft::device_matrix_view<uint32_t, int64_t, raft::row_major> d_final_pids,
  int query_bits)
{
  auto d_sorted_pairs =
    raft::make_device_vector<ClusterQueryPair, int64_t>(searcher.get_handle(), 0);
  auto d_G_k1xSumq = raft::make_device_vector<float, int64_t>(searcher.get_handle(), 0);
  auto d_G_kbxSumq = raft::make_device_vector<float, int64_t>(searcher.get_handle(), 0);
  PrepareClusterSearchInputs(queries, nprobe, searcher, d_sorted_pairs, d_G_k1xSumq, d_G_kbxSumq);
  searcher.SearchClusterQueryPairsQuantizeQuery(*this,
                                                cluster_meta_.data_handle(),
                                                d_sorted_pairs.data_handle(),
                                                batch_size,
                                                queries,
                                                d_G_k1xSumq.data_handle(),
                                                d_G_kbxSumq.data_handle(),
                                                nprobe,
                                                k,
                                                d_final_dists,
                                                d_final_pids,
                                                query_bits == 4);
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
