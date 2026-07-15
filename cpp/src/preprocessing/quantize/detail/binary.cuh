/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/preprocessing/quantize/binary.hpp>
#include <raft/core/copy.cuh>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/init.cuh>
#include <raft/matrix/sample_rows.cuh>
#include <raft/random/rng.cuh>
#include <raft/random/sample_without_replacement.cuh>
#include <raft/stats/mean.cuh>
#include <raft/util/cudart_utils.hpp>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace cuvs::preprocessing::quantize::detail {

template <class T>
_RAFT_HOST_DEVICE bool is_positive(const T& a)
{
  return a > 0;
}

template <>
_RAFT_HOST_DEVICE bool is_positive<half>(const half& a)
{
  return is_positive(static_cast<float>(a));
}

template <class T, uint32_t block_size, class pack_t = uint8_t>
RAFT_KERNEL binary_quantization_kernel(const T* const in_ptr,
                                       const uint32_t ldi,
                                       const T* const threshold_ptr,
                                       const size_t dataset_size,
                                       const uint32_t dataset_dim,
                                       pack_t* const out_ptr,
                                       const uint32_t ldo)
{
  constexpr uint32_t warp_size = 32;
  const uint32_t bits_per_pack = sizeof(pack_t) * 8;
  const auto output_dim        = raft::div_rounding_up_safe(dataset_dim, bits_per_pack);

  const auto vector_id = (blockDim.x * blockIdx.x + threadIdx.x) / warp_size;
  if (vector_id >= dataset_size) { return; }

  __shared__ pack_t smem[block_size];
  auto local_smem = smem + (threadIdx.x / warp_size) * warp_size;

  const auto lane_id = threadIdx.x % warp_size;
  for (uint32_t j_offset = 0; j_offset < dataset_dim; j_offset += warp_size * bits_per_pack) {
    // Load dataset vector elements with coalesce access. The mapping of the vector element position
    // and the `pack` register is as follows:
    //
    // lane_id | LSB  (pack(u8))  MSB
    //       0 |  0, 32, 64, ..., 224
    //       1 |  1, 33, 65, ..., 225
    //       ...
    //      31 | 31, 63, 95, ..., 255
    pack_t pack = 0;
    for (uint32_t bit_offset = 0; bit_offset < bits_per_pack; bit_offset++) {
      const auto j = j_offset + lane_id + bit_offset * warp_size;
      if (j < dataset_dim) {
        auto v = in_ptr[static_cast<size_t>(vector_id) * ldi + j];
        if (threshold_ptr != nullptr) { v -= threshold_ptr[j]; }
        if (is_positive(v)) { pack |= (1u << bit_offset); }
      }
    }

    // Store the local result in smem so that the other threads in the same warp can read
    local_smem[lane_id] = pack;

    // Store the result with (a kind of) transposition so that the the coalesce access can be used.
    // The mapping of the result `pack` register bit position and (smem_index, bit_position) is as
    // follows:
    //
    // lane_id | LSB          (pack(u8))         MSB
    //       0 | ( 0,0), ( 1,0), ( 2,0), ..., ( 7,0)
    //       1 | ( 8,0), ( 9,0), (10,0), ..., (15,0)
    //       ...
    //       4 | ( 0,1), ( 1,1), ( 2,1), ..., ( 7,1)
    //       ...
    //      31 | (24,7), (25,7), (26,7), ..., (31,7)
    //
    // The `bit_position`-th bit of `local_smem[smem_index]` is copied to the corresponding `pack`
    // bit. By this mapping, the quantization result of 8*i-th ~ (8*(i+1)-1)-th vector elements is
    // stored by the lane_id=i thread.
    pack                    = 0;
    const auto smem_start_i = (lane_id % (warp_size / bits_per_pack)) * bits_per_pack;
    const auto mask         = 1u << (lane_id / (warp_size / bits_per_pack));
    for (uint32_t j = 0; j < bits_per_pack; j++) {
      if (local_smem[smem_start_i + j] & mask) { pack |= (1u << j); }
    }

    const auto out_j = j_offset / bits_per_pack + lane_id;
    if (out_j < output_dim) { out_ptr[vector_id * ldo + out_j] = pack; }
  }
}

RAFT_KERNEL mean_f16_in_f32_kernel_sum(float* const result_ptr,
                                       const half* const src_ptr,
                                       const uint32_t ld,
                                       const uint32_t dataset_dim,
                                       const size_t dataset_size,
                                       const size_t dataset_size_per_cta)
{
  const auto dim_i = threadIdx.x + blockIdx.x * blockDim.x;
  if (dim_i >= dataset_dim) { return; }

  const auto start_dataset_index = blockIdx.y * dataset_size_per_cta;
  const auto end_dataset_index   = std::min((blockIdx.y + 1) * dataset_size_per_cta, dataset_size);

  if (start_dataset_index >= dataset_size) { return; }

  float sum = 0;
  for (size_t dataset_index = start_dataset_index; dataset_index < end_dataset_index;
       dataset_index++) {
    sum += static_cast<float>(src_ptr[dim_i + dataset_index * ld]);
  }
  atomicAdd(result_ptr + dim_i, sum);
}

RAFT_KERNEL mean_f16_in_f32_kernel_div(half* const result_ptr,
                                       const float* const src_ptr,
                                       const uint32_t dataset_dim,
                                       const size_t dataset_size)
{
  const auto dim_i = threadIdx.x + blockIdx.x * blockDim.x;
  if (dim_i >= dataset_dim) { return; }

  result_ptr[dim_i] = src_ptr[dim_i] / dataset_size;
}

void mean_f16_in_f32(raft::resources const& res,
                     half* const result_ptr,
                     const half* const src_ptr,
                     const uint32_t ld,
                     const uint32_t dataset_dim,
                     const size_t dataset_size,
                     cudaStream_t cuda_stream)
{
  auto mr = raft::resource::get_workspace_resource_ref(res);
  auto f32_result_vec =
    raft::make_device_mdarray<float, int64_t>(res, mr, raft::make_extents<int64_t>(dataset_dim));
  raft::matrix::fill(res, f32_result_vec.view(), float(0));

  constexpr uint32_t dataset_size_per_cta = 2048;
  constexpr uint32_t block_size           = 256;
  const dim3 grid_size(raft::div_rounding_up_safe<size_t>(dataset_dim, block_size),
                       raft::div_rounding_up_safe<size_t>(dataset_size, dataset_size_per_cta));
  mean_f16_in_f32_kernel_sum<<<grid_size, block_size, 0, cuda_stream>>>(
    f32_result_vec.data_handle(), src_ptr, ld, dataset_dim, dataset_size, dataset_size_per_cta);
  mean_f16_in_f32_kernel_div<<<grid_size.x, block_size, 0, cuda_stream>>>(
    result_ptr, f32_result_vec.data_handle(), dataset_dim, dataset_size);
}

template <typename T>
auto train(raft::resources const& res,
           const cuvs::preprocessing::quantize::binary::params& params,
           raft::device_matrix_view<const T, int64_t> dataset)
  -> cuvs::preprocessing::quantize::binary::quantizer<T>
{
  cuvs::preprocessing::quantize::binary::quantizer<T> quantizer(res);
  if (params.threshold == cuvs::preprocessing::quantize::binary::bit_threshold::zero) {
    quantizer.threshold = raft::make_device_vector<T, int64_t>(res, 0);
    return quantizer;
  }

  const size_t dataset_size  = dataset.extent(0);
  const uint32_t dataset_dim = dataset.extent(1);

  quantizer.threshold = raft::make_device_vector<T, int64_t>(res, dataset_dim);
  auto threshold_ptr  = quantizer.threshold.data_handle();

  if (params.threshold == cuvs::preprocessing::quantize::binary::bit_threshold::mean) {
    // stats::mean does not support F16
    if constexpr (!std::is_same_v<T, half>) {
      raft::stats::mean(
        res, dataset, raft::make_device_vector_view<T, int64_t>(threshold_ptr, dataset_dim), false);
    } else {
      // Use a custom CUDA kernel because 1) raft::stats::mean does not support f16 and 2) it is
      // better to use f32 in accumulation data type for a large dataset.
      mean_f16_in_f32(res,
                      threshold_ptr,
                      dataset.data_handle(),
                      dataset_dim,
                      dataset_dim,
                      dataset_size,
                      raft::resource::get_cuda_stream(res));
    }
  } else if (params.threshold ==
             cuvs::preprocessing::quantize::binary::bit_threshold::sampling_median) {
    RAFT_EXPECTS(params.sampling_ratio > 0 && params.sampling_ratio <= 1,
                 "The sampling ratio must be within the range (0, 1].");
    // Make the number of samples odd so that the median is calculated by only sort and memcpy
    const size_t num_samples =
      std::max(
        raft::div_rounding_up_safe(static_cast<size_t>(dataset_size * params.sampling_ratio), 2lu) *
          2lu,
        2lu) -
      1;
    const size_t data_size_per_vector = sizeof(T) * dataset_dim * 2;
    const uint32_t max_dim_chunk =
      std::min(std::max(1lu, raft::resource::get_workspace_free_bytes(res) / data_size_per_vector),
               static_cast<std::size_t>(dataset_dim));

    raft::random::RngState rng(29837lu);
    auto mr                    = raft::resource::get_workspace_resource_ref(res);
    auto sampled_dataset_chunk = raft::make_device_mdarray<T, int64_t>(
      res, mr, raft::make_extents<int64_t>(num_samples, max_dim_chunk));
    auto transposed_sampled_dataset_chunk = raft::make_device_mdarray<T, int64_t>(
      res, mr, raft::make_extents<int64_t>(max_dim_chunk, num_samples));
    for (uint32_t dim_offset = 0; dim_offset < dataset_dim; dim_offset += max_dim_chunk) {
      const auto dim_chunk = std::min(max_dim_chunk, dataset_dim - dim_offset);

      auto sampled_view =
        raft::make_device_matrix_view<T, int64_t>(sampled_dataset_chunk.data_handle(),
                                                  static_cast<int64_t>(num_samples),
                                                  static_cast<int64_t>(dim_chunk));
      raft::matrix::sample_rows(
        res,
        rng,
        raft::make_device_strided_matrix_view<const T, int64_t>(
          dataset.data_handle() + dim_offset, dataset_size, dim_chunk, dataset_dim),
        sampled_view);
      auto transposed_sampled_view = raft::make_device_matrix_view<T, int64_t>(
        transposed_sampled_dataset_chunk.data_handle(), dim_chunk, num_samples);
      raft::linalg::transpose(res, sampled_view, transposed_sampled_view);

      for (uint32_t i = 0; i < dim_chunk; i++) {
        auto start_ptr = transposed_sampled_dataset_chunk.data_handle() + i * num_samples;
        thrust::sort(thrust::device, start_ptr, start_ptr + num_samples);
      }

      raft::copy_matrix(threshold_ptr + dim_offset,
                        size_t(1),
                        sampled_dataset_chunk.data_handle() + (num_samples - 1) / 2,
                        num_samples,
                        size_t(1),
                        dim_chunk,
                        raft::resource::get_cuda_stream(res));
    }
  }
  return quantizer;
}

template <typename T>
auto train(raft::resources const& res,
           const cuvs::preprocessing::quantize::binary::params& params,
           raft::host_matrix_view<const T, int64_t> dataset)
  -> cuvs::preprocessing::quantize::binary::quantizer<T>
{
  cuvs::preprocessing::quantize::binary::quantizer<T> quantizer(res);
  if (params.threshold == cuvs::preprocessing::quantize::binary::bit_threshold::zero) {
    quantizer.threshold = raft::make_device_vector<T, int64_t>(res, 0);
    return quantizer;
  }

  const uint32_t dataset_dim = dataset.extent(1);
  const size_t dataset_size  = dataset.extent(0);
  quantizer.threshold        = raft::make_device_vector<T, int64_t>(res, dataset_dim);

  using compute_t = std::conditional_t<std::is_same_v<half, T>, float, T>;
  std::vector<compute_t> host_threshold_vec(dataset_dim);
  auto threshold_ptr = host_threshold_vec.data();

  if (params.threshold == cuvs::preprocessing::quantize::binary::bit_threshold::mean) {
    for (uint32_t j = 0; j < dataset_dim; j++) {
      threshold_ptr[j] = 0;
    }
#pragma omp parallel for reduction(+ : threshold_ptr[ : dataset_dim])
    for (size_t i = 0; i < dataset_size; i++) {
      for (uint32_t j = 0; j < dataset_dim; j++) {
        threshold_ptr[j] += static_cast<compute_t>(dataset.data_handle()[i * dataset_dim + j]);
      }
    }
    for (uint32_t j = 0; j < dataset_dim; j++) {
      threshold_ptr[j] /= dataset_size;
    }
  } else if (params.threshold ==
             cuvs::preprocessing::quantize::binary::bit_threshold::sampling_median) {
    RAFT_EXPECTS(params.sampling_ratio > 0 && params.sampling_ratio <= 1,
                 "The sampling ratio must be within the range (0, 1].");

    // Make the number of samples odd so that the median is calculated by only sort and memcpy
    const size_t num_samples =
      std::max(
        raft::div_rounding_up_safe(static_cast<size_t>(dataset_size * params.sampling_ratio), 2lu) *
          2lu,
        2lu) -
      1;

    // Calculate stride
    size_t stride_prime_list[] = {611323lu, 611333lu, 611389lu, 611393};
    uint32_t prime_i           = 0;
    while (dataset_size % stride_prime_list[prime_i] == 0) {
      prime_i++;
    }
    const auto stride = stride_prime_list[prime_i];

    // Transposed
    auto sampled_dataset = raft::make_host_matrix<compute_t, int64_t>(dataset_dim, num_samples);
#pragma omp parallel for
    for (size_t out_i = 0; out_i < num_samples; out_i++) {
      const auto in_i = (out_i * stride) % dataset_size;
      for (uint32_t j = 0; j < dataset_dim; j++) {
        sampled_dataset.data_handle()[j * num_samples + out_i] =
          static_cast<compute_t>(dataset.data_handle()[in_i * dataset_dim + j]);
      }
    }

#pragma omp parallel for
    for (uint32_t j = 0; j < dataset_dim; j++) {
      auto start_ptr = sampled_dataset.data_handle() + j * num_samples;
      std::sort(start_ptr, start_ptr + num_samples);
      threshold_ptr[j] = start_ptr[(num_samples - 1) / 2];
    }
  }

  if constexpr (std::is_same_v<T, compute_t>) {
    raft::copy(
      res,
      raft::make_device_vector_view(quantizer.threshold.data_handle(), (int64_t)dataset_dim),
      raft::make_host_vector_view<const compute_t>(host_threshold_vec.data(),
                                                   (int64_t)dataset_dim));
  } else {
    auto mr         = raft::resource::get_workspace_resource_ref(res);
    auto casted_vec = raft::make_device_mdarray<compute_t, int64_t>(
      res, mr, raft::make_extents<int64_t>(dataset_dim));
    raft::copy(res,
               casted_vec.view(),
               raft::make_host_vector_view<const compute_t>(host_threshold_vec.data(),
                                                            (int64_t)dataset_dim));
    raft::linalg::map(res,
                      quantizer.threshold.view(),
                      raft::cast_op<T>{},
                      raft::make_const_mdspan(casted_vec.view()));
  }
  return quantizer;
}

template <typename T, typename QuantI>
void transform(raft::resources const& res,
               const cuvs::preprocessing::quantize::binary::quantizer<T>& quantizer,
               raft::device_matrix_view<const T, int64_t> dataset,
               raft::device_matrix_view<QuantI, int64_t> out)
{
  cudaStream_t stream            = raft::resource::get_cuda_stream(res);
  const uint32_t bits_per_pack   = sizeof(QuantI) * 8;
  const uint32_t dataset_dim     = dataset.extent(1);
  const uint32_t out_dim         = out.extent(1);
  const size_t dataset_size      = dataset.extent(0);
  const size_t out_dataset_size  = out.extent(0);
  const uint32_t minimul_out_dim = raft::div_rounding_up_safe(dataset_dim, bits_per_pack);
  RAFT_EXPECTS(out_dim >= minimul_out_dim,
               "The quantized dataset dimension must be larger or equal to "
               "%u but is %u passed",
               minimul_out_dim,
               out_dim);
  RAFT_EXPECTS(out_dataset_size >= dataset_size,
               "The quantized dataset size must be larger or equal to "
               "the input dataset size (%lu) but is %lu passed",
               dataset_size,
               out_dataset_size);
  const T* threshold_ptr = nullptr;
  if (quantizer.threshold.size() != 0) { threshold_ptr = quantizer.threshold.data_handle(); }

  constexpr uint32_t warp_size    = 32;
  constexpr uint32_t block_size   = 256;
  constexpr uint32_t vecs_per_cta = block_size / warp_size;
  const auto grid_size =
    raft::div_rounding_up_safe(dataset_size, static_cast<size_t>(vecs_per_cta));

  binary_quantization_kernel<T, block_size>
    <<<grid_size, block_size, 0, stream>>>(dataset.data_handle(),
                                           dataset.stride(0),
                                           threshold_ptr,
                                           dataset_size,
                                           dataset_dim,
                                           out.data_handle(),
                                           out.stride(0));
}

template <typename T, typename QuantI>
void transform(raft::resources const& res,
               const cuvs::preprocessing::quantize::binary::quantizer<T>& quantizer,
               raft::host_matrix_view<const T, int64_t> dataset,
               raft::host_matrix_view<QuantI, int64_t> out)
{
  const uint32_t bits_per_pack   = sizeof(QuantI) * 8;
  const uint32_t dataset_dim     = dataset.extent(1);
  const uint32_t out_dim         = out.extent(1);
  const size_t dataset_size      = dataset.extent(0);
  const size_t out_dataset_size  = out.extent(0);
  const uint32_t minimul_out_dim = raft::div_rounding_up_safe(dataset_dim, bits_per_pack);
  RAFT_EXPECTS(out_dim >= minimul_out_dim,
               "The quantized dataset dimension must be larger or equal to "
               "%u but is %u passed",
               minimul_out_dim,
               out_dim);
  RAFT_EXPECTS(out_dataset_size >= dataset_size,
               "The quantized dataset size must be larger or equal to "
               "the input dataset size (%lu) but is %lu passed",
               dataset_size,
               out_dataset_size);

  using compute_t          = std::conditional_t<std::is_same_v<half, T>, float, T>;
  auto threshold_vec       = raft::make_host_vector<compute_t, int64_t>(0);
  compute_t* threshold_ptr = nullptr;

  if (quantizer.threshold.size() != 0) {
    threshold_vec = raft::make_host_vector<compute_t, int64_t>(dataset_dim);
    threshold_ptr = threshold_vec.data_handle();

    if constexpr (std::is_same_v<compute_t, T>) {
      raft::copy(res,
                 raft::make_host_vector_view(threshold_ptr, (int64_t)dataset_dim),
                 raft::make_device_vector_view<const T>(quantizer.threshold.data_handle(),
                                                        (int64_t)dataset_dim));
    } else {
      auto mr         = raft::resource::get_workspace_resource_ref(res);
      auto casted_vec = raft::make_device_mdarray<float, int64_t>(
        res, mr, raft::make_extents<int64_t>(dataset_dim));
      raft::linalg::map(res,
                        casted_vec.view(),
                        raft::cast_op<compute_t>{},
                        raft::make_const_mdspan(quantizer.threshold.view()));
      raft::copy(
        res,
        raft::make_host_vector_view(threshold_ptr, (int64_t)dataset_dim),
        raft::make_device_vector_view<const float>(casted_vec.data_handle(), (int64_t)dataset_dim));
    }
    // Populate the threshold_ptr on the host side before the host parallel loop.
    raft::resource::sync_stream(res);
  }

#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < dataset_size; ++i) {
    for (uint32_t out_j = 0; out_j < out_dim; ++out_j) {
      if (out_j >= minimul_out_dim) {
        out(i, out_j) = 0;
        continue;
      }
      QuantI pack = 0;
      for (uint32_t pack_j = 0; pack_j < bits_per_pack; ++pack_j) {
        const uint32_t in_j = out_j * bits_per_pack + pack_j;
        if (in_j < dataset_dim) {
          if (threshold_ptr == nullptr) {
            if (is_positive(dataset(i, in_j))) { pack |= (1u << pack_j); }
          } else {
            if (is_positive(static_cast<compute_t>(dataset(i, in_j)) - threshold_ptr[in_j])) {
              pack |= (1u << pack_j);
            }
          }
        }
      }
      out(i, out_j) = pack;
    }
  }
}
}  // namespace cuvs::preprocessing::quantize::detail
