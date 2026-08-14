/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cmath>
#include <cuvs/neighbors/common.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/matrix/init.cuh>

#include "scann_common.cuh"
using namespace cuvs::neighbors;

namespace cuvs::neighbors::experimental::scann::detail {

/** Fix the internal indexing type to avoid integer underflows/overflows */
using ix_t = int64_t;

/**
 * @brief Subtract cluster center coordinates from each dataset vector.
 *
 * residual[i, k] = dataset[i ,k] - centers[l, k],
 * where l = labels[i], the cluster label corresponding to vector i.
 *
 * @tparam T
 * @tparam LabelT
 * @param res raft resources
 * @param dataset dataset vectors, size [n_rows, dim]
 * @param centers cluster center coordinates, size [n_clusters, dim]
 * @param labels cluster labels, size [n_rows]
 * @return device matrix with the residuals, size [n_rows, dim]
 */
template <typename T, typename LabelT>
auto compute_residuals(raft::resources const& res,
                       raft::device_matrix_view<const T, int64_t> dataset,
                       raft::device_matrix_view<const T, int64_t> centers,
                       raft::device_vector_view<const LabelT, int64_t> labels)
  -> raft::device_matrix<T, int64_t, raft::row_major>
{
  auto dim       = dataset.extent(1);
  auto residuals = raft::make_device_matrix<T, int64_t>(res, labels.extent(0), dim);

  // compute residuals for AVQ assignments
  raft::linalg::map_offset(
    res, residuals.view(), [dataset, centers, labels, dim] __device__(size_t i) {
      int row_idx = i / dim;
      int el_idx  = i % dim;
      return dataset(row_idx, el_idx) - centers(labels(row_idx), el_idx);
    });

  return residuals;
}

/**
 * @brief Unpack VPQ codes into 1-byte per code
 *
 * VPQ gives codes in a "packed" form. In the case of 4 bit PQ, each byte stores
 * codes for 2 subspaces in a packed form.
 *
 * This function unpacks the  subspace codes into one byte each. This is for
 * interoperability with open source ScaNN, which doesn't pack codes
 *
 * @tparam IdxT
 * @param res raft resources
 * @param unpacked_codes_view matrix of unpacked codes, size  [n_rows, dim / pq_dim]
 * @param codes_view packed codes from vpq, size [n_rows, ceil((dim / pq_dim * pq_bits) / 8)]
 * @param pq_bits number of bits used for PQ
 * @param num_subspaces the number of pq_subspaces (dim / pq_dim)
 */
template <typename IdxT>
void unpack_codes(raft::resources const& res,
                  raft::device_matrix_view<uint8_t, IdxT> unpacked_codes_view,
                  raft::device_matrix_view<const uint8_t, IdxT> codes_view,
                  int pq_bits,
                  int num_subspaces)
{
  if (pq_bits == 4) {
    raft::linalg::map_offset(
      res, unpacked_codes_view, [codes_view, num_subspaces] __device__(size_t i) {
        int64_t row_idx             = i / num_subspaces;
        int64_t subspace_idx        = i % num_subspaces;
        int64_t packed_subspace_idx = subspace_idx / 2;
        uint8_t mask                = subspace_idx % 2;

        uint8_t packed_labels = codes_view(row_idx, packed_subspace_idx);
        uint8_t first         = packed_labels >> 4;
        uint8_t second        = (packed_labels & 15);

        return (mask)*first + (1 - mask) * second;
      });
  }
}

/**
 * @brief compute eta for AVQ according to Theorem 3.4 in https://arxiv.org/abs/1908.10396
 *
 * @tparam IdxT
 * @param dim the dataset dimension
 * @param sq_norm the squared norm of the vector
 * @param noise_shaping_threshold the threshold T in the Theorem
 * @return eta
 */
template <typename IdxT>
__device__ inline float compute_avq_eta(IdxT dim, const float sq_norm, const float threshold)
{
  return (dim - 1) * (threshold * threshold / sq_norm) / (1 - threshold * threshold / sq_norm);
}

/**
 * @brief helper to convert a float to bfloat16 (represented as int16_t)
 *
 * @param f the float value
 * @return the bflaot16 value (as int16_t)
 */
__device__ inline int16_t float_to_bfloat16(const float& f)
{
  nv_bfloat16 val = __float2bfloat16(f);
  return reinterpret_cast<int16_t&>(val);
}

/**
 * @brief helper to convert a bfloat16 (represented as int16_t) to float
 *
 * @param bf16 the bf16 value (represented as int16_t)
 * @return the float value
 */
__device__ inline float bfloat16_to_float(int16_t& bf16)
{
  nv_bfloat16 nv_bf16 = reinterpret_cast<nv_bfloat16&>(bf16);
  return __bfloat162float(nv_bf16);
}

/**
 * @brief Select the next bfloat16 value to try during coordinate descent
 *
 * Based on the signs of the current residual and quantized value,
 * increment or decrement the quantized value to push residual closer to 0
 *
 * Note that the bfloat16 value is encoded as an int16_t, and the
 * increment/decrement is applied to encoded value. In terms of the float
 * representation, it is the mantissa that is being incremented/decremented,
 * which could carryover to the exponent
 *
 * @param res the float residual
 * @param current the current quantized dimension
 * @return the other possible quantized value
 */
__device__ inline int16_t bfloat16_next_delta(float& res, int16_t& current)
{
  uint32_t res_sign  = ((int32_t)res & (1u << 31) >> 31);
  uint32_t curr_sign = (current & (1 << 15)) >> 15;

  if (res_sign == curr_sign) { return current - 1; }

  return current + 1;
}

template <uint32_t BlockSize, typename IdxT>
__launch_bounds__(BlockSize) RAFT_KERNEL
  quantize_bfloat16_noise_shaped_kernel(raft::device_matrix_view<const float, IdxT> dataset,
                                        raft::device_matrix_view<int16_t, IdxT> bf16_dataset,
                                        raft::device_vector_view<const float, IdxT> sq_norms,
                                        float noise_shaping_threshold)
{
  IdxT row_idx = raft::Pow2<32>::div(IdxT{threadIdx.x} + IdxT{BlockSize} * IdxT{blockIdx.x});

  if (row_idx >= dataset.extent(0)) { return; }

  uint32_t lane_id = raft::Pow2<32>::mod(threadIdx.x);

  IdxT dim = dataset.extent(1);

  // 1 / ||x||
  float inv_norm = 1 / sqrtf(sq_norms[row_idx]);
  float eta      = compute_avq_eta(dim, sq_norms[row_idx], noise_shaping_threshold);

  // < r, x >
  float residual_dot = 0.0;

  for (int i = lane_id; i < dim; i += 32) {
    bf16_dataset(row_idx, i) = float_to_bfloat16(dataset(row_idx, i));

    float residual = dataset(row_idx, i) - bfloat16_to_float(bf16_dataset(row_idx, i));
    residual_dot += dataset(row_idx, i) * residual * inv_norm;
  }

  // reduce  and broadcast residual_dot across warp
  for (uint32_t offset = 16; offset > 0; offset >>= 1) {
    residual_dot += raft::shfl_xor(residual_dot, offset, 32);
  }

  constexpr uint32_t kMaxRounds = 10;

  bool round_changes = true;
  for (int round = 0; round < kMaxRounds && round_changes; round++) {
    round_changes = false;

    for (int i = lane_id; i < dim; i += 32) {
      // coaleseced reads of required data
      float original    = dataset(row_idx, i);
      int16_t quantized = bf16_dataset(row_idx, i);

      float old_residual    = original - bfloat16_to_float(quantized);
      int16_t quantized_new = bfloat16_next_delta(old_residual, quantized);

      float new_residual       = original - bfloat16_to_float(quantized_new);
      float residual_dot_delta = (new_residual - old_residual) * dataset(row_idx, i) * inv_norm;

      float residual_norm_delta = new_residual * new_residual - old_residual * old_residual;

      // we want to compute the change in cost = eta || r_parallel || ^2 + || r_perpendicular|| ^2
      // The change in || r_parallel ||^2 can be written (residual_dot + residual_dot_delta) ^ 2
      // the change in || r_perpendicular || ^2 can be written residual_norm_delta -
      // parallel_norm_delta Thus cost_delta = eta * (residual_dot + residual_dot_delta) ^2 +
      // (residual_norm_delta - (residual_dot + residual_dot_delta)^2 Expanding and simplifying,
      // cost_delta = a + b * resdiaul_dot, where a and b are as below. Since only residual_dot is
      // unknown (because updates must be made synchronously) we can compute a and b in parallel
      // across threads in the warp and minimize computation in the update step of the coordinate
      // descent
      float a = residual_norm_delta + (eta - 1) * residual_dot_delta * residual_dot_delta;
      float b = 2 * (eta - 1) * residual_dot_delta;

      // Dim may not be divisible by 32
      // Only synchronize/shuffle for active threads
      int active_threads = std::min<int>(32, dim - i + lane_id);
      int mask           = (1 << active_threads) - 1;

      // Update step for coordinate descent. Compute the cost_delta for
      // each thread, update the quantized value and residual_dot if applicable,
      // then broadcast the new residual dot to the warp
      // AVQ loss the not separable, so we must optimize each dimension separately
      for (int j = 0; j < active_threads; j++) {
        if (lane_id == j) {
          // change in AVQ loss
          float cost_delta = b * residual_dot + a;

          if (cost_delta < 0.0) {
            quantized = quantized_new;
            residual_dot += residual_dot_delta;
            round_changes = true;
          }
        }

        // broadcast new dot product to all lanes
        residual_dot = raft::shfl(residual_dot, j, active_threads, mask);
      }

      // coalesced write of possibly updated quantized values
      bf16_dataset(row_idx, i) = quantized;
    }

    // reduce round_changes across warp
    for (uint32_t offset = 16; offset > 0; offset >>= 1) {
      round_changes |= raft::shfl_xor(round_changes, offset, 32);
    }
  }
}

/**
 * @brief Quantized a float dataset as bfloat16, with noise shaping (AVQ)
 *
 * During quantization we replace each input vector coordinate `f` of type float32 with a bfloat16
 * coordinate `b`. One way to do this would be to simply assign the nearest bfloat16 value to
 * each coordinate. This would be the best way to quantize if we want to minimize the L2
 * distance between the quantized and the original vector.
 *
 * In the AVQ method, we use a different cost function. To minimize that, we consider nearest
 * representable bfloat16 values (`b1`, `b2`) around `f`, and select the one that minimizes the AVQ
 * cost function. In two dimensions we need to consider the four neighboring quantized vectors:
 * b1       b2
 *        f
 * b3      b4
 *
 * In N dimension we will select one the vertices of an N dimensional hypercube as the quantized
 * vector. To find the minimum without enumerating all the combinations, a coordinate descent
 * method is used.
 * @tparam IdxT
 * @param res raft resources
 * @param dataset the dataset (device only) size [n_rows, dim]
 * @param bf16_dataset the quantized dataset (device only) size [n_rows, dim]
 * @param noise_shaping_threshold the threshold for AVQ
 */
template <typename IdxT>
void quantize_bfloat16_noise_shaped(raft::resources const& res,
                                    raft::device_matrix_view<const float, IdxT> dataset,
                                    raft::device_matrix_view<int16_t, IdxT> bf16_dataset,
                                    float noise_shaping_threshold)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res);

  IdxT n_rows = dataset.extent(0);
  auto norms  = raft::make_device_vector<float, IdxT>(res, n_rows);

  // populate square norms
  raft::linalg::norm<raft::linalg::NormType::L2Norm, raft::Apply::ALONG_ROWS>(
    res, dataset, norms.view());

  constexpr int64_t kBlockSize = 256;

  dim3 threads(kBlockSize, 1, 1);
  dim3 blocks(raft::div_rounding_up_safe<ix_t>(n_rows, kBlockSize / 32), 1, 1);

  quantize_bfloat16_noise_shaped_kernel<kBlockSize, IdxT><<<blocks, threads, 0, stream>>>(
    dataset, bf16_dataset, raft::make_const_mdspan(norms.view()), noise_shaping_threshold);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * @brief Quantized a float dataset as bfloat16, with optional noise shaping (AVQ)
 *
 * @tparam IdxT
 * @param res raft resources
 * @param dataset the dataset (device only) size [n_rows, dim]
 * @param bf16_dataset the quantized dataset (device only) size [n_rows, dim]
 * @param noise_shaping_threshold the threshold for AVQ (nan when not using AVQ)
 */
template <typename IdxT>
void quantize_bfloat16(raft::resources const& res,
                       raft::device_matrix_view<const float, IdxT> dataset,
                       raft::device_matrix_view<int16_t, IdxT> bf16_dataset,
                       float noise_shaping_threshold)
{
  if (!std::isnan(noise_shaping_threshold)) {
    quantize_bfloat16_noise_shaped(res, dataset, bf16_dataset, noise_shaping_threshold);
  } else {
    raft::linalg::map(
      res,
      raft::make_device_vector_view<int16_t, int64_t>(bf16_dataset.data_handle(),
                                                      (int64_t)bf16_dataset.size()),
      [] __device__(float x) { return float_to_bfloat16(x); },
      raft::make_const_mdspan(raft::make_device_vector_view<const float, int64_t>(
        dataset.data_handle(), (int64_t)dataset.size())));
  }
}

/**
 * @brief sample dataset vectors/labels and compute their residuals for PQ training
 *
 * @tparam T
 * @tparms LabelT
 * @tparam Accessor
 * @param res raft resources
 * @param random_state state for random generator
 * @param dataset the dataset (host or device), size [n_rows, dim]
 * @param centroids the centers from kmeans training, size [n_clusters, dim]
 * @param labels the idx of the nearest center for each dataset vector, size [n_rows]
 * @param n_samples number of samples
 * @return the sampled residuals for PQ training, size [n_samples, dim]
 */
template <typename T,
          typename LabelT,
          typename Accessor =
            raft::host_device_accessor<cuda::std::default_accessor<T>, raft::memory_type::host>>
auto sample_training_residuals(
  raft::resources const& res,
  random::RngState random_state,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  raft::device_matrix_view<const T, int64_t> centroids,
  raft::device_vector_view<const LabelT, int64_t> labels,
  int64_t n_samples) -> raft::device_matrix<T, int64_t, raft::row_major>
{
  int64_t n_dim = dataset.extent(1);

  raft::device_vector<int64_t, int64_t> train_indices =
    raft::random::excess_subsample<int64_t, int64_t>(
      res, random_state, dataset.extent(0), n_samples);

  auto trainset = raft::make_device_matrix<T, int64_t>(res, n_samples, n_dim);

  gather_functor<T, int64_t>{}(res,
                               dataset,
                               raft::make_const_mdspan(train_indices.view()),
                               trainset.view(),
                               raft::resource::get_cuda_stream(res));

  // Considering labels as a single column matrix for use in gather
  auto labels_view =
    raft::make_device_matrix_view<const LabelT, int64_t>(labels.data_handle(), labels.extent(0), 1);
  auto trainset_labels = raft::make_device_matrix<LabelT, int64_t>(res, n_samples, 1);

  raft::matrix::gather(
    res, labels_view, raft::make_const_mdspan(train_indices.view()), trainset_labels.view());

  return compute_residuals<T, LabelT>(
    res,
    raft::make_const_mdspan(trainset.view()),
    centroids,
    raft::make_device_vector_view<const LabelT, int64_t>(trainset_labels.data_handle(), n_samples));
}

}  // namespace cuvs::neighbors::experimental::scann::detail
