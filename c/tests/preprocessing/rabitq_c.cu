/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>

#include "../../src/core/interop.hpp"
#include <cuvs/preprocessing/quantize/rabitq.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

template <typename T>
DLManagedTensor make_row_major_tensor(T* data, int64_t rows, int64_t cols)
{
  DLManagedTensor tensor{};
  cuvs::core::to_dlpack(
    raft::make_device_matrix_view<T, int64_t, raft::row_major>(data, rows, cols), &tensor);
  return tensor;
}

template <typename T>
DLManagedTensor make_vector_tensor(T* data, int64_t size)
{
  DLManagedTensor tensor{};
  cuvs::core::to_dlpack(raft::make_device_vector_view<T, int64_t>(data, size), &tensor);
  return tensor;
}

void free_tensor(DLManagedTensor& t)
{
  if (t.deleter) { t.deleter(&t); }
}

template <typename T>
std::vector<T> to_host(raft::device_resources const& handle, T const* d_ptr, size_t len)
{
  std::vector<T> h(len);
  raft::copy(h.data(), d_ptr, len, raft::resource::get_cuda_stream(handle));
  handle.sync_stream();
  return h;
}

/** Mirrors detail::padded_dim(): the working dimension is `dim` rounded up to 32. */
constexpr int64_t expected_padded_dim(int64_t dim) { return ((dim + 31) / 32) * 32; }

/** Mirrors detail::flip_bytes(): four sign-flip passes of `padded_dim` bits each. */
constexpr int64_t expected_flip_bytes(int64_t padded_dim) { return 4 * padded_dim / 8; }

/**
 * Exercises the whole C surface for one rotator kind: params -> pipeline ->
 * getters -> transform -> state readback -> teardown.
 */
void run_rabitq_c_smoke(enum cuvsRabitqRotatorKind kind)
{
  raft::device_resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  constexpr int64_t n_rows = 64;
  // Deliberately not a multiple of 32, so the dim -> padded_dim padding is exercised.
  constexpr int64_t dim        = 40;
  constexpr int64_t padded_dim = expected_padded_dim(dim);
  constexpr uint32_t ex_bits   = 3;
  // Codes hold `ex_bits + 1` bit unsigned values, so this is the largest legal code.
  constexpr uint32_t max_code = (1u << (ex_bits + 1)) - 1u;
  // Pre-fill `codes` with a byte larger than max_code: any element the transform
  // fails to write stays out of range and is caught by the range check below.
  constexpr int unwritten_sentinel = 0xEE;

  static_assert(padded_dim == 64, "test expects dim=40 to pad up to 64");
  static_assert(unwritten_sentinel > static_cast<int>(max_code), "sentinel must be detectable");

  rmm::device_uvector<float> dataset(n_rows * dim, stream);
  rmm::device_uvector<float> centroid(dim, stream);
  raft::random::RngState rng(1234ULL);
  raft::random::uniform(handle, rng, dataset.data(), n_rows * dim, -1.0f, 1.0f);
  raft::random::uniform(handle, rng, centroid.data(), dim, -0.5f, 0.5f);
  handle.sync_stream();

  cuvsResources_t res;
  ASSERT_EQ(cuvsResourcesCreate(&res), CUVS_SUCCESS);

  // --- params: defaults must mirror the C++ member initializers ---
  cuvsRabitqQuantizerParams_t params;
  ASSERT_EQ(cuvsRabitqQuantizerParamsCreate(&params), CUVS_SUCCESS);
  EXPECT_EQ(params->ex_bits, 3u);
  EXPECT_EQ(params->rotator, CUVS_RABITQ_ROTATOR_KIND_FHT_KAC);
  EXPECT_EQ(params->seed, 12345ULL);
  EXPECT_EQ(params->delta_mode, CUVS_RABITQ_DELTA_KIND_RECONSTRUCTION);
  EXPECT_TRUE(params->use_fast);
  EXPECT_EQ(params->coarse_samples, 64u);
  EXPECT_EQ(params->fine_samples, 64u);

  params->ex_bits = ex_bits;
  params->rotator = kind;
  params->seed    = 42ULL;

  // --- factories: one call populates both peers ---
  cuvsRabitqRotator_t rotator;
  cuvsRabitqQuantizer_t quantizer;
  ASSERT_EQ(cuvsRabitqRotatorCreate(&rotator), CUVS_SUCCESS);
  ASSERT_EQ(cuvsRabitqQuantizerCreate(&quantizer), CUVS_SUCCESS);
  ASSERT_EQ(cuvsRabitqPipelineMake(res, params, dim, rotator, quantizer), CUVS_SUCCESS);

  // --- metadata getters ---
  enum cuvsRabitqRotatorKind read_kind = CUVS_RABITQ_ROTATOR_KIND_MATMUL;
  ASSERT_EQ(cuvsRabitqRotatorGetKind(rotator, &read_kind), CUVS_SUCCESS);
  EXPECT_EQ(read_kind, kind);

  int64_t rotator_padded_dim   = 0;
  int64_t quantizer_padded_dim = 0;
  int64_t quantizer_dim        = 0;
  ASSERT_EQ(cuvsRabitqRotatorGetPaddedDim(rotator, &rotator_padded_dim), CUVS_SUCCESS);
  ASSERT_EQ(cuvsRabitqQuantizerGetPaddedDim(quantizer, &quantizer_padded_dim), CUVS_SUCCESS);
  ASSERT_EQ(cuvsRabitqQuantizerGetDim(quantizer, &quantizer_dim), CUVS_SUCCESS);
  EXPECT_EQ(rotator_padded_dim, padded_dim);
  EXPECT_EQ(quantizer_padded_dim, padded_dim);
  EXPECT_EQ(quantizer_dim, dim);

  // use_fast is on, so the factor was actually estimated on the GPU.
  float const_scaling_factor = 0.0f;
  ASSERT_EQ(cuvsRabitqQuantizerGetConstScalingFactor(quantizer, &const_scaling_factor),
            CUVS_SUCCESS);
  EXPECT_TRUE(std::isfinite(const_scaling_factor));
  EXPECT_GT(const_scaling_factor, 0.0f);

  // --- transform ---
  rmm::device_uvector<uint8_t> codes(n_rows * padded_dim, stream);
  rmm::device_uvector<float> delta(n_rows, stream);
  rmm::device_uvector<float> vl(n_rows, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(codes.data(), unwritten_sentinel, codes.size(), stream));
  handle.sync_stream();

  auto dataset_t  = make_row_major_tensor(dataset.data(), n_rows, dim);
  auto centroid_t = make_vector_tensor(centroid.data(), dim);
  auto codes_t    = make_row_major_tensor(codes.data(), n_rows, padded_dim);
  auto delta_t    = make_vector_tensor(delta.data(), n_rows);
  auto vl_t       = make_vector_tensor(vl.data(), n_rows);

  // A NULL centroid must be accepted and mean "no centering".
  ASSERT_EQ(
    cuvsRabitqTransform(res, quantizer, rotator, &dataset_t, nullptr, &codes_t, &delta_t, &vl_t),
    CUVS_SUCCESS);
  ASSERT_EQ(cuvsStreamSync(res), CUVS_SUCCESS);

  {
    auto h_codes = to_host(handle, codes.data(), codes.size());
    size_t n_out_of_range = 0;
    bool all_zero         = true;
    for (auto c : h_codes) {
      if (static_cast<uint32_t>(c) > max_code) { ++n_out_of_range; }
      if (c != 0) { all_zero = false; }
    }
    EXPECT_EQ(n_out_of_range, 0u) << n_out_of_range << " of " << h_codes.size()
                                  << " codes are outside the " << (ex_bits + 1)
                                  << "-bit range [0, " << max_code
                                  << "] (an unwritten element keeps the 0xEE sentinel)";
    EXPECT_FALSE(all_zero) << "quantized codes are all zero";

    auto h_delta = to_host(handle, delta.data(), delta.size());
    auto h_vl    = to_host(handle, vl.data(), vl.size());
    // vl is documented as `delta * -(2^ex_bits - 0.5)`.
    const float cb = -(static_cast<float>(1u << ex_bits) - 0.5f);
    for (int64_t i = 0; i < n_rows; ++i) {
      EXPECT_TRUE(std::isfinite(h_delta[i])) << " row " << i;
      EXPECT_GT(h_delta[i], 0.0f) << " row " << i;
      EXPECT_NEAR(h_vl[i], h_delta[i] * cb, 1e-5f * std::fabs(h_delta[i] * cb) + 1e-20f)
        << " row " << i;
    }
  }

  // A non-NULL centroid takes the residual path and must produce different codes.
  auto h_codes_uncentered = to_host(handle, codes.data(), codes.size());
  RAFT_CUDA_TRY(cudaMemsetAsync(codes.data(), unwritten_sentinel, codes.size(), stream));
  handle.sync_stream();
  ASSERT_EQ(cuvsRabitqTransform(
              res, quantizer, rotator, &dataset_t, &centroid_t, &codes_t, &delta_t, &vl_t),
            CUVS_SUCCESS);
  ASSERT_EQ(cuvsStreamSync(res), CUVS_SUCCESS);
  {
    auto h_codes          = to_host(handle, codes.data(), codes.size());
    size_t n_out_of_range = 0;
    size_t n_diff         = 0;
    for (size_t i = 0; i < h_codes.size(); ++i) {
      if (static_cast<uint32_t>(h_codes[i]) > max_code) { ++n_out_of_range; }
      if (h_codes[i] != h_codes_uncentered[i]) { ++n_diff; }
    }
    EXPECT_EQ(n_out_of_range, 0u) << "centered codes outside the " << (ex_bits + 1) << "-bit range";
    EXPECT_GT(n_diff, 0u) << "passing a centroid did not change the codes";
  }

  // --- rotator state readback ---
  DLManagedTensor state{};
  ASSERT_EQ(cuvsRabitqRotatorGetState(rotator, &state), CUVS_SUCCESS);
  EXPECT_NE(state.dl_tensor.data, nullptr);
  EXPECT_TRUE(cuvs::core::is_dlpack_device_compatible(state.dl_tensor));
  if (kind == CUVS_RABITQ_ROTATOR_KIND_MATMUL) {
    // padded_dim x padded_dim row-major float32 rotation matrix
    ASSERT_EQ(state.dl_tensor.ndim, 2);
    EXPECT_EQ(state.dl_tensor.dtype.code, kDLFloat);
    EXPECT_EQ(state.dl_tensor.dtype.bits, 32);
    EXPECT_EQ(state.dl_tensor.shape[0], padded_dim);
    EXPECT_EQ(state.dl_tensor.shape[1], padded_dim);
    // Non-empty in substance, not just in shape: a rotation matrix cannot be all zeros.
    auto h_state = to_host(
      handle, static_cast<float const*>(state.dl_tensor.data), padded_dim * padded_dim);
    bool all_zero = true;
    for (auto v : h_state) {
      if (v != 0.0f) { all_zero = false; }
    }
    EXPECT_FALSE(all_zero) << "rotation matrix read back as all zeros";
  } else {
    // 4 * padded_dim / 8 packed uint8 sign bits
    ASSERT_EQ(state.dl_tensor.ndim, 1);
    EXPECT_EQ(state.dl_tensor.dtype.code, kDLUInt);
    EXPECT_EQ(state.dl_tensor.dtype.bits, 8);
    ASSERT_EQ(state.dl_tensor.shape[0], expected_flip_bytes(padded_dim));
    auto h_state = to_host(handle,
                           static_cast<uint8_t const*>(state.dl_tensor.data),
                           static_cast<size_t>(expected_flip_bytes(padded_dim)));
    size_t n_set_bits = 0;
    for (auto b : h_state) {
      for (int i = 0; i < 8; ++i) {
        n_set_bits += (b >> i) & 1u;
      }
    }
    // Neither all-zero nor all-one: the flip bits are genuinely random.
    EXPECT_GT(n_set_bits, 0u);
    EXPECT_LT(n_set_bits, h_state.size() * 8);
  }
  free_tensor(state);

  // --- teardown ---
  free_tensor(dataset_t);
  free_tensor(centroid_t);
  free_tensor(codes_t);
  free_tensor(delta_t);
  free_tensor(vl_t);

  EXPECT_EQ(cuvsRabitqRotatorDestroy(rotator), CUVS_SUCCESS);
  EXPECT_EQ(cuvsRabitqQuantizerDestroy(quantizer), CUVS_SUCCESS);
  EXPECT_EQ(cuvsRabitqQuantizerParamsDestroy(params), CUVS_SUCCESS);
  EXPECT_EQ(cuvsResourcesDestroy(res), CUVS_SUCCESS);
}

}  // namespace

TEST(RabitqC, PipelineTransformFhtKac) { run_rabitq_c_smoke(CUVS_RABITQ_ROTATOR_KIND_FHT_KAC); }

TEST(RabitqC, PipelineTransformMatmul) { run_rabitq_c_smoke(CUVS_RABITQ_ROTATOR_KIND_MATMUL); }
