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
#include <cuvs/preprocessing/pca.h>

#include <cmath>
#include <vector>

namespace {

DLManagedTensor make_col_major_tensor(float* data, int64_t rows, int64_t cols)
{
  DLManagedTensor tensor{};
  cuvs::core::to_dlpack(
    raft::make_device_matrix_view<float, int64_t, raft::col_major>(data, rows, cols), &tensor);
  return tensor;
}

DLManagedTensor make_vector_tensor(float* data, int64_t size)
{
  DLManagedTensor tensor{};
  cuvs::core::to_dlpack(raft::make_device_vector_view<float, int64_t>(data, size), &tensor);
  return tensor;
}

void free_tensor(DLManagedTensor& t)
{
  if (t.deleter) { t.deleter(&t); }
}

}  // namespace

TEST(PcaC, FitTransformInverseTransform)
{
  raft::device_resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  int64_t n_rows   = 256;
  int64_t n_cols   = 32;
  int n_components = 32;

  rmm::device_uvector<float> input(n_rows * n_cols, stream);
  rmm::device_uvector<float> input_copy(n_rows * n_cols, stream);
  rmm::device_uvector<float> trans(n_rows * n_components, stream);
  rmm::device_uvector<float> components(n_components * n_cols, stream);
  rmm::device_uvector<float> explained_var(n_components, stream);
  rmm::device_uvector<float> explained_var_ratio(n_components, stream);
  rmm::device_uvector<float> singular_vals(n_components, stream);
  rmm::device_uvector<float> mu(n_cols, stream);
  rmm::device_uvector<float> noise_vars(1, stream);
  rmm::device_uvector<float> output(n_rows * n_cols, stream);

  raft::random::RngState rng(1234ULL);
  raft::random::uniform(handle, rng, input.data(), n_rows * n_cols, -1.0f, 1.0f);

  raft::copy(input_copy.data(), input.data(), n_rows * n_cols, stream);
  handle.sync_stream();

  cuvsResources_t res;
  cuvsResourcesCreate(&res);

  cuvsPcaParams_t params;
  cuvsPcaParamsCreate(&params);
  params->n_components = n_components;

  auto input_t  = make_col_major_tensor(input.data(), n_rows, n_cols);
  auto trans_t  = make_col_major_tensor(trans.data(), n_rows, n_components);
  auto comp_t   = make_col_major_tensor(components.data(), n_components, n_cols);
  auto ev_t     = make_vector_tensor(explained_var.data(), n_components);
  auto evr_t    = make_vector_tensor(explained_var_ratio.data(), n_components);
  auto sv_t     = make_vector_tensor(singular_vals.data(), n_components);
  auto mu_t     = make_vector_tensor(mu.data(), n_cols);
  auto nv_t     = make_vector_tensor(noise_vars.data(), 1);
  auto output_t = make_col_major_tensor(output.data(), n_rows, n_cols);

  cuvsPcaFitTransform(
    res, params, &input_t, &trans_t, &comp_t, &ev_t, &evr_t, &sv_t, &mu_t, &nv_t, false);

  cuvsPcaInverseTransform(res, params, &trans_t, &comp_t, &sv_t, &mu_t, &output_t);

  cuvsStreamSync(res);
  cuvsPcaParamsDestroy(params);
  cuvsResourcesDestroy(res);

  free_tensor(input_t);
  free_tensor(trans_t);
  free_tensor(comp_t);
  free_tensor(ev_t);
  free_tensor(evr_t);
  free_tensor(sv_t);
  free_tensor(mu_t);
  free_tensor(nv_t);
  free_tensor(output_t);

  std::vector<float> input_h(n_rows * n_cols);
  std::vector<float> output_h(n_rows * n_cols);
  raft::copy(input_h.data(), input_copy.data(), n_rows * n_cols, stream);
  raft::copy(output_h.data(), output.data(), n_rows * n_cols, stream);
  handle.sync_stream();

  float max_err = 0.0f;
  for (int64_t i = 0; i < n_rows * n_cols; ++i) {
    max_err = std::max(max_err, std::abs(input_h[i] - output_h[i]));
  }
  EXPECT_LT(max_err, 1e-3f) << "Reconstruction with all components should be near-lossless";
}

TEST(PcaC, FitThenTransform)
{
  raft::device_resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  int64_t n_rows   = 256;
  int64_t n_cols   = 32;
  int n_components = 32;

  rmm::device_uvector<float> input(n_rows * n_cols, stream);
  rmm::device_uvector<float> input_copy(n_rows * n_cols, stream);
  rmm::device_uvector<float> trans(n_rows * n_components, stream);
  rmm::device_uvector<float> components(n_components * n_cols, stream);
  rmm::device_uvector<float> explained_var(n_components, stream);
  rmm::device_uvector<float> explained_var_ratio(n_components, stream);
  rmm::device_uvector<float> singular_vals(n_components, stream);
  rmm::device_uvector<float> mu(n_cols, stream);
  rmm::device_uvector<float> noise_vars(1, stream);
  rmm::device_uvector<float> output(n_rows * n_cols, stream);

  raft::random::RngState rng(1234ULL);
  raft::random::uniform(handle, rng, input.data(), n_rows * n_cols, -1.0f, 1.0f);

  raft::copy(input_copy.data(), input.data(), n_rows * n_cols, stream);
  handle.sync_stream();

  cuvsResources_t res;
  cuvsResourcesCreate(&res);

  cuvsPcaParams_t params;
  cuvsPcaParamsCreate(&params);
  params->n_components = n_components;
  params->copy         = true;

  auto input_t  = make_col_major_tensor(input.data(), n_rows, n_cols);
  auto trans_t  = make_col_major_tensor(trans.data(), n_rows, n_components);
  auto comp_t   = make_col_major_tensor(components.data(), n_components, n_cols);
  auto ev_t     = make_vector_tensor(explained_var.data(), n_components);
  auto evr_t    = make_vector_tensor(explained_var_ratio.data(), n_components);
  auto sv_t     = make_vector_tensor(singular_vals.data(), n_components);
  auto mu_t     = make_vector_tensor(mu.data(), n_cols);
  auto nv_t     = make_vector_tensor(noise_vars.data(), 1);
  auto output_t = make_col_major_tensor(output.data(), n_rows, n_cols);

  cuvsPcaFit(res, params, &input_t, &comp_t, &ev_t, &evr_t, &sv_t, &mu_t, &nv_t, false);
  cuvsPcaTransform(res, params, &input_t, &comp_t, &sv_t, &mu_t, &trans_t);
  cuvsPcaInverseTransform(res, params, &trans_t, &comp_t, &sv_t, &mu_t, &output_t);

  cuvsStreamSync(res);
  cuvsPcaParamsDestroy(params);
  cuvsResourcesDestroy(res);

  free_tensor(input_t);
  free_tensor(trans_t);
  free_tensor(comp_t);
  free_tensor(ev_t);
  free_tensor(evr_t);
  free_tensor(sv_t);
  free_tensor(mu_t);
  free_tensor(nv_t);
  free_tensor(output_t);

  std::vector<float> input_h(n_rows * n_cols);
  std::vector<float> output_h(n_rows * n_cols);
  raft::copy(input_h.data(), input_copy.data(), n_rows * n_cols, stream);
  raft::copy(output_h.data(), output.data(), n_rows * n_cols, stream);
  handle.sync_stream();

  float max_err = 0.0f;
  for (int64_t i = 0; i < n_rows * n_cols; ++i) {
    max_err = std::max(max_err, std::abs(input_h[i] - output_h[i]));
  }
  EXPECT_LT(max_err, 1e-3f) << "Reconstruction with all components should be near-lossless";
}

TEST(PcaC, DimReduction)
{
  raft::device_resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  int64_t n_rows   = 512;
  int64_t n_cols   = 64;
  int n_components = 16;

  rmm::device_uvector<float> input(n_rows * n_cols, stream);
  rmm::device_uvector<float> input_copy(n_rows * n_cols, stream);
  rmm::device_uvector<float> trans(n_rows * n_components, stream);
  rmm::device_uvector<float> components(n_components * n_cols, stream);
  rmm::device_uvector<float> explained_var(n_components, stream);
  rmm::device_uvector<float> explained_var_ratio(n_components, stream);
  rmm::device_uvector<float> singular_vals(n_components, stream);
  rmm::device_uvector<float> mu(n_cols, stream);
  rmm::device_uvector<float> noise_vars(1, stream);
  rmm::device_uvector<float> output(n_rows * n_cols, stream);

  raft::random::RngState rng(5678ULL);
  raft::random::uniform(handle, rng, input.data(), n_rows * n_cols, -1.0f, 1.0f);

  raft::copy(input_copy.data(), input.data(), n_rows * n_cols, stream);
  handle.sync_stream();

  cuvsResources_t res;
  cuvsResourcesCreate(&res);

  cuvsPcaParams_t params;
  cuvsPcaParamsCreate(&params);
  params->n_components = n_components;

  auto input_t  = make_col_major_tensor(input.data(), n_rows, n_cols);
  auto trans_t  = make_col_major_tensor(trans.data(), n_rows, n_components);
  auto comp_t   = make_col_major_tensor(components.data(), n_components, n_cols);
  auto ev_t     = make_vector_tensor(explained_var.data(), n_components);
  auto evr_t    = make_vector_tensor(explained_var_ratio.data(), n_components);
  auto sv_t     = make_vector_tensor(singular_vals.data(), n_components);
  auto mu_t     = make_vector_tensor(mu.data(), n_cols);
  auto nv_t     = make_vector_tensor(noise_vars.data(), 1);
  auto output_t = make_col_major_tensor(output.data(), n_rows, n_cols);

  cuvsPcaFitTransform(
    res, params, &input_t, &trans_t, &comp_t, &ev_t, &evr_t, &sv_t, &mu_t, &nv_t, false);

  cuvsPcaInverseTransform(res, params, &trans_t, &comp_t, &sv_t, &mu_t, &output_t);

  cuvsStreamSync(res);
  cuvsPcaParamsDestroy(params);
  cuvsResourcesDestroy(res);

  free_tensor(input_t);
  free_tensor(trans_t);
  free_tensor(comp_t);
  free_tensor(ev_t);
  free_tensor(evr_t);
  free_tensor(sv_t);
  free_tensor(mu_t);
  free_tensor(nv_t);
  free_tensor(output_t);

  std::vector<float> input_h(n_rows * n_cols);
  std::vector<float> output_h(n_rows * n_cols);
  raft::copy(input_h.data(), input_copy.data(), n_rows * n_cols, stream);
  raft::copy(output_h.data(), output.data(), n_rows * n_cols, stream);
  handle.sync_stream();

  float max_err = 0.0f;
  for (int64_t i = 0; i < n_rows * n_cols; ++i) {
    max_err = std::max(max_err, std::abs(input_h[i] - output_h[i]));
  }
  EXPECT_GT(max_err, 1e-5f) << "With fewer components, reconstruction error should be non-zero";
  EXPECT_LT(max_err, 2.0f) << "Reconstruction error should still be bounded";
}
