/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <dlpack/dlpack.h>

#include <raft/core/error.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/core/c_api.h>
#include <cuvs/preprocessing/pca.h>
#include <cuvs/preprocessing/pca.hpp>

#include "../core/exceptions.hpp"
#include "../core/interop.hpp"

namespace {

cuvs::preprocessing::pca::params to_cpp_params(const cuvsPcaParams& c_params)
{
  cuvs::preprocessing::pca::params cpp_params;
  cpp_params.n_components = c_params.n_components;
  cpp_params.copy         = c_params.copy;
  cpp_params.whiten       = c_params.whiten;
  cpp_params.tol          = c_params.tol;
  cpp_params.n_iterations = c_params.n_iterations;

  switch (c_params.algorithm) {
    case CUVS_PCA_COV_EIG_DQ:
      cpp_params.algorithm = cuvs::preprocessing::pca::solver::COV_EIG_DQ;
      break;
    case CUVS_PCA_COV_EIG_JACOBI:
      cpp_params.algorithm = cuvs::preprocessing::pca::solver::COV_EIG_JACOBI;
      break;
    default: cpp_params.algorithm = cuvs::preprocessing::pca::solver::COV_EIG_DQ; break;
  }

  return cpp_params;
}

template <typename LayoutT>
void _fit(cuvsResources_t res,
          const cuvsPcaParams& params,
          DLManagedTensor* input_tensor,
          DLManagedTensor* components_tensor,
          DLManagedTensor* explained_var_tensor,
          DLManagedTensor* explained_var_ratio_tensor,
          DLManagedTensor* singular_vals_tensor,
          DLManagedTensor* mu_tensor,
          DLManagedTensor* noise_vars_tensor,
          bool flip_signs_based_on_U)
{
  auto res_ptr    = reinterpret_cast<raft::resources*>(res);
  auto cpp_params = to_cpp_params(params);

  using matrix_type = raft::device_matrix_view<float, int64_t, LayoutT>;
  using vector_type = raft::device_vector_view<float, int64_t>;
  using scalar_type = raft::device_scalar_view<float, int64_t>;

  auto input               = cuvs::core::from_dlpack<matrix_type>(input_tensor);
  auto components          = cuvs::core::from_dlpack<matrix_type>(components_tensor);
  auto explained_var       = cuvs::core::from_dlpack<vector_type>(explained_var_tensor);
  auto explained_var_ratio = cuvs::core::from_dlpack<vector_type>(explained_var_ratio_tensor);
  auto singular_vals       = cuvs::core::from_dlpack<vector_type>(singular_vals_tensor);
  auto mu                  = cuvs::core::from_dlpack<vector_type>(mu_tensor);
  auto noise_vars          = cuvs::core::from_dlpack<scalar_type>(noise_vars_tensor);

  cuvs::preprocessing::pca::fit(*res_ptr,
                                cpp_params,
                                input,
                                components,
                                explained_var,
                                explained_var_ratio,
                                singular_vals,
                                mu,
                                noise_vars,
                                flip_signs_based_on_U);
}

template <typename LayoutT>
void _fit_transform(cuvsResources_t res,
                    const cuvsPcaParams& params,
                    DLManagedTensor* input_tensor,
                    DLManagedTensor* trans_input_tensor,
                    DLManagedTensor* components_tensor,
                    DLManagedTensor* explained_var_tensor,
                    DLManagedTensor* explained_var_ratio_tensor,
                    DLManagedTensor* singular_vals_tensor,
                    DLManagedTensor* mu_tensor,
                    DLManagedTensor* noise_vars_tensor,
                    bool flip_signs_based_on_U)
{
  auto res_ptr    = reinterpret_cast<raft::resources*>(res);
  auto cpp_params = to_cpp_params(params);

  using matrix_type = raft::device_matrix_view<float, int64_t, LayoutT>;
  using vector_type = raft::device_vector_view<float, int64_t>;
  using scalar_type = raft::device_scalar_view<float, int64_t>;

  auto input               = cuvs::core::from_dlpack<matrix_type>(input_tensor);
  auto trans_input         = cuvs::core::from_dlpack<matrix_type>(trans_input_tensor);
  auto components          = cuvs::core::from_dlpack<matrix_type>(components_tensor);
  auto explained_var       = cuvs::core::from_dlpack<vector_type>(explained_var_tensor);
  auto explained_var_ratio = cuvs::core::from_dlpack<vector_type>(explained_var_ratio_tensor);
  auto singular_vals       = cuvs::core::from_dlpack<vector_type>(singular_vals_tensor);
  auto mu                  = cuvs::core::from_dlpack<vector_type>(mu_tensor);
  auto noise_vars          = cuvs::core::from_dlpack<scalar_type>(noise_vars_tensor);

  cuvs::preprocessing::pca::fit_transform(*res_ptr,
                                          cpp_params,
                                          input,
                                          trans_input,
                                          components,
                                          explained_var,
                                          explained_var_ratio,
                                          singular_vals,
                                          mu,
                                          noise_vars,
                                          flip_signs_based_on_U);
}

template <typename LayoutT>
void _transform(cuvsResources_t res,
                const cuvsPcaParams& params,
                DLManagedTensor* input_tensor,
                DLManagedTensor* components_tensor,
                DLManagedTensor* singular_vals_tensor,
                DLManagedTensor* mu_tensor,
                DLManagedTensor* trans_input_tensor)
{
  auto res_ptr    = reinterpret_cast<raft::resources*>(res);
  auto cpp_params = to_cpp_params(params);

  using matrix_type = raft::device_matrix_view<float, int64_t, LayoutT>;
  using vector_type = raft::device_vector_view<float, int64_t>;

  auto input         = cuvs::core::from_dlpack<matrix_type>(input_tensor);
  auto components    = cuvs::core::from_dlpack<matrix_type>(components_tensor);
  auto singular_vals = cuvs::core::from_dlpack<vector_type>(singular_vals_tensor);
  auto mu            = cuvs::core::from_dlpack<vector_type>(mu_tensor);
  auto trans_input   = cuvs::core::from_dlpack<matrix_type>(trans_input_tensor);

  cuvs::preprocessing::pca::transform(
    *res_ptr, cpp_params, input, components, singular_vals, mu, trans_input);
}

template <typename LayoutT>
void _inverse_transform(cuvsResources_t res,
                        const cuvsPcaParams& params,
                        DLManagedTensor* trans_input_tensor,
                        DLManagedTensor* components_tensor,
                        DLManagedTensor* singular_vals_tensor,
                        DLManagedTensor* mu_tensor,
                        DLManagedTensor* output_tensor)
{
  auto res_ptr    = reinterpret_cast<raft::resources*>(res);
  auto cpp_params = to_cpp_params(params);

  using matrix_type = raft::device_matrix_view<float, int64_t, LayoutT>;
  using vector_type = raft::device_vector_view<float, int64_t>;

  auto trans_input   = cuvs::core::from_dlpack<matrix_type>(trans_input_tensor);
  auto components    = cuvs::core::from_dlpack<matrix_type>(components_tensor);
  auto singular_vals = cuvs::core::from_dlpack<vector_type>(singular_vals_tensor);
  auto mu            = cuvs::core::from_dlpack<vector_type>(mu_tensor);
  auto output        = cuvs::core::from_dlpack<matrix_type>(output_tensor);

  cuvs::preprocessing::pca::inverse_transform(
    *res_ptr, cpp_params, trans_input, components, singular_vals, mu, output);
}

}  // namespace

extern "C" cuvsError_t cuvsPcaParamsCreate(cuvsPcaParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    *params = new cuvsPcaParams{
      .n_components = 1,
      .copy         = true,
      .whiten       = false,
      .algorithm    = CUVS_PCA_COV_EIG_DQ,
      .tol          = 0.0f,
      .n_iterations = 15,
    };
  });
}

extern "C" cuvsError_t cuvsPcaParamsDestroy(cuvsPcaParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsPcaFit(cuvsResources_t res,
                                  cuvsPcaParams_t params,
                                  DLManagedTensor* input,
                                  DLManagedTensor* components,
                                  DLManagedTensor* explained_var,
                                  DLManagedTensor* explained_var_ratio,
                                  DLManagedTensor* singular_vals,
                                  DLManagedTensor* mu,
                                  DLManagedTensor* noise_vars,
                                  bool flip_signs_based_on_U)
{
  return cuvs::core::translate_exceptions([=] {
    auto dtype = input->dl_tensor.dtype;
    RAFT_EXPECTS(dtype.code == kDLFloat && dtype.bits == 32,
                 "PCA input must be float32 (kDLFloat, 32 bits)");
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(input->dl_tensor),
                 "PCA input must be device-accessible memory");

    if (cuvs::core::is_f_contiguous(input)) {
      _fit<raft::col_major>(res,
                            *params,
                            input,
                            components,
                            explained_var,
                            explained_var_ratio,
                            singular_vals,
                            mu,
                            noise_vars,
                            flip_signs_based_on_U);
    } else if (cuvs::core::is_c_contiguous(input)) {
      _fit<raft::row_major>(res,
                            *params,
                            input,
                            components,
                            explained_var,
                            explained_var_ratio,
                            singular_vals,
                            mu,
                            noise_vars,
                            flip_signs_based_on_U);
    } else {
      RAFT_FAIL("PCA input must be contiguous (C- or F-order)");
    }
  });
}

extern "C" cuvsError_t cuvsPcaFitTransform(cuvsResources_t res,
                                           cuvsPcaParams_t params,
                                           DLManagedTensor* input,
                                           DLManagedTensor* trans_input,
                                           DLManagedTensor* components,
                                           DLManagedTensor* explained_var,
                                           DLManagedTensor* explained_var_ratio,
                                           DLManagedTensor* singular_vals,
                                           DLManagedTensor* mu,
                                           DLManagedTensor* noise_vars,
                                           bool flip_signs_based_on_U)
{
  return cuvs::core::translate_exceptions([=] {
    auto dtype = input->dl_tensor.dtype;
    RAFT_EXPECTS(dtype.code == kDLFloat && dtype.bits == 32,
                 "PCA input must be float32 (kDLFloat, 32 bits)");
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(input->dl_tensor),
                 "PCA input must be device-accessible memory");

    if (cuvs::core::is_f_contiguous(input)) {
      _fit_transform<raft::col_major>(res,
                                      *params,
                                      input,
                                      trans_input,
                                      components,
                                      explained_var,
                                      explained_var_ratio,
                                      singular_vals,
                                      mu,
                                      noise_vars,
                                      flip_signs_based_on_U);
    } else if (cuvs::core::is_c_contiguous(input)) {
      _fit_transform<raft::row_major>(res,
                                      *params,
                                      input,
                                      trans_input,
                                      components,
                                      explained_var,
                                      explained_var_ratio,
                                      singular_vals,
                                      mu,
                                      noise_vars,
                                      flip_signs_based_on_U);
    } else {
      RAFT_FAIL("PCA input must be contiguous (C- or F-order)");
    }
  });
}

extern "C" cuvsError_t cuvsPcaTransform(cuvsResources_t res,
                                        cuvsPcaParams_t params,
                                        DLManagedTensor* input,
                                        DLManagedTensor* components,
                                        DLManagedTensor* singular_vals,
                                        DLManagedTensor* mu,
                                        DLManagedTensor* trans_input)
{
  return cuvs::core::translate_exceptions([=] {
    auto dtype = input->dl_tensor.dtype;
    RAFT_EXPECTS(dtype.code == kDLFloat && dtype.bits == 32,
                 "PCA input must be float32 (kDLFloat, 32 bits)");
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(input->dl_tensor),
                 "PCA input must be device-accessible memory");

    if (cuvs::core::is_f_contiguous(input)) {
      _transform<raft::col_major>(res, *params, input, components, singular_vals, mu, trans_input);
    } else if (cuvs::core::is_c_contiguous(input)) {
      _transform<raft::row_major>(res, *params, input, components, singular_vals, mu, trans_input);
    } else {
      RAFT_FAIL("PCA input must be contiguous (C- or F-order)");
    }
  });
}

extern "C" cuvsError_t cuvsPcaInverseTransform(cuvsResources_t res,
                                               cuvsPcaParams_t params,
                                               DLManagedTensor* trans_input,
                                               DLManagedTensor* components,
                                               DLManagedTensor* singular_vals,
                                               DLManagedTensor* mu,
                                               DLManagedTensor* output)
{
  return cuvs::core::translate_exceptions([=] {
    auto dtype = trans_input->dl_tensor.dtype;
    RAFT_EXPECTS(dtype.code == kDLFloat && dtype.bits == 32,
                 "PCA trans_input must be float32 (kDLFloat, 32 bits)");
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(trans_input->dl_tensor),
                 "PCA trans_input must be device-accessible memory");

    if (cuvs::core::is_f_contiguous(trans_input)) {
      _inverse_transform<raft::col_major>(
        res, *params, trans_input, components, singular_vals, mu, output);
    } else if (cuvs::core::is_c_contiguous(trans_input)) {
      _inverse_transform<raft::row_major>(
        res, *params, trans_input, components, singular_vals, mu, output);
    } else {
      RAFT_FAIL("PCA trans_input must be contiguous (C- or F-order)");
    }
  });
}
