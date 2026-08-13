/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./detail/pca.cuh"

#include <cuvs/preprocessing/pca.hpp>

namespace cuvs::preprocessing::pca {

#define CUVS_INST_PCA_FIT(DataT, IndexT, LayoutT)                       \
  void fit(raft::resources const& handle,                               \
           const params& config,                                        \
           raft::device_matrix_view<DataT, IndexT, LayoutT> input,      \
           raft::device_matrix_view<DataT, IndexT, LayoutT> components, \
           raft::device_vector_view<DataT, IndexT> explained_var,       \
           raft::device_vector_view<DataT, IndexT> explained_var_ratio, \
           raft::device_vector_view<DataT, IndexT> singular_vals,       \
           raft::device_vector_view<DataT, IndexT> mu,                  \
           raft::device_scalar_view<DataT, IndexT> noise_vars,          \
           bool flip_signs_based_on_U)                                  \
  {                                                                     \
    detail::fit(handle,                                                 \
                config,                                                 \
                input,                                                  \
                components,                                             \
                explained_var,                                          \
                explained_var_ratio,                                    \
                singular_vals,                                          \
                mu,                                                     \
                noise_vars,                                             \
                flip_signs_based_on_U);                                 \
  }

CUVS_INST_PCA_FIT(float, int64_t, raft::col_major);
CUVS_INST_PCA_FIT(float, int64_t, raft::row_major);
#undef CUVS_INST_PCA_FIT

#define CUVS_INST_PCA_FIT_TRANSFORM(DataT, IndexT, LayoutT)                        \
  void fit_transform(raft::resources const& handle,                                \
                     const params& config,                                         \
                     raft::device_matrix_view<DataT, IndexT, LayoutT> input,       \
                     raft::device_matrix_view<DataT, IndexT, LayoutT> trans_input, \
                     raft::device_matrix_view<DataT, IndexT, LayoutT> components,  \
                     raft::device_vector_view<DataT, IndexT> explained_var,        \
                     raft::device_vector_view<DataT, IndexT> explained_var_ratio,  \
                     raft::device_vector_view<DataT, IndexT> singular_vals,        \
                     raft::device_vector_view<DataT, IndexT> mu,                   \
                     raft::device_scalar_view<DataT, IndexT> noise_vars,           \
                     bool flip_signs_based_on_U)                                   \
  {                                                                                \
    detail::fit_transform(handle,                                                  \
                          config,                                                  \
                          input,                                                   \
                          trans_input,                                             \
                          components,                                              \
                          explained_var,                                           \
                          explained_var_ratio,                                     \
                          singular_vals,                                           \
                          mu,                                                      \
                          noise_vars,                                              \
                          flip_signs_based_on_U);                                  \
  }

CUVS_INST_PCA_FIT_TRANSFORM(float, int64_t, raft::col_major);
CUVS_INST_PCA_FIT_TRANSFORM(float, int64_t, raft::row_major);
#undef CUVS_INST_PCA_FIT_TRANSFORM

#define CUVS_INST_PCA_TRANSFORM(DataT, IndexT, LayoutT)                                   \
  void transform(raft::resources const& handle,                                           \
                 const params& config,                                                    \
                 raft::device_matrix_view<DataT, IndexT, LayoutT> input,                  \
                 raft::device_matrix_view<DataT, IndexT, LayoutT> components,             \
                 raft::device_vector_view<DataT, IndexT> singular_vals,                   \
                 raft::device_vector_view<DataT, IndexT> mu,                              \
                 raft::device_matrix_view<DataT, IndexT, LayoutT> trans_input)            \
  {                                                                                       \
    detail::transform(handle, config, input, components, singular_vals, mu, trans_input); \
  }

CUVS_INST_PCA_TRANSFORM(float, int64_t, raft::col_major);
CUVS_INST_PCA_TRANSFORM(float, int64_t, raft::row_major);
#undef CUVS_INST_PCA_TRANSFORM

#define CUVS_INST_PCA_INVERSE_TRANSFORM(DataT, IndexT, LayoutT)                                    \
  void inverse_transform(raft::resources const& handle,                                            \
                         const params& config,                                                     \
                         raft::device_matrix_view<DataT, IndexT, LayoutT> trans_input,             \
                         raft::device_matrix_view<DataT, IndexT, LayoutT> components,              \
                         raft::device_vector_view<DataT, IndexT> singular_vals,                    \
                         raft::device_vector_view<DataT, IndexT> mu,                               \
                         raft::device_matrix_view<DataT, IndexT, LayoutT> output)                  \
  {                                                                                                \
    detail::inverse_transform(handle, config, trans_input, components, singular_vals, mu, output); \
  }

CUVS_INST_PCA_INVERSE_TRANSFORM(float, int64_t, raft::col_major);
CUVS_INST_PCA_INVERSE_TRANSFORM(float, int64_t, raft::row_major);
#undef CUVS_INST_PCA_INVERSE_TRANSFORM

}  // namespace cuvs::preprocessing::pca
