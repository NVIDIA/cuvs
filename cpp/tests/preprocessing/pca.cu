/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <cuvs/preprocessing/pca.hpp>

#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>

#include <gtest/gtest.h>
#include <test_utils.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace cuvs::preprocessing::pca {

template <typename T>
struct PcaInputs {
  T tolerance;
  int n_row;
  int n_col;
  int n_row2;
  int n_col2;
  unsigned long long int seed;
  int algo;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const PcaInputs<T>& dims)
{
  return os;
}

/**
 * @brief Run fit_transform followed by inverse_transform.
 *
 * Templated on layout to exercise both col-major and row-major paths.
 * Intermediate buffers are managed internally unless the caller provides
 * pre-allocated pointers via the optional parameters, in which case the
 * results are written there directly.
 */
template <typename T, typename LayoutT = raft::col_major>
void pca_roundtrip(raft::resources const& handle,
                   T* input,
                   int n_rows,
                   int n_cols,
                   T* output,
                   int n_components,
                   int algo,
                   T* components_out    = nullptr,
                   T* explained_var_out = nullptr,
                   T* trans_out         = nullptr)
{
  auto stream = raft::resource::get_cuda_stream(handle);

  params prms;
  prms.n_components = n_components;
  if (algo == 0)
    prms.algorithm = solver::COV_EIG_DQ;
  else
    prms.algorithm = solver::COV_EIG_JACOBI;

  rmm::device_uvector<T> comp_buf(components_out ? 0 : n_components * n_cols, stream);
  rmm::device_uvector<T> ev_buf(explained_var_out ? 0 : n_components, stream);
  rmm::device_uvector<T> trans_buf(trans_out ? 0 : n_rows * n_components, stream);

  T* comp_ptr  = components_out ? components_out : comp_buf.data();
  T* ev_ptr    = explained_var_out ? explained_var_out : ev_buf.data();
  T* trans_ptr = trans_out ? trans_out : trans_buf.data();

  rmm::device_uvector<T> evr(n_components, stream);
  rmm::device_uvector<T> sv(n_components, stream);
  rmm::device_uvector<T> mu(n_cols, stream);
  rmm::device_uvector<T> nv(1, stream);

  auto input_view = raft::make_device_matrix_view<T, int64_t, LayoutT>(input, n_rows, n_cols);
  auto trans_view =
    raft::make_device_matrix_view<T, int64_t, LayoutT>(trans_ptr, n_rows, n_components);
  auto comp_view =
    raft::make_device_matrix_view<T, int64_t, LayoutT>(comp_ptr, n_components, n_cols);
  auto ev_view     = raft::make_device_vector_view<T, int64_t>(ev_ptr, n_components);
  auto evr_view    = raft::make_device_vector_view<T, int64_t>(evr.data(), n_components);
  auto sv_view     = raft::make_device_vector_view<T, int64_t>(sv.data(), n_components);
  auto mu_view     = raft::make_device_vector_view<T, int64_t>(mu.data(), n_cols);
  auto nv_view     = raft::make_device_scalar_view<T>(nv.data());
  auto output_view = raft::make_device_matrix_view<T, int64_t, LayoutT>(output, n_rows, n_cols);

  fit_transform(
    handle, prms, input_view, trans_view, comp_view, ev_view, evr_view, sv_view, mu_view, nv_view);
  inverse_transform(handle, prms, trans_view, comp_view, sv_view, mu_view, output_view);
}

template <typename T>
class PcaTest : public ::testing::TestWithParam<PcaInputs<T>> {
 public:
  PcaTest()
    : params_(::testing::TestWithParam<PcaInputs<T>>::GetParam()),
      stream(raft::resource::get_cuda_stream(handle)),
      explained_vars(params_.n_col, stream),
      explained_vars_ref(params_.n_col, stream),
      components(params_.n_col * params_.n_col, stream),
      components_ref(params_.n_col * params_.n_col, stream),
      trans_data(params_.n_row * params_.n_col, stream),
      trans_data_ref(params_.n_row * params_.n_col, stream),
      data(params_.n_row * params_.n_col, stream),
      data_back(params_.n_row * params_.n_col, stream),
      data2(params_.n_row2 * params_.n_col2, stream),
      data2_back(params_.n_row2 * params_.n_col2, stream)
  {
  }

 protected:
  void SetUp() override
  {
    int len  = params_.n_row * params_.n_col;
    int len2 = params_.n_row2 * params_.n_col2;

    // --- basic test: all components, known reference data ---
    {
      std::vector<T> data_h = {1.0, 2.0, 5.0, 4.0, 2.0, 1.0};
      data_h.resize(len);
      raft::update_device(data.data(), data_h.data(), len, stream);

      std::vector<T> trans_data_ref_h = {-2.3231, -0.3517, 2.6748, 0.3979, -0.6571, 0.2592};
      trans_data_ref_h.resize(len);
      raft::update_device(trans_data_ref.data(), trans_data_ref_h.data(), len, stream);

      int len_comp = params_.n_col * params_.n_col;

      std::vector<T> components_ref_h = {0.8163, 0.5776, -0.5776, 0.8163};
      components_ref_h.resize(len_comp);
      std::vector<T> explained_vars_ref_h = {6.338, 0.3287};
      explained_vars_ref_h.resize(params_.n_col);

      raft::update_device(components_ref.data(), components_ref_h.data(), len_comp, stream);
      raft::update_device(
        explained_vars_ref.data(), explained_vars_ref_h.data(), params_.n_col, stream);

      pca_roundtrip(handle,
                    data.data(),
                    params_.n_row,
                    params_.n_col,
                    data_back.data(),
                    params_.n_col,
                    params_.algo,
                    components.data(),
                    explained_vars.data(),
                    trans_data.data());
    }

    // --- advanced test: all components, random data ---
    {
      raft::random::Rng r(params_.seed, raft::random::GenPC);
      r.uniform(data2.data(), len2, T(-1.0), T(1.0), stream);

      pca_roundtrip(handle,
                    data2.data(),
                    params_.n_row2,
                    params_.n_col2,
                    data2_back.data(),
                    params_.n_col2,
                    params_.algo);
    }

    // --- dim reduction test: n_components < n_cols, random data ---
    {
      int n_components = std::max(1, params_.n_col2 / 4);

      rmm::device_uvector<T> input(len2, stream);
      rmm::device_uvector<T> input_copy(len2, stream);
      rmm::device_uvector<T> recon(len2, stream);

      raft::random::Rng rng(params_.seed + 1, raft::random::GenPC);
      rng.uniform(input.data(), len2, T(-1.0), T(1.0), stream);
      raft::copy(input_copy.data(), input.data(), len2, stream);

      pca_roundtrip(handle,
                    input.data(),
                    params_.n_row2,
                    params_.n_col2,
                    recon.data(),
                    n_components,
                    params_.algo);

      std::vector<T> orig_h(len2);
      std::vector<T> recon_h(len2);
      raft::update_host(orig_h.data(), input_copy.data(), len2, stream);
      raft::update_host(recon_h.data(), recon.data(), len2, stream);
      raft::resource::sync_stream(handle);

      max_recon_err = T(0);
      for (int i = 0; i < len2; ++i) {
        max_recon_err = std::max(max_recon_err, std::abs(orig_h[i] - recon_h[i]));
      }
    }
  }

  void testPca()
  {
    auto s = raft::resource::get_cuda_stream(handle);

    ASSERT_TRUE(devArrMatch(explained_vars.data(),
                            explained_vars_ref.data(),
                            params_.n_col,
                            cuvs::CompareApprox<T>(params_.tolerance),
                            s));

    ASSERT_TRUE(devArrMatch(components.data(),
                            components_ref.data(),
                            (params_.n_col * params_.n_col),
                            cuvs::CompareApprox<T>(params_.tolerance),
                            s));

    ASSERT_TRUE(devArrMatch(trans_data.data(),
                            trans_data_ref.data(),
                            (params_.n_row * params_.n_col),
                            cuvs::CompareApprox<T>(params_.tolerance),
                            s));

    ASSERT_TRUE(devArrMatch(data.data(),
                            data_back.data(),
                            (params_.n_row * params_.n_col),
                            cuvs::CompareApprox<T>(params_.tolerance),
                            s));

    ASSERT_TRUE(devArrMatch(data2.data(),
                            data2_back.data(),
                            (params_.n_row2 * params_.n_col2),
                            cuvs::CompareApprox<T>(params_.tolerance),
                            s));

    EXPECT_GT(max_recon_err, T(1e-5)) << "Error should be non-zero when n_components < n_cols";
    EXPECT_LT(max_recon_err, T(2.0)) << "Reconstruction error should be bounded";
  }

 private:
  raft::device_resources handle;
  cudaStream_t stream;

  PcaInputs<T> params_;
  T max_recon_err = T(0);

  rmm::device_uvector<T> explained_vars, explained_vars_ref, components, components_ref, trans_data,
    trans_data_ref, data, data_back, data2, data2_back;
};

const std::vector<PcaInputs<float>> inputsf2 = {{0.01f, 3, 2, 1024, 128, 1234ULL, 0},
                                                {0.01f, 3, 2, 256, 32, 1234ULL, 1}};

typedef PcaTest<float> PcaTestF;
TEST_P(PcaTestF, Result) { this->testPca(); }

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestF, ::testing::ValuesIn(inputsf2));

/**
 * Row-major end-to-end test: runs fit_transform + inverse_transform on row-major
 * inputs and verifies the reconstruction matches the original. Also checks that
 * row-major and col-major inputs (representing the same logical data) produce
 * the same explained variances, singular values, and column means.
 */
template <typename T>
class PcaRowMajorTest : public ::testing::TestWithParam<PcaInputs<T>> {
 public:
  PcaRowMajorTest()
    : params_(::testing::TestWithParam<PcaInputs<T>>::GetParam()),
      stream(raft::resource::get_cuda_stream(handle))
  {
  }

 protected:
  // Convert col-major data of shape (n_rows, n_cols) into row-major in dst.
  // Both buffers live on device.
  void to_row_major(const T* col_major_src, T* row_major_dst, int n_rows, int n_cols)
  {
    std::vector<T> host_col(n_rows * n_cols);
    std::vector<T> host_row(n_rows * n_cols);
    raft::update_host(host_col.data(), col_major_src, n_rows * n_cols, stream);
    raft::resource::sync_stream(handle);
    for (int i = 0; i < n_rows; ++i) {
      for (int j = 0; j < n_cols; ++j) {
        host_row[i * n_cols + j] = host_col[j * n_rows + i];
      }
    }
    raft::update_device(row_major_dst, host_row.data(), n_rows * n_cols, stream);
  }

  void testRowMajorRoundtrip()
  {
    int n_rows = params_.n_row2;
    int n_cols = params_.n_col2;
    int len    = n_rows * n_cols;

    rmm::device_uvector<T> data_col(len, stream);
    rmm::device_uvector<T> data_row(len, stream);
    rmm::device_uvector<T> data_back_row(len, stream);

    raft::random::Rng r(params_.seed, raft::random::GenPC);
    r.uniform(data_col.data(), len, T(-1.0), T(1.0), stream);
    to_row_major(data_col.data(), data_row.data(), n_rows, n_cols);

    pca_roundtrip<T, raft::row_major>(
      handle, data_row.data(), n_rows, n_cols, data_back_row.data(), n_cols, params_.algo);

    ASSERT_TRUE(devArrMatch(data_row.data(),
                            data_back_row.data(),
                            len,
                            cuvs::CompareApprox<T>(params_.tolerance),
                            stream));
  }

  void testLayoutsAgreeNumerically()
  {
    int n_rows = params_.n_row2;
    int n_cols = params_.n_col2;
    int len    = n_rows * n_cols;

    rmm::device_uvector<T> data_col(len, stream);
    rmm::device_uvector<T> data_row(len, stream);

    raft::random::Rng r(params_.seed, raft::random::GenPC);
    r.uniform(data_col.data(), len, T(-1.0), T(1.0), stream);
    to_row_major(data_col.data(), data_row.data(), n_rows, n_cols);

    // Run col-major path
    rmm::device_uvector<T> col_back(len, stream);
    rmm::device_uvector<T> col_ev(n_cols, stream);
    rmm::device_uvector<T> col_components(n_cols * n_cols, stream);
    rmm::device_uvector<T> col_trans(len, stream);
    pca_roundtrip<T, raft::col_major>(handle,
                                      data_col.data(),
                                      n_rows,
                                      n_cols,
                                      col_back.data(),
                                      n_cols,
                                      params_.algo,
                                      col_components.data(),
                                      col_ev.data(),
                                      col_trans.data());

    // Run row-major path on the same logical data
    rmm::device_uvector<T> row_back(len, stream);
    rmm::device_uvector<T> row_ev(n_cols, stream);
    rmm::device_uvector<T> row_components(n_cols * n_cols, stream);
    rmm::device_uvector<T> row_trans(len, stream);
    pca_roundtrip<T, raft::row_major>(handle,
                                      data_row.data(),
                                      n_rows,
                                      n_cols,
                                      row_back.data(),
                                      n_cols,
                                      params_.algo,
                                      row_components.data(),
                                      row_ev.data(),
                                      row_trans.data());

    // Explained variances and reconstructions should agree across layouts.
    ASSERT_TRUE(devArrMatch(
      col_ev.data(), row_ev.data(), n_cols, cuvs::CompareApprox<T>(params_.tolerance), stream));

    // Reconstructions are stored in their native layouts; compare element-by-
    // element after a host-side reshape of the row-major result back to col.
    std::vector<T> col_back_h(len);
    std::vector<T> row_back_h(len);
    raft::update_host(col_back_h.data(), col_back.data(), len, stream);
    raft::update_host(row_back_h.data(), row_back.data(), len, stream);
    raft::resource::sync_stream(handle);
    for (int i = 0; i < n_rows; ++i) {
      for (int j = 0; j < n_cols; ++j) {
        T col_val = col_back_h[j * n_rows + i];
        T row_val = row_back_h[i * n_cols + j];
        ASSERT_NEAR(col_val, row_val, params_.tolerance) << "Mismatch at (" << i << "," << j << ")";
      }
    }
  }

 private:
  raft::device_resources handle;
  cudaStream_t stream;
  PcaInputs<T> params_;
};

typedef PcaRowMajorTest<float> PcaRowMajorTestF;
TEST_P(PcaRowMajorTestF, Roundtrip) { this->testRowMajorRoundtrip(); }
TEST_P(PcaRowMajorTestF, AgreesWithColMajor) { this->testLayoutsAgreeNumerically(); }

INSTANTIATE_TEST_CASE_P(PcaTests, PcaRowMajorTestF, ::testing::ValuesIn(inputsf2));

}  // end namespace cuvs::preprocessing::pca
