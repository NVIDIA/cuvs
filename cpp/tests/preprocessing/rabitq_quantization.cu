/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <cuvs/preprocessing/quantize/rabitq.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/itertools.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <ostream>
#include <vector>

/*
 * RaBitQ is lossy and exposes no inverse_transform, so nothing below checks a
 * round trip. What is checked instead:
 *   - the shape / consistency contract of the three factories, and that the
 *     rotator state is non-degenerate and reproducible from `params::seed`;
 *   - that rotate() is an orthogonal transform (norm preserving) - a cheap but
 *     strong numeric check that covers both rotator kinds and both FHT paths
 *     (power-of-two and non-power-of-two padded dims);
 *   - that transform() emits in-range, not-all-zero codes and finite factors;
 *   - that transform() is deterministic;
 *   - that transform() agrees with rotate() + transform_residuals() whenever no
 *     zero padding is involved (dim % 32 == 0).
 */

namespace cuvs::preprocessing::quantize::rabitq {

/** Mirrors detail::padded_dim(): the working dimension is `dim` rounded up to 32. */
constexpr int64_t expected_padded_dim(int64_t dim) { return ((dim + 31) / 32) * 32; }

/** Mirrors detail::flip_bytes(): four sign-flip passes of `padded_dim` bits each. */
constexpr int64_t expected_flip_bytes(int64_t padded_dim) { return 4 * padded_dim / 8; }

struct RabitqQuantizationInputs {
  int rows;
  int dim;
  rotator_kind rotator_type;
  uint32_t ex_bits;
  bool use_fast;
};

std::ostream& operator<<(std::ostream& os, const RabitqQuantizationInputs& inputs)
{
  return os << "rows:" << inputs.rows << " dim:" << inputs.dim
            << " padded_dim:" << expected_padded_dim(inputs.dim) << " rotator:"
            << (inputs.rotator_type == rotator_kind::matmul ? "matmul" : "fht_kac")
            << " ex_bits:" << inputs.ex_bits << " use_fast:" << inputs.use_fast;
}

template <typename CodeT>
class RabitqQuantizationTest : public ::testing::TestWithParam<RabitqQuantizationInputs> {
 public:
  RabitqQuantizationTest()
    : ps_(::testing::TestWithParam<RabitqQuantizationInputs>::GetParam()),
      stream_(raft::resource::get_cuda_stream(handle_)),
      dataset_(raft::make_device_matrix<float, int64_t>(handle_, ps_.rows, ps_.dim)),
      centroid_(raft::make_device_vector<float, int64_t>(handle_, ps_.dim))
  {
  }

 protected:
  void SetUp() override
  {
    raft::random::RngState r(1234ULL);
    raft::random::uniform(handle_, r, dataset_.data_handle(), dataset_.size(), -1.0f, 1.0f);
    raft::random::uniform(handle_, r, centroid_.data_handle(), centroid_.size(), -0.5f, 0.5f);
    raft::resource::sync_stream(handle_, stream_);
  }

  // -------------------------------------------------------------------------
  // helpers
  // -------------------------------------------------------------------------

  params make_params() const
  {
    params p;
    p.ex_bits    = ps_.ex_bits;
    p.rotator    = ps_.rotator_type;
    p.seed       = 137ULL;
    p.delta_mode = delta_kind::reconstruction;
    p.use_fast   = ps_.use_fast;
    // The per-vector rescale search only runs when use_fast is false; keep it cheap.
    p.coarse_samples = 16;
    p.fine_samples   = 16;
    return p;
  }

  template <typename T>
  std::vector<T> to_host(const T* d_ptr, size_t len)
  {
    std::vector<T> h(len);
    raft::update_host(h.data(), d_ptr, len, stream_);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream_));
    return h;
  }

  static std::vector<double> row_norms(const std::vector<float>& h, int64_t rows, int64_t cols)
  {
    std::vector<double> norms(rows);
    for (int64_t i = 0; i < rows; ++i) {
      double sum = 0.0;
      for (int64_t j = 0; j < cols; ++j) {
        const double v = h[static_cast<size_t>(i) * static_cast<size_t>(cols) + j];
        sum += v * v;
      }
      norms[i] = std::sqrt(sum);
    }
    return norms;
  }

  /** The two peers must agree on the documented padding rule. */
  void verify_shape(rotator const& rot, quantizer const& quant)
  {
    const int64_t padded = expected_padded_dim(ps_.dim);
    ASSERT_EQ(quant.dim, ps_.dim);
    ASSERT_EQ(quant.padded_dim, padded);
    ASSERT_EQ(rot.padded_dim, padded);
    ASSERT_EQ(padded % 32, 0);
    ASSERT_GE(padded, ps_.dim);
    ASSERT_LT(padded - ps_.dim, 32);
    ASSERT_EQ(static_cast<int>(rot.kind), static_cast<int>(ps_.rotator_type));
    ASSERT_EQ(quant.params_quantizer.ex_bits, ps_.ex_bits);
    if (ps_.use_fast) {
      ASSERT_TRUE(std::isfinite(quant.const_scaling_factor));
      ASSERT_GT(quant.const_scaling_factor, 0.0f);
    } else {
      // Unused on the slow path; the kernel substitutes 1.0 itself.
      ASSERT_EQ(quant.const_scaling_factor, 1.0f);
    }
  }

  /** The dense rotation matrix must be padded_dim x padded_dim and orthonormal. */
  void verify_matmul_state(rotator const& rot)
  {
    const int64_t padded = rot.padded_dim;
    ASSERT_EQ(rot.rotation_matrix.extent(0), padded);
    ASSERT_EQ(rot.rotation_matrix.extent(1), padded);
    ASSERT_EQ(rot.flip_bits.extent(0), 0);

    auto h = to_host(rot.rotation_matrix.data_handle(),
                     static_cast<size_t>(padded) * static_cast<size_t>(padded));
    // Spot-check the leading rows: unit norm and mutually orthogonal.
    const int64_t n_check = std::min<int64_t>(padded, 8);
    for (int64_t i = 0; i < n_check; ++i) {
      for (int64_t j = i; j < n_check; ++j) {
        double dot = 0.0;
        for (int64_t k = 0; k < padded; ++k) {
          dot += double(h[static_cast<size_t>(i) * padded + k]) *
                 double(h[static_cast<size_t>(j) * padded + k]);
        }
        ASSERT_NEAR(dot, (i == j) ? 1.0 : 0.0, 1e-4)
          << "rotation_matrix rows " << i << " and " << j << " are not orthonormal";
      }
    }
  }

  /** The packed Kac's-walk sign bits must have the documented size and be mixed. */
  void verify_fht_kac_state(rotator const& rot)
  {
    const int64_t n_bytes = expected_flip_bytes(rot.padded_dim);
    ASSERT_EQ(rot.flip_bits.extent(0), n_bytes);
    ASSERT_EQ(rot.rotation_matrix.extent(0), 0);
    ASSERT_EQ(rot.rotation_matrix.extent(1), 0);

    auto h            = to_host(rot.flip_bits.data_handle(), static_cast<size_t>(n_bytes));
    size_t n_set_bits = 0;
    for (auto byte : h) {
      for (int k = 0; k < 8; ++k) {
        n_set_bits += (byte >> k) & 1u;
      }
    }
    // An all-zero or all-one flip pattern would degenerate the random rotation
    // into a plain Hadamard transform.
    ASSERT_GT(n_set_bits, 0u);
    ASSERT_LT(n_set_bits, static_cast<size_t>(n_bytes) * 8);
  }

  void verify_state(rotator const& rot)
  {
    if (ps_.rotator_type == rotator_kind::matmul) {
      verify_matmul_state(rot);
    } else {
      verify_fht_kac_state(rot);
    }
  }

  /** Codes hold `ex_bits + 1` bit unsigned values and must not be all zero. */
  std::vector<CodeT> verify_codes(const CodeT* d_codes, size_t len)
  {
    auto h                  = to_host(d_codes, len);
    const uint32_t max_code = (1u << (ps_.ex_bits + 1)) - 1u;
    size_t n_out_of_range   = 0;
    bool all_zero           = true;
    for (size_t i = 0; i < len; ++i) {
      if (static_cast<uint32_t>(h[i]) > max_code) { ++n_out_of_range; }
      if (h[i] != CodeT{0}) { all_zero = false; }
    }
    EXPECT_EQ(n_out_of_range, 0u) << n_out_of_range << " of " << len << " codes exceed the "
                                  << (ps_.ex_bits + 1) << "-bit range [0, " << max_code << "]";
    EXPECT_FALSE(all_zero) << "quantized codes are all zero";
    return h;
  }

  void verify_finite(const float* d_ptr, size_t len, const char* what, bool positive = false)
  {
    auto h       = to_host(d_ptr, len);
    size_t n_bad = 0;
    for (size_t i = 0; i < len; ++i) {
      if (!std::isfinite(h[i]) || (positive && !(h[i] > 0.0f))) { ++n_bad; }
    }
    EXPECT_EQ(n_bad, 0u) << n_bad << " of " << len << " " << what << " values are not finite"
                         << (positive ? " and strictly positive" : "");
  }

  /** vl is documented as `delta * -(2^ex_bits - 0.5)`. */
  void verify_vl(const float* d_delta, const float* d_vl, int64_t rows)
  {
    auto h_delta   = to_host(d_delta, static_cast<size_t>(rows));
    auto h_vl      = to_host(d_vl, static_cast<size_t>(rows));
    const float cb = -(static_cast<float>(1u << ps_.ex_bits) - 0.5f);
    for (int64_t i = 0; i < rows; ++i) {
      const float expected = h_delta[i] * cb;
      ASSERT_NEAR(h_vl[i], expected, 1e-5f * std::fabs(expected) + 1e-20f) << " row " << i;
    }
  }

  void expect_same_codes(std::vector<CodeT> const& a,
                         std::vector<CodeT> const& b,
                         const char* what)
  {
    ASSERT_EQ(a.size(), b.size());
    size_t n_diff = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      if (a[i] != b[i]) { ++n_diff; }
    }
    EXPECT_EQ(n_diff, 0u) << what << ": " << n_diff << " of " << a.size() << " codes differ";
  }

  // -------------------------------------------------------------------------
  // 1. factories: consistent padded_dim, non-degenerate + reproducible state
  // -------------------------------------------------------------------------
  void testInit()
  {
    const auto p = make_params();

    auto pipe = init_pipeline(handle_, p, ps_.dim);
    verify_shape(pipe.rotator, pipe.quantizer);
    verify_state(pipe.rotator);

    // The peers built on their own must be interchangeable with the pipeline's:
    // that is what makes the header's "store the seed and re-run init_rotator"
    // recipe safe.
    auto rot   = init_rotator(handle_, p, ps_.dim);
    auto quant = init_quantizer(handle_, p, ps_.dim);
    verify_shape(rot, quant);
    verify_state(rot);
    ASSERT_EQ(rot.padded_dim, pipe.quantizer.padded_dim);
    ASSERT_NEAR(quant.const_scaling_factor,
                pipe.quantizer.const_scaling_factor,
                1e-6f * std::fabs(pipe.quantizer.const_scaling_factor) + 1e-20f);
    if (ps_.rotator_type == rotator_kind::matmul) {
      const size_t n = static_cast<size_t>(rot.padded_dim) * static_cast<size_t>(rot.padded_dim);
      ASSERT_TRUE(devArrMatch(pipe.rotator.rotation_matrix.data_handle(),
                              rot.rotation_matrix.data_handle(),
                              n,
                              cuvs::Compare<float>(),
                              stream_));
    } else {
      ASSERT_TRUE(devArrMatch(pipe.rotator.flip_bits.data_handle(),
                              rot.flip_bits.data_handle(),
                              static_cast<size_t>(expected_flip_bytes(rot.padded_dim)),
                              cuvs::Compare<uint8_t>(),
                              stream_));
    }

    // dim must be positive, and ex_bits must stay inside the tight-start table.
    EXPECT_THROW(init_rotator(handle_, p, 0), raft::logic_error);
    EXPECT_THROW(init_quantizer(handle_, p, -1), raft::logic_error);
    auto bad_params    = p;
    bad_params.ex_bits = 9;
    EXPECT_THROW(init_quantizer(handle_, bad_params, ps_.dim), raft::logic_error);
  }

  // -------------------------------------------------------------------------
  // 2. rotate() is an orthogonal transform
  // -------------------------------------------------------------------------
  void testRotate()
  {
    const auto p     = make_params();
    auto rot         = init_rotator(handle_, p, ps_.dim);
    const int64_t pd = rot.padded_dim;
    const int64_t n  = ps_.rows;

    auto in  = raft::make_device_matrix<float, int64_t>(handle_, n, pd);
    auto out = raft::make_device_matrix<float, int64_t>(handle_, n, pd);
    raft::random::RngState r(4242ULL);
    raft::random::uniform(handle_, r, in.data_handle(), in.size(), -1.0f, 1.0f);
    RAFT_CUDA_TRY(cudaMemsetAsync(out.data_handle(), 0, out.size() * sizeof(float), stream_));
    raft::resource::sync_stream(handle_, stream_);

    rotate(handle_, rot, raft::make_const_mdspan(in.view()), out.view());
    raft::resource::sync_stream(handle_, stream_);

    auto h_in      = to_host(in.data_handle(), in.size());
    auto h_out     = to_host(out.data_handle(), out.size());
    auto norms_in  = row_norms(h_in, n, pd);
    auto norms_out = row_norms(h_out, n, pd);
    for (int64_t i = 0; i < n; ++i) {
      ASSERT_GT(norms_in[i], 0.0) << " row " << i;
      // Orthogonal transform: ||Rx|| == ||x||. The tolerance is float32 slack
      // over pd accumulations, not a modelling allowance.
      ASSERT_NEAR(norms_out[i], norms_in[i], 1e-3 * norms_in[i]) << " row " << i;
    }

    // ...and it must actually mix the coordinates rather than copy them.
    size_t n_same = 0;
    for (size_t i = 0; i < h_in.size(); ++i) {
      if (h_in[i] == h_out[i]) { ++n_same; }
    }
    EXPECT_LT(n_same, h_in.size() / 2) << "rotate() left most coordinates untouched";

    if (ps_.rotator_type == rotator_kind::fht_kac) {
      // In-place rotation is documented as supported for fht_kac only.
      auto inplace = raft::make_device_matrix<float, int64_t>(handle_, n, pd);
      raft::copy(inplace.data_handle(), in.data_handle(), in.size(), stream_);
      raft::resource::sync_stream(handle_, stream_);
      rotate(handle_, rot, raft::make_const_mdspan(inplace.view()), inplace.view());
      raft::resource::sync_stream(handle_, stream_);
      ASSERT_TRUE(devArrMatch(out.data_handle(),
                              inplace.data_handle(),
                              out.size(),
                              cuvs::CompareApprox<float>(1e-5f),
                              stream_));
    } else {
      EXPECT_THROW(rotate(handle_, rot, raft::make_const_mdspan(in.view()), in.view()),
                   raft::logic_error);
    }

    // rotate() does not pad: an unpadded view must be rejected.
    if (ps_.dim != pd) {
      auto unpadded = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
        (const float*)in.data_handle(), n, static_cast<int64_t>(ps_.dim));
      auto unpadded_out = raft::make_device_matrix_view<float, int64_t, raft::row_major>(
        out.data_handle(), n, static_cast<int64_t>(ps_.dim));
      EXPECT_THROW(rotate(handle_, rot, unpadded, unpadded_out), raft::logic_error);
    }
  }

  // -------------------------------------------------------------------------
  // 3 + 4. the codes / factors are sane, and transform() is deterministic
  // -------------------------------------------------------------------------
  void testTransform()
  {
    const auto p     = make_params();
    auto pipe        = init_pipeline(handle_, p, ps_.dim);
    const int64_t pd = pipe.quantizer.padded_dim;
    const int64_t n  = ps_.rows;
    const size_t len = static_cast<size_t>(n) * static_cast<size_t>(pd);
    auto data        = raft::make_const_mdspan(dataset_.view());

    auto codes_a = raft::make_device_matrix<CodeT, int64_t>(handle_, n, pd);
    auto codes_b = raft::make_device_matrix<CodeT, int64_t>(handle_, n, pd);
    auto delta_a = raft::make_device_vector<float, int64_t>(handle_, n);
    auto delta_b = raft::make_device_vector<float, int64_t>(handle_, n);
    auto vl_a    = raft::make_device_vector<float, int64_t>(handle_, n);
    auto vl_b    = raft::make_device_vector<float, int64_t>(handle_, n);

    // --- (codes, delta, vl) output mode, no centroid ---
    transform(handle_,
              pipe.quantizer,
              pipe.rotator,
              data,
              std::nullopt,
              codes_a.view(),
              delta_a.view(),
              vl_a.view());
    transform(handle_,
              pipe.quantizer,
              pipe.rotator,
              data,
              std::nullopt,
              codes_b.view(),
              delta_b.view(),
              vl_b.view());
    raft::resource::sync_stream(handle_, stream_);

    auto h_codes_a = verify_codes(codes_a.data_handle(), len);
    auto h_codes_b = verify_codes(codes_b.data_handle(), len);
    verify_finite(delta_a.data_handle(), static_cast<size_t>(n), "delta", /* positive */ true);
    verify_finite(vl_a.data_handle(), static_cast<size_t>(n), "vl");
    verify_vl(delta_a.data_handle(), vl_a.data_handle(), n);

    // determinism: the same pipeline on the same input -> identical output
    expect_same_codes(h_codes_a, h_codes_b, "transform() (delta, vl) is not deterministic");
    ASSERT_TRUE(devArrMatch(delta_a.data_handle(),
                            delta_b.data_handle(),
                            static_cast<size_t>(n),
                            cuvs::Compare<float>(),
                            stream_));
    ASSERT_TRUE(devArrMatch(vl_a.data_handle(),
                            vl_b.data_handle(),
                            static_cast<size_t>(n),
                            cuvs::Compare<float>(),
                            stream_));

    // --- (codes, factors) output mode, with a centroid ---
    auto ff_codes_a = raft::make_device_matrix<CodeT, int64_t>(handle_, n, pd);
    auto ff_codes_b = raft::make_device_matrix<CodeT, int64_t>(handle_, n, pd);
    auto factors_a  = raft::make_device_matrix<float, int64_t>(handle_, n, 3);
    auto factors_b  = raft::make_device_matrix<float, int64_t>(handle_, n, 3);

    auto centroid_view = raft::make_const_mdspan(centroid_.view());
    std::optional<raft::device_vector_view<const float, int64_t>> centroid_opt(centroid_view);
    transform(handle_,
              pipe.quantizer,
              pipe.rotator,
              data,
              centroid_opt,
              ff_codes_a.view(),
              factors_a.view());
    transform(handle_,
              pipe.quantizer,
              pipe.rotator,
              data,
              centroid_opt,
              ff_codes_b.view(),
              factors_b.view());
    raft::resource::sync_stream(handle_, stream_);

    auto h_ff_a = verify_codes(ff_codes_a.data_handle(), len);
    auto h_ff_b = verify_codes(ff_codes_b.data_handle(), len);
    verify_finite(factors_a.data_handle(), static_cast<size_t>(n) * 3, "factor");
    {
      // f_error is an error bound, so it can never be negative.
      auto h_factors = to_host(factors_a.data_handle(), static_cast<size_t>(n) * 3);
      size_t n_bad   = 0;
      for (int64_t i = 0; i < n; ++i) {
        if (!(h_factors[static_cast<size_t>(i) * 3 + 2] >= 0.0f)) { ++n_bad; }
      }
      EXPECT_EQ(n_bad, 0u) << n_bad << " of " << n << " f_error factors are negative";
    }
    expect_same_codes(h_ff_a, h_ff_b, "transform() (factors) is not deterministic");
    ASSERT_TRUE(devArrMatch(factors_a.data_handle(),
                            factors_b.data_handle(),
                            static_cast<size_t>(n) * 3,
                            cuvs::Compare<float>(),
                            stream_));

    // Codes are computed on residuals w.r.t. the centroid, so passing one must
    // change the result.
    {
      size_t n_diff = 0;
      for (size_t i = 0; i < len; ++i) {
        if (h_codes_a[i] != h_ff_a[i]) { ++n_diff; }
      }
      EXPECT_GT(n_diff, 0u) << "subtracting a centroid did not change the codes";
    }

    if (ps_.dim % 32 == 0) { testResidualsMatchTransform(pipe, h_codes_a, delta_a, vl_a); }
  }

  /**
   * With no padding needed, transform() is exactly rotate() followed by
   * transform_residuals() - same rotator, same launch shapes, same quantizer
   * kernel - so the two paths must agree.
   */
  void testResidualsMatchTransform(pipeline const& pipe,
                                   std::vector<CodeT> const& h_codes_ref,
                                   raft::device_vector<float, int64_t> const& delta_ref,
                                   raft::device_vector<float, int64_t> const& vl_ref)
  {
    const int64_t pd = pipe.quantizer.padded_dim;
    const int64_t n  = ps_.rows;
    const size_t len = static_cast<size_t>(n) * static_cast<size_t>(pd);

    auto residuals = raft::make_device_matrix<float, int64_t>(handle_, n, pd);
    rotate(handle_, pipe.rotator, raft::make_const_mdspan(dataset_.view()), residuals.view());
    raft::resource::sync_stream(handle_, stream_);
    auto residuals_view = raft::make_const_mdspan(residuals.view());

    auto codes = raft::make_device_matrix<CodeT, int64_t>(handle_, n, pd);
    auto delta = raft::make_device_vector<float, int64_t>(handle_, n);
    auto vl    = raft::make_device_vector<float, int64_t>(handle_, n);
    transform_residuals(
      handle_, pipe.quantizer, residuals_view, codes.view(), delta.view(), vl.view());
    raft::resource::sync_stream(handle_, stream_);

    auto h_codes = verify_codes(codes.data_handle(), len);
    expect_same_codes(
      h_codes_ref, h_codes, "transform() and rotate() + transform_residuals() disagree");
    ASSERT_TRUE(devArrMatch(delta_ref.data_handle(),
                            delta.data_handle(),
                            static_cast<size_t>(n),
                            cuvs::CompareApprox<float>(1e-6f),
                            stream_));
    ASSERT_TRUE(devArrMatch(vl_ref.data_handle(),
                            vl.data_handle(),
                            static_cast<size_t>(n),
                            cuvs::CompareApprox<float>(1e-6f),
                            stream_));

    // The full-factor residual overload treats std::nullopt as a zero centroid,
    // which is what the (delta, vl) call above used too.
    auto ff_codes = raft::make_device_matrix<CodeT, int64_t>(handle_, n, pd);
    auto factors  = raft::make_device_matrix<float, int64_t>(handle_, n, 3);
    transform_residuals(
      handle_, pipe.quantizer, residuals_view, std::nullopt, ff_codes.view(), factors.view());
    raft::resource::sync_stream(handle_, stream_);
    auto h_ff_codes = verify_codes(ff_codes.data_handle(), len);
    if (ps_.use_fast) {
      // Both modes then quantize with the same shared scaling factor, so the
      // codes must match. (On the slow path they need not: the full-factor
      // launcher hardcodes 64/64 rescale samples instead of forwarding
      // params::coarse_samples / params::fine_samples.)
      expect_same_codes(h_codes_ref,
                        h_ff_codes,
                        "the (delta, vl) and (factors) output modes emit different codes");
    }
    verify_finite(factors.data_handle(), static_cast<size_t>(n) * 3, "residual factor");

    // Wrong extents must be rejected rather than silently reinterpreted.
    if (n > 1) {
      auto short_delta = raft::make_device_vector<float, int64_t>(handle_, n - 1);
      EXPECT_THROW(transform_residuals(handle_,
                                       pipe.quantizer,
                                       residuals_view,
                                       codes.view(),
                                       short_delta.view(),
                                       vl.view()),
                   raft::logic_error);
    }
  }

 private:
  raft::resources handle_;
  RabitqQuantizationInputs ps_;
  cudaStream_t stream_;
  raft::device_matrix<float, int64_t, raft::row_major> dataset_;
  raft::device_vector<float, int64_t> centroid_;
};

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/**
 * Codes of `ex_bits + 1 <= 8` bits fit uint8_t. The dim sweep covers
 * power-of-two padded dims (64, 128), non-power-of-two ones (96, 960 -> the
 * truncated FHT plus Kac's walk) and one dim that actually gets padded
 * (100 -> 128). The two row counts straddle the FHT warp/cooperative kernel
 * switch (1000 vectors).
 */
const std::vector<RabitqQuantizationInputs> generate_inputs_narrow()
{
  auto inputs = raft::util::itertools::product<RabitqQuantizationInputs>(
    {13, 1000},
    {64, 96, 100, 128},
    {rotator_kind::matmul, rotator_kind::fht_kac},
    {1u, 3u},
    {true});
  // dim = 960 exercises the non-power-of-two FHT with trunc_dim = 512. Only one
  // matmul instance there: its rotation matrix costs O(dim^3) host work.
  auto fht_960 = raft::util::itertools::product<RabitqQuantizationInputs>(
    {13, 1000}, {960}, {rotator_kind::fht_kac}, {3u}, {true});
  auto matmul_960 = raft::util::itertools::product<RabitqQuantizationInputs>(
    {1000}, {960}, {rotator_kind::matmul}, {3u}, {true});
  // use_fast = false runs the per-vector rescale search instead of the shared
  // pre-estimated scaling factor.
  auto slow = raft::util::itertools::product<RabitqQuantizationInputs>(
    {64}, {96, 128}, {rotator_kind::fht_kac}, {1u, 3u}, {false});

  inputs.insert(inputs.end(), fht_960.begin(), fht_960.end());
  inputs.insert(inputs.end(), matmul_960.begin(), matmul_960.end());
  inputs.insert(inputs.end(), slow.begin(), slow.end());
  return inputs;
}

/** ex_bits = 8 needs 9 code bits, so those configurations require uint16_t. */
const std::vector<RabitqQuantizationInputs> generate_inputs_wide()
{
  return raft::util::itertools::product<RabitqQuantizationInputs>(
    {512}, {96, 128}, {rotator_kind::matmul, rotator_kind::fht_kac}, {5u, 8u}, {true});
}

typedef RabitqQuantizationTest<uint8_t> RabitqQuantizationTest_uint8t;
TEST_P(RabitqQuantizationTest_uint8t, Init) { this->testInit(); }
TEST_P(RabitqQuantizationTest_uint8t, Rotate) { this->testRotate(); }
TEST_P(RabitqQuantizationTest_uint8t, Transform) { this->testTransform(); }

typedef RabitqQuantizationTest<uint16_t> RabitqQuantizationTest_uint16t;
TEST_P(RabitqQuantizationTest_uint16t, Init) { this->testInit(); }
TEST_P(RabitqQuantizationTest_uint16t, Rotate) { this->testRotate(); }
TEST_P(RabitqQuantizationTest_uint16t, Transform) { this->testTransform(); }

INSTANTIATE_TEST_CASE_P(RabitqQuantizationTest,
                        RabitqQuantizationTest_uint8t,
                        ::testing::ValuesIn(generate_inputs_narrow()));
INSTANTIATE_TEST_CASE_P(RabitqQuantizationTest,
                        RabitqQuantizationTest_uint16t,
                        ::testing::ValuesIn(generate_inputs_wide()));

// ---------------------------------------------------------------------------
// Checks that do not depend on the sweep
// ---------------------------------------------------------------------------

/** A code element type too narrow for `ex_bits + 1` bits must be rejected. */
TEST(RabitqQuantizationBasicTest, CodeWidthValidation)
{
  raft::resources handle;
  constexpr int64_t kDim  = 128;
  constexpr int64_t kRows = 8;

  params p;
  p.ex_bits = 8;  // 9 bits per code element
  auto pipe = init_pipeline(handle, p, kDim);

  const int64_t kPadded = pipe.quantizer.padded_dim;

  auto dataset = raft::make_device_matrix<float, int64_t>(handle, kRows, kDim);
  raft::random::RngState r(1234ULL);
  raft::random::uniform(handle, r, dataset.data_handle(), dataset.size(), -1.0f, 1.0f);

  auto data  = raft::make_const_mdspan(dataset.view());
  auto delta = raft::make_device_vector<float, int64_t>(handle, kRows);
  auto vl    = raft::make_device_vector<float, int64_t>(handle, kRows);

  auto narrow = raft::make_device_matrix<uint8_t, int64_t>(handle, kRows, kPadded);
  EXPECT_THROW(transform(handle,
                         pipe.quantizer,
                         pipe.rotator,
                         data,
                         std::nullopt,
                         narrow.view(),
                         delta.view(),
                         vl.view()),
               raft::logic_error);

  auto wide = raft::make_device_matrix<uint16_t, int64_t>(handle, kRows, kPadded);
  EXPECT_NO_THROW(transform(handle,
                            pipe.quantizer,
                            pipe.rotator,
                            data,
                            std::nullopt,
                            wide.view(),
                            delta.view(),
                            vl.view()));
  raft::resource::sync_stream(handle);
}

/** A rotator and a quantizer built for different dims are not peers. */
TEST(RabitqQuantizationBasicTest, MismatchedPeers)
{
  raft::resources handle;
  params p;
  auto rot   = init_rotator(handle, p, 128);
  auto quant = init_quantizer(handle, p, 256);
  ASSERT_NE(rot.padded_dim, quant.padded_dim);

  auto dataset = raft::make_device_matrix<float, int64_t>(handle, 4, 256);
  raft::random::RngState r(1234ULL);
  raft::random::uniform(handle, r, dataset.data_handle(), dataset.size(), -1.0f, 1.0f);
  auto codes = raft::make_device_matrix<uint8_t, int64_t>(handle, 4, quant.padded_dim);
  auto delta = raft::make_device_vector<float, int64_t>(handle, 4);
  auto vl    = raft::make_device_vector<float, int64_t>(handle, 4);
  EXPECT_THROW(transform(handle,
                         quant,
                         rot,
                         raft::make_const_mdspan(dataset.view()),
                         std::nullopt,
                         codes.view(),
                         delta.view(),
                         vl.view()),
               raft::logic_error);
}

/** An empty batch is a no-op, not a zero-sized kernel launch. */
TEST(RabitqQuantizationBasicTest, EmptyInput)
{
  raft::resources handle;
  constexpr int64_t kDim = 64;
  params p;
  auto pipe = init_pipeline(handle, p, kDim);

  auto dataset = raft::make_device_matrix<float, int64_t>(handle, 0, kDim);
  auto codes   = raft::make_device_matrix<uint8_t, int64_t>(handle, 0, pipe.quantizer.padded_dim);
  auto delta   = raft::make_device_vector<float, int64_t>(handle, 0);
  auto vl      = raft::make_device_vector<float, int64_t>(handle, 0);
  EXPECT_NO_THROW(transform(handle,
                            pipe.quantizer,
                            pipe.rotator,
                            raft::make_const_mdspan(dataset.view()),
                            std::nullopt,
                            codes.view(),
                            delta.view(),
                            vl.view()));
  raft::resource::sync_stream(handle);
}

}  // namespace cuvs::preprocessing::quantize::rabitq
