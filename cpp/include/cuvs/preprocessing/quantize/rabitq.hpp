/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/core/export.hpp>

#include <cstdint>
#include <optional>

namespace CUVS_EXPORT cuvs {
namespace preprocessing {
namespace quantize {
namespace rabitq {

/**
 * @defgroup rabitq RaBitQ quantizer utilities
 * @{
 */

/**
 * @brief Random-rotation implementation used before quantization.
 *
 * Both kinds produce an orthogonal transform of the padded working dimension;
 * they differ only in cost and in the amount of state they carry.
 */
enum class rotator_kind : uint8_t {
  /** Dense `padded_dim x padded_dim` random orthogonal matrix (cuBLAS GEMM), O(n * dim^2). */
  matmul,
  /** Fast Hadamard Transform combined with Kac's walk, O(n * dim * log(dim)). */
  fht_kac
};

/**
 * @brief Rule used to derive the per-vector reconstruction scale `delta`.
 *
 * Only relevant for the `(codes, delta, vl)` output mode; the full-factor mode
 * emits `(f_add, f_rescale, f_error)` instead and ignores this setting.
 */
enum class delta_kind : uint8_t {
  /** `delta = (|residual| / |code|) * cos(residual, code)` - minimizes reconstruction error. */
  reconstruction = 0,
  /** `delta = (|residual| / |code|) / cos(residual, code)` - unbiased inner-product estimate. */
  unbiased = 1,
  /** `delta = |residual| / |code|` - plain norm ratio, no angular correction. */
  plain = 2
};

/**
 * @brief RaBitQ quantizer parameters.
 *
 * These parameters fully describe the quantizer: RaBitQ is data-oblivious, so
 * none of them are learned from a dataset (see make_quantizer / make_rotator).
 */
struct params {
  /**
   * Number of *extended* bits per dimension. The total code width is
   * `ex_bits + 1` bits (one sign bit plus `ex_bits` magnitude bits), so
   * `ex_bits = 0` is plain 1-bit RaBitQ and `ex_bits = 3` is the common
   * 4-bit configuration.
   *
   * Possible value range: [0, 8].
   */
  uint32_t ex_bits = 3;
  /** Random-rotation implementation. */
  rotator_kind rotator = rotator_kind::fht_kac;
  /**
   * Seed for the host RNG that draws the rotation state and the random probe
   * vectors used to estimate the constant scaling factor.
   */
  uint64_t seed = 12345ULL;
  /** Rule used to derive the per-vector `delta` / `vl` pair. */
  delta_kind delta_mode = delta_kind::reconstruction;
  /**
   * When true, every vector is quantized with a single pre-estimated scaling
   * factor (`quantizer::const_scaling_factor`) instead of running the
   * per-vector rescale search. This is substantially faster and is the
   * recommended setting; disable it to trade build throughput for slightly
   * tighter codes.
   */
  bool use_fast = true;
  /** Number of samples in the coarse stage of the per-vector rescale search. */
  uint32_t coarse_samples = 64;
  /** Number of samples in the refinement stage of the per-vector rescale search. */
  uint32_t fine_samples = 64;
};

/**
 * @brief Random rotation applied to the data before quantization.
 *
 * The rotator carries no learned information - its state is drawn from
 * `params::seed` - but it *is* part of the code representation: codes produced
 * with one rotator are only comparable to queries transformed by the very same
 * rotator. There is deliberately no serialization API here, so the caller owns
 * this object and is responsible for keeping it alongside the codes (e.g. by
 * storing `params::seed` and re-running make_rotator, or by embedding the
 * buffers below in its own index format).
 *
 * A rotator is an independent peer of the quantizer: neither contains the
 * other, and a single rotator may be shared by several quantizers.
 */
struct rotator {
  /** Which implementation the state below belongs to. */
  rotator_kind kind = rotator_kind::fht_kac;
  /**
   * Working dimension: `dim` rounded up to a multiple of 32.
   *
   * This is the rotator's full identity alongside `kind` and the state below —
   * both transforms operate purely on the padded width (`fht_kac` derives its
   * segment size as `2^floor_log2(padded_dim)`), so the original unpadded `dim`
   * is not part of the rotator.
   */
  int64_t padded_dim = 0;
  /**
   * `padded_dim x padded_dim` row-major random orthogonal matrix.
   * Empty unless `kind == rotator_kind::matmul`.
   */
  raft::device_matrix<float, int64_t, raft::row_major> rotation_matrix;
  /**
   * Packed sign bits of the four Kac's-walk passes (`4 * padded_dim / 8` bytes).
   * Empty unless `kind == rotator_kind::fht_kac`.
   */
  raft::device_vector<uint8_t, int64_t> flip_bits;

  /** @brief Construct an empty (uninitialized) rotator. */
  explicit rotator(raft::resources const& res)
    : rotation_matrix(raft::make_device_matrix<float, int64_t, raft::row_major>(res, 0, 0)),
      flip_bits(raft::make_device_vector<uint8_t, int64_t>(res, 0))
  {
  }
};

/**
 * @brief Stores the (data-independent) configuration used to quantize rotated residuals.
 *
 * RaBitQ needs no training data, so this is plain metadata plus the single
 * scalar estimated at init time. Like the rotator, it is owned by the caller
 * and has no serialization API - it is cheap to rebuild from `params` and
 * `dim`.
 */
struct quantizer {
  /** Parameters used to build this quantizer. */
  params params_quantizer;
  /**
   * Scaling factor shared by all vectors on the `params::use_fast` path,
   * estimated from random Gaussian probe vectors at init time.
   * Left at 1.0 (and unused) when `params::use_fast` is false.
   */
  float const_scaling_factor = 1.0f;
  /** Dimensionality of the un-rotated input vectors. */
  int64_t dim = 0;
  /** Working dimension: `dim` rounded up to a multiple of 32. */
  int64_t padded_dim = 0;
};

/**
 * @brief Convenience composition of the two peers for the common end-to-end flow.
 *
 * Holds one rotator and one quantizer; it adds no state and no behaviour of its
 * own. Pass `pipeline::quantizer` and `pipeline::rotator` to transform().
 */
struct pipeline {
  /** Rotation applied before quantization. */
  rabitq::rotator rotator;
  /** Quantizer applied to the rotated residuals. */
  rabitq::quantizer quantizer;

  /** @brief Construct an empty (uninitialized) pipeline. */
  explicit pipeline(raft::resources const& res) : rotator(res) {}
};

/**
 * @brief Initializes a RaBitQ rotator.
 *
 * No training data is required: the rotation is drawn from `params::seed`
 * alone. The returned object owns its device buffers; keep it for as long as
 * the codes it produced are in use (see rotator).
 *
 * Usage example:
 * @code{.cpp}
 * raft::resources res;
 * cuvs::preprocessing::quantize::rabitq::params params;
 * auto rotator =
 *   cuvs::preprocessing::quantize::rabitq::make_rotator(res, params, dataset.extent(1));
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure the rotator, e.g. kind and seed
 * @param[in] dim dimensionality of the un-rotated input vectors
 *
 * @return rotator
 */
rotator make_rotator(raft::resources const& res, params const& params, int64_t dim);

/**
 * @brief Initializes a RaBitQ quantizer.
 *
 * No training data is required. When `params::use_fast` is set, this estimates
 * the constant scaling factor on the GPU, which is the only non-trivial work
 * performed here.
 *
 * Usage example:
 * @code{.cpp}
 * raft::resources res;
 * cuvs::preprocessing::quantize::rabitq::params params;
 * auto quantizer =
 *   cuvs::preprocessing::quantize::rabitq::make_quantizer(res, params, dataset.extent(1));
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure the quantizer, e.g. ex_bits
 * @param[in] dim dimensionality of the un-rotated input vectors
 *
 * @return quantizer
 */
quantizer make_quantizer(raft::resources const& res, params const& params, int64_t dim);

/**
 * @brief Initializes a rotator and a quantizer from the same parameters.
 *
 * Equivalent to calling make_rotator() and make_quantizer() with the same
 * arguments. No training data is required.
 *
 * Usage example:
 * @code{.cpp}
 * raft::resources res;
 * cuvs::preprocessing::quantize::rabitq::params params;
 * auto pipeline =
 *   cuvs::preprocessing::quantize::rabitq::make_pipeline(res, params, dataset.extent(1));
 * cuvs::preprocessing::quantize::rabitq::transform(res, pipeline.quantizer, pipeline.rotator,
 *   dataset, std::nullopt, codes.view(), delta.view(), vl.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configure both components
 * @param[in] dim dimensionality of the un-rotated input vectors
 *
 * @return pipeline
 */
pipeline make_pipeline(raft::resources const& res, params const& params, int64_t dim);

/**
 * @brief Applies the random rotation to already-padded rows.
 *
 * Both `in` and `out` must be `n_rows x rotator::padded_dim`; this entry point
 * does not pad. In-place operation (`in.data_handle() == out.data_handle()`) is
 * supported by `rotator_kind::fht_kac` only.
 *
 * Use this to rotate queries or externally computed centroids so that they live
 * in the same rotated space as the codes.
 *
 * Usage example:
 * @code{.cpp}
 * auto rotated = raft::make_device_matrix<float, int64_t>(res, n_queries, rotator.padded_dim);
 * cuvs::preprocessing::quantize::rabitq::rotate(res, rotator, padded_queries, rotated.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] rotator a RaBitQ rotator
 * @param[in] in a row-major `n_rows x padded_dim` matrix view on device
 * @param[out] out a row-major `n_rows x padded_dim` matrix view on device
 *
 */
void rotate(raft::resources const& res,
            rotator const& rotator,
            raft::device_matrix_view<const float, int64_t, raft::row_major> in,
            raft::device_matrix_view<float, int64_t, raft::row_major> out);

/**
 * @brief Quantizes a raw dataset, emitting per-vector `(delta, vl)` factors.
 *
 * Performs the whole chain: zero-pad `dim -> padded_dim`, rotate, optionally
 * subtract the rotated centroid, then quantize the residuals. One code element
 * is written per padded dimension; each holds an unsigned
 * `params::ex_bits + 1` bit value, so `codes` must be `n_rows x padded_dim`.
 *
 * A temporary `n_rows x padded_dim` float buffer holding the rotated residuals
 * is allocated from the workspace resource, so large datasets should be
 * transformed in batches (a pool workspace resource is recommended).
 *
 * Usage example:
 * @code{.cpp}
 * raft::resources res;
 * cuvs::preprocessing::quantize::rabitq::params params;
 * auto quantizer = cuvs::preprocessing::quantize::rabitq::make_quantizer(res, params, dim);
 * auto rotator   = cuvs::preprocessing::quantize::rabitq::make_rotator(res, params, dim);
 * auto codes = raft::make_device_matrix<uint8_t, int64_t>(res, n_rows, quantizer.padded_dim);
 * auto delta = raft::make_device_vector<float, int64_t>(res, n_rows);
 * auto vl    = raft::make_device_vector<float, int64_t>(res, n_rows);
 * cuvs::preprocessing::quantize::rabitq::transform(res, quantizer, rotator, dataset,
 *   std::nullopt, codes.view(), delta.view(), vl.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a RaBitQ quantizer
 * @param[in] rotator the rotator whose `padded_dim` matches the quantizer
 * @param[in] dataset a row-major `n_rows x dim` matrix view on device
 * @param[in] centroid optional un-rotated `dim` centroid on device; when set, codes are
 *            computed on the residuals w.r.t. this centroid
 * @param[out] codes a row-major `n_rows x padded_dim` matrix view on device
 * @param[out] delta per-vector scale, `n_rows` elements on device
 * @param[out] vl per-vector offset (`delta * -(2^ex_bits - 0.5)`), `n_rows` elements on device
 *
 */
void transform(raft::resources const& res,
               quantizer const& quantizer,
               rotator const& rotator,
               raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
               std::optional<raft::device_vector_view<const float, int64_t>> centroid,
               raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
               raft::device_vector_view<float, int64_t> delta,
               raft::device_vector_view<float, int64_t> vl);

/** @copydoc transform */
void transform(raft::resources const& res,
               quantizer const& quantizer,
               rotator const& rotator,
               raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
               std::optional<raft::device_vector_view<const float, int64_t>> centroid,
               raft::device_matrix_view<uint16_t, int64_t, raft::row_major> codes,
               raft::device_vector_view<float, int64_t> delta,
               raft::device_vector_view<float, int64_t> vl);

/**
 * @brief Quantizes a raw dataset, emitting the full `(f_add, f_rescale, f_error)` factor triplet.
 *
 * Same chain as the `(delta, vl)` overload, but the per-vector factors are the
 * ones used for approximate-distance estimation during search:
 * `factors[i] = { f_add, f_rescale, f_error }`.
 *
 * A temporary `n_rows x padded_dim` float buffer holding the rotated residuals
 * is allocated from the workspace resource, so large datasets should be
 * transformed in batches (a pool workspace resource is recommended).
 *
 * The rotated centroid is an internal temporary as well. A search implementation
 * that needs it can reproduce it exactly by zero-padding `centroid` to
 * `padded_dim` and calling rotate() with the same rotator.
 *
 * Usage example:
 * @code{.cpp}
 * auto codes   = raft::make_device_matrix<uint8_t, int64_t>(res, n_rows, quantizer.padded_dim);
 * auto factors = raft::make_device_matrix<float, int64_t>(res, n_rows, 3);
 * cuvs::preprocessing::quantize::rabitq::transform(res, quantizer, rotator, dataset,
 *   std::make_optional(centroid.view()), codes.view(), factors.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a RaBitQ quantizer
 * @param[in] rotator the rotator whose `padded_dim` matches the quantizer
 * @param[in] dataset a row-major `n_rows x dim` matrix view on device
 * @param[in] centroid optional un-rotated `dim` centroid on device; treated as zero when unset
 * @param[out] codes a row-major `n_rows x padded_dim` matrix view on device
 * @param[out] factors a row-major `n_rows x 3` matrix view on device
 *
 */
void transform(raft::resources const& res,
               quantizer const& quantizer,
               rotator const& rotator,
               raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
               std::optional<raft::device_vector_view<const float, int64_t>> centroid,
               raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
               raft::device_matrix_view<float, int64_t, raft::row_major> factors);

/** @copydoc transform */
void transform(raft::resources const& res,
               quantizer const& quantizer,
               rotator const& rotator,
               raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
               std::optional<raft::device_vector_view<const float, int64_t>> centroid,
               raft::device_matrix_view<uint16_t, int64_t, raft::row_major> codes,
               raft::device_matrix_view<float, int64_t, raft::row_major> factors);

/**
 * @brief Quantizes pre-rotated residuals, emitting per-vector `(delta, vl)` factors.
 *
 * The rotator is not needed here: `residuals` must already be rotated, padded
 * to `quantizer::padded_dim` and centered (centroid subtracted). Use this when
 * the residuals are produced elsewhere, e.g. by an IVF build that has already
 * rotated its data, and rotate() plus a subtraction of your own.
 *
 * Usage example:
 * @code{.cpp}
 * auto codes = raft::make_device_matrix<uint8_t, int64_t>(res, n_rows, quantizer.padded_dim);
 * auto delta = raft::make_device_vector<float, int64_t>(res, n_rows);
 * auto vl    = raft::make_device_vector<float, int64_t>(res, n_rows);
 * cuvs::preprocessing::quantize::rabitq::transform_residuals(res, quantizer, residuals,
 *   codes.view(), delta.view(), vl.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a RaBitQ quantizer
 * @param[in] residuals a row-major `n_rows x padded_dim` matrix view on device, rotated and
 *            centered
 * @param[out] codes a row-major `n_rows x padded_dim` matrix view on device
 * @param[out] delta per-vector scale, `n_rows` elements on device
 * @param[out] vl per-vector offset (`delta * -(2^ex_bits - 0.5)`), `n_rows` elements on device
 *
 */
void transform_residuals(raft::resources const& res,
                         quantizer const& quantizer,
                         raft::device_matrix_view<const float, int64_t, raft::row_major> residuals,
                         raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
                         raft::device_vector_view<float, int64_t> delta,
                         raft::device_vector_view<float, int64_t> vl);

/** @copydoc transform_residuals */
void transform_residuals(raft::resources const& res,
                         quantizer const& quantizer,
                         raft::device_matrix_view<const float, int64_t, raft::row_major> residuals,
                         raft::device_matrix_view<uint16_t, int64_t, raft::row_major> codes,
                         raft::device_vector_view<float, int64_t> delta,
                         raft::device_vector_view<float, int64_t> vl);

/**
 * @brief Quantizes pre-rotated residuals, emitting the full factor triplet.
 *
 * As above, `residuals` must already be rotated, padded to
 * `quantizer::padded_dim` and centered. The full-factor formula also reads the
 * centroid, which must be supplied *in the rotated, padded space* (rotate() it
 * with the same rotator that produced the residuals); pass `std::nullopt` to
 * treat it as zero.
 *
 * Usage example:
 * @code{.cpp}
 * auto codes   = raft::make_device_matrix<uint8_t, int64_t>(res, n_rows, quantizer.padded_dim);
 * auto factors = raft::make_device_matrix<float, int64_t>(res, n_rows, 3);
 * cuvs::preprocessing::quantize::rabitq::transform_residuals(res, quantizer, residuals,
 *   std::make_optional(rotated_centroid.view()), codes.view(), factors.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] quantizer a RaBitQ quantizer
 * @param[in] residuals a row-major `n_rows x padded_dim` matrix view on device, rotated and
 *            centered
 * @param[in] rotated_centroid optional rotated, padded centroid (`padded_dim` elements on device)
 * @param[out] codes a row-major `n_rows x padded_dim` matrix view on device
 * @param[out] factors a row-major `n_rows x 3` matrix view on device
 *
 */
void transform_residuals(
  raft::resources const& res,
  quantizer const& quantizer,
  raft::device_matrix_view<const float, int64_t, raft::row_major> residuals,
  std::optional<raft::device_vector_view<const float, int64_t>> rotated_centroid,
  raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
  raft::device_matrix_view<float, int64_t, raft::row_major> factors);

/** @copydoc transform_residuals */
void transform_residuals(
  raft::resources const& res,
  quantizer const& quantizer,
  raft::device_matrix_view<const float, int64_t, raft::row_major> residuals,
  std::optional<raft::device_vector_view<const float, int64_t>> rotated_centroid,
  raft::device_matrix_view<uint16_t, int64_t, raft::row_major> codes,
  raft::device_matrix_view<float, int64_t, raft::row_major> factors);

/** @} */  // end of group rabitq

}  // namespace rabitq
}  // namespace quantize
}  // namespace preprocessing
}  // namespace CUVS_EXPORT cuvs
