/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>
#include <dlpack/dlpack.h>
#include <stdint.h>

#include <cuvs/core/export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup preprocessing_c_rabitq C API for RaBitQ Quantizer
 * @{
 */

/**
 * @brief Random-rotation implementation used before quantization.
 *
 * Both kinds produce an orthogonal transform of the padded working dimension;
 * they differ only in cost and in the amount of state they carry.
 */
enum cuvsRabitqRotatorKind {
  /** Dense `padded_dim x padded_dim` random orthogonal matrix (cuBLAS GEMM), O(n * dim^2). */
  CUVS_RABITQ_ROTATOR_KIND_MATMUL = 0,
  /** Fast Hadamard Transform combined with Kac's walk, O(n * dim * log(dim)). */
  CUVS_RABITQ_ROTATOR_KIND_FHT_KAC = 1,
};

/**
 * @brief Rule used to derive the per-vector reconstruction scale `delta`.
 *
 * Only relevant for cuvsRabitqTransform / cuvsRabitqTransformResiduals, which
 * emit `(codes, delta, vl)`; the full-factor entry points emit
 * `(f_add, f_rescale, f_error)` instead and ignore this setting.
 */
enum cuvsRabitqDeltaKind {
  /** `delta = (|residual| / |code|) * cos(residual, code)` - minimizes reconstruction error. */
  CUVS_RABITQ_DELTA_KIND_RECONSTRUCTION = 0,
  /** `delta = (|residual| / |code|) / cos(residual, code)` - unbiased inner-product estimate. */
  CUVS_RABITQ_DELTA_KIND_UNBIASED = 1,
  /** `delta = |residual| / |code|` - plain norm ratio, no angular correction. */
  CUVS_RABITQ_DELTA_KIND_PLAIN = 2,
};

/**
 * @brief RaBitQ quantizer parameters.
 *
 * These parameters fully describe both the rotator and the encoder: RaBitQ is
 * data-oblivious, so none of them are learned from a dataset. This is why the
 * factories below take a `dim` instead of a training dataset.
 */
struct cuvsRabitqQuantizerParams {
  /**
   * Number of *extended* bits per dimension. The total code width is
   * `ex_bits + 1` bits (one sign bit plus `ex_bits` magnitude bits), so
   * `ex_bits = 0` is plain 1-bit RaBitQ and `ex_bits = 3` is the common
   * 4-bit configuration.
   *
   * Possible value range: [0, 8].
   */
  uint32_t ex_bits;
  /** Random-rotation implementation. */
  enum cuvsRabitqRotatorKind rotator;
  /**
   * Seed for the RNGs that draw the rotation state and the random probe
   * vectors used to estimate the constant scaling factor.
   */
  uint64_t seed;
  /** Rule used to derive the per-vector `delta` / `vl` pair. */
  enum cuvsRabitqDeltaKind delta_mode;
  /**
   * When true, every vector is quantized with a single pre-estimated scaling
   * factor (see cuvsRabitqEncoderGetConstScalingFactor) instead of running
   * the per-vector rescale search. This is substantially faster and is the
   * recommended setting; disable it to trade build throughput for slightly
   * tighter codes.
   */
  bool use_fast;
  /** Number of samples in the coarse stage of the per-vector rescale search. */
  uint32_t coarse_samples;
  /** Number of samples in the refinement stage of the per-vector rescale search. */
  uint32_t fine_samples;
};

typedef struct cuvsRabitqQuantizerParams* cuvsRabitqQuantizerParams_t;

/**
 * @brief Allocate RaBitQ quantizer params, and populate with default values
 *
 * @param[in] params cuvsRabitqQuantizerParams_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRabitqQuantizerParamsCreate(cuvsRabitqQuantizerParams_t* params);

/**
 * @brief De-allocate RaBitQ quantizer params
 *
 * @param[in] params
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRabitqQuantizerParamsDestroy(cuvsRabitqQuantizerParams_t params);

/**
 * @brief Opaque handle to the random rotation applied before quantization.
 *
 * The rotator carries no learned information - its state is drawn from
 * cuvsRabitqQuantizerParams::seed - but it *is* part of the code
 * representation: codes produced with one rotator are only comparable to
 * queries transformed by the very same rotator. There is deliberately no
 * serialization API, so the caller owns this handle and is responsible for
 * keeping it alongside the codes, by copying the buffers returned by
 * cuvsRabitqRotatorGetState into its own index format and restoring them with
 * cuvsRabitqRotatorMakeFromState. Re-running cuvsRabitqRotatorMake with the
 * same params reproduces them only on the same GPU model and library version.
 *
 * A rotator is an independent peer of the encoder: neither contains the
 * other, and a single rotator may be shared by several encoders.
 */
typedef struct {
  uintptr_t addr;
} cuvsRabitqRotator;

typedef cuvsRabitqRotator* cuvsRabitqRotator_t;

/**
 * @brief Allocate a RaBitQ rotator handle
 *
 * The handle starts out empty; populate it with cuvsRabitqRotatorMake,
 * cuvsRabitqRotatorMakeFromState or cuvsRabitqQuantizerMake before use.
 *
 * @param[in] rotator cuvsRabitqRotator_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRabitqRotatorCreate(cuvsRabitqRotator_t* rotator);

/**
 * @brief De-allocate a RaBitQ rotator, freeing its device buffers
 *
 * Invalidates every tensor previously filled in by cuvsRabitqRotatorGetState.
 *
 * @param[in] rotator
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRabitqRotatorDestroy(cuvsRabitqRotator_t rotator);

/**
 * @brief Opaque handle to the (data-independent) configuration used to quantize
 * rotated residuals.
 *
 * RaBitQ needs no training data, so this is plain metadata plus the single
 * scalar estimated at init time. Like the rotator it is owned by the caller and
 * has no serialization API - it is cheap to rebuild from the params and `dim`.
 */
typedef struct {
  uintptr_t addr;
} cuvsRabitqEncoder;

typedef cuvsRabitqEncoder* cuvsRabitqEncoder_t;

/**
 * @brief Allocate a RaBitQ encoder handle
 *
 * The handle starts out empty; populate it with cuvsRabitqEncoderMake or
 * cuvsRabitqQuantizerMake before use.
 *
 * @param[in] encoder cuvsRabitqEncoder_t to allocate
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRabitqEncoderCreate(cuvsRabitqEncoder_t* encoder);

/**
 * @brief De-allocate a RaBitQ encoder
 *
 * @param[in] encoder
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRabitqEncoderDestroy(cuvsRabitqEncoder_t encoder);

/**
 * @brief Initializes a RaBitQ rotator.
 *
 * No training data is required: the rotation is drawn from
 * cuvsRabitqQuantizerParams::seed alone, which is why this takes a `dim`
 * rather than a dataset. The rotator owns its device buffers; keep it alive for
 * as long as the codes it produced are in use.
 *
 * @param[in] res raft resource
 * @param[in] params configure the rotator, e.g. kind and seed
 * @param[in] dim dimensionality of the un-rotated input vectors
 * @param[out] rotator an allocated rotator handle to initialize
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRabitqRotatorMake(cuvsResources_t res,
                                              cuvsRabitqQuantizerParams_t params,
                                              int64_t dim,
                                              cuvsRabitqRotator_t rotator);

/**
 * @brief Initializes a RaBitQ rotator from saved state.
 *
 * The inverse of cuvsRabitqRotatorGetKind + cuvsRabitqRotatorGetState: restores
 * a rotator persisted next to the codes it produced. `state` must have the
 * shape and dtype cuvsRabitqRotatorGetState documents for `kind`, and the
 * padded dimension is taken from it. It is copied, so it may live in device or
 * host memory and may be freed once this returns.
 *
 * @param[in] res raft resource
 * @param[in] kind the rotation implementation `state` belongs to
 * @param[in] state the saved rotator state
 * @param[out] rotator an allocated rotator handle to initialize
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRabitqRotatorMakeFromState(cuvsResources_t res,
                                                       enum cuvsRabitqRotatorKind kind,
                                                       DLManagedTensor* state,
                                                       cuvsRabitqRotator_t rotator);

/**
 * @brief Initializes a RaBitQ encoder.
 *
 * No training data is required, which is why this takes a `dim` rather than a
 * dataset. When cuvsRabitqQuantizerParams::use_fast is set, this estimates the
 * constant scaling factor on the GPU, which is the only non-trivial work
 * performed here.
 *
 * @param[in] res raft resource
 * @param[in] params configure the encoder, e.g. ex_bits
 * @param[in] dim dimensionality of the un-rotated input vectors
 * @param[out] encoder an allocated encoder handle to initialize
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRabitqEncoderMake(cuvsResources_t res,
                                              cuvsRabitqQuantizerParams_t params,
                                              int64_t dim,
                                              cuvsRabitqEncoder_t encoder);

/**
 * @brief Initializes a RaBitQ quantizer: a rotator and an encoder from the same parameters.
 *
 * Equivalent to calling cuvsRabitqRotatorMake and cuvsRabitqEncoderMake with
 * the same arguments. No training data is required.
 *
 * There is deliberately no opaque quantizer handle: the C++ `quantizer` struct
 * is only a composition of the two independent peers - it adds no state and no
 * behaviour of its own - so two out-parameters are the honest C spelling of it.
 *
 * @param[in] res raft resource
 * @param[in] params configure both components
 * @param[in] dim dimensionality of the un-rotated input vectors
 * @param[out] rotator an allocated rotator handle to initialize
 * @param[out] encoder an allocated encoder handle to initialize
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRabitqQuantizerMake(cuvsResources_t res,
                                                cuvsRabitqQuantizerParams_t params,
                                                int64_t dim,
                                                cuvsRabitqRotator_t rotator,
                                                cuvsRabitqEncoder_t encoder);

/**
 * @brief Applies the random rotation to already-padded rows.
 *
 * Both `in` and `out` must be `n_rows x padded_dim` row-major float32 device
 * matrices (see cuvsRabitqRotatorGetPaddedDim); this entry point does not pad.
 * In-place operation (`in` and `out` pointing at the same buffer) is supported
 * by ::CUVS_RABITQ_ROTATOR_KIND_FHT_KAC only.
 *
 * Use this to rotate queries or externally computed centroids so that they live
 * in the same rotated space as the codes.
 *
 * @param[in] res raft resource
 * @param[in] rotator a RaBitQ rotator
 * @param[in] in a row-major `n_rows x padded_dim` float32 device matrix
 * @param[out] out a row-major `n_rows x padded_dim` float32 device matrix
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRabitqRotate(cuvsResources_t res,
                                         cuvsRabitqRotator_t rotator,
                                         DLManagedTensor* in,
                                         DLManagedTensor* out);

/**
 * @brief Quantizes a raw dataset, emitting per-vector `(delta, vl)` factors.
 *
 * Performs the whole chain: zero-pad `dim -> padded_dim`, rotate, optionally
 * subtract the rotated centroid, then quantize the residuals. One code element
 * is written per padded dimension; each holds an unsigned
 * `cuvsRabitqQuantizerParams::ex_bits + 1` bit value, so `codes` must be
 * `n_rows x padded_dim`.
 *
 * Rows are processed in internal batches, so the temporary float buffer holding
 * the rotated residuals, allocated from the workspace resource, covers one batch
 * rather than the whole dataset (a pool workspace resource is recommended).
 *
 * `dataset` may also be in host memory (kDLCPU). Its rows are then copied to the
 * device one batch at a time, and are no longer read once the call returns.
 * Pinned (kDLCUDAHost) and managed tensors are device-accessible and are read in
 * place. `centroid` and the outputs must be on device.
 *
 * @param[in] res raft resource
 * @param[in] encoder a RaBitQ encoder
 * @param[in] rotator the rotator whose padded dim matches the encoder
 * @param[in] dataset a row-major `n_rows x dim` float32 host or device matrix
 * @param[in] centroid optional un-rotated `dim` float32 device vector; when non-NULL, codes are
 *            computed on the residuals w.r.t. this centroid. Pass NULL to skip centering.
 * @param[out] codes a row-major `n_rows x padded_dim` device matrix of uint8 or uint16 codes
 * @param[out] delta per-vector scale, an `n_rows` float32 device vector
 * @param[out] vl per-vector offset (`delta * -(2^ex_bits - 0.5)`), an `n_rows` float32 device
 *             vector
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRabitqTransform(cuvsResources_t res,
                                            cuvsRabitqEncoder_t encoder,
                                            cuvsRabitqRotator_t rotator,
                                            DLManagedTensor* dataset,
                                            DLManagedTensor* centroid,
                                            DLManagedTensor* codes,
                                            DLManagedTensor* delta,
                                            DLManagedTensor* vl);

/**
 * @brief Quantizes a raw dataset, emitting the full `(f_add, f_rescale, f_error)` factor triplet.
 *
 * Same chain as cuvsRabitqTransform, but the per-vector factors are the ones
 * used for approximate-distance estimation during search:
 * `factors[i] = { f_add, f_rescale, f_error }`.
 *
 * Rows are processed in internal batches, so the temporary float buffer holding
 * the rotated residuals, allocated from the workspace resource, covers one batch
 * rather than the whole dataset (a pool workspace resource is recommended).
 * As for cuvsRabitqTransform, `dataset` may also be in host memory.
 *
 * The rotated centroid is an internal temporary as well. A search
 * implementation that needs it can reproduce it exactly by zero-padding
 * `centroid` to `padded_dim` and calling cuvsRabitqRotate with the same
 * rotator.
 *
 * @param[in] res raft resource
 * @param[in] encoder a RaBitQ encoder
 * @param[in] rotator the rotator whose padded dim matches the encoder
 * @param[in] dataset a row-major `n_rows x dim` float32 host or device matrix
 * @param[in] centroid optional un-rotated `dim` float32 device vector; treated as zero when NULL
 * @param[out] codes a row-major `n_rows x padded_dim` device matrix of uint8 or uint16 codes
 * @param[out] factors a row-major `n_rows x 3` float32 device matrix
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRabitqTransformFull(cuvsResources_t res,
                                                cuvsRabitqEncoder_t encoder,
                                                cuvsRabitqRotator_t rotator,
                                                DLManagedTensor* dataset,
                                                DLManagedTensor* centroid,
                                                DLManagedTensor* codes,
                                                DLManagedTensor* factors);

/**
 * @brief Quantizes pre-rotated residuals, emitting per-vector `(delta, vl)` factors.
 *
 * The rotator is not needed here: `residuals` must already be rotated, padded
 * to the encoder's padded dim and centered (centroid subtracted). Use this
 * when the residuals are produced elsewhere, e.g. by an IVF build that has
 * already rotated its data, or by cuvsRabitqRotate plus a subtraction of your
 * own.
 *
 * @param[in] res raft resource
 * @param[in] encoder a RaBitQ encoder
 * @param[in] residuals a row-major `n_rows x padded_dim` float32 device matrix, rotated and
 *            centered
 * @param[out] codes a row-major `n_rows x padded_dim` device matrix of uint8 or uint16 codes
 * @param[out] delta per-vector scale, an `n_rows` float32 device vector
 * @param[out] vl per-vector offset (`delta * -(2^ex_bits - 0.5)`), an `n_rows` float32 device
 *             vector
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRabitqTransformResiduals(cuvsResources_t res,
                                                     cuvsRabitqEncoder_t encoder,
                                                     DLManagedTensor* residuals,
                                                     DLManagedTensor* codes,
                                                     DLManagedTensor* delta,
                                                     DLManagedTensor* vl);

/**
 * @brief Quantizes pre-rotated residuals, emitting the full factor triplet.
 *
 * As above, `residuals` must already be rotated, padded to the encoder's
 * padded dim and centered. The full-factor formula also reads the centroid,
 * which must be supplied *in the rotated, padded space* (run cuvsRabitqRotate
 * on it with the same rotator that produced the residuals); pass NULL to treat
 * it as zero.
 *
 * @param[in] res raft resource
 * @param[in] encoder a RaBitQ encoder
 * @param[in] residuals a row-major `n_rows x padded_dim` float32 device matrix, rotated and
 *            centered
 * @param[in] rotated_centroid optional rotated, padded centroid (a `padded_dim` float32 device
 *            vector); treated as zero when NULL
 * @param[out] codes a row-major `n_rows x padded_dim` device matrix of uint8 or uint16 codes
 * @param[out] factors a row-major `n_rows x 3` float32 device matrix
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRabitqTransformResidualsFull(cuvsResources_t res,
                                                         cuvsRabitqEncoder_t encoder,
                                                         DLManagedTensor* residuals,
                                                         DLManagedTensor* rotated_centroid,
                                                         DLManagedTensor* codes,
                                                         DLManagedTensor* factors);

/**
 * @brief Get which random-rotation implementation a rotator uses.
 *
 * Together with cuvsRabitqRotatorGetPaddedDim and cuvsRabitqRotatorGetState
 * this is the rotator's full identity. Because RaBitQ has no serialization API,
 * reading these three back out is how a C caller persists the rotator next to
 * the codes it produced; cuvsRabitqRotatorMakeFromState restores it.
 *
 * @param[in] rotator a RaBitQ rotator
 * @param[out] kind the rotation implementation
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRabitqRotatorGetKind(cuvsRabitqRotator_t rotator,
                                                 enum cuvsRabitqRotatorKind* kind);

/**
 * @brief Get the working dimension of a rotator: `dim` rounded up to a multiple of 32.
 *
 * Both transforms operate purely on the padded width, so the original unpadded
 * `dim` is not part of the rotator.
 *
 * @param[in] rotator a RaBitQ rotator
 * @param[out] padded_dim the working dimension
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRabitqRotatorGetPaddedDim(cuvsRabitqRotator_t rotator,
                                                      int64_t* padded_dim);

/**
 * @brief Get the device-side state of a rotator.
 *
 * The shape and dtype depend on cuvsRabitqRotatorGetKind:
 * - ::CUVS_RABITQ_ROTATOR_KIND_MATMUL fills in the `padded_dim x padded_dim`
 *   row-major float32 rotation matrix `R`; cuvsRabitqRotate maps each row `x`
 *   to `R x`.
 * - ::CUVS_RABITQ_ROTATOR_KIND_FHT_KAC fills in the packed sign bits of the four
 *   Kac's-walk passes, a `4 * padded_dim / 8` element uint8 vector.
 *
 * The returned DLManagedTensor is a **non-owning view**: the device buffer stays
 * owned by `rotator` and becomes dangling once cuvsRabitqRotatorDestroy is
 * called on it. Copy the contents out if you need them to outlive the rotator.
 * Only the tensor's own shape/stride metadata is allocated here, and it is
 * released by invoking the tensor's `deleter`.
 *
 * @param[in] rotator a RaBitQ rotator
 * @param[out] state a view of the rotator's device state
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRabitqRotatorGetState(cuvsRabitqRotator_t rotator,
                                                  DLManagedTensor* state);

/**
 * @brief Get the dimensionality of the un-rotated input vectors an encoder expects.
 *
 * @param[in] encoder a RaBitQ encoder
 * @param[out] dim dimensionality of the un-rotated input vectors
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRabitqEncoderGetDim(cuvsRabitqEncoder_t encoder, int64_t* dim);

/**
 * @brief Get the working dimension of an encoder: `dim` rounded up to a multiple of 32.
 *
 * This is the number of code elements emitted per vector, i.e. the second
 * extent that the `codes` tensor must have.
 *
 * @param[in] encoder a RaBitQ encoder
 * @param[out] padded_dim the working dimension
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRabitqEncoderGetPaddedDim(cuvsRabitqEncoder_t encoder,
                                                      int64_t* padded_dim);

/**
 * @brief Get the scaling factor shared by all vectors on the fast path.
 *
 * Estimated from random Gaussian probe vectors when the encoder was made.
 * Left at 1.0 (and unused) when cuvsRabitqQuantizerParams::use_fast is false.
 *
 * @param[in] encoder a RaBitQ encoder
 * @param[out] const_scaling_factor the shared scaling factor
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsRabitqEncoderGetConstScalingFactor(cuvsRabitqEncoder_t encoder,
                                                               float* const_scaling_factor);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif
