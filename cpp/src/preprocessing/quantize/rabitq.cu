/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../neighbors/detail/ann_utils.cuh"
#include "./detail/rabitq/pipeline.cuh"
#include "./detail/rabitq/quantize.cuh"
#include "./detail/rabitq/rescale_search.cuh"
#include "./detail/rabitq/rotator.cuh"

#include <cuvs/preprocessing/quantize/rabitq.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace cuvs::preprocessing::quantize::rabitq {

namespace {

/** Largest `ex_bits` the detail layer's tight-start table covers. */
constexpr uint32_t kMaxExBits = 8;

/** Number of per-vector factors in the full-factor output mode. */
constexpr int64_t kNumFactors = 3;

/** Borrow the public rotator's buffers for the detail launchers. */
[[nodiscard]] auto as_ref(rabitq::rotator const& rot) -> detail::rotator_ref
{
  detail::rotator_ref ref;
  ref.padded_dim      = rot.padded_dim;
  ref.use_fht_kac     = rot.kind == rotator_kind::fht_kac;
  ref.rotation_matrix = ref.use_fht_kac ? nullptr : rot.rotation_matrix.data_handle();
  ref.flip_bits       = ref.use_fht_kac ? rot.flip_bits.data_handle() : nullptr;
  return ref;
}

void check_params(params const& p)
{
  RAFT_EXPECTS(p.ex_bits <= kMaxExBits, "rabitq: params.ex_bits must be in [0, 8]");
  RAFT_EXPECTS(p.coarse_samples >= 1 && p.fine_samples >= 1,
               "rabitq: params.coarse_samples and params.fine_samples must be >= 1");
}

void check_rotator(rabitq::rotator const& rot)
{
  RAFT_EXPECTS(rot.padded_dim > 0,
               "rabitq: rotator is not initialized (padded_dim == 0); use make_rotator()");
  RAFT_EXPECTS(rot.padded_dim % 32 == 0, "rabitq: rotator.padded_dim must be a multiple of 32");
  if (rot.kind == rotator_kind::matmul) {
    RAFT_EXPECTS(rot.rotation_matrix.extent(0) == rot.padded_dim &&
                   rot.rotation_matrix.extent(1) == rot.padded_dim,
                 "rabitq: rotator.rotation_matrix must be padded_dim x padded_dim");
  } else {
    RAFT_EXPECTS(rot.flip_bits.extent(0) == detail::flip_bytes(rot.padded_dim),
                 "rabitq: rotator.flip_bits must hold flip_bytes(padded_dim) bytes");
  }
}

void check_quantizer(rabitq::quantizer const& q)
{
  RAFT_EXPECTS(q.dim > 0, "rabitq: quantizer is not initialized (dim == 0); use make_quantizer()");
  RAFT_EXPECTS(q.padded_dim == detail::padded_dim(q.dim),
               "rabitq: quantizer.padded_dim is inconsistent with quantizer.dim");
  check_params(q.params_quantizer);
}

/** The quantizer and the rotator must agree on the padded working dimension. */
void check_peers(rabitq::quantizer const& q, rabitq::rotator const& rot)
{
  check_quantizer(q);
  check_rotator(rot);
  RAFT_EXPECTS(rot.padded_dim == q.padded_dim,
               "rabitq: rotator.padded_dim must match quantizer.padded_dim");
}

/** One code element per padded dimension, each holding an `ex_bits + 1` bit value. */
template <typename CodeT>
void check_codes(rabitq::quantizer const& q,
                 int64_t n_rows,
                 raft::device_matrix_view<CodeT, int64_t, raft::row_major> codes)
{
  constexpr uint32_t kCodeBits = sizeof(CodeT) * 8;
  RAFT_EXPECTS(q.params_quantizer.ex_bits + 1 <= kCodeBits,
               "rabitq: the code element type is too narrow for ex_bits + 1 bits");
  RAFT_EXPECTS(codes.extent(0) == n_rows, "rabitq: codes.extent(0) must match the row count");
  RAFT_EXPECTS(codes.extent(1) == q.padded_dim,
               "rabitq: codes.extent(1) must equal quantizer.padded_dim");
}

void check_rows(int64_t n_rows)
{
  RAFT_EXPECTS(n_rows >= 0 && n_rows <= static_cast<int64_t>(std::numeric_limits<int>::max()),
               "rabitq: the number of rows passed in a single call must fit in a 32-bit int; "
               "process larger datasets in batches");
}

void check_delta_vl(int64_t n_rows,
                    raft::device_vector_view<float, int64_t> delta,
                    raft::device_vector_view<float, int64_t> vl)
{
  RAFT_EXPECTS(delta.extent(0) == n_rows && vl.extent(0) == n_rows,
               "rabitq: delta and vl must hold one element per row");
}

void check_factors(int64_t n_rows,
                   raft::device_matrix_view<float, int64_t, raft::row_major> factors)
{
  RAFT_EXPECTS(factors.extent(0) == n_rows && factors.extent(1) == kNumFactors,
               "rabitq: factors must be a n_rows x 3 matrix");
}

/**
 * Rows per internal transform() batch.
 *
 * Every stage after the input read (pad, rotate, centroid subtraction, quantize)
 * sweeps the batch's `batch x padded_dim` residual scratch, so a batch whose
 * scratch fits in ~3/8 of the L2 cache keeps those passes in L2 instead of DRAM:
 * 1.1-1.7x faster than one unbatched call for fht_kac on an RTX PRO 6000 (128 MB
 * L2). Below 2K rows the fixed per-batch cost outweighs that, and beyond 64K rows
 * (32K for a narrow matmul) nothing more is gained. A matmul rotator wider than
 * 256 is GEMM-bound instead and runs best on large batches. As in
 * IVF-PQ's extend, the batch is then halved until its scratch fits the free
 * workspace, here half of it: the rest covers the smaller temporaries (rotated
 * centroid, padding) and fragmentation. Host input adds `staging_row_bytes` per
 * row for the buffers its rows are copied into.
 */
[[nodiscard]] auto transform_batch_rows(raft::resources const& res,
                                        rabitq::quantizer const& q,
                                        rabitq::rotator const& rot,
                                        int64_t n_rows,
                                        size_t staging_row_bytes) -> int64_t
{
  constexpr int64_t kMinL2Rows        = 2048;
  constexpr int64_t kMaxL2Rows        = 65536;
  constexpr int64_t kMaxL2RowsMatmul  = 32768;
  constexpr int64_t kGemmRows         = 131072;
  constexpr int64_t kMinWorkspaceRows = 128;

  const bool matmul       = rot.kind == rotator_kind::matmul;
  const size_t row_floats = static_cast<size_t>(q.padded_dim);
  int64_t batch           = kGemmRows;
  if (!matmul || q.padded_dim <= 256) {
    const size_t l2 = raft::resource::get_device_properties(res).l2CacheSize;
    const auto fit  = static_cast<int64_t>(l2 * 3 / 8 / (row_floats * sizeof(float)));
    const auto cap  = matmul ? kMaxL2RowsMatmul : kMaxL2Rows;
    batch           = kMinL2Rows;
    while (batch * 2 <= std::min(fit, cap)) {
      batch *= 2;
    }
  }
  // matmul pads into a second batch x padded_dim temporary (pad_and_rotate).
  const size_t row_bytes =
    row_floats * sizeof(float) * (matmul && q.dim != q.padded_dim ? 2 : 1) + staging_row_bytes;
  const size_t budget = raft::resource::get_workspace_free_bytes(res) / 2;
  while (batch > kMinWorkspaceRows && static_cast<size_t>(batch) * row_bytes > budget) {
    batch /= 2;
  }
  return std::min(batch, n_rows);
}

/** A row-major `n_rows x dim` input dataset, on device or on host. */
template <typename Accessor>
using dataset_view =
  raft::mdspan<const float, raft::matrix_extent<int64_t>, raft::row_major, Accessor>;

/** Bytes per row of the device buffers that host input is copied into (see for_each_batch). */
template <typename Accessor>
[[nodiscard]] auto staging_row_bytes(raft::resources const& res, int64_t dim) -> size_t
{
  if constexpr (Accessor::is_device_accessible) {
    return 0;
  } else {
    const bool prefetch = cuvs::spatial::knn::detail::utils::get_prefetch_stream(res).second;
    return (prefetch ? 2 : 1) * static_cast<size_t>(dim) * sizeof(float);
  }
}

/**
 * Calls `quantize_batch(rows, r0, n)` on consecutive `batch`-row slices of
 * `dataset`, `rows` pointing at the slice's `n` rows on device. Device input is
 * read in place. Host input is copied by batch_load_iterator into workspace
 * buffers, as in IVF-PQ's extend: on the main stream, or, when `res` has a
 * stream pool, on a pool stream into two buffers, so that the next batch is
 * copied while the current one is quantized. No host row is read after this
 * returns.
 */
template <typename Accessor, typename QuantizeBatch>
void for_each_batch(raft::resources const& res,
                    dataset_view<Accessor> dataset,
                    int64_t batch,
                    QuantizeBatch&& quantize_batch)
{
  const int64_t n_rows = dataset.extent(0);
  if constexpr (Accessor::is_device_accessible) {
    for (int64_t r0 = 0; r0 < n_rows; r0 += batch) {
      quantize_batch(
        dataset.data_handle() + r0 * dataset.extent(1), r0, std::min(batch, n_rows - r0));
    }
  } else {
    namespace utils              = cuvs::spatial::knn::detail::utils;
    auto [copy_stream, prefetch] = utils::get_prefetch_stream(res);
    // The pool stream is not ordered after the work already queued on res's
    // stream, which may still be writing the host rows (e.g. into pinned memory).
    if (prefetch) { raft::resource::sync_stream(res); }
    utils::batch_load_iterator<dataset_view<Accessor>> batches(
      res,
      dataset,
      static_cast<size_t>(batch),
      copy_stream,
      raft::resource::get_workspace_resource_ref(res),
      prefetch);
    batches.prefetch_next_batch();
    for (auto const& b : batches) {
      quantize_batch(b.data(), static_cast<int64_t>(b.offset()), static_cast<int64_t>(b.size()));
      batches.prefetch_next_batch();
      // The last prefetch_next_batch() returns without waiting for res's stream,
      // and the buffers are freed on the pool stream: wait for the kernels still
      // reading them, as IVF-SQ's extend does.
      if (prefetch) { raft::resource::sync_stream(res); }
    }
  }
}

template <typename CodeT, typename Accessor>
void transform_impl(raft::resources const& res,
                    rabitq::quantizer const& q,
                    rabitq::rotator const& rot,
                    dataset_view<Accessor> dataset,
                    std::optional<raft::device_vector_view<const float, int64_t>> centroid,
                    raft::device_matrix_view<CodeT, int64_t, raft::row_major> codes,
                    raft::device_vector_view<float, int64_t> delta,
                    raft::device_vector_view<float, int64_t> vl)
{
  const int64_t n_rows = dataset.extent(0);
  check_peers(q, rot);
  RAFT_EXPECTS(dataset.extent(1) == q.dim,
               "rabitq::transform: dataset.extent(1) must equal quantizer.dim");
  check_codes(q, n_rows, codes);
  check_delta_vl(n_rows, delta, vl);
  if (centroid.has_value()) {
    RAFT_EXPECTS(centroid->extent(0) == q.dim,
                 "rabitq::transform: centroid must hold quantizer.dim elements");
  }
  if (n_rows == 0) { return; }

  auto stream    = raft::resource::get_cuda_stream(res);
  auto workspace = raft::resource::get_workspace_resource_ref(res);
  const int64_t batch =
    transform_batch_rows(res, q, rot, n_rows, staging_row_bytes<Accessor>(res, q.dim));
  rmm::device_uvector<float> residuals(
    static_cast<size_t>(batch) * static_cast<size_t>(q.padded_dim), stream, workspace);
  rmm::device_uvector<float> rotated_centroid(
    centroid.has_value() ? static_cast<size_t>(q.padded_dim) : 0, stream, workspace);

  // Rows are quantized independently, so batching only bounds the scratch.
  for_each_batch(res, dataset, batch, [&](const float* rows, int64_t r0, int64_t n) {
    detail::quantize_data(res,
                          rows,
                          static_cast<size_t>(n),
                          static_cast<size_t>(q.dim),
                          as_ref(rot),
                          centroid.has_value() ? centroid->data_handle() : nullptr,
                          centroid.has_value() ? rotated_centroid.data() : nullptr,
                          static_cast<size_t>(q.params_quantizer.ex_bits),
                          q.const_scaling_factor,
                          q.params_quantizer.use_fast,
                          codes.data_handle() + r0 * q.padded_dim,
                          delta.data_handle() + r0,
                          vl.data_handle() + r0,
                          residuals.data(),
                          static_cast<int>(q.params_quantizer.delta_mode),
                          static_cast<int>(q.params_quantizer.coarse_samples),
                          static_cast<int>(q.params_quantizer.fine_samples));
  });
}

template <typename CodeT, typename Accessor>
void transform_full_impl(raft::resources const& res,
                         rabitq::quantizer const& q,
                         rabitq::rotator const& rot,
                         dataset_view<Accessor> dataset,
                         std::optional<raft::device_vector_view<const float, int64_t>> centroid,
                         raft::device_matrix_view<CodeT, int64_t, raft::row_major> codes,
                         raft::device_matrix_view<float, int64_t, raft::row_major> factors)
{
  const int64_t n_rows = dataset.extent(0);
  check_peers(q, rot);
  RAFT_EXPECTS(dataset.extent(1) == q.dim,
               "rabitq::transform: dataset.extent(1) must equal quantizer.dim");
  check_codes(q, n_rows, codes);
  check_factors(n_rows, factors);
  if (centroid.has_value()) {
    RAFT_EXPECTS(centroid->extent(0) == q.dim,
                 "rabitq::transform: centroid must hold quantizer.dim elements");
  }
  if (n_rows == 0) { return; }

  auto stream    = raft::resource::get_cuda_stream(res);
  auto workspace = raft::resource::get_workspace_resource_ref(res);
  const int64_t batch =
    transform_batch_rows(res, q, rot, n_rows, staging_row_bytes<Accessor>(res, q.dim));
  rmm::device_uvector<float> residuals(
    static_cast<size_t>(batch) * static_cast<size_t>(q.padded_dim), stream, workspace);
  // The full-factor formula always reads the rotated centroid; the detail layer
  // zero-fills this buffer when no centroid is given.
  rmm::device_uvector<float> rotated_centroid(
    static_cast<size_t>(q.padded_dim), stream, workspace);

  // Rows are quantized independently, so batching only bounds the scratch.
  for_each_batch(res, dataset, batch, [&](const float* rows, int64_t r0, int64_t n) {
    detail::quantize_data_full(res,
                               rows,
                               static_cast<size_t>(n),
                               static_cast<size_t>(q.dim),
                               as_ref(rot),
                               centroid.has_value() ? centroid->data_handle() : nullptr,
                               rotated_centroid.data(),
                               static_cast<size_t>(q.params_quantizer.ex_bits),
                               q.const_scaling_factor,
                               q.params_quantizer.use_fast,
                               codes.data_handle() + r0 * q.padded_dim,
                               factors.data_handle() + r0 * kNumFactors,
                               residuals.data());
  });
}

template <typename CodeT>
void transform_residuals_impl(
  raft::resources const& res,
  rabitq::quantizer const& q,
  raft::device_matrix_view<const float, int64_t, raft::row_major> residuals,
  raft::device_matrix_view<CodeT, int64_t, raft::row_major> codes,
  raft::device_vector_view<float, int64_t> delta,
  raft::device_vector_view<float, int64_t> vl)
{
  const int64_t n_rows = residuals.extent(0);
  check_quantizer(q);
  check_rows(n_rows);
  RAFT_EXPECTS(residuals.extent(1) == q.padded_dim,
               "rabitq::transform_residuals: residuals.extent(1) must equal quantizer.padded_dim");
  check_codes(q, n_rows, codes);
  check_delta_vl(n_rows, delta, vl);
  if (n_rows == 0) { return; }

  detail::quantize_fused_on_residuals(raft::resource::get_cuda_stream(res),
                                      residuals.data_handle(),
                                      static_cast<size_t>(n_rows),
                                      static_cast<size_t>(q.padded_dim),
                                      static_cast<size_t>(q.params_quantizer.ex_bits),
                                      q.const_scaling_factor,
                                      q.params_quantizer.use_fast,
                                      codes.data_handle(),
                                      delta.data_handle(),
                                      vl.data_handle(),
                                      static_cast<int>(q.params_quantizer.delta_mode),
                                      static_cast<int>(q.params_quantizer.coarse_samples),
                                      static_cast<int>(q.params_quantizer.fine_samples));
}

template <typename CodeT>
void transform_residuals_full_impl(
  raft::resources const& res,
  rabitq::quantizer const& q,
  raft::device_matrix_view<const float, int64_t, raft::row_major> residuals,
  std::optional<raft::device_vector_view<const float, int64_t>> rotated_centroid,
  raft::device_matrix_view<CodeT, int64_t, raft::row_major> codes,
  raft::device_matrix_view<float, int64_t, raft::row_major> factors)
{
  const int64_t n_rows = residuals.extent(0);
  check_quantizer(q);
  check_rows(n_rows);
  RAFT_EXPECTS(residuals.extent(1) == q.padded_dim,
               "rabitq::transform_residuals: residuals.extent(1) must equal quantizer.padded_dim");
  check_codes(q, n_rows, codes);
  check_factors(n_rows, factors);
  if (rotated_centroid.has_value()) {
    RAFT_EXPECTS(
      rotated_centroid->extent(0) == q.padded_dim,
      "rabitq::transform_residuals: rotated_centroid must hold quantizer.padded_dim elements");
  }
  if (n_rows == 0) { return; }

  auto stream    = raft::resource::get_cuda_stream(res);
  auto workspace = raft::resource::get_workspace_resource_ref(res);
  // The kernel always reads a centroid; stand in with zeros when there is none.
  rmm::device_uvector<float> zero_centroid(
    rotated_centroid.has_value() ? 0 : static_cast<size_t>(q.padded_dim), stream, workspace);
  const float* d_centroid = nullptr;
  if (rotated_centroid.has_value()) {
    d_centroid = rotated_centroid->data_handle();
  } else {
    RAFT_CUDA_TRY(cudaMemsetAsync(
      zero_centroid.data(), 0, static_cast<size_t>(q.padded_dim) * sizeof(float), stream));
    d_centroid = zero_centroid.data();
  }

  detail::quantize_full_on_residuals(stream,
                                     residuals.data_handle(),
                                     d_centroid,
                                     static_cast<size_t>(n_rows),
                                     static_cast<size_t>(q.padded_dim),
                                     static_cast<size_t>(q.params_quantizer.ex_bits),
                                     q.const_scaling_factor,
                                     q.params_quantizer.use_fast,
                                     codes.data_handle(),
                                     factors.data_handle());
}

}  // namespace

rotator make_rotator(raft::resources const& res, params const& params, int64_t dim)
{
  RAFT_EXPECTS(dim > 0, "rabitq::make_rotator: dim must be positive");
  check_params(params);

  rotator rot{res};
  rot.kind       = params.rotator;
  rot.padded_dim = detail::padded_dim(dim);

  auto stream = raft::resource::get_cuda_stream(res);
  if (rot.kind == rotator_kind::matmul) {
    rot.rotation_matrix = raft::make_device_matrix<float, int64_t, raft::row_major>(
      res, rot.padded_dim, rot.padded_dim);
    detail::init_state_matmul(res, params.seed, rot.padded_dim, rot.rotation_matrix.data_handle());
  } else {
    rot.flip_bits =
      raft::make_device_vector<uint8_t, int64_t>(res, detail::flip_bytes(rot.padded_dim));
    detail::init_state_fht_kac(stream, params.seed, rot.padded_dim, rot.flip_bits.data_handle());
  }
  return rot;
}

// RaBitQ needs no training data: `res` only supplies the stream, the workspace
// memory resource and the device properties used by the scaling-factor
// estimator below (and is unused when `params.use_fast` is false).
quantizer make_quantizer(raft::resources const& res, params const& params, int64_t dim)
{
  RAFT_EXPECTS(dim > 0, "rabitq::make_quantizer: dim must be positive");
  check_params(params);

  quantizer q;
  q.params_quantizer = params;
  q.dim              = dim;
  q.padded_dim       = detail::padded_dim(dim);
  // RaBitQ is data-oblivious: the only thing to "learn" is the shared scaling
  // factor of the fast path, and it is estimated from random probe vectors.
  if (params.use_fast) {
    q.const_scaling_factor =
      detail::get_const_scaling_factor(res,
                                       static_cast<size_t>(q.padded_dim),
                                       static_cast<size_t>(params.ex_bits),
                                       params.seed,
                                       static_cast<int>(params.coarse_samples),
                                       static_cast<int>(params.fine_samples));
  }
  return q;
}

pipeline make_pipeline(raft::resources const& res, params const& params, int64_t dim)
{
  pipeline p{res};
  p.rotator   = make_rotator(res, params, dim);
  p.quantizer = make_quantizer(res, params, dim);
  return p;
}

void rotate(raft::resources const& res,
            rotator const& rotator,
            raft::device_matrix_view<const float, int64_t, raft::row_major> in,
            raft::device_matrix_view<float, int64_t, raft::row_major> out)
{
  check_rotator(rotator);
  const int64_t n_rows = in.extent(0);
  check_rows(n_rows);
  RAFT_EXPECTS(out.extent(0) == n_rows, "rabitq::rotate: in and out must have the same row count");
  RAFT_EXPECTS(in.extent(1) == rotator.padded_dim && out.extent(1) == rotator.padded_dim,
               "rabitq::rotate: in and out must be n_rows x rotator.padded_dim (pad the rows "
               "yourself; rotate() does not pad)");
  auto ref = as_ref(rotator);
  RAFT_EXPECTS(in.data_handle() != out.data_handle() || detail::supports_inplace_rotate(ref),
               "rabitq::rotate: in-place rotation is only supported by rotator_kind::fht_kac");
  if (n_rows == 0) { return; }
  detail::rotate(res, ref, in.data_handle(), out.data_handle(), n_rows);
}

void transform(raft::resources const& res,
               quantizer const& quantizer,
               rotator const& rotator,
               raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
               std::optional<raft::device_vector_view<const float, int64_t>> centroid,
               raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
               raft::device_vector_view<float, int64_t> delta,
               raft::device_vector_view<float, int64_t> vl)
{
  transform_impl(res, quantizer, rotator, dataset, centroid, codes, delta, vl);
}

void transform(raft::resources const& res,
               quantizer const& quantizer,
               rotator const& rotator,
               raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
               std::optional<raft::device_vector_view<const float, int64_t>> centroid,
               raft::device_matrix_view<uint16_t, int64_t, raft::row_major> codes,
               raft::device_vector_view<float, int64_t> delta,
               raft::device_vector_view<float, int64_t> vl)
{
  transform_impl(res, quantizer, rotator, dataset, centroid, codes, delta, vl);
}

void transform(raft::resources const& res,
               quantizer const& quantizer,
               rotator const& rotator,
               raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
               std::optional<raft::device_vector_view<const float, int64_t>> centroid,
               raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
               raft::device_vector_view<float, int64_t> delta,
               raft::device_vector_view<float, int64_t> vl)
{
  transform_impl(res, quantizer, rotator, dataset, centroid, codes, delta, vl);
}

void transform(raft::resources const& res,
               quantizer const& quantizer,
               rotator const& rotator,
               raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
               std::optional<raft::device_vector_view<const float, int64_t>> centroid,
               raft::device_matrix_view<uint16_t, int64_t, raft::row_major> codes,
               raft::device_vector_view<float, int64_t> delta,
               raft::device_vector_view<float, int64_t> vl)
{
  transform_impl(res, quantizer, rotator, dataset, centroid, codes, delta, vl);
}

void transform(raft::resources const& res,
               quantizer const& quantizer,
               rotator const& rotator,
               raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
               std::optional<raft::device_vector_view<const float, int64_t>> centroid,
               raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
               raft::device_matrix_view<float, int64_t, raft::row_major> factors)
{
  transform_full_impl(res, quantizer, rotator, dataset, centroid, codes, factors);
}

void transform(raft::resources const& res,
               quantizer const& quantizer,
               rotator const& rotator,
               raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
               std::optional<raft::device_vector_view<const float, int64_t>> centroid,
               raft::device_matrix_view<uint16_t, int64_t, raft::row_major> codes,
               raft::device_matrix_view<float, int64_t, raft::row_major> factors)
{
  transform_full_impl(res, quantizer, rotator, dataset, centroid, codes, factors);
}

void transform(raft::resources const& res,
               quantizer const& quantizer,
               rotator const& rotator,
               raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
               std::optional<raft::device_vector_view<const float, int64_t>> centroid,
               raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
               raft::device_matrix_view<float, int64_t, raft::row_major> factors)
{
  transform_full_impl(res, quantizer, rotator, dataset, centroid, codes, factors);
}

void transform(raft::resources const& res,
               quantizer const& quantizer,
               rotator const& rotator,
               raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
               std::optional<raft::device_vector_view<const float, int64_t>> centroid,
               raft::device_matrix_view<uint16_t, int64_t, raft::row_major> codes,
               raft::device_matrix_view<float, int64_t, raft::row_major> factors)
{
  transform_full_impl(res, quantizer, rotator, dataset, centroid, codes, factors);
}

void transform_residuals(raft::resources const& res,
                         quantizer const& quantizer,
                         raft::device_matrix_view<const float, int64_t, raft::row_major> residuals,
                         raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
                         raft::device_vector_view<float, int64_t> delta,
                         raft::device_vector_view<float, int64_t> vl)
{
  transform_residuals_impl(res, quantizer, residuals, codes, delta, vl);
}

void transform_residuals(raft::resources const& res,
                         quantizer const& quantizer,
                         raft::device_matrix_view<const float, int64_t, raft::row_major> residuals,
                         raft::device_matrix_view<uint16_t, int64_t, raft::row_major> codes,
                         raft::device_vector_view<float, int64_t> delta,
                         raft::device_vector_view<float, int64_t> vl)
{
  transform_residuals_impl(res, quantizer, residuals, codes, delta, vl);
}

void transform_residuals(
  raft::resources const& res,
  quantizer const& quantizer,
  raft::device_matrix_view<const float, int64_t, raft::row_major> residuals,
  std::optional<raft::device_vector_view<const float, int64_t>> rotated_centroid,
  raft::device_matrix_view<uint8_t, int64_t, raft::row_major> codes,
  raft::device_matrix_view<float, int64_t, raft::row_major> factors)
{
  transform_residuals_full_impl(res, quantizer, residuals, rotated_centroid, codes, factors);
}

void transform_residuals(
  raft::resources const& res,
  quantizer const& quantizer,
  raft::device_matrix_view<const float, int64_t, raft::row_major> residuals,
  std::optional<raft::device_vector_view<const float, int64_t>> rotated_centroid,
  raft::device_matrix_view<uint16_t, int64_t, raft::row_major> codes,
  raft::device_matrix_view<float, int64_t, raft::row_major> factors)
{
  transform_residuals_full_impl(res, quantizer, residuals, rotated_centroid, codes, factors);
}

}  // namespace cuvs::preprocessing::quantize::rabitq
