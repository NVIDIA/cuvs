/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./detail/rabitq/pipeline.cuh"
#include "./detail/rabitq/quantize.cuh"
#include "./detail/rabitq/rescale_search.cuh"
#include "./detail/rabitq/rotator.cuh"

#include <cuvs/preprocessing/quantize/rabitq.hpp>

#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

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
               "rabitq: rotator is not initialized (padded_dim == 0); use init_rotator()");
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
  RAFT_EXPECTS(q.dim > 0, "rabitq: quantizer is not initialized (dim == 0); use init_quantizer()");
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

template <typename CodeT>
void transform_impl(raft::resources const& res,
                    rabitq::quantizer const& q,
                    rabitq::rotator const& rot,
                    raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
                    std::optional<raft::device_vector_view<const float, int64_t>> centroid,
                    raft::device_matrix_view<CodeT, int64_t, raft::row_major> codes,
                    raft::device_vector_view<float, int64_t> delta,
                    raft::device_vector_view<float, int64_t> vl)
{
  const int64_t n_rows = dataset.extent(0);
  check_peers(q, rot);
  check_rows(n_rows);
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
  rmm::device_uvector<float> residuals(
    static_cast<size_t>(n_rows) * static_cast<size_t>(q.padded_dim), stream, workspace);
  rmm::device_uvector<float> rotated_centroid(
    centroid.has_value() ? static_cast<size_t>(q.padded_dim) : 0, stream, workspace);

  detail::quantize_data(res,
                        dataset.data_handle(),
                        static_cast<size_t>(n_rows),
                        static_cast<size_t>(q.dim),
                        as_ref(rot),
                        centroid.has_value() ? centroid->data_handle() : nullptr,
                        centroid.has_value() ? rotated_centroid.data() : nullptr,
                        static_cast<size_t>(q.params_quantizer.ex_bits),
                        q.const_scaling_factor,
                        q.params_quantizer.use_fast,
                        codes.data_handle(),
                        delta.data_handle(),
                        vl.data_handle(),
                        residuals.data(),
                        static_cast<int>(q.params_quantizer.delta_mode),
                        static_cast<int>(q.params_quantizer.coarse_samples),
                        static_cast<int>(q.params_quantizer.fine_samples));
}

template <typename CodeT>
void transform_full_impl(raft::resources const& res,
                         rabitq::quantizer const& q,
                         rabitq::rotator const& rot,
                         raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
                         std::optional<raft::device_vector_view<const float, int64_t>> centroid,
                         raft::device_matrix_view<CodeT, int64_t, raft::row_major> codes,
                         raft::device_matrix_view<float, int64_t, raft::row_major> factors)
{
  const int64_t n_rows = dataset.extent(0);
  check_peers(q, rot);
  check_rows(n_rows);
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
  rmm::device_uvector<float> residuals(
    static_cast<size_t>(n_rows) * static_cast<size_t>(q.padded_dim), stream, workspace);
  // The full-factor formula always reads the rotated centroid; the detail layer
  // zero-fills this buffer when no centroid is given.
  rmm::device_uvector<float> rotated_centroid(
    static_cast<size_t>(q.padded_dim), stream, workspace);

  detail::quantize_data_full(res,
                             dataset.data_handle(),
                             static_cast<size_t>(n_rows),
                             static_cast<size_t>(q.dim),
                             as_ref(rot),
                             centroid.has_value() ? centroid->data_handle() : nullptr,
                             rotated_centroid.data(),
                             static_cast<size_t>(q.params_quantizer.ex_bits),
                             q.const_scaling_factor,
                             q.params_quantizer.use_fast,
                             codes.data_handle(),
                             factors.data_handle(),
                             residuals.data());
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

rotator init_rotator(raft::resources const& res, params const& params, int64_t dim)
{
  RAFT_EXPECTS(dim > 0, "rabitq::init_rotator: dim must be positive");
  check_params(params);

  rotator rot{res};
  rot.kind       = params.rotator;
  rot.padded_dim = detail::padded_dim(dim);

  auto stream = raft::resource::get_cuda_stream(res);
  if (rot.kind == rotator_kind::matmul) {
    rot.rotation_matrix = raft::make_device_matrix<float, int64_t, raft::row_major>(
      res, rot.padded_dim, rot.padded_dim);
    detail::init_state_matmul(
      stream, params.seed, rot.padded_dim, rot.rotation_matrix.data_handle());
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
quantizer init_quantizer(raft::resources const& res, params const& params, int64_t dim)
{
  RAFT_EXPECTS(dim > 0, "rabitq::init_quantizer: dim must be positive");
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

pipeline init_pipeline(raft::resources const& res, params const& params, int64_t dim)
{
  pipeline p{res};
  p.rotator   = init_rotator(res, params, dim);
  p.quantizer = init_quantizer(res, params, dim);
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
