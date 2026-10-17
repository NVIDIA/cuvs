/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/distance/distance.hpp>
#include <cuvs/preprocessing/quantize/bbq.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/math.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/norm.cuh>
#include <raft/stats/mean.cuh>
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/reduction.cuh>

#include <cstdint>
#include <limits>

namespace cuvs::preprocessing::quantize::bbq::detail {

// The initial interval is set to the minimum MSE grid for each number of bits.
// These starting points are derintervaled from the optimal MSE grid for a uniform distribution.
_RAFT_HOST_DEVICE constexpr float2 minimum_mse_grid(uint32_t bits)
{
  constexpr float MINIMUM_MSE_GRID[8][2] = {{-0.798f, 0.798f},
                                            {-1.493f, 1.493f},
                                            {-2.051f, 2.051f},
                                            {-2.514f, 2.514f},
                                            {-2.916f, 2.916f},
                                            {-3.278f, 3.278f},
                                            {-3.611f, 3.611f},
                                            {-3.922f, 3.922f}};
  return float2{MINIMUM_MSE_GRID[bits - 1][0], MINIMUM_MSE_GRID[bits - 1][1]};
}

/** Lucene stores ||x - c||^2 as the correction for euclidean metrics and x.c for the rest. */
constexpr bool is_euclidean(cuvs::distance::DistanceType metric)
{
  return metric == cuvs::distance::DistanceType::L2Expanded ||
         metric == cuvs::distance::DistanceType::L2SqrtExpanded;
}

template <typename DataT>
_RAFT_HOST_DEVICE DataT clamp(DataT value, DataT min, DataT max)
{
  return raft::min(raft::max(value, min), max);
}

/** Java Math.round, which rounds half up rather than half to even. */
template <typename MathT>
__device__ __forceinline__ int lucene_round(MathT x)
{
  return static_cast<int>(::floor(x + MathT(0.5)));
}

/** Everything the interval search needs from a row. Warp-uniform on return. */
struct row_stats {
  /** Bounds of the centered row, which the starting interval is clamped to. */
  float min;
  float max;
  /** ||x - c||^2, the norm the anisotropic term of the loss is relatintervale to. */
  float norm2;
  /** ||x||^2 in the original, un-centered space. */
  float row_norm;
  /** x.c, the additional correction of the non-euclidean metrics. */
  float centroid_dot;
  float mean;
  float std;
};

template <typename MathT>
__device__ inline row_stats compute_row_stats(const float* row_ptr,
                                              const float* centroid,
                                              uint32_t dim,
                                              bool euclidean)
{
  constexpr float kInf = std::numeric_limits<float>::infinity();
  float min_value      = kInf;
  float max_value      = -kInf;
  MathT sum            = MathT(0.0);
  MathT sum_sq         = MathT(0.0);
  MathT row_norm       = MathT(0.0);
  MathT centroid_dot   = MathT(0.0);
  for (uint32_t d = raft::laneId(); d < dim; d += raft::WarpSize) {
    const float x = row_ptr[d];
    const float v = x - centroid[d];
    min_value     = raft::min(min_value, v);
    max_value     = raft::max(max_value, v);
    sum += v;
    sum_sq += static_cast<MathT>(v) * v;
    row_norm += static_cast<MathT>(x) * x;
    if (!euclidean) { centroid_dot += static_cast<MathT>(x) * centroid[d]; }
  }
  min_value    = raft::warpReduce(min_value, raft::min_op{});
  max_value    = raft::warpReduce(max_value, raft::max_op{});
  sum          = raft::warpReduce(sum, raft::add_op{});
  sum_sq       = raft::warpReduce(sum_sq, raft::add_op{});
  row_norm     = raft::warpReduce(row_norm, raft::add_op{});
  centroid_dot = raft::warpReduce(centroid_dot, raft::add_op{});

  const MathT mean     = static_cast<MathT>(sum / dim);
  const MathT variance = raft::max(static_cast<MathT>(sum_sq / dim - mean * mean), MathT(0.0));
  return row_stats{min_value,
                   max_value,
                   static_cast<float>(sum_sq),
                   static_cast<float>(row_norm),
                   static_cast<float>(centroid_dot),
                   static_cast<float>(mean),
                   static_cast<float>(raft::sqrt(variance))};
}

/**
 * Loss of quantizing the row onto @p points levels spread over @p interval: the squared error, plus
 * the error along the row itself weighted by `1 - lambda`, which is what keeps the reconstruction's
 * inner product with the original unbiased.
 */
template <typename MathT>
__device__ inline float interval_loss(const float* row_ptr,
                                      const float* centroid,
                                      uint32_t dim,
                                      float2 interval,
                                      int points,
                                      float norm2,
                                      float lambda)
{
  const float lower    = interval.x;
  const float upper    = interval.y;
  const MathT step     = static_cast<MathT>((upper - lower) / (points - MathT(1.0f)));
  const MathT step_inv = MathT(1.0f) / step;
  MathT xe             = 0.0f;
  MathT e              = 0.0f;

  if (dim % 2 == 0) {
    for (uint32_t d = raft::laneId() * 2; d < dim; d += raft::WarpSize * 2) {
      raft::TxN_t<float, 2> xi;
      xi.load(row_ptr, d);
      xi.val.data[0] -= centroid[d];
      xi.val.data[1] -= centroid[d + 1];
      const MathT xiq1 =
        lower +
        step * lucene_round(static_cast<MathT>((clamp(xi.val.data[0], lower, upper) - lower)) *
                            step_inv);
      const MathT xiq2 =
        lower +
        step * lucene_round(static_cast<MathT>((clamp(xi.val.data[1], lower, upper) - lower)) *
                            step_inv);
      xe += xi.val.data[0] * (xi.val.data[0] - xiq1) + xi.val.data[1] * (xi.val.data[1] - xiq2);
      e += (xi.val.data[0] - xiq1) * (xi.val.data[0] - xiq1) +
           (xi.val.data[1] - xiq2) * (xi.val.data[1] - xiq2);
    }
  } else {
    for (uint32_t d = raft::laneId(); d < dim; d += raft::WarpSize) {
      const float xi = row_ptr[d] - centroid[d];
      const MathT xiq =
        lower +
        step * lucene_round(static_cast<MathT>((clamp(xi, lower, upper) - lower)) * step_inv);
      xe += xi * (xi - xiq);
      e += (xi - xiq) * (xi - xiq);
    }
  }
  xe = raft::warpReduce(xe, raft::add_op{});
  e  = raft::warpReduce(e, raft::add_op{});
  return (MathT(1.0f) - lambda) * xe * xe / norm2 + lambda * e;
}

/**
 * Coordinate descent on the interval: each step solves the 2x2 normal equations for the endpoints
 * with the code assignments held fixed, and stops as soon as a step stops paying for itself.
 */
template <typename MathT>
__device__ inline float2 optimize_intervals(float2 interval,
                                            const float* row_ptr,
                                            const float* centroid,
                                            uint32_t dim,
                                            float norm2,
                                            int points,
                                            float lambda,
                                            int iters)
{
  float initial_loss =
    interval_loss<MathT>(row_ptr, centroid, dim, interval, points, norm2, lambda);
  const float scale = (1.0f - lambda) / norm2;
  if (!::isfinite(scale)) { return interval; }
  for (int i = 0; i < iters; ++i) {
    const float lower    = interval.x;
    const float upper    = interval.y;
    const float step_inv = (points - 1.0f) / (upper - lower);
    MathT daa = 0.0f, dab = 0.0f, dbb = 0.0f, dax = 0.0f, dbx = 0.0f;
    auto accumulate = [&](float xi) {
      const MathT clamped = (clamp(xi, lower, upper) - lower) * step_inv;
      const MathT k       = lucene_round<MathT>(clamped);
      const MathT s       = k / (points - 1);
      daa += (MathT(1.0f) - s) * (MathT(1.0f) - s);
      dab += (MathT(1.0f) - s) * s;
      dbb += s * s;
      dax += xi * (MathT(1.0f) - s);
      dbx += xi * s;
    };
    if (dim % 4 == 0) {
      for (uint32_t d = raft::laneId() * 4; d < dim; d += raft::WarpSize * 4) {
        raft::TxN_t<float, 4> xi;
        xi.load(row_ptr, d);
        accumulate(xi.val.data[0] - centroid[d]);
        accumulate(xi.val.data[1] - centroid[d + 1]);
        accumulate(xi.val.data[2] - centroid[d + 2]);
        accumulate(xi.val.data[3] - centroid[d + 3]);
      }
    } else {
      for (uint32_t d = raft::laneId(); d < dim; d += raft::WarpSize) {
        accumulate(row_ptr[d] - centroid[d]);
      }
    }
    daa = raft::warpReduce(daa, raft::add_op{});
    dab = raft::warpReduce(dab, raft::add_op{});
    dbb = raft::warpReduce(dbb, raft::add_op{});
    dax = raft::warpReduce(dax, raft::add_op{});
    dbx = raft::warpReduce(dbx, raft::add_op{});

    const MathT m0  = scale * dax * dax + lambda * daa;
    const MathT m1  = scale * dax * dbx + lambda * dab;
    const MathT m2  = scale * dbx * dbx + lambda * dbb;
    const MathT det = m0 * m2 - m1 * m1;
    if (det == 0) { return interval; }
    const float2 candidate = {static_cast<float>((m2 * dax - m1 * dbx) / det),
                              static_cast<float>((m0 * dbx - m1 * dax) / det)};
    if (raft::abs(interval.x - candidate.x) < 1e-8f &&
        raft::abs(interval.y - candidate.y) < 1e-8f) {
      return interval;
    }
    const float new_loss =
      interval_loss<MathT>(row_ptr, centroid, dim, candidate, points, norm2, lambda);
    if (new_loss > initial_loss) { return interval; }
    interval     = candidate;
    initial_loss = new_loss;
  }
  return interval;
}

/**
 * Writes the codes of dimensions [@p base, base + WarpSize) of one row, one code per lane, in
 * @p Layout. Lanes past the end of the row must pass a zero code; the whole warp has to take part
 * because the packed layouts span lanes.
 *
 * Mirrors the host reference's pack_codes, which in turn follows Lucene packAsBinary, packNibbles,
 * transposeDibit and transposeHalfByte.
 */
template <bbq_code_layout Layout>
__device__ inline void write_code_group(uint8_t* code_row,
                                        uint32_t dim,
                                        uint32_t base,
                                        uint32_t code)
{
  constexpr uint32_t bits = get_bit_width(Layout);
  const uint32_t lane     = raft::laneId();
  const uint32_t d        = base + lane;

  if constexpr (Layout == bbq_code_layout::packed_8b || Layout == bbq_code_layout::packed_7b) {
    if (d < dim) { code_row[d] = static_cast<uint8_t>(code); }
  } else if constexpr (Layout == bbq_code_layout::packed_4b) {
    // Contiguous pairing: dims 2k and 2k+1 share byte k, high nibble first. This is not Lucene's
    // packNibbles (which pairs dim i with dim dim/2 + i) -- see the host reference for why.
    // An odd dim drops its last component, whose byte the caller has already zeroed.
    const uint32_t low = __shfl_down_sync(0xffffffffu, code, 1);
    if (lane % 2 == 0 && d + 1 < dim) {
      code_row[d / 2] = static_cast<uint8_t>((code << 4) | (low & 0x0fu));
    }
  } else {
    // One bit plane per code bit, least significant plane first, each plane holding dimension d in
    // bit 7 - d % 8 of its (d / 8)-th byte. packed_1b is the single-plane case of this.
    const uint32_t stripe = (dim + 7) / 8;
    for (uint32_t bit = 0; bit < bits; ++bit) {
      // The ballot puts lane 0 in bit 0, so reversing it lands dimension `base` in the most
      // significant bit of the group's first byte, where Lucene's bit order wants it.
      const uint32_t plane = __brev(__ballot_sync(0xffffffffu, ((code >> bit) & 1u) != 0u));
      // The group covers 4 bytes, but a short tail can leave some of them past the plane.
      if (lane < sizeof(uint32_t) && base / 8 + lane < stripe) {
        code_row[bit * stripe + base / 8 + lane] = static_cast<uint8_t>(plane >> (24 - 8 * lane));
      }
    }
  }
}

/** The per-row arrays to fill a `bbq_dataset_owning_storage`, as raw pointers. */
struct quantizer_output {
  uint8_t* codes;
  uint32_t code_row_length;
  float* lower_intervals;
  float* upper_intervals;
  float* additional_corrections;
  int32_t* quantized_component_sums;
  float* row_norm;
};

/** One warp per row: find the row's interval, then quantize and pack the row against it. */
template <bbq_code_layout Layout, typename MathT>
__global__ void quantize_kernel(
  raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
  const float* centroid,
  bool stage_centroid,
  bool euclidean,
  float lambda,
  int iters,
  quantizer_output out)
{
  constexpr uint32_t bits = get_bit_width(Layout);
  constexpr int points    = 1 << bits;
  constexpr float n_steps = static_cast<float>(points - 1);

  const auto dim = static_cast<uint32_t>(dataset.extent(1));

  // Every warp in the block re-reads the whole centroid on each of the dozen or so passes a row
  // takes, so it is staged once per block. `stage_centroid` is false when the centroid does not fit
  // in shared memory, and the global copy is then read in place. The staging has to happen before
  // the bounds check below, so that the barrier sees the whole block.
  extern __shared__ float staged_centroid[];
  if (stage_centroid) {
    for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
      staged_centroid[d] = centroid[d];
    }
    __syncthreads();
    centroid = staged_centroid;
  }

  const int64_t row_id =
    (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / raft::WarpSize;
  // A warp shares its row, so the whole warp leaves together and the reductions below stay whole.
  if (row_id >= dataset.extent(0)) { return; }

  const auto* row_ptr = &dataset(row_id, 0);
  const auto stats    = compute_row_stats<MathT>(row_ptr, centroid, dim, euclidean);
  const auto grid     = minimum_mse_grid(bits);
  float2 interval     = {
    clamp(static_cast<float>(grid.x * stats.std + stats.mean), stats.min, stats.max),
    clamp(static_cast<float>(grid.y * stats.std + stats.mean), stats.min, stats.max)};
  interval =
    optimize_intervals<MathT>(interval, row_ptr, centroid, dim, stats.norm2, points, lambda, iters);

  const float lower = interval.x;
  const float upper = interval.y;
  const float step  = (upper - lower) / n_steps;
  uint8_t* code_row =
    out.codes + static_cast<size_t>(row_id) * static_cast<size_t>(out.code_row_length);
  int component_sum = 0;
  // Uniform over the warp so that the whole warp reaches the packing of every group.
  for (uint32_t base = 0; base < dim; base += raft::WarpSize) {
    const uint32_t d = base + raft::laneId();
    uint32_t code    = 0;
    if (d < dim) {
      const float xi = clamp(row_ptr[d] - centroid[d], lower, upper);
      code           = static_cast<uint32_t>(lucene_round((xi - lower) / step));
      component_sum += static_cast<int>(code);
    }
    write_code_group<Layout>(code_row, dim, base, code);
  }
  component_sum = raft::warpReduce(component_sum, raft::add_op{});

  if (raft::laneId() == 0) {
    out.lower_intervals[row_id]          = interval.x;
    out.upper_intervals[row_id]          = interval.y;
    out.additional_corrections[row_id]   = euclidean ? stats.norm2 : stats.centroid_dot;
    out.quantized_component_sums[row_id] = component_sum;
    out.row_norm[row_id]                 = stats.row_norm;
  }
}

constexpr int kQuantizeBlockSize        = 128;
constexpr int64_t kQuantizeRowsPerBlock = kQuantizeBlockSize / raft::WarpSize;

template <bbq_code_layout Layout>
void launch_quantize_kernel(raft::resources const& res,
                            const cuvs::preprocessing::quantize::bbq::params& params,
                            raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
                            const float* centroid,
                            quantizer_output out)
{
  const auto n_blocks =
    static_cast<uint32_t>(raft::ceildiv<int64_t>(dataset.extent(0), kQuantizeRowsPerBlock));
  const size_t centroid_bytes = dataset.extent(1) * sizeof(float);
  const bool stage_centroid =
    centroid_bytes <= raft::resource::get_device_properties(res).sharedMemPerBlock;
  quantize_kernel<Layout, float>
    <<<n_blocks,
       kQuantizeBlockSize,
       stage_centroid ? centroid_bytes : 0,
       raft::resource::get_cuda_stream(res).get()>>>(dataset,
                                                     centroid,
                                                     stage_centroid,
                                                     is_euclidean(params.metric),
                                                     params.lambda,
                                                     params.iters,
                                                     out);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

inline void quantize_rows(raft::resources const& res,
                          const cuvs::preprocessing::quantize::bbq::params& params,
                          bbq_code_layout layout,
                          raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
                          const float* centroid,
                          quantizer_output out)
{
  switch (layout) {
    case bbq_code_layout::packed_1b:
      return launch_quantize_kernel<bbq_code_layout::packed_1b>(
        res, params, dataset, centroid, out);
    case bbq_code_layout::transposed_2b:
      return launch_quantize_kernel<bbq_code_layout::transposed_2b>(
        res, params, dataset, centroid, out);
    case bbq_code_layout::transposed_4b:
      return launch_quantize_kernel<bbq_code_layout::transposed_4b>(
        res, params, dataset, centroid, out);
    case bbq_code_layout::packed_4b:
      return launch_quantize_kernel<bbq_code_layout::packed_4b>(
        res, params, dataset, centroid, out);
    case bbq_code_layout::packed_7b:
      return launch_quantize_kernel<bbq_code_layout::packed_7b>(
        res, params, dataset, centroid, out);
    case bbq_code_layout::packed_8b:
      return launch_quantize_kernel<bbq_code_layout::packed_8b>(
        res, params, dataset, centroid, out);
  }
  RAFT_FAIL("Unknown BBQ code layout %d", static_cast<int>(layout));
}

/**
 * Computes the single centroid shared by every row, as Lucene and Elasticsearch do.
 */
inline auto build(raft::resources const& res,
                  raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> quantizer<float>
{
  using DataT       = float;
  const auto n_rows = dataset.extent(0);
  const auto dim    = dataset.extent(1);
  RAFT_EXPECTS(n_rows > 0 && dim > 0, "BBQ build needs a non-empty dataset");

  quantizer<DataT> quant{res, static_cast<uint32_t>(dim)};
  raft::stats::mean(res, dataset, quant.centroid.view());
  auto centroid_norm_sq = raft::make_device_vector<float, int64_t>(res, 1);
  raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
    res,
    raft::make_device_matrix_view<const DataT, int64_t>(quant.centroid.data_handle(), 1, dim),
    centroid_norm_sq.view());
  const auto stream = raft::resource::get_cuda_stream(res).get();
  raft::copy(&quant.centroid_norm_sq, centroid_norm_sq.data_handle(), 1, stream);
  raft::resource::sync_stream(res);
  return quant;
}

/**
 * Quantizes @p dataset against the centroid of @p quant in the document layout, and again in the
 * query layout when it differs, and returns the codes together with the per-row corrections search
 * needs.
 */
inline auto transform(raft::resources const& res,
                      const cuvs::preprocessing::quantize::bbq::params& params,
                      quantizer_view<float> quant,
                      raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::device_bbq_dataset<float, int64_t>
{
  constexpr int64_t kMaxQuantizeRows =
    kQuantizeBlockSize * int64_t{std::numeric_limits<int32_t>::max()} / raft::WarpSize;
  const auto n_rows = dataset.extent(0);
  const auto dim    = dataset.extent(1);
  RAFT_EXPECTS(n_rows > 0 && dim > 0, "BBQ transform needs a non-empty dataset");
  RAFT_EXPECTS(dim == static_cast<int64_t>(quant.dim()),
               "BBQ transform: the dataset has %lld dimensions, the quantizer %u",
               static_cast<long long>(dim),
               quant.dim());
  RAFT_EXPECTS(n_rows <= kMaxQuantizeRows,
               "BBQ transform: at most %lld rows are supported, got %lld",
               static_cast<long long>(kMaxQuantizeRows),
               static_cast<long long>(n_rows));

  const auto stream = raft::resource::get_cuda_stream(res).get();
  quantizer<float> dataset_quantizer{res, quant.dim()};
  raft::copy(
    dataset_quantizer.centroid.data_handle(), quant.centroid.data_handle(), quant.dim(), stream);
  dataset_quantizer.centroid_norm_sq = quant.centroid_norm_sq;
  cuvs::neighbors::device_bbq_dataset<float, int64_t> encoded{std::move(dataset_quantizer)};

  auto encode = [&](bbq_code_layout layout) {
    bbq_dataset_owning_storage<float, int64_t> storage{
      res, encoded.quantizer.view(), n_rows, layout, params.metric};
    RAFT_CUDA_TRY(cudaMemsetAsync(storage.codes.data_handle(), 0, storage.codes.size(), stream));
    quantize_rows(res,
                  params,
                  layout,
                  dataset,
                  encoded.quantizer.centroid.data_handle(),
                  quantizer_output{storage.codes.data_handle(),
                                   storage.encoded_row_length(),
                                   storage.lower_intervals.data_handle(),
                                   storage.upper_intervals.data_handle(),
                                   storage.additional_corrections.data_handle(),
                                   storage.quantized_component_sums.data_handle(),
                                   storage.row_norm.data_handle()});
    helpers::resolve_dequant_factors(
      res,
      storage.dequant_delta.view(),
      storage.dequant_sum_delta.view(),
      raft::make_const_mdspan(storage.lower_intervals.view()),
      raft::make_const_mdspan(storage.upper_intervals.view()),
      raft::make_const_mdspan(storage.quantized_component_sums.view()),
      layout);
    encoded.add_storage(std::move(storage));
  };
  encode(params.doc_layout);
  if (params.query_layout != params.doc_layout) { encode(params.query_layout); }
  return encoded;
}
}  // namespace cuvs::preprocessing::quantize::bbq::detail
