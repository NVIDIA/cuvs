/*
 * SPDX-FileCopyrightText: Copyright the Apache Software Foundation (ASF)
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/distance/distance.hpp>
#include <cuvs/preprocessing/quantize/bbq.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/**
 * Host-side reference implementation of Lucene's OptimizedScalarQuantizer, used to feed the
 * BBQ NN-Descent build path. This is prototype scaffolding: it is a slow CPU quantizer that
 * exists so the compressed build can be exercised end to end, and it is expected to be replaced
 * by a GPU quantizer. It is shared between the unit tests and the ann-bench CAGRA wrapper so the
 * two can never disagree about the code format; libcuvs itself only ever sees the uploaded codes.
 */
namespace cuvs_internal::bbq {

using cuvs::preprocessing::quantize::bbq::bbq_code_layout;
using cuvs::preprocessing::quantize::bbq::get_bit_width;
using cuvs::preprocessing::quantize::bbq::get_encoded_row_length;

/**
 * Host-resident mirror of the arrays in a BBQ quantizer. The library quantizer is device-only,
 * so the CPU reference implementation and its on-disk cache need their own staging type; it is
 * uploaded by copy_bbq_owning_storage_host_to_device below.
 */
struct host_quantizer_storage {
  raft::host_matrix<uint8_t, int64_t> codes;
  raft::host_vector<float, int64_t> lower_intervals;
  raft::host_vector<float, int64_t> upper_intervals;
  raft::host_vector<float, int64_t> additional_corrections;
  raft::host_vector<int32_t, int64_t> quantized_component_sums;
  raft::host_vector<float, int64_t> centroid;
  raft::host_vector<float, int64_t> dequant_delta;
  raft::host_vector<float, int64_t> dequant_sum_delta;
  raft::host_vector<float, int64_t> row_norm;
  bbq_code_layout layout{bbq_code_layout::packed_1b};
  cuvs::distance::DistanceType metric{cuvs::distance::DistanceType::L2Expanded};
  float centroid_norm_sq{};
};

/**
 * Fills the derived per-row dequantization factors. The device quantizer requires them up front
 * (deriving them there would need a stream to allocate with), so they are computed here, where
 * the intervals are already host-resident.
 */
inline void derive_dequant_factors(host_quantizer_storage& q)
{
  const float scale = 1.0f / static_cast<float>((uint32_t{1} << get_bit_width(q.layout)) - 1);
  for (int64_t i = 0; i < q.lower_intervals.extent(0); ++i) {
    const float delta      = (q.upper_intervals(i) - q.lower_intervals(i)) * scale;
    q.dequant_delta(i)     = delta;
    q.dequant_sum_delta(i) = delta * static_cast<float>(q.quantized_component_sums(i));
  }
}

constexpr float kMinimumMseGrid[8][2] = {{-0.798f, 0.798f},
                                         {-1.493f, 1.493f},
                                         {-2.051f, 2.051f},
                                         {-2.514f, 2.514f},
                                         {-2.916f, 2.916f},
                                         {-3.278f, 3.278f},
                                         {-3.611f, 3.611f},
                                         {-3.922f, 3.922f}};

constexpr float kDefaultLambda = 0.1f;
constexpr int kDefaultIters    = 5;

inline long round(double x) { return static_cast<long>(std::floor(x + 0.5)); }

inline double clamp(double x, double a, double b) { return std::min(std::max(x, a), b); }

inline double loss(
  const std::vector<float>& vector, const float interval[2], int points, float norm2, float lambda)
{
  const double a        = interval[0];
  const double b        = interval[1];
  const double step     = (b - a) / (points - 1.0);
  const double step_inv = 1.0 / step;
  double xe             = 0.0;
  double e              = 0.0;
  for (double xi : vector) {
    const double xiq = a + step * static_cast<double>(round((clamp(xi, a, b) - a) * step_inv));
    xe += xi * (xi - xiq);
    e += (xi - xiq) * (xi - xiq);
  }
  return (1.0 - lambda) * xe * xe / norm2 + lambda * e;
}

inline void optimize_intervals(float interval[2],
                               const std::vector<float>& vector,
                               float norm2,
                               int points,
                               float lambda = kDefaultLambda,
                               int iters    = kDefaultIters)
{
  double initial_loss = loss(vector, interval, points, norm2, lambda);
  const float scale   = (1.0f - lambda) / norm2;
  if (!std::isfinite(scale)) { return; }
  for (int i = 0; i < iters; ++i) {
    const float a        = interval[0];
    const float b        = interval[1];
    const float step_inv = (points - 1.0f) / (b - a);
    double daa = 0.0, dab = 0.0, dbb = 0.0, dax = 0.0, dbx = 0.0;
    for (float xi : vector) {
      const float k =
        static_cast<float>(round(static_cast<float>((clamp(xi, a, b) - a) * step_inv)));
      const float s = k / (points - 1);
      daa += (1.0 - s) * (1.0 - s);
      dab += (1.0 - s) * s;
      dbb += s * s;
      dax += xi * (1.0 - s);
      dbx += xi * s;
    }
    const double m0  = scale * dax * dax + lambda * daa;
    const double m1  = scale * dax * dbx + lambda * dab;
    const double m2  = scale * dbx * dbx + lambda * dbb;
    const double det = m0 * m2 - m1 * m1;
    if (det == 0) { return; }
    const float a_opt = static_cast<float>((m2 * dax - m1 * dbx) / det);
    const float b_opt = static_cast<float>((m0 * dbx - m1 * dax) / det);
    if (std::abs(interval[0] - a_opt) < 1e-8 && std::abs(interval[1] - b_opt) < 1e-8) { return; }
    float new_interval[2] = {a_opt, b_opt};
    const double new_loss = loss(vector, new_interval, points, norm2, lambda);
    if (new_loss > initial_loss) { return; }
    interval[0]  = a_opt;
    interval[1]  = b_opt;
    initial_loss = new_loss;
  }
}

struct row_result {
  float lower_interval;
  float upper_interval;
  float additional_correction;
  int32_t quantized_component_sum;
};

inline row_result scalar_quantize(std::vector<float>& vector,
                                  std::vector<uint8_t>& destination,
                                  uint8_t bits,
                                  const float* centroid,
                                  bool euclidean)
{
  const int n        = static_cast<int>(vector.size());
  const int points   = 1 << bits;
  double vec_mean    = 0.0;
  double vec_var     = 0.0;
  float norm2        = 0.0f;
  float centroid_dot = 0.0f;
  float min          = FLT_MAX;
  float max          = -FLT_MAX;
  for (int i = 0; i < n; ++i) {
    if (!euclidean) { centroid_dot += vector[i] * centroid[i]; }
    vector[i] = vector[i] - centroid[i];
    min       = std::min(min, vector[i]);
    max       = std::max(max, vector[i]);
    norm2 += vector[i] * vector[i];
    const double delta = vector[i] - vec_mean;
    vec_mean += delta / (i + 1);
    vec_var += delta * (vector[i] - vec_mean);
  }
  vec_var /= n;
  const double vec_std = std::sqrt(vec_var);

  float interval[2];
  interval[0] =
    static_cast<float>(clamp(kMinimumMseGrid[bits - 1][0] * vec_std + vec_mean, min, max));
  interval[1] =
    static_cast<float>(clamp(kMinimumMseGrid[bits - 1][1] * vec_std + vec_mean, min, max));
  optimize_intervals(interval, vector, norm2, points);

  const float n_steps = static_cast<float>((1 << bits) - 1);
  const float a       = interval[0];
  const float b       = interval[1];
  const float step    = (b - a) / n_steps;
  int sum_query       = 0;
  for (int h = 0; h < n; ++h) {
    const float xi       = static_cast<float>(clamp(vector[h], a, b));
    const int assignment = static_cast<int>(round((xi - a) / step));
    sum_query += assignment;
    destination[h] = static_cast<uint8_t>(assignment);
  }
  return row_result{interval[0], interval[1], euclidean ? norm2 : centroid_dot, sum_query};
}

// Packs one-byte-per-component codes into packed_1b / transposed_2b /
// packed_4b / transposed_4b (or leaves unpacked). Matches Lucene packAsBinary,
// packNibbles, transposeDibit, transposeHalfByte.
inline std::vector<uint8_t> pack_codes(const std::vector<uint8_t>& unpacked,
                                       size_t n_rows,
                                       size_t dim,
                                       bbq_code_layout layout)
{
  const uint32_t bits     = get_bit_width(layout);
  const size_t row_length = get_encoded_row_length(static_cast<uint32_t>(dim), layout);
  if (layout == bbq_code_layout::packed_8b || layout == bbq_code_layout::packed_7b) {
    return unpacked;
  }

  std::vector<uint8_t> packed(n_rows * row_length, 0);
#pragma omp parallel for
  for (int64_t row = 0; row < static_cast<int64_t>(n_rows); ++row) {
    auto* output      = packed.data() + static_cast<size_t>(row) * row_length;
    const auto* input = unpacked.data() + static_cast<size_t>(row) * dim;
    if (layout == bbq_code_layout::packed_4b) {
      // Contiguous: dims 2k / 2k+1 share byte k. NOT Lucene packNibbles, which pairs dim i with
      // dim dim/2 + i. A self-join is position-agnostic so either works there, but an asymmetric
      // pair (packed_1b or transposed_2b document promoted to 4-bit width against this query)
      // needs dimension k of both operands in the same slot -- halves-pairing silently
      // multiplies mismatched dimensions and costs recall.
      const size_t pairs = dim / 2;
      for (size_t i = 0; i < pairs; ++i) {
        output[i] = static_cast<uint8_t>((input[2 * i] << 4) | (input[2 * i + 1] & 0x0f));
      }
      continue;
    }
    for (size_t d = 0; d < dim; ++d) {
      const uint8_t code = input[d];
      if (layout == bbq_code_layout::packed_1b) {
        for (uint32_t bit = 0; bit < bits; ++bit) {
          const size_t position = d * bits + bit;
          output[position / 8] |=
            static_cast<uint8_t>(((code >> (bits - 1 - bit)) & 1u) << (7 - position % 8));
        }
      } else {
        // transposed_2b / transposed_4b bit-planes (LSB plane first)
        const size_t stripe = (dim + 7) / 8;
        for (uint32_t bit = 0; bit < bits; ++bit) {
          output[bit * stripe + d / 8] |= static_cast<uint8_t>(((code >> bit) & 1u) << (7 - d % 8));
        }
      }
    }
  }
  return packed;
}

inline host_quantizer_storage quantize(const float* data,
                                       int64_t n_rows,
                                       int64_t dim,
                                       cuvs::distance::DistanceType metric,
                                       bbq_code_layout layout = bbq_code_layout::packed_8b)
{
  const auto bits      = static_cast<uint8_t>(get_bit_width(layout));
  const bool euclidean = metric == cuvs::distance::DistanceType::L2Expanded ||
                         metric == cuvs::distance::DistanceType::L2SqrtExpanded;

  auto centroid          = raft::make_host_vector<float, int64_t>(dim);
  float centroid_norm_sq = 0.0f;
  std::fill_n(centroid.data_handle(), static_cast<size_t>(dim), 0.0f);
  for (int64_t i = 0; i < n_rows; ++i) {
    for (int64_t d = 0; d < dim; ++d) {
      centroid(d) += data[i * dim + d];
    }
  }
  for (int64_t d = 0; d < dim; ++d) {
    centroid(d) /= static_cast<float>(n_rows);
    centroid_norm_sq += centroid(d) * centroid(d);
  }

  std::vector<uint8_t> unpacked(static_cast<size_t>(n_rows * dim));
  auto lower_intervals          = raft::make_host_vector<float, int64_t>(n_rows);
  auto upper_intervals          = raft::make_host_vector<float, int64_t>(n_rows);
  auto additional_corrections   = raft::make_host_vector<float, int64_t>(n_rows);
  auto quantized_component_sums = raft::make_host_vector<int32_t, int64_t>(n_rows);
  auto row_norm                 = raft::make_host_vector<float, int64_t>(n_rows);

#pragma omp parallel for
  for (int64_t i = 0; i < n_rows; ++i) {
    std::vector<float> row(data + i * dim, data + (i + 1) * dim);
    std::vector<uint8_t> codes(dim);
    float orig_norm2 = 0.0f;
    for (int64_t d = 0; d < dim; ++d) {
      orig_norm2 += row[d] * row[d];
    }
    row_norm(i)       = orig_norm2;
    const auto result = scalar_quantize(row, codes, bits, centroid.data_handle(), euclidean);
    std::copy(codes.begin(), codes.end(), unpacked.begin() + i * dim);
    lower_intervals(i)          = result.lower_interval;
    upper_intervals(i)          = result.upper_interval;
    additional_corrections(i)   = result.additional_correction;
    quantized_component_sums(i) = result.quantized_component_sum;
  }

  auto packed = pack_codes(unpacked, static_cast<size_t>(n_rows), static_cast<size_t>(dim), layout);
  auto codes  = raft::make_host_matrix<uint8_t, int64_t>(
    n_rows, get_encoded_row_length(static_cast<uint32_t>(dim), layout));
  std::copy(packed.begin(), packed.end(), codes.data_handle());

  host_quantizer_storage out{std::move(codes),
                             std::move(lower_intervals),
                             std::move(upper_intervals),
                             std::move(additional_corrections),
                             std::move(quantized_component_sums),
                             std::move(centroid),
                             raft::make_host_vector<float, int64_t>(n_rows),
                             raft::make_host_vector<float, int64_t>(n_rows),
                             std::move(row_norm),
                             layout,
                             metric,
                             centroid_norm_sq};
  derive_dequant_factors(out);
  return out;
}

/** The CPU reference quantizer works in float, so it only ever feeds a float-valued dataset. */
template <typename IdxT>
auto copy_bbq_owning_storage_host_to_device(raft::resources const& res,
                                            host_quantizer_storage const& host_storage) ->
  typename cuvs::neighbors::device_bbq_dataset<float, IdxT>::owning_storage_type
{
  using device_storage =
    typename cuvs::neighbors::device_bbq_dataset<float, IdxT>::owning_storage_type;
  auto stream = raft::resource::get_cuda_stream(res);
  device_storage device{res,
                        static_cast<IdxT>(host_storage.codes.extent(0)),
                        static_cast<uint32_t>(host_storage.centroid.extent(0)),
                        host_storage.layout,
                        host_storage.metric};

  raft::copy(
    device.codes.data_handle(), host_storage.codes.data_handle(), device.codes.size(), stream);
  raft::copy(device.lower_intervals.data_handle(),
             host_storage.lower_intervals.data_handle(),
             device.lower_intervals.size(),
             stream);
  raft::copy(device.upper_intervals.data_handle(),
             host_storage.upper_intervals.data_handle(),
             device.upper_intervals.size(),
             stream);
  raft::copy(device.additional_corrections.data_handle(),
             host_storage.additional_corrections.data_handle(),
             device.additional_corrections.size(),
             stream);
  raft::copy(device.quantized_component_sums.data_handle(),
             host_storage.quantized_component_sums.data_handle(),
             device.quantized_component_sums.size(),
             stream);
  raft::copy(device.centroid.data_handle(),
             host_storage.centroid.data_handle(),
             device.centroid.size(),
             stream);
  raft::copy(device.dequant_delta.data_handle(),
             host_storage.dequant_delta.data_handle(),
             device.dequant_delta.size(),
             stream);
  raft::copy(device.dequant_sum_delta.data_handle(),
             host_storage.dequant_sum_delta.data_handle(),
             device.dequant_sum_delta.size(),
             stream);
  raft::copy(device.row_norm.data_handle(),
             host_storage.row_norm.data_handle(),
             device.row_norm.size(),
             stream);
  device.centroid_norm_sq = host_storage.centroid_norm_sq;
  return device;
}

template <typename IdxT>
auto make_device_bbq_dataset(raft::resources const& res,
                             std::vector<host_quantizer_storage> const& host)
  -> cuvs::neighbors::device_bbq_dataset<float, IdxT>
{
  RAFT_EXPECTS(host.size() != 0, "host BBQ dataset has no storage");
  cuvs::neighbors::device_bbq_dataset<float, IdxT> device{
    copy_bbq_owning_storage_host_to_device<IdxT>(res, host[0])};
  for (std::size_t i = 1; i < host.size(); ++i) {
    device.add_quantizer(copy_bbq_owning_storage_host_to_device<IdxT>(res, host[i]));
  }
  return device;
}

struct bbq_layout_token {
  std::string_view token;
  bbq_code_layout layout;
};

// Same convention as my_tests/bbq's CLI tokens: bare N = densely packed (tensor-core-eligible),
// N + "t" = transposed/bitplane (SIMT). At 1 bit the two coincide, so there is no "1t". There is
// no densely-packed 2-bit layout at all (packed_2b was retired): transposed_2b is SIMT-only, so
// "2t" is the only 2-bit token and it is not tensor-core-eligible.
constexpr bbq_layout_token kBbqLayoutTokens[] = {
  {"1", bbq_code_layout::packed_1b},
  {"2t", bbq_code_layout::transposed_2b},
  {"4", bbq_code_layout::packed_4b},
  {"4t", bbq_code_layout::transposed_4b},
  {"7", bbq_code_layout::packed_7b},
  {"8", bbq_code_layout::packed_8b},
};

/** Parses a layout token. Width follows from the layout (bits=4 is packed_4b vs transposed_4b). */
inline auto parse_bbq_layout_token(std::string_view token) -> bbq_code_layout
{
  for (const auto& t : kBbqLayoutTokens) {
    if (t.token == token) { return t.layout; }
  }
  RAFT_FAIL("Unknown BBQ layout token '%s'; expected one of 1, 2t, 4, 4t, 7, 8.",
            std::string(token).c_str());
}

/**
 * Reject layout pairs the local-join kernels cannot serve. Without this the library either
 * silently falls back to a symmetric join on the first quantizer (which looks like a working
 * asymmetric run), or reaches an unsupported-layout failure deep inside GNND::build. Must be kept
 * in sync with the dispatch in GNND<Data_t, Index_t>::local_join (nn_descent.cuh).
 */
inline void validate_layout_pair(bbq_code_layout query_layout, bbq_code_layout doc_layout)
{
  if (query_layout == doc_layout) {
    const bool supported = query_layout == bbq_code_layout::packed_1b ||
                           query_layout == bbq_code_layout::transposed_2b ||
                           query_layout == bbq_code_layout::packed_4b ||
                           query_layout == bbq_code_layout::packed_7b ||
                           query_layout == bbq_code_layout::packed_8b;
    RAFT_EXPECTS(supported,
                 "Symmetric BBQ NN-Descent has no local-join kernel for layout %d -- "
                 "transposed_4b is only supported as one side of an asymmetric pair.",
                 static_cast<int>(query_layout));
    return;
  }
  const bool supported =
    (doc_layout == bbq_code_layout::packed_1b && query_layout == bbq_code_layout::packed_4b) ||
    (doc_layout == bbq_code_layout::packed_1b && query_layout == bbq_code_layout::transposed_2b) ||
    (doc_layout == bbq_code_layout::packed_1b && query_layout == bbq_code_layout::transposed_4b) ||
    (doc_layout == bbq_code_layout::transposed_2b &&
     query_layout == bbq_code_layout::transposed_4b);
  RAFT_EXPECTS(supported,
               "Asymmetric BBQ NN-Descent supports only (doc, query) layouts of (packed_1b, "
               "packed_4b), (packed_1b, transposed_2b), (packed_1b, transposed_4b), or "
               "(transposed_2b, transposed_4b); got (%d, %d).",
               static_cast<int>(doc_layout),
               static_cast<int>(query_layout));
}

/** Every parameter that affects the codes is in the name, so the cache self-invalidates. */
inline auto cache_path(int64_t n_rows,
                       int64_t dim,
                       bbq_code_layout layout,
                       cuvs::distance::DistanceType metric) -> std::string
{
  const char* dir = std::getenv("CUVS_BBQ_CACHE_DIR");
  return std::string{dir != nullptr && dir[0] != '\0' ? dir : "/tmp"} + "/bbq-n" +
         std::to_string(n_rows) + "-d" + std::to_string(dim) + "-b" +
         std::to_string(get_bit_width(layout)) + "-l" + std::to_string(static_cast<int>(layout)) +
         "-m" + std::to_string(static_cast<int>(metric)) + ".bin";
}

/** Visits every raw buffer of @p q in a fixed order; this is the on-disk layout. */
template <typename OpT>
void for_each_buffer(host_quantizer_storage& q, OpT op)
{
  op(q.codes.data_handle(), q.codes.size() * sizeof(uint8_t));
  op(q.lower_intervals.data_handle(), q.lower_intervals.size() * sizeof(float));
  op(q.upper_intervals.data_handle(), q.upper_intervals.size() * sizeof(float));
  op(q.additional_corrections.data_handle(), q.additional_corrections.size() * sizeof(float));
  op(q.quantized_component_sums.data_handle(), q.quantized_component_sums.size() * sizeof(int32_t));
  op(q.centroid.data_handle(), q.centroid.size() * sizeof(float));
  // Unlike dequant_delta/dequant_sum_delta, row_norm (original-space ||x||^2) isn't derivable
  // from the other cached fields, so it has to round-trip through the cache.
  op(q.row_norm.data_handle(), q.row_norm.size() * sizeof(float));
}

/** Allocates the arrays of the given shape, leaving their contents undefined. */
inline auto make_host_quantizer_storage(int64_t n_rows,
                                        int64_t dim,
                                        bbq_code_layout layout,
                                        cuvs::distance::DistanceType metric)
  -> host_quantizer_storage
{
  return host_quantizer_storage{
    raft::make_host_matrix<uint8_t, int64_t>(
      n_rows, get_encoded_row_length(static_cast<uint32_t>(dim), layout)),
    raft::make_host_vector<float, int64_t>(n_rows),
    raft::make_host_vector<float, int64_t>(n_rows),
    raft::make_host_vector<float, int64_t>(n_rows),
    raft::make_host_vector<int32_t, int64_t>(n_rows),
    raft::make_host_vector<float, int64_t>(dim),
    raft::make_host_vector<float, int64_t>(n_rows),
    raft::make_host_vector<float, int64_t>(n_rows),
    raft::make_host_vector<float, int64_t>(n_rows),
    layout,
    metric,
    0.0f};
}

/** Reads the codes back if the file is present and has exactly the expected length. */
inline auto cache_load(const std::string& path,
                       int64_t n_rows,
                       int64_t dim,
                       bbq_code_layout layout,
                       cuvs::distance::DistanceType metric) -> std::optional<host_quantizer_storage>
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { return std::nullopt; }
  auto q               = make_host_quantizer_storage(n_rows, dim, layout, metric);
  std::streamoff bytes = 0;
  for_each_buffer(q, [&bytes](void*, size_t n) { bytes += static_cast<std::streamoff>(n); });
  if (f.tellg() != bytes) {
    RAFT_LOG_WARN("Ignoring BBQ cache of unexpected size: %s", path.c_str());
    return std::nullopt;
  }
  f.seekg(0);
  for_each_buffer(
    q, [&f](void* p, size_t n) { f.read(static_cast<char*>(p), static_cast<std::streamsize>(n)); });
  if (!f) {
    RAFT_LOG_WARN("Failed to read BBQ cache, re-quantizing: %s", path.c_str());
    return std::nullopt;
  }
  // Cheaper to recompute than to store and validate. dequant_delta/dequant_sum_delta are likewise
  // a deterministic function of the cached lower/upper_intervals, quantized_component_sums, and
  // layout, so they are recomputed here rather than added to the on-disk layout.
  for (int64_t d = 0; d < dim; ++d) {
    q.centroid_norm_sq += q.centroid(d) * q.centroid(d);
  }
  derive_dequant_factors(q);
  RAFT_LOG_INFO("Loaded BBQ codes from %s", path.c_str());
  return q;
}

/** Writes via a temporary so an interrupted run cannot leave a truncated cache behind. */
inline void cache_store(const std::string& path, host_quantizer_storage& q)
{
  const std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) {
      RAFT_LOG_WARN("Cannot write BBQ cache: %s", tmp.c_str());
      return;
    }
    for_each_buffer(q, [&f](void* p, size_t n) {
      f.write(static_cast<const char*>(p), static_cast<std::streamsize>(n));
    });
    if (!f) {
      RAFT_LOG_WARN("Failed to write BBQ cache: %s", tmp.c_str());
      f.close();
      std::remove(tmp.c_str());
      return;
    }
  }
  if (std::rename(tmp.c_str(), path.c_str()) != 0) {
    RAFT_LOG_WARN("Cannot move BBQ cache into place: %s", path.c_str());
    std::remove(tmp.c_str());
    return;
  }
  RAFT_LOG_INFO("Wrote BBQ codes to %s", path.c_str());
}

inline auto quantize_cached(const float* rows,
                            int64_t n_rows,
                            int64_t dim,
                            bbq_code_layout layout,
                            cuvs::distance::DistanceType metric) -> host_quantizer_storage
{
  const auto path = cache_path(n_rows, dim, layout, metric);
  if (auto cached = cache_load(path, n_rows, dim, layout, metric)) { return std::move(*cached); }
  auto quantized = quantize(rows, n_rows, dim, metric, layout);
  cache_store(path, quantized);
  return quantized;
}

/**
 * Quantize host-resident @p rows and upload the codes, ready for the BBQ NN-Descent build.
 *
 * `query_token == doc_token` produces a single quantizer and a symmetric join; differing tokens
 * produce two quantizers and an asymmetric join, where the wider codes serve the query side.
 */
inline auto quantize_to_device(raft::resources const& res,
                               const float* rows,
                               int64_t n_rows,
                               int64_t dim,
                               cuvs::distance::DistanceType metric,
                               bbq_code_layout query_layout,
                               bbq_code_layout doc_layout)
  -> cuvs::neighbors::device_bbq_dataset<float, int64_t>
{
  validate_layout_pair(query_layout, doc_layout);
  std::vector<host_quantizer_storage> host;
  host.push_back(quantize_cached(rows, n_rows, dim, query_layout, metric));
  if (doc_layout != query_layout) {
    host.push_back(quantize_cached(rows, n_rows, dim, doc_layout, metric));
  }
  auto device = make_device_bbq_dataset<int64_t>(res, host);
  // The uploads are stream-ordered against `host`, which dies with this frame.
  raft::resource::sync_stream(res);
  return device;
}

inline auto quantize_to_device(raft::resources const& res,
                               const float* rows,
                               int64_t n_rows,
                               int64_t dim,
                               cuvs::distance::DistanceType metric,
                               std::string_view query_token,
                               std::string_view doc_token)
  -> cuvs::neighbors::device_bbq_dataset<float, int64_t>
{
  return quantize_to_device(res,
                            rows,
                            n_rows,
                            dim,
                            metric,
                            parse_bbq_layout_token(query_token),
                            parse_bbq_layout_token(doc_token));
}
}  // namespace cuvs_internal::bbq
