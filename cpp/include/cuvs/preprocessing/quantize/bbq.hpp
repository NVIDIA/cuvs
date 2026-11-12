/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/dataset/bbq.hpp>
#include <cuvs/core/export.hpp>
#include <cuvs/distance/distance.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cstdint>

namespace CUVS_EXPORT cuvs {

namespace preprocessing::quantize::bbq {

/**
 * @defgroup bbq-preprocessing Better Binary Quantization utilities
 * @{
 */

/**
 * doc_layout: The code layout of the encoded rows on the document side of a distance
 * query_layout: The code layout of the encoded rows on the query side of a distance. When it
 * differs from doc_layout, the dataset is encoded a second time in this layout.
 * metric: Used for selecting the per-row additional correction: ||x - c||^2 for L2, x.c otherwise.
 * lambda: Relative interval weight of the quantization error against the norm-preserving term of
 * the loss.
 * iters: Coordinate descent iterations per row.
 */
struct params {
  bbq_code_layout doc_layout          = bbq_code_layout::packed_1b;
  bbq_code_layout query_layout        = bbq_code_layout::transposed_4b;
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded;
  float lambda                        = 0.1f;
  int iters                           = 5;
};

namespace helpers {
/**
 * Derives dequant_delta from lower/upper_intervals and the layout's code width, and
 * dequant_sum_delta from that delta and quantized_component_sums.
 */
void resolve_dequant_factors(
  raft::resources const& res,
  raft::device_vector_view<float, int64_t> dequant_delta,
  raft::device_vector_view<float, int64_t> dequant_sum_delta,
  raft::device_vector_view<const float, int64_t> lower_intervals,
  raft::device_vector_view<const float, int64_t> upper_intervals,
  raft::device_vector_view<const int32_t, int64_t> quantized_component_sums,
  bbq_code_layout layout);
}  // namespace helpers

/**
 * @brief Builds a BBQ quantizer from a dataset: computes the centroid shared by every row, and its
 * squared norm, into @p quantizer.
 *
 * @p quantizer is typically the manifest of a BBQ dataset from `make_device_bbq_dataset`.
 * Its dimension has to match the one of @p dense_dataset.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::bbq::params params;
 * auto bbq_dataset = cuvs::preprocessing::quantize::bbq::make_device_bbq_dataset(
 *   handle, params, dense_dataset.extent(0), dense_dataset.extent(1));
 * cuvs::preprocessing::quantize::bbq::build(handle, dense_dataset, bbq_dataset.manifest_view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] dense_dataset a row-major matrix view on device
 * @param[out] quantizer a view of the BBQ quantizer to write the centroid into
 */
void build(raft::resources const& res,
           raft::device_matrix_view<const float, int64_t> dense_dataset,
           quantizer_view<float> quantizer);

/**
 * @brief Allocates a BBQ dataset of the shape `transform` writes, leaving its contents
 * uninitialized.
 *
 * The rows are allocated in `params.doc_layout` and, when `params.query_layout` differs, a second
 * time in that layout; the metric is recorded from `params.metric`, and the quantizer holds room
 * for a centroid of @p dim components.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::bbq::params params;
 * auto bbq_dataset =
 *   cuvs::preprocessing::quantize::bbq::make_device_bbq_dataset(handle, params, n_rows, dim);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configuration for BBQ encoding; only doc_layout, query_layout and metric are
 * used
 * @param[in] n_rows number of rows
 * @param[in] dim number of components of each row before encoding
 *
 * @return an uninitialized BBQ dataset
 */
cuvs::device_bbq_dataset<float, int64_t> make_device_bbq_dataset(raft::resources const& res,
                                                                 const params params,
                                                                 int64_t n_rows,
                                                                 uint32_t dim);

/**
 * @brief Encodes a dataset into a BBQ dataset, against the centroid in the manifest of the BBQ
 * dataset.
 *
 * @p out receives the rows encoded in `params.doc_layout` format, and when specified in
 * `params.query_layout` format as well. The manifest of @p out has to hold the quantizer already,
 * from a previous call to `build`.
 *
 * Usage example:
 * @code{.cpp}
 * raft::handle_t handle;
 * cuvs::preprocessing::quantize::bbq::params params;
 * auto bbq_dataset = cuvs::preprocessing::quantize::bbq::make_device_bbq_dataset(
 *   handle, params, dense_dataset.extent(0), dense_dataset.extent(1));
 * cuvs::preprocessing::quantize::bbq::build(handle, dense_dataset, bbq_dataset.manifest_view());
 * cuvs::preprocessing::quantize::bbq::transform(handle, params, dense_dataset, bbq_dataset.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] params configuration for BBQ encoding, e.g. doc_layout, query_layout, metric, lambda,
 * iters
 * @param[in] dense_dataset a row-major matrix view on device
 * @param[inout] out a view of the BBQ dataset whose manifest holds the quantizer, to write the
 * encoding into
 */
void transform(raft::resources const& res,
               const params params,
               raft::device_matrix_view<const float, int64_t> dense_dataset,
               cuvs::device_bbq_dataset<float, int64_t>::view_type out);

/** @} */  // end of bbq group

}  // namespace preprocessing::quantize::bbq
}  // namespace CUVS_EXPORT cuvs
