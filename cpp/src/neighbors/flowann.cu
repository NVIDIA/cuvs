/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/flowann.hpp>

#include "detail/ann_utils.cuh"
#include "detail/flowann/search_multi_cta.cuh"

#include <raft/linalg/norm.cuh>
#include <raft/linalg/reduce.cuh>

#include <algorithm>
#include <optional>
#include <type_traits>
#include <typeinfo>

namespace cuvs::neighbors::cagra::experimental::flowann {

template <typename T, typename IdxT, cuvs::neighbors::ann_dataset_view DatasetViewT>
void index<T, IdxT, DatasetViewT>::compute_dataset_norms_(raft::resources const& res)
{
  if constexpr (cuvs::neighbors::is_device_padded_dataset_view_v<DatasetViewT>) {
    auto dataset = dataset_.view();
    if (!dataset_norms_.has_value() || dataset_norms_->extent(0) != dataset.extent(0)) {
      dataset_norms_.reset();
      dataset_norms_.emplace(raft::make_device_vector<float, int64_t>(res, dataset.extent(0)));
    }

    constexpr float scale = cuvs::spatial::knn::detail::utils::config<T>::kDivisor /
                            cuvs::spatial::knn::detail::utils::config<float>::kDivisor;
    auto scaled_sq_op =
      raft::compose_op(raft::sq_op{}, raft::div_const_op<float>{scale}, raft::cast_op<float>());
    raft::linalg::reduce<raft::Apply::ALONG_ROWS>(res,
                                                  dataset,
                                                  dataset_norms_->view(),
                                                  0.0f,
                                                  false,
                                                  scaled_sq_op,
                                                  raft::add_op(),
                                                  raft::sqrt_op{});
  }
}

namespace detail {

template <typename T,
          typename OutputIdxT,
          typename SampleFilterT,
          cuvs::neighbors::ann_dataset_view DatasetViewT>
void search_with_filtering(raft::resources const& res,
                           search_params const& params,
                           flowann::index<T, std::uint32_t, DatasetViewT> const& index,
                           search_context& context,
                           raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
                           raft::device_matrix_view<OutputIdxT, int64_t, raft::row_major> neighbors,
                           raft::device_matrix_view<float, int64_t, raft::row_major> distances,
                           SampleFilterT sample_filter)
{
  search_dispatch(res, params, index, context, queries, neighbors, distances, sample_filter);
}

template <typename T, typename OutputIdxT, cuvs::neighbors::ann_dataset_view DatasetViewT>
void search_with_base_filter(
  raft::resources const& res,
  search_params const& params,
  flowann::index<T, std::uint32_t, DatasetViewT> const& index,
  search_context& context,
  raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
  raft::device_matrix_view<OutputIdxT, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances,
  cuvs::neighbors::filtering::base_filter const& sample_filter_ref)
{
  try {
    using filter_type    = cuvs::neighbors::filtering::none_sample_filter;
    auto const& filter   = dynamic_cast<filter_type const&>(sample_filter_ref);
    auto adjusted_params = params;
    if (adjusted_params.filtering_rate < 0.0f) { adjusted_params.filtering_rate = 0.0f; }
    return search_with_filtering(
      res, adjusted_params, index, context, queries, neighbors, distances, filter);
  } catch (std::bad_cast const&) {
  }

  try {
    using filter_type    = cuvs::neighbors::filtering::bitset_filter<std::uint32_t, int64_t>;
    auto const& filter   = dynamic_cast<filter_type const&>(sample_filter_ref);
    auto adjusted_params = params;
    if (adjusted_params.filtering_rate < 0.0f) {
      auto const num_set_bits = filter.bitset_view_.count(res);
      auto const filtering_rate =
        static_cast<float>(index.dataset().n_rows() - num_set_bits) / index.dataset().n_rows();
      adjusted_params.filtering_rate = std::clamp(filtering_rate, 0.0f, 0.999f);
    }
    return search_with_filtering(
      res, adjusted_params, index, context, queries, neighbors, distances, filter);
  } catch (std::bad_cast const&) {
  }

  try {
    using filter_type    = cuvs::neighbors::filtering::bloom_filter;
    auto const& filter   = dynamic_cast<filter_type const&>(sample_filter_ref);
    auto adjusted_params = params;
    if (adjusted_params.filtering_rate < 0.0f) {
      auto const* bloom = static_cast<cuvs::core::bloom_filter const*>(filter.filter_data);
      RAFT_EXPECTS(bloom != nullptr,
                   "bloom_filter must carry a valid cuvs::core::bloom_filter handle");
      adjusted_params.filtering_rate = bloom->estimate_filtering_rate();
    }
    return search_with_filtering(
      res, adjusted_params, index, context, queries, neighbors, distances, filter);
  } catch (std::bad_cast const&) {
  }

  try {
    using filter_type    = cuvs::neighbors::filtering::udf_filter;
    auto const& filter   = dynamic_cast<filter_type const&>(sample_filter_ref);
    auto adjusted_params = params;
    if (adjusted_params.filtering_rate < 0.0f) {
      adjusted_params.filtering_rate =
        filter.filtering_rate < 0.0f ? 0.0f : std::clamp(filter.filtering_rate, 0.0f, 0.999f);
    }
    return search_with_filtering(
      res, adjusted_params, index, context, queries, neighbors, distances, filter);
  } catch (std::bad_cast const&) {
    RAFT_FAIL("Unsupported FlowANN sample filter type");
  }
}

}  // namespace detail

#define CUVS_FLOWANN_DEFINE_SEARCH(T, DatasetAlias, OutputIdxT)                         \
  void search(raft::resources const& res,                                               \
              search_params const& params,                                              \
              DatasetAlias<T, std::uint32_t> const& index,                              \
              search_context& context,                                                  \
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries,      \
              raft::device_matrix_view<OutputIdxT, int64_t, raft::row_major> neighbors, \
              raft::device_matrix_view<float, int64_t, raft::row_major> distances,      \
              cuvs::neighbors::filtering::base_filter const& sample_filter)             \
  {                                                                                     \
    detail::search_with_base_filter(                                                    \
      res, params, index, context, queries, neighbors, distances, sample_filter);       \
  }

#define CUVS_FLOWANN_DEFINE_SEARCH_FOR_TYPE(T)                                       \
  CUVS_FLOWANN_DEFINE_SEARCH(T, device_padded_index, std::uint32_t)                  \
  CUVS_FLOWANN_DEFINE_SEARCH(T, device_padded_index, std::int64_t)                   \
  CUVS_FLOWANN_DEFINE_SEARCH(T, vpq_f16_index, std::uint32_t)                        \
  CUVS_FLOWANN_DEFINE_SEARCH(T, vpq_f16_index, std::int64_t)                         \
  template CUVS_EXPORT void                                                          \
  index<T, std::uint32_t, cuvs::neighbors::device_padded_dataset_view<T, int64_t>>:: \
    compute_dataset_norms_(raft::resources const& res)

CUVS_FLOWANN_DEFINE_SEARCH_FOR_TYPE(float);
CUVS_FLOWANN_DEFINE_SEARCH_FOR_TYPE(half);
CUVS_FLOWANN_DEFINE_SEARCH_FOR_TYPE(std::int8_t);
CUVS_FLOWANN_DEFINE_SEARCH_FOR_TYPE(std::uint8_t);

#undef CUVS_FLOWANN_DEFINE_SEARCH_FOR_TYPE
#undef CUVS_FLOWANN_DEFINE_SEARCH

}  // namespace cuvs::neighbors::cagra::experimental::flowann
