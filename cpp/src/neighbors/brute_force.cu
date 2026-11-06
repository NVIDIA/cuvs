/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/cuda_fp16.hpp>

#include "./detail/int8_brute_force.cuh"
#include "./detail/knn_brute_force.cuh"

#include <cuvs/neighbors/brute_force.hpp>

#include <raft/core/copy.hpp>

namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace brute_force {

template <typename T, typename DistT>
index<T, DistT, void>::index(raft::resources const& res)
  // this constructor is just for a temporary index, for use in the deserialization
  // api. all the parameters here will get replaced with loaded values - that aren't
  // necessarily known ahead of time before deserialization.
  // TODO: do we even need a handle here - could just construct one?
  : cuvs::neighbors::index(),
    metric_(cuvs::distance::DistanceType::L2Expanded),
    dataset_(raft::make_device_matrix<T, int64_t>(res, 0, 0)),
    norms_(std::nullopt),
    metric_arg_(0)
{
}

template <typename T, typename DistT>
index<T, DistT, void>::index(raft::resources const& res,
                             raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,
                             std::optional<raft::device_vector<DistT, int64_t>>&& norms,
                             cuvs::distance::DistanceType metric,
                             DistT metric_arg)
  : cuvs::neighbors::index(),
    metric_(metric),
    dataset_(raft::make_device_matrix<T, int64_t>(res, 0, 0)),
    norms_(std::move(norms)),
    metric_arg_(metric_arg)
{
  if (norms_) { norms_view_ = raft::make_const_mdspan(norms_.value().view()); }
  update_dataset(res, dataset);
  raft::resource::sync_stream(res);
}

template <typename T, typename DistT>
index<T, DistT, void>::index(raft::resources const&,
                             raft::device_matrix<T, int64_t, raft::row_major>&& dataset,
                             std::optional<raft::device_vector<DistT, int64_t>>&& norms,
                             cuvs::distance::DistanceType metric,
                             DistT metric_arg)
  : cuvs::neighbors::index(),
    metric_(metric),
    dataset_(std::move(dataset)),
    norms_(std::move(norms)),
    metric_arg_(metric_arg)
{
  dataset_view_ = raft::make_const_mdspan(dataset_.view());
  if (norms_) { norms_view_ = raft::make_const_mdspan(norms_->view()); }
}

template <typename T, typename DistT>
index<T, DistT, void>::index(raft::resources const& res,
                             raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,
                             std::optional<raft::device_vector<DistT, int64_t>>&& norms,
                             cuvs::distance::DistanceType metric,
                             DistT metric_arg)
  : cuvs::neighbors::index(),
    metric_(metric),
    dataset_(raft::make_device_matrix<T, int64_t>(res, 0, 0)),
    norms_(std::move(norms)),
    metric_arg_(metric_arg)
{
  if (norms_) { norms_view_ = raft::make_const_mdspan(norms_.value().view()); }
  update_dataset(res, dataset);
}

template <typename T, typename DistT>
index<T, DistT, void>::index(
  raft::resources const& res,
  raft::device_matrix_view<const T, int64_t, raft::row_major> dataset_view,
  std::optional<raft::device_vector_view<const DistT, int64_t>> norms_view,
  cuvs::distance::DistanceType metric,
  DistT metric_arg)
  : cuvs::neighbors::index(),
    metric_(metric),
    dataset_(raft::make_device_matrix<T, int64_t>(res, 0, 0)),
    dataset_view_(dataset_view),
    norms_view_(norms_view),
    metric_arg_(metric_arg)
{
}

template <typename T, typename DistT>
index<T, DistT, void>::index(
  raft::resources const& res,
  raft::device_matrix_view<const T, int64_t, raft::col_major> dataset_view,
  std::optional<raft::device_vector<DistT, int64_t>>&& norms,
  cuvs::distance::DistanceType metric,
  DistT metric_arg)
  : cuvs::neighbors::index(),
    metric_(metric),
    dataset_(
      raft::make_device_matrix<T, int64_t>(res, dataset_view.extent(0), dataset_view.extent(1))),
    norms_(std::move(norms)),
    metric_arg_(metric_arg)
{
  // currently we don't support col_major inside tiled_brute_force_knn, because
  // of limitations of the pairwise_distance API:
  // 1) paiwise_distance takes a single 'isRowMajor' parameter - and we have
  // multiple options here (both dataset and queries)
  // 2) because of tiling, we need to be able to set a custom stride in the PW
  // api, which isn't supported
  // Instead, transpose the input matrices if they are passed as col-major.
  // (note: we're doing the transpose here to avoid doing per query)
  raft::linalg::transpose(res,
                          const_cast<T*>(dataset_view.data_handle()),
                          dataset_.data_handle(),
                          dataset_view.extent(0),
                          dataset_view.extent(1),
                          raft::resource::get_cuda_stream(res).get());
  dataset_view_ = raft::make_const_mdspan(dataset_.view());
}

template <typename T, typename DistT>
index<T, DistT, void>::index(
  raft::resources const& res,
  raft::device_matrix_view<const T, int64_t, raft::col_major> dataset_view,
  std::optional<raft::device_vector_view<const DistT, int64_t>> norms_view,
  cuvs::distance::DistanceType metric,
  DistT metric_arg)
  : cuvs::neighbors::index(),
    metric_(metric),
    dataset_(
      raft::make_device_matrix<T, int64_t>(res, dataset_view.extent(0), dataset_view.extent(1))),
    norms_view_(norms_view),
    metric_arg_(metric_arg)
{
  // currently we don't support col_major inside tiled_brute_force_knn, because
  // of limitations of the pairwise_distance API:
  // 1) paiwise_distance takes a single 'isRowMajor' parameter - and we have
  // multiple options here (both dataset and queries)
  // 2) because of tiling, we need to be able to set a custom stride in the PW
  // api, which isn't supported
  // Instead, transpose the input matrices if they are passed as col-major.
  // (note: we're doing the transpose here to avoid doing per query)
  raft::linalg::transpose(res,
                          const_cast<T*>(dataset_view.data_handle()),
                          dataset_.data_handle(),
                          dataset_view.extent(0),
                          dataset_view.extent(1),
                          raft::resource::get_cuda_stream(res).get());
  dataset_view_ = raft::make_const_mdspan(dataset_.view());
}

template <typename T, typename DistT>
void index<T, DistT, void>::update_dataset(
  raft::resources const& res, raft::device_matrix_view<const T, int64_t, raft::row_major> dataset)
{
  dataset_view_ = dataset;
}

template <typename T, typename DistT>
void index<T, DistT, void>::update_dataset(
  raft::resources const& res, raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)
{
  dataset_ = raft::make_device_matrix<T, int64_t>(res, dataset.extent(0), dataset.extent(1));
  raft::copy(res, dataset_.view(), dataset);
  dataset_view_ = raft::make_const_mdspan(dataset_.view());
}

#define CUVS_INST_BFKNN(T, DistT)                                                               \
  auto build(raft::resources const& res,                                                        \
             const cuvs::neighbors::brute_force::index_params& index_params,                    \
             raft::device_matrix_view<const T, int64_t, raft::row_major> dataset)               \
    -> cuvs::neighbors::brute_force::index<T, DistT>                                            \
  {                                                                                             \
    return detail::build<T, DistT>(res, dataset, index_params.metric, index_params.metric_arg); \
  }                                                                                             \
  auto build(raft::resources const& res,                                                        \
             const cuvs::neighbors::brute_force::index_params& index_params,                    \
             raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)                 \
    -> cuvs::neighbors::brute_force::index<T, DistT>                                            \
  {                                                                                             \
    return detail::build<T, DistT>(res, dataset, index_params.metric, index_params.metric_arg); \
  }                                                                                             \
  auto build(raft::resources const& res,                                                        \
             raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,               \
             cuvs::distance::DistanceType metric,                                               \
             DistT metric_arg) -> cuvs::neighbors::brute_force::index<T, DistT>                 \
  {                                                                                             \
    return detail::build<T, DistT>(res, dataset, metric, metric_arg);                           \
  }                                                                                             \
  auto build(raft::resources const& res,                                                        \
             const cuvs::neighbors::brute_force::index_params& index_params,                    \
             raft::device_matrix_view<const T, int64_t, raft::col_major> dataset)               \
    -> cuvs::neighbors::brute_force::index<T, DistT>                                            \
  {                                                                                             \
    return detail::build<T, DistT>(res, dataset, index_params.metric, index_params.metric_arg); \
  }                                                                                             \
  auto build(raft::resources const& res,                                                        \
             raft::device_matrix_view<const T, int64_t, raft::col_major> dataset,               \
             cuvs::distance::DistanceType metric,                                               \
             DistT metric_arg) -> cuvs::neighbors::brute_force::index<T, DistT>                 \
  {                                                                                             \
    return detail::build<T, DistT>(res, dataset, metric, metric_arg);                           \
  }                                                                                             \
                                                                                                \
  void search(raft::resources const& res,                                                       \
              const cuvs::neighbors::brute_force::search_params& params,                        \
              const cuvs::neighbors::brute_force::index<T, DistT>& idx,                         \
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries,              \
              raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,            \
              raft::device_matrix_view<DistT, int64_t, raft::row_major> distances,              \
              const cuvs::neighbors::filtering::base_filter& sample_filter)                     \
  {                                                                                             \
    detail::search<T, int64_t, DistT, raft::row_major>(                                         \
      res, idx, queries, neighbors, distances, sample_filter);                                  \
  }                                                                                             \
  void search(raft::resources const& res,                                                       \
              const cuvs::neighbors::brute_force::index<T, DistT>& idx,                         \
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries,              \
              raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,            \
              raft::device_matrix_view<DistT, int64_t, raft::row_major> distances,              \
              const cuvs::neighbors::filtering::base_filter& sample_filter)                     \
  {                                                                                             \
    detail::search<T, int64_t, DistT, raft::row_major>(                                         \
      res, idx, queries, neighbors, distances, sample_filter);                                  \
  }                                                                                             \
  void search(raft::resources const& res,                                                       \
              const cuvs::neighbors::brute_force::search_params& params,                        \
              const cuvs::neighbors::brute_force::index<T, DistT>& idx,                         \
              raft::device_matrix_view<const T, int64_t, raft::col_major> queries,              \
              raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,            \
              raft::device_matrix_view<DistT, int64_t, raft::row_major> distances,              \
              const cuvs::neighbors::filtering::base_filter& sample_filter)                     \
  {                                                                                             \
    detail::search<T, int64_t, DistT, raft::col_major>(                                         \
      res, idx, queries, neighbors, distances, sample_filter);                                  \
  }                                                                                             \
  void search(raft::resources const& res,                                                       \
              const cuvs::neighbors::brute_force::index<T, DistT>& idx,                         \
              raft::device_matrix_view<const T, int64_t, raft::col_major> queries,              \
              raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,            \
              raft::device_matrix_view<DistT, int64_t, raft::row_major> distances,              \
              const cuvs::neighbors::filtering::base_filter& sample_filter)                     \
  {                                                                                             \
    detail::search<T, int64_t, DistT, raft::col_major>(                                         \
      res, idx, queries, neighbors, distances, sample_filter);                                  \
  }

CUVS_INST_BFKNN(float, float);
CUVS_INST_BFKNN(half, float);

template class cuvs::neighbors::brute_force::index<float, float>;
template class cuvs::neighbors::brute_force::index<half, float>;

#undef CUVS_INST_BFKNN

template <typename T, typename DistT, typename DatasetViewT>
index<T, DistT, DatasetViewT>::index(raft::resources const& res,
                                     index_params const& params,
                                     DatasetViewT dataset)
  : dataset_(dataset),
    metric_(params.metric),
    metric_arg_(params.metric_arg),
    norms_(raft::make_device_vector<double, int64_t>(
      res, params.metric != cuvs::distance::DistanceType::InnerProduct ? dataset.n_rows() : 0)),
    dot_corrections_(raft::make_device_vector<double, int64_t>(
      res,
      params.metric == cuvs::distance::DistanceType::InnerProduct ||
          params.metric == cuvs::distance::DistanceType::CosineExpanded
        ? dataset.n_rows()
        : 0))
{
  RAFT_EXPECTS(size() > 0, "SQ brute-force dataset must be nonempty");
  RAFT_EXPECTS(dim() > 0 && dim() % 4 == 0,
               "SQ brute-force requires a positive dimension divisible by 4");
  RAFT_EXPECTS(dim() <= std::numeric_limits<std::int32_t>::max() / (128 * 128),
               "SQ brute-force dimension is too large for exact INT32 dot-product accumulation");
  RAFT_EXPECTS(detail::int8_supported_metric(metric_),
               "SQ brute-force supports InnerProduct, L2, and CosineExpanded metrics");
  auto const affine = detail::get_sq_affine_parameters<T>(dataset_.const_manifest_view(), metric_);
  detail::compute_sq_statistics(
    res, dataset_.data_const_view(), affine, metric_, norms_.view(), dot_corrections_.view());
}

#define CUVS_INST_SQ_BFKNN(T)                                                              \
  auto build(raft::resources const& res,                                                   \
             const cuvs::neighbors::brute_force::index_params& params,                     \
             cuvs::device_sq_dataset<T, int64_t, T>::const_view_type dataset)              \
    -> cuvs::neighbors::brute_force::device_sq_index<T>                                    \
  {                                                                                        \
    return device_sq_index<T>(res, params, dataset);                                       \
  }                                                                                        \
  void search(raft::resources const& res,                                                  \
              const cuvs::neighbors::brute_force::search_params& params,                   \
              const cuvs::neighbors::brute_force::device_sq_index<T>& idx,                 \
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries,         \
              raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,       \
              raft::device_matrix_view<float, int64_t, raft::row_major> distances,         \
              const cuvs::neighbors::filtering::base_filter& sample_filter)                \
  {                                                                                        \
    RAFT_EXPECTS(                                                                          \
      dynamic_cast<const cuvs::neighbors::filtering::none_sample_filter*>(&sample_filter), \
      "filtered search is not available for SQ brute-force indexes");                      \
    detail::int8_brute_force_search(res, idx, queries, neighbors, distances);              \
  }                                                                                        \
  template class index<T, float, cuvs::device_sq_dataset<T, int64_t, T>::const_view_type>;

CUVS_INST_SQ_BFKNN(float);
CUVS_INST_SQ_BFKNN(half);
CUVS_INST_SQ_BFKNN(double);

#undef CUVS_INST_SQ_BFKNN

}  // namespace brute_force
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
