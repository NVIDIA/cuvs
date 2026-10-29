/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * FlowANN is described in "Disentangling Graph Dependencies for Efficient
 * Billion-Scale GPU Vector Search" (OSDI 2026).
 */

#include <neighbors/detail/flowann/flowann.hpp>

#include "detail/ann_utils.cuh"

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

#define CUVS_FLOWANN_DEFINE_STORAGE(T)                                               \
  template CUVS_EXPORT void                                                          \
  index<T, std::uint32_t, cuvs::neighbors::device_padded_dataset_view<T, int64_t>>:: \
    compute_dataset_norms_(raft::resources const& res)

CUVS_FLOWANN_DEFINE_STORAGE(float);
CUVS_FLOWANN_DEFINE_STORAGE(half);
CUVS_FLOWANN_DEFINE_STORAGE(std::int8_t);
CUVS_FLOWANN_DEFINE_STORAGE(std::uint8_t);

#undef CUVS_FLOWANN_DEFINE_STORAGE

}  // namespace cuvs::neighbors::cagra::experimental::flowann
