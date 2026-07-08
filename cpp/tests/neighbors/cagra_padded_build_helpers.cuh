/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/neighbors/common.hpp>
#include <raft/core/device_mdspan.hpp>

#include <memory>

namespace cuvs::neighbors::test {

/**
 * Prepares a device_padded_dataset_view for cagra::build: uses make_device_padded_dataset_view when
 * the source row stride already matches alignment, otherwise make_device_padded_dataset and keeps
 * the copy in
 * \p owned. The caller must keep this object alive for the lifetime of any index that only holds a
 * view over the data.
 */
template <typename DataT>
struct padded_device_matrix_for_cagra {
  std::unique_ptr<cuvs::neighbors::device_padded_dataset<DataT, int64_t>> owned;
  cuvs::neighbors::device_padded_dataset_view<DataT, int64_t> view;

  padded_device_matrix_for_cagra(
    raft::resources const& res, raft::device_matrix_view<const DataT, int64_t, raft::row_major> src)
    : padded_device_matrix_for_cagra{build(res, src)}
  {
  }

 private:
  struct build_result {
    std::unique_ptr<cuvs::neighbors::device_padded_dataset<DataT, int64_t>> owned;
    cuvs::neighbors::device_padded_dataset_view<DataT, int64_t> view;
  };

  // device_padded_dataset_view has no default constructor; fill both members from one build step.
  explicit padded_device_matrix_for_cagra(build_result&& br)
    : owned{std::move(br.owned)}, view{std::move(br.view)}
  {
  }

  static auto build(raft::resources const& res,
                    raft::device_matrix_view<const DataT, int64_t, raft::row_major> src)
    -> build_result
  {
    using namespace cuvs::neighbors;
    if (matrix_row_width_matches_cagra_required(src)) {
      return build_result{nullptr, make_device_padded_dataset_view(res, src)};
    } else {
      auto own = make_device_padded_dataset(res, src);
      auto vw  = own->as_dataset_view();
      return build_result{std::move(own), vw};
    }
  }
};

}  // namespace cuvs::neighbors::test
