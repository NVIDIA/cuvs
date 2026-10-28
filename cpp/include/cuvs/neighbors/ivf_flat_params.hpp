/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/neighbors/params.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>

namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace ivf_flat {
/**
 * @defgroup ivf_flat_cpp_index_params IVF-Flat index build parameters
 * @{
 */
struct index_params : cuvs::neighbors::index_params {
  /** The number of inverted lists (clusters) */
  uint32_t n_lists = 1024;
  /** The number of iterations searching for kmeans centers (index building). */
  uint32_t kmeans_n_iters = 20;
  /** The fraction of data to use during iterative kmeans building. */
  double kmeans_trainset_fraction = 0.5;
  /**
   * By default (adaptive_centers = false), the cluster centers are trained in `ivf_flat::build`,
   * and never modified in `ivf_flat::extend`. As a result, you may need to retrain the index
   * from scratch after invoking (`ivf_flat::extend`) a few times with new data, the distribution of
   * which is no longer representative of the original training set.
   *
   * The alternative behavior (adaptive_centers = true) is to update the cluster centers for new
   * data when it is added. In this case, `index.centers()` are always exactly the centroids of the
   * data in the corresponding clusters. The drawback of this behavior is that the centroids depend
   * on the order of adding new data (through the classification of the added data); that is,
   * `index.centers()` "drift" together with the changing distribution of the newly added data.
   */
  bool adaptive_centers = false;
  /**
   * By default, the algorithm allocates more space than necessary for individual clusters
   * (`list_data`). This allows to amortize the cost of memory allocation and reduce the number of
   * data copies during repeated calls to `extend` (extending the database).
   *
   * The alternative is the conservative allocation behavior; when enabled, the algorithm always
   * allocates the minimum amount of memory required to store the given number of records. Set this
   * flag to `true` if you prefer to use as little GPU memory for the database as possible.
   */
  bool conservative_memory_allocation = false;
  /**
   * Whether to add the dataset content to the index, i.e.:
   *
   *  - `true` means the index is filled with the dataset vectors and ready to search after calling
   * `build`.
   *  - `false` means `build` only trains the underlying model (e.g. quantizer or clustering), but
   * the index is left empty; you'd need to call `extend` on the index afterwards to populate it.
   */
  bool add_data_on_build = true;
};
/**
 * @}
 */

/**
 * @defgroup ivf_flat_cpp_search_params IVF-Flat index search parameters
 * @{
 */
struct search_params : cuvs::neighbors::search_params {
  /** The number of clusters to search. */
  uint32_t n_probes = 20;
  /** Custom metric UDF code. */
  std::optional<std::string> metric_udf = std::nullopt;
};

/** @} */

static_assert(std::is_aggregate_v<index_params>);
static_assert(std::is_aggregate_v<search_params>);

}  // namespace ivf_flat
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
