/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/neighbors/hnsw.hpp>
#include <cuvs/util/file_io.hpp>

#include <hnswlib/hnswalg.h>
#include <hnswlib/hnswlib.h>

#include <raft/core/logger.hpp>

#include <cuda_fp16.h>
#include <filesystem>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <type_traits>

namespace cuvs::neighbors::hnsw::detail {

// This is needed as hnswlib hardcodes the distance type to float
// or int32_t in certain places. However, we can solve uint8 or int8
// natively with the patch cuVS applies. We could potentially remove
// all the hardcodes and propagate templates throughout hnswlib, but
// as of now it's not needed.
template <typename T>
struct hnsw_dist_t {
  using type = void;
};

template <>
struct hnsw_dist_t<float> {
  using type = float;
};

template <>
struct hnsw_dist_t<half> {
  using type = float;
};

template <>
struct hnsw_dist_t<uint8_t> {
  using type = int;
};

template <>
struct hnsw_dist_t<int8_t> {
  using type = int;
};

template <typename T>
struct index_impl : index<T> {
 public:
  /**
   * @brief load a base-layer-only hnswlib index originally saved from a built CAGRA index
   *
   * @param[in] filepath path to the index
   * @param[in] dim dimensions of the training dataset
   * @param[in] metric distance metric to search. Supported metrics ("L2Expanded", "InnerProduct")
   * @param[in] hierarchy hierarchy used for upper HNSW layers
   */
  index_impl(int dim, cuvs::distance::DistanceType metric, HnswHierarchy hierarchy)
    : index<T>{dim, metric, hierarchy}
  {
    if (metric == cuvs::distance::DistanceType::InnerProduct) {
      space_ = std::make_unique<hnswlib::InnerProductSpace<T, typename hnsw_dist_t<T>::type>>(dim);
    } else if (metric == cuvs::distance::DistanceType::L2Expanded) {
      if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
        space_ = std::make_unique<hnswlib::L2Space<T, typename hnsw_dist_t<T>::type>>(dim);
      } else if constexpr (std::is_same_v<T, std::int8_t> or std::is_same_v<T, std::uint8_t>) {
        space_ = std::make_unique<hnswlib::L2SpaceI<T>>(dim);
      }
    }

    RAFT_EXPECTS(space_ != nullptr, "Unsupported metric type was used");
  }

  /**
  @brief Get hnswlib index
  */
  auto get_index() const -> void const* override { return appr_alg_.get(); }

  /**
  @brief Set ef for search
  */
  void set_ef(int ef) const override
  {
    ensure_loaded();
    appr_alg_->ef_ = ef;
  }

  /**
  @brief Set index
   */
  void set_index(std::unique_ptr<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>&& index)
  {
    appr_alg_ = std::move(index);
  }

  /**
  @brief Get space
   */
  auto get_space() const -> hnswlib::SpaceInterface<typename hnsw_dist_t<T>::type>*
  {
    return space_.get();
  }

  /**
  @brief Set file descriptor for disk-backed index
   */
  void set_file_descriptor(cuvs::util::file_descriptor&& fd) { hnsw_fd_.emplace(std::move(fd)); }

  /**
  @brief Get file descriptor
   */
  auto file_descriptor() const -> const std::optional<cuvs::util::file_descriptor>&
  {
    return hnsw_fd_;
  }

  /**
  @brief Get file path for disk-backed index
   */
  std::string file_path() const override
  {
    if (hnsw_fd_.has_value() && hnsw_fd_->is_valid()) { return hnsw_fd_->get_path(); }
    return "";
  }

  /**
  @brief Ensure the index is loaded into memory.
         If the index is disk-backed and not yet loaded, this will load it from the file.
   */
  void ensure_loaded() const
  {
    if (appr_alg_ != nullptr) { return; }  // Already loaded

    // Check if we have a file descriptor to load from
    if (!hnsw_fd_.has_value() || !hnsw_fd_->is_valid()) {
      RAFT_FAIL("Cannot load HNSW index: no file descriptor available and index not in memory");
    }

    std::string filepath = hnsw_fd_->get_path();
    RAFT_EXPECTS(!filepath.empty(), "Cannot load HNSW index: file path is empty");
    RAFT_EXPECTS(std::filesystem::exists(filepath),
                 "Cannot load HNSW index: file does not exist: %s",
                 filepath.c_str());

    RAFT_LOG_INFO("Loading HNSW index from disk: %s", filepath.c_str());

    try {
      appr_alg_ = std::make_unique<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>>(
        space_.get(), filepath);
      if (this->hierarchy() == HnswHierarchy::NONE) { appr_alg_->base_layer_only = true; }
    } catch (const std::bad_alloc& e) {
      RAFT_FAIL(
        "Failed to load HNSW index from '%s': insufficient host memory. "
        "The index is too large to fit in available RAM. "
        "Consider using a machine with more memory or reducing the dataset size.",
        filepath.c_str());
    }
  }

 private:
  mutable std::unique_ptr<hnswlib::HierarchicalNSW<typename hnsw_dist_t<T>::type>> appr_alg_;
  std::unique_ptr<hnswlib::SpaceInterface<typename hnsw_dist_t<T>::type>> space_;
  std::optional<cuvs::util::file_descriptor> hnsw_fd_;
};

}  // namespace cuvs::neighbors::hnsw::detail
