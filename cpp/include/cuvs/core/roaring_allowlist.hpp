/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace CUVS_EXPORT cuvs {
namespace core {

/**
 * @brief Non-owning device view of one row in a batched Roaring allowlist.
 *
 * The view contains an opaque pointer to an already initialized device-side cuCollections
 * reference plus immutable shape and cardinality metadata. Creating or copying it is O(1) and
 * performs no allocation, parsing, kernel launch, or synchronization. The owning
 * @ref roaring_allowlist must outlive the view and every operation that uses it.
 */
class CUVS_EXPORT roaring_allowlist_view {
 public:
  roaring_allowlist_view() = default;

  [[nodiscard]] std::size_t dataset_rows() const noexcept { return dataset_rows_; }
  [[nodiscard]] std::size_t cardinality() const noexcept { return cardinality_; }
  [[nodiscard]] bool empty() const noexcept { return cardinality_ == 0; }
  [[nodiscard]] bool valid() const noexcept { return valid_; }

  /** @brief Opaque device pointer to the pre-parsed cuCollections reference, or null if empty. */
  [[nodiscard]] void const* device_reference() const noexcept { return device_reference_; }

 private:
  friend class roaring_allowlist;

  roaring_allowlist_view(void const* device_reference,
                         std::size_t dataset_rows,
                         std::size_t cardinality) noexcept
    : device_reference_(device_reference),
      dataset_rows_(dataset_rows),
      cardinality_(cardinality),
      valid_(true)
  {
  }

  void const* device_reference_{};
  std::size_t dataset_rows_{};
  std::size_t cardinality_{};
  bool valid_{};
};

/**
 * @brief Owning immutable batch of exact per-query Roaring allowlists.
 *
 * Logically, the owner is a sparse matrix with one allowlist row per query and one possible column
 * per dataset row. @ref from_ids accepts one contiguous ID vector plus an indptr vector that
 * delimits independently sized query rows. Every row is sorted and encoded independently. All
 * variable-length portable Roaring streams and their initialized
 * `cuco::experimental::roaring_bitmap_ref<uint32_t>` objects share one packed device allocation.
 *
 * For multiple rows, the builder uses indptr directly for segmented device radix sort and
 * schedules analysis/encoding over all containers in all rows. A one-row input retains the tuned
 * single-allowlist builder. Final encoding and reference initialization remain stream ordered.
 *
 * IDs must be unique within each row. Setting @p pre_sorted skips sorting and promises that every
 * row is strictly increasing; ordering and uniqueness are not checked. Every ID must be smaller
 * than  dataset_rows.
 *
 * @see https://github.com/RoaringBitmap/RoaringFormatSpec
 * @see
 * https://github.com/NVIDIA/cuCollections/blob/6001618aaa7f17ea2bbcd444650e9573c4f3d6c5/include/cuco/roaring_bitmap_ref.cuh
 */
class CUVS_EXPORT roaring_allowlist {
 private:
  struct impl;

 public:
  using key_type    = std::uint32_t;
  using indptr_type = std::int64_t;

  /**
   * @brief Build ragged allowlist rows from contiguous host IDs and row offsets.
   *
   * `indptr` contains `num_allowlists + 1` entries, starts at zero, is nondecreasing,
   * and ends at `ids.extent(0)`. Empty slices are valid allowlists.
   */
  static roaring_allowlist from_ids(raft::resources const& res,
                                    std::size_t dataset_rows,
                                    raft::host_vector_view<const key_type, std::int64_t> ids,
                                    raft::host_vector_view<const indptr_type, std::int64_t> indptr,
                                    bool pre_sorted = false);

  /**
   * @brief Build ragged allowlist rows from contiguous device IDs and row offsets.
   *
   * The same indptr invariants as the host overload apply. The row offsets are copied to the host
   * once for validation, shape-aware dispatch, and exact packed allocation.
   *
   * The input must remain valid until the construction stream reaches the work enqueued by this
   * call. Temporary memory is O(total input IDs + total containers); no dense dataset-sized bitmap
   * is materialized.
   */
  static roaring_allowlist from_ids(
    raft::resources const& res,
    std::size_t dataset_rows,
    raft::device_vector_view<const key_type, std::int64_t> ids,
    raft::device_vector_view<const indptr_type, std::int64_t> indptr,
    bool pre_sorted = false);

  /**
   * @brief Import packed standard 32-bit portable Roaring rows.
   *
   * `byte_offsets` has `num_allowlists + 1` entries, starts at zero, is nondecreasing, and ends at
   * `bytes.extent(0)`. Empty slices represent empty allowlists. Every row is strictly validated on
   * the host before its bytes are copied. The host buffers must remain valid until the construction
   * stream completes; pinned bytes are recommended when overlap matters.
   */
  static roaring_allowlist from_serialized(
    raft::resources const& res,
    std::size_t dataset_rows,
    raft::host_vector_view<const std::byte, std::int64_t> bytes,
    raft::host_vector_view<const std::uint64_t, std::int64_t> byte_offsets);

  ~roaring_allowlist();

  roaring_allowlist(roaring_allowlist const&)            = delete;
  roaring_allowlist& operator=(roaring_allowlist const&) = delete;
  roaring_allowlist(roaring_allowlist&&) noexcept;
  roaring_allowlist& operator=(roaring_allowlist&&) noexcept;

  [[nodiscard]] std::size_t num_allowlists() const noexcept;
  [[nodiscard]] std::size_t dataset_rows() const noexcept;
  [[nodiscard]] std::size_t cardinality(std::size_t allowlist_id) const;
  [[nodiscard]] bool empty(std::size_t allowlist_id) const;
  [[nodiscard]] std::size_t total_cardinality() const noexcept;

  /** @brief Total device bytes retained by packed rows, references, and row pointer metadata. */
  [[nodiscard]] std::size_t size_bytes() const noexcept;

  /** @brief Return a zero-copy view of one row. */
  [[nodiscard]] roaring_allowlist_view view(std::size_t allowlist_id) const;

  /**
   * @brief Test a matrix of row IDs and synchronize the resource stream.
   *
   * `row_ids[q][i]` is tested against allowlist row `q`. Input and output shapes must match, and
   * their first extent must equal @ref num_allowlists.
   */
  void contains(raft::resources const& res,
                raft::device_matrix_view<const key_type, std::int64_t, raft::row_major> row_ids,
                raft::device_matrix_view<std::uint8_t, std::int64_t, raft::row_major> output) const;

  /** @brief Stream-ordered asynchronous version of @ref contains. */
  void contains_async(
    raft::resources const& res,
    raft::device_matrix_view<const key_type, std::int64_t, raft::row_major> row_ids,
    raft::device_matrix_view<std::uint8_t, std::int64_t, raft::row_major> output) const;

 private:
  explicit roaring_allowlist(std::unique_ptr<impl> impl) noexcept;

  std::unique_ptr<impl> impl_;
};

}  // namespace core
}  // namespace CUVS_EXPORT cuvs
