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
 * @brief Non-owning device view of one immutable Roaring allowlist.
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

  /** @brief Opaque device pointer to the initialized cuCollections reference, or null if empty. */
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
 * @brief Owning immutable exact Roaring allowlist over CAGRA dataset-row IDs.
 *
 * Build an allowlist through @ref from_ids, then pass its zero-copy @ref view to a
 * cuvs::neighbors::filtering::roaring_filter. A filter maps one such view to each query; owners
 * remain independent and can therefore be reused across filters and queries.
 *
 * Construction sorts IDs on the GPU unless @p pre_sorted is true. Setting @p pre_sorted promises
 * that IDs are already in strictly increasing order; this promise is not verified. IDs must be
 * unique and smaller than @p dataset_rows. The encoded bytes and the initialized
 * cuco::experimental::roaring_bitmap_ref<uint32_t> are retained on the device. Creating a view
 * never copies or reparses them, and CAGRA search performs no Roaring initialization.
 *
 * ID-based construction emits the standard portable 32-bit Roaring array and bitmap container
 * forms. Each ID is partitioned by its high 16 bits; the low 16 bits are stored as an array for at
 * most 4,096 values in a partition and as an 8 KiB bitmap otherwise.
 *
 * @see https://github.com/RoaringBitmap/RoaringFormatSpec
 * @see https://github.com/NVIDIA/cuCollections/pull/839
 */
class CUVS_EXPORT roaring_allowlist {
 private:
  struct impl;

 public:
  using key_type = std::uint32_t;

  /**
   * @brief Build one allowlist from host IDs.
   *
   * Host IDs are copied to the construction stream and then use the same device builder as the
   * device overload. IDs must be smaller than dataset_rows; this precondition is not checked.
   * Empty input is valid and rejects every candidate.
   */
  static roaring_allowlist from_ids(raft::resources const& res,
                                    std::size_t dataset_rows,
                                    raft::host_vector_view<const key_type, std::int64_t> ids,
                                    bool pre_sorted = false);

  /**
   * @brief Build one allowlist from device IDs.
   *
   * The input must remain valid until the construction stream reaches the enqueued work. IDs must
   * be smaller than dataset_rows; this precondition is not checked.
   * Temporary memory is O(cardinality + container count); no dataset-sized dense bitmap is used.
   */
  static roaring_allowlist from_ids(raft::resources const& res,
                                    std::size_t dataset_rows,
                                    raft::device_vector_view<const key_type, std::int64_t> ids,
                                    bool pre_sorted = false);

  ~roaring_allowlist();

  roaring_allowlist(roaring_allowlist const&)            = delete;
  roaring_allowlist& operator=(roaring_allowlist const&) = delete;
  roaring_allowlist(roaring_allowlist&&) noexcept;
  roaring_allowlist& operator=(roaring_allowlist&&) noexcept;

  [[nodiscard]] std::size_t dataset_rows() const noexcept;
  [[nodiscard]] std::size_t cardinality() const noexcept;
  [[nodiscard]] bool empty() const noexcept;

  /** @brief Total device bytes retained by the encoded allowlist and initialized reference. */
  [[nodiscard]] std::size_t size_bytes() const noexcept;

  /** @brief Return a zero-copy view. */
  [[nodiscard]] roaring_allowlist_view view() const noexcept;

 private:
  explicit roaring_allowlist(std::unique_ptr<impl> impl) noexcept;

  std::unique_ptr<impl> impl_;
};

}  // namespace core
}  // namespace CUVS_EXPORT cuvs
