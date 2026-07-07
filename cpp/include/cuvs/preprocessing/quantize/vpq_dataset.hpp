/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/common.hpp>
#include <raft/core/logger.hpp>

#include <optional>

namespace cuvs::preprocessing::quantize::pq {

/**
 * @brief Abstract interface for VPQ codebook access.
 *
 * @tparam MathT the type of elements in the codebooks
 */
template <typename MathT>
class vpq_codebooks_iface {
 public:
  using math_type = MathT;

  virtual ~vpq_codebooks_iface() = default;

  /**
   * VQ codebook [vq_n_centers, dim].
   *
   * Returns std::nullopt when no VQ codebook is configured (i.e. PQ-only,
   * use_vq=false). Callers that need to forward a `device_matrix_view`
   * downstream should materialize an empty 0x0 view themselves on nullopt.
   */
  [[nodiscard]] virtual auto vq_code_book() const noexcept
    -> std::optional<raft::device_matrix_view<const math_type, uint32_t, raft::row_major>> = 0;

  /** PQ codebook [pq_n_centers (× pq_dim for subspaces), pq_len]. */
  [[nodiscard]] virtual auto pq_code_book() const noexcept
    -> raft::device_matrix_view<const math_type, uint32_t, raft::row_major> = 0;

  [[nodiscard]] virtual auto dim() const noexcept -> uint32_t
  {
    auto vq = vq_code_book();
    if (!vq.has_value()) {
      RAFT_LOG_WARN(
        "vpq_codebooks_iface::dim() returns 0 when no VQ codebook is present; "
        "the original vector dimension cannot be recovered from PQ codebooks alone.");
      return 0;
    }
    return vq->extent(1);
  }
  [[nodiscard]] virtual auto vq_n_centers() const noexcept -> uint32_t
  {
    auto vq = vq_code_book();
    return vq.has_value() ? vq->extent(0) : 0;
  }
  [[nodiscard]] virtual auto pq_len() const noexcept -> uint32_t
  {
    return pq_code_book().extent(1);
  }
  [[nodiscard]] virtual auto pq_n_centers() const noexcept -> uint32_t
  {
    return pq_code_book().extent(0);
  }
  [[nodiscard]] virtual auto pq_bits() const noexcept -> uint32_t
  {
    auto w        = pq_n_centers();
    uint32_t bits = 0;
    while (w > 1) {
      bits++;
      w >>= 1;
    }
    return bits;
  }
  [[nodiscard]] virtual auto pq_dim() const noexcept -> uint32_t
  {
    auto l = pq_len();
    return l > 0 ? (dim() + l - 1) / l : 0;
  }
};

/**
 * @addtogroup pq
 * @{
 */

/**
 * @brief PIMPL wrapper for VQ + PQ codebooks.
 *
 * Internally delegates to either an owning implementation (holds device
 * matrices) or a view implementation (references external device memory).
 *
 * @tparam MathT the type of elements in the codebooks
 */
template <typename MathT>
class vpq_codebooks {
 public:
  using math_type = MathT;

  vpq_codebooks() = default;

  explicit vpq_codebooks(std::unique_ptr<vpq_codebooks_iface<MathT>> impl) : impl_{std::move(impl)}
  {
  }

  vpq_codebooks(const vpq_codebooks&)            = delete;
  vpq_codebooks& operator=(const vpq_codebooks&) = delete;
  vpq_codebooks(vpq_codebooks&&)                 = default;
  vpq_codebooks& operator=(vpq_codebooks&&)      = default;
  ~vpq_codebooks()                               = default;

  /**
   * VQ codebook view, or std::nullopt when no VQ codebook is configured.
   */
  [[nodiscard]] auto vq_code_book() const noexcept
    -> std::optional<raft::device_matrix_view<const math_type, uint32_t, raft::row_major>>
  {
    return impl_->vq_code_book();
  }

  [[nodiscard]] auto pq_code_book() const noexcept
    -> raft::device_matrix_view<const math_type, uint32_t, raft::row_major>
  {
    return impl_->pq_code_book();
  }

  [[nodiscard]] auto dim() const noexcept -> uint32_t { return impl_->dim(); }
  [[nodiscard]] auto vq_n_centers() const noexcept -> uint32_t { return impl_->vq_n_centers(); }
  [[nodiscard]] auto pq_bits() const noexcept -> uint32_t { return impl_->pq_bits(); }
  [[nodiscard]] auto pq_dim() const noexcept -> uint32_t { return impl_->pq_dim(); }
  [[nodiscard]] auto pq_len() const noexcept -> uint32_t { return impl_->pq_len(); }
  [[nodiscard]] auto pq_n_centers() const noexcept -> uint32_t { return impl_->pq_n_centers(); }

 private:
  std::unique_ptr<vpq_codebooks_iface<MathT>> impl_;
};

/**
 * @brief VPQ compressed dataset.
 *
 * Holds a set of VQ + PQ codebooks together with the encoded dataset.
 * Both the codebooks and the encoded data are always owned by this object.
 *
 * @tparam MathT the type of elements in the codebooks
 * @tparam IdxT  type of the vector indices (represent dataset.extent(0))
 */
template <typename MathT, typename IdxT>
class vpq_dataset : public cuvs::neighbors::dataset<IdxT> {
 public:
  using index_type = IdxT;
  using math_type  = MathT;

  /** VQ + PQ codebooks (owning or view). */
  vpq_codebooks<MathT> codebooks;
  /** Encoded (compressed) data [n_rows, encoded_row_length]. */
  raft::device_matrix<uint8_t, IdxT, raft::row_major> data;

  vpq_dataset() = default;

  vpq_dataset(vpq_codebooks<MathT>&& codebooks_in,
              raft::device_matrix<uint8_t, IdxT, raft::row_major>&& data_in)
    : codebooks{std::move(codebooks_in)}, data{std::move(data_in)}
  {
  }

  vpq_dataset(const vpq_dataset&)            = delete;
  vpq_dataset& operator=(const vpq_dataset&) = delete;
  vpq_dataset(vpq_dataset&&)                 = default;
  vpq_dataset& operator=(vpq_dataset&&)      = default;
  ~vpq_dataset() override                    = default;

  [[nodiscard]] index_type n_rows() const noexcept override { return data.extent(0); }
  [[nodiscard]] uint32_t dim() const noexcept override { return codebooks.dim(); }
  [[nodiscard]] bool is_owning() const noexcept override { return true; }

  /**
   * VQ codebook view, or std::nullopt when no VQ codebook is configured.
   */
  [[nodiscard]] auto vq_code_book() const noexcept
    -> std::optional<raft::device_matrix_view<const math_type, uint32_t, raft::row_major>>
  {
    return codebooks.vq_code_book();
  }
  [[nodiscard]] auto pq_code_book() const noexcept
    -> raft::device_matrix_view<const math_type, uint32_t, raft::row_major>
  {
    return codebooks.pq_code_book();
  }
  [[nodiscard]] auto encoded_row_length() const noexcept -> uint32_t { return data.extent(1); }
  [[nodiscard]] auto vq_n_centers() const noexcept -> uint32_t { return codebooks.vq_n_centers(); }
  [[nodiscard]] auto pq_bits() const noexcept -> uint32_t { return codebooks.pq_bits(); }
  [[nodiscard]] auto pq_dim() const noexcept -> uint32_t { return codebooks.pq_dim(); }
  [[nodiscard]] auto pq_len() const noexcept -> uint32_t { return codebooks.pq_len(); }
  [[nodiscard]] auto pq_n_centers() const noexcept -> uint32_t { return codebooks.pq_n_centers(); }
};

template <typename DatasetT>
struct is_vpq_dataset : std::false_type {};

template <typename MathT, typename IdxT>
struct is_vpq_dataset<vpq_dataset<MathT, IdxT>> : std::true_type {};

template <typename DatasetT>
inline constexpr bool is_vpq_dataset_v = is_vpq_dataset<DatasetT>::value;

/** @} */  // end of group pq

}  // namespace cuvs::preprocessing::quantize::pq
