/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>

#include <raft/core/coo_matrix.hpp>
#include <raft/core/csr_matrix.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/util/integer_utils.hpp>

#include <atomic>
#include <concepts>
#include <memory>
#include <type_traits>
#include <utility>

namespace CUVS_EXPORT cuvs {

/************************************************************************************************
 * The core dataset type
 ************************************************************************************************/

/**
 * A spec is compressing if it defines a dictionary: the codebooks, the quantization range, or
 * whatever else is needed to interpret the data. Plain specs declare `std::monostate` instead, so
 * "there is no dictionary" and "the dictionary slot is empty" are the same vocabulary type.
 */
template <typename SpecT>
concept compressed_dataset_spec = requires {
  typename SpecT::dictionary_type;
  typename SpecT::dictionary_view_type;
  typename SpecT::code_type;
} && !std::same_as<typename SpecT::dictionary_type, std::monostate>;

template <typename T, typename IdxT, typename SpecT>
struct dataset {
  using spec_type  = typename SpecT::template apply<T, IdxT>;
  using value_type = typename spec_type::value_type;
  using index_type = typename spec_type::index_type;

  using data_type       = typename spec_type::data_type;
  using view_type       = typename spec_type::view_type;
  using const_view_type = typename spec_type::const_view_type;

  /* The second, optional storage slot; `std::monostate` for an uncompressed dataset. */
  using dictionary_type = typename spec_type::dictionary_type;

  static constexpr bool is_compressed = compressed_dataset_spec<spec_type>;

  explicit dataset(data_type&& data)
    requires(!is_compressed)
    : data_{std::make_shared<data_type>(std::move(data))}
  {
  }

  dataset(data_type&& data, std::shared_ptr<const dictionary_type> dictionary)
    requires(is_compressed)
    : data_{std::make_shared<data_type>(std::move(data))}, dictionary_{std::move(dictionary)}
  {
  }

  /*
   @return A non-owning view of the data, such as mdspan.
  */
  [[nodiscard]] auto data_view() noexcept -> view_type { return spec_type::get_data_view(*data_); }
  [[nodiscard]] auto data_const_view() const noexcept -> const_view_type
  {
    return spec_type::get_data_const_view(*data_);
  }
  [[nodiscard]] auto n_rows() const noexcept -> index_type { return spec_type::get_n_rows(*data_); }
  // NOTE: a compressed dataset may not be able to tell its dimension from the encoded data alone
  [[nodiscard]] auto dim() const noexcept -> uint32_t
  {
    return spec_type::get_dim(*data_, dict_ref());
  }

  /*
   * Check if the caller is the only owner of the data.
   * This is useful to determine if the data can be modified in place without copying.
   */
  [[nodiscard]] auto is_data_unique() const noexcept -> bool
  {
    /*
    The count cannot grow behind our back: only copying a `dataset` increments it and `data_` never
    escapes the class, so no weak_ptr can resurrect a reference. Hence, if it reads one, the sole
    reference is the one we hold and no other thread has anything to copy from. The converse does
    not hold - a concurrent release may not be visible yet - but that only costs the optimization.

    The fence must follow the load: releasing a copy decrements with release ordering, yet
    `use_count()` reads relaxed and shared_ptr pairs the decrement with an acquire only once the
    counter reaches zero. Without the fence we could miss the writes of a copy that another thread
    has already destroyed.
    */
    if (data_.use_count() != 1) { return false; }
    std::atomic_thread_fence(std::memory_order_acquire);
    return true;
  }

  /*
   @return A non-owning view of the dictionary, such as the two codebooks of a VPQ dataset.
  */
  [[nodiscard]] auto dictionary_view() const noexcept
    requires(is_compressed)
  {
    return spec_type::get_dictionary_view(*dictionary_);
  }
  /*
   A dictionary is immutable once trained, so several datasets may encode against the same one.

   @return A shared handle to the dictionary.
  */
  [[nodiscard]] auto share_dictionary() const noexcept -> std::shared_ptr<const dictionary_type>
    requires(is_compressed)
  {
    return dictionary_;
  }

 private:
  std::shared_ptr<data_type> data_;
  /* Either a shared handle to the dictionary, or the empty dictionary itself. */
  [[no_unique_address]] std::conditional_t<is_compressed,
                                           std::shared_ptr<const dictionary_type>,
                                           dictionary_type> dictionary_;

  [[nodiscard]] auto dict_ref() const noexcept -> const dictionary_type&
  {
    if constexpr (is_compressed) {
      return *dictionary_;
    } else {
      return dictionary_;
    }
  }
};

/************************************************************************************************
 * Dataset specifications
 ************************************************************************************************/

/**
 * Empty dataset specification: contains no data, but keeps the dimension of the dataset implement
 * the common interface.
 */
struct empty_spec {
  struct empty_dataset_rep {
    uint32_t dim;
  };

  template <typename T, typename IdxT>
  struct apply {
    using data_type       = empty_dataset_rep;
    using view_type       = empty_dataset_rep;
    using const_view_type = const empty_dataset_rep;
    using value_type      = std::remove_cv_t<T>;
    using index_type      = std::remove_cv_t<IdxT>;
    using dictionary_type = std::monostate;

    [[nodiscard]] static auto get_data_view(data_type& data) noexcept -> view_type { return data; }
    [[nodiscard]] static auto get_data_const_view(const data_type& data) noexcept -> const_view_type
    {
      return data;
    }
    [[nodiscard]] static auto get_n_rows(const data_type& data) noexcept -> index_type { return 0; }
    [[nodiscard]] static auto get_dim(const data_type& data, const dictionary_type&) noexcept
      -> uint32_t
    {
      return data.dim;
    }
  };
};

/**
 * Dense plain/padded dataset implemented via raft::mdarray.
 */
template <typename LayoutPolicy, typename ContainerPolicy>
struct mdarray_spec {
  template <typename T, typename IdxT>
  struct apply {
    /* NOTE: index type != extents type
    The index type can vary depending on the use case; we often may want to store indices in 32-bit
    slots to save memory on large datasets. The extents are always fixed to 64-bit signed integers
    to simplify the conversion and avoid integer overflow issues.
    */
    using data_type = raft::mdarray<T, raft::matrix_extent<int64_t>, LayoutPolicy, ContainerPolicy>;
    using view_type = typename data_type::view_type;
    using const_view_type = typename data_type::const_view_type;
    using value_type      = std::remove_cv_t<T>;
    using index_type      = std::remove_cv_t<IdxT>;
    using dictionary_type = std::monostate;

    [[nodiscard]] static auto get_data_view(data_type& data) noexcept -> view_type
    {
      return data.view();
    }

    [[nodiscard]] static auto get_data_const_view(const data_type& data) noexcept -> const_view_type
    {
      return data.view();
    }
    [[nodiscard]] static auto get_n_rows(const data_type& data) noexcept -> index_type
    {
      return static_cast<index_type>(data.extent(0));
    }
    [[nodiscard]] static auto get_dim(const data_type& data, const dictionary_type&) noexcept
      -> uint32_t
    {
      return static_cast<uint32_t>(data.extent(1));
    }
  };
};

/**
 * Sparse dataset specification implemented via the raft sparse matrix hierarchy. `SparseLayoutT`
 * selects the representation (`csr_layout`, `coo_layout`).
 *
 * NOTE: only the memory space of the container policy is used here; raft's sparse types bind the
 * policy themselves, separately for the values, the offsets and the indices.
 */
template <typename SparseLayoutT, typename ContainerPolicy>
struct sparse_spec {
  template <typename T, typename IdxT>
  struct apply {
    using value_type      = std::remove_cv_t<T>;
    using index_type      = std::remove_cv_t<IdxT>;
    using dictionary_type = std::monostate;

    static constexpr raft::memory_type mem_type = ContainerPolicy::mem_type;
    static_assert(mem_type == raft::memory_type::host || mem_type == raft::memory_type::device,
                  "raft's sparse hierarchy only provides host and device containers");

    using data_type = std::conditional_t<
      mem_type == raft::memory_type::device,
      typename SparseLayoutT::
        template matrix_type<value_type, index_type, true, raft::device_container_policy>,
      typename SparseLayoutT::
        template matrix_type<value_type, index_type, false, raft::host_container_policy>>;
    using view_type = typename data_type::view_type;
    /* NOTE: raft's sparse hierarchy provides no read-only view yet (see the TODO in
    raft/core/sparse_types.hpp); until then, even reading the structure requires a mutable matrix.
    */
    using const_view_type = view_type;

    [[nodiscard]] static auto get_data_view(data_type& data) noexcept -> view_type
    {
      return data.view();
    }

    [[nodiscard]] static auto get_data_const_view(const data_type& data) noexcept -> const_view_type
    {
      return const_cast<data_type&>(data).view();
    }
    [[nodiscard]] static auto get_n_rows(const data_type& data) noexcept -> index_type
    {
      return static_cast<index_type>(const_cast<data_type&>(data).structure_view().get_n_rows());
    }
    [[nodiscard]] static auto get_dim(const data_type& data, const dictionary_type&) noexcept
      -> uint32_t
    {
      return static_cast<uint32_t>(const_cast<data_type&>(data).structure_view().get_n_cols());
    }
  };
};

/**
 * CSR representation: one compressed row per vector of the dataset.
 *
 * NOTE: the row offsets are 64-bit for the same reason the dense extents are: the number of
 * non-zeros is not bounded by the index type. The column indices are feature ids, so they are
 * bounded by the dataset dimension rather than by its size.
 */
struct csr_layout {
  template <typename T, typename IdxT, bool IsDevice, template <typename> typename ContainerPolicy>
  using matrix_type = raft::csr_matrix<T, int64_t, uint32_t, uint64_t, IsDevice, ContainerPolicy>;
};

/**
 * COO representation: one (row, column, value) triple per non-zero.
 *
 * NOTE: unlike CSR, the row component is an index into the dataset rather than an offset into the
 * non-zeros, hence the dataset index type.
 */
struct coo_layout {
  template <typename T, typename IdxT, bool IsDevice, template <typename> typename ContainerPolicy>
  using matrix_type = raft::coo_matrix<T, IdxT, uint32_t, uint64_t, IsDevice, ContainerPolicy>;
};

/**
 * VQ+PQ compressed dataset specification, mirroring the VPQ dataset of cuvs/neighbors/common.hpp:
 * the two codebooks form the dictionary, while the data slot keeps the encoded rows - the VQ label
 * in the row prefix, followed by the packed PQ codes.
 *
 * `BookPolicy` is the container policy of the codebooks (elements of `MathT`) and `CodePolicy` the
 * one of the encoded rows (elements of `uint8_t`); a bound policy cannot be rebound to another
 * element type, hence the two parameters.
 */
template <typename MathT, typename BookPolicy, typename CodePolicy>
struct vpq_spec {
  /* The encoded rows are stored like the rows of any other dense dataset; only how a row is read
  and where the dimension comes from are specific to the compression.
  */
  template <typename IdxT>
  using storage_spec =
    typename mdarray_spec<raft::layout_c_contiguous, CodePolicy>::template apply<uint8_t, IdxT>;

  template <typename T, typename IdxT>
  struct apply : storage_spec<IdxT> {
    /* The dataset stores codes, but it still represents vectors of `T`. */
    using value_type = std::remove_cv_t<T>;
    /* NOTE: the members of a dependent base are not visible to unqualified lookup, so `data_type`
    has to be pulled into this scope to be usable in the signatures below.
    */
    using typename storage_spec<IdxT>::data_type;
    /* The type the codebooks are stored in; the rows themselves are always packed codes. */
    using math_type = MathT;
    using code_type = uint8_t;

    /* [vq_n_centers, dim] */
    using vq_book_type =
      raft::mdarray<math_type, raft::matrix_extent<int64_t>, raft::layout_c_contiguous, BookPolicy>;
    /* [n_books, pq_n_centers, pq_len], where `n_books` is 1 for a codebook shared by all the
    subspaces, or `pq_dim` for one codebook per subspace. A row-major 3d array with `n_books == 1`
    is bit-identical to the flat 2d codebook of the current implementation.
    */
    using pq_book_type =
      raft::mdarray<math_type, raft::extent_3d<int64_t>, raft::layout_c_contiguous, BookPolicy>;

    struct dictionary_type {
      vq_book_type vq_code_book;
      pq_book_type pq_code_book;
    };

    struct dictionary_view_type {
      typename vq_book_type::const_view_type vq_code_book;
      typename pq_book_type::const_view_type pq_code_book;

      /* NOTE: the dimension of the dataset is a property of the VQ codebook: the encoded rows tell
      only how many bytes it took to compress a vector, which the padding makes ambiguous.
      */
      [[nodiscard]] auto dim() const noexcept -> uint32_t
      {
        return static_cast<uint32_t>(vq_code_book.extent(1));
      }
      [[nodiscard]] auto vq_n_centers() const noexcept -> uint32_t
      {
        return static_cast<uint32_t>(vq_code_book.extent(0));
      }
      [[nodiscard]] auto n_books() const noexcept -> uint32_t
      {
        return static_cast<uint32_t>(pq_code_book.extent(0));
      }
      [[nodiscard]] auto pq_n_centers() const noexcept -> uint32_t
      {
        return static_cast<uint32_t>(pq_code_book.extent(1));
      }
      [[nodiscard]] auto pq_len() const noexcept -> uint32_t
      {
        return static_cast<uint32_t>(pq_code_book.extent(2));
      }
      [[nodiscard]] auto pq_bits() const noexcept -> uint32_t
      {
        return static_cast<uint32_t>(std::countr_zero(pq_n_centers()));
      }
      [[nodiscard]] auto pq_dim() const noexcept -> uint32_t
      {
        return raft::div_rounding_up_safe(dim(), pq_len());
      }
      /* Whether every subspace has a codebook of its own, as opposed to sharing a single one. */
      [[nodiscard]] auto per_subspace() const noexcept -> bool { return n_books() > 1; }
    };

    /* NOTE: the data accessors are inherited; only `get_dim` differs from a dense dataset. */
    [[nodiscard]] static auto get_dim(const data_type&, const dictionary_type& dict) noexcept
      -> uint32_t
    {
      return static_cast<uint32_t>(dict.vq_code_book.extent(1));
    }
    [[nodiscard]] static auto get_dictionary_view(const dictionary_type& dict) noexcept
      -> dictionary_view_type
    {
      return {dict.vq_code_book.view(), dict.pq_code_book.view()};
    }
    /* The length of an encoded row in bytes, including the inlined VQ label. */
    [[nodiscard]] static auto get_encoded_row_length(const data_type& data) noexcept -> uint32_t
    {
      return static_cast<uint32_t>(data.extent(1));
    }
  };
};

/**
 * Scalar-quantized dataset specification: every component of a vector becomes one code, so the
 * dictionary is only the range the codes are mapped back onto and the dimension is still a property
 * of the data.
 */
template <typename MathT, typename CodePolicy>
struct sq_spec {
  /* One code per component, so the codes are laid out exactly like a dense dataset. */
  template <typename IdxT>
  using storage_spec =
    typename mdarray_spec<raft::layout_c_contiguous, CodePolicy>::template apply<int8_t, IdxT>;

  template <typename T, typename IdxT>
  struct apply : storage_spec<IdxT> {
    /* The dataset stores codes, but it still represents vectors of `T`. */
    using value_type = std::remove_cv_t<T>;
    /* NOTE: the members of a dependent base are not visible to unqualified lookup, so `data_type`
    has to be pulled into this scope to be usable in the signature below.
    */
    using typename storage_spec<IdxT>::data_type;
    using math_type = MathT;
    using code_type = int8_t;

    /* Nothing to allocate: the whole dictionary is the interval the codes are dequantized into.
    The members are named as in cuvs::preprocessing::quantize::scalar::quantizer.
    */
    struct dictionary_type {
      math_type min_;
      math_type max_;
    };
    using dictionary_view_type = dictionary_type;

    /* NOTE: only the signature differs from the inherited accessor, which cannot be reused because
    it expects an empty dictionary.
    */
    [[nodiscard]] static auto get_dim(const data_type& data, const dictionary_type&) noexcept
      -> uint32_t
    {
      return static_cast<uint32_t>(data.extent(1));
    }
    [[nodiscard]] static auto get_dictionary_view(const dictionary_type& dict) noexcept
      -> dictionary_view_type
    {
      return dict;
    }
  };
};

/************************************************************************************************
 * Convenience aliases
 ************************************************************************************************/

// Empty
template <typename T, typename IdxT>
using empty_dataset = dataset<T, IdxT, empty_spec>;

// Dense
template <typename T, typename IdxT, typename ContainerPolicy>
using contiguous_dataset =
  dataset<T, IdxT, mdarray_spec<raft::layout_c_contiguous, ContainerPolicy>>;
template <typename T, typename IdxT, typename ContainerPolicy>
using padded_dataset = dataset<T, IdxT, mdarray_spec<raft::layout_left_padded<T>, ContainerPolicy>>;
template <typename T, typename IdxT>
using device_contiguous_dataset =
  contiguous_dataset<T, IdxT, raft::device_accessor<raft::device_container_policy<T>>>;
template <typename T, typename IdxT>
using device_padded_dataset =
  padded_dataset<T, IdxT, raft::device_accessor<raft::device_container_policy<T>>>;
template <typename T, typename IdxT>
using host_contiguous_dataset =
  contiguous_dataset<T, IdxT, raft::host_accessor<raft::host_container_policy<T>>>;
template <typename T, typename IdxT>
using host_padded_dataset =
  padded_dataset<T, IdxT, raft::host_accessor<raft::host_container_policy<T>>>;

// Sparse
template <typename T, typename IdxT, typename ContainerPolicy>
using csr_dataset = dataset<T, IdxT, sparse_spec<csr_layout, ContainerPolicy>>;
template <typename T, typename IdxT, typename ContainerPolicy>
using coo_dataset = dataset<T, IdxT, sparse_spec<coo_layout, ContainerPolicy>>;
template <typename T, typename IdxT>
using device_csr_dataset =
  csr_dataset<T, IdxT, raft::device_accessor<raft::device_container_policy<T>>>;
template <typename T, typename IdxT>
using device_coo_dataset =
  coo_dataset<T, IdxT, raft::device_accessor<raft::device_container_policy<T>>>;
template <typename T, typename IdxT>
using host_csr_dataset = csr_dataset<T, IdxT, raft::host_accessor<raft::host_container_policy<T>>>;
template <typename T, typename IdxT>
using host_coo_dataset = coo_dataset<T, IdxT, raft::host_accessor<raft::host_container_policy<T>>>;

// Compressed
template <typename T, typename IdxT, typename MathT, typename BookPolicy, typename CodePolicy>
using vpq_dataset = dataset<T, IdxT, vpq_spec<MathT, BookPolicy, CodePolicy>>;
template <typename T, typename IdxT, typename MathT, typename CodePolicy>
using sq_dataset = dataset<T, IdxT, sq_spec<MathT, CodePolicy>>;
template <typename T, typename IdxT, typename MathT>
using device_vpq_dataset =
  vpq_dataset<T,
              IdxT,
              MathT,
              raft::device_accessor<raft::device_container_policy<MathT>>,
              raft::device_accessor<raft::device_container_policy<uint8_t>>>;
template <typename T, typename IdxT, typename MathT>
using host_vpq_dataset = vpq_dataset<T,
                                     IdxT,
                                     MathT,
                                     raft::host_accessor<raft::host_container_policy<MathT>>,
                                     raft::host_accessor<raft::host_container_policy<uint8_t>>>;
template <typename T, typename IdxT, typename MathT>
using device_sq_dataset =
  sq_dataset<T, IdxT, MathT, raft::device_accessor<raft::device_container_policy<int8_t>>>;
template <typename T, typename IdxT, typename MathT>
using host_sq_dataset =
  sq_dataset<T, IdxT, MathT, raft::host_accessor<raft::host_container_policy<int8_t>>>;

}  // namespace CUVS_EXPORT cuvs
