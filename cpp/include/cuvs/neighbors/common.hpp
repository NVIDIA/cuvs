/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>
#include <raft/core/device_container_policy.hpp>
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_container_policy.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>   // get_device_for_address, copy_matrix
#include <raft/util/integer_utils.hpp>  // rounding up

#include <cuvs/core/bitmap.hpp>
#include <cuvs/core/bitset.hpp>
#include <cuvs/core/export.hpp>
#include <raft/core/detail/macros.hpp>

#include <cuda_fp16.h>

#include <concepts>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#ifdef __cpp_lib_bitops
#include <bit>
#endif

namespace CUVS_EXPORT cuvs {
namespace core {
class bloom_filter;
}
namespace neighbors {
/**
 * @addtogroup cagra_cpp_index_params
 * @{
 */

/* Graph build algo used in cagra and all_neighbors */
enum GRAPH_BUILD_ALGO { BRUTE_FORCE = 0, IVF_PQ = 1, NN_DESCENT = 2, ACE = 3 };

/** Parameters for VPQ compression. */
struct vpq_params {
  /**
   * The bit length of the vector element after compression by PQ.
   *
   * Possible values: [4, 5, 6, 7, 8].
   *
   * Hint: the smaller the 'pq_bits', the smaller the index size and the better the search
   * performance, but the lower the recall.
   */
  uint32_t pq_bits = 8;
  /**
   * The dimensionality of the vector after compression by PQ.
   * When zero, an optimal value is selected using a heuristic.
   *
   * TODO: at the moment `dim` must be a multiple `pq_dim`.
   */
  uint32_t pq_dim = 0;
  /**
   * Vector Quantization (VQ) codebook size - number of "coarse cluster centers".
   * When zero, an optimal value is selected using a heuristic.
   */
  uint32_t vq_n_centers = 0;
  /** The number of iterations searching for kmeans centers (both VQ & PQ phases). */
  uint32_t kmeans_n_iters = 25;
  /**
   * The fraction of data to use during iterative kmeans building (VQ phase).
   * When zero, an optimal value is selected using a heuristic.
   * @deprecated Prefer using `max_train_points_per_vq_cluster` instead.
   */
  double vq_kmeans_trainset_fraction = 0;
  /**
   * The fraction of data to use during iterative kmeans building (PQ phase).
   * When zero, an optimal value is selected using a heuristic.
   * @deprecated Prefer using `max_train_points_per_pq_code` instead.
   */
  double pq_kmeans_trainset_fraction = 0;
  /**
   * Type of k-means algorithm for PQ training.
   * Balanced k-means tends to be faster than regular k-means for PQ training, for
   * problem sets where the number of points per cluster are approximately equal.
   * Regular k-means may be better for skewed cluster distributions.
   */
  cuvs::cluster::kmeans::kmeans_type pq_kmeans_type =
    cuvs::cluster::kmeans::kmeans_type::KMeansBalanced;
  /**
   * The max number of data points to use per PQ code during PQ codebook training. Using more data
   * points per PQ code may increase the quality of PQ codebook but may also increase the build
   * time. We will use `pq_n_centers * max_train_points_per_pq_code` training
   * points to train each PQ codebook.
   */
  uint32_t max_train_points_per_pq_code = 256;
  /**
   * The max number of data points to use per VQ cluster during training.
   */
  uint32_t max_train_points_per_vq_cluster = 1024;
};

/** @} */  // end group cagra_cpp_index_params

/**
 * @defgroup neighbors_index Approximate Nearest Neighbors Types
 * @{
 */

/** The base for approximate KNN index structures. */
struct index {};

/** The base for KNN index parameters. */
struct index_params {
  /** Distance type. */
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded;
  /** The argument used by some distance metrics. */
  float metric_arg = 2.0f;
};

struct search_params {};

/**
 * @brief Strategy for merging indices.
 *
 * This enum is declared separately to avoid namespace pollution when including common.hpp.
 * It provides a generic merge strategy that can be used across different index types.
 */
enum class MergeStrategy {
  /** Merge indices physically by combining their data structures */
  MERGE_STRATEGY_PHYSICAL = 0,
  /** Merge indices logically by creating a composite wrapper */
  MERGE_STRATEGY_LOGICAL = 1
};

/** @} */  // end group neighbors_index

/**
 * @brief Spec-based `dataset` / `dataset_view`.
 *
 * `dataset<T,IdxT,SpecT>` and `dataset_view<T,IdxT,SpecT>` are single generic templates with zero
 * per-kind dispatch inside them: every member is a one-line forward to `spec_type::get_*(...)`,
 * and all kind-specific logic lives in the per-kind Spec structs below (`empty_dataset_spec`,
 * `padded_dataset_spec`, `standard_dataset_spec`, `vpq_dataset_spec`), which `dataset`/
 * `dataset_view` never name or branch on. `dataset` and `dataset_view` are deliberately two
 * independent, non-inheriting types (no shared_ptr, no "sometimes owning" object): `dataset` holds
 * owning storage (mdarray-shaped), `dataset_view` holds the corresponding view storage
 * (mdspan-shaped). The same `get_n_rows`/`get_dim` spec functions serve both, since
 * `raft::mdarray`/`raft::mdspan` both expose `.extent(r)`.
 */

template <typename T, typename IdxT, typename SpecT>
struct dataset;

template <typename T, typename IdxT, typename SpecT>
struct dataset_view;

/**
 * A spec defines a dictionary iff it needs a second storage slot to interpret the data (e.g. PQ
 * codebooks). Non-compressed specs declare `dictionary_type = std::monostate` -- the same
 * vocabulary type for "no dictionary," not just an omitted member -- so `dataset`/`dataset_view`
 * never need to branch on whether the slot exists; they just always have one, sometimes empty.
 */
template <typename SpecT>
concept compressed_dataset_spec = requires {
  typename SpecT::dictionary_type;
  typename SpecT::dictionary_view_type;
} && !std::is_same_v<typename SpecT::dictionary_type, std::monostate>;

namespace detail {

// Default owning/view accessors for public dataset aliases.
template <typename T>
using device_owning_accessor = raft::device_accessor<raft::device_container_policy<T>>;

template <typename T>
using host_owning_accessor = raft::host_accessor<raft::host_container_policy<T>>;

template <typename T>
using device_view_accessor = raft::device_accessor<cuda::std::default_accessor<const T>>;

template <typename T>
using host_view_accessor = raft::host_accessor<cuda::std::default_accessor<const T>>;

/** View accessor paired with an owning dataset accessor (same residency). */
template <typename DataT, typename Accessor>
using dataset_view_accessor_for_owning = std::conditional_t<Accessor::is_device_accessible,
                                                            device_view_accessor<DataT>,
                                                            host_view_accessor<DataT>>;

/** Owning accessor paired with a view accessor (same residency). */
template <typename DataT, typename Accessor>
using dataset_owning_accessor_for_view = std::conditional_t<Accessor::is_device_accessible,
                                                            device_owning_accessor<DataT>,
                                                            host_owning_accessor<DataT>>;

// Accessor here is already device_owning_accessor<DataT> / host_owning_accessor<DataT> at every
// call site -- exactly the container policy raft::device_mdarray/host_mdarray default to for
// element type DataT -- so pass it straight through instead of re-deriving a
// raft::device_matrix/host_matrix from scratch.
template <typename DataT, typename IdxT, typename Accessor>
using dense_owning_matrix =
  raft::mdarray<DataT, raft::matrix_extent<IdxT>, raft::row_major, Accessor>;

template <typename DataT, typename IdxT, typename Accessor>
using dense_view_matrix = raft::mdspan<const DataT,
                                       raft::matrix_extent<IdxT>,
                                       raft::row_major,
                                       dataset_view_accessor_for_owning<DataT, Accessor>>;

template <typename MathT, typename IdxT, typename Accessor>
using vpq_vq_book_matrix =
  raft::mdarray<MathT, raft::matrix_extent<uint32_t>, raft::row_major, Accessor>;

// VPQ codes are always uint8_t regardless of MathT, so retarget the owning accessor's element
// type instead of re-deriving a device/host matrix; residency is still driven by Accessor.
template <typename NewT, typename Accessor>
using owning_accessor_with_value_type = std::conditional_t<Accessor::is_device_accessible,
                                                           device_owning_accessor<NewT>,
                                                           host_owning_accessor<NewT>>;

template <typename IdxT, typename Accessor>
using vpq_data_matrix = raft::mdarray<uint8_t,
                                      raft::matrix_extent<IdxT>,
                                      raft::row_major,
                                      owning_accessor_with_value_type<uint8_t, Accessor>>;

// -----------------------------------------------------------------------------
// empty
// -----------------------------------------------------------------------------

template <typename IdxT>
struct empty_dataset_storage {
  uint32_t suggested_dim{};
  empty_dataset_storage() noexcept = default;
  explicit empty_dataset_storage(uint32_t dim) noexcept : suggested_dim(dim) {}
  [[nodiscard]] auto n_rows() const noexcept -> IdxT { return 0; }
  [[nodiscard]] auto dim() const noexcept -> uint32_t { return suggested_dim; }
};

// -----------------------------------------------------------------------------
// dense row-major (logical dim may differ from row pitch; shared by padded & standard)
// -----------------------------------------------------------------------------

/**
 * Dense row-major owning storage shared by padded and standard dataset specs. Publicly inherits
 * from MatrixT (a `raft::mdarray`) so `view()`/`data_handle()`/`extent()` etc. are reused as-is
 * rather than hand-forwarded; `logical_dim_` is the only state this struct adds.
 *
 * Template parameters:
 * - MatrixT: owning matrix type that stores the payload (host/device matrix).
 * - ViewT: non-owning row-major view type returned by `view()`.
 * - DataT: scalar element type of the dataset payload.
 * - IdxT: index type used for row counts (`n_rows()` return type).
 */
template <typename MatrixT, typename ViewT, typename DataT, typename IdxT>
struct dense_row_major_dataset_owning_storage : public MatrixT {
  uint32_t logical_dim_;

  // MatrixT (mdarray) also has its own stride(size_t); pull it back into scope since declaring
  // our own no-arg stride() below would otherwise hide it entirely (C++ name hiding).
  using MatrixT::stride;

  dense_row_major_dataset_owning_storage(MatrixT&& data, uint32_t logical_dim) noexcept
    : MatrixT{std::move(data)}, logical_dim_{logical_dim}
  {
  }

  [[nodiscard]] auto n_rows() const noexcept -> IdxT { return this->extent(0); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t { return logical_dim_; }
  [[nodiscard]] auto stride() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(this->extent(1));
  }
  // view() and data_handle() are inherited directly from MatrixT (raft::mdarray); no hand-written
  // forwarding needed since MatrixT::view() const already returns exactly ViewT.
};

template <typename ViewT, typename DataT, typename IdxT>
struct dense_row_major_dataset_view_storage : public ViewT {
  uint32_t logical_dim_;

  // ViewT (mdspan) also has its own stride(size_t); pull it back into scope since declaring our
  // own no-arg stride() below would otherwise hide it entirely (C++ name hiding), and the body of
  // that stride() itself needs to call the inherited one.
  using ViewT::stride;

  dense_row_major_dataset_view_storage() noexcept = default;

  explicit dense_row_major_dataset_view_storage(ViewT v) noexcept
    : ViewT(v), logical_dim_(static_cast<uint32_t>(v.extent(1)))
  {
  }

  dense_row_major_dataset_view_storage(ViewT v, uint32_t logical_dim) noexcept
    : ViewT(v), logical_dim_(logical_dim)
  {
  }

  [[nodiscard]] auto n_rows() const noexcept -> IdxT { return this->extent(0); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t { return logical_dim_; }
  [[nodiscard]] auto stride() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(ViewT::stride(0) > 0 ? ViewT::stride(0) : this->extent(1));
  }
  // ViewT (mdspan) has no view() of its own -- it already *is* the view -- so this shrinks to a
  // plain upcast instead of reaching into a wrapped field.
  [[nodiscard]] auto view() const noexcept -> ViewT { return *this; }
};

/** Spec-side implementation shared by `padded_dataset_spec`/`standard_dataset_spec`; those two
 * stay distinct top-level types (identical bodies) purely so classification traits can tell them
 * apart -- exactly mirroring today's `padded_dataset_container`/`standard_dataset_container`,
 * which are likewise two differently-named tags over one shared storage implementation. */
template <typename ContainerPolicy>
struct dense_dataset_spec_impl {
  template <typename T, typename IdxT>
  struct apply {
    using value_type           = std::remove_cv_t<T>;
    using index_type           = std::remove_cv_t<IdxT>;
    using MatrixT              = dense_owning_matrix<T, IdxT, ContainerPolicy>;
    using ViewT                = dense_view_matrix<T, IdxT, ContainerPolicy>;
    using data_type            = dense_row_major_dataset_owning_storage<MatrixT, ViewT, T, IdxT>;
    using view_type            = dense_row_major_dataset_view_storage<ViewT, T, IdxT>;
    using dictionary_type      = std::monostate;
    using dictionary_view_type = std::monostate;

    [[nodiscard]] static auto get_data_view(data_type const& data) noexcept -> view_type
    {
      return view_type(data.view(), data.dim());
    }
    template <typename AnyDatasetOrView>
    [[nodiscard]] static auto get_n_rows(AnyDatasetOrView const& data) noexcept -> index_type
    {
      return data.n_rows();
    }
    template <typename AnyDatasetOrView>
    [[nodiscard]] static auto get_dim(AnyDatasetOrView const& data, dictionary_type const&) noexcept
      -> uint32_t
    {
      return data.dim();
    }
    [[nodiscard]] static auto get_dictionary_view(dictionary_type const&) noexcept
      -> dictionary_view_type
    {
      return {};
    }
  };
};

}  // namespace detail

// -----------------------------------------------------------------------------
// Public specs -- the only place per-kind logic lives.
// -----------------------------------------------------------------------------

template <typename Accessor>
struct empty_dataset_spec {
  using accessor_type = Accessor;

  template <typename T, typename IdxT>
  struct apply {
    using value_type           = std::remove_cv_t<T>;
    using index_type           = std::remove_cv_t<IdxT>;
    using data_type            = detail::empty_dataset_storage<IdxT>;
    using view_type            = detail::empty_dataset_storage<IdxT>;
    using dictionary_type      = std::monostate;
    using dictionary_view_type = std::monostate;

    [[nodiscard]] static auto get_data_view(data_type const& data) noexcept -> view_type
    {
      return data;
    }
    [[nodiscard]] static auto get_n_rows(data_type const& data) noexcept -> index_type
    {
      return static_cast<index_type>(data.n_rows());
    }
    [[nodiscard]] static auto get_dim(data_type const& data, dictionary_type const&) noexcept
      -> uint32_t
    {
      return data.dim();
    }
    [[nodiscard]] static auto get_dictionary_view(dictionary_type const&) noexcept
      -> dictionary_view_type
    {
      return {};
    }
  };
};

template <typename ContainerPolicy>
struct padded_dataset_spec {
  using accessor_type = ContainerPolicy;
  template <typename T, typename IdxT>
  struct apply : detail::dense_dataset_spec_impl<ContainerPolicy>::template apply<T, IdxT> {};
};

template <typename ContainerPolicy>
struct standard_dataset_spec {
  using accessor_type = ContainerPolicy;
  template <typename T, typename IdxT>
  struct apply : detail::dense_dataset_spec_impl<ContainerPolicy>::template apply<T, IdxT> {};
};

/** `Accessor` drives both codebook and code residency, mirroring today's
 * single-`Accessor`-per-VPQ-dataset design (`vpq_vq_book_matrix`/`vpq_data_matrix` are both keyed
 * off one `Accessor`). Data = encoded rows (uint8_t codes); dictionary = {vq_code_book,
 * pq_code_book}. Inlined directly (unlike padded/standard) since no second tag shares this body. */
template <typename MathT, typename Accessor>
struct vpq_dataset_spec {
  using accessor_type = Accessor;

  template <typename T, typename IdxT>
  struct apply {
    using value_type = std::remove_cv_t<T>;
    using index_type = std::remove_cv_t<IdxT>;
    using math_type  = MathT;

    using data_type = detail::vpq_data_matrix<IdxT, Accessor>;
    using view_type = raft::mdspan<const uint8_t,
                                   raft::matrix_extent<IdxT>,
                                   raft::row_major,
                                   detail::dataset_view_accessor_for_owning<uint8_t, Accessor>>;

    using vq_book_type = detail::vpq_vq_book_matrix<MathT, IdxT, Accessor>;
    using pq_book_type = detail::vpq_vq_book_matrix<MathT, IdxT, Accessor>;

    struct dictionary_type {
      vq_book_type vq_code_book;
      pq_book_type pq_code_book;
    };
    struct dictionary_view_type {
      typename vq_book_type::const_view_type vq_code_book;
      typename pq_book_type::const_view_type pq_code_book;

      [[nodiscard]] auto dim() const noexcept -> uint32_t
      {
        return static_cast<uint32_t>(vq_code_book.extent(1));
      }
      [[nodiscard]] auto vq_n_centers() const noexcept -> uint32_t
      {
        return static_cast<uint32_t>(vq_code_book.extent(0));
      }
      [[nodiscard]] auto pq_n_centers() const noexcept -> uint32_t
      {
        return static_cast<uint32_t>(pq_code_book.extent(0));
      }
      [[nodiscard]] auto pq_len() const noexcept -> uint32_t
      {
        return static_cast<uint32_t>(pq_code_book.extent(1));
      }
      [[nodiscard]] auto pq_bits() const noexcept -> uint32_t
      {
        auto pq_width = pq_n_centers();
#ifdef __cpp_lib_bitops
        return std::countr_zero(pq_width);
#else
        uint32_t bits = 0;
        while (pq_width > 1) {
          bits++;
          pq_width >>= 1;
        }
        return bits;
#endif
      }
      [[nodiscard]] auto pq_dim() const noexcept -> uint32_t
      {
        return raft::div_rounding_up_unsafe(dim(), pq_len());
      }
    };

    [[nodiscard]] static auto get_data_view(data_type const& data) noexcept -> view_type
    {
      return data.view();
    }
    template <typename AnyExtentShaped>
    [[nodiscard]] static auto get_n_rows(AnyExtentShaped const& data) noexcept -> index_type
    {
      return static_cast<index_type>(data.extent(0));
    }
    /* get_dim differs from a plain dense dataset: the dimension comes from the VQ codebook, not
    the encoded rows (row padding makes the encoded-row width ambiguous as a dimension). */
    template <typename AnyData>
    [[nodiscard]] static auto get_dim(AnyData const&, dictionary_type const& dict) noexcept
      -> uint32_t
    {
      return static_cast<uint32_t>(dict.vq_code_book.extent(1));
    }
    template <typename AnyData>
    [[nodiscard]] static auto get_dim(AnyData const&, dictionary_view_type const& dict) noexcept
      -> uint32_t
    {
      return dict.dim();
    }
    [[nodiscard]] static auto get_dictionary_view(dictionary_type const& dict) noexcept
      -> dictionary_view_type
    {
      return {dict.vq_code_book.view(), dict.pq_code_book.view()};
    }
    [[nodiscard]] static auto get_encoded_row_length(data_type const& data) noexcept -> uint32_t
    {
      return static_cast<uint32_t>(data.extent(1));
    }
    [[nodiscard]] static auto get_encoded_row_length(view_type const& data) noexcept -> uint32_t
    {
      return static_cast<uint32_t>(data.extent(1));
    }
  };
};

// -----------------------------------------------------------------------------
// dataset / dataset_view
// -----------------------------------------------------------------------------

/** Owning dataset: value-held storage (no shared_ptr -- exclusive ownership). Every member is a
 * one-line forward to `spec_type::get_*`; all per-kind logic lives in `SpecT`, never inside this
 * struct. */
template <typename T, typename IdxT, typename SpecT>
struct dataset {
  using spec_type       = typename SpecT::template apply<T, IdxT>;
  using value_type      = typename spec_type::value_type;
  using index_type      = typename spec_type::index_type;
  using data_type       = typename spec_type::data_type;
  using dictionary_type = typename spec_type::dictionary_type;

  // Non-compressed: forward constructor args straight to data_type's own constructor (e.g.
  // (MatrixT&&, uint32_t logical_dim) for dense, (uint32_t dim) for empty) -- preserves today's
  // construction call sites unchanged.
  template <typename... Args>
  explicit dataset(Args&&... args)
    requires(!compressed_dataset_spec<spec_type> && std::is_constructible_v<data_type, Args...>)
    : data_(std::forward<Args>(args)...), dictionary_{}
  {
  }

  // Compressed: data (codes) and dictionary (codebooks) constructed independently.
  dataset(data_type&& data, dictionary_type&& dictionary)
    requires(compressed_dataset_spec<spec_type>)
    : data_(std::move(data)), dictionary_(std::move(dictionary))
  {
  }

  [[nodiscard]] auto n_rows() const noexcept -> index_type { return spec_type::get_n_rows(data_); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t
  {
    return spec_type::get_dim(data_, dictionary_);
  }
  [[nodiscard]] auto data_view() const noexcept { return spec_type::get_data_view(data_); }
  [[nodiscard]] auto dictionary_view() const noexcept
  {
    return spec_type::get_dictionary_view(dictionary_);
  }

  [[nodiscard]] auto as_dataset_view() const noexcept -> dataset_view<T, IdxT, SpecT>
  {
    return dataset_view<T, IdxT, SpecT>(data_view(), dictionary_view());
  }

  // Move the owning storage out (e.g. to reuse an already-encoded codes matrix while rebuilding
  // only the dictionary at a different math_type, as in VPQ's f32->f16 conversion path).
  [[nodiscard]] auto release_data() noexcept -> data_type&& { return std::move(data_); }
  [[nodiscard]] auto release_dictionary() noexcept -> dictionary_type&&
  {
    return std::move(dictionary_);
  }

  // Dictionary-derived helpers (VPQ: encoded_row_length/vq_n_centers/pq_bits/pq_dim/pq_len/
  // pq_n_centers) forward through dictionary_view() when the dictionary provides them; SFINAE'd
  // away for kinds without a dictionary, matching today's VPQ-only surface without dataset<>
  // itself branching on which kind it is.
  [[nodiscard]] auto encoded_row_length() const noexcept
    requires requires(data_type const& d) { spec_type::get_encoded_row_length(d); }
  {
    return spec_type::get_encoded_row_length(data_);
  }
  [[nodiscard]] auto vq_n_centers() const noexcept
    requires requires(decltype(dictionary_view()) const& d) { d.vq_n_centers(); }
  {
    return dictionary_view().vq_n_centers();
  }
  [[nodiscard]] auto pq_n_centers() const noexcept
    requires requires(decltype(dictionary_view()) const& d) { d.pq_n_centers(); }
  {
    return dictionary_view().pq_n_centers();
  }
  [[nodiscard]] auto pq_len() const noexcept
    requires requires(decltype(dictionary_view()) const& d) { d.pq_len(); }
  {
    return dictionary_view().pq_len();
  }
  [[nodiscard]] auto pq_bits() const noexcept
    requires requires(decltype(dictionary_view()) const& d) { d.pq_bits(); }
  {
    return dictionary_view().pq_bits();
  }
  [[nodiscard]] auto pq_dim() const noexcept
    requires requires(decltype(dictionary_view()) const& d) { d.pq_dim(); }
  {
    return dictionary_view().pq_dim();
  }

 private:
  data_type data_;
  [[no_unique_address]] dictionary_type dictionary_;
};

/** Non-owning dataset view: holds only view-shaped storage (mdspan, not mdarray). Deliberately not
 * derived from `dataset` -- a view type holds "all view state" with no inheritance and no shared
 * ownership tying it to the owning type. Reuses the same `get_n_rows`/`get_dim` spec functions as
 * `dataset`, fed view-shaped arguments instead of owning ones. */
template <typename T, typename IdxT, typename SpecT>
struct dataset_view {
  using spec_type            = typename SpecT::template apply<T, IdxT>;
  using value_type           = typename spec_type::value_type;
  using index_type           = typename spec_type::index_type;
  using view_type            = typename spec_type::view_type;
  using dictionary_view_type = typename spec_type::dictionary_view_type;

  dataset_view() noexcept = default;

  // Already-constructed (view_type, dictionary_view_type) pair -- the shape `as_dataset_view()`
  // always constructs with, for every kind (dictionary_view_type is std::monostate and
  // defaults away when there's no dictionary). Not a template, so it's preferred over the
  // forwarding constructor below whenever both could apply.
  dataset_view(view_type data_view, dictionary_view_type dictionary_view = {}) noexcept
    : data_view_{data_view}, dictionary_view_{dictionary_view}
  {
  }

  // Forward raw constructor args straight to view_type's own constructor (e.g. (ViewT, uint32_t
  // logical_dim) for dense, (uint32_t dim) for empty) -- preserves today's direct-construction
  // call sites (e.g. `device_padded_dataset_view<T,IdxT>(raw_mdspan, dim)`) unchanged. `view_type`
  // is never itself constructible from `(view_type, dictionary_view_type)` (its own constructors
  // only take mdspan-shaped args), so this and the plain constructor above never both match the
  // same call -- no ambiguity.
  template <typename... Args>
  explicit dataset_view(Args&&... args)
    requires(std::is_constructible_v<view_type, Args...>)
    : data_view_(std::forward<Args>(args)...), dictionary_view_{}
  {
  }

  [[nodiscard]] auto n_rows() const noexcept -> index_type
  {
    return spec_type::get_n_rows(data_view_);
  }
  [[nodiscard]] auto dim() const noexcept -> uint32_t
  {
    return spec_type::get_dim(data_view_, dictionary_view_);
  }
  [[nodiscard]] auto data_view() const noexcept -> view_type { return data_view_; }
  [[nodiscard]] auto dictionary_view() const noexcept -> dictionary_view_type
  {
    return dictionary_view_;
  }

  // See dataset<>'s equivalent block: VPQ-only helpers, SFINAE'd away for kinds without a
  // dictionary.
  [[nodiscard]] auto encoded_row_length() const noexcept
    requires requires(view_type const& d) { spec_type::get_encoded_row_length(d); }
  {
    return spec_type::get_encoded_row_length(data_view_);
  }
  [[nodiscard]] auto vq_n_centers() const noexcept
    requires requires(dictionary_view_type const& d) { d.vq_n_centers(); }
  {
    return dictionary_view_.vq_n_centers();
  }
  [[nodiscard]] auto pq_n_centers() const noexcept
    requires requires(dictionary_view_type const& d) { d.pq_n_centers(); }
  {
    return dictionary_view_.pq_n_centers();
  }
  [[nodiscard]] auto pq_len() const noexcept
    requires requires(dictionary_view_type const& d) { d.pq_len(); }
  {
    return dictionary_view_.pq_len();
  }
  [[nodiscard]] auto pq_bits() const noexcept
    requires requires(dictionary_view_type const& d) { d.pq_bits(); }
  {
    return dictionary_view_.pq_bits();
  }
  [[nodiscard]] auto pq_dim() const noexcept
    requires requires(dictionary_view_type const& d) { d.pq_dim(); }
  {
    return dictionary_view_.pq_dim();
  }

 private:
  view_type data_view_{};
  [[no_unique_address]] dictionary_view_type dictionary_view_{};
};

/**
 * @brief Aliases for concrete `dataset` / `dataset_view` layouts.
 */
template <typename IdxT>
using device_empty_dataset =
  dataset<void, IdxT, empty_dataset_spec<detail::device_view_accessor<char>>>;

template <typename IdxT>
using device_empty_dataset_view =
  dataset_view<void, IdxT, empty_dataset_spec<detail::device_view_accessor<char>>>;

template <typename IdxT>
using host_empty_dataset =
  dataset<void, IdxT, empty_dataset_spec<detail::host_view_accessor<char>>>;

template <typename IdxT>
using host_empty_dataset_view =
  dataset_view<void, IdxT, empty_dataset_spec<detail::host_view_accessor<char>>>;

template <typename DataT, typename IdxT>
using device_padded_dataset =
  dataset<DataT, IdxT, padded_dataset_spec<detail::device_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using device_padded_dataset_view =
  dataset_view<DataT, IdxT, padded_dataset_spec<detail::device_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using host_padded_dataset =
  dataset<DataT, IdxT, padded_dataset_spec<detail::host_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using host_padded_dataset_view =
  dataset_view<DataT, IdxT, padded_dataset_spec<detail::host_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using device_standard_dataset =
  dataset<DataT, IdxT, standard_dataset_spec<detail::device_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using device_standard_dataset_view =
  dataset_view<DataT, IdxT, standard_dataset_spec<detail::device_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using host_standard_dataset =
  dataset<DataT, IdxT, standard_dataset_spec<detail::host_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using host_standard_dataset_view =
  dataset_view<DataT, IdxT, standard_dataset_spec<detail::host_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using device_vpq_dataset =
  dataset<DataT, IdxT, vpq_dataset_spec<DataT, detail::device_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using device_vpq_dataset_view =
  dataset_view<DataT, IdxT, vpq_dataset_spec<DataT, detail::device_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using host_vpq_dataset =
  dataset<DataT, IdxT, vpq_dataset_spec<DataT, detail::host_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using host_vpq_dataset_view =
  dataset_view<DataT, IdxT, vpq_dataset_spec<DataT, detail::host_owning_accessor<DataT>>>;

// Maps a dataset view type to its owning (allocating) dataset counterpart. Trivial and total under
// the Spec design: the owning type for `dataset_view<T,IdxT,SpecT>` is always
// `dataset<T,IdxT,SpecT>`
// -- no per-kind specialization table needed (unlike the old Container-tagged design).
template <typename DatasetViewT>
struct owning_dataset_for_view;

template <typename T, typename IdxT, typename SpecT>
struct owning_dataset_for_view<dataset_view<T, IdxT, SpecT>> {
  using type = dataset<T, IdxT, SpecT>;
};

template <typename DatasetViewT>
using owning_dataset_for_view_t = typename owning_dataset_for_view<DatasetViewT>::type;

// -----------------------------------------------------------------------------
// Spec-kind classification (all derived from SpecT; dataset/dataset_view never branch on kind).
// -----------------------------------------------------------------------------

template <typename SpecT>
struct is_empty_spec : std::false_type {};
template <typename Accessor>
struct is_empty_spec<empty_dataset_spec<Accessor>> : std::true_type {};
template <typename SpecT>
inline constexpr bool is_empty_spec_v = is_empty_spec<SpecT>::value;

template <typename SpecT>
struct is_padded_spec : std::false_type {};
template <typename ContainerPolicy>
struct is_padded_spec<padded_dataset_spec<ContainerPolicy>> : std::true_type {};
template <typename SpecT>
inline constexpr bool is_padded_spec_v = is_padded_spec<SpecT>::value;

template <typename SpecT>
struct is_standard_spec : std::false_type {};
template <typename ContainerPolicy>
struct is_standard_spec<standard_dataset_spec<ContainerPolicy>> : std::true_type {};
template <typename SpecT>
inline constexpr bool is_standard_spec_v = is_standard_spec<SpecT>::value;

template <typename SpecT>
struct is_vpq_spec : std::false_type {};
template <typename MathT, typename Accessor>
struct is_vpq_spec<vpq_dataset_spec<MathT, Accessor>> : std::true_type {};
template <typename SpecT>
inline constexpr bool is_vpq_spec_v = is_vpq_spec<SpecT>::value;

template <typename SpecT>
struct vpq_spec_math_type {};
template <typename MathT, typename Accessor>
struct vpq_spec_math_type<vpq_dataset_spec<MathT, Accessor>> {
  using type = MathT;
};
template <typename SpecT>
using vpq_spec_math_type_t = typename vpq_spec_math_type<SpecT>::type;

/** Owning-side kind traits (mirror today's `is_padded_dataset_v`/`is_standard_dataset_v`/
 * `is_vpq_dataset_v`, used for SFINAE overload selection in factory.cuh/compute_distance_vpq.hpp).
 */
template <typename DatasetT>
struct is_padded_dataset : std::false_type {};
template <typename T, typename IdxT, typename SpecT>
struct is_padded_dataset<dataset<T, IdxT, SpecT>> : std::bool_constant<is_padded_spec_v<SpecT>> {};
template <typename T, typename IdxT, typename SpecT>
struct is_padded_dataset<dataset_view<T, IdxT, SpecT>>
  : std::bool_constant<is_padded_spec_v<SpecT>> {};
template <typename DatasetT>
inline constexpr bool is_padded_dataset_v = is_padded_dataset<DatasetT>::value;

template <typename DatasetT>
struct is_standard_dataset : std::false_type {};
template <typename T, typename IdxT, typename SpecT>
struct is_standard_dataset<dataset<T, IdxT, SpecT>>
  : std::bool_constant<is_standard_spec_v<SpecT>> {};
template <typename T, typename IdxT, typename SpecT>
struct is_standard_dataset<dataset_view<T, IdxT, SpecT>>
  : std::bool_constant<is_standard_spec_v<SpecT>> {};
template <typename DatasetT>
inline constexpr bool is_standard_dataset_v = is_standard_dataset<DatasetT>::value;

template <typename DatasetT>
struct is_vpq_dataset : std::false_type {};
template <typename T, typename IdxT, typename SpecT>
struct is_vpq_dataset<dataset<T, IdxT, SpecT>> : std::bool_constant<is_vpq_spec_v<SpecT>> {};
template <typename DatasetT>
inline constexpr bool is_vpq_dataset_v = is_vpq_dataset<DatasetT>::value;

// -----------------------------------------------------------------------------
// Dataset view compile-time classification (replaces runtime std::variant dispatch).
// -----------------------------------------------------------------------------

/** Any non-owning dataset view exposing row count and logical dimension. */
template <typename V, typename IdxT = int64_t>
concept ann_dataset_view = requires(V const& v) {
  { v.n_rows() } -> std::convertible_to<IdxT>;
  { v.dim() } -> std::convertible_to<uint32_t>;
};

enum class dataset_view_kind {
  // TODO(removal): Remove `unknown` once all deprecated host_matrix_view / device_matrix_view /
  // mdspan overloads are deleted. It exists solely so that overload resolution on the deprecated
  // build(host_matrix_view) / build(device_matrix_view) shims does not cause a hard error when
  // the compiler evaluates is_host/device_dataset_view_v for a plain mdspan type.
  unknown,
  empty,
  padded,
  standard,
  vpq_f16,
  vpq_f32,
};

template <typename V>
using dataset_view_type_t = std::remove_cvref_t<V>;

/** Primary template returns `unknown` so traits safely return `false` for non-dataset-view types.
 */
template <typename V>
struct dataset_view_kind_of {
  static constexpr dataset_view_kind value = dataset_view_kind::unknown;
};

template <typename T, typename IdxT, typename SpecT>
struct dataset_view_kind_of<dataset_view<T, IdxT, SpecT>> {
  static constexpr dataset_view_kind value = []() constexpr {
    if constexpr (is_empty_spec_v<SpecT>) {
      return dataset_view_kind::empty;
    } else if constexpr (is_padded_spec_v<SpecT>) {
      return dataset_view_kind::padded;
    } else if constexpr (is_standard_spec_v<SpecT>) {
      return dataset_view_kind::standard;
    } else if constexpr (is_vpq_spec_v<SpecT>) {
      static_assert(std::is_same_v<vpq_spec_math_type_t<SpecT>, half> ||
                      std::is_same_v<vpq_spec_math_type_t<SpecT>, float>,
                    "VPQ dataset_view_kind_of expects MathT to be half or float");
      return std::is_same_v<vpq_spec_math_type_t<SpecT>, half> ? dataset_view_kind::vpq_f16
                                                               : dataset_view_kind::vpq_f32;
    } else {
      return dataset_view_kind::unknown;
    }
  }();
};

/** True when the dataset view accessor is device-accessible. */
template <typename V>
struct dataset_view_is_device_accessible : std::false_type {};

template <typename T, typename IdxT, typename SpecT>
struct dataset_view_is_device_accessible<dataset_view<T, IdxT, SpecT>>
  : std::bool_constant<SpecT::accessor_type::is_device_accessible> {};

template <typename V>
inline constexpr bool dataset_view_is_device_accessible_v =
  dataset_view_is_device_accessible<dataset_view_type_t<V>>::value;

template <typename V>
inline constexpr dataset_view_kind dataset_view_kind_v =
  dataset_view_kind_of<dataset_view_type_t<V>>::value;

template <typename V>
inline constexpr bool is_device_empty_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::empty && dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_host_empty_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::empty && !dataset_view_is_device_accessible_v<V>;

/** True for any empty dataset view (device or host). */
template <typename V>
inline constexpr bool is_empty_dataset_view_v =
  is_device_empty_dataset_view_v<V> || is_host_empty_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_device_padded_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::padded && dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_host_padded_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::padded && !dataset_view_is_device_accessible_v<V>;

/** True for either `device_padded_dataset_view` or `host_padded_dataset_view`. */
template <typename V>
inline constexpr bool is_padded_dataset_view_v =
  is_device_padded_dataset_view_v<V> || is_host_padded_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_device_standard_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::standard && dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_host_standard_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::standard && !dataset_view_is_device_accessible_v<V>;

/** True for either `device_standard_dataset_view` or `host_standard_dataset_view`. */
template <typename V>
inline constexpr bool is_standard_dataset_view_v =
  is_device_standard_dataset_view_v<V> || is_host_standard_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_device_vpq_f16_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::vpq_f16 && dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_host_vpq_f16_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::vpq_f16 && !dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_vpq_f16_dataset_view_v =
  is_device_vpq_f16_dataset_view_v<V> || is_host_vpq_f16_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_device_vpq_f32_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::vpq_f32 && dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_host_vpq_f32_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::vpq_f32 && !dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_vpq_f32_dataset_view_v =
  is_device_vpq_f32_dataset_view_v<V> || is_host_vpq_f32_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_device_vpq_dataset_view_v =
  is_device_vpq_f16_dataset_view_v<V> || is_device_vpq_f32_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_host_vpq_dataset_view_v =
  is_host_vpq_f16_dataset_view_v<V> || is_host_vpq_f32_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_vpq_dataset_view_v =
  is_device_vpq_dataset_view_v<V> || is_host_vpq_dataset_view_v<V>;

/** True for any device-resident dataset view. */
template <typename V>
inline constexpr bool is_device_dataset_view_v =
  dataset_view_kind_v<V> != dataset_view_kind::unknown && dataset_view_is_device_accessible_v<V>;

/** True for any host-resident dataset view. */
template <typename V>
inline constexpr bool is_host_dataset_view_v =
  dataset_view_kind_v<V> != dataset_view_kind::unknown && !dataset_view_is_device_accessible_v<V>;

/**
 * True when a host view `H` and device view `D` represent the same storage kind and differ
 * only in residency (host vs. device). Used by host/device conversion helpers.
 */
template <typename HostViewT, typename DeviceViewT>
inline constexpr bool compatible_host_device_dataset_views_v =
  is_host_dataset_view_v<HostViewT> && is_device_dataset_view_v<DeviceViewT> &&
  (dataset_view_kind_v<HostViewT> == dataset_view_kind_v<DeviceViewT>);

/**
 * Generic accessor retargeting while preserving the dataset tag/layout and value/index types:
 * `dataset<T, IdxT, SpecT<..., OldAccessor>>      -> dataset<T, IdxT, SpecT<..., NewAccessor>>`
 * `dataset_view<T, IdxT, SpecT<..., OldAccessor>> -> dataset_view<T, IdxT, SpecT<...,
 * NewAccessor>>`
 */
template <typename DatasetLikeT, typename NewAccessor>
struct with_accessor;

template <typename T, typename IdxT, typename NewAccessor>
struct with_accessor<dataset<T, IdxT, empty_dataset_spec<NewAccessor>>, NewAccessor> {
  using type = dataset<T, IdxT, empty_dataset_spec<NewAccessor>>;
};

template <typename T, typename IdxT, typename OldAccessor, typename NewAccessor>
struct with_accessor<dataset<T, IdxT, padded_dataset_spec<OldAccessor>>, NewAccessor> {
  using type = dataset<T, IdxT, padded_dataset_spec<NewAccessor>>;
};

template <typename T, typename IdxT, typename OldAccessor, typename NewAccessor>
struct with_accessor<dataset<T, IdxT, standard_dataset_spec<OldAccessor>>, NewAccessor> {
  using type = dataset<T, IdxT, standard_dataset_spec<NewAccessor>>;
};

template <typename T, typename IdxT, typename MathT, typename OldAccessor, typename NewAccessor>
struct with_accessor<dataset<T, IdxT, vpq_dataset_spec<MathT, OldAccessor>>, NewAccessor> {
  using type = dataset<T, IdxT, vpq_dataset_spec<MathT, NewAccessor>>;
};

template <typename T, typename IdxT, typename OldAccessor, typename NewAccessor>
struct with_accessor<dataset_view<T, IdxT, empty_dataset_spec<OldAccessor>>, NewAccessor> {
  using type = dataset_view<T, IdxT, empty_dataset_spec<NewAccessor>>;
};

template <typename T, typename IdxT, typename OldAccessor, typename NewAccessor>
struct with_accessor<dataset_view<T, IdxT, padded_dataset_spec<OldAccessor>>, NewAccessor> {
  using type = dataset_view<T, IdxT, padded_dataset_spec<NewAccessor>>;
};

template <typename T, typename IdxT, typename OldAccessor, typename NewAccessor>
struct with_accessor<dataset_view<T, IdxT, standard_dataset_spec<OldAccessor>>, NewAccessor> {
  using type = dataset_view<T, IdxT, standard_dataset_spec<NewAccessor>>;
};

template <typename T, typename IdxT, typename MathT, typename OldAccessor, typename NewAccessor>
struct with_accessor<dataset_view<T, IdxT, vpq_dataset_spec<MathT, OldAccessor>>, NewAccessor> {
  using type = dataset_view<T, IdxT, vpq_dataset_spec<MathT, NewAccessor>>;
};

template <typename DatasetLikeT, typename NewAccessor>
using with_accessor_t =
  typename with_accessor<dataset_view_type_t<DatasetLikeT>, NewAccessor>::type;

/** Map any host accessor to its device counterpart (same payload policy). */
template <typename Accessor>
struct to_device_accessor {
  using type = Accessor;
};

template <typename T>
struct to_device_accessor<detail::host_view_accessor<T>> {
  using type = detail::device_view_accessor<T>;
};

template <typename T>
struct to_device_accessor<detail::host_owning_accessor<T>> {
  using type = detail::device_owning_accessor<T>;
};

template <typename Accessor>
using to_device_accessor_t = typename to_device_accessor<Accessor>::type;

/** Maps a host dataset view type to its device-resident counterpart. */
template <typename HostViewT>
struct device_counterpart;

template <typename T, typename IdxT, typename SpecT>
struct device_counterpart<dataset_view<T, IdxT, SpecT>> {
  using type = with_accessor_t<dataset_view<T, IdxT, SpecT>,
                               to_device_accessor_t<typename SpecT::accessor_type>>;
};

template <typename HostViewT>
using device_counterpart_t = typename device_counterpart<dataset_view_type_t<HostViewT>>::type;

/** True for device padded or standard views accepted by dense graph build (VPQ excluded). */
template <typename V>
inline constexpr bool is_dense_row_major_device_dataset_view_v =
  is_device_padded_dataset_view_v<V> || is_device_standard_dataset_view_v<V>;

/** True for host or device padded/standard views (iterative graph build; VPQ excluded). */
template <typename V>
inline constexpr bool is_dense_row_major_dataset_view_v =
  is_padded_dataset_view_v<V> || is_standard_dataset_view_v<V>;

/** Element type `T` for `cagra::build(res, params, dataset_view)` (deduced, not a template arg).
 * Trivial under the Spec design: every `dataset_view<T,IdxT,SpecT>` already carries `T` directly.
 */
template <typename V>
using cagra_view_element_type_t = typename dataset_view_type_t<V>::value_type;

// -----------------------------------------------------------------------------
// CAGRA row width in elements (same for make_device_padded_dataset* and index layout checks).
// -----------------------------------------------------------------------------

/**
 * @brief Required row width in elements for CAGRA: minimum leading dimension (LDA) per row for the
 *        default per-row byte alignment (16 bytes, combined with `sizeof` element type), given
 *        `logical_columns` feature columns.
 */
[[nodiscard]] inline uint32_t cagra_required_row_width(uint32_t logical_columns,
                                                       std::size_t sizeof_value,
                                                       uint32_t align_bytes = 16)
{
  return static_cast<uint32_t>(
    raft::round_up_safe<std::size_t>(static_cast<std::size_t>(logical_columns) * sizeof_value,
                                     std::lcm(align_bytes, static_cast<uint32_t>(sizeof_value))) /
    sizeof_value);
}

template <typename ValueT>
[[nodiscard]] inline uint32_t cagra_required_row_width(uint32_t logical_columns,
                                                       uint32_t align_bytes = 16)
{
  return cagra_required_row_width(logical_columns, sizeof(ValueT), align_bytes);
}

/** Actual row width in elements (leading dimension) of a 2D row-major matrix view. */
template <typename T, typename I, typename L>
[[nodiscard]] inline uint32_t matrix_actual_row_width(raft::device_matrix_view<T, I, L> m)
{
  return m.stride(0) > 0 ? static_cast<uint32_t>(m.stride(0)) : static_cast<uint32_t>(m.extent(1));
}

template <typename T, typename I, typename L>
[[nodiscard]] inline uint32_t matrix_actual_row_width(raft::host_matrix_view<T, I, L> m)
{
  return m.stride(0) > 0 ? static_cast<uint32_t>(m.stride(0)) : static_cast<uint32_t>(m.extent(1));
}

/**
 * @brief True if the matrix's row width in elements matches `cagra_required_row_width` for
 *        `m.extent(1)` and element type `T` (CAGRA row layout is satisfied for this view).
 */
template <typename T, typename I, typename L>
[[nodiscard]] inline bool matrix_row_width_matches_cagra_required(
  raft::device_matrix_view<T, I, L> m, uint32_t align_bytes = 16)
{
  using value_type = std::remove_const_t<T>;
  const uint32_t need =
    cagra_required_row_width<value_type>(static_cast<uint32_t>(m.extent(1)), align_bytes);
  return matrix_actual_row_width(m) == need;
}

template <typename T, typename I, typename L>
[[nodiscard]] inline bool matrix_row_width_matches_cagra_required(raft::host_matrix_view<T, I, L> m,
                                                                  uint32_t align_bytes = 16)
{
  using value_type = std::remove_const_t<T>;
  const uint32_t need =
    cagra_required_row_width<value_type>(static_cast<uint32_t>(m.extent(1)), align_bytes);
  return matrix_actual_row_width(m) == need;
}

namespace detail {

template <typename SrcT>
[[nodiscard]] inline uint32_t mdspan_row_stride_elements(SrcT const& src)
{
  return src.stride(0) > 0 ? static_cast<uint32_t>(src.stride(0))
                           : static_cast<uint32_t>(src.extent(1));
}

template <typename ValueT, typename SrcT>
[[nodiscard]] inline ValueT* expect_device_accessible_data_handle(SrcT const& src,
                                                                  char const* error_msg)
{
  cudaPointerAttributes ptr_attrs;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&ptr_attrs, src.data_handle()));
  // `devicePointer` is relative to the *current* device: it is null for an allocation owned by
  // another device without peer access, even though that allocation is perfectly usable once the
  // caller switches to the owning device (as the multi-GPU paths do). Accept device and managed
  // allocations on their own merit and only consult `devicePointer` for host memory, which needs a
  // mapping to be reachable at all.
  if (ptr_attrs.type == cudaMemoryTypeDevice || ptr_attrs.type == cudaMemoryTypeManaged) {
    return const_cast<ValueT*>(src.data_handle());
  }
  auto* device_ptr = reinterpret_cast<ValueT*>(ptr_attrs.devicePointer);
  RAFT_EXPECTS(device_ptr != nullptr, "%s", error_msg);
  return device_ptr;
}

template <typename ValueT, typename IndexT, typename ViewT, typename SrcT>
[[nodiscard]] inline ViewT make_device_dense_row_major_view_from_src(SrcT const& src,
                                                                     uint32_t logical_dim)
{
  auto* device_ptr = expect_device_accessible_data_handle<ValueT>(
    src, "make_device_*_dataset_view: source must be device-accessible.");
  auto v = raft::make_device_matrix_view(
    device_ptr, src.extent(0), static_cast<IndexT>(mdspan_row_stride_elements(src)));
  return ViewT(v, logical_dim);
}

template <typename ValueT, typename IndexT, typename ViewT, typename SrcT>
[[nodiscard]] inline ViewT make_host_dense_row_major_view_from_src(SrcT const& src,
                                                                   uint32_t logical_dim)
{
  RAFT_EXPECTS(raft::get_device_for_address(src.data_handle()) == -1,
               "make_host_*_dataset_view: source must be host-accessible.");
  auto v = raft::make_host_matrix_view(const_cast<ValueT*>(src.data_handle()),
                                       src.extent(0),
                                       static_cast<IndexT>(mdspan_row_stride_elements(src)));
  return ViewT(v, logical_dim);
}

template <typename DatasetT, typename ValueT, typename IndexT, typename SrcT>
auto make_device_dense_row_major_dataset_from_src(raft::resources const& res,
                                                  SrcT const& src,
                                                  uint32_t logical_dim,
                                                  uint32_t target_stride,
                                                  char const* view_factory_name)
  -> std::unique_ptr<DatasetT>
{
  uint32_t const src_stride = mdspan_row_stride_elements(src);
  RAFT_EXPECTS(logical_dim <= target_stride,
               "logical dim (%u) must not exceed row stride (%u).",
               static_cast<unsigned>(logical_dim),
               static_cast<unsigned>(target_stride));
  RAFT_EXPECTS(static_cast<uint32_t>(src.extent(1)) <= target_stride,
               "Source row length must not exceed required stride.");
  cudaPointerAttributes ptr_attrs;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&ptr_attrs, src.data_handle()));
  bool const device_src =
    (ptr_attrs.type == cudaMemoryTypeDevice) || (ptr_attrs.type == cudaMemoryTypeManaged);
  if (device_src && src_stride == target_stride) {
    RAFT_EXPECTS(false,
                 "source is device and stride is already correct. "
                 "Use %s() to get a view instead.",
                 view_factory_name);
  }
  auto out_array = raft::make_device_matrix<ValueT, IndexT>(res, src.extent(0), target_stride);
  RAFT_CUDA_TRY(cudaMemsetAsync(out_array.data_handle(),
                                0,
                                out_array.size() * sizeof(ValueT),
                                raft::resource::get_cuda_stream(res)));
  raft::copy_matrix(out_array.data_handle(),
                    target_stride,
                    src.data_handle(),
                    src_stride,
                    logical_dim,
                    src.extent(0),
                    raft::resource::get_cuda_stream(res));
  return std::make_unique<DatasetT>(std::move(out_array), logical_dim);
}

template <typename DatasetT, typename ValueT, typename IndexT, typename SrcT>
auto make_host_dense_row_major_dataset_from_src(raft::resources const& res,
                                                SrcT const& src,
                                                uint32_t logical_dim,
                                                uint32_t target_stride,
                                                char const* view_factory_name)
  -> std::unique_ptr<DatasetT>
{
  uint32_t const src_stride = mdspan_row_stride_elements(src);
  constexpr bool device_src = SrcT::accessor_type::is_device_accessible;
  RAFT_EXPECTS(logical_dim <= target_stride,
               "logical dim (%u) must not exceed row stride (%u).",
               static_cast<unsigned>(logical_dim),
               static_cast<unsigned>(target_stride));
  if (!device_src && src_stride == target_stride) {
    RAFT_EXPECTS(false,
                 "source stride is already correct. Use %s() to get a view instead.",
                 view_factory_name);
  }
  RAFT_EXPECTS(static_cast<uint32_t>(src.extent(1)) <= target_stride,
               "Source row length must not exceed required stride.");
  auto out_array = raft::make_host_matrix<ValueT, IndexT>(src.extent(0), target_stride);
  std::memset(out_array.data_handle(), 0, out_array.size() * sizeof(ValueT));
  raft::copy_matrix(out_array.data_handle(),
                    target_stride,
                    src.data_handle(),
                    src_stride,
                    logical_dim,
                    src.extent(0),
                    raft::resource::get_cuda_stream(res));
  if (device_src) { raft::resource::sync_stream(res); }
  return std::make_unique<DatasetT>(std::move(out_array), logical_dim);
}

}  // namespace detail

template <typename SrcT>
auto make_device_padded_dataset_view(const raft::resources& res,
                                     SrcT const& src,
                                     uint32_t align_bytes = 16)
  -> device_padded_dataset_view<typename SrcT::value_type, typename SrcT::index_type>
{
  using value_type = typename SrcT::value_type;
  using index_type = typename SrcT::index_type;
  uint32_t required_stride =
    cagra_required_row_width<value_type>(static_cast<uint32_t>(src.extent(1)), align_bytes);
  RAFT_EXPECTS(
    detail::mdspan_row_stride_elements(src) == required_stride,
    "make_device_padded_dataset_view: stride is incorrect (required stride for alignment). "
    "Use make_device_padded_dataset() to get an owning padded copy.");
  return detail::make_device_dense_row_major_view_from_src<
    value_type,
    index_type,
    device_padded_dataset_view<value_type, index_type>>(src, static_cast<uint32_t>(src.extent(1)));
}

template <typename SrcT>
auto make_device_padded_dataset(const raft::resources& res,
                                SrcT const& src,
                                uint32_t align_bytes = 16)
  -> std::unique_ptr<device_padded_dataset<typename SrcT::value_type, typename SrcT::index_type>>
{
  using value_type               = typename SrcT::value_type;
  using index_type               = typename SrcT::index_type;
  uint32_t const logical_dim     = static_cast<uint32_t>(src.extent(1));
  uint32_t const required_stride = cagra_required_row_width<value_type>(logical_dim, align_bytes);
  return detail::make_device_dense_row_major_dataset_from_src<
    device_padded_dataset<value_type, index_type>,
    value_type,
    index_type>(res, src, logical_dim, required_stride, "make_device_padded_dataset_view");
}

template <typename SrcT>
auto make_host_padded_dataset_view(SrcT const& src, uint32_t align_bytes = 16)
  -> host_padded_dataset_view<typename SrcT::value_type, typename SrcT::index_type>
{
  using value_type = typename SrcT::value_type;
  using index_type = typename SrcT::index_type;
  uint32_t required_stride =
    cagra_required_row_width<value_type>(static_cast<uint32_t>(src.extent(1)), align_bytes);
  RAFT_EXPECTS(
    detail::mdspan_row_stride_elements(src) == required_stride,
    "make_host_padded_dataset_view: stride is incorrect (required stride for alignment). "
    "Use make_host_padded_dataset() to get an owning padded copy.");
  return detail::make_host_dense_row_major_view_from_src<
    value_type,
    index_type,
    host_padded_dataset_view<value_type, index_type>>(src, static_cast<uint32_t>(src.extent(1)));
}

template <typename SrcT>
auto make_host_padded_dataset(const raft::resources& res,
                              SrcT const& src,
                              uint32_t align_bytes = 16)
  -> std::unique_ptr<host_padded_dataset<typename SrcT::value_type, typename SrcT::index_type>>
{
  using value_type               = typename SrcT::value_type;
  using index_type               = typename SrcT::index_type;
  uint32_t const logical_dim     = static_cast<uint32_t>(src.extent(1));
  uint32_t const required_stride = cagra_required_row_width<value_type>(logical_dim, align_bytes);
  return detail::make_host_dense_row_major_dataset_from_src<
    host_padded_dataset<value_type, index_type>,
    value_type,
    index_type>(res, src, logical_dim, required_stride, "make_host_padded_dataset_view");
}

template <typename SrcT>
auto make_device_standard_dataset_view(SrcT const& src)
  -> device_standard_dataset_view<typename SrcT::value_type, typename SrcT::index_type>
{
  using value_type = typename SrcT::value_type;
  using index_type = typename SrcT::index_type;
  return detail::make_device_dense_row_major_view_from_src<
    value_type,
    index_type,
    device_standard_dataset_view<value_type, index_type>>(src,
                                                          static_cast<uint32_t>(src.extent(1)));
}

/**
 * @brief Create an owning device standard dataset with explicit row layout.
 *
 * Internal use only: the sole call site today is
 * `cuvs::neighbors::detail::deserialize_standard()` in `dataset_serialize.hpp`, which must pass
 * wire-format `(logical_dim, stride)` because the deserialized host buffer is tight `[n_rows x
 * dim]` while the on-disk stride may be larger. Do not call from user code; prefer
 * `make_device_standard_dataset_view()` when wrapping existing correctly-strided storage.
 */
template <typename SrcT>
auto make_device_standard_dataset(const raft::resources& res,
                                  SrcT const& src,
                                  uint32_t logical_dim,
                                  uint32_t target_stride)
  -> std::unique_ptr<device_standard_dataset<typename SrcT::value_type, typename SrcT::index_type>>
{
  using value_type = typename SrcT::value_type;
  using index_type = typename SrcT::index_type;
  return detail::make_device_dense_row_major_dataset_from_src<
    device_standard_dataset<value_type, index_type>,
    value_type,
    index_type>(res, src, logical_dim, target_stride, "make_device_standard_dataset_view");
}

template <typename SrcT>
auto make_host_standard_dataset_view(SrcT const& src)
  -> host_standard_dataset_view<typename SrcT::value_type, typename SrcT::index_type>
{
  using value_type = typename SrcT::value_type;
  using index_type = typename SrcT::index_type;
  return detail::make_host_dense_row_major_view_from_src<
    value_type,
    index_type,
    host_standard_dataset_view<value_type, index_type>>(src, static_cast<uint32_t>(src.extent(1)));
}

namespace filtering {

/**
 * @defgroup neighbors_filtering Filtering for ANN Types
 * @{
 */

enum class FilterType : int { None = 0, Bitmap = 1, Bitset = 2, Bloom = 3, UDF = 100 };

struct base_filter {
  ~base_filter()                             = default;
  virtual FilterType get_filter_type() const = 0;
};

/* A filter that filters nothing. This is the default behavior. */
struct none_sample_filter : public base_filter {
  /** \cond */
  constexpr __forceinline__ _RAFT_HOST_DEVICE bool operator()(
    // query index
    const uint32_t query_ix,
    // the current inverted list index
    const uint32_t cluster_ix,
    // the index of the current sample inside the current inverted list
    const uint32_t sample_ix) const;

  constexpr __forceinline__ _RAFT_HOST_DEVICE bool operator()(
    // query index
    const uint32_t query_ix,
    // the index of the current sample
    const uint32_t sample_ix) const;
  /** \endcond */
  FilterType get_filter_type() const override { return FilterType::None; }
};

/**
 * @brief Filter used to convert the cluster index and sample index
 * of an IVF search into a sample index. This can be used as an
 * intermediate filter.
 *
 * @tparam index_t Indexing type
 * @tparam filter_t
 */
template <typename index_t, typename filter_t>
struct ivf_to_sample_filter : public base_filter {
  const index_t* const* inds_ptrs_;
  const filter_t next_filter_;

  _RAFT_HOST_DEVICE ivf_to_sample_filter(const index_t* const* inds_ptrs,
                                         const filter_t next_filter);

  /** \cond */
  /** If the original filter takes three arguments, then don't modify the arguments.
   * If the original filter takes two arguments, then we are using `inds_ptr_` to obtain the sample
   * index.
   */
  inline _RAFT_HOST_DEVICE bool operator()(
    // query index
    const uint32_t query_ix,
    // the current inverted list index
    const uint32_t cluster_ix,
    // the index of the current sample inside the current inverted list
    const uint32_t sample_ix) const;

  FilterType get_filter_type() const override { return next_filter_.get_filter_type(); }
  /** \endcond */
};

/**
 * @brief Filter an index with a bitmap
 *
 * @tparam bitmap_t Data type of the bitmap
 * @tparam index_t Indexing type
 */
template <typename bitmap_t, typename index_t>
struct bitmap_filter : public base_filter {
  using view_t = cuvs::core::bitmap_view<bitmap_t, index_t>;

  // View of the bitset to use as a filter
  const view_t bitmap_view_;

  bitmap_filter(const view_t bitmap_for_filtering);
  /** \cond */
  inline _RAFT_HOST_DEVICE bool operator()(
    // query index
    const uint32_t query_ix,
    // the index of the current sample
    const uint32_t sample_ix) const;
  /** \endcond */

  FilterType get_filter_type() const override { return FilterType::Bitmap; }

  view_t view() const { return bitmap_view_; }

  template <typename csr_matrix_t>
  void to_csr(raft::resources const& handle, csr_matrix_t& csr);
};

/**
 * @brief Filter an index with a bitset
 *
 * This filter holds a non-owning view of the bitset; it does not allocate or copy the underlying
 * device buffer. The library performs no caching of the bitset across search calls. Allocating and
 * populating the device bitset may be more expensive than a single filtered search, so callers that
 * issue repeated searches against the same filter (e.g. many queries over one index) should build
 * the bitset once and reuse it across those calls rather than rebuild it per search. Reusing the
 * bitset is essential for realizing the full throughput of filtered search.
 *
 * @tparam bitset_t Data type of the bitset
 * @tparam index_t Indexing type
 */
template <typename bitset_t, typename index_t>
struct bitset_filter : public base_filter {
  using view_t = cuvs::core::bitset_view<bitset_t, index_t>;

  // View of the bitset to use as a filter
  const view_t bitset_view_;

  /** \cond */
  _RAFT_HOST_DEVICE bitset_filter(const view_t bitset_for_filtering);
  constexpr __forceinline__ _RAFT_HOST_DEVICE bool operator()(
    // query index
    const uint32_t query_ix,
    // the index of the current sample
    const uint32_t sample_ix) const;
  /** \endcond */

  FilterType get_filter_type() const override { return FilterType::Bitset; }

  view_t view() const { return bitset_view_; }

  template <typename csr_matrix_t>
  void to_csr(raft::resources const& handle, csr_matrix_t& csr);
};

/**
 * @brief Filter CAGRA candidates with a global @c cuvs::core::bloom_filter over the index.
 *
 * Build the filter once on the host with bulk @c add() over the allowed dataset row ids and pass
 * the owning @c cuvs::core::bloom_filter to this wrapper. CAGRA internals build/cache the device
 * payload, similar to @ref bitset_filter, and the linked JIT-LTO fragment probes the same filter
 * for every query and candidate with probabilistic membership tests.
 *
 * Bloom filters have no false negatives: if a row was inserted, @c contains returns @c true. False
 * positives are possible, so highly selective predicates may still need a bitset or UDF for exact
 * filtering.
 *
 * This adapter is non-owning. The referenced @c cuvs::core::bloom_filter must outlive the adapter
 * and any searches that use it, and must not be moved or mutated concurrently with a search.
 */
struct bloom_filter : public base_filter {
  void* filter_data{nullptr};

  bloom_filter() = default;

  explicit bloom_filter(const cuvs::core::bloom_filter& bloom_filter)
    : filter_data(const_cast<cuvs::core::bloom_filter*>(&bloom_filter))
  {
  }

  FilterType get_filter_type() const override { return FilterType::Bloom; }
};

/**
 * @brief JIT-LTO user-defined filter predicate.
 *
 * The source must define a device function named by @c function_name with signature:
 *
 * @code{.cpp}
 * __device__ bool cuvs_filter_udf(uint32_t query_id, source_index_t source_id, void* filter_data);
 * @endcode
 *
 * Return @c true to allow a source vector to appear in the results and @c false to reject it.
 * @c filter_data is passed through unchanged and must point to device-accessible memory when the
 * UDF dereferences it. CAGRA currently provides @c source_index_t as @c uint32_t in the generated
 * JIT fragment.
 */
struct udf_filter : public base_filter {
  /** CUDA C++ source containing the device predicate. */
  std::string source;
  /** Opaque device-accessible pointer passed to the predicate. */
  void* filter_data = nullptr;
  /** Estimated fraction of rows rejected by the predicate, or negative if unknown. */
  float filtering_rate = -1.0f;
  /** Device function name to call from the generated CAGRA sample filter. */
  std::string function_name = "cuvs_filter_udf";

  udf_filter() = default;

  explicit udf_filter(std::string source,
                      void* filter_data         = nullptr,
                      float filtering_rate      = -1.0f,
                      std::string function_name = "cuvs_filter_udf")
    : source(std::move(source)),
      filter_data(filter_data),
      filtering_rate(filtering_rate),
      function_name(std::move(function_name))
  {
  }

  FilterType get_filter_type() const override { return FilterType::UDF; }
};

/** @} */  // end group neighbors_filtering

/**
 * If the filtering depends on the index of a sample, then the following
 * filter template can be used:
 *
 * template <typename IdxT>
 * struct index_ivf_sample_filter {
 *   using index_type = IdxT;
 *
 *   const index_type* const* inds_ptr = nullptr;
 *
 *   index_ivf_sample_filter() {}
 *   index_ivf_sample_filter(const index_type* const* _inds_ptr)
 *       : inds_ptr{_inds_ptr} {}
 *   index_ivf_sample_filter(const index_ivf_sample_filter&) = default;
 *   index_ivf_sample_filter(index_ivf_sample_filter&&) = default;
 *   index_ivf_sample_filter& operator=(const index_ivf_sample_filter&) = default;
 *   index_ivf_sample_filter& operator=(index_ivf_sample_filter&&) = default;
 *
 *   inline _RAFT_HOST_DEVICE bool operator()(
 *       const uint32_t query_ix,
 *       const uint32_t cluster_ix,
 *       const uint32_t sample_ix) const {
 *     index_type database_idx = inds_ptr[cluster_ix][sample_ix];
 *
 *     // return true or false, depending on the database_idx
 *     return true;
 *   }
 * };
 *
 * Initialize it as:
 *   using filter_type = index_ivf_sample_filter<idx_t>;
 *   filter_type filter(cuvs_ivfpq_index.inds_ptrs().data_handle());
 *
 * Use it as:
 *   cuvs::neighbors::ivf_pq::search_with_filtering<data_t, idx_t, filter_type>(
 *     ...regular parameters here...,
 *     filter
 *   );
 *
 * Another example would be the following filter that greenlights samples according
 * to a contiguous bit mask vector.
 *
 * template <typename IdxT>
 * struct bitmask_ivf_sample_filter {
 *   using index_type = IdxT;
 *
 *   const index_type* const* inds_ptr = nullptr;
 *   const uint64_t* const bit_mask_ptr = nullptr;
 *   const int64_t bit_mask_stride_64 = 0;
 *
 *   bitmask_ivf_sample_filter() {}
 *   bitmask_ivf_sample_filter(
 *       const index_type* const* _inds_ptr,
 *       const uint64_t* const _bit_mask_ptr,
 *       const int64_t _bit_mask_stride_64)
 *       : inds_ptr{_inds_ptr},
 *         bit_mask_ptr{_bit_mask_ptr},
 *         bit_mask_stride_64{_bit_mask_stride_64} {}
 *   bitmask_ivf_sample_filter(const bitmask_ivf_sample_filter&) = default;
 *   bitmask_ivf_sample_filter(bitmask_ivf_sample_filter&&) = default;
 *   bitmask_ivf_sample_filter& operator=(const bitmask_ivf_sample_filter&) = default;
 *   bitmask_ivf_sample_filter& operator=(bitmask_ivf_sample_filter&&) = default;
 *
 *   inline _RAFT_HOST_DEVICE bool operator()(
 *       const uint32_t query_ix,
 *       const uint32_t cluster_ix,
 *       const uint32_t sample_ix) const {
 *     const index_type database_idx = inds_ptr[cluster_ix][sample_ix];
 *     const uint64_t bit_mask_element =
 *         bit_mask_ptr[query_ix * bit_mask_stride_64 + database_idx / 64];
 *     const uint64_t masked_bool =
 *         bit_mask_element & (1ULL << (uint64_t)(database_idx % 64));
 *     const bool is_bit_set = (masked_bool != 0);
 *
 *     return is_bit_set;
 *   }
 * };
 */
}  // namespace filtering

namespace ivf {

/**
 * Default value filled in the `indices` array.
 * One may encounter it trying to access a record within a list that is outside of the
 * `size` bound or whenever the list is allocated but not filled-in yet.
 */
template <typename IdxT>
constexpr static IdxT kInvalidRecord =
  (std::is_signed_v<IdxT> ? IdxT{0} : std::numeric_limits<IdxT>::max()) - 1;

/**
 * Abstract base class for IVF list data.
 * This allows polymorphic access to list data regardless of the underlying layout.
 *
 * @tparam ValueT The data element type (e.g., uint8_t for PQ codes, float for raw vectors)
 * @tparam IdxT The index type for source indices
 * @tparam SizeT The size type
 *
 * TODO: Make this struct internal (tracking issue: https://github.com/nvidia/cuvs/issues/1726)
 */
template <typename ValueT, typename IdxT, typename SizeT = uint32_t>
struct list_base {
  using value_type = ValueT;
  using index_type = IdxT;
  using size_type  = SizeT;

  virtual ~list_base() = default;

  /** Get the raw data pointer. */
  virtual value_type* data_ptr() noexcept             = 0;
  virtual const value_type* data_ptr() const noexcept = 0;

  /** Get the indices pointer. */
  virtual index_type* indices_ptr() noexcept             = 0;
  virtual const index_type* indices_ptr() const noexcept = 0;

  /** Get the current size (number of records). */
  virtual size_type get_size() const noexcept = 0;

  /** Set the current size (number of records). */
  virtual void set_size(size_type new_size) noexcept = 0;

  /** Get the total size of the data array in bytes. */
  virtual size_t data_byte_size() const noexcept = 0;

  /** Get the capacity (number of indices that can be stored). */
  virtual size_type indices_capacity() const noexcept = 0;
};

/** The data for a single IVF list. */
template <template <typename, typename...> typename SpecT,
          typename SizeT,
          typename... SpecExtraArgs>
struct list : public list_base<typename SpecT<SizeT, SpecExtraArgs...>::value_type,
                               typename SpecT<SizeT, SpecExtraArgs...>::index_type,
                               SizeT> {
  using size_type    = SizeT;
  using spec_type    = SpecT<size_type, SpecExtraArgs...>;
  using value_type   = typename spec_type::value_type;
  using index_type   = typename spec_type::index_type;
  using list_extents = typename spec_type::list_extents;

  /** Possibly encoded data; it's layout is defined by `SpecT`. */
  raft::device_mdarray<value_type, list_extents, raft::row_major> data;
  /** Source indices. */
  raft::device_mdarray<index_type, raft::extent_1d<size_type>, raft::row_major> indices;
  /** The actual size of the content. */
  std::atomic<size_type> size;

  /** Allocate a new list capable of holding at least `n_rows` data records and indices. */
  list(raft::resources const& res, const spec_type& spec, size_type n_rows);

  value_type* data_ptr() noexcept override { return data.data_handle(); }
  const value_type* data_ptr() const noexcept override { return data.data_handle(); }

  index_type* indices_ptr() noexcept override { return indices.data_handle(); }
  const index_type* indices_ptr() const noexcept override { return indices.data_handle(); }

  size_type get_size() const noexcept override { return size.load(); }
  void set_size(size_type new_size) noexcept override { size.store(new_size); }

  size_t data_byte_size() const noexcept override { return data.size() * sizeof(value_type); }
  size_type indices_capacity() const noexcept override { return indices.extent(0); }
};

template <typename ListT, class T = void>
struct enable_if_valid_list {};

template <class T,
          template <typename, typename...> typename SpecT,
          typename SizeT,
          typename... SpecExtraArgs>
struct enable_if_valid_list<list<SpecT, SizeT, SpecExtraArgs...>, T> {
  using type = T;
};

/**
 * Designed after `std::enable_if_t`, this trait is helpful in the instance resolution;
 * plug this in the return type of a function that has an instance of `ivf::list` as
 * a template parameter.
 */
template <typename ListT, class T = void>
using enable_if_valid_list_t = typename enable_if_valid_list<ListT, T>::type;

/**
 * Resize a list by the given id, so that it can contain the given number of records;
 * copy the data if necessary.
 *
 * @note This is an internal function that requires the concrete list type.
 *       For IVF-PQ indexes, prefer using the helper functions in
 *       `cuvs::neighbors::ivf_pq::helpers::resize_list` which handle type casting internally.
 */
template <typename ListT>
CUVS_EXPORT void resize_list(raft::resources const& res,
                             std::shared_ptr<ListT>& orig_list,  // NOLINT
                             const typename ListT::spec_type& spec,
                             typename ListT::size_type new_used_size,
                             typename ListT::size_type old_used_size);

/**
 * Serialize a list to an output stream.
 *
 * @note This function requires the concrete list type (not the base class) because:
 *       1. It needs access to the spec_type to determine the data layout for serialization
 *       2. The serialized format depends on the spec's make_list_extents() method
 *       When calling from code that only has a base class pointer, use std::static_pointer_cast
 *       to obtain the typed pointer first.
 */
template <typename ListT>
enable_if_valid_list_t<ListT> serialize_list(
  const raft::resources& handle,
  std::ostream& os,
  const ListT& ld,
  const typename ListT::spec_type& store_spec,
  std::optional<typename ListT::size_type> size_override = std::nullopt);

template <typename ListT>
enable_if_valid_list_t<ListT> serialize_list(
  const raft::resources& handle,
  std::ostream& os,
  const std::shared_ptr<ListT>& ld,
  const typename ListT::spec_type& store_spec,
  std::optional<typename ListT::size_type> size_override = std::nullopt);

/**
 * Deserialize a list from an arbitrary input stream.
 *
 * This compatibility path stages list data through host memory because a std::istream does not
 * expose a portable file path or descriptor. Index filename overloads use KvikIO and transfer list
 * payloads directly to device memory when GDS is available.
 */
template <typename ListT>
enable_if_valid_list_t<ListT> deserialize_list(const raft::resources& handle,
                                               std::istream& is,
                                               std::shared_ptr<ListT>& ld,
                                               const typename ListT::spec_type& store_spec,
                                               const typename ListT::spec_type& device_spec);
}  // namespace ivf

using namespace raft;

template <typename AnnIndexType, typename T, typename IdxT>
struct iface {
  iface()
    : cagra_owned_padded_dataset_(nullptr),
      cagra_owned_standard_dataset_(nullptr),
      mutex_(std::make_shared<std::mutex>())
  {
  }

  const IdxT size() const { return index_.value().size(); }

  std::optional<AnnIndexType> index_;
  /** Used by CAGRA when deserializing an index that contains a dataset; keeps it alive for the
   * view. */
  std::unique_ptr<cuvs::neighbors::device_padded_dataset<T, int64_t>> cagra_owned_padded_dataset_;
  /** Used by CAGRA standard-layout paths to keep deserialized/attached dataset views alive. */
  std::unique_ptr<cuvs::neighbors::device_standard_dataset<T, int64_t>>
    cagra_owned_standard_dataset_;
  std::shared_ptr<std::mutex> mutex_;
};

template <typename AnnIndexType, typename T, typename IdxT, typename Accessor>
void build(const raft::resources& handle,
           cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
           const cuvs::neighbors::index_params* index_params,
           raft::mdspan<const T, matrix_extent<int64_t>, row_major, Accessor> index_dataset);

template <typename AnnIndexType, typename T, typename IdxT, typename Accessor1, typename Accessor2>
void extend(
  const raft::resources& handle,
  cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
  raft::mdspan<const T, matrix_extent<int64_t>, row_major, Accessor1> new_vectors,
  std::optional<raft::mdspan<const IdxT, vector_extent<int64_t>, layout_c_contiguous, Accessor2>>
    new_indices);

template <typename AnnIndexType, typename T, typename IdxT, typename searchIdxT>
void search(const raft::resources& handle,
            const cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
            const cuvs::neighbors::search_params* search_params,
            raft::device_matrix_view<const T, int64_t, row_major> h_queries,
            raft::device_matrix_view<searchIdxT, int64_t, row_major> d_neighbors,
            raft::device_matrix_view<float, int64_t, row_major> d_distances);

template <typename AnnIndexType, typename T, typename IdxT>
void serialize(const raft::resources& handle,
               const cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
               std::ostream& os);

template <typename AnnIndexType, typename T, typename IdxT>
void deserialize(const raft::resources& handle,
                 cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
                 std::istream& is);

template <typename AnnIndexType, typename T, typename IdxT>
void deserialize(const raft::resources& handle,
                 cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
                 const std::string& filename);

/// \defgroup mg_cpp_index_params ANN MG index build parameters

/** Distribution mode */
/// \ingroup mg_cpp_index_params
enum distribution_mode {
  /** Index is replicated on each device, favors throughput */
  REPLICATED,
  /** Index is split on several devices, favors scaling */
  SHARDED
};

/// \defgroup mg_cpp_search_params ANN MG search parameters

/** Search mode when using a replicated index */
/// \ingroup mg_cpp_search_params
enum replicated_search_mode {
  /** Search queries are split to maintain equal load on GPUs */
  LOAD_BALANCER,
  /** Each search query is processed by a single GPU in a round-robin fashion */
  ROUND_ROBIN
};

/** Merge mode when using a sharded index */
/// \ingroup mg_cpp_search_params
enum sharded_merge_mode {
  /** Search batches are merged on the root rank */
  MERGE_ON_ROOT_RANK,
  /** Search batches are merged in a tree reduction fashion */
  TREE_MERGE
};

/** Build parameters */
/// \ingroup mg_cpp_index_params
template <typename Upstream>
struct mg_index_params : public Upstream {
  mg_index_params() : mode(SHARDED) {}

  mg_index_params(const Upstream& sp) : Upstream(sp), mode(SHARDED) {}

  /** Distribution mode */
  cuvs::neighbors::distribution_mode mode = SHARDED;
};

/** Search parameters */
/// \ingroup mg_cpp_search_params
template <typename Upstream>
struct mg_search_params : public Upstream {
  mg_search_params() : search_mode(LOAD_BALANCER), merge_mode(TREE_MERGE) {}

  mg_search_params(const Upstream& sp)
    : Upstream(sp), search_mode(LOAD_BALANCER), merge_mode(TREE_MERGE)
  {
  }

  /** Replicated search mode */
  cuvs::neighbors::replicated_search_mode search_mode = LOAD_BALANCER;
  /** Sharded merge mode */
  cuvs::neighbors::sharded_merge_mode merge_mode = TREE_MERGE;
  /** Number of rows per batch */
  int64_t n_rows_per_batch = 1 << 20;
};

template <typename AnnIndexType, typename T, typename IdxT>
struct mg_index {
  mg_index(const raft::resources& clique);
  mg_index(const raft::resources& clique, distribution_mode mode);
  mg_index(const raft::resources& clique, const std::string& filename);

  mg_index(const mg_index&)                    = delete;
  mg_index(mg_index&&)                         = default;
  auto operator=(const mg_index&) -> mg_index& = delete;
  auto operator=(mg_index&&) -> mg_index&      = default;

  distribution_mode mode_;
  int num_ranks_;
  std::vector<iface<AnnIndexType, T, IdxT>> ann_interfaces_;

  // for load balancing mechanism
  std::shared_ptr<std::atomic<int64_t>> round_robin_counter_;
};

}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
