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
#include <cuvs/core/dataset.hpp>
#include <cuvs/core/export.hpp>
#include <raft/core/detail/macros.hpp>

#include <cuda_fp16.h>

#include <concepts>
#include <cstring>
#include <memory>
#include <numeric>
#include <span>
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
class roaring_allowlist_view;
class roaring_allowlist;
}  // namespace core
namespace neighbors {
/**
 * @addtogroup cagra_cpp_index_params
 * @{
 */

/* Graph build algo used in cagra and all_neighbors */
enum GRAPH_BUILD_ALGO { BRUTE_FORCE = 0, IVF_PQ = 1, NN_DESCENT = 2, ACE = 3 };

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

namespace filtering {

/**
 * @defgroup neighbors_filtering Filtering for ANN Types
 * @{
 */

enum class FilterType : int { None = 0, Bitmap = 1, Bitset = 2, Bloom = 3, Roaring = 4, UDF = 100 };

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
 * @brief Reusable per-query mapping to immutable exact Roaring allowlists.
 *
 * Entry @c q selects view @c q. CAGRA retains candidate dataset row @c r when the selected
 * allowlist contains @c r. Construction copies only already initialized device-reference pointers
 * and empty flags into the filter payload; encoded bytes are neither copied nor parsed. Search
 * therefore performs no Roaring allocation, initialization, synchronization, or preprocessing.
 *
 * @code{.cpp}
 * auto first = cuvs::core::roaring_allowlist::from_ids(
 *   res, dataset_rows,
 *   raft::make_host_vector_view<const std::uint32_t, std::int64_t>(first_ids.data(),
 *                                                                    first_ids.size()));
 * auto second = cuvs::core::roaring_allowlist::from_ids(
 *   res, dataset_rows,
 *   raft::make_host_vector_view<const std::uint32_t, std::int64_t>(second_ids.data(),
 *                                                                    second_ids.size()));
 * std::array views{first.view(), second.view()};
 * auto filter = cuvs::neighbors::filtering::roaring_bitmap_filter(res, views);
 * @endcode
 *
 * Owners and views can be reused across filters and queries. This filter owns its mapping tables
 * and device payload, but not the referenced owners, which must outlive the filter and all searches
 * using it. Copies are cheap shared handles required by CAGRA query-offset wrappers.
 *
 * Roaring filters currently support direct @c cagra::search only. Dynamic batching can combine
 * requests into a different query-row layout, and tiered search applies one filter to partitions
 * with different row domains; both paths reject this filter type.
 *
 * @see cuvs::core::roaring_allowlist
 * @see https://github.com/RoaringBitmap/RoaringFormatSpec
 */
struct roaring_bitmap_filter : public base_filter {
 private:
  struct impl;

 public:
  /** @brief Construct an invalid handle. It cannot be passed to CAGRA search. */
  roaring_bitmap_filter() = default;

  /**
   * @brief Materialize the query-to-allowlist device pointer table.
   *
   * @p allowlists must be nonempty, every view must be valid, and every view must have the same
   * `dataset_rows()`. Query count is inferred from the span length.
   */
  explicit roaring_bitmap_filter(raft::resources const& res,
                                 std::span<const cuvs::core::roaring_allowlist_view> allowlists);

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::size_t num_queries() const noexcept;
  [[nodiscard]] std::size_t dataset_rows() const noexcept;
  [[nodiscard]] std::size_t cardinality(std::size_t query_id) const;
  [[nodiscard]] bool empty(std::size_t query_id) const;

  /**
   * @brief Conservative maximum rejected fraction among all query allowlists.
   *
   * CAGRA uses this precomputed value when `search_params::filtering_rate` is unset. Basing one
   * batch-wide scalar on the sparsest query avoids under-provisioning that query, but a very sparse
   * or empty allowlist can increase the search work performed for every query in the batch. Callers
   * may set `search_params::filtering_rate` explicitly when another tradeoff is preferable.
   */
  [[nodiscard]] float filtering_rate() const noexcept;

  /** @brief Device bytes owned by this mapping, excluding the referenced allowlists. */
  [[nodiscard]] std::size_t size_bytes() const noexcept;

  /**
   * @brief Replace one query's allowlist pointer outside the search path.
   *
   * The replacement must have the same `dataset_rows()`. Copies share the underlying mapping, so
   * the replacement is visible through every copy of this filter. The method copies one pointer
   * and one empty flag to the device and synchronizes @p res before returning. Do not call it
   * concurrently with a search, and keep the replacement owner alive for all subsequent searches.
   */
  void set_allowlist(raft::resources const& res,
                     std::size_t query_id,
                     cuvs::core::roaring_allowlist_view replacement);

  /** @brief Internal device payload already prepared for the linked CAGRA predicate. */
  [[nodiscard]] void* device_payload() const noexcept;

  FilterType get_filter_type() const override { return FilterType::Roaring; }

 private:
  std::shared_ptr<impl> impl_;
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
  std::unique_ptr<cuvs::core::device_padded_dataset<T, int64_t>> cagra_owned_padded_dataset_;
  /** Used by CAGRA standard-layout paths to keep deserialized/attached dataset views alive. */
  std::unique_ptr<cuvs::core::device_standard_dataset<T, int64_t>> cagra_owned_standard_dataset_;
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
