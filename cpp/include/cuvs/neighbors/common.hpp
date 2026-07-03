/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>   // get_device_for_address
#include <raft/util/integer_utils.hpp>  // rounding up

#include <cuvs/core/bitmap.hpp>
#include <cuvs/core/bitset.hpp>
#include <cuvs/core/export.hpp>
#include <raft/core/detail/macros.hpp>

#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>

#ifdef __cpp_lib_bitops
#include <bit>
#endif

namespace CUVS_EXPORT cuvs {
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

/** Two-dimensional dataset; maybe owning, maybe compressed, maybe strided. */
template <typename IdxT>
struct dataset {
  using index_type = IdxT;
  /**  Size of the dataset. */
  [[nodiscard]] virtual auto n_rows() const noexcept -> index_type = 0;
  /** Dimensionality of the dataset. */
  [[nodiscard]] virtual auto dim() const noexcept -> uint32_t = 0;
  /** Whether the object owns the data. */
  [[nodiscard]] virtual auto is_owning() const noexcept -> bool = 0;
  virtual ~dataset() noexcept                                   = default;
};

template <typename IdxT>
struct empty_dataset : public dataset<IdxT> {
  using index_type = IdxT;
  uint32_t suggested_dim;
  explicit empty_dataset(uint32_t dim) noexcept : suggested_dim(dim) {}
  [[nodiscard]] auto n_rows() const noexcept -> index_type final { return 0; }
  [[nodiscard]] auto dim() const noexcept -> uint32_t final { return suggested_dim; }
  [[nodiscard]] auto is_owning() const noexcept -> bool final { return true; }
};

template <typename DataT, typename IdxT>
struct strided_dataset : public dataset<IdxT> {
  using index_type = IdxT;
  using value_type = DataT;
  using view_type  = raft::device_matrix_view<const value_type, index_type, raft::layout_stride>;
  [[nodiscard]] auto n_rows() const noexcept -> index_type final { return view().extent(0); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t final
  {
    return static_cast<uint32_t>(view().extent(1));
  }
  /** Leading dimension of the dataset. */
  [[nodiscard]] constexpr auto stride() const noexcept -> uint32_t
  {
    auto v = view();
    return static_cast<uint32_t>(v.stride(0) > 0 ? v.stride(0) : v.extent(1));
  }
  /** Get the view of the data. */
  [[nodiscard]] virtual auto view() const noexcept -> view_type = 0;
};

template <typename DataT, typename IdxT>
struct non_owning_dataset : public strided_dataset<DataT, IdxT> {
  using index_type = IdxT;
  using value_type = DataT;
  using typename strided_dataset<value_type, index_type>::view_type;
  view_type data;
  explicit non_owning_dataset(view_type v) noexcept : data(v) {}
  [[nodiscard]] auto is_owning() const noexcept -> bool final { return false; }
  [[nodiscard]] auto view() const noexcept -> view_type final { return data; };
};

template <typename DataT, typename IdxT, typename LayoutPolicy, typename ContainerPolicy>
struct owning_dataset : public strided_dataset<DataT, IdxT> {
  using index_type = IdxT;
  using value_type = DataT;
  using typename strided_dataset<value_type, index_type>::view_type;
  using storage_type =
    raft::mdarray<value_type, raft::matrix_extent<index_type>, LayoutPolicy, ContainerPolicy>;
  using mapping_type = typename view_type::mapping_type;
  storage_type data;
  mapping_type view_mapping;
  owning_dataset(storage_type&& store, mapping_type view_mapping) noexcept
    : data{std::move(store)}, view_mapping{view_mapping}
  {
  }

  [[nodiscard]] auto is_owning() const noexcept -> bool final { return true; }
  [[nodiscard]] auto view() const noexcept -> view_type final
  {
    return view_type{data.data_handle(), view_mapping};
  };
};

template <typename DatasetT>
struct is_strided_dataset : std::false_type {};

template <typename DataT, typename IdxT>
struct is_strided_dataset<strided_dataset<DataT, IdxT>> : std::true_type {};

template <typename DataT, typename IdxT>
struct is_strided_dataset<non_owning_dataset<DataT, IdxT>> : std::true_type {};

template <typename DataT, typename IdxT, typename LayoutPolicy, typename ContainerPolicy>
struct is_strided_dataset<owning_dataset<DataT, IdxT, LayoutPolicy, ContainerPolicy>>
  : std::true_type {};

template <typename DatasetT>
inline constexpr bool is_strided_dataset_v = is_strided_dataset<DatasetT>::value;

/**
 * @brief Construct a strided matrix from any mdarray or mdspan.
 *
 * This function constructs a non-owning view if the input satisfied two conditions:
 *
 *   1) The data is accessible from the current device
 *   2) The memory layout is the same as expected (row-major matrix with the required stride)
 *
 * Otherwise, this function constructs an owning device matrix and copies the data.
 * When the data is copied, padding elements are filled with zeroes.
 *
 * @tparam SrcT the source mdarray or mdspan
 *
 * @param[in] res raft resources handle
 * @param[in] src the source mdarray or mdspan
 * @param[in] required_stride the leading dimension (in elements)
 * @return maybe owning current-device-accessible strided matrix
 */
template <typename SrcT>
auto make_strided_dataset(const raft::resources& res, const SrcT& src, uint32_t required_stride)
  -> std::unique_ptr<strided_dataset<typename SrcT::value_type, typename SrcT::index_type>>
{
  using extents_type = typename SrcT::extents_type;
  using value_type   = typename SrcT::value_type;
  using index_type   = typename SrcT::index_type;
  using layout_type  = typename SrcT::layout_type;
  static_assert(extents_type::rank() == 2, "The input must be a matrix.");
  static_assert(std::is_same_v<layout_type, raft::layout_right> ||
                  std::is_same_v<layout_type, raft::layout_right_padded<value_type>> ||
                  std::is_same_v<layout_type, raft::layout_stride>,
                "The input must be row-major");
  RAFT_EXPECTS(src.extent(1) <= required_stride,
               "The input row length must be not larger than the desired stride.");
  cudaPointerAttributes ptr_attrs;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&ptr_attrs, src.data_handle()));
  auto* device_ptr             = reinterpret_cast<value_type*>(ptr_attrs.devicePointer);
  const uint32_t src_stride    = src.stride(0) > 0 ? src.stride(0) : src.extent(1);
  const bool device_accessible = device_ptr != nullptr;
  const bool row_major         = src.stride(1) <= 1;
  const bool stride_matches    = required_stride == src_stride;

  if (device_accessible && row_major && stride_matches) {
    // Everything matches: make a non-owning dataset
    return std::make_unique<non_owning_dataset<value_type, index_type>>(
      raft::make_device_strided_matrix_view<const value_type, index_type>(
        device_ptr, src.extent(0), src.extent(1), required_stride));
  }
  // Something is wrong: have to make a copy and produce an owning dataset
  auto out_layout =
    raft::make_strided_layout(src.extents(), cuda::std::array<index_type, 2>{required_stride, 1});
  auto out_array =
    raft::make_device_matrix<value_type, index_type>(res, src.extent(0), required_stride);

  using out_mdarray_type          = decltype(out_array);
  using out_layout_type           = typename out_mdarray_type::layout_type;
  using out_container_policy_type = typename out_mdarray_type::container_policy_type;
  using out_owning_type =
    owning_dataset<value_type, index_type, out_layout_type, out_container_policy_type>;

  RAFT_CUDA_TRY(cudaMemsetAsync(out_array.data_handle(),
                                0,
                                out_array.size() * sizeof(value_type),
                                raft::resource::get_cuda_stream(res)));
  raft::copy_matrix(out_array.data_handle(),
                    required_stride,
                    src.data_handle(),
                    src_stride,
                    src.extent(1),
                    src.extent(0),
                    raft::resource::get_cuda_stream(res));

  return std::make_unique<out_owning_type>(std::move(out_array), out_layout);
}

/**
 * @brief Construct a strided matrix from any mdarray.
 *
 * This function constructs an owning device matrix and copies the data.
 * When the data is copied, padding elements are filled with zeroes.
 *
 * @tparam DataT
 * @tparam IdxT
 * @tparam LayoutPolicy
 * @tparam ContainerPolicy
 *
 * @param[in] res raft resources handle
 * @param[in] src the source mdarray or mdspan
 * @param[in] required_stride the leading dimension (in elements)
 * @return owning current-device-accessible strided matrix
 */
template <typename DataT, typename IdxT, typename LayoutPolicy, typename ContainerPolicy>
auto make_strided_dataset(
  const raft::resources& res,
  raft::mdarray<DataT, raft::matrix_extent<IdxT>, LayoutPolicy, ContainerPolicy>&& src,
  uint32_t required_stride) -> std::unique_ptr<strided_dataset<DataT, IdxT>>
{
  using value_type            = DataT;
  using index_type            = IdxT;
  using layout_type           = LayoutPolicy;
  using container_policy_type = ContainerPolicy;
  static_assert(std::is_same_v<layout_type, raft::layout_right> ||
                  std::is_same_v<layout_type, raft::layout_right_padded<value_type>> ||
                  std::is_same_v<layout_type, raft::layout_stride>,
                "The input must be row-major");
  RAFT_EXPECTS(src.extent(1) <= required_stride,
               "The input row length must be not larger than the desired stride.");
  const uint32_t src_stride = src.stride(0) > 0 ? src.stride(0) : src.extent(1);
  const bool stride_matches = required_stride == src_stride;

  auto out_layout =
    raft::make_strided_layout(src.extents(), cuda::std::array<index_type, 2>{required_stride, 1});

  using out_mdarray_type          = raft::device_matrix<value_type, index_type>;
  using out_layout_type           = typename out_mdarray_type::layout_type;
  using out_container_policy_type = typename out_mdarray_type::container_policy_type;
  using out_owning_type =
    owning_dataset<value_type, index_type, out_layout_type, out_container_policy_type>;

  if constexpr (std::is_same_v<layout_type, out_layout_type> &&
                std::is_same_v<container_policy_type, out_container_policy_type>) {
    if (stride_matches) {
      // Everything matches, we can own the mdarray
      return std::make_unique<out_owning_type>(std::move(src), out_layout);
    }
  }
  // Something is wrong: have to make a copy and produce an owning dataset
  auto out_array =
    raft::make_device_matrix<value_type, index_type>(res, src.extent(0), required_stride);

  RAFT_CUDA_TRY(cudaMemsetAsync(out_array.data_handle(),
                                0,
                                out_array.size() * sizeof(value_type),
                                raft::resource::get_cuda_stream(res)));
  raft::copy_matrix(out_array.data_handle(),
                    required_stride,
                    src.data_handle(),
                    src_stride,
                    src.extent(1),
                    src.extent(0),
                    raft::resource::get_cuda_stream(res));

  return std::make_unique<out_owning_type>(std::move(out_array), out_layout);
}

/**
 * @brief Construct a strided matrix from any mdarray or mdspan.
 *
 * A variant `make_strided_dataset` that allows specifying the byte alignment instead of the
 * explicit stride length.
 *
 * @tparam SrcT the source mdarray or mdspan
 *
 * @param[in] res raft resources handle
 * @param[in] src the source mdarray or mdspan
 * @param[in] align_bytes the required byte alignment for the dataset rows.
 * @return maybe owning current-device-accessible strided matrix
 */
template <typename SrcT>
auto make_aligned_dataset(const raft::resources& res, SrcT src, uint32_t align_bytes = 16)
  -> std::unique_ptr<strided_dataset<typename SrcT::value_type, typename SrcT::index_type>>
{
  using source_type      = std::remove_cv_t<std::remove_reference_t<SrcT>>;
  using value_type       = typename source_type::value_type;
  constexpr size_t kSize = sizeof(value_type);
  uint32_t required_stride =
    raft::round_up_safe<size_t>(src.extent(1) * kSize, std::lcm(align_bytes, kSize)) / kSize;
  return make_strided_dataset(res, std::forward<SrcT>(src), required_stride);
}
/**
 * @brief VPQ compressed dataset.
 *
 * The dataset is compressed using two level quantization
 *
 *   1. Vector Quantization
 *   2. Product Quantization of residuals
 *
 * @tparam MathT the type of elements in the codebooks
 * @tparam IdxT type of the vector indices (represent dataset.extent(0))
 *
 */
template <typename MathT, typename IdxT>
struct vpq_dataset : public dataset<IdxT> {
  using index_type = IdxT;
  using math_type  = MathT;
  /** Vector Quantization codebook - "coarse cluster centers". */
  raft::device_matrix<math_type, uint32_t, raft::row_major> vq_code_book;
  /** Product Quantization codebook - "fine cluster centers".  */
  raft::device_matrix<math_type, uint32_t, raft::row_major> pq_code_book;
  /** Compressed dataset.  */
  raft::device_matrix<uint8_t, index_type, raft::row_major> data;

  vpq_dataset(raft::device_matrix<math_type, uint32_t, raft::row_major>&& vq_code_book,
              raft::device_matrix<math_type, uint32_t, raft::row_major>&& pq_code_book,
              raft::device_matrix<uint8_t, index_type, raft::row_major>&& data)
    : vq_code_book{std::move(vq_code_book)},
      pq_code_book{std::move(pq_code_book)},
      data{std::move(data)}
  {
  }

  [[nodiscard]] auto n_rows() const noexcept -> index_type final { return data.extent(0); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t final { return vq_code_book.extent(1); }
  [[nodiscard]] auto is_owning() const noexcept -> bool final { return true; }

  /** Row length of the encoded data in bytes. */
  [[nodiscard]] constexpr inline auto encoded_row_length() const noexcept -> uint32_t
  {
    return data.extent(1);
  }
  /** The number of "coarse cluster centers" */
  [[nodiscard]] constexpr inline auto vq_n_centers() const noexcept -> uint32_t
  {
    return vq_code_book.extent(0);
  }
  /** The bit length of an encoded vector element after compression by PQ. */
  [[nodiscard]] constexpr inline auto pq_bits() const noexcept -> uint32_t
  {
    /*
    NOTE: pq_bits and the book size

    Normally, we'd store `pq_bits` as a part of the index.
    However, we know there's an invariant `pq_n_centers = 1 << pq_bits`, i.e. the codebook size is
    the same as the number of possible code values. Hence, we don't store the pq_bits and derive it
    from the array dimensions instead.
     */
    auto pq_width = pq_n_centers();
#ifdef __cpp_lib_bitops
    return std::countr_zero(pq_width);
#else
    uint32_t pq_bits = 0;
    while (pq_width > 1) {
      pq_bits++;
      pq_width >>= 1;
    }
    return pq_bits;
#endif
  }
  /** The dimensionality of an encoded vector after compression by PQ. */
  [[nodiscard]] constexpr inline auto pq_dim() const noexcept -> uint32_t
  {
    return raft::div_rounding_up_unsafe(dim(), pq_len());
  }
  /** Dimensionality of a subspaces, i.e. the number of vector components mapped to a subspace */
  [[nodiscard]] constexpr inline auto pq_len() const noexcept -> uint32_t
  {
    return pq_code_book.extent(1);
  }
  /** The number of vectors in a PQ codebook (`1 << pq_bits`). */
  [[nodiscard]] constexpr inline auto pq_n_centers() const noexcept -> uint32_t
  {
    return pq_code_book.extent(0);
  }
};

template <typename DatasetT>
struct is_vpq_dataset : std::false_type {};

template <typename MathT, typename IdxT>
struct is_vpq_dataset<vpq_dataset<MathT, IdxT>> : std::true_type {};

template <typename DatasetT>
inline constexpr bool is_vpq_dataset_v = is_vpq_dataset<DatasetT>::value;

namespace filtering {

/**
 * @defgroup neighbors_filtering Filtering for ANN Types
 * @{
 */

enum class FilterType { None, Bitmap, Bitset, UDF };

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
 * TODO: Make this struct internal (tracking issue: https://github.com/rapidsai/cuvs/issues/1726)
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
  iface() : mutex_(std::make_shared<std::mutex>()) {}

  const IdxT size() const { return index_.value().size(); }

  std::optional<AnnIndexType> index_;
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
