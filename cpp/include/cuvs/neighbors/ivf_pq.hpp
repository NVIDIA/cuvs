/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/cuda_fp16.hpp>

#include <cuvs/neighbors/common.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/integer_utils.hpp>

#include <cuvs/core/export.hpp>
#include <optional>
#include <tuple>
#include <variant>
#include <vector>

namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace ivf_pq {

/**
 * @defgroup ivf_pq_cpp_index_params IVF-PQ index build parameters
 * @{
 */
/** A type for specifying how PQ codebooks are created. */
enum class codebook_gen {  // NOLINT
  PER_SUBSPACE = 0,        // NOLINT
  PER_CLUSTER  = 1,        // NOLINT
};

/** A type for specifying the memory layout of PQ codes in IVF lists. */
enum class list_layout {  // NOLINT
  /** Flat layout: each vector's PQ codes stored contiguously [n_rows, bytes_per_vector]. */
  FLAT = 0,  // NOLINT
  /** Interleaved layout: codes from multiple vectors interleaved for coalesced memory access. */
  INTERLEAVED = 1,  // NOLINT
};

struct index_params : cuvs::neighbors::index_params {
  /**
   * The number of inverted lists (clusters)
   *
   * Hint: the number of vectors per cluster (`n_rows/n_lists`) should be approximately 1,000 to
   * 10,000.
   */
  uint32_t n_lists = 1024;
  /** The number of iterations searching for kmeans centers (index building). */
  uint32_t kmeans_n_iters = 20;
  /** The fraction of data to use during iterative kmeans building. */
  double kmeans_trainset_fraction = 0.5;
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
   * The dimensionality of the vector after compression by PQ. When zero, an optimal value is
   * selected using a heuristic.
   *
   * NB: `pq_dim * pq_bits` must be a multiple of 8.
   *
   * Hint: a smaller 'pq_dim' results in a smaller index size and better search performance, but
   * lower recall. If 'pq_bits' is 8, 'pq_dim' can be set to any number, but multiple of 8 are
   * desirable for good performance. If 'pq_bits' is not 8, 'pq_dim' should be a multiple of 8.
   * For good performance, it is desirable that 'pq_dim' is a multiple of 32. Ideally, 'pq_dim'
   * should be also a divisor of the dataset dim.
   */
  uint32_t pq_dim = 0;
  /** How PQ codebooks are created. */
  codebook_gen codebook_kind = codebook_gen::PER_SUBSPACE;
  /**
   * Memory layout of PQ codes in IVF lists.
   *
   * - INTERLEAVED (default): Codes from multiple vectors are interleaved for coalesced GPU memory
   *   access during search. This is optimized for search performance.
   * - FLAT: Each vector's PQ codes are stored contiguously.
   */
  list_layout codes_layout = list_layout::INTERLEAVED;
  /**
   * Apply a random rotation matrix on the input data and queries even if `dim % pq_dim == 0`.
   *
   * Note: if `dim` is not multiple of `pq_dim`, a random rotation is always applied to the input
   * data and queries to transform the working space from `dim` to `rot_dim`, which may be slightly
   * larger than the original space and and is a multiple of `pq_dim` (`rot_dim % pq_dim == 0`).
   * However, this transform is not necessary when `dim` is multiple of `pq_dim`
   *   (`dim == rot_dim`, hence no need in adding "extra" data columns / features).
   *
   * By default, if `dim == rot_dim`, the rotation transform is initialized with the identity
   * matrix. When `force_random_rotation == true`, a random orthogonal transform matrix is generated
   * regardless of the values of `dim` and `pq_dim`.
   */
  bool force_random_rotation = false;
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

  /**
   * The max number of data points to use per PQ code during PQ codebook training. Using more data
   * points per PQ code may increase the quality of PQ codebook but may also increase the build
   * time. The parameter is applied to both PQ codebook generation methods, i.e., PER_SUBSPACE and
   * PER_CLUSTER. In both cases, we will use `pq_book_size * max_train_points_per_pq_code` training
   * points to train each codebook.
   */
  uint32_t max_train_points_per_pq_code = 256;

  /**
   * Creates index_params based on shape of the input dataset.
   * Usage example:
   * @code{.cpp}
   *   using namespace cuvs::neighbors;
   *   raft::resources res;
   *   // create index_params for a [N. D] dataset and have InnerProduct as the distance metric
   *   auto dataset = raft::make_device_matrix<float, int64_t>(res, N, D);
   *   ivf_pq::index_params index_params =
   *     ivf_pq::index_params::from_dataset(dataset.extents(), cuvs::distance::InnerProduct);
   *   // modify/update index_params as needed
   *   index_params.add_data_on_build = true;
   * @endcode
   */
  static index_params from_dataset(
    raft::matrix_extent<int64_t> dataset,
    cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded);
};
/**
 * @}
 */

/**
 * @defgroup ivf_pq_cpp_search_params IVF-PQ index search parameters
 * @{
 */
struct search_params : cuvs::neighbors::search_params {
  /** The number of clusters to search. */
  uint32_t n_probes = 20;
  /**
   * Data type of look up table to be created dynamically at search time.
   *
   * Possible values: [CUDA_R_32F, CUDA_R_16F, CUDA_R_8U]
   *
   * The use of low-precision types reduces the amount of shared memory required at search time, so
   * fast shared memory kernels can be used even for datasets with large dimansionality. Note that
   * the recall is slightly degraded when low-precision type is selected.
   */
  cudaDataType_t lut_dtype = CUDA_R_32F;
  /**
   * Storage data type for distance/similarity computed at search time.
   *
   * Possible values: [CUDA_R_16F, CUDA_R_32F]
   *
   * If the performance limiter at search time is device memory access, selecting FP16 will improve
   * performance slightly.
   */
  cudaDataType_t internal_distance_dtype = CUDA_R_32F;
  /**
   * Preferred fraction of SM's unified memory / L1 cache to be used as shared memory.
   *
   * Possible values: [0.0 - 1.0] as a fraction of the `sharedMemPerMultiprocessor`.
   *
   * One wants to increase the carveout to make sure a good GPU occupancy for the main search
   * kernel, but not to keep it too high to leave some memory to be used as L1 cache. Note, this
   * value is interpreted only as a hint. Moreover, a GPU usually allows only a fixed set of cache
   * configurations, so the provided value is rounded up to the nearest configuration. Refer to the
   * NVIDIA tuning guide for the target GPU architecture.
   *
   * Note, this is a low-level tuning parameter that can have drastic negative effects on the search
   * performance if tweaked incorrectly.
   */
  double preferred_shmem_carveout = 1.0;
  /**
   * [Experimental] The data type to use as the GEMM element type when searching the clusters to
   * probe.
   *
   * Possible values: [CUDA_R_8I, CUDA_R_16F, CUDA_R_32F].
   *
   * - Legacy default: CUDA_R_32F (float)
   * - Recommended for performance: CUDA_R_16F (half)
   * - Experimental/low-precision: CUDA_R_8I (int8_t)
   *    (WARNING: int8_t variant degrades recall unless data is normalized and low-dimensional)
   */
  cudaDataType_t coarse_search_dtype = CUDA_R_32F;
  /**
   * Set the internal batch size to improve GPU utilization at the cost of larger memory footprint.
   */
  uint32_t max_internal_batch_size = 4096;
};
/**
 * @}
 */

static_assert(std::is_aggregate_v<index_params>);
static_assert(std::is_aggregate_v<search_params>);

/** Size of the interleaved group. */
constexpr static uint32_t kIndexGroupSize = 32;
/** Stride of the interleaved group for vectorized loads. */
constexpr static uint32_t kIndexGroupVecLen = 16;

/**
 * Default value returned by `search` when the `n_probes` is too small and top-k is too large.
 * One may encounter it if the combined size of probed clusters is smaller than the requested
 * number of results per query.
 */
template <typename IdxT>
constexpr static IdxT kOutOfBoundsRecord = std::numeric_limits<IdxT>::max();

template <typename SizeT, typename IdxT>
struct list_spec_interleaved {
  using value_type = uint8_t;
  using index_type = IdxT;
  /** PQ-encoded data stored in the interleaved format:
   *
   *    [ ceildiv(list_size, kIndexGroupSize)
   *    , ceildiv(pq_dim, (kIndexGroupVecLen * 8u) / pq_bits)
   *    , kIndexGroupSize
   *    , kIndexGroupVecLen
   *    ].
   */
  using list_extents = raft::
    extents<SizeT, raft::dynamic_extent, raft::dynamic_extent, kIndexGroupSize, kIndexGroupVecLen>;

  SizeT align_max;
  SizeT align_min;
  uint32_t pq_bits;
  uint32_t pq_dim;

  constexpr list_spec_interleaved(uint32_t pq_bits,
                                  uint32_t pq_dim,
                                  bool conservative_memory_allocation);

  // Allow casting between different size-types (for safer size and offset calculations)
  template <typename OtherSizeT>
  constexpr explicit list_spec_interleaved(
    const list_spec_interleaved<OtherSizeT, IdxT>& other_spec);

  /** Determine the extents of an array enough to hold a given amount of data. */
  constexpr list_extents make_list_extents(SizeT n_rows) const;
};

template <typename SizeT, typename IdxT>
constexpr list_spec_interleaved<SizeT, IdxT>::list_spec_interleaved(
  uint32_t pq_bits, uint32_t pq_dim, bool conservative_memory_allocation)
  : pq_bits(pq_bits),
    pq_dim(pq_dim),
    align_min(kIndexGroupSize),
    align_max(conservative_memory_allocation ? kIndexGroupSize : 1024)
{
}

template <typename SizeT, typename IdxT>
template <typename OtherSizeT>
constexpr list_spec_interleaved<SizeT, IdxT>::list_spec_interleaved(
  const list_spec_interleaved<OtherSizeT, IdxT>& other_spec)
  : pq_bits{other_spec.pq_bits},
    pq_dim{other_spec.pq_dim},
    align_min{other_spec.align_min},
    align_max{other_spec.align_max}
{
}

template <typename SizeT, typename IdxT>
constexpr typename list_spec_interleaved<SizeT, IdxT>::list_extents
list_spec_interleaved<SizeT, IdxT>::make_list_extents(SizeT n_rows) const
{
  // how many elems of pq_dim fit into one kIndexGroupVecLen-byte chunk
  auto pq_chunk = (kIndexGroupVecLen * 8u) / pq_bits;
  return list_extents{raft::div_rounding_up_safe<SizeT>(n_rows, kIndexGroupSize),
                      raft::div_rounding_up_safe<SizeT>(pq_dim, pq_chunk)};
}

template <typename IdxT, typename SizeT = uint32_t>
using list_data_interleaved = ivf::list<list_spec_interleaved, SizeT, IdxT>;

/**
 * Flat (non-interleaved) storage specification for PQ-encoded data.
 *
 * This stores each vector's PQ codes contiguously:
 *   [n_rows, bytes_per_vector] where bytes_per_vector = ceildiv(pq_dim * pq_bits, 8)
 */
template <typename SizeT, typename IdxT>
struct list_spec_flat {
  using value_type   = uint8_t;
  using index_type   = IdxT;
  using list_extents = raft::matrix_extent<SizeT>;

  SizeT align_max;
  SizeT align_min;
  uint32_t pq_bits;
  uint32_t pq_dim;

  constexpr list_spec_flat(uint32_t pq_bits, uint32_t pq_dim, bool conservative_memory_allocation)
    : pq_bits(pq_bits), pq_dim(pq_dim), align_min(1), align_max(1)
  {
  }

  // Allow casting between different size-types (for safer size and offset calculations)
  template <typename OtherSizeT>
  constexpr explicit list_spec_flat(const list_spec_flat<OtherSizeT, IdxT>& other_spec);

  /** Number of bytes per encoded vector. */
  constexpr SizeT bytes_per_vector() const
  {
    return raft::div_rounding_up_safe<SizeT>(pq_dim * pq_bits, 8u);
  }

  /** Determine the extents of an array enough to hold a given amount of data. */
  constexpr list_extents make_list_extents(SizeT n_rows) const
  {
    return list_extents{n_rows, bytes_per_vector()};
  }
};

template <typename SizeT, typename IdxT>
template <typename OtherSizeT>
constexpr list_spec_flat<SizeT, IdxT>::list_spec_flat(
  const list_spec_flat<OtherSizeT, IdxT>& other_spec)
  : pq_bits{other_spec.pq_bits},
    pq_dim{other_spec.pq_dim},
    align_min{other_spec.align_min},
    align_max{other_spec.align_max}
{
}

template <typename IdxT, typename SizeT = uint32_t>
using list_data_flat = ivf::list<list_spec_flat, SizeT, IdxT>;

/**
 * Type alias for the polymorphic base class for IVF-PQ list data.
 * IVF-PQ uses uint8_t for PQ-encoded data.
 * Both list_data_interleaved and list_data_flat now inherit from this via ivf::list.
 */
template <typename IdxT, typename SizeT = uint32_t>
using list_data_base = ivf::list_base<uint8_t, IdxT, SizeT>;

using pq_centers_extents =
  raft::extents<uint32_t, raft::dynamic_extent, raft::dynamic_extent, raft::dynamic_extent>;

template <typename IdxT>
class index_iface {
 public:
  virtual ~index_iface() = default;

  virtual cuvs::distance::DistanceType metric() const noexcept  = 0;
  virtual codebook_gen codebook_kind() const noexcept           = 0;
  virtual list_layout codes_layout() const noexcept             = 0;
  virtual IdxT size() const noexcept                            = 0;
  virtual uint32_t dim() const noexcept                         = 0;
  virtual uint32_t dim_ext() const noexcept                     = 0;
  virtual uint32_t rot_dim() const noexcept                     = 0;
  virtual uint32_t pq_bits() const noexcept                     = 0;
  virtual uint32_t pq_dim() const noexcept                      = 0;
  virtual uint32_t pq_len() const noexcept                      = 0;
  virtual uint32_t pq_book_size() const noexcept                = 0;
  virtual uint32_t n_lists() const noexcept                     = 0;
  virtual bool conservative_memory_allocation() const noexcept  = 0;
  virtual uint32_t get_list_size_in_bytes(uint32_t label) const = 0;

  virtual std::vector<std::shared_ptr<list_data_base<IdxT>>>& lists() noexcept             = 0;
  virtual const std::vector<std::shared_ptr<list_data_base<IdxT>>>& lists() const noexcept = 0;

  virtual raft::device_vector_view<uint32_t, uint32_t, raft::row_major> list_sizes() noexcept = 0;
  virtual raft::device_vector_view<const uint32_t, uint32_t, raft::row_major> list_sizes()
    const noexcept = 0;

  virtual raft::device_vector_view<uint8_t*, uint32_t, raft::row_major> data_ptrs() noexcept = 0;
  virtual raft::device_vector_view<const uint8_t* const, uint32_t, raft::row_major> data_ptrs()
    const noexcept = 0;

  virtual raft::device_vector_view<IdxT*, uint32_t, raft::row_major> inds_ptrs() noexcept = 0;
  virtual raft::device_vector_view<const IdxT* const, uint32_t, raft::row_major> inds_ptrs()
    const noexcept = 0;

  virtual raft::host_vector_view<IdxT, uint32_t, raft::row_major> accum_sorted_sizes() noexcept = 0;
  virtual raft::host_vector_view<const IdxT, uint32_t, raft::row_major> accum_sorted_sizes()
    const noexcept = 0;

  virtual raft::device_mdspan<const float, pq_centers_extents, raft::row_major> pq_centers()
    const noexcept = 0;

  virtual raft::device_matrix_view<const float, uint32_t, raft::row_major> centers()
    const noexcept = 0;

  virtual raft::device_matrix_view<const float, uint32_t, raft::row_major> centers_rot()
    const noexcept = 0;

  virtual raft::device_matrix_view<const float, uint32_t, raft::row_major> rotation_matrix()
    const noexcept = 0;

  virtual raft::device_matrix_view<const int8_t, uint32_t, raft::row_major> rotation_matrix_int8(
    const raft::resources& res) const = 0;
  virtual raft::device_matrix_view<const half, uint32_t, raft::row_major> rotation_matrix_half(
    const raft::resources& res) const = 0;
  virtual raft::device_matrix_view<const int8_t, uint32_t, raft::row_major> centers_int8(
    const raft::resources& res) const = 0;
  virtual raft::device_matrix_view<const half, uint32_t, raft::row_major> centers_half(
    const raft::resources& res) const = 0;
};

/**
 * @defgroup ivf_pq_cpp_index IVF-PQ index
 * @{
 */
/**
 * @brief IVF-PQ index.
 *
 * In the IVF-PQ index, a database vector y is approximated with two level quantization:
 *
 * y = Q_1(y) + Q_2(y - Q_1(y))
 *
 * The first level quantizer (Q_1), maps the vector y to the nearest cluster center. The number of
 * clusters is n_lists.
 *
 * The second quantizer encodes the residual, and it is defined as a product quantizer [1].
 *
 * A product quantizer encodes a `dim` dimensional vector with a `pq_dim` dimensional vector.
 * First we split the input vector into `pq_dim` subvectors (denoted by u), where each u vector
 * contains `pq_len` distinct components of y
 *
 * y_1, y_2, ... y_{pq_len}, y_{pq_len+1}, ... y_{2*pq_len}, ... y_{dim-pq_len+1} ... y_{dim}
 *  \___________________/     \____________________________/      \______________________/
 *         u_1                         u_2                          u_{pq_dim}
 *
 * Then each subvector encoded with a separate quantizer q_i, end the results are concatenated
 *
 * Q_2(y) = q_1(u_1),q_2(u_2),...,q_{pq_dim}(u_pq_dim})
 *
 * Each quantizer q_i outputs a code with pq_bit bits. The second level quantizers are also defined
 * by k-means clustering in the corresponding sub-space: the reproduction values are the centroids,
 * and the set of reproduction values is the codebook.
 *
 * When the data dimensionality `dim` is not multiple of `pq_dim`, the feature space is transformed
 * using a random orthogonal matrix to have `rot_dim = pq_dim * pq_len` dimensions
 * (`rot_dim >= dim`).
 *
 * The second-level quantizers are trained either for each subspace or for each cluster:
 *   (a) codebook_gen::PER_SUBSPACE:
 *         creates `pq_dim` second-level quantizers - one for each slice of the data along features;
 *   (b) codebook_gen::PER_CLUSTER:
 *         creates `n_lists` second-level quantizers - one for each first-level cluster.
 * In either case, the centroids are again found using k-means clustering interpreting the data as
 * having pq_len dimensions.
 *
 * [1] Product quantization for nearest neighbor search Herve Jegou, Matthijs Douze, Cordelia Schmid
 *
 * @tparam IdxT type of the indices in the source dataset
 *
 */
template <typename IdxT>
class index : public index_iface<IdxT>, cuvs::neighbors::index {
 public:
  using index_params_type  = ivf_pq::index_params;
  using search_params_type = ivf_pq::search_params;
  using index_type         = IdxT;
  static_assert(!raft::is_narrowing_v<uint32_t, IdxT>,
                "IdxT must be able to represent all values of uint32_t");

  index(const index&)                    = delete;
  index(index&&)                         = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index&      = default;
  ~index()                               = default;

  /**
   * @brief Construct an empty index.
   *
   * Constructs an empty index. This index will either need to be trained with `build`
   * or loaded from a saved copy with `deserialize`
   */
  index(raft::resources const& handle);

  /**
   * @brief Construct an index with specified parameters.
   *
   * This constructor creates an owning index with the given parameters.
   *
   * @param handle RAFT resources handle
   * @param metric Distance metric for clustering
   * @param codebook_kind How PQ codebooks are created
   * @param n_lists Number of inverted lists (clusters)
   * @param dim Dimensionality of the input data
   * @param pq_bits Bit length of vector elements after PQ compression
   * @param pq_dim Dimensionality after PQ compression (0 = auto-select)
   * @param conservative_memory_allocation Memory allocation strategy
   */
  index(raft::resources const& handle,
        cuvs::distance::DistanceType metric,
        codebook_gen codebook_kind,
        uint32_t n_lists,
        uint32_t dim,
        uint32_t pq_bits                    = 8,
        uint32_t pq_dim                     = 0,
        bool conservative_memory_allocation = false);

  /**
   * @brief Construct an index from index parameters.
   *
   * @param handle RAFT resources handle
   * @param params Index parameters
   * @param dim Dimensionality of the input data
   */
  index(raft::resources const& handle, const index_params& params, uint32_t dim);

  /** Total length of the index. */
  IdxT size() const noexcept override;

  /** Dimensionality of the input data. */
  uint32_t dim() const noexcept override;

  /**
   * Dimensionality of the cluster centers:
   * input data dim extended with vector norms and padded to 8 elems.
   */
  uint32_t dim_ext() const noexcept;

  /**
   * Dimensionality of the data after transforming it for PQ processing
   * (rotated and augmented to be muplitple of `pq_dim`).
   */
  uint32_t rot_dim() const noexcept;

  /** The bit length of an encoded vector element after compression by PQ. */
  uint32_t pq_bits() const noexcept override;

  /** The dimensionality of an encoded vector after compression by PQ. */
  uint32_t pq_dim() const noexcept override;

  /** Dimensionality of a subspace, i.e. the number of vector components mapped to a subspace */
  uint32_t pq_len() const noexcept;

  /** The number of vectors in a PQ codebook (`1 << pq_bits`). */
  uint32_t pq_book_size() const noexcept;

  /** Distance metric used for clustering. */
  cuvs::distance::DistanceType metric() const noexcept override;

  /** How PQ codebooks are created. */
  codebook_gen codebook_kind() const noexcept override;

  /** Memory layout of PQ codes in IVF lists. */
  list_layout codes_layout() const noexcept override;

  /** Number of clusters/inverted lists (first level quantization). */
  uint32_t n_lists() const noexcept;

  /**
   * Whether to use conservative memory allocation when extending the list (cluster) data
   * (see index_params.conservative_memory_allocation).
   */
  bool conservative_memory_allocation() const noexcept override;

  /**
   * PQ cluster centers
   *
   *   - codebook_gen::PER_SUBSPACE: [pq_dim , pq_len, pq_book_size]
   *   - codebook_gen::PER_CLUSTER:  [n_lists, pq_len, pq_book_size]
   */
  raft::device_mdspan<const float, pq_centers_extents, raft::row_major> pq_centers()
    const noexcept override;

  /** Lists' data and indices (polymorphic, works for both FLAT and INTERLEAVED layouts). */
  std::vector<std::shared_ptr<list_data_base<IdxT>>>& lists() noexcept override;
  const std::vector<std::shared_ptr<list_data_base<IdxT>>>& lists() const noexcept override;

  /** Pointers to the inverted lists (clusters) data  [n_lists]. */
  raft::device_vector_view<uint8_t*, uint32_t, raft::row_major> data_ptrs() noexcept override;
  raft::device_vector_view<const uint8_t* const, uint32_t, raft::row_major> data_ptrs()
    const noexcept override;

  /** Pointers to the inverted lists (clusters) indices  [n_lists]. */
  raft::device_vector_view<IdxT*, uint32_t, raft::row_major> inds_ptrs() noexcept override;
  raft::device_vector_view<const IdxT* const, uint32_t, raft::row_major> inds_ptrs()
    const noexcept override;

  /** The transform matrix (original space -> rotated padded space) [rot_dim, dim] */
  raft::device_matrix_view<const float, uint32_t, raft::row_major> rotation_matrix()
    const noexcept override;

  raft::device_matrix_view<const int8_t, uint32_t, raft::row_major> rotation_matrix_int8(
    const raft::resources& res) const override;
  raft::device_matrix_view<const half, uint32_t, raft::row_major> rotation_matrix_half(
    const raft::resources& res) const override;

  /**
   * Accumulated list sizes, sorted in descending order [n_lists + 1].
   * The last value contains the total length of the index.
   * The value at index zero is always zero.
   *
   * That is, the content of this span is as if the `list_sizes` was sorted and then accumulated.
   *
   * This span is used during search to estimate the maximum size of the workspace.
   */
  raft::host_vector_view<IdxT, uint32_t, raft::row_major> accum_sorted_sizes() noexcept override;
  raft::host_vector_view<const IdxT, uint32_t, raft::row_major> accum_sorted_sizes()
    const noexcept override;

  /** Sizes of the lists [n_lists]. */
  raft::device_vector_view<uint32_t, uint32_t, raft::row_major> list_sizes() noexcept override;
  raft::device_vector_view<const uint32_t, uint32_t, raft::row_major> list_sizes()
    const noexcept override;

  /** Cluster centers corresponding to the lists in the original space [n_lists, dim_ext] */
  raft::device_matrix_view<const float, uint32_t, raft::row_major> centers()
    const noexcept override;

  raft::device_matrix_view<const int8_t, uint32_t, raft::row_major> centers_int8(
    const raft::resources& res) const override;
  raft::device_matrix_view<const half, uint32_t, raft::row_major> centers_half(
    const raft::resources& res) const override;

  /** Cluster centers corresponding to the lists in the rotated space [n_lists, rot_dim] */
  raft::device_matrix_view<const float, uint32_t, raft::row_major> centers_rot()
    const noexcept override;

  /** fetch size of a particular IVF list in bytes using the list extents.
   * Usage example:
   * @code{.cpp}
   *   raft::resources res;
   *   // use default index params
   *   ivf_pq::index_params index_params;
   *   // extend the IVF lists while building the index
   *   index_params.add_data_on_build = true;
   *   // create and fill the index from a [N, D] dataset
   *   auto index = cuvs::neighbors::ivf_pq::build(res, index_params, dataset);
   *   // Fetch the size of the fourth list
   *   uint32_t size = index.get_list_size_in_bytes(3);
   * @endcode
   *
   * @param[in] label list ID
   */
  uint32_t get_list_size_in_bytes(uint32_t label) const override;

  /**
   * @brief Construct index from implementation pointer.
   *
   * This constructor is used internally by build/extend/deserialize functions.
   *
   * @param impl Implementation pointer (owning or view)
   */
  explicit index(std::unique_ptr<index_iface<IdxT>> impl);

  static pq_centers_extents make_pq_centers_extents(
    uint32_t dim, uint32_t pq_dim, uint32_t pq_bits, codebook_gen codebook_kind, uint32_t n_lists);

  static uint32_t calculate_pq_dim(uint32_t dim);

 private:
  std::unique_ptr<index_iface<IdxT>> impl_;
};
/**
 * @}
 */

/**
 * @defgroup ivf_pq_cpp_index_build IVF-PQ index build
 * @{
 */
/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a device matrix view to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-pq index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   ivf_pq::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_pq::build(handle, index_params, dataset, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::device_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_pq::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_pq::index<int64_t>* idx);

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a device matrix view to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-pq index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::device_matrix_view<const half, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   ivf_pq::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_pq::build(handle, index_params, dataset, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::device_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_pq::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_pq::index<int64_t>* idx);
/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a device matrix view to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-pq index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   ivf_pq::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_pq::build(handle, index_params, dataset, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::device_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_pq::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_pq::index<int64_t>* idx);

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a device matrix view to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-pq index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   ivf_pq::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_pq::build(handle, index_params, dataset, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::device_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_pq::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_pq::index<int64_t>* idx);
/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Note, if index_params.add_data_on_build is set to true, the user can set a
 * stream pool in the input raft::resource with at least one stream to enable kernel and copy
 * overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a host_matrix_view to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-pq index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Note, if index_params.add_data_on_build is set to true, the user can set a
 * stream pool in the input raft::resource with at least one stream to enable kernel and copy
 * overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   ivf_pq::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_pq::build(handle, index_params, dataset, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::host_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_pq::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_pq::index<int64_t>* idx);

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Note, if index_params.add_data_on_build is set to true, the user can set a
 * stream pool in the input raft::resource with at least one stream to enable kernel and copy
 * overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a host_matrix_view to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-pq index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::host_matrix_view<const half, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   ivf_pq::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_pq::build(handle, index_params, dataset, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::host_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_pq::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::host_matrix_view<const half, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_pq::index<int64_t>* idx);

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a host_matrix_view to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-pq index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Note, if index_params.add_data_on_build is set to true, the user can set a
 * stream pool in the input raft::resource with at least one stream to enable kernel and copy
 * overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   ivf_pq::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_pq::build(handle, index_params, dataset, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::host_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_pq::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_pq::index<int64_t>* idx);

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Note, if index_params.add_data_on_build is set to true, the user can set a
 * stream pool in the input raft::resource with at least one stream to enable kernel and copy
 * overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset a host_matrix_view to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-pq index
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 * - CosineExpanded
 *
 * Note, if index_params.add_data_on_build is set to true, the user can set a
 * stream pool in the input raft::resource with at least one stream to enable kernel and copy
 * overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping. This is only applicable if index_params.add_data_on_build is set to true
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // create and fill the index from a [N, D] dataset
 *   ivf_pq::index<decltype(dataset::value_type), decltype(dataset::index_type)> index;
 *   ivf_pq::build(handle, index_params, dataset, index);
 * @endcode
 *
 * @param[in] handle
 * @param[in] index_params configure the index building
 * @param[in] dataset raft::host_matrix_view to a row-major matrix [n_rows, dim]
 * @param[out] idx reference to ivf_pq::index
 *
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_pq::index<int64_t>* idx);

/**
 * @brief Build a view-type IVF-PQ index from device memory centroids and codebook.
 *
 * This function creates a non-owning index that stores a reference to the provided device data.
 * All parameters must be provided with correct extents. The caller is responsible for ensuring
 * the lifetime of the input data exceeds the lifetime of the returned index.
 *
 * The index_params must be consistent with the provided matrices. Specifically:
 * - index_params.codebook_kind determines the expected shape of pq_centers
 * - index_params.metric will be stored in the index
 * - index_params.conservative_memory_allocation will be stored in the index
 * The function will verify consistency between index_params, dim, and the matrix extents.
 *
 * @param[in] handle raft resources handle
 * @param[in] index_params configure the index (metric, codebook_kind, etc.). Must be consistent
 *   with the provided matrices.
 * @param[in] dim dimensionality of the input data
 * @param[in] pq_centers PQ codebook on device memory with required extents:
 *   - codebook_gen::PER_SUBSPACE: [pq_dim, pq_len, pq_book_size]
 *   - codebook_gen::PER_CLUSTER:  [n_lists, pq_len, pq_book_size]
 * @param[in] centers Cluster centers in the original space [n_lists, dim_ext]
 *   where dim_ext = round_up(dim + 1, 8)
 * @param[in] centers_rot Rotated cluster centers [n_lists, rot_dim]
 *   where rot_dim = pq_len * pq_dim
 * @param[in] rotation_matrix Transform matrix (original space -> rotated padded space) [rot_dim,
 * dim]
 *
 * @return A view-type ivf_pq index that references the provided data
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           const uint32_t dim,
           raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
           raft::device_matrix_view<const float, uint32_t, raft::row_major> centers,
           raft::device_matrix_view<const float, uint32_t, raft::row_major> centers_rot,
           raft::device_matrix_view<const float, uint32_t, raft::row_major> rotation_matrix)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Build an IVF-PQ index from device memory centroids and codebook.
 *
 * This function creates a non-owning index that references the provided device data directly.
 * All parameters must be provided with correct extents. The caller is responsible for ensuring
 * the lifetime of the input data exceeds the lifetime of the returned index.
 *
 * The index_params must be consistent with the provided matrices. Specifically:
 * - index_params.codebook_kind determines the expected shape of pq_centers
 * - index_params.metric will be stored in the index
 * - index_params.conservative_memory_allocation will be stored in the index
 * The function will verify consistency between index_params, dim, and the matrix extents.
 *
 * @param[in] handle raft resources handle
 * @param[in] index_params configure the index (metric, codebook_kind, etc.). Must be consistent
 *   with the provided matrices.
 * @param[in] dim dimensionality of the input data
 * @param[in] pq_centers PQ codebook on device memory with required extents:
 *   - codebook_gen::PER_SUBSPACE: [pq_dim, pq_len, pq_book_size]
 *   - codebook_gen::PER_CLUSTER:  [n_lists, pq_len, pq_book_size]
 * @param[in] centers Cluster centers in the original space [n_lists, dim_ext]
 *   where dim_ext = round_up(dim + 1, 8)
 * @param[in] centers_rot Rotated cluster centers [n_lists, rot_dim]
 *   where rot_dim = pq_len * pq_dim
 * @param[in] rotation_matrix Transform matrix (original space -> rotated padded space) [rot_dim,
 * dim]
 * @param[out] idx pointer to ivf_pq::index
 */
void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_pq::index_params& index_params,
           const uint32_t dim,
           raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
           raft::device_matrix_view<const float, uint32_t, raft::row_major> centers,
           raft::device_matrix_view<const float, uint32_t, raft::row_major> centers_rot,
           raft::device_matrix_view<const float, uint32_t, raft::row_major> rotation_matrix,
           cuvs::neighbors::ivf_pq::index<int64_t>* idx);

/**
 * @brief Build an IVF-PQ index from host memory centroids and codebook (in-place).
 *
 * @param[in] handle raft resources handle
 * @param[in] index_params configure the index building
 * @param[in] dim dimensionality of the input data
 * @param[in] pq_centers PQ codebook
 * @param[in] centers Cluster centers
 * @param[in] centers_rot Optional rotated cluster centers
 * @param[in] rotation_matrix Optional rotation matrix
 */
auto build(
  raft::resources const& handle,
  const cuvs::neighbors::ivf_pq::index_params& index_params,
  const uint32_t dim,
  raft::host_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
  raft::host_matrix_view<const float, uint32_t, raft::row_major> centers,
  std::optional<raft::host_matrix_view<const float, uint32_t, raft::row_major>> centers_rot,
  std::optional<raft::host_matrix_view<const float, uint32_t, raft::row_major>> rotation_matrix)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Build an IVF-PQ index from host memory centroids and codebook (in-place).
 *
 * @param[in] handle raft resources handle
 * @param[in] index_params configure the index building
 * @param[in] dim dimensionality of the input data
 * @param[in] pq_centers PQ codebook on host memory
 * @param[in] centers Cluster centers on host memory
 * @param[in] centers_rot Optional rotated cluster centers on host
 * @param[in] rotation_matrix Optional rotation matrix on host
 * @param[out] idx pointer to IVF-PQ index to be built
 */
void build(
  raft::resources const& handle,
  const cuvs::neighbors::ivf_pq::index_params& index_params,
  const uint32_t dim,
  raft::host_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
  raft::host_matrix_view<const float, uint32_t, raft::row_major> centers,
  std::optional<raft::host_matrix_view<const float, uint32_t, raft::row_major>> centers_rot,
  std::optional<raft::host_matrix_view<const float, uint32_t, raft::row_major>> rotation_matrix,
  cuvs::neighbors::ivf_pq::index<int64_t>* idx);
/**
 * @}
 */

/**
 * @defgroup ivf_pq_cpp_index_extend IVF-PQ index extend
 * @{
 */
/**
 * @brief Extend the index with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_pq::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a device matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a device vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
auto extend(raft::resources const& handle,
            raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Extend the index with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_pq::extend(handle, new_vectors, no_op, &index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a device matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a device vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
void extend(raft::resources const& handle,
            raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_pq::index<int64_t>* idx);

/**
 * @brief Extend the index with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_pq::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a device matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a device vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
auto extend(raft::resources const& handle,
            raft::device_matrix_view<const half, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Extend the index with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_pq::extend(handle, new_vectors, no_op, &index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a device matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a device vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
void extend(raft::resources const& handle,
            raft::device_matrix_view<const half, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_pq::index<int64_t>* idx);

/**
 * @brief Extend the index with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_pq::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a device matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a device vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
auto extend(raft::resources const& handle,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Extend the index with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_pq::extend(handle, new_vectors, no_op, &index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a device matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a device vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
void extend(raft::resources const& handle,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_pq::index<int64_t>* idx);

/**
 * @brief Extend the index with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_pq::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a device matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a device vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
auto extend(raft::resources const& handle,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Extend the index with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // fill the index with the data
 *   std::optional<raft::device_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_pq::extend(handle, new_vectors, no_op, &index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a device matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a device vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
void extend(raft::resources const& handle,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_pq::index<int64_t>* idx);

/**
 * @brief Extend the index with the new data.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_pq::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a host matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a host vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
auto extend(raft::resources const& handle,
            raft::host_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Extend the index with the new data.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_pq::extend(handle, new_vectors, no_op, &index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a host matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a host vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
void extend(raft::resources const& handle,
            raft::host_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_pq::index<int64_t>* idx);

/**
 * @brief Extend the index with the new data.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_pq::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a host matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a host vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
auto extend(raft::resources const& handle,
            raft::host_matrix_view<const half, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Extend the index with the new data.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_pq::extend(handle, new_vectors, no_op, &index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a host matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a host vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
void extend(raft::resources const& handle,
            raft::host_matrix_view<const half, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_pq::index<int64_t>* idx);

/**
 * @brief Extend the index with the new data.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_pq::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a host matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a host vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
auto extend(raft::resources const& handle,
            raft::host_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Extend the index with the new data.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_pq::extend(handle, new_vectors, no_op, &index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a host matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a host vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
void extend(raft::resources const& handle,
            raft::host_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_pq::index<int64_t>* idx);

/**
 * @brief Extend the index with the new data.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   auto index = ivf_pq::extend(handle, new_vectors, no_op, index_empty);
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a host matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a host vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
auto extend(raft::resources const& handle,
            raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
  -> cuvs::neighbors::ivf_pq::index<int64_t>;

/**
 * @brief Extend the index with the new data.
 *
 * Note, the user can set a stream pool in the input raft::resource with
 * at least one stream to enable kernel and copy overlapping.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset);
 *   // optional: create a stream pool with at least one stream to enable kernel and copy
 *   // overlapping
 *   raft::resource::set_cuda_stream_pool(handle, std::make_shared<rmm::cuda_stream_pool>(1));
 *   // fill the index with the data
 *   std::optional<raft::host_vector_view<const IdxT, IdxT>> no_op = std::nullopt;
 *   ivf_pq::extend(handle, new_vectors, no_op, &index_empty);
 *
 * @endcode
 *
 * @param[in] handle
 * @param[in] new_vectors a host matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a host vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
void extend(raft::resources const& handle,
            raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            cuvs::neighbors::ivf_pq::index<int64_t>* idx);
/**
 * @}
 */

/**
 * @defgroup ivf_pq_cpp_index_search IVF-PQ index search
 * @{
 */
/**
 * @brief Search ANN using the constructed index.
 *
 * See the [ivf_pq::build](#ivf_pq::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`.
 * The exact size of the temporary buffer depends on multiple factors and is an implementation
 * detail. However, you can safely specify a small initial size for the memory pool, so that only a
 * few allocations happen to grow it during the first invocations of the `search`.
 *
 * @code{.cpp}
 *   ...
 *   // use default search parameters
 *   ivf_pq::search_params search_params;
 *   // Use the same allocator across multiple searches to reduce the number of
 *   // cuda memory allocations
 *   ivf_pq::search(handle, search_params, index, queries1, out_inds1, out_dists1);
 *   ivf_pq::search(handle, search_params, index, queries2, out_inds2, out_dists2);
 *   ivf_pq::search(handle, search_params, index, queries3, out_inds3, out_dists3);
 *   ...
 * @endcode
 *
 * @param[in] handle
 * @param[in] search_params configure the search
 * @param[in] index ivf-pq constructed index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering)
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_pq::search_params& search_params,
            const cuvs::neighbors::ivf_pq::index<int64_t>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [ivf_pq::build](#ivf_pq::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`.
 * The exact size of the temporary buffer depends on multiple factors and is an implementation
 * detail. However, you can safely specify a small initial size for the memory pool, so that only a
 * few allocations happen to grow it during the first invocations of the `search`.
 *
 * @code{.cpp}
 *   ...
 *   // use default search parameters
 *   ivf_pq::search_params search_params;
 *   // Use the same allocator across multiple searches to reduce the number of
 *   // cuda memory allocations
 *   ivf_pq::search(handle, search_params, index, queries1, out_inds1, out_dists1);
 *   ivf_pq::search(handle, search_params, index, queries2, out_inds2, out_dists2);
 *   ivf_pq::search(handle, search_params, index, queries3, out_inds3, out_dists3);
 *   ...
 * @endcode
 *
 * @param[in] handle
 * @param[in] search_params configure the search
 * @param[in] index ivf-pq constructed index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering)
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_pq::search_params& search_params,
            const cuvs::neighbors::ivf_pq::index<int64_t>& index,
            raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [ivf_pq::build](#ivf_pq::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`.
 * The exact size of the temporary buffer depends on multiple factors and is an implementation
 * detail. However, you can safely specify a small initial size for the memory pool, so that only a
 * few allocations happen to grow it during the first invocations of the `search`.
 *
 * @code{.cpp}
 *   ...
 *   // use default search parameters
 *   ivf_pq::search_params search_params;
 *   // Use the same allocator across multiple searches to reduce the number of
 *   // cuda memory allocations
 *   ivf_pq::search(handle, search_params, index, queries1, out_inds1, out_dists1);
 *   ivf_pq::search(handle, search_params, index, queries2, out_inds2, out_dists2);
 *   ivf_pq::search(handle, search_params, index, queries3, out_inds3, out_dists3);
 *   ...
 * @endcode
 *
 * @param[in] handle
 * @param[in] search_params configure the search
 * @param[in] index ivf-pq constructed index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering)
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_pq::search_params& search_params,
            const cuvs::neighbors::ivf_pq::index<int64_t>& index,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [ivf_pq::build](#ivf_pq::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`.
 * The exact size of the temporary buffer depends on multiple factors and is an implementation
 * detail. However, you can safely specify a small initial size for the memory pool, so that only a
 * few allocations happen to grow it during the first invocations of the `search`.
 *
 * @code{.cpp}
 *   ...
 *   // use default search parameters
 *   ivf_pq::search_params search_params;
 *   // Use the same allocator across multiple searches to reduce the number of
 *   // cuda memory allocations
 *   ivf_pq::search(handle, search_params, index, queries1, out_inds1, out_dists1);
 *   ivf_pq::search(handle, search_params, index, queries2, out_inds2, out_dists2);
 *   ivf_pq::search(handle, search_params, index, queries3, out_inds3, out_dists3);
 *   ...
 * @endcode
 *
 * @param[in] handle
 * @param[in] search_params configure the search
 * @param[in] index ivf-pq constructed index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter an optional device filter function object that greenlights samples
 * for a given query. (none_sample_filter for no filtering)
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_pq::search_params& search_params,
            const cuvs::neighbors::ivf_pq::index<int64_t>& index,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @}
 */

/**
 * @defgroup ivf_pq_cpp_transform IVF-PQ index transform
 * @{
 */
/**
 * @brief Transform a dataset by applying pq-encoding to each vector
 *
 * @param[in] handle
 * @param[in] index ivf-pq constructed index
 * @param[in] dataset a device matrix view to a row-major matrix [n_rows, index.dim()]
 * @param[out] output_labels a device vector view [n_rows] that will get populaterd with the
 * cluster ids (labels) for each vector in the input dataset
 * @param[out] output_dataset a device matrix view [n_rows, ceildiv(index.pq_dim() *
 * index.pq_bits(), 8)]] that will get populated with the pq-encoded dataset
 */
void transform(raft::resources const& handle,
               const cuvs::neighbors::ivf_pq::index<int64_t>& index,
               raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
               raft::device_vector_view<uint32_t, int64_t> output_labels,
               raft::device_matrix_view<uint8_t, int64_t> output_dataset);
/** @copydoc transform */
void transform(raft::resources const& handle,
               const cuvs::neighbors::ivf_pq::index<int64_t>& index,
               raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
               raft::device_vector_view<uint32_t, int64_t> output_labels,
               raft::device_matrix_view<uint8_t, int64_t> output_dataset);
/** @copydoc transform */
void transform(raft::resources const& handle,
               const cuvs::neighbors::ivf_pq::index<int64_t>& index,
               raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
               raft::device_vector_view<uint32_t, int64_t> output_labels,
               raft::device_matrix_view<uint8_t, int64_t> output_dataset);
/** @copydoc transform */
void transform(raft::resources const& handle,
               const cuvs::neighbors::ivf_pq::index<int64_t>& index,
               raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
               raft::device_vector_view<uint32_t, int64_t> output_labels,
               raft::device_matrix_view<uint8_t, int64_t> output_dataset);
/**
 * @}
 */

/**
 * @defgroup ivf_pq_cpp_serialize IVF-PQ index serialize
 * @{
 */
/**
 * Write the index to an output stream
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = ivf_pq::build(...);`
 * cuvs::neighbors::ivf_pq::serialize(handle, os, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index IVF-PQ index
 *
 */
void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::ivf_pq::index<int64_t>& index);

/**
 * Save the index to file.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = ivf_pq::build(...);`
 * cuvs::neighbors::ivf_pq::serialize(handle, filename, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index IVF-PQ index
 *
 */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::ivf_pq::index<int64_t>& index);

/**
 * Load index from input stream
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 *
 * raft::resources handle;
 *
 * // create an input stream
 * std::istream is(std::cin.rdbuf());
 *
 * using IdxT = int64_t; // type of the index
 * // create an empty index
 * cuvs::neighbors::ivf_pq::index<IdxT> index(handle);
 *
 * cuvs::neighbors::ivf_pq::deserialize(handle, is, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] str the name of the file that stores the index
 * @param[out] index IVF-PQ index
 *
 */
void deserialize(raft::resources const& handle,
                 std::istream& str,
                 cuvs::neighbors::ivf_pq::index<int64_t>* index);
/**
 * Load index from file.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * using IdxT = int64_t; // type of the index
 * // create an empty index
 * ivf_pq::index<IdxT> index(handle);
 *
 * cuvs::neighbors::ivf_pq::deserialize(handle, filename, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[out] index IVF-PQ index
 *
 */
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::ivf_pq::index<int64_t>* index);
/**
 * @}
 */

/// \defgroup mg_cpp_index_build ANN MG index build

/// \ingroup mg_cpp_index_build
/**
 * @brief Builds a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_pq::build(clique, index_params, index_dataset);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed IVF-PQ MG index
 */
auto build(const raft::resources& clique,
           const cuvs::neighbors::mg_index_params<ivf_pq::index_params>& index_params,
           raft::host_matrix_view<const float, int64_t, row_major> index_dataset)
  -> cuvs::neighbors::mg_index<ivf_pq::index<int64_t>, float, int64_t>;

/// \ingroup mg_cpp_index_build
/**
 * @brief Builds a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_pq::build(clique, index_params, index_dataset);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed IVF-PQ MG index
 */
auto build(const raft::resources& clique,
           const cuvs::neighbors::mg_index_params<ivf_pq::index_params>& index_params,
           raft::host_matrix_view<const half, int64_t, row_major> index_dataset)
  -> cuvs::neighbors::mg_index<ivf_pq::index<int64_t>, half, int64_t>;

/// \ingroup mg_cpp_index_build
/**
 * @brief Builds a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_pq::build(clique, index_params, index_dataset);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed IVF-PQ MG index
 */
auto build(const raft::resources& clique,
           const cuvs::neighbors::mg_index_params<ivf_pq::index_params>& index_params,
           raft::host_matrix_view<const int8_t, int64_t, row_major> index_dataset)
  -> cuvs::neighbors::mg_index<ivf_pq::index<int64_t>, int8_t, int64_t>;

/// \ingroup mg_cpp_index_build
/**
 * @brief Builds a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_pq::build(clique, index_params, index_dataset);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index_params configure the index building
 * @param[in] index_dataset a row-major matrix on host [n_rows, dim]
 *
 * @return the constructed IVF-PQ MG index
 */
auto build(const raft::resources& clique,
           const cuvs::neighbors::mg_index_params<ivf_pq::index_params>& index_params,
           raft::host_matrix_view<const uint8_t, int64_t, row_major> index_dataset)
  -> cuvs::neighbors::mg_index<ivf_pq::index<int64_t>, uint8_t, int64_t>;

/// \defgroup mg_cpp_index_extend ANN MG index extend

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_pq::build(clique, index_params, index_dataset);
 * cuvs::neighbors::ivf_pq::extend(clique, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::resources& clique,
            cuvs::neighbors::mg_index<ivf_pq::index<int64_t>, float, int64_t>& index,
            raft::host_matrix_view<const float, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_pq::build(clique, index_params, index_dataset);
 * cuvs::neighbors::ivf_pq::extend(clique, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::resources& clique,
            cuvs::neighbors::mg_index<ivf_pq::index<int64_t>, half, int64_t>& index,
            raft::host_matrix_view<const half, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_pq::build(clique, index_params, index_dataset);
 * cuvs::neighbors::ivf_pq::extend(clique, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::resources& clique,
            cuvs::neighbors::mg_index<ivf_pq::index<int64_t>, int8_t, int64_t>& index,
            raft::host_matrix_view<const int8_t, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices);

/// \ingroup mg_cpp_index_extend
/**
 * @brief Extends a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_pq::build(clique, index_params, index_dataset);
 * cuvs::neighbors::ivf_pq::extend(clique, index, new_vectors, std::nullopt);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] new_vectors a row-major matrix on host [n_rows, dim]
 * @param[in] new_indices optional vector on host [n_rows],
 * `std::nullopt` means default continuous range `[0...n_rows)`
 *
 */
void extend(const raft::resources& clique,
            cuvs::neighbors::mg_index<ivf_pq::index<int64_t>, uint8_t, int64_t>& index,
            raft::host_matrix_view<const uint8_t, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices);

/// \defgroup mg_cpp_index_search ANN MG index search

/// \ingroup mg_cpp_index_search
/**
 * @brief Searches a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_pq::build(clique, index_params, index_dataset);
 * cuvs::neighbors::mg_search_params<ivf_pq::search_params> search_params;
 * cuvs::neighbors::ivf_pq::search(clique, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 *
 */
void search(const raft::resources& clique,
            const cuvs::neighbors::mg_index<ivf_pq::index<int64_t>, float, int64_t>& index,
            const cuvs::neighbors::mg_search_params<ivf_pq::search_params>& search_params,
            raft::host_matrix_view<const float, int64_t, row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances);

/// \ingroup mg_cpp_index_search
/**
 * @brief Searches a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_pq::build(clique, index_params, index_dataset);
 * cuvs::neighbors::mg_search_params<ivf_pq::search_params> search_params;
 * cuvs::neighbors::ivf_pq::search(clique, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 *
 */
void search(const raft::resources& clique,
            const cuvs::neighbors::mg_index<ivf_pq::index<int64_t>, half, int64_t>& index,
            const cuvs::neighbors::mg_search_params<ivf_pq::search_params>& search_params,
            raft::host_matrix_view<const half, int64_t, row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances);

/// \ingroup mg_cpp_index_search
/**
 * @brief Searches a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_pq::build(clique, index_params, index_dataset);
 * cuvs::neighbors::mg_search_params<ivf_pq::search_params> search_params;
 * cuvs::neighbors::ivf_pq::search(clique, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 *
 */
void search(const raft::resources& clique,
            const cuvs::neighbors::mg_index<ivf_pq::index<int64_t>, int8_t, int64_t>& index,
            const cuvs::neighbors::mg_search_params<ivf_pq::search_params>& search_params,
            raft::host_matrix_view<const int8_t, int64_t, row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances);

/// \ingroup mg_cpp_index_search
/**
 * @brief Searches a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_pq::build(clique, index_params, index_dataset);
 * cuvs::neighbors::mg_search_params<ivf_pq::search_params> search_params;
 * cuvs::neighbors::ivf_pq::search(clique, index, search_params, queries, neighbors,
 * distances);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] search_params configure the index search
 * @param[in] queries a row-major matrix on host [n_rows, dim]
 * @param[out] neighbors a row-major matrix on host [n_rows, n_neighbors]
 * @param[out] distances a row-major matrix on host [n_rows, n_neighbors]
 *
 */
void search(const raft::resources& clique,
            const cuvs::neighbors::mg_index<ivf_pq::index<int64_t>, uint8_t, int64_t>& index,
            const cuvs::neighbors::mg_search_params<ivf_pq::search_params>& search_params,
            raft::host_matrix_view<const uint8_t, int64_t, row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances);

/// \defgroup mg_cpp_serialize ANN MG index serialization

/// \ingroup mg_cpp_serialize
/**
 * @brief Serializes a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_pq::build(clique, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::ivf_pq::serialize(clique, index, filename);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(const raft::resources& clique,
               const cuvs::neighbors::mg_index<ivf_pq::index<int64_t>, float, int64_t>& index,
               const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 * @brief Serializes a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_pq::build(clique, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::ivf_pq::serialize(clique, index, filename);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(const raft::resources& clique,
               const cuvs::neighbors::mg_index<ivf_pq::index<int64_t>, half, int64_t>& index,
               const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 * @brief Serializes a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_pq::build(clique, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::ivf_pq::serialize(clique, index, filename);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(const raft::resources& clique,
               const cuvs::neighbors::mg_index<ivf_pq::index<int64_t>, int8_t, int64_t>& index,
               const std::string& filename);

/// \ingroup mg_cpp_serialize
/**
 * @brief Serializes a multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_pq::build(clique, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::ivf_pq::serialize(clique, index, filename);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] index the pre-built index
 * @param[in] filename path to the file to be serialized
 *
 */
void serialize(const raft::resources& clique,
               const cuvs::neighbors::mg_index<ivf_pq::index<int64_t>, uint8_t, int64_t>& index,
               const std::string& filename);

/// \ingroup mg_cpp_deserialize
/**
 * @brief Deserializes an IVF-PQ multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::mg_index_params<ivf_pq::index_params> index_params;
 * auto index = cuvs::neighbors::ivf_pq::build(clique, index_params, index_dataset);
 * const std::string filename = "mg_index.cuvs";
 * cuvs::neighbors::ivf_pq::serialize(clique, index, filename);
 * auto new_index = cuvs::neighbors::ivf_pq::deserialize<float, int64_t>(clique, filename);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] filename path to the file to be deserialized
 *
 */
template <typename T, typename IdxT>
auto deserialize(const raft::resources& clique, const std::string& filename)
  -> cuvs::neighbors::mg_index<ivf_pq::index<IdxT>, T, IdxT>;

/// \defgroup mg_cpp_distribute ANN MG local index distribution

/// \ingroup mg_cpp_distribute
/**
 * @brief Replicates a locally built and serialized IVF-PQ index to all GPUs to form a distributed
 * multi-GPU index
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources_snmg clique;
 * cuvs::neighbors::ivf_pq::index_params index_params;
 * auto index = cuvs::neighbors::ivf_pq::build(clique, index_params, index_dataset);
 * const std::string filename = "local_index.cuvs";
 * cuvs::neighbors::ivf_pq::serialize(clique, filename, index);
 * auto new_index = cuvs::neighbors::ivf_pq::distribute<float, int64_t>(clique, filename);
 * @endcode
 *
 * @param[in] clique a `raft::resources` object specifying the NCCL clique configuration
 * @param[in] filename path to the file to be deserialized : a local index
 *
 */
template <typename T, typename IdxT>
auto distribute(const raft::resources& clique, const std::string& filename)
  -> cuvs::neighbors::mg_index<ivf_pq::index<IdxT>, T, IdxT>;

namespace helpers {
/**
 * @defgroup ivf_pq_cpp_helpers IVF-PQ helper methods
 * @{
 */
namespace codepacker {
/**
 * @addtogroup ivf_pq_cpp_helpers
 * @{
 */
/**
 * @brief Unpack `n_take` consecutive records of a single list (cluster) in the compressed index
 * starting at given `offset`.
 *
 * Bit compression is removed, which means output will have pq_dim dimensional vectors (one code per
 * byte, instead of ceildiv(pq_dim * pq_bits, 8) bytes of pq codes).
 *
 * Usage example:
 * @code{.cpp}
 *   auto list_data = index.lists()[label]->data.view();
 *   // allocate the buffer for the output
 *   uint32_t n_take = 4;
 *   auto codes = raft::make_device_matrix<uint8_t>(res, n_take, index.pq_dim());
 *   uint32_t offset = 0;
 *   // unpack n_take elements from the list
 *   ivf_pq::helpers::codepacker::unpack(res, list_data, index.pq_bits(), offset, codes.view());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] list_data block to read from
 * @param[in] pq_bits bit length of encoded vector elements
 * @param[in] offset
 *   How many records in the list to skip.
 * @param[out] codes
 *   the destination buffer [n_take, index.pq_dim()].
 *   The length `n_take` defines how many records to unpack,
 *   it must be smaller than the list size.
 */
void unpack(raft::resources const& res,
            raft::device_mdspan<const uint8_t,
                                list_spec_interleaved<uint32_t, uint32_t>::list_extents,
                                raft::row_major> list_data,
            uint32_t pq_bits,
            uint32_t offset,
            raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> codes);

/**
 * @brief Unpack `n_rows` consecutive records of a single list (cluster) in the compressed index
 * starting at given `offset`. The output codes of a single vector are contiguous, not expanded to
 * one code per byte, which means the output has ceildiv(pq_dim * pq_bits, 8) bytes per PQ encoded
 * vector.
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   auto list_data = index.lists()[label]->data.view();
 *   // allocate the buffer for the output
 *   uint32_t n_rows = 4;
 *   auto codes = raft::make_device_matrix<uint8_t>(
 *     res, n_rows, raft::ceildiv(index.pq_dim() * index.pq_bits(), 8));
 *   uint32_t offset = 0;
 *   // unpack n_rows elements from the list
 *   ivf_pq::helpers::codepacker::unpack_contiguous(
 *     res, list_data, index.pq_bits(), offset, n_rows, index.pq_dim(), codes.data_handle());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] list_data block to read from
 * @param[in] pq_bits bit length of encoded vector elements
 * @param[in] offset
 *   How many records in the list to skip.
 * @param[in] n_rows How many records to unpack
 * @param[in] pq_dim The dimensionality of the PQ compressed records
 * @param[out] codes
 *   the destination buffer [n_rows, ceildiv(pq_dim * pq_bits, 8)].
 *   The length `n_rows` defines how many records to unpack,
 *   it must be smaller than the list size.
 */
void unpack_contiguous(raft::resources const& res,
                       raft::device_mdspan<const uint8_t,
                                           list_spec_interleaved<uint32_t, uint32_t>::list_extents,
                                           raft::row_major> list_data,
                       uint32_t pq_bits,
                       uint32_t offset,
                       uint32_t n_rows,
                       uint32_t pq_dim,
                       uint8_t* codes);

/**
 * Write flat PQ codes into an existing list by the given offset.
 *
 * NB: no memory allocation happens here; the list must fit the data (offset + n_vec).
 *
 * Usage example:
 * @code{.cpp}
 *   auto list_data  = index.lists()[label]->data.view();
 *   // allocate the buffer for the input codes
 *   auto codes = raft::make_device_matrix<uint8_t>(res, n_vec, index.pq_dim());
 *   ... prepare n_vecs to pack into the list in codes ...
 *   // write codes into the list starting from the 42nd position
 *   ivf_pq::helpers::codepacker::pack(
 *       res, make_const_mdspan(codes.view()), index.pq_bits(), 42, list_data);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] codes flat PQ codes, one code per byte [n_vec, pq_dim]
 * @param[in] pq_bits bit length of encoded vector elements
 * @param[in] offset how many records to skip before writing the data into the list
 * @param[in] list_data block to write into
 */
void pack(raft::resources const& res,
          raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes,
          uint32_t pq_bits,
          uint32_t offset,
          raft::device_mdspan<uint8_t,
                              list_spec_interleaved<uint32_t, uint32_t>::list_extents,
                              raft::row_major> list_data);

/**
 * Write flat PQ codes into an existing list by the given offset. The input codes of a single vector
 * are contiguous (not expanded to one code per byte).
 *
 * NB: no memory allocation happens here; the list must fit the data (offset + n_rows records).
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   auto list_data  = index.lists()[label]->data.view();
 *   // allocate the buffer for the input codes
 *   auto codes = raft::make_device_matrix<uint8_t>(
 *     res, n_rows, raft::ceildiv(index.pq_dim() * index.pq_bits(), 8));
 *   ... prepare compressed vectors to pack into the list in codes ...
 *   // write codes into the list starting from the 42nd position. If the current size of the list
 *   // is greater than 42, this will overwrite the codes starting at this offset.
 *   ivf_pq::helpers::codepacker::pack_contiguous(
 *     res, codes.data_handle(), n_rows, index.pq_dim(), index.pq_bits(), 42, list_data);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] codes flat PQ codes, [n_vec, ceildiv(pq_dim * pq_bits, 8)]
 * @param[in] n_rows number of records
 * @param[in] pq_dim
 * @param[in] pq_bits bit length of encoded vector elements
 * @param[in] offset how many records to skip before writing the data into the list
 * @param[in] list_data block to write into
 */
void pack_contiguous(raft::resources const& res,
                     const uint8_t* codes,
                     uint32_t n_rows,
                     uint32_t pq_dim,
                     uint32_t pq_bits,
                     uint32_t offset,
                     raft::device_mdspan<uint8_t,
                                         list_spec_interleaved<uint32_t, uint32_t>::list_extents,
                                         raft::row_major> list_data);

/**
 * Write flat PQ codes into an existing list by the given offset.
 *
 * The list is identified by its label.
 *
 * NB: no memory allocation happens here; the list must fit the data (offset + n_vec).
 *
 * Usage example:
 * @code{.cpp}
 *   // We will write into the 137th cluster
 *   uint32_t label = 137;
 *   // allocate the buffer for the input codes
 *   auto codes = raft::make_device_matrix<const uint8_t>(res, n_vec, index.pq_dim());
 *   ... prepare n_vecs to pack into the list in codes ...
 *   // write codes into the list starting from the 42nd position
 *   ivf_pq::helpers::codepacker::pack_list_data(res, &index, codes_to_pack, label, 42);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] index IVF-PQ index.
 * @param[in] codes flat PQ codes, one code per byte [n_rows, pq_dim]
 * @param[in] label The id of the list (cluster) into which we write.
 * @param[in] offset how many records to skip before writing the data into the list
 */
void pack_list_data(raft::resources const& res,
                    index<int64_t>* index,
                    raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes,
                    uint32_t label,
                    uint32_t offset);

/**
 * Write flat PQ codes into an existing list by the given offset. Use this when the input
 * vectors are PQ encoded and not expanded to one code per byte.
 *
 * The list is identified by its label.
 *
 * NB: no memory allocation happens here; the list into which the vectors are packed must fit offset
 * + n_rows rows.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   raft::resources res;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(res, index_params, dataset, N, D);
 *   // allocate the buffer for n_rows input codes. Each vector occupies
 *   // raft::ceildiv(index.pq_dim() * index.pq_bits(), 8) bytes because
 *   // codes are compressed and without gaps.
 *   auto codes = raft::make_device_matrix<const uint8_t>(
 *     res, n_rows, raft::ceildiv(index.pq_dim() * index.pq_bits(), 8));
 *   ... prepare the compressed vectors to pack into the list in codes ...
 *   // the first n_rows codes in the fourth IVF list are to be overwritten.
 *   uint32_t label = 3;
 *   // write codes into the list starting from the 0th position
 *   ivf_pq::helpers::codepacker::pack_contiguous_list_data(
 *     res, &index, codes.data_handle(), n_rows, label, 0);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-PQ index
 * @param[in] codes flat contiguous PQ codes [n_rows, ceildiv(pq_dim * pq_bits, 8)]
 * @param[in] n_rows how many records to pack
 * @param[in] label The id of the list (cluster) into which we write.
 * @param[in] offset how many records to skip before writing the data into the list
 */
void pack_contiguous_list_data(raft::resources const& res,
                               index<int64_t>* index,
                               uint8_t* codes,
                               uint32_t n_rows,
                               uint32_t label,
                               uint32_t offset);

/**
 * @brief Unpack `n_take` consecutive records of a single list (cluster) in the compressed index
 * starting at given `offset`, one code per byte (independently of pq_bits).
 *
 * Usage example:
 * @code{.cpp}
 *   // We will unpack the fourth cluster
 *   uint32_t label = 3;
 *   // Get the list size
 *   uint32_t list_size = 0;
 *   raft::copy(&list_size, index.list_sizes().data_handle() + label, 1,
 * resource::get_cuda_stream(res)); resource::sync_stream(res);
 *   // allocate the buffer for the output
 *   auto codes = raft::make_device_matrix<uint8_t>(res, list_size, index.pq_dim());
 *   // unpack the whole list
 *   ivf_pq::helpers::codepacker::unpack_list_data(res, index, codes.view(), label, 0);
 * @endcode
 *
 * @param[in] res
 * @param[in] index
 * @param[out] out_codes
 *   the destination buffer [n_take, index.pq_dim()].
 *   The length `n_take` defines how many records to unpack,
 *   it must be smaller than the list size.
 * @param[in] label
 *   The id of the list (cluster) to decode.
 * @param[in] offset
 *   How many records in the list to skip.
 */
void unpack_list_data(raft::resources const& res,
                      const index<int64_t>& index,
                      raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_codes,
                      uint32_t label,
                      uint32_t offset);

/**
 * @brief Unpack a series of records of a single list (cluster) in the compressed index
 * by their in-list offsets, one code per byte (independently of pq_bits).
 *
 * Usage example:
 * @code{.cpp}
 *   // We will unpack the fourth cluster
 *   uint32_t label = 3;
 *   // Create the selection vector
 *   auto selected_indices = raft::make_device_vector<uint32_t>(res, 4);
 *   ... fill the indices ...
 *   resource::sync_stream(res);
 *   // allocate the buffer for the output
 *   auto codes = raft::make_device_matrix<float>(res, selected_indices.size(), index.pq_dim());
 *   // decode the whole list
 *   ivf_pq::helpers::codepacker::unpack_list_data(
 *       res, index, selected_indices.view(), codes.view(), label);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] index IVF-PQ index (passed by reference)
 * @param[in] in_cluster_indices
 *   The offsets of the selected indices within the cluster.
 * @param[out] out_codes
 *   the destination buffer [n_take, index.pq_dim()].
 *   The length `n_take` defines how many records to unpack,
 *   it must be smaller than the list size.
 * @param[in] label
 *   The id of the list (cluster) to decode.
 */
void unpack_list_data(raft::resources const& res,
                      const index<int64_t>& index,
                      raft::device_vector_view<const uint32_t> in_cluster_indices,
                      raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_codes,
                      uint32_t label);

/**
 * @brief Unpack `n_rows` consecutive PQ encoded vectors of a single list (cluster) in the
 * compressed index starting at given `offset`, not expanded to one code per byte. Each code in the
 * output buffer occupies ceildiv(index.pq_dim() * index.pq_bits(), 8) bytes.
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   // We will unpack the whole fourth cluster
 *   uint32_t label = 3;
 *   // Get the list size
 *   uint32_t list_size = 0;
 *   raft::update_host(&list_size, index.list_sizes().data_handle() + label, 1,
 *     raft::resource::get_cuda_stream(res));
 *   raft::resource::sync_stream(res);
 *   // allocate the buffer for the output
 *   auto codes = raft::make_device_matrix<uint8_t>(res, list_size, raft::ceildiv(index.pq_dim() *
 *      index.pq_bits(), 8));
 *   // unpack the whole list
 *   ivf_pq::helpers::codepacker::unpack_contiguous_list_data(res, index, codes.data_handle(),
 *      list_size, label, 0);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] index IVF-PQ index (passed by reference)
 * @param[out] out_codes
 *   the destination buffer [n_rows, ceildiv(index.pq_dim() * index.pq_bits(), 8)].
 *   The length `n_rows` defines how many records to unpack,
 *   offset + n_rows must be smaller than or equal to the list size.
 * @param[in] n_rows how many codes to unpack
 * @param[in] label
 *   The id of the list (cluster) to decode.
 * @param[in] offset
 *   How many records in the list to skip.
 */
void unpack_contiguous_list_data(raft::resources const& res,
                                 const index<int64_t>& index,
                                 uint8_t* out_codes,
                                 uint32_t n_rows,
                                 uint32_t label,
                                 uint32_t offset);

/**
 * @brief Decode `n_take` consecutive records of a single list (cluster) in the compressed index
 * starting at given `offset`.
 *
 * Usage example:
 * @code{.cpp}
 *   // We will reconstruct the fourth cluster
 *   uint32_t label = 3;
 *   // Get the list size
 *   uint32_t list_size = 0;
 *   raft::copy(&list_size, index.list_sizes().data_handle() + label, 1,
 *   resource::get_cuda_stream(res)); resource::sync_stream(res);
 *   // allocate the buffer for the output
 *   auto decoded_vectors = raft::make_device_matrix<float>(res, list_size, index.dim());
 *   // decode the whole list
 *   ivf_pq::helpers::codepacker::reconstruct_list_data(res, index, decoded_vectors.view(), label,
 * 0);
 * @endcode
 *
 * @param[in] res
 * @param[in] index
 * @param[out] out_vectors
 *   the destination buffer [n_take, index.dim()].
 *   The length `n_take` defines how many records to reconstruct,
 *   it must be smaller than the list size.
 * @param[in] label
 *   The id of the list (cluster) to decode.
 * @param[in] offset
 *   How many records in the list to skip.
 */
void reconstruct_list_data(raft::resources const& res,
                           const index<int64_t>& index,
                           raft::device_matrix_view<float, uint32_t, raft::row_major> out_vectors,
                           uint32_t label,
                           uint32_t offset);

void reconstruct_list_data(raft::resources const& res,
                           const index<int64_t>& index,
                           raft::device_matrix_view<half, uint32_t, raft::row_major> out_vectors,
                           uint32_t label,
                           uint32_t offset);

void reconstruct_list_data(raft::resources const& res,
                           const index<int64_t>& index,
                           raft::device_matrix_view<int8_t, uint32_t, raft::row_major> out_vectors,
                           uint32_t label,
                           uint32_t offset);

void reconstruct_list_data(raft::resources const& res,
                           const index<int64_t>& index,
                           raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_vectors,
                           uint32_t label,
                           uint32_t offset);

/**
 * @brief Decode a series of records of a single list (cluster) in the compressed index
 * by their in-list offsets.
 *
 * Usage example:
 * @code{.cpp}
 *   // We will reconstruct the fourth cluster
 *   uint32_t label = 3;
 *   // Create the selection vector
 *   auto selected_indices = raft::make_device_vector<uint32_t>(res, 4);
 *   ... fill the indices ...
 *   resource::sync_stream(res);
 *   // allocate the buffer for the output
 *   auto decoded_vectors = raft::make_device_matrix<float>(
 *                             res, selected_indices.size(), index.dim());
 *   // decode the whole list
 *   ivf_pq::helpers::codepacker::reconstruct_list_data(
 *       res, index, selected_indices.view(), decoded_vectors.view(), label);
 * @endcode
 *
 * @param[in] res
 * @param[in] index
 * @param[in] in_cluster_indices
 *   The offsets of the selected indices within the cluster.
 * @param[out] out_vectors
 *   the destination buffer [n_take, index.dim()].
 *   The length `n_take` defines how many records to reconstruct,
 *   it must be smaller than the list size.
 * @param[in] label
 *   The id of the list (cluster) to decode.
 */
void reconstruct_list_data(raft::resources const& res,
                           const index<int64_t>& index,
                           raft::device_vector_view<const uint32_t> in_cluster_indices,
                           raft::device_matrix_view<float, uint32_t, raft::row_major> out_vectors,
                           uint32_t label);
void reconstruct_list_data(raft::resources const& res,
                           const index<int64_t>& index,
                           raft::device_vector_view<const uint32_t> in_cluster_indices,
                           raft::device_matrix_view<half, uint32_t, raft::row_major> out_vectors,
                           uint32_t label);
void reconstruct_list_data(raft::resources const& res,
                           const index<int64_t>& index,
                           raft::device_vector_view<const uint32_t> in_cluster_indices,
                           raft::device_matrix_view<int8_t, uint32_t, raft::row_major> out_vectors,
                           uint32_t label);
void reconstruct_list_data(raft::resources const& res,
                           const index<int64_t>& index,
                           raft::device_vector_view<const uint32_t> in_cluster_indices,
                           raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_vectors,
                           uint32_t label);

/**
 * @brief Extend one list of the index in-place, by the list label, skipping the classification and
 * encoding steps.
 *
 * Usage example:
 * @code{.cpp}
 *   // We will extend the fourth cluster
 *   uint32_t label = 3;
 *   // We will fill 4 new vectors
 *   uint32_t n_vec = 4;
 *   // Indices of the new vectors
 *   auto indices = raft::make_device_vector<uint32_t>(res, n_vec);
 *   ... fill the indices ...
 *   auto new_codes = raft::make_device_matrix<uint8_t, uint32_t, row_major> new_codes(
 *       res, n_vec, index.pq_dim());
 *   ... fill codes ...
 *   // extend list with new codes
 *   ivf_pq::helpers::codepacker::extend_list_with_codes(
 *       res, &index, codes.view(), indices.view(), label);
 * @endcode
 *
 * @param[in] res
 * @param[inout] index
 * @param[in] new_codes flat PQ codes, one code per byte [n_rows, index.pq_dim()]
 * @param[in] new_indices source indices [n_rows]
 * @param[in] label the id of the target list (cluster).
 */
void extend_list_with_codes(
  raft::resources const& res,
  index<int64_t>* index,
  raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> new_codes,
  raft::device_vector_view<const int64_t, uint32_t, raft::row_major> new_indices,
  uint32_t label);

/**
 * @brief Extend one list of the index in-place, by the list label, skipping the classification and
 * encoding steps. Uses contiguous/packed codes format.
 *
 * This is similar to extend_list_with_codes but takes codes in contiguous packed format
 * [n_rows, ceildiv(pq_dim * pq_bits, 8)] instead of unpacked format [n_rows, pq_dim].
 * This works correctly with any pq_bits value.
 *
 * Usage example:
 * @code{.cpp}
 *   // We will extend the fourth cluster
 *   uint32_t label = 3;
 *   // We will fill 4 new vectors
 *   uint32_t n_vec = 4;
 *   // Indices of the new vectors
 *   auto indices = raft::make_device_vector<int64_t>(res, n_vec);
 *   ... fill the indices ...
 *   // Allocate buffer for packed codes
 *   uint32_t code_size = raft::ceildiv(index.pq_dim() * index.pq_bits(), 8u);
 *   auto new_codes = raft::make_device_matrix<uint8_t, uint32_t, row_major>(res, n_vec, code_size);
 *   ... fill codes ...
 *   // extend list with new codes
 *   ivf_pq::helpers::codepacker::extend_list_with_contiguous_codes(
 *       res, &index, new_codes.view(), indices.view(), label);
 * @endcode
 *
 * @param[in] res
 * @param[inout] index
 * @param[in] new_codes flat contiguous PQ codes [n_rows, ceildiv(pq_dim * pq_bits, 8)]
 * @param[in] new_indices source indices [n_rows]
 * @param[in] label the id of the target list (cluster).
 */
void extend_list_with_contiguous_codes(
  raft::resources const& res,
  index<int64_t>* index,
  raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> new_codes,
  raft::device_vector_view<const int64_t, uint32_t, raft::row_major> new_indices,
  uint32_t label);

/**
 * @brief Extend one list of the index in-place, by the list label, skipping the classification
 * step.
 *
 *  Usage example:
 * @code{.cpp}
 *   // We will extend the fourth cluster
 *   uint32_t label = 3;
 *   // We will extend with 4 new vectors
 *   uint32_t n_vec = 4;
 *   // Indices of the new vectors
 *   auto indices = raft::make_device_vector<uint32_t>(res, n_vec);
 *   ... fill the indices ...
 *   auto new_vectors = raft::make_device_matrix<float, uint32_t, row_major> new_codes(
 *       res, n_vec, index.dim());
 *   ... fill vectors ...
 *   // extend list with new vectors
 *   ivf_pq::helpers::codepacker::extend_list(
 *       res, &index, new_vectors.view(), indices.view(), label);
 * @endcode
 *
 *
 * @param[in] res
 * @param[inout] index
 * @param[in] new_vectors data to encode [n_rows, index.dim()]
 * @param[in] new_indices source indices [n_rows]
 * @param[in] label the id of the target list (cluster).
 */
void extend_list(raft::resources const& res,
                 index<int64_t>* index,
                 raft::device_matrix_view<const float, uint32_t, raft::row_major> new_vectors,
                 raft::device_vector_view<const int64_t, uint32_t, raft::row_major> new_indices,
                 uint32_t label);
void extend_list(raft::resources const& res,
                 index<int64_t>* index,
                 raft::device_matrix_view<const int8_t, uint32_t, raft::row_major> new_vectors,
                 raft::device_vector_view<const int64_t, uint32_t, raft::row_major> new_indices,
                 uint32_t label);
void extend_list(raft::resources const& res,
                 index<int64_t>* index,
                 raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> new_vectors,
                 raft::device_vector_view<const int64_t, uint32_t, raft::row_major> new_indices,
                 uint32_t label);

/**
 * @}
 */
};  // namespace codepacker

/**
 * @brief Remove all data from a single list (cluster) in the index.
 *
 * Usage example:
 * @code{.cpp}
 *   // We will erase the fourth cluster (label = 3)
 *   ivf_pq::helpers::erase_list(res, &index, 3);
 * @endcode
 *
 *
 * @param[in] res
 * @param[inout] index
 * @param[in] label the id of the target list (cluster).
 */
void erase_list(raft::resources const& res, index<int64_t>* index, uint32_t label);

/**
 * @brief Public helper API to reset the data and indices ptrs, and the list sizes. Useful for
 * externally modifying the index without going through the build stage. The data and indices of the
 * IVF lists will be lost.
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // initialize an empty index
 *   ivf_pq::index<int64_t> index(res, index_params, D);
 *   // reset the index's state and list sizes
 *   ivf_pq::helpers::reset_index(res, &index);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-PQ index
 */
void reset_index(const raft::resources& res, index<int64_t>* index);

/**
 * @brief Pad cluster centers with their L2 norms for efficient GEMM operations.
 *
 * This function takes cluster centers and pads them with their L2 norms to create
 * extended centers suitable for coarse search operations. The output has dimensions
 * [n_centers, dim_ext] where dim_ext = round_up(dim + 1, 8).
 *
 * @param[in] res raft resource
 * @param[in] centers cluster centers [n_centers, dim]
 * @param[out] padded_centers padded centers with norms [n_centers, dim_ext]
 */
void pad_centers_with_norms(
  raft::resources const& res,
  raft::device_matrix_view<const float, uint32_t, raft::row_major> centers,
  raft::device_matrix_view<float, uint32_t, raft::row_major> padded_centers);

/**
 * @brief Pad cluster centers with their L2 norms for efficient GEMM operations.
 *
 * This function takes cluster centers and pads them with their L2 norms to create
 * extended centers suitable for coarse search operations. The output has dimensions
 * [n_centers, dim_ext] where dim_ext = round_up(dim + 1, 8).
 *
 * @param[in] res raft resource
 * @param[in] centers cluster centers [n_centers, dim]
 * @param[out] padded_centers padded centers with norms [n_centers, dim_ext]
 */
void pad_centers_with_norms(
  raft::resources const& res,
  raft::host_matrix_view<const float, uint32_t, raft::row_major> centers,
  raft::device_matrix_view<float, uint32_t, raft::row_major> padded_centers);

/**
 * @brief Rotate padded centers with the rotation matrix.
 *
 * @param[in] res raft resource
 * @param[in] padded_centers padded centers [n_centers, dim_ext]
 * @param[in] rotation_matrix rotation matrix [rot_dim, dim]
 * @param[out] rotated_centers rotated centers [n_centers, rot_dim]
 */
void rotate_padded_centers(
  raft::resources const& res,
  raft::device_matrix_view<const float, uint32_t, raft::row_major> padded_centers,
  raft::device_matrix_view<const float, uint32_t, raft::row_major> rotation_matrix,
  raft::device_matrix_view<float, uint32_t, raft::row_major> rotated_centers);

/**
 * @brief Public helper API for fetching a trained index's IVF centroids
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   // allocate the buffer for the output centers
 *   auto cluster_centers = raft::make_device_matrix<float, uint32_t>(
 *     res, index.n_lists(), index.dim());
 *   // Extract the IVF centroids into the buffer
 *   cuvs::neighbors::ivf_pq::helpers::extract_centers(res, index, cluster_centers.data_handle());
 * @endcode
 *
 * @param[in] res raft resource
 * @param[in] index IVF-PQ index (passed by reference)
 * @param[out] cluster_centers IVF cluster centers [index.n_lists(), index.dim]
 */
void extract_centers(raft::resources const& res,
                     const index<int64_t>& index,
                     raft::device_matrix_view<float, int64_t, raft::row_major> cluster_centers);

/** @copydoc extract_centers */
void extract_centers(raft::resources const& res,
                     const index<int64_t>& index,
                     raft::host_matrix_view<float, uint32_t, raft::row_major> cluster_centers);
/**
 * @brief Helper exposing the re-computation of list sizes and related arrays if IVF lists have been
 * modified externally.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   raft::resources res;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // initialize an empty index
 *   ivf_pq::index<int64_t> index(res, index_params, D);
 *   ivf_pq::helpers::reset_index(res, &index);
 *   // resize the first IVF list to hold 5 records
 *   auto spec = ivf_pq::list_spec_interleaved<uint32_t, int64_t>{
 *     index.pq_bits(), index.pq_dim(), index.conservative_memory_allocation()};
 *   uint32_t new_size = 5;
 *   ivf_pq::helpers::resize_list(res, index.lists()[0], spec, new_size, 0);
 *   raft::update_device(index.list_sizes().data_handle(), &new_size, 1, stream);
 *   // recompute the internal state of the index
 *   ivf_pq::helpers::recompute_internal_state(res, &index);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-PQ index
 */
void recompute_internal_state(const raft::resources& res, index<int64_t>* index);

/**
 * @brief Generate a rotation matrix into user-provided buffer (standalone version).
 *
 * This standalone helper generates a rotation matrix without requiring an index object.
 * Users can call this to prepare a rotation matrix before building from precomputed data.
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   uint32_t dim = 128, pq_dim = 32;
 *   uint32_t rot_dim = pq_dim * ((dim + pq_dim - 1) / pq_dim);  // rounded up
 *
 *   // Allocate rotation matrix buffer [rot_dim, dim]
 *   auto rotation_matrix = raft::make_device_matrix<float, uint32_t>(res, rot_dim, dim);
 *
 *   // Generate the rotation matrix
 *   ivf_pq::helpers::make_rotation_matrix(
 *     res, rotation_matrix.view(), true);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[out] rotation_matrix Output buffer [rot_dim, dim] for the rotation matrix
 * @param[in] force_random_rotation If false and rot_dim == dim, creates identity matrix.
 *                                   If true or rot_dim != dim, creates random orthogonal matrix.
 */
void make_rotation_matrix(
  raft::resources const& res,
  raft::device_matrix_view<float, uint32_t, raft::row_major> rotation_matrix,
  bool force_random_rotation);

/** Calculate the size of the compressed dataset.
 *
 * @param[in] res raft resource
 * @param[in] dataset shape of the dataset
 * @param[in] param ivf-pq compression params
 *
 * @return compressed dataset size in bytes
 */
size_t compressed_dataset_size(raft::resources const& res,
                               raft::matrix_extent<int64_t> dataset,
                               cuvs::neighbors::ivf_pq::index_params params);

/**
 * @brief Resize an IVF-PQ list with flat layout.
 *
 * This helper resizes an IVF list that uses the flat (non-interleaved) PQ code layout.
 * If the new size exceeds the current capacity, a new list is allocated and existing
 * data is copied. The function handles the type casting internally.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   raft::resources res;
 *   // Assuming index uses FLAT layout
 *   auto spec = ivf_pq::list_spec_flat<uint32_t, int64_t>{
 *     index.pq_bits(), index.pq_dim(), index.conservative_memory_allocation()};
 *   uint32_t old_size = current_list_size;
 *   uint32_t new_size = old_size + n_new_vectors;
 *   ivf_pq::helpers::resize_list(res, index.lists()[label], spec, new_size, old_size);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] orig_list the list to resize (may be replaced with a new allocation)
 * @param[in] spec the list specification containing pq_bits, pq_dim, and allocation settings
 * @param[in] new_used_size the new size of the list (number of vectors)
 * @param[in] old_used_size the current size of the list (data up to this size is preserved)
 */
void resize_list(raft::resources const& res,
                 std::shared_ptr<list_data_base<int64_t, uint32_t>>& orig_list,
                 const list_spec_flat<uint32_t, int64_t>& spec,
                 uint32_t new_used_size,
                 uint32_t old_used_size);

/**
 * @brief Resize an IVF-PQ list with interleaved layout.
 *
 * This helper resizes an IVF list that uses the interleaved PQ code layout (default).
 * If the new size exceeds the current capacity, a new list is allocated and existing
 * data is copied. The function handles the type casting internally.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   raft::resources res;
 *   // Assuming index uses INTERLEAVED layout (default)
 *   auto spec = ivf_pq::list_spec_interleaved<uint32_t, int64_t>{
 *     index.pq_bits(), index.pq_dim(), index.conservative_memory_allocation()};
 *   uint32_t old_size = current_list_size;
 *   uint32_t new_size = old_size + n_new_vectors;
 *   ivf_pq::helpers::resize_list(res, index.lists()[label], spec, new_size, old_size);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] orig_list the list to resize (may be replaced with a new allocation)
 * @param[in] spec the list specification containing pq_bits, pq_dim, and allocation settings
 * @param[in] new_used_size the new size of the list (number of vectors)
 * @param[in] old_used_size the current size of the list (data up to this size is preserved)
 */
void resize_list(raft::resources const& res,
                 std::shared_ptr<list_data_base<int64_t, uint32_t>>& orig_list,
                 const list_spec_interleaved<uint32_t, int64_t>& spec,
                 uint32_t new_used_size,
                 uint32_t old_used_size);

/**
 * @}
 */
}  // namespace helpers

}  // namespace ivf_pq
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs

namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace graph_build_params {
/** Specialized parameters utilizing IVF-PQ to build knn graph */
struct ivf_pq_params {
  cuvs::neighbors::ivf_pq::index_params build_params;
  cuvs::neighbors::ivf_pq::search_params search_params;
  float refinement_rate = 1.0;

  ivf_pq_params() = default;

  /**
   * Set default parameters based on shape of the input dataset.
   * Usage example:
   * @code{.cpp}
   *   using namespace cuvs::neighbors;
   *   raft::resources res;
   *   // create index_params for a [N. D] dataset
   *   auto dataset = raft::make_device_matrix<float, int64_t>(res, N, D);
   *   auto pq_params =
   *     graph_build_params::ivf_pq_params(dataset.extents());
   *   // modify/update index_params as needed
   *   pq_params.kmeans_trainset_fraction = 0.1;
   * @endcode
   */
  ivf_pq_params(raft::matrix_extent<int64_t> dataset_extents,
                cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded)
  {
    build_params    = ivf_pq::index_params::from_dataset(dataset_extents, metric);
    auto n_rows     = dataset_extents.extent(0);
    auto n_features = dataset_extents.extent(1);
    if (n_features <= 32) {
      build_params.pq_dim  = 16;
      build_params.pq_bits = 8;
    } else {
      build_params.pq_bits = 4;
      if (n_features <= 64) {
        build_params.pq_dim = 32;
      } else if (n_features <= 128) {
        build_params.pq_dim = 64;
      } else if (n_features <= 192) {
        build_params.pq_dim = 96;
      } else {
        build_params.pq_dim = raft::round_up_safe<uint32_t>(n_features / 2, 128);
      }
    }

    build_params.n_lists        = std::max<uint32_t>(1, n_rows / 2000);
    build_params.kmeans_n_iters = 10;

    const double kMinPointsPerCluster         = 32;
    const double min_kmeans_trainset_points   = kMinPointsPerCluster * build_params.n_lists;
    const double max_kmeans_trainset_fraction = 1.0;
    const double min_kmeans_trainset_fraction =
      std::min(max_kmeans_trainset_fraction, min_kmeans_trainset_points / n_rows);
    build_params.kmeans_trainset_fraction = std::clamp(
      1.0 / std::sqrt(n_rows * 1e-5), min_kmeans_trainset_fraction, max_kmeans_trainset_fraction);
    build_params.codebook_kind = ivf_pq::codebook_gen::PER_SUBSPACE;

    search_params                         = cuvs::neighbors::ivf_pq::search_params{};
    search_params.n_probes                = std::round(std::sqrt(build_params.n_lists) / 20 + 4);
    search_params.lut_dtype               = CUDA_R_16F;
    search_params.internal_distance_dtype = CUDA_R_16F;
    search_params.coarse_search_dtype     = CUDA_R_16F;
    search_params.max_internal_batch_size = 128 * 1024;

    refinement_rate = 1;
  }
};
}  // namespace graph_build_params
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
