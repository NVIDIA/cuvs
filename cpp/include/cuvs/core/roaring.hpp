/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/bitset.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace cuvs::core {

/**
 * @defgroup roaring GPU Roaring Bitmap
 *
 * GPU-native Roaring Bitmap for compressed prefiltering in vector search.
 * Stores attribute bitmaps in compressed Roaring format in GPU memory,
 * supports bulk set operations (AND/OR/ANDNOT/XOR), and decompresses
 * to flat bitsets compatible with cuvs::core::bitset for search.
 *
 * Typical usage:
 * @code{.cpp}
 *   // Build filter bitmaps on CPU, upload to GPU
 *   auto category = gpu_roaring::from_sorted_ids(res, cat_ids, universe);
 *   auto price    = gpu_roaring::from_sorted_ids(res, price_ids, universe);
 *
 *   // Combine on GPU
 *   auto combined = gpu_roaring::set_and(res, category, price);
 *
 *   // Decompress to bitset for search
 *   auto bitset = gpu_roaring::to_bitset(res, combined);
 *   auto filter = cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(bitset.view());
 *   cuvs::neighbors::cagra::search(res, params, index, queries, neighbors, distances, filter);
 * @endcode
 * @{
 */

/** Container type tags for Roaring containers */
enum class roaring_container_type : uint8_t { ARRAY = 0, BITMAP = 1, RUN = 2 };

/**
 * @brief Non-owning device view of a @ref gpu_roaring, usable inside
 * kernels for per-sample membership tests (mirrors
 * raft::core::bitset_view::test()).
 */
struct roaring_view {
  const uint16_t* keys                = nullptr;
  const roaring_container_type* types = nullptr;
  const uint32_t* offsets             = nullptr;
  const uint16_t* cardinalities       = nullptr;
  const uint64_t* bitmap_data         = nullptr;
  const uint16_t* array_data          = nullptr;
  const uint16_t* run_data            = nullptr;
  const uint16_t* key_index           = nullptr;  // optional O(1) key lookup
  uint32_t n_containers               = 0;
  uint32_t max_key                    = 0;
  uint32_t n_rows                     = 0;

#if defined(__CUDACC__)
  /** Whether sample index `id` is a member of the filter. */
  __device__ __forceinline__ bool test(uint32_t id) const
  {
    uint16_t key = static_cast<uint16_t>(id >> 16);
    uint16_t low = static_cast<uint16_t>(id & 0xFFFFu);

    int cid = -1;
    if (key_index != nullptr) {
      if (key > max_key) return false;
      uint16_t slot = key_index[key];
      if (slot == 0xFFFFu) return false;
      cid = slot;
    } else {
      uint32_t lo = 0, hi = n_containers;
      while (lo < hi) {  // lower_bound on sorted keys
        uint32_t mid = (lo + hi) / 2;
        if (keys[mid] < key) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      if (lo == n_containers || keys[lo] != key) return false;
      cid = static_cast<int>(lo);
    }

    uint32_t offset = offsets[cid];
    switch (types[cid]) {
      case roaring_container_type::BITMAP: {
        uint64_t word = bitmap_data[offset / sizeof(uint64_t) + (low >> 6)];
        return (word >> (low & 63u)) & 1u;
      }
      case roaring_container_type::ARRAY: {
        const uint16_t* arr = array_data + offset / sizeof(uint16_t);
        uint32_t lo = 0, hi = cardinalities[cid];
        while (lo < hi) {
          uint32_t mid = (lo + hi) / 2;
          if (arr[mid] < low) {
            lo = mid + 1;
          } else {
            hi = mid;
          }
        }
        return lo < cardinalities[cid] && arr[lo] == low;
      }
      default: {  // RUN
        const uint16_t* runs = run_data + offset / sizeof(uint16_t);
        uint32_t n_runs      = cardinalities[cid];
        uint32_t lo = 0, hi = n_runs;
        while (lo < hi) {  // last run with start <= low
          uint32_t mid = (lo + hi) / 2;
          if (runs[mid * 2] <= low) {
            lo = mid + 1;
          } else {
            hi = mid;
          }
        }
        if (lo == 0) return false;
        uint32_t start = runs[(lo - 1) * 2];
        uint32_t len   = runs[(lo - 1) * 2 + 1];
        return low >= start && low <= start + len;
      }
    }
  }
#endif
};

/**
 * @brief GPU-resident Roaring bitmap in Structure-of-Arrays layout.
 *
 * All device memory is managed via rmm::device_uvector (RAII).
 * Supports move semantics; not copyable.
 */
struct gpu_roaring {
  // Top-level index (one entry per container, sorted by key)
  rmm::device_uvector<uint16_t> keys;
  rmm::device_uvector<roaring_container_type> types;
  rmm::device_uvector<uint32_t> offsets;
  rmm::device_uvector<uint16_t> cardinalities;
  uint32_t n_containers  = 0;
  uint32_t universe_size = 0;

  // Per-type data pools
  rmm::device_uvector<uint64_t> bitmap_data;
  uint32_t n_bitmap_containers = 0;

  rmm::device_uvector<uint16_t> array_data;
  uint32_t n_array_containers = 0;

  rmm::device_uvector<uint16_t> run_data;
  uint32_t n_run_containers = 0;

  // Direct-map key index: key_index[high16] = container index, or 0xFFFF.
  // Replaces O(log n) binary search with O(1) table lookup.
  rmm::device_uvector<uint16_t> key_index;
  uint32_t max_key = 0;

  // Complement optimization: when true, the stored set is the complement
  // of the logical set. contains() results are flipped at query time.
  // Note: never set by from_sorted_ids; filters require negated == false.
  bool negated = false;

  // Total logical cardinality (number of set bits, before complement)
  uint64_t total_cardinality = 0;

  // Host mirrors of the per-container metadata, captured at construction.
  // These make per-filter cardinalities and CSR indptr available with no
  // device->host transfer or count kernels (uint32 element counts: a full
  // 65536-element container wraps to 0 in the device-side uint16 array).
  std::vector<uint16_t> h_keys;
  std::vector<roaring_container_type> h_types;
  std::vector<uint32_t> h_offsets;
  std::vector<uint32_t> h_element_counts;

  /** Non-owning device view for use inside kernels. */
  [[nodiscard]] roaring_view view() const
  {
    roaring_view v;
    v.keys          = keys.data();
    v.types         = types.data();
    v.offsets       = offsets.data();
    v.cardinalities = cardinalities.data();
    v.bitmap_data   = bitmap_data.data();
    v.array_data    = array_data.data();
    v.run_data      = run_data.data();
    v.key_index     = key_index.size() > 0 ? key_index.data() : nullptr;
    v.n_containers  = n_containers;
    v.max_key       = max_key;
    v.n_rows        = universe_size;
    return v;
  }

  /** Construct an empty GPU Roaring bitmap */
  explicit gpu_roaring(rmm::cuda_stream_view stream)
    : keys(0, stream),
      types(0, stream),
      offsets(0, stream),
      cardinalities(0, stream),
      bitmap_data(0, stream),
      array_data(0, stream),
      run_data(0, stream),
      key_index(0, stream)
  {
  }

  gpu_roaring(gpu_roaring&&)            = default;
  gpu_roaring& operator=(gpu_roaring&&) = default;

  /** Total device memory used (bytes) */
  [[nodiscard]] size_t device_memory_bytes() const
  {
    return keys.size() * sizeof(uint16_t) + types.size() * sizeof(roaring_container_type) +
           offsets.size() * sizeof(uint32_t) + cardinalities.size() * sizeof(uint16_t) +
           bitmap_data.size() * sizeof(uint64_t) + array_data.size() * sizeof(uint16_t) +
           run_data.size() * sizeof(uint16_t) + key_index.size() * sizeof(uint16_t);
  }

  /** Equivalent flat bitset size (bytes) */
  [[nodiscard]] size_t flat_bitset_bytes() const
  {
    return (static_cast<size_t>(universe_size) + 31) / 32 * sizeof(uint32_t);
  }

  /** Compression ratio (flat / compressed) */
  [[nodiscard]] double compression_ratio() const
  {
    auto dev_bytes = device_memory_bytes();
    return dev_bytes > 0 ? static_cast<double>(flat_bitset_bytes()) / dev_bytes : 0.0;
  }
};

/** Set operation types */
enum class roaring_set_op : uint8_t { AND = 0, OR = 1, ANDNOT = 2, XOR = 3 };

/**
 * @brief Create a GPU Roaring bitmap from a sorted array of IDs on host.
 *
 * The IDs are partitioned into 65536-element containers following the
 * Roaring bitmap format. Containers with <= 4096 elements use array format;
 * denser containers use bitmap format.
 *
 * @param[in] res RAFT resources (provides CUDA stream)
 * @param[in] sorted_ids Host pointer to sorted, deduplicated uint32_t IDs
 * @param[in] n_ids Number of IDs
 * @param[in] universe_size Maximum representable ID + 1
 * @return GPU Roaring bitmap
 */
gpu_roaring from_sorted_ids(raft::resources const& res,
                            const uint32_t* sorted_ids,
                            uint32_t n_ids,
                            uint32_t universe_size);

/**
 * @brief Perform a pairwise set operation between two GPU Roaring bitmaps.
 *
 * @param[in] res RAFT resources
 * @param[in] a First operand
 * @param[in] b Second operand
 * @param[in] op Set operation (AND, OR, ANDNOT, XOR)
 * @return Result GPU Roaring bitmap
 */
gpu_roaring set_op(raft::resources const& res,
                   const gpu_roaring& a,
                   const gpu_roaring& b,
                   roaring_set_op op);

/** Convenience: result = a AND b */
inline gpu_roaring set_and(raft::resources const& res, const gpu_roaring& a, const gpu_roaring& b)
{
  return set_op(res, a, b, roaring_set_op::AND);
}

/** Convenience: result = a OR b */
inline gpu_roaring set_or(raft::resources const& res, const gpu_roaring& a, const gpu_roaring& b)
{
  return set_op(res, a, b, roaring_set_op::OR);
}

/**
 * @brief Multi-bitmap AND: result = a[0] AND a[1] AND ... AND a[n-1].
 */
gpu_roaring multi_and(raft::resources const& res, const gpu_roaring* bitmaps, uint32_t count);

/**
 * @brief Multi-bitmap OR: result = a[0] OR a[1] OR ... OR a[n-1].
 */
gpu_roaring multi_or(raft::resources const& res, const gpu_roaring* bitmaps, uint32_t count);

/**
 * @brief Decompress a GPU Roaring bitmap to a flat bitset.
 *
 * The output is compatible with cuvs::core::bitset<uint32_t, int64_t>
 * and can be wrapped in a bitset_filter for use with cuVS search.
 *
 * @param[in] res RAFT resources
 * @param[in] bitmap Source GPU Roaring bitmap
 * @return Flat bitset (device memory, RAII)
 */
cuvs::core::bitset<uint32_t, int64_t> to_bitset(raft::resources const& res,
                                                const gpu_roaring& bitmap);

/**
 * @brief Decompress into a pre-allocated flat bitset buffer.
 *
 * @param[in]  res RAFT resources
 * @param[in]  bitmap Source GPU Roaring bitmap
 * @param[out] output Device pointer to uint32_t array, pre-zeroed
 * @param[in]  output_size_words Number of uint32_t words in output
 */
void decompress_to_bitset(raft::resources const& res,
                          const gpu_roaring& bitmap,
                          uint32_t* output,
                          uint32_t output_size_words);

/**
 * @brief Create a GPU Roaring bitmap from sorted, deduplicated device IDs.
 *
 * Equivalent to the host-pointer overload; the IDs are copied to the host
 * for container construction (the per-container metadata must live on the
 * host anyway — it is what makes count-free filtered search possible).
 */
gpu_roaring from_sorted_ids(raft::resources const& res,
                            raft::device_vector_view<const uint32_t, int64_t> sorted_ids,
                            uint32_t universe_size);

/**
 * @brief Emit the sorted member IDs of one or more Roaring bitmaps into the
 * column-index array of a CSR structure, one bitmap per CSR row segment.
 *
 * Row `i`'s segment is `[indptr[i], indptr[i+1])` where the indptr is the
 * exclusive prefix sum of `bitmaps[i]->total_cardinality` — available on
 * the host with no device synchronization. One kernel launch for all
 * bitmaps (one CTA per Roaring container).
 *
 * @param[in]  res RAFT resources
 * @param[in]  bitmaps Host array of `n_bitmaps` bitmap pointers
 * @param[in]  n_bitmaps Number of bitmaps (CSR rows)
 * @param[out] indices Device array of size `sum(cardinalities)`
 */
void to_csr_indices(raft::resources const& res,
                    const gpu_roaring* const* bitmaps,
                    uint32_t n_bitmaps,
                    int64_t* indices);

/**
 * @brief Decompress `n_bitmaps` Roaring bitmaps into one dense bit matrix
 * of logical shape `[n_bitmaps, n_rows]` (row-contiguous bits), compatible
 * with cuvs::core::bitmap_view.
 *
 * @param[in]  res RAFT resources
 * @param[in]  bitmaps Host array of `n_bitmaps` bitmap pointers
 * @param[in]  n_bitmaps Number of bitmap rows
 * @param[in]  n_rows Number of columns (bits per row)
 * @param[out] output Device array of `ceil(n_bitmaps * n_rows / 32)` words,
 *             pre-zeroed by the caller
 */
void decompress_to_bitmap(raft::resources const& res,
                          const gpu_roaring* const* bitmaps,
                          uint32_t n_bitmaps,
                          int64_t n_rows,
                          uint32_t* output);

/** @} */  // end group roaring

}  // namespace cuvs::core
