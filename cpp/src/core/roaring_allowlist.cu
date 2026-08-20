/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/roaring_allowlist.hpp>

#include "nvtx.hpp"

#include <cuco/roaring_bitmap_ref.cuh>

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/device/device_select.cuh>

#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuda/std/cstddef>
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace cuvs::core {
namespace {

/**
 * Portable Roaring serialization used by each allowlist
 * =====================================================
 *
 * This file writes and validates the standard 32-bit portable Roaring format.
 * The authoritative format description is:
 *
 *   https://github.com/RoaringBitmap/RoaringFormatSpec
 *
 * The resulting bytes are consumed on the device by cuCollections:
 *
 *   https://github.com/NVIDIA/cuCollections/blob/6001618aaa7f17ea2bbcd444650e9573c4f3d6c5/include/cuco/roaring_bitmap_ref.cuh
 *   https://github.com/NVIDIA/cuCollections/blob/6001618aaa7f17ea2bbcd444650e9573c4f3d6c5/include/cuco/detail/roaring_bitmap/util.cuh
 *
 * All integers below are little-endian. A 32-bit ID is divided into a container
 * key and a value:
 *
 *   ID = (uint32_t(key) << 16) | value
 *          high 16 bits             low 16 bits
 *
 * IDs with the same key belong to one container. Container keys and array
 * values are strictly increasing. The portable stream for one nonempty
 * allowlist has one of these two headers:
 *
 * With no run containers, the row is laid out as:
 *
 * @code{.unparsed}
 * uint32 cookie = 12346
 * uint32 N
 * descriptor[N]
 * uint32 container_offset[N]
 * container payloads
 * @endcode
 *
 * With at least one run container, the row is laid out as:
 *
 * @code{.unparsed}
 * uint32 cookie = 12347 | ((N - 1) << 16)
 * uint8 run_container_bitmap[ceil(N / 8)]
 * descriptor[N]
 * uint32 container_offset[N]  // present only when N >= 4
 * container payloads
 * @endcode
 *
 * Each four-byte descriptor is:
 *
 *   uint16 key
 *   uint16 cardinality_minus_one
 *
 * An offset is measured from the first byte of this stream. Without run
 * containers, cardinality selects the payload representation: at most 4096
 * values use an array; more than 4096 use a bitmap. The run-container bitmap
 * overrides that choice for marked containers. Its bit order is
 * least-significant bit first.
 *
 * @code{.unparsed}
 * array:
 *   uint16 value[cardinality]
 *
 * bitmap:
 *   uint64 words[1024]  // 8192 bytes; v is bit (v % 64) of word (v / 64)
 *
 * run:
 *   uint16 number_of_runs
 *   { uint16 start; uint16 length_minus_one; } runs[number_of_runs]
 * @endcode
 *
 * The portable format above describes exactly one bitmap. Query-to-allowlist
 * association is a separate concern: `filtering::roaring_filter` stores device
 * pointers to already initialized allowlist references. Consequently neither
 * the serialized payload nor this metadata is copied or parsed by CAGRA search.
 */

// Names and thresholds used by the portable format specification.
constexpr std::uint32_t kCookieNoRun    = 12346;
constexpr std::uint32_t kCookieRun      = 12347;
constexpr std::size_t kArrayCardinality = 4096;
constexpr std::size_t kBitmapBytes      = 8192;
constexpr std::size_t kOffsetThreshold  = 4;

using ref_type = cuco::experimental::roaring_bitmap_ref<std::uint32_t>;

struct row_metadata {
  std::size_t cardinality{};
  std::uint32_t max_id{};
  bool empty{true};
};

std::uint16_t read_u16(std::byte const* data, std::size_t size, std::size_t offset)
{
  RAFT_EXPECTS(offset <= size && size - offset >= 2,
               "Malformed portable Roaring bitmap: truncated uint16 value.");
  return static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(data[offset])) |
         static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(data[offset + 1])) << 8;
}

std::uint32_t read_u32(std::byte const* data, std::size_t size, std::size_t offset)
{
  RAFT_EXPECTS(offset <= size && size - offset >= 4,
               "Malformed portable Roaring bitmap: truncated uint32 value.");
  std::uint32_t value{};
  for (int i = 0; i < 4; ++i) {
    value |= static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(data[offset + i])) << (8 * i);
  }
  return value;
}

void validate_dataset_rows(std::size_t dataset_rows)
{
  constexpr std::uint64_t kKeyDomain = std::uint64_t{1} << 32;
  RAFT_EXPECTS(dataset_rows > 0, "dataset_rows must be greater than zero.");
  RAFT_EXPECTS(static_cast<std::uint64_t>(dataset_rows) <= kKeyDomain,
               "dataset_rows exceeds the uint32_t Roaring key domain.");
}

enum class container_kind : std::uint8_t { array, bitmap, run };

std::size_t align_up(std::size_t offset, std::size_t alignment)
{
  return (offset + alignment - 1) / alignment * alignment;
}

struct device_build_summary {
  std::int64_t cardinality{};
  std::uint64_t payload_bytes{};
  std::uint32_t num_containers{};
  std::uint32_t has_run{};
  std::uint32_t invalid{};
};

struct device_build_result {
  rmm::device_uvector<cuda::std::byte> storage;
  std::size_t serialized_bytes{};
  std::size_t cardinality{};
  bool reference_initialized{};
};

std::size_t reference_offset(std::size_t serialized_bytes)
{
  return align_up(serialized_bytes, alignof(ref_type));
}

std::size_t owned_storage_bytes(std::size_t serialized_bytes)
{
  return serialized_bytes == 0 ? 0 : reference_offset(serialized_bytes) + sizeof(ref_type);
}

constexpr int kBuilderBlockSize = 256;

// Small allowlists do not benefit from the general builder's device-wide sort,
// two scans, and separate per-stage allocations. At this cardinality every
// portable container is necessarily an array or a run (a bitmap requires more
// than 4096 values in one high-16-bit partition), so one CTA can sort, analyze,
// and later encode the complete row. For a single pre-sorted row the cutoff is lower because the
// general path already avoids its most expensive stage, the device-wide sort. Batched rows use the
// 128-ID capacity because the launch is amortized across the matrix; keep both rules tied to the
// construction benchmark.
constexpr int kSparseBuilderBlockSize               = 128;
constexpr std::size_t kSparseBuilderMaxIds          = 128;
constexpr std::size_t kSparseBuilderMaxPreSortedIds = 64;
constexpr int kSparseItemsPerThread =
  static_cast<int>(kSparseBuilderMaxIds) / kSparseBuilderBlockSize;
static_assert(kSparseBuilderMaxIds % kSparseBuilderBlockSize == 0);

struct sparse_container_metadata {
  std::uint32_t begin{};
  std::uint32_t payload_offset{};
  std::uint16_t runs{};
  container_kind kind{};
  std::uint8_t padding{};
};

static_assert(sizeof(sparse_container_metadata) == 12);

struct sparse_scratch_layout {
  explicit sparse_scratch_layout(std::size_t ids, std::size_t containers, bool store_sorted_ids)
  {
    sorted_ids_offset = align_up(sizeof(device_build_summary), alignof(std::uint32_t));
    auto offset       = sorted_ids_offset + (store_sorted_ids ? ids * sizeof(std::uint32_t) : 0);
    metadata_offset   = align_up(offset, alignof(sparse_container_metadata));
    bytes             = metadata_offset + containers * sizeof(sparse_container_metadata);
  }

  std::size_t sorted_ids_offset{};
  std::size_t metadata_offset{};
  std::size_t bytes{};
};

/**
 * One allocation for all general-builder temporaries.
 *
 * The allocation is cardinality/container scaled. The largest CUB workspace is reused by sort,
 * boundary selection, and payload scan because those stages are stream ordered.
 */
struct general_scratch_layout {
  general_scratch_layout(std::size_t ids,
                         std::size_t containers,
                         bool store_sorted_ids,
                         std::size_t workspace_bytes)
  {
    std::size_t cursor{};
    auto reserve = [&](std::size_t count, std::size_t item_size, std::size_t alignment) {
      auto const result = align_up(cursor, alignment);
      cursor            = result + count * item_size;
      return result;
    };

    if (store_sorted_ids) {
      sorted_ids_offset = reserve(ids, sizeof(std::uint32_t), alignof(std::uint32_t));
    }
    id_count_offset         = reserve(1, sizeof(std::int64_t), alignof(std::int64_t));
    valid_count_offset      = reserve(1, sizeof(std::int64_t), alignof(std::int64_t));
    selected_count_offset   = reserve(1, sizeof(std::int64_t), alignof(std::int64_t));
    container_starts_offset = reserve(containers, sizeof(std::int64_t), alignof(std::int64_t));
    num_containers_offset   = reserve(1, sizeof(std::uint32_t), alignof(std::uint32_t));
    kinds_offset            = reserve(containers, sizeof(container_kind), alignof(container_kind));
    payload_sizes_offset    = reserve(containers, sizeof(std::uint64_t), alignof(std::uint64_t));
    payload_offsets_offset  = reserve(containers, sizeof(std::uint64_t), alignof(std::uint64_t));
    has_run_offset          = reserve(1, sizeof(std::uint32_t), alignof(std::uint32_t));
    summary_offset   = reserve(1, sizeof(device_build_summary), alignof(device_build_summary));
    workspace_offset = reserve(workspace_bytes, sizeof(cuda::std::byte), alignof(std::max_align_t));
    bytes            = cursor;
  }

  std::size_t sorted_ids_offset{};
  std::size_t id_count_offset{};
  std::size_t valid_count_offset{};
  std::size_t selected_count_offset{};
  std::size_t container_starts_offset{};
  std::size_t num_containers_offset{};
  std::size_t kinds_offset{};
  std::size_t payload_sizes_offset{};
  std::size_t payload_offsets_offset{};
  std::size_t has_run_offset{};
  std::size_t summary_offset{};
  std::size_t workspace_offset{};
  std::size_t bytes{};
};

int grid_size_for(std::size_t count)
{
  auto const blocks = (count + kBuilderBlockSize - 1) / kBuilderBlockSize;
  return static_cast<int>(std::min<std::size_t>(blocks, 65535));
}

__device__ void write_u16(cuda::std::byte* output, std::size_t offset, std::uint16_t value)
{
  auto* bytes       = reinterpret_cast<std::uint8_t*>(output);
  bytes[offset]     = static_cast<std::uint8_t>(value);
  bytes[offset + 1] = static_cast<std::uint8_t>(value >> 8);
}

__device__ void write_u32(cuda::std::byte* output, std::size_t offset, std::uint32_t value)
{
  auto* bytes = reinterpret_cast<std::uint8_t*>(output);
  for (int byte = 0; byte < 4; ++byte) {
    bytes[offset + byte] = static_cast<std::uint8_t>(value >> (8 * byte));
  }
}

__device__ void write_u64(cuda::std::byte* output, std::size_t offset, std::uint64_t value)
{
  auto* bytes = reinterpret_cast<std::uint8_t*>(output);
  for (int byte = 0; byte < 8; ++byte) {
    bytes[offset + byte] = static_cast<std::uint8_t>(value >> (8 * byte));
  }
}

/** Find the sorted prefix that lies inside the logical dataset shape. */
__global__ void find_valid_count_kernel(std::uint32_t const* ids,
                                        std::int64_t const* id_count,
                                        std::uint64_t dataset_rows,
                                        std::int64_t* valid_count)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) { return; }
  std::int64_t first{};
  auto last = *id_count;
  while (first < last) {
    auto const middle = first + (last - first) / 2;
    if (static_cast<std::uint64_t>(ids[middle]) < dataset_rows) {
      first = middle + 1;
    } else {
      last = middle;
    }
  }
  *valid_count = first;
}

/** Select the first sorted ID belonging to every high-16-bit container. */
struct is_container_start {
  std::uint32_t const* ids{};
  std::int64_t const* valid_count{};

  __device__ bool operator()(std::int64_t i) const
  {
    auto const count = *valid_count;
    return i < count && (i == 0 || (ids[i - 1] >> 16) != (ids[i] >> 16));
  }
};

__global__ void narrow_container_count_kernel(std::int64_t const* selected_count,
                                              std::uint32_t* num_containers)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *num_containers = static_cast<std::uint32_t>(*selected_count);
  }
}

/**
 * Count consecutive runs and select the smallest legal portable payload for
 * each container.
 *
 * Each block owns one container. Threads independently identify run starts in
 * the sorted slice, then a block reduction produces the exact run count. This
 * uses O(number of input IDs) scratch for sorting and scans; it never
 * constructs a dense dataset-sized bitmap.
 */
__global__ void analyze_containers_kernel(std::uint32_t const* ids,
                                          std::int64_t const* id_count,
                                          std::int64_t const* container_starts,
                                          std::uint32_t const* num_containers,
                                          container_kind* kinds,
                                          std::uint64_t* payload_sizes,
                                          std::uint32_t* has_run)
{
  auto const container = static_cast<std::uint32_t>(blockIdx.x);
  auto const count     = *num_containers;
  if (container >= count) { return; }

  auto const begin = container_starts[container];
  auto const end   = container + 1 < count ? container_starts[container + 1] : *id_count;
  std::uint32_t local_runs{};
  for (auto i = begin + threadIdx.x; i < end; i += blockDim.x) {
    local_runs +=
      i == begin || static_cast<std::uint64_t>(ids[i]) != static_cast<std::uint64_t>(ids[i - 1]) + 1
        ? 1u
        : 0u;
  }

  using block_reduce = cub::BlockReduce<std::uint32_t, kBuilderBlockSize>;
  __shared__ typename block_reduce::TempStorage reduction_storage;
  auto const runs = block_reduce(reduction_storage).Sum(local_runs);
  if (threadIdx.x != 0) { return; }

  auto const cardinality = static_cast<std::uint64_t>(end - begin);
  auto const normal_size =
    cardinality <= kArrayCardinality ? cardinality * sizeof(std::uint16_t) : kBitmapBytes;
  auto const run_size = sizeof(std::uint16_t) + runs * 2 * sizeof(std::uint16_t);
  if (run_size < normal_size) {
    kinds[container]         = container_kind::run;
    payload_sizes[container] = run_size;
    atomicExch(has_run, 1u);
  } else if (cardinality <= kArrayCardinality) {
    kinds[container]         = container_kind::array;
    payload_sizes[container] = normal_size;
  } else {
    kinds[container]         = container_kind::bitmap;
    payload_sizes[container] = normal_size;
  }
}

/** Collect the scalar results needed to allocate the exact final portable byte
 * stream. */
__global__ void finish_device_analysis_kernel(std::int64_t const* id_count,
                                              std::int64_t const* valid_count,
                                              std::uint32_t const* num_containers,
                                              std::uint32_t const* has_run,
                                              std::uint64_t const* payload_sizes,
                                              std::uint64_t const* payload_offsets,
                                              device_build_summary* summary)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) { return; }
  auto const cardinality  = *id_count;
  auto const containers   = *num_containers;
  summary->cardinality    = cardinality;
  summary->num_containers = containers;
  summary->has_run        = *has_run;
  summary->payload_bytes =
    containers == 0 ? 0 : payload_offsets[containers - 1] + payload_sizes[containers - 1];
  summary->invalid = cardinality != *valid_count;
}

__host__ __device__ std::size_t portable_header_size(std::uint32_t num_containers, bool has_run)
{
  if (!has_run) {
    return 2 * sizeof(std::uint32_t) +
           num_containers * (2 * sizeof(std::uint16_t) + sizeof(std::uint32_t));
  }
  auto const run_bitmap_bytes = (num_containers + 7) / 8;
  return sizeof(std::uint32_t) + run_bitmap_bytes + num_containers * 2 * sizeof(std::uint16_t) +
         (num_containers >= kOffsetThreshold ? num_containers * sizeof(std::uint32_t) : 0);
}

/** Write the cookie, run bitmap, descriptors, and portable container-offset
 * table. */
__global__ void encode_header_kernel(std::uint32_t const* ids,
                                     std::int64_t const* id_count,
                                     std::int64_t const* container_starts,
                                     std::uint32_t num_containers,
                                     container_kind const* kinds,
                                     std::uint64_t const* payload_offsets,
                                     std::size_t header_size,
                                     bool has_run,
                                     cuda::std::byte* output)
{
  auto const thread           = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto const stride           = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  auto const run_bitmap_bytes = has_run ? (num_containers + 7) / 8 : 0;
  auto const descriptor_offset =
    has_run ? sizeof(std::uint32_t) + run_bitmap_bytes : 2 * sizeof(std::uint32_t);
  auto const offsets_offset = descriptor_offset + num_containers * 2 * sizeof(std::uint16_t);
  bool const store_offsets  = !has_run || num_containers >= kOffsetThreshold;

  if (thread == 0) {
    if (has_run) {
      write_u32(output, 0, kCookieRun | ((num_containers - 1) << 16));
    } else {
      write_u32(output, 0, kCookieNoRun);
      write_u32(output, sizeof(std::uint32_t), num_containers);
    }
  }

  // Write every run-bitmap byte directly. This initializes unused high bits to zero and avoids a
  // memset of the complete serialized allocation.
  auto* output_bytes = reinterpret_cast<std::uint8_t*>(output);
  for (auto byte = thread; byte < run_bitmap_bytes; byte += stride) {
    std::uint8_t value{};
    for (std::uint32_t bit = 0; bit < 8; ++bit) {
      auto const container = static_cast<std::uint32_t>(byte * 8 + bit);
      if (container < num_containers && kinds[container] == container_kind::run) {
        value |= static_cast<std::uint8_t>(1u << bit);
      }
    }
    output_bytes[sizeof(std::uint32_t) + byte] = value;
  }

  for (auto container = thread; container < num_containers; container += stride) {
    auto const begin = container_starts[container];
    auto const end   = container + 1 < num_containers ? container_starts[container + 1] : *id_count;
    auto const descriptor = descriptor_offset + container * 2 * sizeof(std::uint16_t);
    write_u16(output, descriptor, static_cast<std::uint16_t>(ids[begin] >> 16));
    write_u16(
      output, descriptor + sizeof(std::uint16_t), static_cast<std::uint16_t>(end - begin - 1));
    if (store_offsets) {
      write_u32(output,
                offsets_offset + container * sizeof(std::uint32_t),
                static_cast<std::uint32_t>(header_size + payload_offsets[container]));
    }
  }
}

/** Encode array, bitmap, and run payloads directly into their final device offsets. */
__global__ void encode_payloads_kernel(std::uint32_t const* ids,
                                       std::int64_t const* id_count,
                                       std::int64_t const* container_starts,
                                       std::uint32_t num_containers,
                                       container_kind const* kinds,
                                       std::uint64_t const* payload_sizes,
                                       std::uint64_t const* payload_offsets,
                                       std::size_t header_size,
                                       cuda::std::byte* output)
{
  auto const container = static_cast<std::uint32_t>(blockIdx.x);
  if (container >= num_containers) { return; }
  auto const begin   = container_starts[container];
  auto const end     = container + 1 < num_containers ? container_starts[container + 1] : *id_count;
  auto const payload = header_size + payload_offsets[container];

  if (kinds[container] == container_kind::array) {
    for (auto i = begin + threadIdx.x; i < end; i += blockDim.x) {
      write_u16(output,
                payload + static_cast<std::size_t>(i - begin) * sizeof(std::uint16_t),
                static_cast<std::uint16_t>(ids[i] & 0xffffu));
    }
    return;
  }

  using run_scan = cub::BlockScan<std::uint32_t, kBuilderBlockSize>;
  union payload_scratch {
    std::uint64_t bitmap_words[kBitmapBytes / sizeof(std::uint64_t)];
    typename run_scan::TempStorage run_scan_storage;
  };
  __shared__ payload_scratch scratch;
  __shared__ std::uint32_t run_base;
  __shared__ std::uint32_t tile_runs;

  if (kinds[container] == container_kind::bitmap) {
    constexpr std::uint32_t words = kBitmapBytes / sizeof(std::uint64_t);
    for (std::uint32_t word = threadIdx.x; word < words; word += blockDim.x) {
      scratch.bitmap_words[word] = 0;
    }
    __syncthreads();
    for (auto i = begin + threadIdx.x; i < end; i += blockDim.x) {
      auto const lower = ids[i] & 0xffffu;
      atomicOr(reinterpret_cast<unsigned long long*>(&scratch.bitmap_words[lower / 64]),
               static_cast<unsigned long long>(std::uint64_t{1} << (lower % 64)));
    }
    __syncthreads();
    for (std::uint32_t word = threadIdx.x; word < words; word += blockDim.x) {
      write_u64(output, payload + word * sizeof(std::uint64_t), scratch.bitmap_words[word]);
    }
    return;
  }

  auto const num_runs = static_cast<std::uint16_t>(
    (payload_sizes[container] - sizeof(std::uint16_t)) / (2 * sizeof(std::uint16_t)));
  if (threadIdx.x == 0) {
    write_u16(output, payload, num_runs);
    run_base = 0;
  }
  __syncthreads();

  // A block scan assigns stable output positions to run starts in each tile. Each run-start thread
  // walks only its own run, so total work remains O(container cardinality) while high-run-count
  // containers use the complete CTA instead of one serial thread.
  for (auto tile = begin; tile < end; tile += blockDim.x) {
    auto const i = tile + threadIdx.x;
    std::uint32_t const is_run_start =
      i < end && (i == begin ||
                  static_cast<std::uint64_t>(ids[i]) != static_cast<std::uint64_t>(ids[i - 1]) + 1)
        ? 1u
        : 0u;
    std::uint32_t run_rank{};
    std::uint32_t block_runs{};
    run_scan(scratch.run_scan_storage).ExclusiveSum(is_run_start, run_rank, block_runs);
    if (threadIdx.x == 0) { tile_runs = block_runs; }
    __syncthreads();

    if (is_run_start != 0) {
      auto j = i + 1;
      while (j < end &&
             static_cast<std::uint64_t>(ids[j]) == static_cast<std::uint64_t>(ids[j - 1]) + 1) {
        ++j;
      }
      auto const start = static_cast<std::uint16_t>(ids[i] & 0xffffu);
      auto const last  = static_cast<std::uint16_t>(ids[j - 1] & 0xffffu);
      auto const run_offset =
        payload + sizeof(std::uint16_t) +
        static_cast<std::size_t>(run_base + run_rank) * 2 * sizeof(std::uint16_t);
      write_u16(output, run_offset, start);
      write_u16(
        output, run_offset + sizeof(std::uint16_t), static_cast<std::uint16_t>(last - start));
    }
    __syncthreads();
    if (threadIdx.x == 0) { run_base += tile_runs; }
    __syncthreads();
  }
}

/**
 * Sort and analyze an entire sparse allowlist in one CTA.
 *
 * The unsorted specialization uses a blocked 128 x 1 radix sort. Padding uses UINT32_MAX;
 * writing only the first `size` sorted items is still correct when UINT32_MAX itself is a valid ID
 * because all padding values compare equal to that final real value. Thread zero then walks at most
 * 128 normalized IDs to
 * build compact per-container metadata and the exact serialized payload size.
 *
 * Inputs are promised unique by the public API. This kernel deliberately does
 * not spend work or storage checking or collapsing duplicates.
 */
template <bool SortInput>
__global__ void analyze_sparse_ids_kernel(std::uint32_t const* ids,
                                          std::uint32_t size,
                                          std::uint64_t dataset_rows,
                                          std::uint32_t* sorted_ids,
                                          sparse_container_metadata* metadata,
                                          device_build_summary* summary)
{
  if constexpr (SortInput) {
    using block_sort =
      cub::BlockRadixSort<std::uint32_t, kSparseBuilderBlockSize, kSparseItemsPerThread>;
    __shared__ typename block_sort::TempStorage sort_storage;
    std::uint32_t thread_ids[kSparseItemsPerThread];

#pragma unroll
    for (int item = 0; item < kSparseItemsPerThread; ++item) {
      auto const index = static_cast<std::uint32_t>(threadIdx.x) * kSparseItemsPerThread + item;
      thread_ids[item] = index < size ? ids[index] : std::numeric_limits<std::uint32_t>::max();
    }
    block_sort(sort_storage).Sort(thread_ids);
#pragma unroll
    for (int item = 0; item < kSparseItemsPerThread; ++item) {
      auto const index = static_cast<std::uint32_t>(threadIdx.x) * kSparseItemsPerThread + item;
      if (index < size) { sorted_ids[index] = thread_ids[item]; }
    }
    __syncthreads();
  }

  if (threadIdx.x != 0) { return; }
  auto const* normalized_ids = SortInput ? sorted_ids : ids;

  summary->cardinality    = size;
  summary->payload_bytes  = 0;
  summary->num_containers = 0;
  summary->has_run        = 0;
  summary->invalid        = 0;

  // Validate before writing container metadata. For a valid row, the logical
  // dataset shape bounds the number of containers allocated by the host.
  for (std::uint32_t i = 0; i < size; ++i) {
    if (static_cast<std::uint64_t>(normalized_ids[i]) >= dataset_rows) {
      summary->invalid = 1;
      return;
    }
  }

  std::uint32_t container{};
  std::uint32_t begin{};
  std::uint32_t payload_offset{};
  while (begin < size) {
    auto const key = normalized_ids[begin] >> 16;
    auto end       = begin + 1;
    std::uint32_t runs{1};
    while (end < size && (normalized_ids[end] >> 16) == key) {
      runs += static_cast<std::uint64_t>(normalized_ids[end]) !=
                  static_cast<std::uint64_t>(normalized_ids[end - 1]) + 1
                ? 1u
                : 0u;
      ++end;
    }

    auto const cardinality = end - begin;
    auto const array_size  = cardinality * sizeof(std::uint16_t);
    auto const run_size    = sizeof(std::uint16_t) + runs * 2 * sizeof(std::uint16_t);
    auto const use_run     = run_size < array_size;
    metadata[container] =
      sparse_container_metadata{begin,
                                payload_offset,
                                static_cast<std::uint16_t>(runs),
                                use_run ? container_kind::run : container_kind::array,
                                0};
    payload_offset += use_run ? run_size : array_size;
    summary->has_run |= use_run ? 1u : 0u;
    ++container;
    begin = end;
  }

  summary->payload_bytes  = payload_offset;
  summary->num_containers = container;
}

/**
 * Encode a sparse row in one CTA after the host has allocated the exact byte
 * count reported by `analyze_sparse_ids_kernel`.
 *
 * Thread zero writes every header byte, including the run bitmap, so this path
 * needs no output memset. Array values are striped across the CTA; thread zero
 * writes the comparatively small run payloads. Bitmap payloads cannot occur
 * below the sparse cardinality threshold.
 */
__global__ void encode_sparse_row_kernel(std::uint32_t const* ids,
                                         std::uint32_t cardinality,
                                         sparse_container_metadata const* metadata,
                                         std::uint32_t num_containers,
                                         std::size_t header_size,
                                         bool has_run,
                                         cuda::std::byte* output,
                                         ref_type* reference)
{
  bool const store_offsets = !has_run || num_containers >= kOffsetThreshold;
  std::size_t descriptor_offset{};
  std::size_t offsets_offset{};

  if (threadIdx.x == 0) {
    if (has_run) {
      write_u32(output, 0, kCookieRun | ((num_containers - 1) << 16));
      auto const run_bitmap_bytes = (num_containers + 7) / 8;
      for (std::uint32_t byte = 0; byte < run_bitmap_bytes; ++byte) {
        std::uint8_t value{};
        for (std::uint32_t bit = 0; bit < 8; ++bit) {
          auto const container = byte * 8 + bit;
          if (container < num_containers && metadata[container].kind == container_kind::run) {
            value |= static_cast<std::uint8_t>(1u << bit);
          }
        }
        reinterpret_cast<std::uint8_t*>(output)[sizeof(std::uint32_t) + byte] = value;
      }
      descriptor_offset = sizeof(std::uint32_t) + run_bitmap_bytes;
      offsets_offset    = descriptor_offset + num_containers * 2 * sizeof(std::uint16_t);
    } else {
      write_u32(output, 0, kCookieNoRun);
      write_u32(output, sizeof(std::uint32_t), num_containers);
      descriptor_offset = 2 * sizeof(std::uint32_t);
      offsets_offset    = descriptor_offset + num_containers * 2 * sizeof(std::uint16_t);
    }

    for (std::uint32_t container = 0; container < num_containers; ++container) {
      auto const begin = metadata[container].begin;
      auto const end = container + 1 < num_containers ? metadata[container + 1].begin : cardinality;
      auto const descriptor = descriptor_offset + container * 2 * sizeof(std::uint16_t);
      write_u16(output, descriptor, static_cast<std::uint16_t>(ids[begin] >> 16));
      write_u16(
        output, descriptor + sizeof(std::uint16_t), static_cast<std::uint16_t>(end - begin - 1));
      if (store_offsets) {
        write_u32(output,
                  offsets_offset + container * sizeof(std::uint32_t),
                  static_cast<std::uint32_t>(header_size + metadata[container].payload_offset));
      }
    }
  }

  for (std::uint32_t container = 0; container < num_containers; ++container) {
    auto const begin = metadata[container].begin;
    auto const end   = container + 1 < num_containers ? metadata[container + 1].begin : cardinality;
    auto const payload = header_size + metadata[container].payload_offset;
    if (metadata[container].kind == container_kind::array) {
      for (auto i = begin + threadIdx.x; i < end; i += blockDim.x) {
        write_u16(output,
                  payload + static_cast<std::size_t>(i - begin) * sizeof(std::uint16_t),
                  static_cast<std::uint16_t>(ids[i] & 0xffffu));
      }
      continue;
    }

    if (threadIdx.x == 0) {
      write_u16(output, payload, metadata[container].runs);
      std::uint16_t run_index{};
      for (auto i = begin; i < end;) {
        auto const start = static_cast<std::uint16_t>(ids[i] & 0xffffu);
        auto j           = i + 1;
        while (j < end &&
               static_cast<std::uint64_t>(ids[j]) == static_cast<std::uint64_t>(ids[j - 1]) + 1) {
          ++j;
        }
        auto const last       = static_cast<std::uint16_t>(ids[j - 1] & 0xffffu);
        auto const run_offset = payload + sizeof(std::uint16_t) +
                                static_cast<std::size_t>(run_index) * 2 * sizeof(std::uint16_t);
        write_u16(output, run_offset, start);
        write_u16(
          output, run_offset + sizeof(std::uint16_t), static_cast<std::uint16_t>(last - start));
        ++run_index;
        i = j;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) { ::new (static_cast<void*>(reference)) ref_type{output}; }
}

device_build_result build_sparse_from_device_ids(
  raft::resources const& res,
  std::size_t dataset_rows,
  raft::device_vector_view<const std::uint32_t, std::int64_t> ids,
  bool pre_sorted)
{
  common::nvtx::range<common::nvtx::domain::cuvs> build_scope("roaring_allowlist::build_sparse");
  auto const stream = raft::resource::get_cuda_stream(res);
  auto const size   = static_cast<std::size_t>(ids.extent(0));
  auto const num_chunks =
    (static_cast<std::uint64_t>(dataset_rows) + (std::uint64_t{1} << 16) - 1) >> 16;
  auto const max_containers = std::min<std::size_t>(size, static_cast<std::size_t>(num_chunks));
  sparse_scratch_layout const layout{size, max_containers, !pre_sorted};
  rmm::device_uvector<cuda::std::byte> scratch(layout.bytes, stream);

  auto* summary = reinterpret_cast<device_build_summary*>(scratch.data());
  auto* sorted_ids =
    pre_sorted ? nullptr
               : reinterpret_cast<std::uint32_t*>(scratch.data() + layout.sorted_ids_offset);
  auto* metadata =
    reinterpret_cast<sparse_container_metadata*>(scratch.data() + layout.metadata_offset);

  {
    common::nvtx::range<common::nvtx::domain::cuvs> stage_scope(
      "roaring_allowlist::sparse_analysis");
    if (pre_sorted) {
      analyze_sparse_ids_kernel<false>
        <<<1, kSparseBuilderBlockSize, 0, stream>>>(ids.data_handle(),
                                                    static_cast<std::uint32_t>(size),
                                                    dataset_rows,
                                                    nullptr,
                                                    metadata,
                                                    summary);
    } else {
      analyze_sparse_ids_kernel<true>
        <<<1, kSparseBuilderBlockSize, 0, stream>>>(ids.data_handle(),
                                                    static_cast<std::uint32_t>(size),
                                                    dataset_rows,
                                                    sorted_ids,
                                                    metadata,
                                                    summary);
    }
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  device_build_summary host_summary;
  {
    common::nvtx::range<common::nvtx::domain::cuvs> stage_scope("roaring_allowlist::size_readback");
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      &host_summary, summary, sizeof(host_summary), cudaMemcpyDeviceToHost, stream));
    raft::resource::sync_stream(res);
  }
  RAFT_EXPECTS(host_summary.invalid == 0,
               "Roaring allowlist ID must be smaller than dataset_rows.");
  RAFT_EXPECTS(host_summary.cardinality > 0 && host_summary.num_containers > 0,
               "Internal error: nonempty sparse Roaring input produced an empty device build.");

  auto const header_size =
    portable_header_size(host_summary.num_containers, host_summary.has_run != 0);
  auto const serialized_bytes = header_size + static_cast<std::size_t>(host_summary.payload_bytes);
  rmm::device_uvector<cuda::std::byte> output(0, stream);
  {
    common::nvtx::range<common::nvtx::domain::cuvs> stage_scope(
      "roaring_allowlist::final_allocation");
    output.resize(owned_storage_bytes(serialized_bytes), stream);
  }
  auto const* normalized_ids = pre_sorted ? ids.data_handle() : sorted_ids;
  common::nvtx::range<common::nvtx::domain::cuvs> encode_scope(
    "roaring_allowlist::sparse_encode_and_ref");
  encode_sparse_row_kernel<<<1, kSparseBuilderBlockSize, 0, stream>>>(
    normalized_ids,
    static_cast<std::uint32_t>(size),
    metadata,
    host_summary.num_containers,
    header_size,
    host_summary.has_run != 0,
    output.data(),
    reinterpret_cast<ref_type*>(output.data() + reference_offset(serialized_bytes)));
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  return {std::move(output), serialized_bytes, size, true};
}

/**
 * Build a standard portable Roaring row from device IDs.
 *
 * Very sparse rows use the single-CTA builder above. Larger rows use one ID
 * array for device-wide radix sorting. Both pre-sorted paths
 * use the caller's strictly increasing IDs directly and allocate no normalization ID array.
 * Boundary scan data, CUB workspace, and O(min(input IDs, 65536)) container metadata remain
 * cardinality-scaled. In particular, the builder never allocates storage proportional to
 * `dataset_rows` bits.
 *
 * Pre-sorted ordering and uniqueness are unchecked caller promises.
 */
device_build_result build_from_device_ids(
  raft::resources const& res,
  std::size_t dataset_rows,
  raft::device_vector_view<const std::uint32_t, std::int64_t> ids,
  bool pre_sorted)
{
  common::nvtx::range<common::nvtx::domain::cuvs> build_scope("roaring_allowlist::build_from_ids");
  auto const stream = raft::resource::get_cuda_stream(res);
  auto const size   = static_cast<std::size_t>(ids.extent(0));
  if (size == 0) { return {rmm::device_uvector<cuda::std::byte>(0, stream), 0, 0, false}; }
  auto const sparse_cutoff = pre_sorted ? kSparseBuilderMaxPreSortedIds : kSparseBuilderMaxIds;
  if (size <= sparse_cutoff) {
    return build_sparse_from_device_ids(res, dataset_rows, ids, pre_sorted);
  }

  auto const num_chunks =
    (static_cast<std::uint64_t>(dataset_rows) + (std::uint64_t{1} << 16) - 1) >> 16;
  auto const max_containers  = std::min<std::size_t>(size, static_cast<std::size_t>(num_chunks));
  auto const item_count      = static_cast<std::int64_t>(size);
  auto const container_slots = static_cast<std::int64_t>(max_containers);
  constexpr int sort_end_bit = std::numeric_limits<std::uint32_t>::digits;

  std::size_t sort_workspace_bytes{};
  std::size_t select_workspace_bytes{};
  std::size_t payload_scan_workspace_bytes{};
  if (!pre_sorted) {
    RAFT_CUDA_TRY(cub::DeviceRadixSort::SortKeys(nullptr,
                                                 sort_workspace_bytes,
                                                 ids.data_handle(),
                                                 static_cast<std::uint32_t*>(nullptr),
                                                 item_count,
                                                 0,
                                                 sort_end_bit,
                                                 stream));
  }
  auto counting = thrust::make_counting_iterator<std::int64_t>(0);
  RAFT_CUDA_TRY(cub::DeviceSelect::If(nullptr,
                                      select_workspace_bytes,
                                      counting,
                                      static_cast<std::int64_t*>(nullptr),
                                      static_cast<std::int64_t*>(nullptr),
                                      item_count,
                                      is_container_start{ids.data_handle(), nullptr},
                                      stream));
  RAFT_CUDA_TRY(cub::DeviceScan::ExclusiveSum(nullptr,
                                              payload_scan_workspace_bytes,
                                              static_cast<std::uint64_t*>(nullptr),
                                              static_cast<std::uint64_t*>(nullptr),
                                              container_slots,
                                              stream));
  auto const workspace_bytes =
    std::max(sort_workspace_bytes, std::max(select_workspace_bytes, payload_scan_workspace_bytes));
  general_scratch_layout const layout{size, max_containers, !pre_sorted, workspace_bytes};
  rmm::device_uvector<cuda::std::byte> scratch(layout.bytes, stream);

  auto* sorted_ids =
    pre_sorted ? nullptr
               : reinterpret_cast<std::uint32_t*>(scratch.data() + layout.sorted_ids_offset);
  auto* id_count    = reinterpret_cast<std::int64_t*>(scratch.data() + layout.id_count_offset);
  auto* valid_count = reinterpret_cast<std::int64_t*>(scratch.data() + layout.valid_count_offset);
  auto* selected_count =
    reinterpret_cast<std::int64_t*>(scratch.data() + layout.selected_count_offset);
  auto* container_starts =
    reinterpret_cast<std::int64_t*>(scratch.data() + layout.container_starts_offset);
  auto* num_containers =
    reinterpret_cast<std::uint32_t*>(scratch.data() + layout.num_containers_offset);
  auto* kinds = reinterpret_cast<container_kind*>(scratch.data() + layout.kinds_offset);
  auto* payload_sizes =
    reinterpret_cast<std::uint64_t*>(scratch.data() + layout.payload_sizes_offset);
  auto* payload_offsets =
    reinterpret_cast<std::uint64_t*>(scratch.data() + layout.payload_offsets_offset);
  auto* has_run = reinterpret_cast<std::uint32_t*>(scratch.data() + layout.has_run_offset);
  auto* device_summary =
    reinterpret_cast<device_build_summary*>(scratch.data() + layout.summary_offset);
  auto* workspace = scratch.data() + layout.workspace_offset;

  if (!pre_sorted) {
    common::nvtx::range<common::nvtx::domain::cuvs> stage_scope("roaring_allowlist::radix_sort");
    RAFT_CUDA_TRY(cub::DeviceRadixSort::SortKeys(workspace,
                                                 sort_workspace_bytes,
                                                 ids.data_handle(),
                                                 sorted_ids,
                                                 item_count,
                                                 0,
                                                 sort_end_bit,
                                                 stream));
  }
  auto const* normalized_ids = pre_sorted ? ids.data_handle() : sorted_ids;
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(id_count, &item_count, sizeof(item_count), cudaMemcpyHostToDevice, stream));

  {
    common::nvtx::range<common::nvtx::domain::cuvs> stage_scope(
      "roaring_allowlist::container_discovery");
    find_valid_count_kernel<<<1, 1, 0, stream>>>(
      normalized_ids, id_count, dataset_rows, valid_count);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    RAFT_CUDA_TRY(cub::DeviceSelect::If(workspace,
                                        select_workspace_bytes,
                                        counting,
                                        container_starts,
                                        selected_count,
                                        item_count,
                                        is_container_start{normalized_ids, valid_count},
                                        stream));
    narrow_container_count_kernel<<<1, 1, 0, stream>>>(selected_count, num_containers);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  {
    common::nvtx::range<common::nvtx::domain::cuvs> stage_scope(
      "roaring_allowlist::container_analysis");
    RAFT_CUDA_TRY(
      cudaMemsetAsync(payload_sizes, 0, max_containers * sizeof(std::uint64_t), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(has_run, 0, sizeof(std::uint32_t), stream));
    analyze_containers_kernel<<<static_cast<unsigned int>(max_containers),
                                kBuilderBlockSize,
                                0,
                                stream>>>(
      normalized_ids, valid_count, container_starts, num_containers, kinds, payload_sizes, has_run);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    RAFT_CUDA_TRY(cub::DeviceScan::ExclusiveSum(workspace,
                                                payload_scan_workspace_bytes,
                                                payload_sizes,
                                                payload_offsets,
                                                container_slots,
                                                stream));
    finish_device_analysis_kernel<<<1, 1, 0, stream>>>(id_count,
                                                       valid_count,
                                                       num_containers,
                                                       has_run,
                                                       payload_sizes,
                                                       payload_offsets,
                                                       device_summary);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  device_build_summary summary;
  {
    common::nvtx::range<common::nvtx::domain::cuvs> stage_scope("roaring_allowlist::size_readback");
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(&summary, device_summary, sizeof(summary), cudaMemcpyDeviceToHost, stream));
    raft::resource::sync_stream(res);
  }
  RAFT_EXPECTS(summary.invalid == 0, "Roaring allowlist ID must be smaller than dataset_rows.");
  RAFT_EXPECTS(summary.cardinality > 0 && summary.num_containers > 0,
               "Internal error: nonempty Roaring input produced an empty device build.");

  auto const header_size = portable_header_size(summary.num_containers, summary.has_run != 0);
  RAFT_EXPECTS(summary.payload_bytes <= std::numeric_limits<std::uint32_t>::max() - header_size,
               "Portable Roaring row exceeds the 32-bit offset range.");
  auto const serialized_bytes = header_size + static_cast<std::size_t>(summary.payload_bytes);
  rmm::device_uvector<cuda::std::byte> output(0, stream);
  {
    common::nvtx::range<common::nvtx::domain::cuvs> stage_scope(
      "roaring_allowlist::final_allocation");
    output.resize(owned_storage_bytes(serialized_bytes), stream);
  }
  auto const header_items =
    std::max<std::size_t>(summary.num_containers, (summary.num_containers + 7) / 8);
  {
    common::nvtx::range<common::nvtx::domain::cuvs> stage_scope("roaring_allowlist::header_encode");
    encode_header_kernel<<<grid_size_for(header_items), kBuilderBlockSize, 0, stream>>>(
      normalized_ids,
      valid_count,
      container_starts,
      summary.num_containers,
      kinds,
      payload_offsets,
      header_size,
      summary.has_run != 0,
      output.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
  common::nvtx::range<common::nvtx::domain::cuvs> payload_scope(
    "roaring_allowlist::payload_encode");
  encode_payloads_kernel<<<summary.num_containers, kBuilderBlockSize, 0, stream>>>(
    normalized_ids,
    valid_count,
    container_starts,
    summary.num_containers,
    kinds,
    payload_sizes,
    payload_offsets,
    header_size,
    output.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  return {
    std::move(output), serialized_bytes, static_cast<std::size_t>(summary.cardinality), false};
}

struct batched_device_build_result {
  rmm::device_uvector<cuda::std::byte> storage;
  std::vector<std::size_t> row_offsets;
  std::vector<std::size_t> reference_offsets;
  std::vector<std::size_t> serialized_bytes;
  std::vector<std::size_t> cardinalities;
  bool references_initialized{};
};

struct packed_rows_layout {
  std::vector<std::size_t> row_offsets;
  std::vector<std::size_t> reference_offsets;
  std::size_t references_offset{};
  std::size_t bytes{};
};

packed_rows_layout make_packed_rows_layout(std::vector<std::size_t> const& serialized_bytes)
{
  packed_rows_layout result;
  result.row_offsets.resize(serialized_bytes.size());
  result.reference_offsets.resize(serialized_bytes.size());
  std::size_t cursor{};
  bool any_nonempty{};
  for (std::size_t row = 0; row < serialized_bytes.size(); ++row) {
    if (serialized_bytes[row] == 0) { continue; }
    any_nonempty            = true;
    cursor                  = align_up(cursor, alignof(std::max_align_t));
    result.row_offsets[row] = cursor;
    cursor += serialized_bytes[row];
  }
  if (!any_nonempty) { return result; }
  result.references_offset = align_up(cursor, alignof(ref_type));
  for (std::size_t row = 0; row < serialized_bytes.size(); ++row) {
    result.reference_offsets[row] = result.references_offset + row * sizeof(ref_type);
  }
  result.bytes = result.references_offset + serialized_bytes.size() * sizeof(ref_type);
  return result;
}

struct batch_general_scratch_layout {
  batch_general_scratch_layout(std::size_t total_ids,
                               std::size_t rows,
                               std::size_t max_containers,
                               bool store_sorted_ids,
                               std::size_t workspace_bytes)
  {
    std::size_t cursor{};
    auto reserve = [&](std::size_t count, std::size_t item_size, std::size_t alignment) {
      auto const result = align_up(cursor, alignment);
      cursor            = result + count * item_size;
      return result;
    };
    if (store_sorted_ids) {
      sorted_ids_offset = reserve(total_ids, sizeof(std::uint32_t), alignof(std::uint32_t));
    }
    valid_counts_offset     = reserve(rows, sizeof(std::int64_t), alignof(std::int64_t));
    flags_offset            = reserve(total_ids, sizeof(std::uint8_t), alignof(std::uint8_t));
    selected_count_offset   = reserve(1, sizeof(std::int64_t), alignof(std::int64_t));
    container_starts_offset = reserve(max_containers, sizeof(std::int64_t), alignof(std::int64_t));
    row_container_offsets_offset = reserve(rows + 1, sizeof(std::int64_t), alignof(std::int64_t));
    container_rows_offset = reserve(max_containers, sizeof(std::uint32_t), alignof(std::uint32_t));
    kinds_offset         = reserve(max_containers, sizeof(container_kind), alignof(container_kind));
    payload_sizes_offset = reserve(max_containers, sizeof(std::uint64_t), alignof(std::uint64_t));
    payload_offsets_offset = reserve(max_containers, sizeof(std::uint64_t), alignof(std::uint64_t));
    has_run_offset         = reserve(rows, sizeof(std::uint32_t), alignof(std::uint32_t));
    summaries_offset = reserve(rows, sizeof(device_build_summary), alignof(device_build_summary));
    output_offsets_offset = reserve(rows, sizeof(std::uint64_t), alignof(std::uint64_t));
    workspace_offset = reserve(workspace_bytes, sizeof(cuda::std::byte), alignof(std::max_align_t));
    bytes            = cursor;
  }

  std::size_t sorted_ids_offset{};
  std::size_t valid_counts_offset{};
  std::size_t flags_offset{};
  std::size_t selected_count_offset{};
  std::size_t container_starts_offset{};
  std::size_t row_container_offsets_offset{};
  std::size_t container_rows_offset{};
  std::size_t kinds_offset{};
  std::size_t payload_sizes_offset{};
  std::size_t payload_offsets_offset{};
  std::size_t has_run_offset{};
  std::size_t summaries_offset{};
  std::size_t output_offsets_offset{};
  std::size_t workspace_offset{};
  std::size_t bytes{};
};

__global__ void find_batch_valid_counts_kernel(std::uint32_t const* ids,
                                               std::int64_t const* indptr,
                                               std::int64_t rows,
                                               std::uint64_t dataset_rows,
                                               std::int64_t* valid_counts)
{
  auto const row = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) { return; }
  auto const begin = indptr[row];
  auto const width = indptr[row + 1] - begin;
  std::int64_t first{};
  auto last = width;
  while (first < last) {
    auto const middle = first + (last - first) / 2;
    if (static_cast<std::uint64_t>(ids[begin + middle]) < dataset_rows) {
      first = middle + 1;
    } else {
      last = middle;
    }
  }
  valid_counts[row] = first;
}

__global__ void mark_batch_container_starts_kernel(std::uint32_t const* ids,
                                                   std::int64_t const* indptr,
                                                   std::int64_t const* valid_counts,
                                                   std::int64_t rows,
                                                   std::uint8_t* flags)
{
  auto const row = static_cast<std::int64_t>(blockIdx.x);
  if (row >= rows) { return; }
  auto const begin = indptr[row];
  auto const width = indptr[row + 1] - begin;
  auto const valid = valid_counts[row];
  for (std::int64_t column = threadIdx.x; column < width; column += blockDim.x) {
    auto const index = begin + column;
    flags[index] =
      column < valid && (column == 0 || (ids[index - 1] >> 16) != (ids[index] >> 16)) ? 1 : 0;
  }
}

__global__ void find_row_container_offsets_kernel(std::int64_t const* container_starts,
                                                  std::int64_t const* selected_count,
                                                  std::int64_t const* indptr,
                                                  std::int64_t rows,
                                                  std::int64_t* row_offsets)
{
  auto const row = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row > rows) { return; }
  auto const target = indptr[row];
  std::int64_t first{};
  auto last = *selected_count;
  while (first < last) {
    auto const middle = first + (last - first) / 2;
    if (container_starts[middle] < target) {
      first = middle + 1;
    } else {
      last = middle;
    }
  }
  row_offsets[row] = first;
}

__global__ void fill_container_rows_kernel(std::int64_t const* row_container_offsets,
                                           std::int64_t rows,
                                           std::uint32_t* container_rows)
{
  auto const row = static_cast<std::int64_t>(blockIdx.x);
  if (row >= rows) { return; }
  for (auto container = row_container_offsets[row] + threadIdx.x;
       container < row_container_offsets[row + 1];
       container += blockDim.x) {
    container_rows[container] = static_cast<std::uint32_t>(row);
  }
}

__global__ void analyze_batch_containers_kernel(std::uint32_t const* ids,
                                                std::int64_t const* indptr,
                                                std::int64_t const* valid_counts,
                                                std::int64_t const* container_starts,
                                                std::int64_t const* selected_count,
                                                std::uint32_t const* container_rows,
                                                container_kind* kinds,
                                                std::uint64_t* payload_sizes,
                                                std::uint32_t* has_run)
{
  auto const container = static_cast<std::int64_t>(blockIdx.x);
  auto const count     = *selected_count;
  if (container >= count) { return; }
  auto const row     = static_cast<std::int64_t>(container_rows[container]);
  auto const begin   = container_starts[container];
  auto const row_end = indptr[row] + valid_counts[row];
  auto const end     = container + 1 < count && container_rows[container + 1] == row
                         ? container_starts[container + 1]
                         : row_end;
  std::uint32_t local_runs{};
  for (auto i = begin + threadIdx.x; i < end; i += blockDim.x) {
    local_runs +=
      i == begin || static_cast<std::uint64_t>(ids[i]) != static_cast<std::uint64_t>(ids[i - 1]) + 1
        ? 1u
        : 0u;
  }
  using block_reduce = cub::BlockReduce<std::uint32_t, kBuilderBlockSize>;
  __shared__ typename block_reduce::TempStorage reduction_storage;
  auto const runs = block_reduce(reduction_storage).Sum(local_runs);
  if (threadIdx.x != 0) { return; }
  auto const cardinality = static_cast<std::uint64_t>(end - begin);
  auto const normal_size =
    cardinality <= kArrayCardinality ? cardinality * sizeof(std::uint16_t) : kBitmapBytes;
  auto const run_size = sizeof(std::uint16_t) + runs * 2 * sizeof(std::uint16_t);
  if (run_size < normal_size) {
    kinds[container]         = container_kind::run;
    payload_sizes[container] = run_size;
    atomicExch(has_run + row, 1u);
  } else if (cardinality <= kArrayCardinality) {
    kinds[container]         = container_kind::array;
    payload_sizes[container] = normal_size;
  } else {
    kinds[container]         = container_kind::bitmap;
    payload_sizes[container] = normal_size;
  }
}

__global__ void finish_batch_rows_kernel(std::int64_t rows,
                                         std::int64_t const* indptr,
                                         std::int64_t const* valid_counts,
                                         std::int64_t const* row_container_offsets,
                                         std::uint32_t const* has_run,
                                         std::uint64_t const* payload_sizes,
                                         std::uint64_t const* payload_offsets,
                                         device_build_summary* summaries)
{
  auto const row = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) { return; }
  auto const begin       = row_container_offsets[row];
  auto const end         = row_container_offsets[row + 1];
  auto const cardinality = indptr[row + 1] - indptr[row];
  auto& summary          = summaries[row];
  summary.cardinality    = cardinality;
  summary.num_containers = static_cast<std::uint32_t>(end - begin);
  summary.has_run        = has_run[row];
  summary.payload_bytes =
    begin == end ? 0 : payload_offsets[end - 1] + payload_sizes[end - 1] - payload_offsets[begin];
  summary.invalid = valid_counts[row] != cardinality;
}

__global__ void encode_batch_headers_kernel(std::uint32_t const* ids,
                                            std::int64_t const* indptr,
                                            std::int64_t const* valid_counts,
                                            std::int64_t const* container_starts,
                                            std::int64_t const* row_container_offsets,
                                            container_kind const* kinds,
                                            std::uint64_t const* payload_offsets,
                                            device_build_summary const* summaries,
                                            std::uint64_t const* output_offsets,
                                            cuda::std::byte* storage)
{
  auto const row     = static_cast<std::int64_t>(blockIdx.x);
  auto const summary = summaries[row];
  if (summary.num_containers == 0) { return; }
  auto const first_container  = row_container_offsets[row];
  auto* output                = storage + output_offsets[row];
  auto const has_run          = summary.has_run != 0;
  auto const run_bitmap_bytes = has_run ? (summary.num_containers + 7) / 8 : 0;
  auto const descriptor_offset =
    has_run ? sizeof(std::uint32_t) + run_bitmap_bytes : 2 * sizeof(std::uint32_t);
  auto const offsets_offset =
    descriptor_offset + summary.num_containers * 2 * sizeof(std::uint16_t);
  auto const header_size   = portable_header_size(summary.num_containers, has_run);
  bool const store_offsets = !has_run || summary.num_containers >= kOffsetThreshold;
  if (threadIdx.x == 0) {
    if (has_run) {
      write_u32(output, 0, kCookieRun | ((summary.num_containers - 1) << 16));
    } else {
      write_u32(output, 0, kCookieNoRun);
      write_u32(output, sizeof(std::uint32_t), summary.num_containers);
    }
  }
  auto* output_bytes = reinterpret_cast<std::uint8_t*>(output);
  for (std::uint32_t byte = threadIdx.x; byte < run_bitmap_bytes; byte += blockDim.x) {
    std::uint8_t value{};
    for (std::uint32_t bit = 0; bit < 8; ++bit) {
      auto const local = byte * 8 + bit;
      if (local < summary.num_containers && kinds[first_container + local] == container_kind::run) {
        value |= static_cast<std::uint8_t>(1u << bit);
      }
    }
    output_bytes[sizeof(std::uint32_t) + byte] = value;
  }
  auto const row_end = indptr[row] + valid_counts[row];
  for (std::uint32_t local = threadIdx.x; local < summary.num_containers; local += blockDim.x) {
    auto const container = first_container + local;
    auto const begin     = container_starts[container];
    auto const end = local + 1 < summary.num_containers ? container_starts[container + 1] : row_end;
    auto const descriptor = descriptor_offset + local * 2 * sizeof(std::uint16_t);
    write_u16(output, descriptor, static_cast<std::uint16_t>(ids[begin] >> 16));
    write_u16(
      output, descriptor + sizeof(std::uint16_t), static_cast<std::uint16_t>(end - begin - 1));
    if (store_offsets) {
      auto const local_payload = payload_offsets[container] - payload_offsets[first_container];
      write_u32(output,
                offsets_offset + local * sizeof(std::uint32_t),
                static_cast<std::uint32_t>(header_size + local_payload));
    }
  }
}

__global__ void encode_batch_payloads_kernel(std::uint32_t const* ids,
                                             std::int64_t const* indptr,
                                             std::int64_t const* valid_counts,
                                             std::int64_t const* container_starts,
                                             std::int64_t const* row_container_offsets,
                                             std::uint32_t const* container_rows,
                                             std::int64_t num_containers,
                                             container_kind const* kinds,
                                             std::uint64_t const* payload_sizes,
                                             std::uint64_t const* payload_offsets,
                                             device_build_summary const* summaries,
                                             std::uint64_t const* output_offsets,
                                             cuda::std::byte* storage)
{
  auto const container = static_cast<std::int64_t>(blockIdx.x);
  if (container >= num_containers) { return; }
  auto const row             = static_cast<std::int64_t>(container_rows[container]);
  auto const begin           = container_starts[container];
  auto const first_container = row_container_offsets[row];
  auto const local           = container - first_container;
  auto const summary         = summaries[row];
  auto const row_end         = indptr[row] + valid_counts[row];
  auto const end = local + 1 < summary.num_containers ? container_starts[container + 1] : row_end;
  auto* output   = storage + output_offsets[row];
  auto const payload = portable_header_size(summary.num_containers, summary.has_run != 0) +
                       payload_offsets[container] - payload_offsets[first_container];

  if (kinds[container] == container_kind::array) {
    for (auto i = begin + threadIdx.x; i < end; i += blockDim.x) {
      write_u16(output,
                payload + static_cast<std::size_t>(i - begin) * sizeof(std::uint16_t),
                static_cast<std::uint16_t>(ids[i] & 0xffffu));
    }
    return;
  }
  using run_scan = cub::BlockScan<std::uint32_t, kBuilderBlockSize>;
  union payload_scratch {
    std::uint64_t bitmap_words[kBitmapBytes / sizeof(std::uint64_t)];
    typename run_scan::TempStorage run_scan_storage;
  };
  __shared__ payload_scratch scratch;
  __shared__ std::uint32_t run_base;
  __shared__ std::uint32_t tile_runs;
  if (kinds[container] == container_kind::bitmap) {
    constexpr std::uint32_t words = kBitmapBytes / sizeof(std::uint64_t);
    for (std::uint32_t word = threadIdx.x; word < words; word += blockDim.x) {
      scratch.bitmap_words[word] = 0;
    }
    __syncthreads();
    for (auto i = begin + threadIdx.x; i < end; i += blockDim.x) {
      auto const lower = ids[i] & 0xffffu;
      atomicOr(reinterpret_cast<unsigned long long*>(&scratch.bitmap_words[lower / 64]),
               static_cast<unsigned long long>(std::uint64_t{1} << (lower % 64)));
    }
    __syncthreads();
    for (std::uint32_t word = threadIdx.x; word < words; word += blockDim.x) {
      write_u64(output, payload + word * sizeof(std::uint64_t), scratch.bitmap_words[word]);
    }
    return;
  }
  auto const num_runs = static_cast<std::uint16_t>(
    (payload_sizes[container] - sizeof(std::uint16_t)) / (2 * sizeof(std::uint16_t)));
  if (threadIdx.x == 0) {
    write_u16(output, payload, num_runs);
    run_base = 0;
  }
  __syncthreads();
  for (auto tile = begin; tile < end; tile += blockDim.x) {
    auto const i = tile + threadIdx.x;
    std::uint32_t const is_run_start =
      i < end && (i == begin ||
                  static_cast<std::uint64_t>(ids[i]) != static_cast<std::uint64_t>(ids[i - 1]) + 1)
        ? 1u
        : 0u;
    std::uint32_t run_rank{};
    std::uint32_t block_runs{};
    run_scan(scratch.run_scan_storage).ExclusiveSum(is_run_start, run_rank, block_runs);
    if (threadIdx.x == 0) { tile_runs = block_runs; }
    __syncthreads();
    if (is_run_start != 0) {
      auto j = i + 1;
      while (j < end &&
             static_cast<std::uint64_t>(ids[j]) == static_cast<std::uint64_t>(ids[j - 1]) + 1) {
        ++j;
      }
      auto const start = static_cast<std::uint16_t>(ids[i] & 0xffffu);
      auto const last  = static_cast<std::uint16_t>(ids[j - 1] & 0xffffu);
      auto const run_offset =
        payload + sizeof(std::uint16_t) +
        static_cast<std::size_t>(run_base + run_rank) * 2 * sizeof(std::uint16_t);
      write_u16(output, run_offset, start);
      write_u16(
        output, run_offset + sizeof(std::uint16_t), static_cast<std::uint16_t>(last - start));
    }
    __syncthreads();
    if (threadIdx.x == 0) { run_base += tile_runs; }
    __syncthreads();
  }
}

__global__ void initialize_batch_refs_kernel(cuda::std::byte const* storage,
                                             std::uint64_t const* row_offsets,
                                             device_build_summary const* summaries,
                                             ref_type* references,
                                             std::size_t rows)
{
  auto const row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < rows && summaries[row].num_containers != 0) {
    ::new (static_cast<void*>(references + row)) ref_type{storage + row_offsets[row]};
  }
}

batched_device_build_result build_general_rows(
  raft::resources const& res,
  std::size_t dataset_rows,
  raft::device_vector_view<const std::uint32_t, std::int64_t> ids,
  raft::device_vector_view<const std::int64_t, std::int64_t> indptr,
  std::vector<std::int64_t> const& host_indptr,
  bool pre_sorted)
{
  common::nvtx::range<common::nvtx::domain::cuvs> build_scope(
    "roaring_allowlist::build_general_batch");
  auto const stream     = raft::resource::get_cuda_stream(res);
  auto const rows       = host_indptr.size() - 1;
  auto const total_ids  = static_cast<std::size_t>(ids.extent(0));
  auto const item_count = static_cast<std::int64_t>(total_ids);
  auto const row_count  = static_cast<std::int64_t>(rows);
  auto const chunks =
    (static_cast<std::uint64_t>(dataset_rows) + (std::uint64_t{1} << 16) - 1) >> 16;
  std::size_t max_containers{};
  for (std::size_t row = 0; row < rows; ++row) {
    auto const width = static_cast<std::size_t>(host_indptr[row + 1] - host_indptr[row]);
    max_containers += std::min(width, static_cast<std::size_t>(chunks));
  }
  RAFT_EXPECTS(rows <= std::numeric_limits<unsigned int>::max(),
               "Batched Roaring construction has too many rows for one launch.");
  RAFT_EXPECTS(max_containers <= std::numeric_limits<unsigned int>::max(),
               "Batched Roaring construction has too many containers for one launch.");
  constexpr int sort_end_bit = std::numeric_limits<std::uint32_t>::digits;
  auto counting              = thrust::make_counting_iterator<std::int64_t>(0);

  std::size_t sort_workspace_bytes{};
  std::size_t select_workspace_bytes{};
  std::size_t scan_workspace_bytes{};
  if (!pre_sorted) {
    RAFT_CUDA_TRY(cub::DeviceSegmentedRadixSort::SortKeys(nullptr,
                                                          sort_workspace_bytes,
                                                          ids.data_handle(),
                                                          static_cast<std::uint32_t*>(nullptr),
                                                          item_count,
                                                          row_count,
                                                          indptr.data_handle(),
                                                          indptr.data_handle() + 1,
                                                          0,
                                                          sort_end_bit,
                                                          stream));
  }
  RAFT_CUDA_TRY(cub::DeviceSelect::Flagged(nullptr,
                                           select_workspace_bytes,
                                           counting,
                                           static_cast<std::uint8_t const*>(nullptr),
                                           static_cast<std::int64_t*>(nullptr),
                                           static_cast<std::int64_t*>(nullptr),
                                           item_count,
                                           stream));
  RAFT_CUDA_TRY(cub::DeviceScan::ExclusiveSum(nullptr,
                                              scan_workspace_bytes,
                                              static_cast<std::uint64_t*>(nullptr),
                                              static_cast<std::uint64_t*>(nullptr),
                                              static_cast<std::int64_t>(max_containers),
                                              stream));
  auto const workspace_bytes =
    std::max(sort_workspace_bytes, std::max(select_workspace_bytes, scan_workspace_bytes));
  batch_general_scratch_layout const layout{
    total_ids, rows, max_containers, !pre_sorted, workspace_bytes};
  rmm::device_uvector<cuda::std::byte> scratch(layout.bytes, stream);
  auto* sorted_ids =
    pre_sorted ? nullptr
               : reinterpret_cast<std::uint32_t*>(scratch.data() + layout.sorted_ids_offset);
  auto* valid_counts = reinterpret_cast<std::int64_t*>(scratch.data() + layout.valid_counts_offset);
  auto* flags        = reinterpret_cast<std::uint8_t*>(scratch.data() + layout.flags_offset);
  auto* selected_count =
    reinterpret_cast<std::int64_t*>(scratch.data() + layout.selected_count_offset);
  auto* container_starts =
    reinterpret_cast<std::int64_t*>(scratch.data() + layout.container_starts_offset);
  auto* row_container_offsets =
    reinterpret_cast<std::int64_t*>(scratch.data() + layout.row_container_offsets_offset);
  auto* container_rows =
    reinterpret_cast<std::uint32_t*>(scratch.data() + layout.container_rows_offset);
  auto* kinds = reinterpret_cast<container_kind*>(scratch.data() + layout.kinds_offset);
  auto* payload_sizes =
    reinterpret_cast<std::uint64_t*>(scratch.data() + layout.payload_sizes_offset);
  auto* payload_offsets =
    reinterpret_cast<std::uint64_t*>(scratch.data() + layout.payload_offsets_offset);
  auto* has_run = reinterpret_cast<std::uint32_t*>(scratch.data() + layout.has_run_offset);
  auto* summaries =
    reinterpret_cast<device_build_summary*>(scratch.data() + layout.summaries_offset);
  auto* output_offsets =
    reinterpret_cast<std::uint64_t*>(scratch.data() + layout.output_offsets_offset);
  auto* workspace = scratch.data() + layout.workspace_offset;

  if (!pre_sorted) {
    RAFT_CUDA_TRY(cub::DeviceSegmentedRadixSort::SortKeys(workspace,
                                                          sort_workspace_bytes,
                                                          ids.data_handle(),
                                                          sorted_ids,
                                                          item_count,
                                                          row_count,
                                                          indptr.data_handle(),
                                                          indptr.data_handle() + 1,
                                                          0,
                                                          sort_end_bit,
                                                          stream));
  }
  auto const* normalized = pre_sorted ? ids.data_handle() : sorted_ids;
  find_batch_valid_counts_kernel<<<grid_size_for(rows), kBuilderBlockSize, 0, stream>>>(
    normalized, indptr.data_handle(), row_count, dataset_rows, valid_counts);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  mark_batch_container_starts_kernel<<<static_cast<unsigned int>(rows),
                                       kBuilderBlockSize,
                                       0,
                                       stream>>>(
    normalized, indptr.data_handle(), valid_counts, row_count, flags);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  RAFT_CUDA_TRY(cub::DeviceSelect::Flagged(workspace,
                                           select_workspace_bytes,
                                           counting,
                                           flags,
                                           container_starts,
                                           selected_count,
                                           item_count,
                                           stream));
  find_row_container_offsets_kernel<<<grid_size_for(rows + 1), kBuilderBlockSize, 0, stream>>>(
    container_starts, selected_count, indptr.data_handle(), row_count, row_container_offsets);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  fill_container_rows_kernel<<<static_cast<unsigned int>(rows), kBuilderBlockSize, 0, stream>>>(
    row_container_offsets, row_count, container_rows);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  RAFT_CUDA_TRY(cudaMemsetAsync(payload_sizes, 0, max_containers * sizeof(std::uint64_t), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(has_run, 0, rows * sizeof(std::uint32_t), stream));
  analyze_batch_containers_kernel<<<static_cast<unsigned int>(max_containers),
                                    kBuilderBlockSize,
                                    0,
                                    stream>>>(normalized,
                                              indptr.data_handle(),
                                              valid_counts,
                                              container_starts,
                                              selected_count,
                                              container_rows,
                                              kinds,
                                              payload_sizes,
                                              has_run);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  RAFT_CUDA_TRY(cub::DeviceScan::ExclusiveSum(workspace,
                                              scan_workspace_bytes,
                                              payload_sizes,
                                              payload_offsets,
                                              static_cast<std::int64_t>(max_containers),
                                              stream));
  finish_batch_rows_kernel<<<grid_size_for(rows), kBuilderBlockSize, 0, stream>>>(
    row_count,
    indptr.data_handle(),
    valid_counts,
    row_container_offsets,
    has_run,
    payload_sizes,
    payload_offsets,
    summaries);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  std::vector<device_build_summary> host_summaries(rows);
  RAFT_CUDA_TRY(cudaMemcpyAsync(host_summaries.data(),
                                summaries,
                                rows * sizeof(device_build_summary),
                                cudaMemcpyDeviceToHost,
                                stream));
  raft::resource::sync_stream(res);
  std::vector<std::size_t> serialized(rows);
  std::vector<std::size_t> cardinalities(rows);
  std::size_t actual_containers{};
  for (std::size_t row = 0; row < rows; ++row) {
    auto const& summary    = host_summaries[row];
    auto const cardinality = static_cast<std::size_t>(host_indptr[row + 1] - host_indptr[row]);
    RAFT_EXPECTS(summary.invalid == 0, "Roaring allowlist ID must be smaller than dataset_rows.");
    RAFT_EXPECTS(summary.cardinality == static_cast<std::int64_t>(cardinality) &&
                   (cardinality == 0 || summary.num_containers > 0),
                 "Internal error: general batched row analysis failed.");
    cardinalities[row] = cardinality;
    if (cardinality == 0) { continue; }
    auto const header = portable_header_size(summary.num_containers, summary.has_run != 0);
    RAFT_EXPECTS(summary.payload_bytes <= std::numeric_limits<std::uint32_t>::max() - header,
                 "Portable Roaring row exceeds the 32-bit offset range.");
    serialized[row] = header + static_cast<std::size_t>(summary.payload_bytes);
    actual_containers += summary.num_containers;
  }
  auto packed = make_packed_rows_layout(serialized);
  rmm::device_uvector<cuda::std::byte> storage(packed.bytes, stream);
  static_assert(sizeof(std::size_t) == sizeof(std::uint64_t));
  RAFT_CUDA_TRY(cudaMemcpyAsync(output_offsets,
                                packed.row_offsets.data(),
                                rows * sizeof(std::uint64_t),
                                cudaMemcpyHostToDevice,
                                stream));
  encode_batch_headers_kernel<<<static_cast<unsigned int>(rows), kBuilderBlockSize, 0, stream>>>(
    normalized,
    indptr.data_handle(),
    valid_counts,
    container_starts,
    row_container_offsets,
    kinds,
    payload_offsets,
    summaries,
    output_offsets,
    storage.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  encode_batch_payloads_kernel<<<static_cast<unsigned int>(actual_containers),
                                 kBuilderBlockSize,
                                 0,
                                 stream>>>(normalized,
                                           indptr.data_handle(),
                                           valid_counts,
                                           container_starts,
                                           row_container_offsets,
                                           container_rows,
                                           static_cast<std::int64_t>(actual_containers),
                                           kinds,
                                           payload_sizes,
                                           payload_offsets,
                                           summaries,
                                           output_offsets,
                                           storage.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  initialize_batch_refs_kernel<<<grid_size_for(rows), kBuilderBlockSize, 0, stream>>>(
    storage.data(),
    output_offsets,
    summaries,
    reinterpret_cast<ref_type*>(storage.data() + packed.references_offset),
    rows);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  return {std::move(storage),
          std::move(packed.row_offsets),
          std::move(packed.reference_offsets),
          std::move(serialized),
          std::move(cardinalities),
          true};
}

void validate_indptr(std::vector<std::int64_t> const& indptr, std::size_t rows, std::size_t nnz)
{
  RAFT_EXPECTS(rows > 0, "Roaring input must contain at least one allowlist row.");
  RAFT_EXPECTS(indptr.size() == rows + 1, "Roaring indptr must contain num_rows + 1 entries.");
  RAFT_EXPECTS(indptr.front() == 0, "Roaring indptr must start at zero.");
  for (std::size_t row = 0; row < rows; ++row) {
    RAFT_EXPECTS(indptr[row] <= indptr[row + 1] && indptr[row] >= 0,
                 "Roaring indptr must be nonnegative and nondecreasing.");
  }
  RAFT_EXPECTS(indptr.back() >= 0 && static_cast<std::size_t>(indptr.back()) == nnz,
               "The final Roaring indptr entry must equal nnz.");
}

batched_device_build_result build_batched_from_device_ids(
  raft::resources const& res,
  std::size_t dataset_rows,
  raft::device_vector_view<const std::uint32_t, std::int64_t> ids,
  raft::device_vector_view<const std::int64_t, std::int64_t> indptr,
  std::vector<std::int64_t> const& host_indptr,
  bool pre_sorted)
{
  auto const stream = raft::resource::get_cuda_stream(res);
  auto const rows   = host_indptr.size() - 1;
  auto const size   = static_cast<std::size_t>(ids.extent(0));
  validate_indptr(host_indptr, rows, size);
  if (size == 0) {
    return {rmm::device_uvector<cuda::std::byte>(0, stream),
            std::vector<std::size_t>(rows),
            std::vector<std::size_t>(rows),
            std::vector<std::size_t>(rows),
            std::vector<std::size_t>(rows),
            true};
  }
  if (rows == 1) {
    auto row_view = raft::make_device_vector_view<const std::uint32_t, std::int64_t>(
      ids.data_handle(), static_cast<std::int64_t>(size));
    auto built = build_from_device_ids(res, dataset_rows, row_view, pre_sorted);
    return {std::move(built.storage),
            {0},
            {built.cardinality == 0 ? 0 : reference_offset(built.serialized_bytes)},
            {built.serialized_bytes},
            {built.cardinality},
            built.reference_initialized};
  }
  return build_general_rows(res, dataset_rows, ids, indptr, host_indptr, pre_sorted);
}

/**
 * Validate one externally supplied portable row before cuco sees it.
 *
 * cuco's raw-byte reference constructor assumes a valid stream and is not given
 * the row's byte length. We therefore walk the complete row on the host first,
 * checking every read, the required header variant, ordered keys and values,
 * payload cardinalities, exact container offsets, the logical dataset bound,
 * and that no trailing bytes remain. Besides rejecting malformed input, this
 * walk records cardinality without retaining another decoded representation.
 *
 * A zero-length input is a cuVS convenience for an empty allowlist. The
 * standard serialized empty form (no-run cookie followed by N = 0) is accepted
 * as well.
 */
row_metadata validate_serialized_row(std::byte const* data,
                                     std::size_t size,
                                     std::size_t dataset_rows)
{
  // Empty here means the outer cuVS byte offsets selected no portable bytes for
  // this row.
  if (size == 0) { return {}; }

  // Decode the cookie first because it determines both how N is stored and
  // whether a run bitmap follows. Every subsequent read advances `cursor`
  // through exactly one format field.
  auto const cookie  = read_u32(data, size, 0);
  bool const has_run = (cookie & 0xffffu) == kCookieRun;
  std::size_t num_containers{};
  std::size_t cursor = 4;
  if (has_run) {
    num_containers = (cookie >> 16) + 1;
  } else {
    RAFT_EXPECTS(cookie == kCookieNoRun, "Malformed portable Roaring bitmap: unsupported cookie.");
    num_containers = read_u32(data, size, cursor);
    cursor += 4;
  }
  RAFT_EXPECTS(num_containers <= (std::size_t{1} << 16),
               "Malformed portable Roaring bitmap: too many containers.");
  if (num_containers == 0) {
    RAFT_EXPECTS(!has_run && cursor == size,
                 "Malformed portable Roaring bitmap: invalid empty representation.");
    return {};
  }

  std::size_t run_bitmap_offset{};
  if (has_run) {
    auto const run_bitmap_bytes = (num_containers + 7) / 8;
    RAFT_EXPECTS(cursor <= size && size - cursor >= run_bitmap_bytes,
                 "Malformed portable Roaring bitmap: truncated run bitmap.");
    run_bitmap_offset = cursor;
    cursor += run_bitmap_bytes;
  }

  // The descriptive header is common to both cookie forms. Reconstruct
  // cardinality by adding one to its encoded value and require container keys
  // to be strictly increasing.
  std::vector<std::uint16_t> keys(num_containers);
  std::vector<std::uint32_t> cards(num_containers);
  for (std::size_t i = 0; i < num_containers; ++i) {
    keys[i]  = read_u16(data, size, cursor);
    cards[i] = static_cast<std::uint32_t>(read_u16(data, size, cursor + 2)) + 1;
    cursor += 4;
    if (i > 0) {
      RAFT_EXPECTS(keys[i - 1] < keys[i],
                   "Malformed portable Roaring bitmap: container keys are not ordered.");
    }
  }

  // These are offsets INSIDE this portable row. The no-run form always stores
  // them; the run form stores them only at the specification's four-container
  // threshold.
  auto const store_offsets = !has_run || num_containers >= kOffsetThreshold;
  std::vector<std::uint32_t> container_offsets;
  if (store_offsets) {
    container_offsets.resize(num_containers);
    for (std::size_t i = 0; i < num_containers; ++i) {
      container_offsets[i] = read_u32(data, size, cursor);
      cursor += 4;
    }
  }

  // Validate payloads in descriptor order. Requiring every stored offset to
  // equal `cursor` also rejects gaps, overlaps, and offsets that point into a
  // header or a different container.
  row_metadata metadata;
  metadata.empty = false;
  for (std::size_t i = 0; i < num_containers; ++i) {
    if (store_offsets) {
      RAFT_EXPECTS(container_offsets[i] == cursor,
                   "Malformed portable Roaring bitmap: invalid container offset.");
    }
    auto const is_run =
      has_run && ((std::to_integer<std::uint8_t>(data[run_bitmap_offset + i / 8]) >> (i % 8)) & 1u);
    std::uint32_t lower_max{};
    if (is_run) {
      auto const num_runs = read_u16(data, size, cursor);
      cursor += 2;
      std::uint32_t run_cardinality{};
      std::uint32_t previous_end{};
      for (std::size_t run_index = 0; run_index < num_runs; ++run_index) {
        auto const start  = static_cast<std::uint32_t>(read_u16(data, size, cursor));
        auto const length = static_cast<std::uint32_t>(read_u16(data, size, cursor + 2));
        cursor += 4;
        auto const end = start + length;
        RAFT_EXPECTS(end <= std::numeric_limits<std::uint16_t>::max(),
                     "Malformed portable Roaring bitmap: run exceeds uint16 range.");
        if (run_index > 0) {
          RAFT_EXPECTS(start > previous_end,
                       "Malformed portable Roaring bitmap: runs overlap or are "
                       "unordered.");
        }
        previous_end = end;
        lower_max    = end;
        run_cardinality += length + 1;
      }
      RAFT_EXPECTS(num_runs > 0 && run_cardinality == cards[i],
                   "Malformed portable Roaring bitmap: invalid run cardinality.");
    } else if (cards[i] <= kArrayCardinality) {
      std::uint32_t previous{};
      for (std::size_t j = 0; j < cards[i]; ++j) {
        auto const value = static_cast<std::uint32_t>(read_u16(data, size, cursor));
        cursor += 2;
        if (j > 0) {
          RAFT_EXPECTS(previous < value,
                       "Malformed portable Roaring bitmap: "
                       "array values are not ordered.");
        }
        previous  = value;
        lower_max = value;
      }
    } else {
      RAFT_EXPECTS(cursor <= size && size - cursor >= kBitmapBytes,
                   "Malformed portable Roaring bitmap: truncated bitmap container.");
      std::size_t popcount{};
      bool found_max = false;
      for (std::size_t j = 0; j < kBitmapBytes; ++j) {
        popcount += std::popcount(std::to_integer<std::uint8_t>(data[cursor + j]));
      }
      for (std::size_t j = kBitmapBytes; j-- > 0 && !found_max;) {
        auto const byte = std::to_integer<std::uint8_t>(data[cursor + j]);
        if (byte != 0) {
          lower_max = static_cast<std::uint32_t>(j * 8 + (7 - std::countl_zero(byte)));
          found_max = true;
        }
      }
      RAFT_EXPECTS(found_max && popcount == cards[i],
                   "Malformed portable Roaring bitmap: invalid bitmap cardinality.");
      cursor += kBitmapBytes;
    }

    auto const max_id = (static_cast<std::uint32_t>(keys[i]) << 16) | lower_max;
    RAFT_EXPECTS(static_cast<std::uint64_t>(max_id) < dataset_rows,
                 "Portable Roaring bitmap contains an ID outside dataset_rows.");
    metadata.cardinality += cards[i];
    metadata.max_id = max_id;
  }
  RAFT_EXPECTS(cursor == size, "Malformed portable Roaring bitmap: trailing or unconsumed bytes.");
  return metadata;
}

/**
 * Construct the lightweight cuco reference once, outside the search path.
 *
 * The raw-byte constructor parses the portable header and stores small
 * container-location metadata by value while retaining pointers into `data`. It
 * neither allocates nor copies the serialized payload. Empty allowlists skip
 * this kernel because the pinned parser expects at least one container.
 */
__global__ void initialize_imported_refs_kernel(cuda::std::byte const* storage,
                                                std::uint64_t const* row_offsets,
                                                std::uint64_t const* serialized_bytes,
                                                ref_type* references,
                                                std::size_t rows)
{
  auto const row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < rows && serialized_bytes[row] != 0) {
    ::new (static_cast<void*>(references + row)) ref_type{storage + row_offsets[row]};
  }
}

__global__ void initialize_view_tables_kernel(cuda::std::byte const* storage,
                                              std::uint64_t const* reference_offsets,
                                              std::uint64_t const* serialized_bytes,
                                              ref_type const** references,
                                              std::uint8_t* empty_rows,
                                              std::size_t rows)
{
  auto const row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) { return; }
  auto const empty = serialized_bytes[row] == 0;
  references[row] =
    empty ? nullptr : reinterpret_cast<ref_type const*>(storage + reference_offsets[row]);
  empty_rows[row] = empty ? 1 : 0;
}

__global__ void contains_kernel(ref_type const* const* references,
                                std::uint8_t const* empty_rows,
                                std::uint64_t dataset_rows,
                                std::uint32_t const* row_ids,
                                std::uint8_t* output,
                                std::size_t columns,
                                std::size_t size)
{
  auto const i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size) { return; }
  auto const query = i / columns;
  auto const row   = row_ids[i];
  output[i]        = empty_rows[query] == 0 && static_cast<std::uint64_t>(row) < dataset_rows &&
              references[query]->contains(row);
}

}  // namespace

struct roaring_allowlist::impl {
  rmm::device_uvector<cuda::std::byte> storage;
  std::vector<std::size_t> row_offsets_;
  std::vector<std::size_t> reference_offsets_;
  std::vector<std::size_t> serialized_bytes_;
  std::vector<std::size_t> cardinalities_;
  rmm::device_uvector<ref_type const*> references;
  rmm::device_uvector<std::uint8_t> empty_rows;
  std::size_t dataset_rows_{};
  std::size_t total_cardinality_{};
  bool references_initialized_{};

  impl(raft::resources const& res, std::size_t dataset_rows, batched_device_build_result&& built)
    : storage(std::move(built.storage)),
      row_offsets_(std::move(built.row_offsets)),
      reference_offsets_(std::move(built.reference_offsets)),
      serialized_bytes_(std::move(built.serialized_bytes)),
      cardinalities_(std::move(built.cardinalities)),
      references(cardinalities_.size(), raft::resource::get_cuda_stream(res)),
      empty_rows(cardinalities_.size(), raft::resource::get_cuda_stream(res)),
      dataset_rows_(dataset_rows),
      total_cardinality_(
        std::accumulate(cardinalities_.begin(), cardinalities_.end(), std::size_t{})),
      references_initialized_(built.references_initialized)
  {
    static_assert(std::is_trivially_destructible_v<ref_type>);
    auto const rows = cardinalities_.size();
    RAFT_EXPECTS(rows > 0 && row_offsets_.size() == rows && reference_offsets_.size() == rows &&
                   serialized_bytes_.size() == rows,
                 "Internal error: inconsistent batched Roaring metadata.");
    if (rows == 0) { return; }

    auto const stream = raft::resource::get_cuda_stream(res);
    static_assert(sizeof(std::size_t) == sizeof(std::uint64_t));
    rmm::device_uvector<std::uint64_t> device_row_offsets(rows, stream);
    rmm::device_uvector<std::uint64_t> device_reference_offsets(rows, stream);
    rmm::device_uvector<std::uint64_t> device_serialized_bytes(rows, stream);
    RAFT_CUDA_TRY(cudaMemcpyAsync(device_row_offsets.data(),
                                  row_offsets_.data(),
                                  rows * sizeof(std::uint64_t),
                                  cudaMemcpyHostToDevice,
                                  stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(device_reference_offsets.data(),
                                  reference_offsets_.data(),
                                  rows * sizeof(std::uint64_t),
                                  cudaMemcpyHostToDevice,
                                  stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(device_serialized_bytes.data(),
                                  serialized_bytes_.data(),
                                  rows * sizeof(std::uint64_t),
                                  cudaMemcpyHostToDevice,
                                  stream));
    if (!references_initialized_ && storage.size() != 0) {
      initialize_imported_refs_kernel<<<grid_size_for(rows), kBuilderBlockSize, 0, stream>>>(
        storage.data(),
        device_row_offsets.data(),
        device_serialized_bytes.data(),
        reinterpret_cast<ref_type*>(storage.data() + reference_offsets_.front()),
        rows);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
    initialize_view_tables_kernel<<<grid_size_for(rows), kBuilderBlockSize, 0, stream>>>(
      storage.data(),
      device_reference_offsets.data(),
      device_serialized_bytes.data(),
      references.data(),
      empty_rows.data(),
      rows);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  [[nodiscard]] ref_type const* reference(std::size_t allowlist_id) const noexcept
  {
    return cardinalities_[allowlist_id] == 0
             ? nullptr
             : reinterpret_cast<ref_type const*>(storage.data() + reference_offsets_[allowlist_id]);
  }
};

roaring_allowlist::roaring_allowlist(std::unique_ptr<impl> impl) noexcept : impl_(std::move(impl))
{
}

roaring_allowlist roaring_allowlist::from_ids(
  raft::resources const& res,
  std::size_t dataset_rows,
  raft::host_vector_view<const key_type, std::int64_t> ids,
  raft::host_vector_view<const indptr_type, std::int64_t> indptr,
  bool pre_sorted)
{
  validate_dataset_rows(dataset_rows);
  RAFT_EXPECTS(indptr.extent(0) >= 2, "Roaring indptr must contain at least two entries.");
  auto const rows = static_cast<std::size_t>(indptr.extent(0) - 1);
  auto const size = static_cast<std::size_t>(ids.extent(0));
  std::vector<indptr_type> host_indptr(indptr.data_handle(),
                                       indptr.data_handle() + indptr.extent(0));
  validate_indptr(host_indptr, rows, size);

  auto const stream = raft::resource::get_cuda_stream(res);
  rmm::device_uvector<key_type> device_ids(size, stream);
  rmm::device_uvector<indptr_type> device_indptr(rows + 1, stream);
  if (size != 0) {
    RAFT_CUDA_TRY(cudaMemcpyAsync(device_ids.data(),
                                  ids.data_handle(),
                                  size * sizeof(key_type),
                                  cudaMemcpyHostToDevice,
                                  stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(device_indptr.data(),
                                  host_indptr.data(),
                                  (rows + 1) * sizeof(indptr_type),
                                  cudaMemcpyHostToDevice,
                                  stream));
  }
  auto device_ids_view = raft::make_device_vector_view<const key_type, std::int64_t>(
    device_ids.data(), static_cast<std::int64_t>(size));
  auto device_indptr_view = raft::make_device_vector_view<const indptr_type, std::int64_t>(
    device_indptr.data(), static_cast<std::int64_t>(rows + 1));
  auto built = build_batched_from_device_ids(
    res, dataset_rows, device_ids_view, device_indptr_view, host_indptr, pre_sorted);
  return roaring_allowlist{std::make_unique<impl>(res, dataset_rows, std::move(built))};
}

roaring_allowlist roaring_allowlist::from_ids(
  raft::resources const& res,
  std::size_t dataset_rows,
  raft::device_vector_view<const key_type, std::int64_t> ids,
  raft::device_vector_view<const indptr_type, std::int64_t> indptr,
  bool pre_sorted)
{
  validate_dataset_rows(dataset_rows);
  RAFT_EXPECTS(indptr.extent(0) >= 2, "Roaring indptr must contain at least two entries.");
  auto const rows   = static_cast<std::size_t>(indptr.extent(0) - 1);
  auto const size   = static_cast<std::size_t>(ids.extent(0));
  auto const stream = raft::resource::get_cuda_stream(res);
  std::vector<indptr_type> host_indptr(rows + 1);
  RAFT_CUDA_TRY(cudaMemcpyAsync(host_indptr.data(),
                                indptr.data_handle(),
                                (rows + 1) * sizeof(indptr_type),
                                cudaMemcpyDeviceToHost,
                                stream));
  raft::resource::sync_stream(res);
  validate_indptr(host_indptr, rows, size);
  auto built =
    build_batched_from_device_ids(res, dataset_rows, ids, indptr, host_indptr, pre_sorted);
  return roaring_allowlist{std::make_unique<impl>(res, dataset_rows, std::move(built))};
}

roaring_allowlist roaring_allowlist::from_serialized(
  raft::resources const& res,
  std::size_t dataset_rows,
  raft::host_vector_view<const std::byte, std::int64_t> bytes,
  raft::host_vector_view<const std::uint64_t, std::int64_t> byte_offsets)
{
  validate_dataset_rows(dataset_rows);
  RAFT_EXPECTS(byte_offsets.extent(0) >= 2,
               "Roaring byte_offsets must contain at least two entries.");
  auto const rows = static_cast<std::size_t>(byte_offsets.extent(0) - 1);
  auto const size = static_cast<std::size_t>(bytes.extent(0));
  RAFT_EXPECTS(byte_offsets(0) == 0, "Roaring byte_offsets must start at zero.");
  RAFT_EXPECTS(byte_offsets(static_cast<std::int64_t>(rows)) == size,
               "The final Roaring byte offset must equal bytes.extent(0).");

  std::vector<std::size_t> serialized_bytes(rows);
  std::vector<std::size_t> cardinalities(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    auto const begin = byte_offsets(static_cast<std::int64_t>(row));
    auto const end   = byte_offsets(static_cast<std::int64_t>(row + 1));
    RAFT_EXPECTS(begin <= end && end <= size,
                 "Roaring byte_offsets must be nondecreasing and in bounds.");
    auto const row_size = static_cast<std::size_t>(end - begin);
    auto const metadata =
      validate_serialized_row(bytes.data_handle() + begin, row_size, dataset_rows);
    serialized_bytes[row] = metadata.empty ? 0 : row_size;
    cardinalities[row]    = metadata.cardinality;
  }

  auto const packed = make_packed_rows_layout(serialized_bytes);
  auto const stream = raft::resource::get_cuda_stream(res);
  rmm::device_uvector<cuda::std::byte> storage(packed.bytes, stream);
  for (std::size_t row = 0; row < rows; ++row) {
    if (serialized_bytes[row] == 0) { continue; }
    auto const begin = byte_offsets(static_cast<std::int64_t>(row));
    RAFT_CUDA_TRY(cudaMemcpyAsync(storage.data() + packed.row_offsets[row],
                                  bytes.data_handle() + begin,
                                  serialized_bytes[row],
                                  cudaMemcpyHostToDevice,
                                  stream));
  }

  batched_device_build_result built{std::move(storage),
                                    packed.row_offsets,
                                    packed.reference_offsets,
                                    std::move(serialized_bytes),
                                    std::move(cardinalities),
                                    false};
  return roaring_allowlist{std::make_unique<impl>(res, dataset_rows, std::move(built))};
}

roaring_allowlist::~roaring_allowlist()                                       = default;
roaring_allowlist::roaring_allowlist(roaring_allowlist&&) noexcept            = default;
roaring_allowlist& roaring_allowlist::operator=(roaring_allowlist&&) noexcept = default;

std::size_t roaring_allowlist::num_allowlists() const noexcept
{
  return impl_->cardinalities_.size();
}

std::size_t roaring_allowlist::dataset_rows() const noexcept { return impl_->dataset_rows_; }

std::size_t roaring_allowlist::cardinality(std::size_t allowlist_id) const
{
  RAFT_EXPECTS(allowlist_id < num_allowlists(), "Roaring allowlist_id is out of range.");
  return impl_->cardinalities_[allowlist_id];
}

bool roaring_allowlist::empty(std::size_t allowlist_id) const
{
  return cardinality(allowlist_id) == 0;
}

std::size_t roaring_allowlist::total_cardinality() const noexcept
{
  return impl_->total_cardinality_;
}

std::size_t roaring_allowlist::size_bytes() const noexcept
{
  return impl_->storage.size() * sizeof(cuda::std::byte) +
         impl_->references.size() * sizeof(ref_type const*) +
         impl_->empty_rows.size() * sizeof(std::uint8_t);
}

roaring_allowlist_view roaring_allowlist::view(std::size_t allowlist_id) const
{
  RAFT_EXPECTS(allowlist_id < num_allowlists(), "Roaring allowlist_id is out of range.");
  return roaring_allowlist_view{
    impl_->reference(allowlist_id), dataset_rows(), cardinality(allowlist_id)};
}

void roaring_allowlist::contains(
  raft::resources const& res,
  raft::device_matrix_view<const key_type, std::int64_t, raft::row_major> row_ids,
  raft::device_matrix_view<std::uint8_t, std::int64_t, raft::row_major> output) const
{
  contains_async(res, row_ids, output);
  raft::resource::sync_stream(res);
}

void roaring_allowlist::contains_async(
  raft::resources const& res,
  raft::device_matrix_view<const key_type, std::int64_t, raft::row_major> row_ids,
  raft::device_matrix_view<std::uint8_t, std::int64_t, raft::row_major> output) const
{
  RAFT_EXPECTS(row_ids.extent(0) == static_cast<std::int64_t>(num_allowlists()),
               "Roaring membership matrix must have one row per allowlist.");
  RAFT_EXPECTS(output.extent(0) == row_ids.extent(0) && output.extent(1) == row_ids.extent(1),
               "Roaring membership output shape must match the input shape.");
  auto const rows    = static_cast<std::size_t>(row_ids.extent(0));
  auto const columns = static_cast<std::size_t>(row_ids.extent(1));
  if (columns == 0) { return; }
  RAFT_EXPECTS(rows <= std::numeric_limits<std::size_t>::max() / columns,
               "Roaring membership matrix is too large.");
  auto const size          = rows * columns;
  constexpr int block_size = 256;
  contains_kernel<<<grid_size_for(size), block_size, 0, raft::resource::get_cuda_stream(res)>>>(
    impl_->references.data(),
    impl_->empty_rows.data(),
    static_cast<std::uint64_t>(dataset_rows()),
    row_ids.data_handle(),
    output.data_handle(),
    columns,
    size);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace cuvs::core
