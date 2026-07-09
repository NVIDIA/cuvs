/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * reserved. SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../sample_filter.cuh"  // public filter types
#include "../sample_filter_data.cuh"
#include "jit_lto_kernels/cagra_filter_payload.cuh"

#include <cuvs/core/bloom_filter.hpp>
#include <raft/core/error.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <list>
#include <mutex>
#include <type_traits>
#include <unordered_map>

namespace cuvs::neighbors::cagra::detail {

template <typename PayloadT>
std::uint64_t cagra_payload_hash(PayloadT const& payload)
{
  static_assert(std::is_trivially_copyable_v<PayloadT>);
  constexpr std::uint64_t kOffset = 1469598103934665603ull;
  constexpr std::uint64_t kPrime  = 1099511628211ull;
  auto const* bytes               = reinterpret_cast<unsigned char const*>(&payload);
  std::uint64_t hash              = kOffset;
  for (std::size_t i = 0; i < sizeof(PayloadT); ++i) {
    hash ^= bytes[i];
    hash *= kPrime;
  }
  return hash;
}

template <typename PayloadT>
struct cagra_device_payload_owner {
  struct state {
    PayloadT host_payload{};
    PayloadT* device_payload{nullptr};
    cudaStream_t stream{};
    cudaEvent_t ready_event{};
    int device{-1};
    std::mutex mutex;

    explicit state(PayloadT payload) : host_payload(payload) {}

    ~state() noexcept
    {
      if (device_payload != nullptr) {
        RAFT_CUDA_TRY_NO_THROW(cudaFreeAsync(device_payload, stream));
      }
      if (ready_event != nullptr) { RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(ready_event)); }
    }

    PayloadT* dev_ptr(cudaStream_t cuda_stream)
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (device_payload == nullptr) {
        RAFT_CUDA_TRY(cudaGetDevice(&device));
        RAFT_CUDA_TRY(cudaMallocAsync(
          reinterpret_cast<void**>(&device_payload), sizeof(PayloadT), cuda_stream));
        RAFT_CUDA_TRY(cudaMemcpyAsync(
          device_payload, &host_payload, sizeof(PayloadT), cudaMemcpyHostToDevice, cuda_stream));
        RAFT_CUDA_TRY(cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming));
        RAFT_CUDA_TRY(cudaEventRecord(ready_event, cuda_stream));
        stream = cuda_stream;
      } else {
        RAFT_CUDA_TRY(cudaStreamWaitEvent(cuda_stream, ready_event, 0));
      }
      return device_payload;
    }
  };

  // PayloadT is copied to device by value. Pointer fields inside PayloadT are shallow-copied and
  // must already point to device-addressable memory that remains valid while the cached payload is
  // usable.
  struct cache_key {
    std::uint64_t payload_hash{};
    int device{};

    bool operator==(cache_key const& other) const
    {
      return payload_hash == other.payload_hash && device == other.device;
    }
  };

  struct cache_key_hash {
    std::size_t operator()(cache_key const& key) const
    {
      auto seed = static_cast<std::size_t>(key.payload_hash);
      seed ^= static_cast<std::size_t>(key.device) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      return seed;
    }
  };

  cagra_device_payload_owner() = default;

  void* dev_ptr(PayloadT payload, cudaStream_t stream) const
  {
    int device{};
    RAFT_CUDA_TRY(cudaGetDevice(&device));

    // Keep cached payload copies for process lifetime to avoid per-search allocation/copy churn.
    // Cross-stream reuse is ordered by each state's ready_event before kernels consume the pointer.
    const auto key = cache_key{cagra_payload_hash(payload), device};
    state* selected_state{};
    {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      auto& entries = cache_[key];
      for (auto& cached : entries) {
        if (std::memcmp(&cached.host_payload, &payload, sizeof(PayloadT)) == 0) {
          selected_state = &cached;
          break;
        }
      }
      if (selected_state == nullptr) {
        entries.emplace_back(payload);
        selected_state = &entries.back();
      }
    }

    return selected_state->dev_ptr(stream);
  }

 private:
  mutable std::mutex cache_mutex_;
  mutable std::unordered_map<cache_key, std::list<state>, cache_key_hash> cache_;
};

template <typename T>
struct is_bitset_filter : std::false_type {};

template <typename bitset_t, typename index_t>
struct is_bitset_filter<::cuvs::neighbors::filtering::bitset_filter<bitset_t, index_t>>
  : std::true_type {};

template <typename T>
struct is_bloom_filter : std::false_type {};

template <>
struct is_bloom_filter<::cuvs::neighbors::filtering::bloom_filter> : std::true_type {};

template <typename T>
struct is_udf_filter : std::false_type {};

template <>
struct is_udf_filter<::cuvs::neighbors::filtering::udf_filter> : std::true_type {};

template <typename SourceIndexT, typename FilterT>
::cuvs::neighbors::detail::bitset_filter_data_t<SourceIndexT> make_cagra_bitset_filter_storage(
  const FilterT& filter)
{
  const auto bitset_view = filter.view();
  return ::cuvs::neighbors::detail::bitset_filter_data_t<SourceIndexT>{
    const_cast<std::uint32_t*>(bitset_view.data()),
    static_cast<SourceIndexT>(bitset_view.size()),
    static_cast<SourceIndexT>(bitset_view.get_original_nbits())};
}

template <typename FilterT>
::cuvs::neighbors::detail::bloom_filter_data_t<std::uint32_t> make_cagra_bloom_filter_storage(
  const FilterT& filter)
{
  using payload_t = ::cuvs::neighbors::detail::bloom_filter_data_t<std::uint32_t>;
  payload_t payload{};
  RAFT_EXPECTS(filter.bloom_filter_ptr != nullptr,
               "bloom_filter requires a cuvs::core::bloom_filter object or prebuilt filter_data.");
  filter.bloom_filter_ptr->export_payload(&payload, sizeof(payload));
  return payload;
}

template <typename PayloadT>
void* get_cagra_device_payload(PayloadT payload, cudaStream_t stream)
{
  static cagra_device_payload_owner<PayloadT> owner;
  return owner.dev_ptr(payload, stream);
}

template <typename SourceIndexT, typename FilterT>
void* make_cagra_bitset_filter_payload(const FilterT& filter, cudaStream_t stream)
{
  return get_cagra_device_payload(make_cagra_bitset_filter_storage<SourceIndexT>(filter), stream);
}

template <typename SourceIndexT, typename FilterT>
void fill_cagra_sample_filter(cagra_sample_filter<SourceIndexT>& out,
                              const FilterT& filter,
                              cudaStream_t stream)
{
  using DecayedFilter = std::decay_t<FilterT>;
  if constexpr (is_bitset_filter<DecayedFilter>::value) {
    out.filter_data = make_cagra_bitset_filter_payload<SourceIndexT>(filter, stream);
  } else if constexpr (is_bloom_filter<DecayedFilter>::value) {
    if (filter.filter_data != nullptr) {
      out.filter_data = filter.filter_data;
    } else {
      out.filter_data = get_cagra_device_payload(make_cagra_bloom_filter_storage(filter), stream);
    }
  } else if constexpr (is_udf_filter<DecayedFilter>::value) {
    out.filter_data = filter.filter_data;
  }
}

template <typename SourceIndexT, typename FilterT>
std::uint64_t cagra_filter_payload_hash(const FilterT& filter)
{
  using DecayedFilter = std::decay_t<FilterT>;
  if constexpr (is_bitset_filter<DecayedFilter>::value) {
    return cagra_payload_hash(make_cagra_bitset_filter_storage<SourceIndexT>(filter));
  } else if constexpr (is_bloom_filter<DecayedFilter>::value) {
    if (filter.filter_data != nullptr) { return 0; }
    return cagra_payload_hash(make_cagra_bloom_filter_storage(filter));
  } else if constexpr (requires { filter.filter; }) {
    return cagra_filter_payload_hash<SourceIndexT>(filter.filter);
  } else {
    return 0;
  }
}

template <typename FilterT>
void* cagra_filter_data_ptr(const FilterT& filter)
{
  using DecayedFilter = std::decay_t<FilterT>;
  if constexpr (is_bloom_filter<DecayedFilter>::value || is_udf_filter<DecayedFilter>::value) {
    return filter.filter_data;
  } else if constexpr (requires { filter.filter; }) {
    return cagra_filter_data_ptr(filter.filter);
  } else {
    return nullptr;
  }
}

template <typename SampleFilterT>
std::uint32_t cagra_filter_query_id_offset(const SampleFilterT& sample_filter)
{
  if constexpr (requires {
                  sample_filter.filter;
                  sample_filter.offset;
                }) {
    return sample_filter.offset;
  } else {
    return 0;
  }
}

/// Host: fill @ref cagra_sample_filter from a CAGRA filter object.
template <typename SourceIndexT, typename SampleFilterT>
cagra_sample_filter<SourceIndexT> extract_cagra_sample_filter(const SampleFilterT& sample_filter,
                                                              cudaStream_t stream)
{
  cagra_sample_filter<SourceIndexT> out;
  if constexpr (requires {
                  sample_filter.filter;
                  sample_filter.offset;
                }) {
    out.query_id_offset = sample_filter.offset;
    fill_cagra_sample_filter(out, sample_filter.filter, stream);
  } else {
    fill_cagra_sample_filter(out, sample_filter, stream);
  }
  return out;
}

/// Host: find UDF compile/link metadata only. Query offsets stay in the runtime payload produced
/// by @ref extract_cagra_sample_filter and are applied before calling the linked sample_filter.
template <typename SampleFilterT>
const ::cuvs::neighbors::filtering::udf_filter* get_cagra_udf_filter(
  const SampleFilterT& sample_filter)
{
  using DecayedFilter = std::decay_t<SampleFilterT>;
  if constexpr (is_udf_filter<DecayedFilter>::value) {
    return &sample_filter;
  } else if constexpr (requires { sample_filter.filter; }) {
    return get_cagra_udf_filter(sample_filter.filter);
  } else {
    return nullptr;
  }
}

}  // namespace cuvs::neighbors::cagra::detail
