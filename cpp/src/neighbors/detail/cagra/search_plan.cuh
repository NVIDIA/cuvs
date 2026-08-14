/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "hashmap.hpp"

#include <cuvs/neighbors/common.hpp>
#include <neighbors/detail/cagra/compute_distance-ext.cuh>
#include <raft/core/resource/cuda_stream.hpp>
// #include "topk_for_cagra/topk.h"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/distance/distance.hpp>

#include <cuvs/neighbors/cagra.hpp>
#include <raft/util/pow2_utils.cuh>

#include <optional>
#include <tuple>
#include <variant>

namespace cuvs::neighbors::cagra::detail {

/**
 * A lightweight version of rmm::device_uvector.
 * This version avoids calling cudaSetDevice / cudaGetDevice, and therefore it is required that
 * the current cuda device does not change during the lifetime of this object. This is expected
 * to be useful in multi-threaded scenarios where we want to minimize overhead due to
 * thread sincronization during cuda API calls.
 * If the size stays at zero, this struct never calls any CUDA driver / RAFT resource functions.
 */
template <typename T>
struct lightweight_uvector {
 private:
  using raft_res_type = const raft::resources*;
  using rmm_res_type  = std::tuple<rmm::device_async_resource_ref, rmm::cuda_stream_view>;
  static constexpr size_t kAlign = 256;

  std::variant<raft_res_type, rmm_res_type> res_;
  T* ptr_;
  size_t size_;

 public:
  explicit lightweight_uvector(const raft::resources& res) : res_(&res), ptr_{nullptr}, size_{0} {}

  [[nodiscard]] auto data() noexcept -> T* { return ptr_; }
  [[nodiscard]] auto data() const noexcept -> const T* { return ptr_; }
  [[nodiscard]] auto size() const noexcept -> size_t { return size_; }

  void resize(size_t new_size)
  {
    if (new_size == size_) { return; }
    if (std::holds_alternative<raft_res_type>(res_)) {
      auto& h = std::get<raft_res_type>(res_);
      res_    = rmm_res_type{raft::resource::get_workspace_resource_ref(*h),
                          raft::resource::get_cuda_stream(*h)};
    }
    auto& [r, s] = std::get<rmm_res_type>(res_);
    T* new_ptr   = nullptr;
    if (new_size > 0) {
      new_ptr = reinterpret_cast<T*>(r.allocate(s, new_size * sizeof(T), kAlign));
    }
    auto copy_size = std::min(size_, new_size);
    if (copy_size > 0) {
      cudaMemcpyAsync(new_ptr, ptr_, copy_size * sizeof(T), cudaMemcpyDefault, s);
    }
    if (size_ > 0) { r.deallocate(s, ptr_, size_ * sizeof(T), kAlign); }
    ptr_  = new_ptr;
    size_ = new_size;
  }

  void resize(size_t new_size, rmm::cuda_stream_view stream)
  {
    if (new_size == size_) { return; }
    if (std::holds_alternative<raft_res_type>(res_)) {
      auto& h = std::get<raft_res_type>(res_);
      res_    = rmm_res_type{raft::resource::get_workspace_resource_ref(*h), stream};
    } else {
      std::get<rmm::cuda_stream_view>(std::get<rmm_res_type>(res_)) = stream;
    }
    resize(new_size);
  }

  ~lightweight_uvector() noexcept
  {
    if (size_ > 0) {
      auto& [r, s] = std::get<rmm_res_type>(res_);
      r.deallocate(s, ptr_, size_ * sizeof(T), kAlign);
    }
  }
};

struct search_plan_impl_base : public search_params {
  int64_t dataset_size;
  int64_t dim;
  int64_t graph_degree;
  uint32_t topk;
  search_plan_impl_base(
    search_params params, int64_t dim, int64_t dataset_size, int64_t graph_degree, uint32_t topk)
    : search_params(params),
      dim(dim),
      dataset_size(dataset_size),
      graph_degree(graph_degree),
      topk(topk)
  {
    if (persistent) {
      if (algo == search_algo::AUTO) {
        algo = search_algo::SINGLE_CTA;
        RAFT_LOG_DEBUG("Auto strategy: persistent enabled, selecting single-cta");
      } else if (algo != search_algo::SINGLE_CTA) {
        RAFT_LOG_WARN(
          "CAGRA persistent kernel is only supported with SINGLE_CTA search (algo=%d); "
          "persistent will have no effect",
          static_cast<int>(algo));
      }
    } else if (algo == search_algo::AUTO) {
      const size_t num_sm = raft::getMultiProcessorCount();
      if (itopk_size <= 512 && search_params::max_queries >= num_sm * 2lu) {
        algo = search_algo::SINGLE_CTA;
        RAFT_LOG_DEBUG("Auto strategy: selecting single-cta");
      } else {
        algo = search_algo::MULTI_CTA;
        RAFT_LOG_DEBUG("Auto strategy: selecting multi-cta");
      }
    }
  }
};

template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SAMPLE_FILTER_T,
          typename SourceIndexT = IndexT,
          typename OutputIndexT = SourceIndexT>
struct search_plan_impl : public search_plan_impl_base {
  using DATA_T     = DataT;
  using INDEX_T    = IndexT;
  using DISTANCE_T = DistanceT;

  int64_t hash_bitlen;

  size_t small_hash_bitlen;
  size_t small_hash_reset_interval;
  size_t hashmap_size;
  uint32_t result_buffer_size;

  uint32_t smem_size;
  uint32_t num_seeds;

  lightweight_uvector<INDEX_T> hashmap;
  lightweight_uvector<uint32_t> num_executed_iterations;  // device or managed?
  lightweight_uvector<INDEX_T> dev_seed;
  dataset_descriptor_host<DataT, IndexT, DistanceT> dataset_desc;

  search_plan_impl(raft::resources const& res,
                   search_params params,
                   const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
                   int64_t dim,
                   int64_t dataset_size,
                   int64_t graph_degree,
                   uint32_t topk)
    : search_plan_impl_base(params, dim, dataset_size, graph_degree, topk),
      hashmap(res),
      num_executed_iterations(res),
      dev_seed(res),
      num_seeds(0),
      dataset_desc(dataset_desc)
  {
    adjust_search_params();
    check_params();
    calc_hashmap_params(res);
    if (!persistent) {  // Persistent kernel does not provide this functionality
      num_executed_iterations.resize(max_queries, raft::resource::get_cuda_stream(res));
    }
    RAFT_LOG_DEBUG("# algo = %d", static_cast<int>(algo));
  }

  virtual ~search_plan_impl() {}

  virtual void operator()(
    raft::resources const& res,
    raft::device_matrix_view<const INDEX_T, int64_t, raft::row_major> graph,
    std::optional<raft::device_vector_view<const SourceIndexT, int64_t>> source_indices,
    OutputIndexT* const result_indices_ptr,  // [num_queries, topk]
    DISTANCE_T* const result_distances_ptr,  // [num_queries, topk]
    const DATA_T* const queries_ptr,         // [num_queries, dataset_dim]
    const std::uint32_t num_queries,
    const INDEX_T* dev_seed_ptr,                   // [num_queries, num_seeds]
    std::uint32_t* const num_executed_iterations,  // [num_queries]
    uint32_t topk,
    SAMPLE_FILTER_T sample_filter) {};

  void adjust_search_params()
  {
    uint32_t _max_iterations = max_iterations;
    if (max_iterations == 0) {
      if (algo == search_algo::MULTI_CTA) {
        constexpr uint32_t mc_itopk_size   = 32;
        constexpr uint32_t mc_search_width = 1;
        _max_iterations                    = mc_itopk_size / mc_search_width;
      } else {
        _max_iterations = itopk_size / search_width;
      }
      int64_t num_reachable_nodes = 1;
      while (num_reachable_nodes < dataset_size) {
        num_reachable_nodes *= max((int64_t)2, graph_degree / 2);
        _max_iterations += 1;
      }
    }
    if (max_iterations < min_iterations) { _max_iterations = min_iterations; }
    if (max_iterations < _max_iterations) {
      RAFT_LOG_DEBUG(
        "# max_iterations is increased from %lu to %u.", max_iterations, _max_iterations);
      max_iterations = _max_iterations;
    }
    if (algo == search_algo::MULTI_CTA && (0.0 < filtering_rate && filtering_rate < 1.0)) {
      size_t adjusted_itopk_size =
        (size_t)((float)topk / (1.0 - filtering_rate) +
                 (float)(itopk_size - topk) / std::sqrt(1.0 - filtering_rate));
      if (adjusted_itopk_size % 32) { adjusted_itopk_size += 32 - (adjusted_itopk_size % 32); }
      if (itopk_size < adjusted_itopk_size) {
        RAFT_LOG_DEBUG(
          "# internal_topk is increased from %lu to %lu, considering fintering rate %f.",
          itopk_size,
          adjusted_itopk_size,
          filtering_rate);
        itopk_size = adjusted_itopk_size;
      }
    }
    if (itopk_size % 32) {
      uint32_t itopk32 = itopk_size;
      itopk32 += 32 - (itopk_size % 32);
      RAFT_LOG_DEBUG("# internal_topk is increased from %lu to %u, as it must be multiple of 32.",
                     itopk_size,
                     itopk32);
      itopk_size = itopk32;
    }
    team_size = dataset_desc.team_size;
  }

  // defines hash_bitlen, small_hash_bitlen, small_hash_reset interval, hash_size
  inline void calc_hashmap_params(raft::resources const& res)
  {
    // Determine hash size (bit length)
    hashmap_size              = 0;
    hash_bitlen               = 0;
    small_hash_bitlen         = 0;
    small_hash_reset_interval = 1024 * 1024;
    float max_fill_rate       = hashmap_max_fill_rate;
    if (algo == search_algo::MULTI_CTA) {
      const uint32_t mc_itopk_size = 32;
      const uint32_t mc_num_cta_per_query =
        max(search_width, raft::ceildiv(itopk_size, (size_t)mc_itopk_size));
      RAFT_LOG_DEBUG("# mc_itopk_size: %u", mc_itopk_size);
      RAFT_LOG_DEBUG("# mc_num_cta_per_query: %u", mc_num_cta_per_query);
      //
      // [visited_hash_table]
      // In the multi CTA algo, which node has been visited is managed in a hash
      // table that each CTA has in the shared memory. This hash table is not
      // shared among CTAs. This hash table is reset and restored in each iteration.
      //
      const uint32_t max_visited_nodes = mc_itopk_size + (graph_degree * 2);
      small_hash_bitlen                = 8;  // 256
      while (max_visited_nodes > hashmap::get_size(small_hash_bitlen) * max_fill_rate) {
        small_hash_bitlen += 1;
      }
      RAFT_EXPECTS(small_hash_bitlen <= 14, "small_hash_bitlen cannot be largen than 14 (16K)");
      //
      // [traversed_hash_table]
      // Whether a node has ever been used as the starting point for a traversal
      // in each iteration is managed in a separate hash table, which is shared
      // among the CTAs.
      //
      const auto max_traversed_nodes =
        mc_num_cta_per_query * max((size_t)mc_itopk_size, max_iterations);
      unsigned min_bitlen = 11;  // 2K
      if (min_bitlen < hashmap_min_bitlen) { min_bitlen = hashmap_min_bitlen; }
      hash_bitlen = min_bitlen;
      while (max_traversed_nodes > hashmap::get_size(hash_bitlen) * max_fill_rate) {
        hash_bitlen += 1;
      }
      RAFT_EXPECTS(hash_bitlen <= 25, "hash_bitlen cannot be largen than 25 (32M)");
    } else {
      while (hashmap_mode == hash_mode::AUTO || hashmap_mode == hash_mode::SMALL) {
        //
        // The small-hash reduces hash table size by initializing the hash table
        // for each iteration and re-registering only the nodes that should not be
        // re-visited in that iteration. Therefore, the size of small-hash should
        // be determined based on the internal topk size and the number of nodes
        // visited per iteration.
        //
        const auto max_visited_nodes = itopk_size + (search_width * graph_degree * 1);
        unsigned min_bitlen          = 8;   // 256
        unsigned max_bitlen          = 13;  // 8K
        if (min_bitlen < hashmap_min_bitlen) { min_bitlen = hashmap_min_bitlen; }
        hash_bitlen = min_bitlen;
        while (max_visited_nodes > hashmap::get_size(hash_bitlen) * max_fill_rate) {
          hash_bitlen += 1;
        }
        if (hash_bitlen > max_bitlen) {
          // Switch to normal hash if hashmap_mode is AUTO, otherwise exit.
          if (hashmap_mode == hash_mode::AUTO) {
            hash_bitlen = 0;
            break;
          } else {
            RAFT_FAIL(
              "small-hash cannot be used because the required hash size exceeds the limit (%u)",
              hashmap::get_size(max_bitlen));
          }
        }
        small_hash_bitlen = hash_bitlen;
        //
        // Sincc the hash table size is limited to a power of 2, the requirement,
        // the maximum fill rate, may be satisfied even if the frequency of hash
        // table reset is reduced to once every 2 or more iterations without
        // changing the hash table size. In that case, reduce the reset frequency.
        //
        small_hash_reset_interval = 1;
        while (1) {
          const auto max_visited_nodes =
            itopk_size + (search_width * graph_degree * (small_hash_reset_interval + 1));
          if (max_visited_nodes > hashmap::get_size(hash_bitlen) * max_fill_rate) { break; }
          small_hash_reset_interval += 1;
        }
        break;
      }
      if (hash_bitlen == 0) {
        //
        // The size of hash table is determined based on the maximum number of
        // nodes that may be visited before the search is completed and the
        // maximum fill rate of the hash table.
        //
        uint32_t max_visited_nodes = itopk_size + (search_width * graph_degree * max_iterations);
        unsigned min_bitlen        = 11;  // 2K
        if (min_bitlen < hashmap_min_bitlen) { min_bitlen = hashmap_min_bitlen; }
        hash_bitlen = min_bitlen;
        while (max_visited_nodes > hashmap::get_size(hash_bitlen) * max_fill_rate) {
          hash_bitlen += 1;
        }
        RAFT_EXPECTS(hash_bitlen <= 20,
                     "hash_bitlen cannot be largen than 20 (1M). You can decrease itopk_size, "
                     "search_width or max_iterations to reduce the required hashmap size.");
      }
    }
    RAFT_LOG_DEBUG("# internal topK = %lu", itopk_size);
    RAFT_LOG_DEBUG("# parent size = %lu", search_width);
    RAFT_LOG_DEBUG("# min_iterations = %lu", min_iterations);
    RAFT_LOG_DEBUG("# max_iterations = %lu", max_iterations);
    RAFT_LOG_DEBUG("# max_queries = %lu", max_queries);
    RAFT_LOG_DEBUG("# hashmap mode = %s%s-%u",
                   (small_hash_bitlen > 0 ? "small-" : ""),
                   "hash",
                   hashmap::get_size(hash_bitlen));
    if (small_hash_bitlen > 0) {
      RAFT_LOG_DEBUG("# small_hash_reset_interval = %lu", small_hash_reset_interval);
    }
    hashmap_size = sizeof(INDEX_T) * max_queries * hashmap::get_size(hash_bitlen);
    RAFT_LOG_DEBUG("# hashmap size: %lu", hashmap_size);
    if (hashmap_size >= 1024 * 1024 * 1024) {
      RAFT_LOG_DEBUG(" (%.2f GiB)", (double)hashmap_size / (1024 * 1024 * 1024));
    } else if (hashmap_size >= 1024 * 1024) {
      RAFT_LOG_DEBUG(" (%.2f MiB)", (double)hashmap_size / (1024 * 1024));
    } else if (hashmap_size >= 1024) {
      RAFT_LOG_DEBUG(" (%.2f KiB)", (double)hashmap_size / (1024));
    }
  }

  virtual void check(const uint32_t topk)
  {
    // For single-CTA and multi kernel
    RAFT_EXPECTS(
      topk <= itopk_size, "topk = %u must be smaller than itopk_size = %lu", topk, itopk_size);
  }

  inline void check_params()
  {
    std::string error_message = "";

    if (itopk_size > 1024) {
      if ((algo == search_algo::MULTI_CTA) || (algo == search_algo::MULTI_KERNEL)) {
      } else {
        error_message += std::string("- `internal_topk` (" + std::to_string(itopk_size) +
                                     ") must be smaller or equal to 1024");
      }
    }
    if (algo != search_algo::SINGLE_CTA && algo != search_algo::MULTI_CTA &&
        algo != search_algo::MULTI_KERNEL) {
      error_message += "An invalid kernel mode has been given: " + std::to_string((int)algo) + "";
    }
    if (thread_block_size != 0 && thread_block_size != 64 && thread_block_size != 128 &&
        thread_block_size != 256 && thread_block_size != 512 && thread_block_size != 1024) {
      error_message += "`thread_block_size` must be 0, 64, 128, 256 or 512. " +
                       std::to_string(thread_block_size) + " has been given.";
    }
    if (hashmap_min_bitlen > 20) {
      error_message += "`hashmap_min_bitlen` must be equal to or smaller than 20. " +
                       std::to_string(hashmap_min_bitlen) + " has been given.";
    }
    if (hashmap_max_fill_rate < 0.1 || hashmap_max_fill_rate >= 0.9) {
      error_message +=
        "`hashmap_max_fill_rate` must be equal to or greater than 0.1 and smaller than 0.9. " +
        std::to_string(hashmap_max_fill_rate) + " has been given.";
    }
    if constexpr (!std::is_same<SAMPLE_FILTER_T,
                                cuvs::neighbors::filtering::none_sample_filter>::value) {
      if (hashmap_mode == hash_mode::SMALL) {
        error_message += "`SMALL` hash is not available when filtering";
      } else {
        hashmap_mode = hash_mode::HASH;
      }
    }
    if (algo == search_algo::MULTI_CTA) {
      if (hashmap_mode == hash_mode::SMALL) {
        error_message += "`small_hash` is not available when 'search_mode' is \"multi-cta\"";
      } else {
        hashmap_mode = hash_mode::HASH;
      }
    }

    if (error_message.length() != 0) { THROW("[CAGRA Error] %s", error_message.c_str()); }
  }
};

}  // namespace cuvs::neighbors::cagra::detail
