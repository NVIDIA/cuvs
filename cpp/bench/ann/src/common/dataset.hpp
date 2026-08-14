/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "ann_types.hpp"
#include "blob.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <thread>

namespace cuvs::bench {

template <typename CarrierT>
void generate_bernoulli(CarrierT* data, size_t words, double p)
{
  constexpr size_t kBitsPerCarrierValue = sizeof(CarrierT) * 8;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::bernoulli_distribution d(p);
  for (size_t i = 0; i < words; i++) {
    CarrierT word = 0;
    for (size_t j = 0; j < kBitsPerCarrierValue; j++) {
      word |= CarrierT{d(gen)} << j;
    }
    data[i] = word;
  }
};

template <typename T>
struct ground_truth_map {
  using bitset_carrier_type                      = uint32_t;
  static constexpr uint32_t kMaxQueriesForRecall = 10'000;

  explicit ground_truth_map(std::string file_name,
                            uint32_t n_queries,
                            std::optional<blob<bitset_carrier_type>>& filter_bitset)
    : gt_maps_(n_queries)
  {
    // Eagerly iterate over and optionally filter the ground truth set to build gt_maps_ for up to
    // kMaxQueriesForRecall queries
    /* NOTE: recall correctness & filtering

    We generate the filtered ground truth values and build unordered_maps with them to
    enable O(1) lookup. We need enough ground truth values to compute recall correctly
    though. But the ground truth file only contains `max_k_` values per row; if there are
    less valid values than k among them, we overestimate the recall. Essentially, we compare
    the first `gt_maps_[query_idx].size()` values of the algorithm output, and this value can be
    less than `k`. In the extreme case of very high filtering rate, we may be bypassing
    entire rows of results. However, this is still better than no recall estimate at all.

    */
    auto ground_truth_set = blob<T>(file_name);
    max_k_                = ground_truth_set.n_cols();
    auto filter           = [&](T i) -> bool {
      if (!filter_bitset.has_value()) { return true; }
      // bitset is `32 = bitset_carrier_type * 8` times more dense than the data
      // use bitwise arithmetic to get the `row_id` and correct bit pos in the `word`
      auto word = filter_bitset->data(MemoryType::kHostMmap)[i >> 5];
      return word & (1 << (i & 31));
    };
    // Avoid CPU oversubscription when parallelizing recall calculation loop
    int num_map_building_worker_threads =
      std::thread::hardware_concurrency() - 1;  // -1 for the main thread
    // ensure non-negative number of workers (possible if hardware_concurrency()
    // does not return an expected value) by clamping to 0
    if (num_map_building_worker_threads < 0) { num_map_building_worker_threads = 0; }
    std::vector<std::thread> gt_map_building_workers;
    gt_map_building_workers.reserve(num_map_building_worker_threads);
    int chunk_size    = n_queries / (num_map_building_worker_threads + 1);
    int remainder     = n_queries % (num_map_building_worker_threads + 1);
    int stride        = (n_queries - 1) / kMaxQueriesForRecall + 1;  // round-up division
    auto build_gt_map = [&](int start, int end, int tid) -> void {
      for (int query_idx = start; query_idx < end; ++query_idx) {
        if (query_idx % stride) continue;
        for (std::uint32_t neighbor_rank = 0; neighbor_rank < max_k_; ++neighbor_rank) {
          auto id = ground_truth_set.data()[query_idx * max_k_ + neighbor_rank];
          if (!filter(id)) { continue; }
          if (gt_maps_[query_idx].count(id)) {
            throw std::invalid_argument(
              "Duplicate neighbor id found in ground truth set for query " +
              std::to_string(query_idx));
          }
          gt_maps_[query_idx][id] = neighbor_rank;
        }
      }
    };
    // Prewarm any lazily-initialized state on the main thread (e.g. memory-mapped
    // buffers inside `blob<T>::data()`) before dispatching worker threads, so the
    // workers only ever read already-initialized state through `data()`. This
    // protects this parallel section against any future lazy paths added inside
    // blob.hpp that may not be internally synchronized.
    (void)ground_truth_set.data();
    if (filter_bitset.has_value()) { (void)filter_bitset->data(MemoryType::kHostMmap); }
    // launch worker threads
    int start = 0;
    for (int tid = 0; tid < num_map_building_worker_threads; tid++) {
      int end = start + chunk_size;
      if (tid < remainder) { ++end; }
      gt_map_building_workers.emplace_back(build_gt_map, start, end, tid);
      start = end;
    }
    // main thread works on last chunk
    build_gt_map(start, n_queries, num_map_building_worker_threads);
    // join all worker threads
    for (auto& worker : gt_map_building_workers) {
      worker.join();
    }
  }

  [[nodiscard]] auto max_k() const -> uint32_t { return max_k_; }

  template <typename index_type>
  [[nodiscard]] auto count_matches(size_t query_idx, const index_type* candidates, uint32_t k) const
    -> std::pair<size_t, size_t>
  {
    if (query_idx >= gt_maps_.size() || gt_maps_[query_idx].empty()) return {0, 0};

    size_t matching = 0;
    for (uint32_t i = 0; i < k; ++i) {
      auto act_idx = candidates[i];
      if (gt_maps_[query_idx].count(act_idx) &&
          static_cast<uint32_t>(gt_maps_[query_idx].at(act_idx)) < k) {
        ++matching;
      }
    }
    size_t total = std::min(gt_maps_[query_idx].size(), static_cast<size_t>(k));
    return {matching, total};
  }

 private:
  // Hash maps of {id, neighbor_rank} for up to kMaxQueriesForRecall queries in the ground truth set
  // e.g. gt_maps_[i][j] = k means that for the i-th query in the ground truth set, the neighbor
  // with idx j is the k-th nearest. Note that the nearest neighbor rank starts from 0.
  std::vector<std::unordered_map<T, T>> gt_maps_;
  uint32_t max_k_ = 0;  // number of nearest neighbors in the ground truth
};

template <typename DataT, typename IdxT = int32_t>
struct dataset {
 public:
  using bitset_carrier_type = typename ground_truth_map<IdxT>::bitset_carrier_type;
  static inline constexpr size_t kBitsPerCarrierValue = sizeof(bitset_carrier_type) * 8;

 private:
  std::string name_;
  std::string distance_;
  blob<DataT> base_set_;
  blob<DataT> query_set_;
  std::optional<blob<bitset_carrier_type>> filter_bitset_;
  std::optional<ground_truth_map<IdxT>> ground_truth_map_;

  // Protects the lazy mutations of the blobs accessed by multiple threads
  mutable std::mutex mutex_;
  // The dim can be read either from the training set or from the query set.
  // This cache variable is filled from either of the two sets loaded first.
  mutable std::atomic<int> dim_ = -1;

  // Cache the dim value from the passed blob.
  inline void cache_dim(const blob<DataT>& blob) const
  {
    if (dim_.load(std::memory_order_relaxed) == -1) {
      dim_.store(static_cast<int>(blob.n_cols()), std::memory_order_relaxed);
    }
  }

 public:
  dataset(std::string name,
          std::string base_file,
          uint32_t subset_first_row,
          uint32_t subset_size,
          std::string query_file,
          std::string distance,
          std::optional<std::string> groundtruth_neighbors_file,
          std::optional<double> filtering_rate = std::nullopt)
    : name_{std::move(name)},
      distance_{std::move(distance)},
      base_set_{base_file, subset_first_row, subset_size},
      query_set_{query_file}
  {
    if (filtering_rate.has_value()) {
      // Generate a random bitset for filtering
      auto n_rows = static_cast<size_t>(subset_size) + static_cast<size_t>(subset_first_row);
      if (subset_size == 0) {
        // Read the base set size as a last resort only - for better laziness
        n_rows = base_set_size();
      }
      auto bitset_size = (n_rows - 1) / kBitsPerCarrierValue + 1;
      blob_file<bitset_carrier_type> bitset_blob_file{static_cast<uint32_t>(bitset_size), 1};
      blob_mmap<bitset_carrier_type> bitset_blob{
        std::move(bitset_blob_file), false, HugePages::kDisable};
      generate_bernoulli(const_cast<bitset_carrier_type*>(bitset_blob.data()),
                         bitset_size,
                         1.0 - filtering_rate.value());
      filter_bitset_.emplace(std::move(bitset_blob));
    }

    if (groundtruth_neighbors_file.has_value()) {
      ground_truth_map_.emplace(ground_truth_map<IdxT>{
        groundtruth_neighbors_file.value(), query_set_.n_rows(), filter_bitset_});
    }
  }

  [[nodiscard]] auto name() const -> std::string { return name_; }
  [[nodiscard]] auto distance() const -> std::string { return distance_; }
  [[nodiscard]] auto dim() const -> int
  {
    auto d = dim_.load(std::memory_order_relaxed);
    if (d > -1) { return d; }
    std::lock_guard<std::mutex> lock(mutex_);
    // Otherwise, try reading both (one of the two sets may be missing)
    try {
      d = static_cast<int>(query_set_.n_cols());
    } catch (const std::runtime_error& e) {
      // Any exception raised above will re-raise next time we try to access the query set.
      query_set_.reset_lazy_state();
      // If the query set is not accessible, use the base set.
      // Don't catch the exception here, because we have nothing else to do anyway.
      d = static_cast<int>(base_set_.n_cols());
    }
    dim_.store(d, std::memory_order_relaxed);
    return d;
  }
  [[nodiscard]] auto max_k() const -> uint32_t
  {
    if (ground_truth_map_.has_value()) { return ground_truth_map_->max_k(); }
    return 0;
  }
  [[nodiscard]] auto base_set_size() const -> size_t
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto r = base_set_.n_rows();
    cache_dim(base_set_);
    return r;
  }
  [[nodiscard]] auto query_set_size() const -> size_t
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto r = query_set_.n_rows();
    cache_dim(query_set_);
    return r;
  }

  [[nodiscard]] auto gt_maps() const -> const std::optional<ground_truth_map<IdxT>>&
  {
    return ground_truth_map_;
  }

  [[nodiscard]] auto query_set() const -> const DataT*
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* r = query_set_.data();
    cache_dim(query_set_);
    return r;
  }
  [[nodiscard]] auto query_set(MemoryType memory_type,
                               HugePages request_hugepages_2mb = HugePages::kDisable) const
    -> const DataT*
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* r = query_set_.data(memory_type, request_hugepages_2mb);
    cache_dim(query_set_);
    return r;
  }

  [[nodiscard]] auto base_set() const -> const DataT*
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* r = base_set_.data();
    cache_dim(base_set_);
    return r;
  }
  [[nodiscard]] auto base_set(MemoryType memory_type,
                              HugePages request_hugepages_2mb = HugePages::kDisable) const
    -> const DataT*
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* r = base_set_.data(memory_type, request_hugepages_2mb);
    cache_dim(base_set_);
    return r;
  }

  [[nodiscard]] auto filter_bitset() const -> const bitset_carrier_type*
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (filter_bitset_.has_value()) { return filter_bitset_->data(); }
    return nullptr;
  }

  [[nodiscard]] auto filter_bitset(MemoryType memory_type,
                                   HugePages request_hugepages_2mb = HugePages::kDisable) const
    -> const bitset_carrier_type*
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (filter_bitset_.has_value()) {
      return filter_bitset_->data(memory_type, request_hugepages_2mb);
    }
    return nullptr;
  }
};

}  // namespace  cuvs::bench
