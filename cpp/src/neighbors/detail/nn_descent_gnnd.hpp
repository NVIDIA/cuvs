/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/nn_descent.hpp>
#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <limits>

namespace cuvs::neighbors::nn_descent::detail {

using DistData_t = float;
constexpr int DEGREE_ON_DEVICE{32};
constexpr int SEGMENT_SIZE{32};
constexpr int counter_interval{100};
template <typename Index_t>
struct InternalID_t;

inline size_t roundUp32(size_t num) { return (num + 31) / 32 * 32; }

// InternalID_t uses 1 bit for marking (new or old).
template <>
struct InternalID_t<int> {
 private:
  using Index_t = int;
  Index_t id_{std::numeric_limits<Index_t>::max()};

 public:
  inline _RAFT_HOST_DEVICE bool is_new() const { return id_ >= 0; }
  inline _RAFT_HOST_DEVICE Index_t& id_with_flag() { return id_; }
  inline _RAFT_HOST_DEVICE Index_t id() const
  {
    if (is_new()) return id_;
    return -id_ - 1;
  }
  inline _RAFT_HOST_DEVICE void mark_old()
  {
    if (id_ >= 0) id_ = -id_ - 1;
  }
  inline _RAFT_HOST_DEVICE bool operator==(const InternalID_t<int>& other) const
  {
    return id() == other.id();
  }
};

struct BuildConfig {
  size_t max_dataset_size;
  size_t dataset_dim;
  size_t node_degree{64};
  size_t internal_node_degree{0};
  // If internal_node_degree == 0, the value of node_degree will be assigned to it
  size_t max_iterations{50};
  float termination_threshold{0.0001};
  size_t output_graph_degree{32};
  cuvs::distance::DistanceType metric{cuvs::distance::DistanceType::L2Expanded};
  cuvs::neighbors::nn_descent::DIST_COMP_DTYPE dist_comp_dtype{
    cuvs::neighbors::nn_descent::DIST_COMP_DTYPE::AUTO};
};

template <typename Index_t>
class BloomFilter {
 public:
  BloomFilter(size_t nrow, size_t num_sets_per_list, size_t num_hashs)
    : nrow_(nrow),
      num_sets_per_list_(num_sets_per_list),
      num_hashs_(num_hashs),
      bitsets_(nrow * num_bits_per_set_ * num_sets_per_list)
  {
  }

  void add(size_t list_id, Index_t key)
  {
    if (is_cleared) { is_cleared = false; }
    uint32_t hash         = hash_0(key);
    size_t global_set_idx = list_id * num_bits_per_set_ * num_sets_per_list_ +
                            key % num_sets_per_list_ * num_bits_per_set_;
    bitsets_[global_set_idx + hash % num_bits_per_set_] = 1;
    for (size_t i = 1; i < num_hashs_; i++) {
      hash                                                = hash + hash_1(key);
      bitsets_[global_set_idx + hash % num_bits_per_set_] = 1;
    }
  }

  void set_nrow(size_t nrow) { nrow_ = nrow; }

  bool check(size_t list_id, Index_t key)
  {
    bool is_present       = true;
    uint32_t hash         = hash_0(key);
    size_t global_set_idx = list_id * num_bits_per_set_ * num_sets_per_list_ +
                            key % num_sets_per_list_ * num_bits_per_set_;
    is_present &= bitsets_[global_set_idx + hash % num_bits_per_set_];

    if (!is_present) return false;
    for (size_t i = 1; i < num_hashs_; i++) {
      hash = hash + hash_1(key);
      is_present &= bitsets_[global_set_idx + hash % num_bits_per_set_];
      if (!is_present) return false;
    }
    return true;
  }

  void clear()
  {
    if (is_cleared) return;
#pragma omp parallel for
    for (size_t i = 0; i < nrow_ * num_bits_per_set_ * num_sets_per_list_; i++) {
      bitsets_[i] = 0;
    }
    is_cleared = true;
  }

 private:
  uint32_t hash_0(uint32_t value)
  {
    value *= 1103515245;
    value += 12345;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return value;
  }

  uint32_t hash_1(uint32_t value)
  {
    value *= 1664525;
    value += 1013904223;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return value;
  }

  static constexpr int num_bits_per_set_ = 512;
  bool is_cleared{true};
  std::vector<bool> bitsets_;
  size_t nrow_;
  size_t num_sets_per_list_;
  size_t num_hashs_;
};

template <typename Index_t>
struct CUVS_EXPORT GnndGraph {
  raft::resources const& res;
  static constexpr int segment_size = 32;
  InternalID_t<Index_t>* h_graph;

  size_t nrow;
  size_t node_degree;
  int num_samples;
  int num_segments;

  raft::host_matrix<DistData_t, size_t, raft::row_major> h_dists;

  raft::pinned_matrix<Index_t, size_t> h_graph_new;
  raft::pinned_vector<int2, size_t> h_list_sizes_new;

  raft::pinned_matrix<Index_t, size_t> h_graph_old;
  raft::pinned_vector<int2, size_t> h_list_sizes_old;
  BloomFilter<Index_t> bloom_filter;

  GnndGraph(const GnndGraph&)            = delete;
  GnndGraph& operator=(const GnndGraph&) = delete;
  GnndGraph(raft::resources const& res,
            const size_t nrow,
            const size_t node_degree,
            const size_t internal_node_degree,
            const size_t num_samples);
  void init_random_graph();
  // TODO: Create a generic bloom filter utility https://github.com/rapidsai/raft/issues/1827
  // Use Bloom filter to sample "new" neighbors for local joining
  void sample_graph_new(InternalID_t<Index_t>* new_neighbors, const size_t width);
  void sample_graph(bool sample_new);
  void update_graph(const InternalID_t<Index_t>* new_neighbors,
                    const DistData_t* new_dists,
                    const size_t width,
                    std::atomic<int64_t>& update_counter);
  void sort_lists();
  void clear();
  ~GnndGraph();
};

template <typename Data_t = float, typename Index_t = int>
class CUVS_EXPORT GNND {
 public:
  GNND(raft::resources const& res, const BuildConfig& build_config);
  GNND(const GNND&)            = delete;
  GNND& operator=(const GNND&) = delete;

  template <typename DistEpilogue_t = raft::identity_op>
  void build(Data_t* data,
             const Index_t nrow,
             Index_t* output_graph,
             bool return_distances,
             DistData_t* output_distances,
             DistEpilogue_t dist_epilogue = DistEpilogue_t{});
  ~GNND()    = default;
  using ID_t = InternalID_t<Index_t>;
  void reset(raft::resources const& res);

 private:
  void add_reverse_edges(Index_t* graph_ptr,
                         Index_t* h_rev_graph_ptr,
                         Index_t* d_rev_graph_ptr,
                         int2* list_sizes,
                         cudaStream_t stream = 0);

  template <typename DistEpilogue_t>
  void local_join(cudaStream_t stream = 0, DistEpilogue_t dist_epilogue = DistEpilogue_t{});

  raft::resources const& res;

  BuildConfig build_config_;
  GnndGraph<Index_t> graph_;
  std::atomic<int64_t> update_counter_;

  size_t nrow_;
  size_t ndim_;

  using input_t = std::remove_const_t<Data_t>;

  // d_data_half_ is used for a special case when input data is fp32 on host and distances will be
  // computed in fp16 (dist_comp_dtype == FP16, or AUTO with dim > 16): we store the dataset on
  // device as fp16 (instead of fp32) to halve the device memory footprint and WMMA kernel read
  // bandwidth.
  std::optional<raft::device_matrix<half, size_t, raft::row_major>> d_data_half_;
  // d_data_direct_ is used when input data is on host, and we need to copy it to device
  std::optional<raft::device_matrix<input_t, size_t, raft::row_major>> d_data_direct_;

  // d_data_ptr_ is used to store the general pointer to the input data
  const void* d_data_ptr_{nullptr};
  raft::device_vector<DistData_t, size_t> l2_norms_;

  raft::device_matrix<ID_t, size_t, raft::row_major> graph_buffer_;
  raft::device_matrix<DistData_t, size_t, raft::row_major> dists_buffer_;

  raft::pinned_matrix<ID_t, size_t> graph_host_buffer_;
  raft::pinned_matrix<DistData_t, size_t> dists_host_buffer_;

  raft::device_vector<int, size_t> d_locks_;

  raft::pinned_matrix<Index_t, size_t> h_rev_graph_new_;
  raft::pinned_matrix<Index_t, size_t> h_graph_old_;
  raft::pinned_matrix<Index_t, size_t> h_rev_graph_old_;
  // int2.x is the number of forward edges, int2.y is the number of reverse edges

  raft::device_vector<int2, size_t> d_list_sizes_new_;
  raft::device_vector<int2, size_t> d_list_sizes_old_;
};

inline BuildConfig get_build_config(raft::resources const& res,
                                    const index_params& params,
                                    size_t num_rows,
                                    size_t num_cols,
                                    const cuvs::distance::DistanceType metric,
                                    size_t& extended_graph_degree,
                                    size_t& graph_degree)
{
  RAFT_EXPECTS(num_rows < std::numeric_limits<int>::max() - 1,
               "The dataset size for GNND should be less than %d",
               std::numeric_limits<int>::max() - 1);
  auto allowed_metrics = params.metric == cuvs::distance::DistanceType::L2Expanded ||
                         params.metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                         params.metric == cuvs::distance::DistanceType::CosineExpanded ||
                         params.metric == cuvs::distance::DistanceType::InnerProduct ||
                         params.metric == cuvs::distance::DistanceType::BitwiseHamming ||
                         params.metric == cuvs::distance::DistanceType::L1;
  RAFT_EXPECTS(allowed_metrics,
               "The metric for NN Descent should be L2Expanded, L2SqrtExpanded, CosineExpanded, "
               "InnerProduct, BitwiseHamming or L1");
  RAFT_EXPECTS(
    metric == params.metric,
    "The metrics set in nn_descent::index_params and nn_descent::index are inconsistent");
  size_t intermediate_degree = params.intermediate_graph_degree;
  graph_degree               = params.graph_degree;

  if (intermediate_degree >= num_rows) {
    RAFT_LOG_WARN(
      "Intermediate graph degree cannot be larger than number of rows in dataset, reducing it to "
      "%lu",
      num_rows - 1);
    intermediate_degree = num_rows - 1;
  }
  if (intermediate_degree < graph_degree) {
    RAFT_LOG_WARN(
      "Graph degree (%lu) cannot be larger than intermediate graph degree (%lu), reducing "
      "graph_degree.",
      graph_degree,
      intermediate_degree);
    graph_degree = intermediate_degree;
  }

  // The elements in each knn-list are partitioned into different buckets, and we need more buckets
  // to mitigate bucket collisions. `intermediate_degree` is OK to larger than
  // extended_graph_degree.
  extended_graph_degree =
    roundUp32(static_cast<size_t>(graph_degree * (graph_degree <= 32 ? 1.0 : 1.3)));
  size_t extended_intermediate_degree =
    roundUp32(static_cast<size_t>(intermediate_degree * (intermediate_degree <= 32 ? 1.0 : 1.3)));

  BuildConfig build_config{.max_dataset_size      = num_rows,
                           .dataset_dim           = num_cols,
                           .node_degree           = extended_graph_degree,
                           .internal_node_degree  = extended_intermediate_degree,
                           .max_iterations        = params.max_iterations,
                           .termination_threshold = params.termination_threshold,
                           .output_graph_degree   = params.graph_degree,
                           .metric                = params.metric,
                           .dist_comp_dtype       = params.dist_comp_dtype};
  return build_config;
}

}  // namespace cuvs::neighbors::nn_descent::detail
