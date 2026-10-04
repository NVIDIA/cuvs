/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef CUVS_ENABLE_FLOWANN_SEARCH

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace cuvs::neighbors::cagra::experimental::flowann {

namespace detail {
struct search_context_access;
}

class search_session;

/** Parameters controlling the host-serviced cross-graph queues. */
struct queue_params {
  /** Number of independent GPU-to-CPU command queues. */
  std::uint32_t num_queues = 1;
  /** Number of CPU pause instructions issued after an empty poll. */
  std::uint32_t empty_pause = 64;
  /** Collect aggregate queue and polling statistics. */
  bool collect_statistics = false;
};

/** Aggregate counters collected by the host-serviced cross-graph queues. */
struct queue_statistics {
  std::uint64_t polls{};
  std::uint64_t empty_polls{};
  std::uint64_t processed_commands{};
};

/** FlowANN search parameters. */
struct search_params : cuvs::neighbors::cagra::search_params {
  /** Maximum number of preloaded medoid seed nodes used to initialize each query. */
  std::uint32_t num_seeds = 0;
  /** Scale used by the adaptive deferred-cross-edge synchronization window. */
  float sync_window_scale = 10.0f;
  /** Parent-position drop that forces synchronization for a submitted cross-edge request. */
  std::uint32_t sync_drop_threshold = 51;
};

/**
 * @brief Owns FlowANN command queues and their CPU polling threads.
 *
 * The cross-graph storage passed to the constructor must outlive this context. A context can be
 * reused across search calls for the same index. Construction and search must use the same current
 * CUDA device; CPU pollers select that device before servicing requests.
 */
class CUVS_EXPORT search_context {
 public:
  search_context(raft::host_matrix_view<const std::uint32_t, int64_t, raft::row_major> cross_graph,
                 queue_params params = {});
  ~search_context();

  search_context(search_context const&)                    = delete;
  auto operator=(search_context const&) -> search_context& = delete;
  search_context(search_context&&) noexcept;
  auto operator=(search_context&&) noexcept -> search_context&;

  /**
   * Pause all CPU pollers after in-flight work has completed.
   *
   * @throws raft::logic_error if a search_session is active.
   */
  void pause();
  /** Resume all CPU pollers before submitting search work. */
  void resume();
  [[nodiscard]] auto num_queues() const noexcept -> std::uint32_t;
  /** Return aggregate counters. Counters are zero unless statistics were enabled. */
  [[nodiscard]] auto statistics() const noexcept -> queue_statistics;

 private:
  struct impl;
  std::unique_ptr<impl> impl_;

  friend struct detail::search_context_access;
  friend class search_session;
};

/**
 * Keep a search context's pollers running for the lifetime of this session.
 *
 * Use a session for repeated low-latency search calls. The search context must outlive the session
 * and must not be moved while the session is active.
 */
class CUVS_EXPORT search_session {
 public:
  explicit search_session(search_context& context);
  ~search_session();

  search_session(search_session const&)                    = delete;
  auto operator=(search_session const&) -> search_session& = delete;
  search_session(search_session&& other) noexcept;
  auto operator=(search_session&& other) noexcept -> search_session&;

 private:
  void reset() noexcept;

  search_context* context_{};
};

/**
 * @brief FlowANN tiered-graph index.
 *
 * The index stores a compressed inner graph in device memory and the cross-graph rows in host
 * memory. The dataset is a non-owning view and its backing storage must outlive the index.
 */
template <typename T,
          typename IdxT,
          ann_dataset_view DatasetViewT = device_padded_dataset_view<T, int64_t>>
class index : public cuvs::neighbors::index {
 public:
  using value_type         = T;
  using index_type         = IdxT;
  using graph_index_type   = std::uint32_t;
  using inner_index_type   = std::uint8_t;
  using dataset_view_type  = DatasetViewT;
  using search_params_type = flowann::search_params;

  static_assert(!raft::is_narrowing_v<std::uint32_t, IdxT>,
                "IdxT must be able to represent all uint32_t graph indices");

  explicit index(raft::resources const& res,
                 cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded,
                 std::uint16_t node_per_cacheline    = 4,
                 std::uint16_t n_bits                = 16)
    : metric_(metric),
      node_per_cacheline_(node_per_cacheline),
      n_bits_(n_bits),
      inner_graph_(raft::make_device_matrix<inner_index_type, int64_t>(res, 0, 0)),
      cross_graph_(raft::make_host_matrix<graph_index_type, int64_t>(0, 0)),
      dataset_norms_(std::nullopt)
  {
    validate_encoding_();
  }

  index(raft::resources const& res,
        cuvs::distance::DistanceType metric,
        DatasetViewT dataset,
        std::uint16_t node_per_cacheline = 4,
        std::uint16_t n_bits             = 16)
    : metric_(metric),
      node_per_cacheline_(node_per_cacheline),
      n_bits_(n_bits),
      inner_graph_(raft::make_device_matrix<inner_index_type, int64_t>(res, 0, 0)),
      cross_graph_(raft::make_host_matrix<graph_index_type, int64_t>(0, 0)),
      dataset_(dataset),
      dataset_norms_(std::nullopt)
  {
    validate_encoding_();
    if constexpr (cuvs::neighbors::is_device_padded_dataset_view_v<DatasetViewT>) {
      if (metric_ == cuvs::distance::DistanceType::CosineExpanded && dataset_.n_rows() > 0) {
        compute_dataset_norms_(res);
        raft::resource::sync_stream(res);
      }
    }
  }

  index(index const&)                    = delete;
  auto operator=(index const&) -> index& = delete;
  index(index&&)                         = default;
  auto operator=(index&&) -> index&      = default;
  ~index()                               = default;

  [[nodiscard]] constexpr auto metric() const noexcept -> cuvs::distance::DistanceType
  {
    return metric_;
  }

  [[nodiscard]] auto size() const noexcept -> IdxT
  {
    return static_cast<IdxT>(cross_graph_.extent(0));
  }

  [[nodiscard]] auto dim() const noexcept -> std::uint32_t { return dataset_.dim(); }

  [[nodiscard]] auto graph_degree() const noexcept -> std::uint32_t { return full_graph_degree_; }

  /** Uniform number of neighbors stored in the packed resident graph. Zero for legacy layouts. */
  [[nodiscard]] constexpr auto resident_degree() const noexcept -> std::uint32_t
  {
    return resident_degree_;
  }

  /** Maximum number of host-resident neighbors stored after the count column. */
  [[nodiscard]] auto cross_degree() const noexcept -> std::uint32_t
  {
    return cross_graph_.extent(1) == 0 ? 0 : static_cast<std::uint32_t>(cross_graph_.extent(1) - 1);
  }

  [[nodiscard]] constexpr auto node_per_cacheline() const noexcept -> std::uint16_t
  {
    return node_per_cacheline_;
  }

  [[nodiscard]] constexpr auto n_bits() const noexcept -> std::uint16_t { return n_bits_; }

  [[nodiscard]] auto dataset() const noexcept -> DatasetViewT const& { return dataset_; }

  /** Dataset norms used by cosine distance. */
  [[nodiscard]] auto dataset_norms() const noexcept
    -> std::optional<raft::device_vector_view<const float, int64_t>>
  {
    if (!dataset_norms_.has_value()) { return std::nullopt; }
    return raft::make_const_mdspan(dataset_norms_->view());
  }

  void update_dataset(raft::resources const& res, DatasetViewT dataset)
  {
    if (cross_graph_.extent(0) > 0) {
      RAFT_EXPECTS(dataset.n_rows() == cross_graph_.extent(0),
                   "Dataset and FlowANN graph must have the same number of rows");
    }
    dataset_ = dataset;
    dataset_norms_.reset();
    if constexpr (cuvs::neighbors::is_device_padded_dataset_view_v<DatasetViewT>) {
      if (metric_ == cuvs::distance::DistanceType::CosineExpanded && dataset_.n_rows() > 0) {
        compute_dataset_norms_(res);
        raft::resource::sync_stream(res);
      }
    }
  }

  [[nodiscard]] auto inner_graph() const noexcept
    -> raft::device_matrix_view<const inner_index_type, int64_t, raft::row_major>
  {
    return raft::make_const_mdspan(inner_graph_.view());
  }

  /**
   * Cross-graph rows have shape `[size, cross_degree + 1]`. Element zero stores the number of valid
   * cross-graph neighbors in the rest of the row. `cross_degree` may be smaller than the complete
   * graph degree when every node has a uniform resident degree.
   */
  [[nodiscard]] auto cross_graph() const noexcept
    -> raft::host_matrix_view<const graph_index_type, int64_t, raft::row_major>
  {
    return raft::make_const_mdspan(cross_graph_.view());
  }

  [[nodiscard]] auto source_indices() const noexcept
    -> std::optional<raft::device_vector_view<const IdxT, int64_t>>
  {
    if (!source_indices_.has_value()) { return std::nullopt; }
    return raft::make_const_mdspan(source_indices_->view());
  }

  [[nodiscard]] auto subgraph_offsets() const noexcept
    -> std::optional<raft::device_vector_view<const std::uint32_t, int64_t>>
  {
    if (!subgraph_offsets_.has_value()) { return std::nullopt; }
    return raft::make_const_mdspan(subgraph_offsets_->view());
  }

  /** Subgraph identifier for every internal node. */
  [[nodiscard]] auto subgraph_ids() const noexcept
    -> std::optional<raft::device_vector_view<const std::uint32_t, int64_t>>
  {
    if (!subgraph_ids_.has_value()) { return std::nullopt; }
    return raft::make_const_mdspan(subgraph_ids_->view());
  }

  /** Optional medoid seed nodes shared by all queries. */
  [[nodiscard]] auto seeds() const noexcept
    -> std::optional<raft::device_vector_view<const IdxT, int64_t>>
  {
    if (!seeds_.has_value()) { return std::nullopt; }
    return raft::make_const_mdspan(seeds_->view());
  }

  /** Cached cross-graph rows for the fixed medoid seeds. */
  [[nodiscard]] auto seed_cross_graph() const noexcept
    -> std::optional<raft::device_matrix_view<const graph_index_type, int64_t, raft::row_major>>
  {
    if (!seed_cross_graph_.has_value()) { return std::nullopt; }
    return raft::make_const_mdspan(seed_cross_graph_->view());
  }

  [[nodiscard]] auto seed_lookup_keys() const noexcept
    -> std::optional<raft::device_vector_view<const IdxT, int64_t>>
  {
    if (!seed_lookup_keys_.has_value()) { return std::nullopt; }
    return raft::make_const_mdspan(seed_lookup_keys_->view());
  }

  [[nodiscard]] auto seed_lookup_values() const noexcept
    -> std::optional<raft::device_vector_view<const std::uint32_t, int64_t>>
  {
    if (!seed_lookup_values_.has_value()) { return std::nullopt; }
    return raft::make_const_mdspan(seed_lookup_values_->view());
  }

  void update_graph(
    raft::resources const& res,
    raft::host_matrix_view<const inner_index_type, int64_t, raft::row_major> inner_graph,
    raft::host_matrix_view<const graph_index_type, int64_t, raft::row_major> cross_graph)
  {
    auto const inferred_degree =
      cross_graph.extent(1) == 0 ? 0u : static_cast<std::uint32_t>(cross_graph.extent(1) - 1);
    update_graph(res, inner_graph, cross_graph, inferred_degree, 0);
  }

  /** Replace the graph and explicitly describe a compact uniform-residency layout. */
  void update_graph(
    raft::resources const& res,
    raft::host_matrix_view<const inner_index_type, int64_t, raft::row_major> inner_graph,
    raft::host_matrix_view<const graph_index_type, int64_t, raft::row_major> cross_graph,
    std::uint32_t full_graph_degree,
    std::uint32_t resident_degree)
  {
    validate_graph_(inner_graph, cross_graph, full_graph_degree, resident_degree);

    auto new_inner = raft::make_device_matrix<inner_index_type, int64_t>(
      res, inner_graph.extent(0), inner_graph.extent(1));
    raft::copy(new_inner.data_handle(),
               inner_graph.data_handle(),
               inner_graph.size(),
               raft::resource::get_cuda_stream(res));

    auto new_cross = raft::make_host_matrix<graph_index_type, int64_t>(cross_graph.extent(0),
                                                                       cross_graph.extent(1));
    std::copy_n(cross_graph.data_handle(), cross_graph.size(), new_cross.data_handle());

    inner_graph_       = std::move(new_inner);
    cross_graph_       = std::move(new_cross);
    full_graph_degree_ = full_graph_degree;
    resident_degree_   = resident_degree;
    rebuild_seed_cross_graph_(res);
  }

  /** Replace the graph while taking ownership of the host cross-graph allocation. */
  void update_graph(raft::resources const& res,
                    raft::host_matrix<inner_index_type, int64_t, raft::row_major>&& inner_graph,
                    raft::host_matrix<graph_index_type, int64_t, raft::row_major>&& cross_graph)
  {
    auto const inferred_degree =
      cross_graph.extent(1) == 0 ? 0u : static_cast<std::uint32_t>(cross_graph.extent(1) - 1);
    update_graph(res, std::move(inner_graph), std::move(cross_graph), inferred_degree, 0);
  }

  /** Replace the graph while taking ownership of a compact uniform-residency layout. */
  void update_graph(raft::resources const& res,
                    raft::host_matrix<inner_index_type, int64_t, raft::row_major>&& inner_graph,
                    raft::host_matrix<graph_index_type, int64_t, raft::row_major>&& cross_graph,
                    std::uint32_t full_graph_degree,
                    std::uint32_t resident_degree)
  {
    validate_graph_(raft::make_const_mdspan(inner_graph.view()),
                    raft::make_const_mdspan(cross_graph.view()),
                    full_graph_degree,
                    resident_degree);

    auto new_inner = raft::make_device_matrix<inner_index_type, int64_t>(
      res, inner_graph.extent(0), inner_graph.extent(1));
    raft::copy(new_inner.data_handle(),
               inner_graph.data_handle(),
               inner_graph.size(),
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);

    inner_graph_       = std::move(new_inner);
    cross_graph_       = std::move(cross_graph);
    full_graph_degree_ = full_graph_degree;
    resident_degree_   = resident_degree;
    rebuild_seed_cross_graph_(res);
  }

  /** Replace a variable-residency grouped graph while retaining the packed rows on the device. */
  void update_graph(raft::resources const& res,
                    raft::device_matrix<inner_index_type, int64_t, raft::row_major>&& inner_graph,
                    raft::host_matrix<graph_index_type, int64_t, raft::row_major>&& cross_graph,
                    bool validate_edges)
  {
    RAFT_EXPECTS(cross_graph.extent(0) > 0 && cross_graph.extent(1) > 1,
                 "FlowANN grouped cross graph must contain rows and neighbors");
    auto const full_degree = static_cast<std::uint32_t>(cross_graph.extent(1) - 1);
    auto const expected_rows =
      raft::div_rounding_up_safe<int64_t>(cross_graph.extent(0), node_per_cacheline_);
    auto const has_inner_graph = inner_graph.extent(0) != 0 || inner_graph.extent(1) != 0;
    if (has_inner_graph) {
      RAFT_EXPECTS(inner_graph.extent(0) == expected_rows,
                   "FlowANN grouped inner graph has an invalid row count");
      RAFT_EXPECTS(inner_graph.extent(1) >= node_per_cacheline_,
                   "FlowANN grouped inner graph rows cannot hold their count headers");
    }
    if (dataset_.n_rows() > 0) {
      RAFT_EXPECTS(dataset_.n_rows() == cross_graph.extent(0),
                   "Dataset and FlowANN grouped graph must have the same number of rows");
    }
    for (int64_t node = 0; node < cross_graph.extent(0); ++node) {
      auto const cross_count = cross_graph(node, 0);
      RAFT_EXPECTS(cross_count <= full_degree,
                   "FlowANN grouped cross-neighbor count exceeds graph degree at row %ld",
                   static_cast<long>(node));
      if (validate_edges) {
        for (std::uint32_t edge = 0; edge < cross_count; ++edge) {
          RAFT_EXPECTS(cross_graph(node, edge + 1) < cross_graph.extent(0),
                       "FlowANN grouped cross graph contains an out-of-range node at row %ld",
                       static_cast<long>(node));
        }
      }
    }

    inner_graph_       = std::move(inner_graph);
    cross_graph_       = std::move(cross_graph);
    full_graph_degree_ = full_degree;
    resident_degree_   = 0;
    rebuild_seed_cross_graph_(res);
  }

  /** Replace a compact graph while taking ownership of both device and host allocations. */
  void update_graph(raft::resources const& res,
                    raft::device_matrix<inner_index_type, int64_t, raft::row_major>&& inner_graph,
                    raft::host_matrix<graph_index_type, int64_t, raft::row_major>&& cross_graph,
                    std::uint32_t full_graph_degree,
                    std::uint32_t resident_degree)
  {
    RAFT_EXPECTS(cross_graph.extent(0) > 0 && cross_graph.extent(1) > 0,
                 "FlowANN cross graph must contain rows and a neighbor-count column");
    RAFT_EXPECTS(full_graph_degree > 0 && resident_degree <= full_graph_degree,
                 "FlowANN compact graph degree metadata is invalid");
    auto const cross_degree = static_cast<std::uint32_t>(cross_graph.extent(1) - 1);
    RAFT_EXPECTS(resident_degree + cross_degree == full_graph_degree,
                 "FlowANN compact graph degrees do not add up to the complete graph degree");
    if (resident_degree == 0) {
      RAFT_EXPECTS(inner_graph.extent(0) == 0 && inner_graph.extent(1) == 0,
                   "FlowANN zero-residency graph must not allocate packed rows");
    } else {
      auto const expected_rows =
        raft::div_rounding_up_safe<int64_t>(cross_graph.extent(0), node_per_cacheline_);
      auto const required_bits =
        static_cast<std::uint64_t>(node_per_cacheline_) * 8 +
        static_cast<std::uint64_t>(node_per_cacheline_) * resident_degree * n_bits_;
      RAFT_EXPECTS(inner_graph.extent(0) == expected_rows,
                   "FlowANN compact inner graph has an invalid row count");
      RAFT_EXPECTS(static_cast<std::uint64_t>(inner_graph.extent(1)) * 8 >= required_bits,
                   "FlowANN compact inner graph rows are too short");
    }
    if (dataset_.n_rows() > 0) {
      RAFT_EXPECTS(dataset_.n_rows() == cross_graph.extent(0),
                   "Dataset and FlowANN compact graph must have the same number of rows");
    }
    for (int64_t node = 0; node < cross_graph.extent(0); ++node) {
      RAFT_EXPECTS(cross_graph(node, 0) == cross_degree,
                   "FlowANN uniform compact graph has an invalid cross count at row %ld",
                   static_cast<long>(node));
      for (std::uint32_t edge = 0; edge < cross_degree; ++edge) {
        RAFT_EXPECTS(cross_graph(node, edge + 1) < cross_graph.extent(0),
                     "FlowANN compact cross graph contains an out-of-range node at row %ld",
                     static_cast<long>(node));
      }
    }

    inner_graph_       = std::move(inner_graph);
    cross_graph_       = std::move(cross_graph);
    full_graph_degree_ = full_graph_degree;
    resident_degree_   = resident_degree;
    rebuild_seed_cross_graph_(res);
  }

  void update_source_indices(
    raft::resources const& res,
    raft::host_vector_view<const IdxT, int64_t, raft::row_major> source_indices)
  {
    RAFT_EXPECTS(source_indices.extent(0) == static_cast<int64_t>(size()),
                 "FlowANN source-index mapping must have one entry per graph row");
    source_indices_.emplace(raft::make_device_vector<IdxT, int64_t>(res, source_indices.extent(0)));
    raft::copy(source_indices_->data_handle(),
               source_indices.data_handle(),
               source_indices.extent(0),
               raft::resource::get_cuda_stream(res));
  }

  void update_seeds(raft::resources const& res,
                    raft::host_vector_view<const IdxT, int64_t, raft::row_major> seeds)
  {
    RAFT_EXPECTS(seeds.extent(0) > 0, "FlowANN seeds must not be empty");
    RAFT_EXPECTS(seeds.extent(0) <= static_cast<int64_t>(size()),
                 "FlowANN seed count must not exceed the index size");
    for (int64_t i = 0; i < seeds.extent(0); ++i) {
      RAFT_EXPECTS(
        seeds(i) < size(), "FlowANN seed at position %ld is out of range", static_cast<long>(i));
    }
    seeds_.emplace(raft::make_device_vector<IdxT, int64_t>(res, seeds.extent(0)));
    raft::copy(seeds_->data_handle(),
               seeds.data_handle(),
               seeds.extent(0),
               raft::resource::get_cuda_stream(res));
    seed_ids_host_.assign(seeds.data_handle(), seeds.data_handle() + seeds.extent(0));
    rebuild_seed_lookup_(res);
    rebuild_seed_cross_graph_(res);
  }

  void update_subgraph_layout(
    raft::resources const& res,
    raft::host_vector_view<const std::uint32_t, int64_t, raft::row_major> offsets,
    raft::host_vector_view<const std::uint32_t, int64_t, raft::row_major> subgraph_ids)
  {
    RAFT_EXPECTS(offsets.extent(0) > 0, "FlowANN subgraph offsets must not be empty");
    RAFT_EXPECTS(offsets.extent(0) <= static_cast<int64_t>(size()),
                 "FlowANN subgraph count must not exceed the index size");
    RAFT_EXPECTS(subgraph_ids.extent(0) == static_cast<int64_t>(size()),
                 "FlowANN must store one subgraph identifier per graph row");
    RAFT_EXPECTS(offsets(0) == 0, "FlowANN first subgraph offset must be zero");
    for (int64_t i = 1; i < offsets.extent(0); ++i) {
      RAFT_EXPECTS(offsets(i - 1) < offsets(i),
                   "FlowANN subgraph offsets must be strictly increasing");
    }
    RAFT_EXPECTS(offsets(offsets.extent(0) - 1) < static_cast<std::uint32_t>(size()),
                 "FlowANN subgraph offsets exceed the index size");
    for (int64_t i = 0; i < subgraph_ids.extent(0); ++i) {
      RAFT_EXPECTS(subgraph_ids(i) < offsets.extent(0),
                   "FlowANN subgraph identifier is out of range at row %ld",
                   static_cast<long>(i));
    }

    subgraph_offsets_.emplace(
      raft::make_device_vector<std::uint32_t, int64_t>(res, offsets.extent(0)));
    raft::copy(subgraph_offsets_->data_handle(),
               offsets.data_handle(),
               offsets.extent(0),
               raft::resource::get_cuda_stream(res));
    subgraph_ids_.emplace(
      raft::make_device_vector<std::uint32_t, int64_t>(res, subgraph_ids.extent(0)));
    raft::copy(subgraph_ids_->data_handle(),
               subgraph_ids.data_handle(),
               subgraph_ids.extent(0),
               raft::resource::get_cuda_stream(res));
  }

 private:
  void validate_graph_(
    raft::host_matrix_view<const inner_index_type, int64_t, raft::row_major> inner_graph,
    raft::host_matrix_view<const graph_index_type, int64_t, raft::row_major> cross_graph,
    std::uint32_t full_graph_degree,
    std::uint32_t resident_degree) const
  {
    RAFT_EXPECTS(cross_graph.extent(1) > 0,
                 "FlowANN cross graph must include a neighbor-count column");
    RAFT_EXPECTS(static_cast<std::uint64_t>(cross_graph.extent(0)) <=
                   static_cast<std::uint64_t>(std::numeric_limits<IdxT>::max()),
                 "FlowANN graph row count exceeds the index type");
    auto const expected_inner_rows =
      raft::div_rounding_up_safe<int64_t>(cross_graph.extent(0), node_per_cacheline_);
    auto const has_inner_graph = inner_graph.extent(0) != 0 || inner_graph.extent(1) != 0;
    if (has_inner_graph) {
      RAFT_EXPECTS(inner_graph.extent(0) == expected_inner_rows,
                   "FlowANN inner graph must contain one packed row per node cacheline");
      RAFT_EXPECTS(inner_graph.extent(1) >= node_per_cacheline_,
                   "FlowANN inner graph rows are too short to contain neighbor counts");
    } else {
      RAFT_EXPECTS(resident_degree == 0,
                   "FlowANN may omit the packed graph only when resident degree is zero");
    }
    RAFT_EXPECTS(full_graph_degree > 0, "FlowANN complete graph degree must be positive");
    RAFT_EXPECTS(resident_degree <= full_graph_degree,
                 "FlowANN resident graph degree exceeds the complete graph degree");
    auto const cross_degree = static_cast<std::uint32_t>(cross_graph.extent(1) - 1);
    RAFT_EXPECTS(cross_degree <= full_graph_degree,
                 "FlowANN cross graph degree exceeds the complete graph degree");
    if (resident_degree > 0 || cross_degree < full_graph_degree) {
      RAFT_EXPECTS(resident_degree + cross_degree == full_graph_degree,
                   "FlowANN compact graph degrees do not add up to the complete graph degree");
    }
    if (dataset_.n_rows() > 0) {
      RAFT_EXPECTS(dataset_.n_rows() == cross_graph.extent(0),
                   "Dataset and FlowANN cross graph must have the same number of rows");
    }

    for (int64_t node = 0; node < cross_graph.extent(0); ++node) {
      auto const cross_count = cross_graph(node, 0);
      RAFT_EXPECTS(cross_count <= cross_degree,
                   "FlowANN cross-neighbor count exceeds graph degree at row %ld",
                   static_cast<long>(node));
      for (std::uint32_t edge = 0; edge < cross_count; ++edge) {
        RAFT_EXPECTS(cross_graph(node, edge + 1) < cross_graph.extent(0),
                     "FlowANN cross graph contains an out-of-range node at row %ld",
                     static_cast<long>(node));
      }

      auto const packed_row = node / node_per_cacheline_;
      auto const packed_col = node % node_per_cacheline_;
      auto const inner_count =
        has_inner_graph ? static_cast<std::uint32_t>(inner_graph(packed_row, packed_col)) : 0u;
      RAFT_EXPECTS(inner_count + cross_count <= full_graph_degree,
                   "FlowANN inner and cross neighbors exceed graph degree at row %ld",
                   static_cast<long>(node));
    }
    for (int64_t packed_row = 0; packed_row < inner_graph.extent(0); ++packed_row) {
      std::uint64_t payload_bits = static_cast<std::uint64_t>(node_per_cacheline_) * 8;
      for (std::uint16_t packed_col = 0; packed_col < node_per_cacheline_; ++packed_col) {
        auto const node = packed_row * node_per_cacheline_ + packed_col;
        if (node < cross_graph.extent(0)) {
          payload_bits += static_cast<std::uint64_t>(inner_graph(packed_row, packed_col)) * n_bits_;
        }
      }
      RAFT_EXPECTS(payload_bits <= static_cast<std::uint64_t>(inner_graph.extent(1)) * 8,
                   "FlowANN inner graph payload exceeds its packed row at row %ld",
                   static_cast<long>(packed_row));
    }
  }

  static auto seed_hash_(IdxT key, std::uint32_t capacity) -> std::uint32_t
  {
    std::uint64_t value = static_cast<std::uint64_t>(key);
    value ^= value >> 16;
    value *= 0x85ebca77u;
    value ^= value >> 13;
    value *= 0xc2b2ae3du;
    value ^= value >> 16;
    return static_cast<std::uint32_t>(value % capacity);
  }

  void rebuild_seed_lookup_(raft::resources const& res)
  {
    if (seed_ids_host_.empty()) {
      seed_lookup_keys_.reset();
      seed_lookup_values_.reset();
      return;
    }
    RAFT_EXPECTS(seed_ids_host_.size() <=
                   (static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) - 1) / 2,
                 "FlowANN seed lookup exceeds uint32 capacity");

    auto const capacity    = static_cast<std::uint32_t>(seed_ids_host_.size() * 2 + 1);
    auto const empty_key   = std::numeric_limits<IdxT>::max();
    auto const empty_value = std::numeric_limits<std::uint32_t>::max();
    std::vector<IdxT> host_keys(capacity, empty_key);
    std::vector<std::uint32_t> host_values(capacity, empty_value);

    for (std::uint32_t seed = 0; seed < seed_ids_host_.size(); ++seed) {
      auto const key = seed_ids_host_[seed];
      auto slot      = seed_hash_(key, capacity);
      while (host_keys[slot] != empty_key) {
        slot = (slot + 1) % capacity;
      }
      host_keys[slot]   = key;
      host_values[slot] = seed;
    }

    seed_lookup_keys_.emplace(raft::make_device_vector<IdxT, int64_t>(res, capacity));
    seed_lookup_values_.emplace(raft::make_device_vector<std::uint32_t, int64_t>(res, capacity));
    auto const stream = raft::resource::get_cuda_stream(res);
    raft::copy(seed_lookup_keys_->data_handle(), host_keys.data(), capacity, stream);
    raft::copy(seed_lookup_values_->data_handle(), host_values.data(), capacity, stream);
  }

  void rebuild_seed_cross_graph_(raft::resources const& res)
  {
    if (seed_ids_host_.empty()) {
      seed_cross_graph_.reset();
      return;
    }
    RAFT_EXPECTS(cross_graph_.extent(0) > 0 && cross_graph_.extent(1) > 0,
                 "FlowANN seed caching requires a non-empty cross graph");

    auto host_cache = raft::make_host_matrix<graph_index_type, int64_t>(
      static_cast<int64_t>(seed_ids_host_.size()), cross_graph_.extent(1));
    for (std::size_t seed = 0; seed < seed_ids_host_.size(); ++seed) {
      auto const node = seed_ids_host_[seed];
      RAFT_EXPECTS(node < size(), "FlowANN seed at position %zu is out of range", seed);
      std::copy_n(
        cross_graph_.data_handle() + static_cast<std::size_t>(node) * cross_graph_.extent(1),
        cross_graph_.extent(1),
        host_cache.data_handle() + seed * cross_graph_.extent(1));
    }

    seed_cross_graph_.emplace(raft::make_device_matrix<graph_index_type, int64_t>(
      res, host_cache.extent(0), host_cache.extent(1)));
    raft::copy(seed_cross_graph_->data_handle(),
               host_cache.data_handle(),
               host_cache.size(),
               raft::resource::get_cuda_stream(res));
  }

  void validate_encoding_() const
  {
    RAFT_EXPECTS(node_per_cacheline_ > 0, "FlowANN node_per_cacheline must be positive");
    RAFT_EXPECTS(n_bits_ > 0 && n_bits_ <= 32, "FlowANN n_bits must be in [1, 32]");
  }

  CUVS_EXPORT void compute_dataset_norms_(raft::resources const& res);

  cuvs::distance::DistanceType metric_;
  std::uint16_t node_per_cacheline_;
  std::uint16_t n_bits_;
  std::uint32_t full_graph_degree_ = 0;
  std::uint32_t resident_degree_   = 0;
  raft::device_matrix<inner_index_type, int64_t, raft::row_major> inner_graph_;
  raft::host_matrix<graph_index_type, int64_t, raft::row_major> cross_graph_;
  DatasetViewT dataset_{};
  std::optional<raft::device_vector<float, int64_t>> dataset_norms_;
  std::optional<raft::device_vector<IdxT, int64_t>> source_indices_;
  std::optional<raft::device_vector<std::uint32_t, int64_t>> subgraph_offsets_;
  std::optional<raft::device_vector<std::uint32_t, int64_t>> subgraph_ids_;
  std::optional<raft::device_vector<IdxT, int64_t>> seeds_;
  std::vector<IdxT> seed_ids_host_;
  std::optional<raft::device_vector<IdxT, int64_t>> seed_lookup_keys_;
  std::optional<raft::device_vector<std::uint32_t, int64_t>> seed_lookup_values_;
  std::optional<raft::device_matrix<graph_index_type, int64_t, raft::row_major>> seed_cross_graph_;
};

template <typename T, typename IdxT = std::uint32_t>
using device_padded_index = index<T, IdxT, cuvs::neighbors::device_padded_dataset_view<T, int64_t>>;

template <typename T, typename IdxT = std::uint32_t>
using vpq_f16_index = index<T, IdxT, cuvs::neighbors::device_vpq_dataset_view<half, int64_t>>;

/** Owns the dataset backing storage required by a FlowANN index. */
template <typename DatasetT, typename IndexT>
struct index_bundle {
  std::unique_ptr<DatasetT> dataset;
  IndexT index;
};

template <typename T, typename IdxT = std::uint32_t>
using device_padded_index_bundle = index_bundle<cuvs::neighbors::device_padded_dataset<T, int64_t>,
                                                flowann::device_padded_index<T, IdxT>>;

template <typename T, typename IdxT = std::uint32_t>
using vpq_f16_index_bundle =
  index_bundle<cuvs::neighbors::device_vpq_dataset<half, int64_t>, flowann::vpq_f16_index<T, IdxT>>;

/**
 * Search a FlowANN index. The overload set matches the data and output-index types supported by
 * CAGRA search. The graph and source-index mapping remain uint32_t internally.
 */
#define CUVS_FLOWANN_DECLARE_SEARCH(T, DatasetAlias, OutputIdxT)              \
  CUVS_EXPORT void search(                                                    \
    raft::resources const& res,                                               \
    search_params const& params,                                              \
    DatasetAlias<T, std::uint32_t> const& index,                              \
    search_context& context,                                                  \
    raft::device_matrix_view<const T, int64_t, raft::row_major> queries,      \
    raft::device_matrix_view<OutputIdxT, int64_t, raft::row_major> neighbors, \
    raft::device_matrix_view<float, int64_t, raft::row_major> distances,      \
    cuvs::neighbors::filtering::base_filter const& sample_filter =            \
      cuvs::neighbors::filtering::none_sample_filter{})

#define CUVS_FLOWANN_DECLARE_SEARCH_FOR_TYPE(T)                       \
  CUVS_FLOWANN_DECLARE_SEARCH(T, device_padded_index, std::uint32_t); \
  CUVS_FLOWANN_DECLARE_SEARCH(T, device_padded_index, std::int64_t);  \
  CUVS_FLOWANN_DECLARE_SEARCH(T, vpq_f16_index, std::uint32_t);       \
  CUVS_FLOWANN_DECLARE_SEARCH(T, vpq_f16_index, std::int64_t)

CUVS_FLOWANN_DECLARE_SEARCH_FOR_TYPE(float);
CUVS_FLOWANN_DECLARE_SEARCH_FOR_TYPE(half);
CUVS_FLOWANN_DECLARE_SEARCH_FOR_TYPE(std::int8_t);
CUVS_FLOWANN_DECLARE_SEARCH_FOR_TYPE(std::uint8_t);

#undef CUVS_FLOWANN_DECLARE_SEARCH_FOR_TYPE
#undef CUVS_FLOWANN_DECLARE_SEARCH

}  // namespace cuvs::neighbors::cagra::experimental::flowann

#endif  // CUVS_ENABLE_FLOWANN_SEARCH
