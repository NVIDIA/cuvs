/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/common.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/distance/distance.hpp>

#include <cuvs/core/cuda_fp16.hpp>
#include <cuvs/core/export.hpp>

#include <utility>

namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace nn_descent {
/**
 * @defgroup nn_descent_cpp_index_params The nn-descent algorithm parameters.
 * @{
 */

/**
 * @brief Dtype to use for distance computation
 * - `AUTO`: Automatically determine the best dtype for distance computation based on the dataset
 * dimensions.
 * - `FP32`: Use fp32 distance computation for better precision at the cost of performance and
 * memory usage.
 * - `FP16`: Use fp16 distance computation.
 */
enum class DIST_COMP_DTYPE { AUTO = 0, FP32 = 1, FP16 = 2 };

/**
 * @brief Parameters used to build an nn-descent index
 * - `graph_degree`: For an input dataset of dimensions (N, D),
 * determines the final dimensions of the all-neighbors knn graph
 * which turns out to be of dimensions (N, graph_degree)
 * - `intermediate_graph_degree`: Internally, nn-descent builds an
 * all-neighbors knn graph of dimensions (N, intermediate_graph_degree)
 * before selecting the final `graph_degree` neighbors. It's recommended
 * that `intermediate_graph_degree` >= 1.5 * graph_degree
 * - `max_iterations`: The number of iterations that nn-descent will refine
 * the graph for. More iterations produce a better quality graph at cost of performance
 * - `termination_threshold`: The delta at which nn-descent will terminate its iterations
 * - `return_distances`: Boolean to decide whether to return distances array
 * - `dist_comp_dtype`: dtype to use for distance computation. Defaults to `AUTO` which
 * automatically determines the best dtype for distance computation based on the dataset dimensions.
 * Use `FP32` for better precision at the cost of performance and memory usage. This option is only
 * valid when data type is fp32. Use `FP16` for better performance and memory usage at the cost of
 * precision.
 */
struct index_params : cuvs::neighbors::index_params {
  size_t graph_degree              = 64;
  size_t intermediate_graph_degree = 128;
  size_t max_iterations            = 20;
  float termination_threshold      = 0.0001;
  bool return_distances            = true;
  DIST_COMP_DTYPE dist_comp_dtype  = DIST_COMP_DTYPE::AUTO;

  /** @brief Construct NN descent parameters for a specific kNN graph degree
   *
   * @param graph_degree output graph degree
   * @param metric distance metric to use
   */
  index_params(size_t graph_degree                 = 64,
               cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded);
};

/**
 * @}
 */

/**
 * @defgroup nn_descent_cpp_index nn-descent index
 * @{
 */
/**
 * @brief nn-descent Build an nn-descent index
 * The index contains an all-neighbors graph of the input dataset
 * stored in host memory of dimensions (n_rows, n_cols)
 *
 * Reference:
 * Hui Wang, Wan-Lei Zhao, Xiangxiang Zeng, and Jianye Yang. 2021.
 * Fast k-NN Graph Construction by GPU based NN-Descent. In Proceedings of the 30th ACM
 * International Conference on Information & Knowledge Management (CIKM '21). Association for
 * Computing Machinery, New York, NY, USA, 1929–1938. https://doi.org/10.1145/3459637.3482344
 *
 * @tparam IdxT dtype to be used for constructing knn-graph
 */
template <typename IdxT>
struct index : cuvs::neighbors::index {
 public:
  /**
   * @brief Construct a new index object
   *
   * This constructor creates an nn-descent index which is a knn-graph in host memory.
   * The type of the knn-graph is a dense raft::host_matrix and dimensions are
   * (n_rows, n_cols).
   *
   * @param res raft::resources is an object managing resources
   * @param n_rows number of rows in knn-graph
   * @param n_cols number of cols in knn-graph
   * @param return_distances whether to return distances
   * @param metric distance metric to use
   */
  index(raft::resources const& res,
        int64_t n_rows,
        int64_t n_cols,
        bool return_distances               = false,
        cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded)
    : cuvs::neighbors::index(),
      res_{res},
      metric_{metric},
      graph_{raft::make_host_matrix<IdxT, int64_t, raft::row_major>(n_rows, n_cols)},
      graph_view_{graph_.view()},
      return_distances_{return_distances}
  {
    if (return_distances) {
      distances_      = raft::make_device_matrix<float, int64_t>(res_, n_rows, n_cols);
      distances_view_ = distances_.value().view();
    }
  }

  /**
   * @brief Construct a new index object
   *
   * This constructor creates an nn-descent index using a user allocated host memory knn-graph.
   * The type of the knn-graph is a dense raft::host_matrix and dimensions are
   * (n_rows, n_cols).
   *
   * @param res raft::resources is an object managing resources
   * @param graph_view raft::host_matrix_view<IdxT, int64_t, raft::row_major> for storing knn-graph
   * @param distances_view optional raft::device_matrix_view<float, int64_t, row_major> for storing
   * distances
   * @param metric distance metric to use
   */
  index(raft::resources const& res,
        raft::host_matrix_view<IdxT, int64_t, raft::row_major> graph_view,
        std::optional<raft::device_matrix_view<float, int64_t, row_major>> distances_view =
          std::nullopt,
        cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded)
    : cuvs::neighbors::index(),
      res_{res},
      metric_{metric},
      graph_{raft::make_host_matrix<IdxT, int64_t, raft::row_major>(0, 0)},
      graph_view_{graph_view},
      distances_view_{distances_view},
      return_distances_{distances_view.has_value()}
  {
  }

  /** Distance metric used for clustering. */
  [[nodiscard]] constexpr inline auto metric() const noexcept -> cuvs::distance::DistanceType
  {
    return metric_;
  }

  // /** Total length of the index (number of vectors). */
  [[nodiscard]] constexpr inline auto size() const noexcept -> IdxT
  {
    return graph_view_.extent(0);
  }

  /** Graph degree */
  [[nodiscard]] constexpr inline auto graph_degree() const noexcept -> uint32_t
  {
    return graph_view_.extent(1);
  }

  /** neighborhood graph [size, graph-degree] */
  [[nodiscard]] inline auto graph() noexcept
    -> raft::host_matrix_view<IdxT, int64_t, raft::row_major>
  {
    return graph_view_;
  }

  /** neighborhood graph distances [size, graph-degree] */
  [[nodiscard]] inline auto distances() noexcept
    -> std::optional<device_matrix_view<float, int64_t, row_major>>
  {
    return distances_view_;
  }

  // Don't allow copying the index for performance reasons (try avoiding copying data)
  index(const index&)                    = delete;
  index(index&&)                         = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index&      = default;
  ~index()                               = default;

 private:
  raft::resources const& res_;
  cuvs::distance::DistanceType metric_;
  raft::host_matrix<IdxT, int64_t, raft::row_major> graph_;  // graph to return for non-int IdxT
  std::optional<raft::device_matrix<float, int64_t, row_major>> distances_;
  raft::host_matrix_view<IdxT, int64_t, raft::row_major>
    graph_view_;  // view of graph for user provided matrix
  std::optional<raft::device_matrix_view<float, int64_t, row_major>> distances_view_;
  bool return_distances_;
};

/** @} */

/**
 * @defgroup nn_descent_cpp_index_build nn-descent index build
 * @{
 */

/**
 * @brief Build nn-descent Index with dataset in device memory
 *
 * The following distance metrics are supported:
 * - L2Expanded
 * - L2SqrtExpanded
 * - CosineExpanded
 * - InnerProduct
 * - L1
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   nn_descent::index_params index_params;
 *   // create and fill the index from a [N, D] raft::device_matrix_view dataset
 *   auto index = nn_descent::build(res, index_params, dataset);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset
 * @endcode
 *
 * @param[in] res raft::resources is an object managing resources
 * @param[in] params an instance of nn_descent::index_params that are parameters
 *               to run the nn-descent algorithm
 * @param[in] dataset raft::device_matrix_view input dataset expected to be located
 *                in device memory
 * @param[in] graph optional raft::host_matrix_view<uint32_t, int64_t, raft::row_major> for owning
 * the output graph
 * @return index<IdxT> index containing all-neighbors knn graph in host memory
 */
auto build(raft::resources const& res,
           index_params const& params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
           std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::row_major>> graph =
             std::nullopt) -> cuvs::neighbors::nn_descent::index<uint32_t>;

/**
 * @brief Build nn-descent Index with dataset in host memory
 *
 * The following distance metrics are supported:
 * - L2Expanded
 * - L2SqrtExpanded
 * - CosineExpanded
 * - InnerProduct
 * - L1
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   nn_descent::index_params index_params;
 *   // create and fill the index from a [N, D] raft::host_matrix_view dataset
 *   auto index = nn_descent::build(res, index_params, dataset);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset
 * @endcode
 *
 * @tparam T data-type of the input dataset
 * @tparam IdxT data-type for the output index
 * @param res raft::resources is an object managing resources
 * @param[in] params an instance of nn_descent::index_params that are parameters
 *               to run the nn-descent algorithm
 * @param[in] dataset raft::host_matrix_view input dataset expected to be located
 *                in host memory
 * @param[in] graph optional raft::host_matrix_view<uint32_t, int64_t, raft::row_major> for owning
 * the output graph
 * @return index<IdxT> index containing all-neighbors knn graph in host memory
 */
auto build(raft::resources const& res,
           index_params const& params,
           raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
           std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::row_major>> graph =
             std::nullopt) -> cuvs::neighbors::nn_descent::index<uint32_t>;

/**
 * @brief Build nn-descent Index with dataset in device memory
 *
 * The following distance metrics are supported:
 * - L2Expanded
 * - L2SqrtExpanded
 * - CosineExpanded
 * - InnerProduct
 * - L1
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   nn_descent::index_params index_params;
 *   // create and fill the index from a [N, D] raft::device_matrix_view dataset
 *   auto index = nn_descent::build(res, index_params, dataset);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset
 * @endcode
 *
 * @param[in] res raft::resources is an object managing resources
 * @param[in] params an instance of nn_descent::index_params that are parameters
 *               to run the nn-descent algorithm
 * @param[in] dataset raft::device_matrix_view input dataset expected to be located
 *                in device memory
 * @param[in] graph optional raft::host_matrix_view<uint32_t, int64_t, raft::row_major> for owning
 * the output graph
 * @return index<IdxT> index containing all-neighbors knn graph in host memory
 */
auto build(raft::resources const& res,
           index_params const& params,
           raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
           std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::row_major>> graph =
             std::nullopt) -> cuvs::neighbors::nn_descent::index<uint32_t>;

/**
 * @brief Build nn-descent Index with dataset in host memory
 *
 * The following distance metrics are supported:
 * - L2Expanded
 * - L2SqrtExpanded
 * - CosineExpanded
 * - InnerProduct
 * - L1
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   nn_descent::index_params index_params;
 *   // create and fill the index from a [N, D] raft::host_matrix_view dataset
 *   auto index = nn_descent::build(res, index_params, dataset);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset
 * @endcode
 *
 * @tparam T data-type of the input dataset
 * @tparam IdxT data-type for the output index
 * @param res raft::resources is an object managing resources
 * @param[in] params an instance of nn_descent::index_params that are parameters
 *               to run the nn-descent algorithm
 * @param[in] dataset raft::host_matrix_view input dataset expected to be located
 *                in host memory
 * @param[in] graph optional raft::host_matrix_view<uint32_t, int64_t, raft::row_major> for owning
 * the output graph
 * @return index<IdxT> index containing all-neighbors knn graph in host memory
 */
auto build(raft::resources const& res,
           index_params const& params,
           raft::host_matrix_view<const half, int64_t, raft::row_major> dataset,
           std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::row_major>> graph =
             std::nullopt) -> cuvs::neighbors::nn_descent::index<uint32_t>;

/**
 * @brief Build nn-descent Index with dataset in device memory
 *
 * The following distance metrics are supported:
 * - L2Expanded
 * - L2SqrtExpanded
 * - CosineExpanded
 * - InnerProduct
 * - L1
 * - BitwiseHamming
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   nn_descent::index_params index_params;
 *   // create and fill the index from a [N, D] raft::device_matrix_view dataset
 *   auto index = nn_descent::build(res, index_params, dataset);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset
 * @endcode
 *
 * @param[in] res raft::resources is an object managing resources
 * @param[in] params an instance of nn_descent::index_params that are parameters
 *               to run the nn-descent algorithm
 * @param[in] dataset raft::device_matrix_view input dataset expected to be located
 *                in device memory
 * @param[in] graph optional raft::host_matrix_view<uint32_t, int64_t, raft::row_major> for owning
 * the output graph
 * @return index<IdxT> index containing all-neighbors knn graph in host memory
 */
auto build(raft::resources const& res,
           index_params const& params,
           raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
           std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::row_major>> graph =
             std::nullopt) -> cuvs::neighbors::nn_descent::index<uint32_t>;

/**
 * @brief Build nn-descent Index with dataset in host memory
 *
 * The following distance metrics are supported:
 * - L2Expanded
 * - L2SqrtExpanded
 * - CosineExpanded
 * - InnerProduct
 * - L1
 * - BitwiseHamming
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   nn_descent::index_params index_params;
 *   // create and fill the index from a [N, D] raft::host_matrix_view dataset
 *   auto index = nn_descent::build(res, index_params, dataset);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset
 * @endcode
 *
 * @tparam T data-type of the input dataset
 * @tparam IdxT data-type for the output index
 * @param res raft::resources is an object managing resources
 * @param[in] params an instance of nn_descent::index_params that are parameters
 *               to run the nn-descent algorithm
 * @param[in] dataset raft::host_matrix_view input dataset expected to be located
 *                in host memory
 * @param[in] graph optional raft::host_matrix_view<uint32_t, int64_t, raft::row_major> for owning
 * the output graph
 * @return index<IdxT> index containing all-neighbors knn graph in host memory
 */
auto build(raft::resources const& res,
           index_params const& params,
           raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
           std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::row_major>> graph =
             std::nullopt) -> cuvs::neighbors::nn_descent::index<uint32_t>;

/**
 * @brief Build nn-descent Index with dataset in device memory
 *
 * The following distance metrics are supported:
 * - L2Expanded
 * - L2SqrtExpanded
 * - CosineExpanded
 * - InnerProduct
 * - L1
 * - BitwiseHamming
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   nn_descent::index_params index_params;
 *   // create and fill the index from a [N, D] raft::device_matrix_view dataset
 *   auto index = nn_descent::build(res, index_params, dataset);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset
 * @endcode
 *
 * @param[in] res raft::resources is an object managing resources
 * @param[in] params an instance of nn_descent::index_params that are parameters
 *               to run the nn-descent algorithm
 * @param[in] dataset raft::device_matrix_view input dataset expected to be located
 *                in device memory
 * @param[in] graph optional raft::host_matrix_view<uint32_t, int64_t, raft::row_major> for owning
 * the output graph
 * @return index<IdxT> index containing all-neighbors knn graph in host memory
 */
auto build(raft::resources const& res,
           index_params const& params,
           raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
           std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::row_major>> graph =
             std::nullopt) -> cuvs::neighbors::nn_descent::index<uint32_t>;

/**
 * @brief Build nn-descent Index with dataset in host memory
 *
 * The following distance metrics are supported:
 * - L2Expanded
 * - L2SqrtExpanded
 * - CosineExpanded
 * - InnerProduct
 * - L1
 * - BitwiseHamming
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   nn_descent::index_params index_params;
 *   // create and fill the index from a [N, D] raft::host_matrix_view dataset
 *   auto index = nn_descent::build(res, index_params, dataset);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset
 * @endcode
 *
 * @tparam T data-type of the input dataset
 * @tparam IdxT data-type for the output index
 * @param res raft::resources is an object managing resources
 * @param[in] params an instance of nn_descent::index_params that are parameters
 *               to run the nn-descent algorithm
 * @param[in] dataset raft::host_matrix_view input dataset expected to be located
 *                in host memory
 * @param[in] graph optional raft::host_matrix_view<uint32_t, int64_t, raft::row_major> for owning
 * the output graph
 * @return index<IdxT> index containing all-neighbors knn graph in host memory
 */
auto build(raft::resources const& res,
           index_params const& params,
           raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
           std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::row_major>> graph =
             std::nullopt) -> cuvs::neighbors::nn_descent::index<uint32_t>;

/** @} */
/**
 * @brief Test if we have enough GPU memory to run NN descent algorithm.
 *
 * @param res
 * @param dataset shape of the dataset
 * @param idx_size the size of index type in bytes
 * @return true if enough GPU memory can be allocated
 * @return false otherwise
 */
bool has_enough_device_memory(raft::resources const& res,
                              raft::matrix_extent<int64_t> dataset,
                              size_t idx_size = 4);

/**
 * @brief Estimate the host and device memory (in bytes) required to build an
 * all-neighbors kNN graph with NN-descent for a dataset of the given shape.
 *
 * The estimate mirrors the persistent allocations performed by the GNND solver.
 * It does not include the input dataset nor the output graph.
 *
 * @param res
 * @param dataset shape of the dataset
 * @param graph_degree the degree of the all-neighbors graph
 * @param idx_size the size of the graph index type in bytes
 * @return std::pair<host_bytes, device_bytes>
 */
std::pair<size_t, size_t> build_mem_usage(raft::resources const& res,
                                          raft::matrix_extent<int64_t> dataset,
                                          size_t graph_degree,
                                          size_t idx_size = 4);

}  // namespace nn_descent
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace graph_build_params {
using nn_descent_params = cuvs::neighbors::nn_descent::index_params;
}
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
