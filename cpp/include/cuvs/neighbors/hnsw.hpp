/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifdef CUVS_BUILD_CAGRA_HNSWLIB

#pragma once

#include "common.hpp"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include "cagra.hpp"
#include <raft/core/host_mdspan.hpp>

#include <sys/types.h>

#include <cstdint>
#include <cuvs/core/export.hpp>
#include <memory>
#include <type_traits>
#include <variant>

namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace hnsw {

// Re-export graph_build_params into hnsw namespace for convenience
namespace graph_build_params = cuvs::neighbors::graph_build_params;

/**
 * @defgroup hnsw_cpp_index_params hnswlib index wrapper params
 * @{
 */

/**
 * @brief Hierarchy for HNSW index when converting from CAGRA index
 *
 * NOTE: When the value is `NONE`, the HNSW index is built as a base-layer-only index.
 */
enum class HnswHierarchy {
  NONE,  // base-layer-only index
  CPU,   // full index with CPU-built hierarchy
  GPU    // full index with GPU-built hierarchy
};

struct index_params : cuvs::neighbors::index_params {
  /** Hierarchy build type for HNSW index when converting from CAGRA index */
  HnswHierarchy hierarchy = HnswHierarchy::GPU;
  /** Size of the candidate list during hierarchy construction when hierarchy is `CPU`*/
  int ef_construction = 200;
  /** Number of host threads to use to construct hierarchy when hierarchy is `CPU` or `GPU`.
      When the value is 0, the number of threads is automatically determined to the
      maximum number of threads available.
      NOTE: When hierarchy is `GPU`, while the majority of the work is done on the GPU,
      initialization of the HNSW index itself and some other work
      is parallelized with the help of CPU threads.
   */
  int num_threads = 0;

  /** HNSW M parameter: number of bi-directional links per node (used when building with ACE).
   */
  size_t M = 32;

  /** Parameters to fine tune GPU graph building. By default we select the parameters based on
   * dataset shape and HNSW build parameters. You can override these parameters to fine tune the
   * graph building process as described in the CAGRA build docs.
   *
   * Set ace_params to configure ACE (Augmented Core Extraction) parameters for building
   * a GPU-accelerated HNSW index. ACE enables building indexes for datasets too large
   * to fit in GPU memory.
   *
   * @code{.cpp}
   * hnsw::index_params params;
   * params.M = 32;
   * params.ef_construction = 120;
   * // Configure ACE parameters
   * params.graph_build_params = hnsw::graph_build_params::ace_params();
   * auto& ace = std::get<hnsw::graph_build_params::ace_params>(params.graph_build_params);
   * ace.npartitions = 4;
   * ace.use_disk = true;
   * ace.build_dir = "/tmp/hnsw_ace_build";
   * @endcode
   */
  std::variant<std::monostate, graph_build_params::ace_params> graph_build_params;
};

/**
 * @brief Create a CAGRA index parameters compatible with HNSW index
 *
 * @param dataset The shape of the input dataset.
 * @param M HNSW index parameter M (graph degree = 2*M).
 * @param ef_construction HNSW index parameter ef_construction.
 * @param metric The distance metric to search.
 *
 *
 * * IMPORTANT NOTE *
 *
 * The reference HNSW index and the corresponding from-CAGRA generated HNSW index will NOT produce
 * the same recalls and QPS for the same parameter `ef`. The graphs are different internally. For
 * the same `ef`, the from-CAGRA index likely has a slightly higher recall and slightly lower QPS.
 * However, the Recall-QPS curves should be similar (i.e. the points are just shifted along the
 * curve).
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   raft::resources res;
 *   auto dataset = raft::make_device_matrix<float, int64_t>(res, N, D);
 *   auto cagra_params = to_cagra_params(dataset.extents(), M, efc);
 *   auto cagra_index = cagra::build(res, cagra_params, dataset);
 *   auto hnsw_index = hnsw::from_cagra(res, hnsw_params, cagra_index);
 * @endcode
 */
[[deprecated("Use cagra::index_params::from_hnsw_params instead")]]
cuvs::neighbors::cagra::index_params to_cagra_params(
  raft::matrix_extent<int64_t> dataset,
  int M,
  int ef_construction,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded);

/**
 * @}
 */

/**
 * @defgroup hnsw_cpp_index hnswlib index wrapper
 * @{
 */

template <typename T>
struct index : cuvs::neighbors::index {
 public:
  /**
   * @brief load a base-layer-only hnswlib index originally saved from a built CAGRA index.
   *  This is a virtual class and it cannot be used directly. To create an index, use the factory
   *  function `cuvs::neighbors::hnsw::from_cagra` from the header
   *  `cuvs/neighbors/hnsw.hpp`
   *
   * @param[in] dim dimensions of the training dataset
   * @param[in] metric distance metric to search. Supported metrics ("L2Expanded", "InnerProduct")
   * @param[in] hierarchy hierarchy used for upper HNSW layers
   */
  index(int dim, cuvs::distance::DistanceType metric, HnswHierarchy hierarchy = HnswHierarchy::NONE)
    : dim_{dim}, metric_{metric}, hierarchy_{hierarchy}
  {
  }

  virtual ~index() {}

  /**
  @brief Get underlying index
  */
  virtual void const* get_index() const = 0;

  auto dim() const -> int const { return dim_; }

  auto metric() const -> cuvs::distance::DistanceType { return metric_; }

  auto hierarchy() const -> HnswHierarchy { return hierarchy_; }

  /**
  @brief Set ef for search
  */
  virtual void set_ef(int ef) const = 0;

  /**
  @brief Get file path for disk-backed index
  */
  virtual std::string file_path() const { return ""; }

 private:
  int dim_;
  cuvs::distance::DistanceType metric_;
  HnswHierarchy hierarchy_;
};

/**
 * @}
 */

/**
 * @defgroup hnsw_cpp_extend_params HNSW index extend parameters
 * @{
 */

struct extend_params {
  /** Number of host threads to use to add additional vectors to the index.
  Value of 0 automatically maximizes parallelism. */
  int num_threads = 0;
};

/**
 * @}
 */

/**
 * @defgroup hnsw_cpp_index_build Build HNSW index on the GPU
 * @{
 */

/**
 * @brief Build an HNSW index on the GPU
 *
 * The resulting graph is compatible for HNSW search, but is not an exact equivalent of the graph
 * built by the HNSW.
 *
 * The HNSW index construction parameters `M` and `ef_construction` are the main parameters to
 * control the graph degree and graph quality.  We have additional options that can be used to fine
 * tune graph building on the GPU (see `cuvs::neighbors::cagra::index_params`). In case the index
 * does not fit the host or GPU memory,  we would use disk as temporary storage. In such cases it is
 * important to set `ace_params.build_dir` to a fast disk with sufficient storage size.
 *
 * NOTE: This function requires CUDA headers to be available at compile time.
 *
 * @param[in] res raft resources
 * @param[in] params hnsw index parameters including ACE configuration
 * @param[in] dataset a host matrix view to a row-major matrix [n_rows, dim]
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   raft::resources res;
 *
 *   // Create index parameters with ACE configuration
 *   hnsw::index_params params;
 *   params.metric = cuvs::distance::DistanceType::L2Expanded;
 *   params.hierarchy = hnsw::HnswHierarchy::GPU;
 *   params.M = 32;
 *   params.ef_construction = 120;
 *
 *   // Configure GPU graph building parameters
 *   auto ace_params = hnsw::graph_build_params::ace_params();
 *   ace_params.npartitions = 4;
 *   ace_params.use_disk = true;
 *   ace_params.build_dir = "/tmp/hnsw_ace_build";
 *   params.graph_build_params = ace_params;
 *
 *   // Build the index
 *   auto dataset = raft::make_host_matrix<float, int64_t>(res, N, D);
 *   // ... fill dataset ...
 *   auto hnsw_index = hnsw::build(res, params, raft::make_const_mdspan(dataset.view()));
 *
 *   // Search the index
 *   hnsw::search_params search_params;
 *   search_params.ef = 200;
 *   auto neighbors = raft::make_host_matrix<uint64_t, int64_t>(res, n_queries, k);
 *   auto distances = raft::make_host_matrix<float, int64_t>(res, n_queries, k);
 *   hnsw::search(res, search_params, *hnsw_index, queries, neighbors.view(), distances.view());
 *
 *   // Serialize the index
 *   hnsw::serialize(res, "index.bin", *hnsw_index);
 * @endcode
 */
std::unique_ptr<index<float>> build(
  raft::resources const& res,
  const index_params& params,
  raft::host_matrix_view<const float, int64_t, raft::row_major> dataset);

/**
 * @brief Build an HNSW index on the GPU
 *
 * The resulting graph is compatible for HNSW search, but is not an exact equivalent of the graph
 * built by the HNSW.
 *
 * The HNSW index construction parameters `M` and `ef_construction` are the main parameters to
 * control the graph degree and graph quality.  We have additional options that can be used to fine
 * tune graph building on the GPU (see `cuvs::neighbors::cagra::index_params`). In case the index
 * does not fit the host or GPU memory,  we would use disk as temporary storage. In such cases it is
 * important to set `ace_params.build_dir` to a fast disk with sufficient storage size.
 *
 * NOTE: This function requires CUDA headers to be available at compile time.
 *
 * @param[in] res raft resources
 * @param[in] params hnsw index parameters including ACE configuration
 * @param[in] dataset a host matrix view to a row-major matrix [n_rows, dim]
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   raft::resources res;
 *
 *   // Create index parameters with ACE configuration
 *   hnsw::index_params params;
 *   params.metric = cuvs::distance::DistanceType::L2Expanded;
 *   params.hierarchy = hnsw::HnswHierarchy::GPU;
 *   params.M = 32;
 *   params.ef_construction = 120;
 *
 *   // Configure GPU graph building parameters
 *   auto ace_params = hnsw::graph_build_params::ace_params();
 *   ace_params.npartitions = 4;
 *   ace_params.use_disk = true;
 *   ace_params.build_dir = "/tmp/hnsw_ace_build";
 *   params.graph_build_params = ace_params;
 *
 *   // Build the index
 *   auto dataset = raft::make_host_matrix<float, int64_t>(res, N, D);
 *   // ... fill dataset ...
 *   auto hnsw_index = hnsw::build(res, params, raft::make_const_mdspan(dataset.view()));
 *
 *   // Search the index
 *   hnsw::search_params search_params;
 *   search_params.ef = 200;
 *   auto neighbors = raft::make_host_matrix<uint64_t, int64_t>(res, n_queries, k);
 *   auto distances = raft::make_host_matrix<float, int64_t>(res, n_queries, k);
 *   hnsw::search(res, search_params, *hnsw_index, queries, neighbors.view(), distances.view());
 *
 *   // Serialize the index
 *   hnsw::serialize(res, "index.bin", *hnsw_index);
 * @endcode
 */
std::unique_ptr<index<half>> build(
  raft::resources const& res,
  const index_params& params,
  raft::host_matrix_view<const half, int64_t, raft::row_major> dataset);

/**
 * @brief Build an HNSW index on the GPU
 *
 * The resulting graph is compatible for HNSW search, but is not an exact equivalent of the graph
 * built by the HNSW.
 *
 * The HNSW index construction parameters `M` and `ef_construction` are the main parameters to
 * control the graph degree and graph quality.  We have additional options that can be used to fine
 * tune graph building on the GPU (see `cuvs::neighbors::cagra::index_params`). In case the index
 * does not fit the host or GPU memory,  we would use disk as temporary storage. In such cases it is
 * important to set `ace_params.build_dir` to a fast disk with sufficient storage size.
 *
 * NOTE: This function requires CUDA headers to be available at compile time.
 *
 * @param[in] res raft resources
 * @param[in] params hnsw index parameters including ACE configuration
 * @param[in] dataset a host matrix view to a row-major matrix [n_rows, dim]
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   raft::resources res;
 *
 *   // Create index parameters with ACE configuration
 *   hnsw::index_params params;
 *   params.metric = cuvs::distance::DistanceType::L2Expanded;
 *   params.hierarchy = hnsw::HnswHierarchy::GPU;
 *   params.M = 32;
 *   params.ef_construction = 120;
 *
 *   // Configure GPU graph building parameters
 *   auto ace_params = hnsw::graph_build_params::ace_params();
 *   ace_params.npartitions = 4;
 *   ace_params.use_disk = true;
 *   ace_params.build_dir = "/tmp/hnsw_ace_build";
 *   params.graph_build_params = ace_params;
 *
 *   // Build the index
 *   auto dataset = raft::make_host_matrix<float, int64_t>(res, N, D);
 *   // ... fill dataset ...
 *   auto hnsw_index = hnsw::build(res, params, raft::make_const_mdspan(dataset.view()));
 *
 *   // Search the index
 *   hnsw::search_params search_params;
 *   search_params.ef = 200;
 *   auto neighbors = raft::make_host_matrix<uint64_t, int64_t>(res, n_queries, k);
 *   auto distances = raft::make_host_matrix<float, int64_t>(res, n_queries, k);
 *   hnsw::search(res, search_params, *hnsw_index, queries, neighbors.view(), distances.view());
 *
 *   // Serialize the index
 *   hnsw::serialize(res, "index.bin", *hnsw_index);
 * @endcode
 */
std::unique_ptr<index<uint8_t>> build(
  raft::resources const& res,
  const index_params& params,
  raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset);

/**
 * @brief Build an HNSW index on the GPU
 *
 * The resulting graph is compatible for HNSW search, but is not an exact equivalent of the graph
 * built by the HNSW.
 *
 * The HNSW index construction parameters `M` and `ef_construction` are the main parameters to
 * control the graph degree and graph quality.  We have additional options that can be used to fine
 * tune graph building on the GPU (see `cuvs::neighbors::cagra::index_params`). In case the index
 * does not fit the host or GPU memory,  we would use disk as temporary storage. In such cases it is
 * important to set `ace_params.build_dir` to a fast disk with sufficient storage size.
 *
 * NOTE: This function requires CUDA headers to be available at compile time.
 *
 * @param[in] res raft resources
 * @param[in] params hnsw index parameters including ACE configuration
 * @param[in] dataset a host matrix view to a row-major matrix [n_rows, dim]
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   raft::resources res;
 *
 *   // Create index parameters with ACE configuration
 *   hnsw::index_params params;
 *   params.metric = cuvs::distance::DistanceType::L2Expanded;
 *   params.hierarchy = hnsw::HnswHierarchy::GPU;
 *   params.M = 32;
 *   params.ef_construction = 120;
 *
 *   // Configure GPU graph building parameters
 *   auto ace_params = hnsw::graph_build_params::ace_params();
 *   ace_params.npartitions = 4;
 *   ace_params.use_disk = true;
 *   ace_params.build_dir = "/tmp/hnsw_ace_build";
 *   params.graph_build_params = ace_params;
 *
 *   // Build the index
 *   auto dataset = raft::make_host_matrix<float, int64_t>(res, N, D);
 *   // ... fill dataset ...
 *   auto hnsw_index = hnsw::build(res, params, raft::make_const_mdspan(dataset.view()));
 *
 *   // Search the index
 *   hnsw::search_params search_params;
 *   search_params.ef = 200;
 *   auto neighbors = raft::make_host_matrix<uint64_t, int64_t>(res, n_queries, k);
 *   auto distances = raft::make_host_matrix<float, int64_t>(res, n_queries, k);
 *   hnsw::search(res, search_params, *hnsw_index, queries, neighbors.view(), distances.view());
 *
 *   // Serialize the index
 *   hnsw::serialize(res, "index.bin", *hnsw_index);
 * @endcode
 */
std::unique_ptr<index<int8_t>> build(
  raft::resources const& res,
  const index_params& params,
  raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset);

/**
 * @}
 */

/**
 * @defgroup hnsw_cpp_index_load Load CAGRA index as hnswlib index
 * @{
 */

/**
 * @brief Construct an hnswlib index from a CAGRA index
 * NOTE: When `hnsw::index_params.hierarchy` is:
 *       1. `NONE`: This method uses the filesystem to write the CAGRA index in
 * `/tmp/<random_number>.bin` before reading it as an hnswlib index, then deleting the temporary
 * file. The returned index is immutable and can only be searched by the hnswlib wrapper in cuVS, as
 * the format is not compatible with the original hnswlib.
 *       2. `CPU`: The returned index is mutable and can be extended with additional vectors. The
 * serialized index is also compatible with the original hnswlib library.
 *
 * @param[in] res raft resources
 * @param[in] params hnsw index parameters
 * @param[in] cagra_index cagra index
 * @param[in] dataset optional dataset to avoid extra memory copy when hierarchy is `CPU`
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *
 *   // Load CAGRA index as an HNSW index
 *   hnsw::index_params hnsw_params;
 *   auto hnsw_index = hnsw::from_cagra(res, hnsw_params, index);
 * @endcode
 */
std::unique_ptr<index<float>> from_cagra(
  raft::resources const& res,
  const index_params& params,
  const cuvs::neighbors::cagra::index<float, uint32_t>& cagra_index,
  std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>> dataset =
    std::nullopt);

/**
 * @brief Construct an hnswlib index from a CAGRA index
 * NOTE: When `hnsw::index_params.hierarchy` is:
 *       1. `NONE`: This method uses the filesystem to write the CAGRA index in
 * `/tmp/<random_number>.bin` before reading it as an hnswlib index, then deleting the temporary
 * file. The returned index is immutable and can only be searched by the hnswlib wrapper in cuVS, as
 * the format is not compatible with the original hnswlib.
 *       2. `CPU`: The returned index is mutable and can be extended with additional vectors. The
 * serialized index is also compatible with the original hnswlib library.
 *
 * @param[in] res raft resources
 * @param[in] params hnsw index parameters
 * @param[in] cagra_index cagra index
 * @param[in] dataset optional dataset to avoid extra memory copy when hierarchy is `CPU`
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *
 *   // Load CAGRA index as an HNSW index
 *   hnsw::index_params hnsw_params;
 *   auto hnsw_index = hnsw::from_cagra(res, hnsw_params, index);
 * @endcode
 */
std::unique_ptr<index<half>> from_cagra(
  raft::resources const& res,
  const index_params& params,
  const cuvs::neighbors::cagra::index<half, uint32_t>& cagra_index,
  std::optional<raft::host_matrix_view<const half, int64_t, raft::row_major>> dataset =
    std::nullopt);

/**
 * @brief Construct an hnswlib index from a CAGRA index
 * NOTE: When `hnsw::index_params.hierarchy` is:
 *       1. `NONE`: This method uses the filesystem to write the CAGRA index in
 * `/tmp/<random_number>.bin` before reading it as an hnswlib index, then deleting the temporary
 * file. The returned index is immutable and can only be searched by the hnswlib wrapper in cuVS, as
 * the format is not compatible with the original hnswlib.
 *       2. `CPU`: The returned index is mutable and can be extended with additional vectors. The
 * serialized index is also compatible with the original hnswlib library.
 *
 * @param[in] res raft resources
 * @param[in] params hnsw index parameters
 * @param[in] cagra_index cagra index
 * @param[in] dataset optional dataset to avoid extra memory copy when hierarchy is `CPU`
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *
 *   // Load CAGRA index as an HNSW index
 *   hnsw::index_params hnsw_params;
 *   auto hnsw_index = hnsw::from_cagra(res, hnsw_params, index);
 * @endcode
 */
std::unique_ptr<index<uint8_t>> from_cagra(
  raft::resources const& res,
  const index_params& params,
  const cuvs::neighbors::cagra::index<uint8_t, uint32_t>& cagra_index,
  std::optional<raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>> dataset =
    std::nullopt);

/**
 * @brief Construct an hnswlib index from a CAGRA index
 * NOTE: When `hnsw::index_params.hierarchy` is:
 *       1. `NONE`: This method uses the filesystem to write the CAGRA index in
 * `/tmp/<random_number>.bin` before reading it as an hnswlib index, then deleting the temporary
 * file. The returned index is immutable and can only be searched by the hnswlib wrapper in cuVS, as
 * the format is not compatible with the original hnswlib.
 *       2. `CPU`: The returned index is mutable and can be extended with additional vectors. The
 * serialized index is also compatible with the original hnswlib library.
 *
 * @param[in] res raft resources
 * @param[in] params hnsw index parameters
 * @param[in] cagra_index cagra index
 * @param[in] dataset optional dataset to avoid extra memory copy when hierarchy is `CPU`
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *
 *   // Load CAGRA index as an HNSW index
 *   hnsw::index_params hnsw_params;
 *   auto hnsw_index = hnsw::from_cagra(res, hnsw_params, index);
 * @endcode
 */
std::unique_ptr<index<int8_t>> from_cagra(
  raft::resources const& res,
  const index_params& params,
  const cuvs::neighbors::cagra::index<int8_t, uint32_t>& cagra_index,
  std::optional<raft::host_matrix_view<const int8_t, int64_t, raft::row_major>> dataset =
    std::nullopt);

/**
 * @}
 */

/**
 * @defgroup hnsw_cpp_index_extend Extend HNSW index with additional vectors
 * @{
 */

/**
 * @brief Add new vectors to an HNSW index
 * NOTE: The HNSW index can only be extended when the `hnsw::index_params.hierarchy` is `CPU`
 *       when converting from a CAGRA index.
 *
 * @param[in] res raft resources
 * @param[in] params configure the extend
 * @param[in] additional_dataset a host matrix view to a row-major matrix [n_rows, index->dim()]
 * @param[inout] idx HNSW index to extend
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace cuvs::neighbors;
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *
 *   // Load CAGRA index as an HNSW index
 *   hnsw::index_params hnsw_params;
 *   hnsw_params.hierarchy = hnsw::HnswHierarchy::CPU;
 *   auto hnsw_index = hnsw::from_cagra(res, hnsw_params, index);
 *
 *   // Extend the HNSW index with additional vectors
 *   auto additional_dataset = raft::make_host_matrix<float>(res, add_size, index->dim());
 *   hnsw::extend_params extend_params;
 *   hnsw::extend(res, extend_params, additional_dataset, *hnsw_index.get());
 * @endcode
 */
void extend(raft::resources const& res,
            const extend_params& params,
            raft::host_matrix_view<const float, int64_t, raft::row_major> additional_dataset,
            index<float>& idx);

/**
 * @brief Add new vectors to an HNSW index
 * NOTE: The HNSW index can only be extended when the `hnsw::index_params.hierarchy` is `CPU`
 *       when converting from a CAGRA index.
 *
 * @param[in] res raft resources
 * @param[in] params configure the extend
 * @param[in] additional_dataset a host matrix view to a row-major matrix [n_rows, index->dim()]
 * @param[inout] idx HNSW index to extend
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace cuvs::neighbors;
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *
 *   // Load CAGRA index as an HNSW index
 *   hnsw::index_params hnsw_params;
 *   hnsw_params.hierarchy = hnsw::HnswHierarchy::CPU;
 *   auto hnsw_index = hnsw::from_cagra(res, hnsw_params, index);
 *
 *   // Extend the HNSW index with additional vectors
 *   auto additional_dataset = raft::make_host_matrix<half>(res, add_size, index->dim());
 *   hnsw::extend_params extend_params;
 *   hnsw::extend(res, extend_params, additional_dataset, *hnsw_index.get());
 * @endcode
 */
void extend(raft::resources const& res,
            const extend_params& params,
            raft::host_matrix_view<const half, int64_t, raft::row_major> additional_dataset,
            index<half>& idx);

/**
 * @brief Add new vectors to an HNSW index
 * NOTE: The HNSW index can only be extended when the `hnsw::index_params.hierarchy` is `CPU`
 *       when converting from a CAGRA index.
 *
 * @param[in] res raft resources
 * @param[in] params configure the extend
 * @param[in] additional_dataset a host matrix view to a row-major matrix [n_rows, index->dim()]
 * @param[inout] idx HNSW index to extend
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace cuvs::neighbors;
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *
 *   // Load CAGRA index as an HNSW index
 *   hnsw::index_params hnsw_params;
 *   hnsw_params.hierarchy = hnsw::HnswHierarchy::CPU;
 *   auto hnsw_index = hnsw::from_cagra(res, hnsw_params, index);
 *
 *   // Extend the HNSW index with additional vectors
 *   auto additional_dataset = raft::make_host_matrix<uint8_t>(res, add_size, index->dim());
 *   hnsw::extend_params extend_params;
 *   hnsw::extend(res, extend_params, additional_dataset, *hnsw_index.get());
 * @endcode
 */
void extend(raft::resources const& res,
            const extend_params& params,
            raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> additional_dataset,
            index<uint8_t>& idx);

/**
 * @brief Add new vectors to an HNSW index
 * NOTE: The HNSW index can only be extended when the `hnsw::index_params.hierarchy` is `CPU`
 *       when converting from a CAGRA index.
 *
 * @param[in] res raft resources
 * @param[in] params configure the extend
 * @param[in] additional_dataset a host matrix view to a row-major matrix [n_rows, index->dim()]
 * @param[inout] idx HNSW index to extend
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace cuvs::neighbors;
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *
 *   // Load CAGRA index as an HNSW index
 *   hnsw::index_params hnsw_params;
 *   hnsw_params.hierarchy = hnsw::HnswHierarchy::CPU;
 *   auto hnsw_index = hnsw::from_cagra(res, hnsw_params, index);
 *
 *   // Extend the HNSW index with additional vectors
 *   auto additional_dataset = raft::make_host_matrix<int8_t>(res, add_size, index->dim());
 *   hnsw::extend_params extend_params;
 *   hnsw::extend(res, extend_params, additional_dataset, *hnsw_index.get());
 * @endcode
 */
void extend(raft::resources const& res,
            const extend_params& params,
            raft::host_matrix_view<const int8_t, int64_t, raft::row_major> additional_dataset,
            index<int8_t>& idx);

/**
 * @}
 */

/**
 * @defgroup hnsw_cpp_search_params Build CAGRA index and search with hnswlib
 * @{
 */

struct search_params : cuvs::neighbors::search_params {
  int ef;               // size of the candidate list
  int num_threads = 0;  // number of host threads to use for concurrent searches. Value of 0
                        // automatically maximizes parallelism
};

/**
 * @}
 */

// TODO: Filtered Search APIs: https://github.com/rapidsai/cuvs/issues/363

/**
 * @defgroup hnsw_cpp_index_search Search hnswlib index
 * @{
 */

/**
 * @brief Search HNSW index constructed from a CAGRA index
 * NOTE: The HNSW index can only be searched by the hnswlib wrapper in cuVS when the hierarchy is
 * `NONE`, as the format is not compatible with the original hnswlib.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] idx cagra index
 * @param[in] queries a host matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a host matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a host matrix view to the distances to the selected neighbors [n_queries,
 * k]
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *
 *   // Load CAGRA index as an HNSW index
 *   hnsw::index_params hnsw_params;
 *   auto hnsw_index = hnsw::from_cagra(res, hnsw_params, index);
 *
 *   // Search K nearest neighbors as an hnswlib index
 *   // using host threads for concurrency
 *   hnsw::search_params search_params;
 *   search_params.ef = 50 // ef >= K;
 *   search_params.num_threads = 10;
 *   auto neighbors = raft::make_host_matrix<uint64_t>(res, n_queries, k);
 *   auto distances = raft::make_host_matrix<float>(res, n_queries, k);
 *   hnsw::search(res, search_params, *index.get(), queries, neighbors, distances);
 * @endcode
 */
void search(raft::resources const& res,
            const search_params& params,
            const index<float>& idx,
            raft::host_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::host_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
            raft::host_matrix_view<float, int64_t, raft::row_major> distances);

/**
 * @brief Search HNSW index constructed from a CAGRA index
 * NOTE: The HNSW index can only be searched by the hnswlib wrapper in cuVS when the hierarchy is
 * `NONE`, as the format is not compatible with the original hnswlib.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] idx cagra index
 * @param[in] queries a host matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a host matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a host matrix view to the distances to the selected neighbors [n_queries,
 * k]
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *
 *   // Load CAGRA index as an HNSW index
 *   hnsw::index_params hnsw_params;
 *   auto hnsw_index = hnsw::from_cagra(res, hnsw_params, index);
 *
 *   // Search K nearest neighbors as an hnswlib index
 *   // using host threads for concurrency
 *   hnsw::search_params search_params;
 *   search_params.ef = 50 // ef >= K;
 *   search_params.num_threads = 10;
 *   auto neighbors = raft::make_host_matrix<uint64_t>(res, n_queries, k);
 *   auto distances = raft::make_host_matrix<float>(res, n_queries, k);
 *   hnsw::search(res, search_params, *index.get(), queries, neighbors, distances);
 * @endcode
 */
void search(raft::resources const& res,
            const search_params& params,
            const index<half>& idx,
            raft::host_matrix_view<const half, int64_t, raft::row_major> queries,
            raft::host_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
            raft::host_matrix_view<float, int64_t, raft::row_major> distances);

/**
 * @brief Search HNSWindex constructed from a CAGRA index
 * NOTE: The HNSW index can only be searched by the hnswlib wrapper in cuVS when the hierarchy is
 * `NONE`, as the format is not compatible with the original hnswlib.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] idx cagra index
 * @param[in] queries a host matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a host matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a host matrix view to the distances to the selected neighbors [n_queries,
 * k]
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *
 *   // Load CAGRA index as an HNSW index
 *   hnsw::index_params hnsw_params;
 *   auto hnsw_index = hnsw::from_cagra(res, hnsw_params, index);
 *
 *   // Search K nearest neighbors as an hnswlib index
 *   // using host threads for concurrency
 *   hnsw::search_params search_params;
 *   search_params.ef = 50 // ef >= K;
 *   search_params.num_threads = 10;
 *   auto neighbors = raft::make_host_matrix<uint64_t>(res, n_queries, k);
 *   auto distances = raft::make_host_matrix<float>(res, n_queries, k);
 *   hnsw::search(res, search_params, *index.get(), queries, neighbors, distances);
 * @endcode
 */
void search(raft::resources const& res,
            const search_params& params,
            const index<uint8_t>& idx,
            raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::host_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
            raft::host_matrix_view<float, int64_t, raft::row_major> distances);

/**
 * @brief Search HNSW index constructed from a CAGRA index
 * NOTE: The HNSW index can only be searched by the hnswlib wrapper in cuVS when the hierarchy is
 * `NONE`, as the format is not compatible with the original hnswlib.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] idx cagra index
 * @param[in] queries a host matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a host matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a host matrix view to the distances to the selected neighbors [n_queries,
 * k]
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *
 *   // Load CAGRA index as an HNSW index
 *   hnsw::index_params hnsw_params;
 *   auto hnsw_index = hnsw::from_cagra(res, hnsw_params, index);
 *
 *   // Search K nearest neighbors as an hnswlib index
 *   // using host threads for concurrency
 *   hnsw::search_params search_params;
 *   search_params.ef = 50 // ef >= K;
 *   search_params.num_threads = 10;
 *   auto neighbors = raft::make_host_matrix<uint64_t>(res, n_queries, k);
 *   auto distances = raft::make_host_matrix<float>(res, n_queries, k);
 *   hnsw::search(res, search_params, *index.get(), queries, neighbors, distances);
 * @endcode
 */
void search(raft::resources const& res,
            const search_params& params,
            const index<int8_t>& idx,
            raft::host_matrix_view<const int8_t, int64_t, raft::row_major> queries,
            raft::host_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
            raft::host_matrix_view<float, int64_t, raft::row_major> distances);

/**
 * @}
 */

/**
 * @defgroup hnsw_cpp_index_serialize Deserialize CAGRA index as hnswlib index
 * @{
 */

/**
 * @brief Serialize the HNSW index to file
 * NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the
 * hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.
 * However, when hierarchy is `CPU`, the saved hnswlib index is compatible with the original hnswlib
 * library.
 *
 * @param[in] res raft resources
 * @param[in] filename path to the file to save the serialized CAGRA index
 * @param[in] idx cagra index
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *
 *   // Load CAGRA index as an HNSW index
 *   hnsw::index_params hnsw_params;
 *   auto hnsw_index = hnsw::from_cagra(res, hnsw_params, index);
 *   // Save the index
 *   hnsw::serialize(res, "index.bin", index);
 * @endcode
 */
void serialize(raft::resources const& res, const std::string& filename, const index<float>& idx);

/**
 * @brief Serialize the HNSW index to file
 * NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the
 * hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.
 * However, when hierarchy is `CPU`, the saved hnswlib index is compatible with the original hnswlib
 * library.
 *
 * @param[in] res raft resources
 * @param[in] filename path to the file to save the serialized CAGRA index
 * @param[in] idx cagra index
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *
 *   // Load CAGRA index as an HNSW index
 *   hnsw::index_params hnsw_params;
 *   auto hnsw_index = hnsw::from_cagra(res, hnsw_params, index);
 *   // Save the index
 *   hnsw::serialize(res, "index.bin", index);
 * @endcode
 */
void serialize(raft::resources const& res, const std::string& filename, const index<half>& idx);

/**
 * @brief Serialize the HNSW index to file
 * NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the
 * hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.
 * However, when hierarchy is `CPU`, the saved hnswlib index is compatible with the original hnswlib
 * library.
 *
 * @param[in] res raft resources
 * @param[in] filename path to the file to save the serialized CAGRA index
 * @param[in] idx cagra index
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *
 *   // Load CAGRA index as an HNSW index
 *   hnsw::index_params hnsw_params;
 *   auto hnsw_index = hnsw::from_cagra(res, hnsw_params, index);
 *   // Save the index
 *   hnsw::serialize(res, "index.bin", index);
 * @endcode
 */
void serialize(raft::resources const& res, const std::string& filename, const index<uint8_t>& idx);

/**
 * @brief Serialize the HNSW index to file
 * NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the
 * hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.
 * However, when hierarchy is `CPU`, the saved hnswlib index is compatible with the original hnswlib
 * library.
 *
 * @param[in] res raft resources
 * @param[in] filename path to the file to save the serialized CAGRA index
 * @param[in] idx cagra index
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *
 *   // Load CAGRA index as an HNSW index
 *   hnsw::index_params hnsw_params;
 *   auto hnsw_index = hnsw::from_cagra(res, hnsw_params, index);
 *   // Save the index
 *   hnsw::serialize(res, "index.bin", index);
 * @endcode
 */
void serialize(raft::resources const& res, const std::string& filename, const index<int8_t>& idx);

/**
 * @brief De-serialize a CAGRA index saved to a file as an hnswlib index
 * NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the
 * hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.
 * However, when hierarchy is `CPU`, the saved hnswlib index is compatible with the original hnswlib
 * library.
 *
 * @param[in] res raft resources
 * @param[in] params hnsw index parameters
 * @param[in] filename path to the file containing the serialized CAGRA index
 * @param[in] dim dimensions of the training dataset
 * @param[in] metric distance metric to search. Supported metrics ("L2Expanded", "InnerProduct")
 * @param[out] index hnsw index
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *
 *   // Load CAGRA index as an HNSW index
 *  hnsw::index_params hnsw_params;
 *  auto hnsw_index = hnsw::from_cagra(res, hnsw_params, index);
 *  // save HNSW index to a file
 *  hnsw::serialize(res, "index.bin", hnsw_index);
 *  // De-serialize the HNSW index
 *  index<float>* hnsw_index = nullptr;
 *  hnsw::deserialize(res, hnsw_params, "index.bin", index->dim(), index->metric(), &hnsw_index);
 *
 *   // Delete index after use
 *   delete hnsw_index;
 * @endcode
 */
void deserialize(raft::resources const& res,
                 const index_params& params,
                 const std::string& filename,
                 int dim,
                 cuvs::distance::DistanceType metric,
                 index<float>** index);

/**
 * @brief De-serialize a CAGRA index saved to a file as an hnswlib index
 * NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the
 * hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.
 * However, when hierarchy is `CPU`, the saved hnswlib index is compatible with the original hnswlib
 * library.
 *
 * @param[in] res raft resources
 * @param[in] params hnsw index parameters
 * @param[in] filename path to the file containing the serialized CAGRA index
 * @param[in] dim dimensions of the training dataset
 * @param[in] metric distance metric to search. Supported metrics ("L2Expanded", "InnerProduct")
 * @param[out] index hnsw index
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *
 *   // Load CAGRA index as an HNSW index
 *  hnsw::index_params hnsw_params;
 *  auto hnsw_index = hnsw::from_cagra(res, hnsw_params, index);
 *  // save HNSW index to a file
 *  hnsw::serialize(res, "index.bin", hnsw_index);
 *  // De-serialize the HNSW index
 *  index<half>* hnsw_index = nullptr;
 *  hnsw::deserialize(res, hnsw_params, "index.bin", index->dim(), index->metric(), &hnsw_index);
 *
 *   // Delete index after use
 *   delete hnsw_index;
 * @endcode
 */
void deserialize(raft::resources const& res,
                 const index_params& params,
                 const std::string& filename,
                 int dim,
                 cuvs::distance::DistanceType metric,
                 index<half>** index);

/**
 * @brief De-serialize a CAGRA index saved to a file as an hnswlib index
 * NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the
 * hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.
 * However, when hierarchy is `CPU`, the saved hnswlib index is compatible with the original hnswlib
 * library.
 *
 * @param[in] res raft resources
 * @param[in] params hnsw index parameters
 * @param[in] filename path to the file containing the serialized CAGRA index
 * @param[in] dim dimensions of the training dataset
 * @param[in] metric distance metric to search. Supported metrics ("L2Expanded", "InnerProduct")
 * @param[out] index hnsw index
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *
 *   // Load CAGRA index as an HNSW index
 *  hnsw::index_params hnsw_params;
 *  auto hnsw_index = hnsw::from_cagra(res, hnsw_params, index);
 *  // save HNSW index to a file
 *  hnsw::serialize(res, "index.bin", hnsw_index);
 *  // De-serialize the HNSW index
 *  index<uint8_t>* hnsw_index = nullptr;
 *  hnsw::deserialize(res, hnsw_params, "index.bin", index->dim(), index->metric(), &hnsw_index);
 *
 *   // Delete index after use
 *   delete hnsw_index;
 * @endcode
 */
void deserialize(raft::resources const& res,
                 const index_params& params,
                 const std::string& filename,
                 int dim,
                 cuvs::distance::DistanceType metric,
                 index<uint8_t>** index);

/**
 * @brief De-serialize a CAGRA index saved to a file as an hnswlib index
 * NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the
 * hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.
 * However, when hierarchy is `CPU`, the saved hnswlib index is compatible with the original hnswlib
 * library.
 *
 * @param[in] res raft resources
 * @param[in] params hnsw index parameters
 * @param[in] filename path to the file containing the serialized CAGRA index
 * @param[in] dim dimensions of the training dataset
 * @param[in] metric distance metric to search. Supported metrics ("L2Expanded", "InnerProduct")
 * @param[out] index hnsw index
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *
 *   // Load CAGRA index as an HNSW index
 *  hnsw::index_params hnsw_params;
 *  auto hnsw_index = hnsw::from_cagra(res, hnsw_params, index);
 *  // save HNSW index to a file
 *  hnsw::serialize(res, "index.bin", hnsw_index);
 *  // De-serialize the HNSW index
 *  index<int8_t>* hnsw_index = nullptr;
 *  hnsw::deserialize(res, hnsw_params, "index.bin", index->dim(), index->metric(), &hnsw_index);
 *
 *   // Delete index after use
 *   delete hnsw_index;
 * @endcode
 */
void deserialize(raft::resources const& res,
                 const index_params& params,
                 const std::string& filename,
                 int dim,
                 cuvs::distance::DistanceType metric,
                 index<int8_t>** index);

/**
 * @}
 */

}  // namespace hnsw
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
#else
#error "This header is only available if cuVS CMake option `BUILD_CAGRA_HNSWLIB=ON"
#endif
