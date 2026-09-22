/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <dlpack/dlpack.h>
#include <type_traits>

#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/all_neighbors.h>
#include <cuvs/neighbors/ivf_pq.h>
#include <cuvs/neighbors/nn_descent.h>
#include <cuvs/neighbors/all_neighbors.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/nn_descent.hpp>

#include "../core/exceptions.hpp"
#include "../core/interop.hpp"

#include "ivf_pq.hpp"

namespace {

void convert_nn_descent_params(cuvsNNDescentIndexParams params,
                               cuvs::neighbors::nn_descent::index_params* out)
{
  out->metric                    = static_cast<cuvs::distance::DistanceType>((int)params.metric);
  out->metric_arg                = params.metric_arg;
  out->graph_degree              = params.graph_degree;
  out->intermediate_graph_degree = params.intermediate_graph_degree;
  out->max_iterations            = params.max_iterations;
  out->termination_threshold     = params.termination_threshold;
  out->return_distances          = params.return_distances;
}

static cuvs::neighbors::all_neighbors::all_neighbors_params convert_params(
  cuvsAllNeighborsIndexParams_t params_ptr, int64_t n_rows, int64_t n_cols)
{
  using namespace cuvs::neighbors;
  cuvs::neighbors::all_neighbors::all_neighbors_params out{};
  auto& p            = *params_ptr;
  out.metric         = static_cast<cuvs::distance::DistanceType>((int)p.metric);
  out.overlap_factor = p.overlap_factor;
  out.n_clusters     = p.n_clusters;

  switch (p.algo) {
    case CUVS_ALL_NEIGHBORS_ALGO_BRUTE_FORCE: {
      graph_build_params::brute_force_params b{};
      b.build_params.metric  = out.metric;
      out.graph_build_params = b;
      break;
    }
    case CUVS_ALL_NEIGHBORS_ALGO_NN_DESCENT: {
      graph_build_params::nn_descent_params n{};
      n.metric = out.metric;
      // Use nn_descent_params if provided, otherwise use defaults
      if (p.nn_descent_params != nullptr) { convert_nn_descent_params(*p.nn_descent_params, &n); }
      out.graph_build_params = n;
      break;
    }
    case CUVS_ALL_NEIGHBORS_ALGO_IVF_PQ: {
      auto dataset_extents = raft::matrix_extent<int64_t>{n_rows, n_cols};
      graph_build_params::ivf_pq_params ivf(dataset_extents, out.metric);
      // Use ivf_pq_params if provided, otherwise use defaults
      if (p.ivf_pq_params != nullptr) {
        cuvs::neighbors::ivf_pq::convert_c_index_params(*p.ivf_pq_params, &ivf.build_params);
      }
      out.graph_build_params = ivf;
      break;
    }
    default: RAFT_FAIL("Unsupported all-neighbors algo");
  }

  return out;
}

static void ensure_indices_dtype(DLManagedTensor* indices)
{
  auto dtype = indices->dl_tensor.dtype;
  RAFT_EXPECTS(dtype.code == kDLInt && dtype.bits == 64, "indices must be int64 output tensor");
}

static void ensure_optional_distance_dtype(DLManagedTensor* distances)
{
  if (distances == nullptr) { return; }
  auto dtype = distances->dl_tensor.dtype;
  RAFT_EXPECTS(dtype.code == kDLFloat && dtype.bits == 32,
               "distances must be float32 output tensor");
}

static void ensure_optional_core_distance_dtype(DLManagedTensor* core_distances)
{
  if (core_distances == nullptr) { return; }
  auto dtype = core_distances->dl_tensor.dtype;
  RAFT_EXPECTS(dtype.code == kDLFloat && dtype.bits == 32,
               "core_distances must be float32 output tensor");
}

// Validate that the outputs (indices/distances/core_distances) all live in the same memory space
static bool validate_output_memory_space(DLManagedTensor* indices,
                                         DLManagedTensor* distances,
                                         DLManagedTensor* core_distances)
{
  const bool host = cuvs::core::is_dlpack_host_compatible(indices->dl_tensor);
  RAFT_EXPECTS(host || cuvs::core::is_dlpack_device_compatible(indices->dl_tensor),
               "indices tensor must be host- or device-compatible");
  auto same_space = [&](DLManagedTensor* t, const char* name) {
    if (t == nullptr) { return; }
    RAFT_EXPECTS(cuvs::core::is_dlpack_host_compatible(t->dl_tensor) == host,
                 "%s tensor must be in the same memory space (host or device) as indices",
                 name);
  };
  same_space(distances, "distances");
  same_space(core_distances, "core_distances");
  return host;
}

// Build with a host-resident dataset. HostOutput selects whether the outputs live on host or
// device.
template <typename T, bool HostOutput>
void _build_host(cuvsResources_t res,
                 cuvsAllNeighborsIndexParams_t params,
                 DLManagedTensor* dataset_tensor,
                 DLManagedTensor* indices_tensor,
                 DLManagedTensor* distances_tensor,
                 DLManagedTensor* core_distances_tensor,
                 float alpha)
{
  auto& cpp_res = *reinterpret_cast<raft::device_resources*>(res);

  auto& dlt = dataset_tensor->dl_tensor;
  RAFT_EXPECTS(cuvs::core::is_dlpack_host_compatible(dlt),
               "Host build expects host-compatible dataset tensor");

  int64_t n_rows = dlt.shape[0];
  int64_t n_cols = dlt.shape[1];

  auto cpp_params = convert_params(params, n_rows, n_cols);

  using dataset_mdspan_t = raft::host_matrix_view<const T, int64_t, raft::row_major>;
  using indices_mdspan_t =
    std::conditional_t<HostOutput,
                       raft::host_matrix_view<int64_t, int64_t, raft::row_major>,
                       raft::device_matrix_view<int64_t, int64_t, raft::row_major>>;
  using distances_mdspan_t =
    std::conditional_t<HostOutput,
                       raft::host_matrix_view<float, int64_t, raft::row_major>,
                       raft::device_matrix_view<float, int64_t, raft::row_major>>;
  using core_mdspan_t = std::conditional_t<HostOutput,
                                           raft::host_vector_view<float, int64_t>,
                                           raft::device_vector_view<float, int64_t>>;

  auto dataset = cuvs::core::from_dlpack<dataset_mdspan_t>(dataset_tensor);
  auto indices = cuvs::core::from_dlpack<indices_mdspan_t>(indices_tensor);

  std::optional<distances_mdspan_t> distances = std::nullopt;
  if (distances_tensor) {
    distances = cuvs::core::from_dlpack<distances_mdspan_t>(distances_tensor);
  }

  std::optional<core_mdspan_t> core_distances = std::nullopt;
  if (core_distances_tensor) {
    core_distances = cuvs::core::from_dlpack<core_mdspan_t>(core_distances_tensor);
  }

  cuvs::neighbors::all_neighbors::build(
    cpp_res, cpp_params, dataset, indices, distances, core_distances, alpha);
}

// Build with a device-resident dataset. Outputs are always device-resident.
template <typename T>
void _build_device(cuvsResources_t device_res,
                   cuvsAllNeighborsIndexParams_t params,
                   DLManagedTensor* dataset_tensor,
                   DLManagedTensor* indices_tensor,
                   DLManagedTensor* distances_tensor,
                   DLManagedTensor* core_distances_tensor,
                   float alpha)
{
  auto& cpp_res = *reinterpret_cast<raft::device_resources*>(device_res);

  auto& dlt = dataset_tensor->dl_tensor;
  RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(dlt),
               "Device build expects device-compatible dataset tensor");

  int64_t n_rows = dlt.shape[0];
  int64_t n_cols = dlt.shape[1];

  auto cpp_params = convert_params(params, n_rows, n_cols);

  using dataset_mdspan_t   = raft::device_matrix_view<const T, int64_t, raft::row_major>;
  using indices_mdspan_t   = raft::device_matrix_view<int64_t, int64_t, raft::row_major>;
  using distances_mdspan_t = raft::device_matrix_view<float, int64_t, raft::row_major>;
  using core_mdspan_t      = raft::device_vector_view<float, int64_t>;

  auto dataset = cuvs::core::from_dlpack<dataset_mdspan_t>(dataset_tensor);
  auto indices = cuvs::core::from_dlpack<indices_mdspan_t>(indices_tensor);

  std::optional<distances_mdspan_t> distances = std::nullopt;
  if (distances_tensor) {
    distances = cuvs::core::from_dlpack<distances_mdspan_t>(distances_tensor);
  }

  std::optional<core_mdspan_t> core_distances = std::nullopt;
  if (core_distances_tensor) {
    core_distances = cuvs::core::from_dlpack<core_mdspan_t>(core_distances_tensor);
  }

  cuvs::neighbors::all_neighbors::build(
    cpp_res, cpp_params, dataset, indices, distances, core_distances, alpha);
}

}  // namespace

extern "C" cuvsError_t cuvsAllNeighborsIndexParamsCreate(cuvsAllNeighborsIndexParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    *params                      = new cuvsAllNeighborsIndexParams{};
    (*params)->algo              = CUVS_ALL_NEIGHBORS_ALGO_BRUTE_FORCE;
    (*params)->overlap_factor    = 1;
    (*params)->n_clusters        = 1;
    (*params)->metric            = L2Expanded;
    (*params)->ivf_pq_params     = nullptr;
    (*params)->nn_descent_params = nullptr;
  });
}

extern "C" cuvsError_t cuvsAllNeighborsIndexParamsDestroy(cuvsAllNeighborsIndexParams_t params)
{
  return cuvs::core::translate_exceptions([=] {
    if (params != nullptr) {
      if (params->ivf_pq_params != nullptr) { cuvsIvfPqIndexParamsDestroy(params->ivf_pq_params); }
      if (params->nn_descent_params != nullptr) {
        cuvsNNDescentIndexParamsDestroy(params->nn_descent_params);
      }
      delete params;
    }
  });
}

extern "C" cuvsError_t cuvsAllNeighborsBuild(cuvsResources_t res,
                                             cuvsAllNeighborsIndexParams_t params,
                                             DLManagedTensor* dataset_tensor,
                                             DLManagedTensor* indices_tensor,
                                             DLManagedTensor* distances_tensor,
                                             DLManagedTensor* core_distances_tensor,
                                             float alpha)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = dataset_tensor->dl_tensor;

    ensure_indices_dtype(indices_tensor);
    ensure_optional_distance_dtype(distances_tensor);
    ensure_optional_core_distance_dtype(core_distances_tensor);
    if (core_distances_tensor != nullptr && distances_tensor == nullptr) {
      RAFT_FAIL("distances tensor must be provided when core_distances tensor is provided");
    }

     const bool host_output =
      validate_output_memory_space(indices_tensor, distances_tensor, core_distances_tensor);

    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      // Check if dataset is host-compatible or device-compatible
      if (cuvs::core::is_dlpack_host_compatible(dataset)) {
        // Host dataset supports both host- and device-resident outputs.
        if (host_output) {
          _build_host<float, true>(
            res, params, dataset_tensor, indices_tensor, distances_tensor, core_distances_tensor,
            alpha);
        } else {
          _build_host<float, false>(
            res, params, dataset_tensor, indices_tensor, distances_tensor, core_distances_tensor,
            alpha);
        }
      } else if (cuvs::core::is_dlpack_device_compatible(dataset)) {
        RAFT_EXPECTS(!host_output,
                    "A device-resident dataset requires device-resident outputs; put the dataset "
                    "on host to produce host-resident outputs.");
        _build_device<float>(res,
                             params,
                             dataset_tensor,
                             indices_tensor,
                             distances_tensor,
                             core_distances_tensor,
                             alpha);
      } else {
        RAFT_FAIL("Dataset tensor must be either host-compatible or device-compatible");
      }
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}
