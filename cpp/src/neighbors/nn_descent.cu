/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/nn_descent.cuh"
#include "detail/nn_descent_gnnd.hpp"
#include <cuvs/neighbors/nn_descent.hpp>
#include <raft/core/logger.hpp>

#include <utility>

using namespace raft;
namespace cuvs::neighbors::nn_descent {

/**
 * @brief Test if we have enough GPU memory to run NN descent algorithm.
 * *
 * @param res
 * @param dataset shape of the dataset
 * @param idx_size the size of index type in bytes
 * @return true if enough GPU memory could be allocated
 * @return false otherwise
 */
bool has_enough_device_memory(raft::resources const& res,
                              raft::matrix_extent<int64_t> dataset,
                              size_t idx_size)
{
  using DistData_t = float;
  try {
    auto d_data_ = raft::make_device_matrix<__half, size_t, raft::row_major>(
      res, dataset.extent(0), dataset.extent(1));
    auto l2_norms_     = raft::make_device_vector<DistData_t, size_t>(res, dataset.extent(0));
    auto graph_buffer_ = raft::make_device_vector<uint32_t, size_t>(
      res, dataset.extent(0) * idx_size * detail::DEGREE_ON_DEVICE);

    auto dists_buffer_ = raft::make_device_matrix<DistData_t, size_t, raft::row_major>(
      res, dataset.extent(0), detail::DEGREE_ON_DEVICE);

    auto d_locks_ = raft::make_device_vector<int, size_t>(res, dataset.extent(0));

    auto d_list_sizes_new_ = raft::make_device_vector<int2, size_t>(res, dataset.extent(0));
    auto d_list_sizes_old_ = raft::make_device_vector<int2, size_t>(res, dataset.extent(0));
    RAFT_LOG_DEBUG("Sufficient memory for NN descent");
    return true;
  } catch (std::bad_alloc& e) {
    RAFT_LOG_DEBUG("Insufficient memory for NN descent");
    return false;
  } catch (raft::logic_error& e) {
    RAFT_LOG_DEBUG("Insufficient memory for NN descent (logic error)");
    return false;
  }
}

std::pair<size_t, size_t> build_mem_usage(raft::resources const& res,
                                          raft::matrix_extent<int64_t> dataset,
                                          size_t graph_degree,
                                          size_t idx_size)
{
  // Mirror the persistent allocations of the GNND solver so that this estimate
  // and the actual build stay in sync. See detail::GNND<>::GNND and
  // has_enough_device_memory above for the corresponding buffers.
  const size_t nrow = static_cast<size_t>(dataset.extent(0));
  const size_t dim  = static_cast<size_t>(dataset.extent(1));

  constexpr size_t degree_on_device = static_cast<size_t>(detail::DEGREE_ON_DEVICE);
  constexpr size_t num_samples      = static_cast<size_t>(detail::NUM_SAMPLES);

  // Device working set (independent of the requested graph_degree; the on-device
  // list width is fixed to DEGREE_ON_DEVICE).
  size_t dev = 0;
  dev += nrow * dim * sizeof(__half);              // d_data_ (dataset cast to fp16)
  dev += nrow * sizeof(float);                     // l2_norms_ (L2/Cosine metrics)
  dev += nrow * degree_on_device * sizeof(int);    // graph_buffer_ (ID_t)
  dev += nrow * degree_on_device * sizeof(float);  // dists_buffer_ (also reused as d_rev_graph)
  dev += nrow * sizeof(int);                       // d_locks_
  dev += 2 * nrow * sizeof(int2);                  // d_list_sizes_new_ / _old_

  // Host working set. The GNND solver keeps a pageable distance matrix sized to
  // the (aligned) output degree plus a number of pinned staging buffers whose
  // width is fixed to DEGREE_ON_DEVICE / NUM_SAMPLES. The caller-owned output
  // graph (nrow * graph_degree) is intentionally excluded here.
  const size_t node_degree = detail::roundUp32(graph_degree);
  size_t host              = 0;
  host += nrow * node_degree * sizeof(float);       // GnndGraph::h_dists
  host += nrow * degree_on_device * sizeof(int);    // graph_host_buffer_
  host += nrow * degree_on_device * sizeof(float);  // dists_host_buffer_
  host += 2 * nrow * num_samples * idx_size;        // h_graph_new / h_graph_old
  host += 2 * nrow * num_samples * idx_size;        // h_rev_graph_new_ / _old_
  host += 4 * nrow * sizeof(int2);                  // h/d list sizes (new + old)

  return std::make_pair(host, dev);
}

}  // namespace cuvs::neighbors::nn_descent
