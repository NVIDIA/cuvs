/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/copy.cuh>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/numpy_serializer.hpp>
#include <raft/core/resource/multi_gpu.hpp>
#include <raft/core/resource/nccl_comm.hpp>
#include <raft/core/serialize.hpp>
#include <raft/linalg/add.cuh>
#include <raft/matrix/init.cuh>
#include <raft/util/cuda_dev_essentials.cuh>

#include "../../core/omp_wrapper.hpp"
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/knn_merge_parts.hpp>

#include <fstream>

namespace cuvs::neighbors {
using namespace raft;

template <typename AnnIndexType, typename T, typename IdxT, typename searchIdxT>
void search(const raft::resources& handle,
            const cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
            const cuvs::neighbors::search_params* search_params,
            raft::host_matrix_view<const T, int64_t, row_major> h_queries,
            raft::device_matrix_view<searchIdxT, int64_t, row_major> d_neighbors,
            raft::device_matrix_view<float, int64_t, row_major> d_distances);
}  // namespace cuvs::neighbors

namespace cuvs::neighbors::snmg::detail {
using namespace cuvs::neighbors;
using namespace raft;

// local index deserialization and distribution
template <typename AnnIndexType, typename T, typename IdxT>
void deserialize_and_distribute(const raft::resources& clique,
                                mg_index<AnnIndexType, T, IdxT>& index,
                                const std::string& filename)
{
  index.ann_interfaces_.reserve(index.num_ranks_);
  for (int rank = 0; rank < index.num_ranks_; rank++) {
    const raft::resources& dev_res = raft::resource::set_current_device_to_rank(clique, rank);
    auto& ann_if                   = index.ann_interfaces_.emplace_back();
    cuvs::neighbors::deserialize(dev_res, ann_if, filename);
  }
}

// MG index deserialization
template <typename AnnIndexType, typename T, typename IdxT>
void deserialize(const raft::resources& clique,
                 mg_index<AnnIndexType, T, IdxT>& index,
                 const std::string& filename)
{
  std::ifstream is(filename, std::ios::in | std::ios::binary);
  if (!is) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  char dtype_string[4];
  is.read(dtype_string, 4);

  const auto& handle = raft::resource::set_current_device_to_root_rank(clique);
  index.mode_ =
    static_cast<cuvs::neighbors::distribution_mode>(deserialize_scalar<int>(handle, is));
  index.num_ranks_ = deserialize_scalar<int>(handle, is);

  if (index.num_ranks_ != raft::resource::get_num_ranks(clique)) {
    RAFT_FAIL("Serialized index has %d ranks whereas NCCL clique has %d ranks",
              index.num_ranks_,
              raft::resource::get_num_ranks(clique));
  }

  index.ann_interfaces_.reserve(index.num_ranks_);
  for (int rank = 0; rank < index.num_ranks_; rank++) {
    const raft::resources& dev_res = raft::resource::set_current_device_to_rank(clique, rank);
    auto& ann_if                   = index.ann_interfaces_.emplace_back();
    cuvs::neighbors::deserialize(dev_res, ann_if, is);
  }

  is.close();
}

template <typename AnnIndexType, typename T, typename IdxT>
void build(const raft::resources& clique,
           mg_index<AnnIndexType, T, IdxT>& index,
           const cuvs::neighbors::index_params* index_params,
           raft::host_matrix_view<const T, int64_t, row_major> index_dataset)
{
  if (index.mode_ == REPLICATED) {
    int64_t n_rows = index_dataset.extent(0);
    RAFT_LOG_DEBUG("REPLICATED BUILD: %d*%drows", index.num_ranks_, n_rows);

    index.ann_interfaces_.resize(index.num_ranks_);

    // Enable nested parallelism
    int saved_nested = cuvs::core::omp::get_nested();
    cuvs::core::omp::set_nested(1);

    int omp_threads      = cuvs::core::omp::get_max_threads();
    int threads_per_rank = std::max(1, omp_threads / index.num_ranks_);

    cuvs::core::omp::check_threads(index.num_ranks_);
#pragma omp parallel for num_threads(index.num_ranks_)
    for (int rank = 0; rank < index.num_ranks_; rank++) {
      // Set thread limit for this rank's nested OpenMP regions
      cuvs::core::omp::set_num_threads(threads_per_rank);

      const raft::resources& dev_res = raft::resource::set_current_device_to_rank(clique, rank);
      auto& ann_if                   = index.ann_interfaces_[rank];
      cuvs::neighbors::build(dev_res, ann_if, index_params, index_dataset);
      resource::sync_stream(dev_res);

      // Restore to allow full parallelism for potential future nested regions
      cuvs::core::omp::set_num_threads(omp_threads);
    }

    // Restore original OpenMP settings
    cuvs::core::omp::set_nested(saved_nested);
  } else if (index.mode_ == SHARDED) {
    int64_t n_rows           = index_dataset.extent(0);
    int64_t n_cols           = index_dataset.extent(1);
    int64_t n_rows_per_shard = raft::ceildiv(n_rows, (int64_t)index.num_ranks_);

    RAFT_LOG_DEBUG("SHARDED BUILD: %d*%drows", index.num_ranks_, n_rows_per_shard);

    index.ann_interfaces_.resize(index.num_ranks_);

    // Enable nested parallelism
    int saved_nested = cuvs::core::omp::get_nested();
    cuvs::core::omp::set_nested(1);

    int omp_threads      = cuvs::core::omp::get_max_threads();
    int threads_per_rank = std::max(1, omp_threads / index.num_ranks_);

    cuvs::core::omp::check_threads(index.num_ranks_);
#pragma omp parallel for num_threads(index.num_ranks_)
    for (int rank = 0; rank < index.num_ranks_; rank++) {
      // Set thread limit for this rank's nested OpenMP regions
      cuvs::core::omp::set_num_threads(threads_per_rank);

      const raft::resources& dev_res  = raft::resource::set_current_device_to_rank(clique, rank);
      int64_t offset                  = rank * n_rows_per_shard;
      int64_t n_rows_of_current_shard = std::min(n_rows_per_shard, n_rows - offset);
      const T* partition_ptr          = index_dataset.data_handle() + (offset * n_cols);
      auto partition                  = raft::make_host_matrix_view<const T, int64_t, row_major>(
        partition_ptr, n_rows_of_current_shard, n_cols);
      auto& ann_if = index.ann_interfaces_[rank];
      cuvs::neighbors::build(dev_res, ann_if, index_params, partition);
      resource::sync_stream(dev_res);

      // Restore to allow full parallelism for potential future nested regions
      cuvs::core::omp::set_num_threads(omp_threads);
    }

    // Restore original OpenMP settings
    cuvs::core::omp::set_nested(saved_nested);
  }
}

template <typename AnnIndexType, typename T, typename IdxT>
void extend(const raft::resources& clique,
            mg_index<AnnIndexType, T, IdxT>& index,
            raft::host_matrix_view<const T, int64_t, row_major> new_vectors,
            std::optional<raft::host_vector_view<const IdxT, int64_t>> new_indices)
{
  int64_t n_rows = new_vectors.extent(0);
  if (index.mode_ == REPLICATED) {
    RAFT_LOG_DEBUG("REPLICATED EXTEND: %d*%drows", index.num_ranks_, n_rows);

    // Enable nested parallelism
    int saved_nested = cuvs::core::omp::get_nested();
    cuvs::core::omp::set_nested(1);

    int omp_threads      = cuvs::core::omp::get_max_threads();
    int threads_per_rank = std::max(1, omp_threads / index.num_ranks_);

    cuvs::core::omp::check_threads(index.num_ranks_);
#pragma omp parallel for num_threads(index.num_ranks_)
    for (int rank = 0; rank < index.num_ranks_; rank++) {
      // Set thread limit for this rank's nested OpenMP regions
      cuvs::core::omp::set_num_threads(threads_per_rank);

      const raft::resources& dev_res = raft::resource::set_current_device_to_rank(clique, rank);
      auto& ann_if                   = index.ann_interfaces_[rank];
      cuvs::neighbors::extend(dev_res, ann_if, new_vectors, new_indices);
      resource::sync_stream(dev_res);

      // Restore to allow full parallelism for potential future nested regions
      cuvs::core::omp::set_num_threads(omp_threads);
    }

    // Restore original OpenMP settings
    cuvs::core::omp::set_nested(saved_nested);
  } else if (index.mode_ == SHARDED) {
    int64_t n_cols           = new_vectors.extent(1);
    int64_t n_rows_per_shard = raft::ceildiv(n_rows, (int64_t)index.num_ranks_);

    RAFT_LOG_DEBUG("SHARDED EXTEND: %d*%drows", index.num_ranks_, n_rows_per_shard);

    // Enable nested parallelism
    int saved_nested = cuvs::core::omp::get_nested();
    cuvs::core::omp::set_nested(1);

    int omp_threads      = cuvs::core::omp::get_max_threads();
    int threads_per_rank = std::max(1, omp_threads / index.num_ranks_);

    cuvs::core::omp::check_threads(index.num_ranks_);
#pragma omp parallel for num_threads(index.num_ranks_)
    for (int rank = 0; rank < index.num_ranks_; rank++) {
      // Set thread limit for this rank's nested OpenMP regions
      cuvs::core::omp::set_num_threads(threads_per_rank);

      const raft::resources& dev_res  = raft::resource::set_current_device_to_rank(clique, rank);
      int64_t offset                  = rank * n_rows_per_shard;
      int64_t n_rows_of_current_shard = std::min(n_rows_per_shard, n_rows - offset);
      const T* new_vectors_ptr        = new_vectors.data_handle() + (offset * n_cols);
      auto new_vectors_part           = raft::make_host_matrix_view<const T, int64_t, row_major>(
        new_vectors_ptr, n_rows_of_current_shard, n_cols);

      std::optional<raft::host_vector_view<const IdxT, int64_t>> new_indices_part = std::nullopt;
      if (new_indices.has_value()) {
        const IdxT* new_indices_ptr = new_indices.value().data_handle() + offset;
        new_indices_part            = raft::make_host_vector_view<const IdxT, int64_t>(
          new_indices_ptr, n_rows_of_current_shard);
      }
      auto& ann_if = index.ann_interfaces_[rank];
      cuvs::neighbors::extend(dev_res, ann_if, new_vectors_part, new_indices_part);
      resource::sync_stream(dev_res);

      // Restore to allow full parallelism for potential future nested regions
      cuvs::core::omp::set_num_threads(omp_threads);
    }

    // Restore original OpenMP settings
    cuvs::core::omp::set_nested(saved_nested);
  }
}

template <typename AnnIndexType, typename T, typename IdxT, typename searchIdxT>
void sharded_search_with_direct_merge(
  const raft::resources& clique,
  const mg_index<AnnIndexType, T, IdxT>& index,
  const cuvs::neighbors::search_params* search_params,
  raft::host_matrix_view<const T, int64_t, row_major> queries,
  raft::host_matrix_view<searchIdxT, int64_t, row_major> neighbors,
  raft::host_matrix_view<float, int64_t, row_major> distances,
  int64_t n_rows_per_batch,
  int64_t n_rows,
  int64_t n_cols,
  int64_t n_neighbors,
  int64_t n_batches)
{
  const auto& root_handle = raft::resource::set_current_device_to_root_rank(clique);
  auto in_neighbors       = raft::make_device_matrix<searchIdxT, int64_t, row_major>(
    root_handle, index.num_ranks_ * n_rows_per_batch, n_neighbors);
  auto in_distances = raft::make_device_matrix<float, int64_t, row_major>(
    root_handle, index.num_ranks_ * n_rows_per_batch, n_neighbors);
  auto out_neighbors = raft::make_device_matrix<searchIdxT, int64_t, row_major>(
    root_handle, n_rows_per_batch, n_neighbors);
  auto out_distances =
    raft::make_device_matrix<float, int64_t, row_major>(root_handle, n_rows_per_batch, n_neighbors);

  for (int64_t batch_idx = 0; batch_idx < n_batches; batch_idx++) {
    int64_t offset                  = batch_idx * n_rows_per_batch;
    int64_t query_offset            = offset * n_cols;
    int64_t output_offset           = offset * n_neighbors;
    int64_t n_rows_of_current_batch = std::min((int64_t)n_rows_per_batch, n_rows - offset);
    int64_t part_size               = n_rows_of_current_batch * n_neighbors;
    auto query_partition            = raft::make_host_matrix_view<const T, int64_t, row_major>(
      queries.data_handle() + query_offset, n_rows_of_current_batch, n_cols);

    const int& requirements = index.num_ranks_;
    cuvs::core::omp::check_threads(
      requirements);  // should use at least num_ranks_ threads to avoid NCCL hang
#pragma omp parallel for num_threads(index.num_ranks_)
    for (int rank = 0; rank < index.num_ranks_; rank++) {
      const raft::resources& dev_res = raft::resource::set_current_device_to_rank(clique, rank);
      auto& ann_if                   = index.ann_interfaces_[rank];

      if (rank == raft::resource::get_root_rank(clique)) {  // root rank
        uint64_t batch_offset = raft::resource::get_root_rank(clique) * part_size;
        auto d_neighbors      = raft::make_device_matrix_view<searchIdxT, int64_t, row_major>(
          in_neighbors.data_handle() + batch_offset, n_rows_of_current_batch, n_neighbors);
        auto d_distances = raft::make_device_matrix_view<float, int64_t, row_major>(
          in_distances.data_handle() + batch_offset, n_rows_of_current_batch, n_neighbors);
        cuvs::neighbors::search(
          dev_res, ann_if, search_params, query_partition, d_neighbors, d_distances);

        // wait for other ranks
        ncclGroupStart();
        for (int from_rank = 0; from_rank < index.num_ranks_; from_rank++) {
          if (from_rank == raft::resource::get_root_rank(clique)) continue;

          batch_offset = from_rank * part_size;
          ncclRecv(in_neighbors.data_handle() + batch_offset,
                   part_size * sizeof(searchIdxT),
                   ncclUint8,
                   from_rank,
                   raft::resource::get_nccl_comm_for_rank(clique, rank),
                   raft::resource::get_cuda_stream(dev_res));
          ncclRecv(in_distances.data_handle() + batch_offset,
                   part_size * sizeof(float),
                   ncclUint8,
                   from_rank,
                   raft::resource::get_nccl_comm_for_rank(clique, rank),
                   raft::resource::get_cuda_stream(dev_res));
        }
        ncclGroupEnd();
        resource::sync_stream(dev_res);
      } else {  // non-root ranks
        auto d_neighbors = raft::make_device_matrix<searchIdxT, int64_t, row_major>(
          dev_res, n_rows_of_current_batch, n_neighbors);
        auto d_distances = raft::make_device_matrix<float, int64_t, row_major>(
          dev_res, n_rows_of_current_batch, n_neighbors);
        cuvs::neighbors::search(
          dev_res, ann_if, search_params, query_partition, d_neighbors.view(), d_distances.view());

        // send results to root rank
        ncclGroupStart();
        ncclSend(d_neighbors.data_handle(),
                 part_size * sizeof(searchIdxT),
                 ncclUint8,
                 raft::resource::get_root_rank(clique),
                 raft::resource::get_nccl_comm_for_rank(clique, rank),
                 raft::resource::get_cuda_stream(dev_res));
        ncclSend(d_distances.data_handle(),
                 part_size * sizeof(float),
                 ncclUint8,
                 raft::resource::get_root_rank(clique),
                 raft::resource::get_nccl_comm_for_rank(clique, rank),
                 raft::resource::get_cuda_stream(dev_res));
        ncclGroupEnd();
        resource::sync_stream(dev_res);
      }
    }

    const auto& root_handle_      = raft::resource::set_current_device_to_root_rank(clique);
    auto h_trans                  = std::vector<searchIdxT>(index.num_ranks_);
    searchIdxT translation_offset = 0;
    for (int rank = 0; rank < index.num_ranks_; rank++) {
      h_trans[rank] = translation_offset;
      translation_offset += index.ann_interfaces_[rank].size();
    }
    auto d_trans = raft::make_device_vector<searchIdxT>(root_handle_, index.num_ranks_);

    raft::copy(root_handle_,
               d_trans.view(),
               raft::make_host_vector_view<const searchIdxT>(h_trans.data(), index.num_ranks_));

    knn_merge_parts(root_handle_,
                    in_distances.view(),
                    in_neighbors.view(),
                    out_distances.view(),
                    out_neighbors.view(),
                    d_trans.view());

    raft::copy(
      root_handle_,
      raft::make_host_vector_view(neighbors.data_handle() + output_offset, part_size),
      raft::make_device_vector_view<const searchIdxT>(out_neighbors.data_handle(), part_size));
    raft::copy(root_handle_,
               raft::make_host_vector_view(distances.data_handle() + output_offset, part_size),
               raft::make_device_vector_view<const float>(out_distances.data_handle(), part_size));

    resource::sync_stream(root_handle_);
  }
}

template <typename AnnIndexType, typename T, typename IdxT, typename searchIdxT>
void sharded_search_with_tree_merge(
  const raft::resources& clique,
  const mg_index<AnnIndexType, T, IdxT>& index,
  const cuvs::neighbors::search_params* search_params,
  raft::host_matrix_view<const T, int64_t, row_major> queries,
  raft::host_matrix_view<searchIdxT, int64_t, row_major> neighbors,
  raft::host_matrix_view<float, int64_t, row_major> distances,
  int64_t n_rows_per_batch,
  int64_t n_rows,
  int64_t n_cols,
  int64_t n_neighbors,
  int64_t n_batches)
{
  for (int64_t batch_idx = 0; batch_idx < n_batches; batch_idx++) {
    int64_t offset                  = batch_idx * n_rows_per_batch;
    int64_t query_offset            = offset * n_cols;
    int64_t output_offset           = offset * n_neighbors;
    int64_t n_rows_of_current_batch = std::min((int64_t)n_rows_per_batch, n_rows - offset);
    auto query_partition            = raft::make_host_matrix_view<const T, int64_t, row_major>(
      queries.data_handle() + query_offset, n_rows_of_current_batch, n_cols);

    const int& requirements = index.num_ranks_;
    cuvs::core::omp::check_threads(
      requirements);  // should use at least num_ranks_ threads to avoid NCCL hang
#pragma omp parallel for num_threads(index.num_ranks_)
    for (int rank = 0; rank < index.num_ranks_; rank++) {
      const raft::resources& dev_res = raft::resource::set_current_device_to_rank(clique, rank);
      auto& ann_if                   = index.ann_interfaces_[rank];

      int64_t part_size = n_rows_of_current_batch * n_neighbors;

      auto tmp_neighbors = raft::make_device_matrix<searchIdxT, int64_t, row_major>(
        dev_res, 2 * n_rows_of_current_batch, n_neighbors);
      auto tmp_distances = raft::make_device_matrix<float, int64_t, row_major>(
        dev_res, 2 * n_rows_of_current_batch, n_neighbors);
      auto neighbors_view = raft::make_device_matrix_view<searchIdxT, int64_t, row_major>(
        tmp_neighbors.data_handle(), n_rows_of_current_batch, n_neighbors);
      auto distances_view = raft::make_device_matrix_view<float, int64_t, row_major>(
        tmp_distances.data_handle(), n_rows_of_current_batch, n_neighbors);
      cuvs::neighbors::search(
        dev_res, ann_if, search_params, query_partition, neighbors_view, distances_view);

      searchIdxT translation_offset = 0;
      for (int r = 0; r < rank; r++) {
        translation_offset += index.ann_interfaces_[r].size();
      }
      raft::linalg::addScalar(neighbors_view.data_handle(),
                              neighbors_view.data_handle(),
                              translation_offset,
                              part_size,
                              raft::resource::get_cuda_stream(dev_res));

      auto d_trans = raft::make_device_vector<searchIdxT>(dev_res, 2);
      raft::matrix::fill(dev_res, d_trans.view(), searchIdxT(0));

      int64_t remaining = index.num_ranks_;
      int64_t radix     = 2;

      while (remaining > 1) {
        bool received_something = false;
        int64_t offset          = radix / 2;
        ncclGroupStart();
        if (rank % radix == 0)  // This is one of the receivers
        {
          int other_id = rank + offset;
          if (other_id < index.num_ranks_)  // Make sure someone's sending anything
          {
            ncclRecv(tmp_neighbors.data_handle() + part_size,
                     part_size * sizeof(searchIdxT),
                     ncclUint8,
                     other_id,
                     raft::resource::get_nccl_comm_for_rank(clique, rank),
                     raft::resource::get_cuda_stream(dev_res));
            ncclRecv(tmp_distances.data_handle() + part_size,
                     part_size * sizeof(float),
                     ncclUint8,
                     other_id,
                     raft::resource::get_nccl_comm_for_rank(clique, rank),
                     raft::resource::get_cuda_stream(dev_res));
            received_something = true;
          }
        } else if (rank % radix == offset)  // This is one of the senders
        {
          int other_id = rank - offset;
          ncclSend(tmp_neighbors.data_handle(),
                   part_size * sizeof(searchIdxT),
                   ncclUint8,
                   other_id,
                   raft::resource::get_nccl_comm_for_rank(clique, rank),
                   raft::resource::get_cuda_stream(dev_res));
          ncclSend(tmp_distances.data_handle(),
                   part_size * sizeof(float),
                   ncclUint8,
                   other_id,
                   raft::resource::get_nccl_comm_for_rank(clique, rank),
                   raft::resource::get_cuda_stream(dev_res));
        }
        ncclGroupEnd();

        remaining = (remaining + 1) / 2;
        radix *= 2;

        if (received_something) {
          auto neighbors_merge_res = raft::make_device_matrix<searchIdxT, int64_t, row_major>(
            dev_res, n_rows_of_current_batch, n_neighbors);
          auto distances_merge_res = raft::make_device_matrix<float, int64_t, row_major>(
            dev_res, n_rows_of_current_batch, n_neighbors);
          knn_merge_parts(dev_res,
                          tmp_distances.view(),
                          tmp_neighbors.view(),
                          distances_merge_res.view(),
                          neighbors_merge_res.view(),
                          d_trans.view());
          raft::copy(dev_res,
                     raft::make_device_vector_view(tmp_neighbors.data_handle(), part_size),
                     raft::make_device_vector_view<const searchIdxT>(
                       neighbors_merge_res.data_handle(), part_size));
          raft::copy(dev_res,
                     raft::make_device_vector_view(tmp_distances.data_handle(), part_size),
                     raft::make_device_vector_view<const float>(distances_merge_res.data_handle(),
                                                                part_size));

          // If done, copy the final result
          if (remaining <= 1) {
            raft::copy(
              dev_res,
              raft::make_host_vector_view(neighbors.data_handle() + output_offset, part_size),
              raft::make_device_vector_view<const searchIdxT>(tmp_neighbors.data_handle(),
                                                              part_size));
            raft::copy(
              dev_res,
              raft::make_host_vector_view(distances.data_handle() + output_offset, part_size),
              raft::make_device_vector_view<const float>(tmp_distances.data_handle(), part_size));
            resource::sync_stream(dev_res);
          }
        }
      }
    }
  }
}

template <typename AnnIndexType, typename T, typename IdxT, typename searchIdxT>
void run_search_batch(const raft::resources& clique,
                      const mg_index<AnnIndexType, T, IdxT>& index,
                      int rank,
                      const cuvs::neighbors::search_params* search_params,
                      raft::host_matrix_view<const T, int64_t, row_major>& queries,
                      raft::host_matrix_view<searchIdxT, int64_t, row_major>& neighbors,
                      raft::host_matrix_view<float, int64_t, row_major>& distances,
                      int64_t query_offset,
                      int64_t output_offset,
                      int64_t n_rows_of_current_batch,
                      int64_t n_cols,
                      int64_t n_neighbors)
{
  const raft::resources& dev_res = raft::resource::set_current_device_to_rank(clique, rank);
  auto& ann_if                   = index.ann_interfaces_[rank];

  auto query_partition = raft::make_host_matrix_view<const T, int64_t, row_major>(
    queries.data_handle() + query_offset, n_rows_of_current_batch, n_cols);
  auto d_neighbors = raft::make_device_matrix<searchIdxT, int64_t, row_major>(
    dev_res, n_rows_of_current_batch, n_neighbors);
  auto d_distances = raft::make_device_matrix<float, int64_t, row_major>(
    dev_res, n_rows_of_current_batch, n_neighbors);

  cuvs::neighbors::search(
    dev_res, ann_if, search_params, query_partition, d_neighbors.view(), d_distances.view());

  raft::copy(dev_res,
             raft::make_host_vector_view(neighbors.data_handle() + output_offset,
                                         n_rows_of_current_batch * n_neighbors),
             raft::make_device_vector_view<const searchIdxT>(
               d_neighbors.data_handle(), n_rows_of_current_batch * n_neighbors));
  raft::copy(dev_res,
             raft::make_host_vector_view(distances.data_handle() + output_offset,
                                         n_rows_of_current_batch * n_neighbors),
             raft::make_device_vector_view<const float>(d_distances.data_handle(),
                                                        n_rows_of_current_batch * n_neighbors));

  resource::sync_stream(dev_res);
}

template <typename AnnIndexType, typename T, typename IdxT, typename searchIdxT>
void search(const raft::resources& clique,
            const mg_index<AnnIndexType, T, IdxT>& index,
            const cuvs::neighbors::search_params* search_params,
            raft::host_matrix_view<const T, int64_t, row_major> queries,
            raft::host_matrix_view<searchIdxT, int64_t, row_major> neighbors,
            raft::host_matrix_view<float, int64_t, row_major> distances)
{
  // Making sure that the NCCL comms are instantiated at this stage.
  // This prevents it being done inside of an OpenMP thread.
  raft::resource::get_nccl_comms(clique);

  int64_t n_rows      = queries.extent(0);
  int64_t n_cols      = queries.extent(1);
  int64_t n_neighbors = neighbors.extent(1);

  int64_t n_rows_per_batch = -1;
  if (index.mode_ == REPLICATED) {
    cuvs::neighbors::replicated_search_mode search_mode;
    if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
      const cuvs::neighbors::mg_search_params<ivf_flat::search_params>* mg_search_params =
        static_cast<const cuvs::neighbors::mg_search_params<ivf_flat::search_params>*>(
          search_params);
      search_mode      = mg_search_params->search_mode;
      n_rows_per_batch = mg_search_params->n_rows_per_batch;
    } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
      const cuvs::neighbors::mg_search_params<ivf_pq::search_params>* mg_search_params =
        static_cast<const cuvs::neighbors::mg_search_params<ivf_pq::search_params>*>(search_params);
      search_mode      = mg_search_params->search_mode;
      n_rows_per_batch = mg_search_params->n_rows_per_batch;
    } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, IdxT>>::value) {
      const cuvs::neighbors::mg_search_params<cagra::search_params>* mg_search_params =
        static_cast<const cuvs::neighbors::mg_search_params<cagra::search_params>*>(search_params);
      search_mode      = mg_search_params->search_mode;
      n_rows_per_batch = mg_search_params->n_rows_per_batch;
    }

    if (search_mode == LOAD_BALANCER) {
      int64_t n_rows_per_rank = raft::ceildiv(n_rows, (int64_t)index.num_ranks_);
      n_rows_per_batch =
        std::min(n_rows_per_batch, n_rows_per_rank);  // get at least num_ranks_ batches
      int64_t n_batches = raft::ceildiv(n_rows, (int64_t)n_rows_per_batch);
      if (n_batches <= 1) n_rows_per_batch = n_rows;

      RAFT_LOG_DEBUG(
        "REPLICATED SEARCH IN LOAD BALANCER MODE: %d*%drows", n_batches, n_rows_per_batch);

      cuvs::core::omp::check_threads(index.num_ranks_);
// Each rank gets its own thread; that thread handles all batches for that rank sequentially.
// This prevents concurrent access to the same GPU from multiple threads. (see
// https://github.com/rapidsai/cuvs/issues/1720)
#pragma omp parallel for num_threads(index.num_ranks_)
      for (int rank = 0; rank < index.num_ranks_; rank++) {
        for (int64_t batch_idx = rank; batch_idx < n_batches; batch_idx += index.num_ranks_) {
          int64_t offset                  = batch_idx * n_rows_per_batch;
          int64_t query_offset            = offset * n_cols;
          int64_t output_offset           = offset * n_neighbors;
          int64_t n_rows_of_current_batch = std::min(n_rows_per_batch, n_rows - offset);

          run_search_batch(clique,
                           index,
                           rank,
                           search_params,
                           queries,
                           neighbors,
                           distances,
                           query_offset,
                           output_offset,
                           n_rows_of_current_batch,
                           n_cols,
                           n_neighbors);
        }
      }
    } else if (search_mode == ROUND_ROBIN) {
      RAFT_LOG_DEBUG("REPLICATED SEARCH IN ROUND ROBIN MODE: %d*%drows", 1, n_rows);

      ASSERT(n_rows <= n_rows_per_batch,
             "In round-robin mode, n_rows must lower or equal to n_rows_per_batch");

      auto& rrc    = *index.round_robin_counter_;
      int64_t rank = rrc++;
      rank %= index.num_ranks_;

      run_search_batch(clique,
                       index,
                       rank,
                       search_params,
                       queries,
                       neighbors,
                       distances,
                       0,
                       0,
                       n_rows,
                       n_cols,
                       n_neighbors);
    }
  } else if (index.mode_ == SHARDED) {
    cuvs::neighbors::sharded_merge_mode merge_mode;
    if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
      const cuvs::neighbors::mg_search_params<ivf_flat::search_params>* mg_search_params =
        static_cast<const cuvs::neighbors::mg_search_params<ivf_flat::search_params>*>(
          search_params);
      merge_mode       = mg_search_params->merge_mode;
      n_rows_per_batch = mg_search_params->n_rows_per_batch;
    } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
      const cuvs::neighbors::mg_search_params<ivf_pq::search_params>* mg_search_params =
        static_cast<const cuvs::neighbors::mg_search_params<ivf_pq::search_params>*>(search_params);
      merge_mode       = mg_search_params->merge_mode;
      n_rows_per_batch = mg_search_params->n_rows_per_batch;
    } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, IdxT>>::value) {
      const cuvs::neighbors::mg_search_params<cagra::search_params>* mg_search_params =
        static_cast<const cuvs::neighbors::mg_search_params<cagra::search_params>*>(search_params);
      merge_mode       = mg_search_params->merge_mode;
      n_rows_per_batch = mg_search_params->n_rows_per_batch;
    }

    int64_t n_batches = raft::ceildiv(n_rows, (int64_t)n_rows_per_batch);
    if (n_batches <= 1) n_rows_per_batch = n_rows;

    if (merge_mode == MERGE_ON_ROOT_RANK && index.num_ranks_ > 1) {
      RAFT_LOG_DEBUG("SHARDED SEARCH WITH MERGE_ON_ROOT_RANK MERGE MODE: %d*%drows",
                     n_batches,
                     n_rows_per_batch);
      sharded_search_with_direct_merge(clique,
                                       index,
                                       search_params,
                                       queries,
                                       neighbors,
                                       distances,
                                       n_rows_per_batch,
                                       n_rows,
                                       n_cols,
                                       n_neighbors,
                                       n_batches);
    } else if (merge_mode == TREE_MERGE && index.num_ranks_ > 1) {
      RAFT_LOG_DEBUG(
        "SHARDED SEARCH WITH TREE_MERGE MERGE MODE %d*%drows", n_batches, n_rows_per_batch);
      sharded_search_with_tree_merge(clique,
                                     index,
                                     search_params,
                                     queries,
                                     neighbors,
                                     distances,
                                     n_rows_per_batch,
                                     n_rows,
                                     n_cols,
                                     n_neighbors,
                                     n_batches);
    } else {
      const int rank = 0;
#pragma omp parallel for
      for (int64_t batch_idx = 0; batch_idx < n_batches; batch_idx++) {
        int64_t offset                  = batch_idx * n_rows_per_batch;
        int64_t query_offset            = offset * n_cols;
        int64_t output_offset           = offset * n_neighbors;
        int64_t n_rows_of_current_batch = std::min(n_rows_per_batch, n_rows - offset);

        run_search_batch(clique,
                         index,
                         rank,
                         search_params,
                         queries,
                         neighbors,
                         distances,
                         query_offset,
                         output_offset,
                         n_rows_of_current_batch,
                         n_cols,
                         n_neighbors);
      }
    }
  }
}

template <typename AnnIndexType, typename T, typename IdxT>
void serialize(const raft::resources& clique,
               const mg_index<AnnIndexType, T, IdxT>& index,
               const std::string& filename)
{
  std::ofstream of(filename, std::ios::out | std::ios::binary);
  if (!of) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  std::string dtype_string = raft::numpy_serializer::get_numpy_dtype<T>().to_string();
  dtype_string.resize(4);
  of << dtype_string;

  const auto& handle = raft::resource::set_current_device_to_root_rank(clique);

  serialize_scalar(handle, of, (int)index.mode_);
  serialize_scalar(handle, of, (int)index.num_ranks_);

  for (int rank = 0; rank < index.num_ranks_; rank++) {
    const raft::resources& dev_res = raft::resource::set_current_device_to_rank(clique, rank);
    auto& ann_if                   = index.ann_interfaces_[rank];
    cuvs::neighbors::serialize(dev_res, ann_if, of);
  }

  of.close();
  if (!of) { RAFT_FAIL("Error writing output %s", filename.c_str()); }
}

}  // namespace cuvs::neighbors::snmg::detail

namespace cuvs::neighbors {
using namespace cuvs::neighbors;
using namespace raft;

template <typename AnnIndexType, typename T, typename IdxT>
mg_index<AnnIndexType, T, IdxT>::mg_index(const raft::resources& clique, distribution_mode mode)
  : mode_(mode), round_robin_counter_(std::make_shared<std::atomic<int64_t>>(0))
{
  num_ranks_ = raft::resource::get_num_ranks(clique);
}

template <typename AnnIndexType, typename T, typename IdxT>
mg_index<AnnIndexType, T, IdxT>::mg_index(const raft::resources& clique,
                                          const std::string& filename)
  : round_robin_counter_(std::make_shared<std::atomic<int64_t>>(0))
{
  cuvs::neighbors::snmg::detail::deserialize(clique, *this, filename);
}
}  // namespace cuvs::neighbors
