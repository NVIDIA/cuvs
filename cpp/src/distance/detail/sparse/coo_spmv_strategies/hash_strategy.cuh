/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "base_strategy.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/util/cuda_dev_essentials.cuh>

#include <cuco/static_map.cuh>
#include <cuda/iterator>
#include <thrust/copy.h>

#include <cooperative_groups.h>
#include <rmm/device_uvector.hpp>

// this is needed by cuco as key, value must be bitwise comparable.
// compilers don't declare float/double as bitwise comparable
// but that is too strict
// for example, the following is true (or 0):
// float a = 5;
// float b = 5;
// memcmp(&a, &b, sizeof(float));
CUCO_DECLARE_BITWISE_COMPARABLE(float);
CUCO_DECLARE_BITWISE_COMPARABLE(double);

namespace cuvs {
namespace distance {
namespace detail {
namespace sparse {

template <typename value_idx, typename value_t, int tpb>
class hash_strategy : public coo_spmv_strategy<value_idx, value_t, tpb> {
 public:
  static constexpr value_idx empty_key_sentinel = value_idx{-1};
  static constexpr value_t empty_value_sentinel = value_t{0};
  using probing_scheme_type = cuco::linear_probing<1, cuco::murmurhash3_32<value_idx>>;
  using storage_ref_type =
    cuco::bucket_storage_ref<cuco::pair<value_idx, value_t>, 1, cuco::extent<int>>;
  using map_type = cuco::static_map_ref<value_idx,
                                        value_t,
                                        cuda::thread_scope_block,
                                        cuda::std::equal_to<value_idx>,
                                        probing_scheme_type,
                                        storage_ref_type,
                                        cuco::op::insert_tag,
                                        cuco::op::find_tag>;

  hash_strategy(const distances_config_t<value_idx, value_t>& config_,
                float capacity_threshold_ = 0.5,
                int map_size_             = get_map_size())
    : coo_spmv_strategy<value_idx, value_t, tpb>(config_),
      capacity_threshold(capacity_threshold_),
      map_size(map_size_)
  {
  }

  void chunking_needed(const value_idx* indptr,
                       const value_idx n_rows,
                       rmm::device_uvector<value_idx>& mask_indptr,
                       std::tuple<value_idx, value_idx>& n_rows_divided,
                       cudaStream_t stream)
  {
    auto policy = raft::resource::get_thrust_policy(this->config.handle);

    auto less                   = thrust::copy_if(policy,
                                cuda::make_counting_iterator(value_idx(0)),
                                cuda::make_counting_iterator(n_rows),
                                mask_indptr.data(),
                                fits_in_hash_table(indptr, 0, capacity_threshold * map_size));
    std::get<0>(n_rows_divided) = less - mask_indptr.data();

    auto more = thrust::copy_if(
      policy,
      cuda::make_counting_iterator(value_idx(0)),
      cuda::make_counting_iterator(n_rows),
      less,
      fits_in_hash_table(
        indptr, capacity_threshold * map_size, std::numeric_limits<value_idx>::max()));
    std::get<1>(n_rows_divided) = more - less;
  }

  template <typename product_f, typename accum_f, typename write_f>
  void dispatch(value_t* out_dists,
                value_idx* coo_rows_b,
                product_f product_func,
                accum_f accum_func,
                write_f write_func,
                int chunk_size)
  {
    auto n_blocks_per_row = raft::ceildiv(this->config.b_nnz, chunk_size * tpb);
    rmm::device_uvector<value_idx> mask_indptr(
      this->config.a_nrows, raft::resource::get_cuda_stream(this->config.handle));
    std::tuple<value_idx, value_idx> n_rows_divided;

    chunking_needed(this->config.a_indptr,
                    this->config.a_nrows,
                    mask_indptr,
                    n_rows_divided,
                    raft::resource::get_cuda_stream(this->config.handle).get());

    auto less_rows = std::get<0>(n_rows_divided);
    if (less_rows > 0) {
      mask_row_it<value_idx> less(this->config.a_indptr, less_rows, mask_indptr.data());

      auto n_less_blocks = less_rows * n_blocks_per_row;
      this->_dispatch_base(*this,
                           map_size,
                           less,
                           out_dists,
                           coo_rows_b,
                           product_func,
                           accum_func,
                           write_func,
                           chunk_size,
                           n_less_blocks,
                           n_blocks_per_row);
    }

    auto more_rows = std::get<1>(n_rows_divided);
    if (more_rows > 0) {
      rmm::device_uvector<value_idx> n_chunks_per_row(
        more_rows + 1, raft::resource::get_cuda_stream(this->config.handle));
      rmm::device_uvector<value_idx> chunk_indices(
        0, raft::resource::get_cuda_stream(this->config.handle));
      chunked_mask_row_it<value_idx>::init(
        this->config.a_indptr,
        mask_indptr.data() + less_rows,
        more_rows,
        capacity_threshold * map_size,
        n_chunks_per_row,
        chunk_indices,
        raft::resource::get_cuda_stream(this->config.handle).get());

      chunked_mask_row_it<value_idx> more(
        this->config.a_indptr,
        more_rows,
        mask_indptr.data() + less_rows,
        capacity_threshold * map_size,
        n_chunks_per_row.data(),
        chunk_indices.data(),
        raft::resource::get_cuda_stream(this->config.handle).get());

      auto n_more_blocks = more.total_row_blocks * n_blocks_per_row;
      this->_dispatch_base(*this,
                           map_size,
                           more,
                           out_dists,
                           coo_rows_b,
                           product_func,
                           accum_func,
                           write_func,
                           chunk_size,
                           n_more_blocks,
                           n_blocks_per_row);
    }
  }

  template <typename product_f, typename accum_f, typename write_f>
  void dispatch_rev(value_t* out_dists,
                    value_idx* coo_rows_a,
                    product_f product_func,
                    accum_f accum_func,
                    write_f write_func,
                    int chunk_size)
  {
    auto n_blocks_per_row = raft::ceildiv(this->config.a_nnz, chunk_size * tpb);
    rmm::device_uvector<value_idx> mask_indptr(
      this->config.b_nrows, raft::resource::get_cuda_stream(this->config.handle));
    std::tuple<value_idx, value_idx> n_rows_divided;

    chunking_needed(this->config.b_indptr,
                    this->config.b_nrows,
                    mask_indptr,
                    n_rows_divided,
                    raft::resource::get_cuda_stream(this->config.handle).get());

    auto less_rows = std::get<0>(n_rows_divided);
    if (less_rows > 0) {
      mask_row_it<value_idx> less(this->config.b_indptr, less_rows, mask_indptr.data());

      auto n_less_blocks = less_rows * n_blocks_per_row;
      this->_dispatch_base_rev(*this,
                               map_size,
                               less,
                               out_dists,
                               coo_rows_a,
                               product_func,
                               accum_func,
                               write_func,
                               chunk_size,
                               n_less_blocks,
                               n_blocks_per_row);
    }

    auto more_rows = std::get<1>(n_rows_divided);
    if (more_rows > 0) {
      rmm::device_uvector<value_idx> n_chunks_per_row(
        more_rows + 1, raft::resource::get_cuda_stream(this->config.handle));
      rmm::device_uvector<value_idx> chunk_indices(
        0, raft::resource::get_cuda_stream(this->config.handle));
      chunked_mask_row_it<value_idx>::init(
        this->config.b_indptr,
        mask_indptr.data() + less_rows,
        more_rows,
        capacity_threshold * map_size,
        n_chunks_per_row,
        chunk_indices,
        raft::resource::get_cuda_stream(this->config.handle).get());

      chunked_mask_row_it<value_idx> more(
        this->config.b_indptr,
        more_rows,
        mask_indptr.data() + less_rows,
        capacity_threshold * map_size,
        n_chunks_per_row.data(),
        chunk_indices.data(),
        raft::resource::get_cuda_stream(this->config.handle).get());

      auto n_more_blocks = more.total_row_blocks * n_blocks_per_row;
      this->_dispatch_base_rev(*this,
                               map_size,
                               more,
                               out_dists,
                               coo_rows_a,
                               product_func,
                               accum_func,
                               write_func,
                               chunk_size,
                               n_more_blocks,
                               n_blocks_per_row);
    }
  }

  __device__ inline map_type init_map(void* storage, const value_idx& cache_size)
  {
    auto map_ref =
      map_type{cuco::empty_key<value_idx>{empty_key_sentinel},
               cuco::empty_value<value_t>{empty_value_sentinel},
               cuda::std::equal_to<value_idx>{},
               probing_scheme_type{},
               cuco::cuda_thread_scope<cuda::thread_scope_block>{},
               storage_ref_type{cuco::extent<int>{cache_size},
                                static_cast<typename storage_ref_type::value_type*>(storage)}};
    map_ref.initialize(cooperative_groups::this_thread_block());

    return map_ref;
  }

  __device__ inline void insert(map_type& map_ref, const value_idx& key, const value_t& value)
  {
    map_ref.insert(cuco::pair{key, value});
  }

  // Note: init_find is now merged with init_map since the new API uses the same ref for both
  // operations

  __device__ inline value_t find(map_type& map_ref, const value_idx& key)
  {
    auto a_pair = map_ref.find(key);

    value_t a_col = 0.0;
    if (a_pair != map_ref.end()) { a_col = a_pair->second; }
    return a_col;
  }

  struct fits_in_hash_table {
   public:
    fits_in_hash_table(const value_idx* indptr_, value_idx degree_l_, value_idx degree_r_)
      : indptr(indptr_), degree_l(degree_l_), degree_r(degree_r_)
    {
    }

    __host__ __device__ bool operator()(const value_idx& i)
    {
      auto degree = indptr[i + 1] - indptr[i];

      return degree >= degree_l && degree < degree_r;
    }

   private:
    const value_idx* indptr;
    const value_idx degree_l, degree_r;
  };

  inline static int get_map_size()
  {
    return (raft::getSharedMemPerBlock() - ((tpb / raft::warp_size()) * sizeof(value_t))) /
           sizeof(cuco::pair<value_idx, value_t>);
  }

 private:
  float capacity_threshold;
  int map_size;
};

}  // namespace sparse
}  // namespace detail
}  // namespace distance
}  // namespace cuvs
