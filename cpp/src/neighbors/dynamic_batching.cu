/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/dynamic_batching.cuh"

#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::neighbors::cagra {

// Single-token names for CUVS_INST_DYNAMIC_BATCHING_INDEX (macro expands Namespace ::__VA_ARGS__).
using cagra_f32_u32_index = device_padded_index<float, uint32_t>;
using cagra_f16_u32_index = device_padded_index<half, uint32_t>;
using cagra_i8_u32_index  = device_padded_index<int8_t, uint32_t>;
using cagra_u8_u32_index  = device_padded_index<uint8_t, uint32_t>;

}  // namespace cuvs::neighbors::cagra

namespace cuvs::neighbors::dynamic_batching {

// NB: the (template) index parameter should be the last; it must be a single preprocessor token
//       into multiple preprocessor token. Then it is consumed as __VA_ARGS__
//
#define CUVS_INST_DYNAMIC_BATCHING_INDEX(T, IdxT, Namespace, ...)                         \
  template <>                                                                             \
  template <>                                                                             \
  CUVS_EXPORT index<T, IdxT>::index<Namespace ::__VA_ARGS__>(                             \
    const raft::resources& res,                                                           \
    const cuvs::neighbors::dynamic_batching::index_params& params,                        \
    const Namespace ::__VA_ARGS__& upstream_index,                                        \
    const typename Namespace ::__VA_ARGS__::search_params_type& upstream_params,          \
    const cuvs::neighbors::filtering::base_filter* sample_filter)                         \
    : runner{new detail::batch_runner<T, IdxT>(                                           \
        res, params, upstream_index, upstream_params, Namespace ::search, sample_filter)} \
  {                                                                                       \
  }

#define CUVS_INST_DYNAMIC_BATCHING_SEARCH(T, IdxT)                                 \
  void search(raft::resources const& res,                                          \
              cuvs::neighbors::dynamic_batching::search_params const& params,      \
              cuvs::neighbors::dynamic_batching::index<T, IdxT> const& index,      \
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries, \
              raft::device_matrix_view<IdxT, int64_t, raft::row_major> neighbors,  \
              raft::device_matrix_view<float, int64_t, raft::row_major> distances) \
  {                                                                                \
    return index.runner->search(res, params, queries, neighbors, distances);       \
  }

// Brute-force search with 64-bit indices
CUVS_INST_DYNAMIC_BATCHING_INDEX(float, int64_t, cuvs::neighbors::brute_force, index<float, float>);

// CAGRA build and search with 32-bit indices
CUVS_INST_DYNAMIC_BATCHING_INDEX(float, uint32_t, cuvs::neighbors::cagra, cagra_f32_u32_index);
CUVS_INST_DYNAMIC_BATCHING_INDEX(half, uint32_t, cuvs::neighbors::cagra, cagra_f16_u32_index);
CUVS_INST_DYNAMIC_BATCHING_INDEX(int8_t, uint32_t, cuvs::neighbors::cagra, cagra_i8_u32_index);
CUVS_INST_DYNAMIC_BATCHING_INDEX(uint8_t, uint32_t, cuvs::neighbors::cagra, cagra_u8_u32_index);

// CAGRA build with 32-bit indices, search with 64-bit indices
CUVS_INST_DYNAMIC_BATCHING_INDEX(float, int64_t, cuvs::neighbors::cagra, cagra_f32_u32_index);
CUVS_INST_DYNAMIC_BATCHING_INDEX(half, int64_t, cuvs::neighbors::cagra, cagra_f16_u32_index);
CUVS_INST_DYNAMIC_BATCHING_INDEX(int8_t, int64_t, cuvs::neighbors::cagra, cagra_i8_u32_index);
CUVS_INST_DYNAMIC_BATCHING_INDEX(uint8_t, int64_t, cuvs::neighbors::cagra, cagra_u8_u32_index);

// IVF-PQ with 64-bit indices
CUVS_INST_DYNAMIC_BATCHING_INDEX(float, int64_t, cuvs::neighbors::ivf_pq, index<int64_t>);
CUVS_INST_DYNAMIC_BATCHING_INDEX(half, int64_t, cuvs::neighbors::ivf_pq, index<int64_t>);
CUVS_INST_DYNAMIC_BATCHING_INDEX(int8_t, int64_t, cuvs::neighbors::ivf_pq, index<int64_t>);
CUVS_INST_DYNAMIC_BATCHING_INDEX(uint8_t, int64_t, cuvs::neighbors::ivf_pq, index<int64_t>);

// IVF-Flat with 64-bit indices
CUVS_INST_DYNAMIC_BATCHING_INDEX(float, int64_t, cuvs::neighbors::ivf_flat, index<float, int64_t>);
CUVS_INST_DYNAMIC_BATCHING_INDEX(int8_t,
                                 int64_t,
                                 cuvs::neighbors::ivf_flat,
                                 index<int8_t, int64_t>);
CUVS_INST_DYNAMIC_BATCHING_INDEX(uint8_t,
                                 int64_t,
                                 cuvs::neighbors::ivf_flat,
                                 index<uint8_t, int64_t>);

CUVS_INST_DYNAMIC_BATCHING_SEARCH(float, int64_t);
CUVS_INST_DYNAMIC_BATCHING_SEARCH(half, int64_t);
CUVS_INST_DYNAMIC_BATCHING_SEARCH(int8_t, int64_t);
CUVS_INST_DYNAMIC_BATCHING_SEARCH(uint8_t, int64_t);
CUVS_INST_DYNAMIC_BATCHING_SEARCH(float, uint32_t);  // uint32_t index type is needed for CAGRA
CUVS_INST_DYNAMIC_BATCHING_SEARCH(half, uint32_t);
CUVS_INST_DYNAMIC_BATCHING_SEARCH(int8_t, uint32_t);
CUVS_INST_DYNAMIC_BATCHING_SEARCH(uint8_t, uint32_t);

#undef CUVS_INST_DYNAMIC_BATCHING_INDEX
#undef CUVS_INST_DYNAMIC_BATCHING_SEARCH

}  // namespace cuvs::neighbors::dynamic_batching
