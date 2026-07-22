/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "search_multi_kernel_launcher_jit.cuh"

#include "set_value_batch.cuh"

#include "hashmap.hpp"
#include "search_plan.cuh"
#include "topk_for_cagra/topk.h"  //todo replace with raft kernel
#include "utils.hpp"
#include <neighbors/detail/cagra/compute_distance-ext.cuh>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>

#include <cuvs/distance/distance.hpp>

#include <cuvs/neighbors/common.hpp>
#include <cuvs/selection/select_k.hpp>

// TODO: This shouldn't be invoking anything from spatial/knn
#include "../ann_utils.cuh"

#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>  // RAFT_CUDA_TRY_NOT_THROW is used TODO(tfeher): consider moving this to cuda_rt_essentials.hpp

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

namespace cuvs::neighbors::cagra::detail {
namespace multi_kernel_search {

template <class T>
RAFT_KERNEL set_value_kernel(T* const dev_ptr, const T val)
{
  *dev_ptr = val;
}

template <class T>
RAFT_KERNEL set_value_kernel(T* const dev_ptr, const T val, const std::size_t count)
{
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= count) { return; }
  dev_ptr[tid] = val;
}

template <class T>
void set_value(T* const dev_ptr, const T val, cudaStream_t cuda_stream)
{
  set_value_kernel<T><<<1, 1, 0, cuda_stream>>>(dev_ptr, val);
}

template <class T>
void set_value(T* const dev_ptr, const T val, const std::size_t count, cudaStream_t cuda_stream)
{
  constexpr std::uint32_t block_size = 256;
  const auto grid_size               = (count + block_size - 1) / block_size;
  set_value_kernel<T><<<grid_size, block_size, 0, cuda_stream>>>(dev_ptr, val, count);
}

template <class T>
RAFT_KERNEL get_value_kernel(T* const host_ptr, const T* const dev_ptr)
{
  *host_ptr = *dev_ptr;
}

template <class T>
void get_value(T* const host_ptr, const T* const dev_ptr, cudaStream_t cuda_stream)
{
  get_value_kernel<T><<<1, 1, 0, cuda_stream>>>(host_ptr, dev_ptr);
}

template <class T>
auto get_value(const T* const dev_ptr, cudaStream_t stream) -> T
{
  T value;
  RAFT_CUDA_TRY(cudaMemcpyAsync(&value, dev_ptr, sizeof(value), cudaMemcpyDefault, stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  return value;
}

template <class INDEX_T>
RAFT_KERNEL pickup_next_parents_kernel(
  INDEX_T* const parent_candidates_ptr,        // [num_queries, lds]
  const std::size_t lds,                       // (*) lds >= parent_candidates_size
  const std::uint32_t parent_candidates_size,  //
  INDEX_T* const visited_hashmap_ptr,          // [num_queries, 1 << hash_bitlen]
  const std::size_t hash_bitlen,
  const std::uint32_t small_hash_bitlen,
  INDEX_T* const parent_list_ptr,      // [num_queries, ldd]
  const std::size_t ldd,               // (*) ldd >= parent_list_size
  const std::size_t parent_list_size,  //
  std::uint32_t* const terminate_flag)
{
  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;

  const std::size_t ldb   = hashmap::get_size(hash_bitlen);
  const uint32_t query_id = blockIdx.x;
  if (threadIdx.x < 32) {
    // pickup next parents with single warp
    for (std::uint32_t i = threadIdx.x; i < parent_list_size; i += 32) {
      parent_list_ptr[i + (ldd * query_id)] = utils::get_max_value<INDEX_T>();
    }
    std::uint32_t parent_candidates_size_max = parent_candidates_size;
    if (parent_candidates_size % 32) {
      parent_candidates_size_max += 32 - (parent_candidates_size % 32);
    }
    std::uint32_t num_new_parents = 0;
    for (std::uint32_t j = threadIdx.x; j < parent_candidates_size_max; j += 32) {
      INDEX_T index;
      int new_parent = 0;
      if (j < parent_candidates_size) {
        index = parent_candidates_ptr[j + (lds * query_id)];
        if ((index & index_msb_1_mask) == 0) {  // check most significant bit
          new_parent = 1;
        }
      }
      const std::uint32_t ballot_mask = __ballot_sync(0xffffffff, new_parent);
      if (new_parent) {
        const auto i = __popc(ballot_mask & ((1 << threadIdx.x) - 1)) + num_new_parents;
        if (i < parent_list_size) {
          parent_list_ptr[i + (ldd * query_id)] = j;
          parent_candidates_ptr[j + (lds * query_id)] |=
            index_msb_1_mask;  // set most significant bit as used node
        }
      }
      num_new_parents += __popc(ballot_mask);
      if (num_new_parents >= parent_list_size) { break; }
    }
    if ((num_new_parents > 0) && (threadIdx.x == 0)) { *terminate_flag = 0; }
  } else if (small_hash_bitlen) {
    // reset small-hash
    hashmap::init(visited_hashmap_ptr + (ldb * query_id), hash_bitlen, 32);
  }

  if (small_hash_bitlen) {
    __syncthreads();
    // insert internal-topk indices into small-hash
    for (unsigned i = threadIdx.x; i < parent_candidates_size; i += blockDim.x) {
      auto key = parent_candidates_ptr[i + (lds * query_id)] &
                 ~index_msb_1_mask;  // clear most significant bit
      hashmap::insert(visited_hashmap_ptr + (ldb * query_id), hash_bitlen, key);
    }
  }
}

template <class INDEX_T>
void pickup_next_parents(INDEX_T* const parent_candidates_ptr,  // [num_queries, lds]
                         const std::size_t lds,                 // (*) lds >= parent_candidates_size
                         const std::size_t parent_candidates_size,  //
                         const std::size_t num_queries,
                         INDEX_T* const visited_hashmap_ptr,  // [num_queries, 1 << hash_bitlen]
                         const std::size_t hash_bitlen,
                         const std::size_t small_hash_bitlen,
                         INDEX_T* const parent_list_ptr,      // [num_queries, ldd]
                         const std::size_t ldd,               // (*) ldd >= parent_list_size
                         const std::size_t parent_list_size,  //
                         std::uint32_t* const terminate_flag,
                         cudaStream_t cuda_stream = 0)
{
  std::uint32_t block_size = 32;
  if (small_hash_bitlen) {
    block_size = 128;
    while (parent_candidates_size > block_size) {
      block_size *= 2;
    }
    block_size = min(block_size, (uint32_t)512);
  }
  pickup_next_parents_kernel<INDEX_T>
    <<<num_queries, block_size, 0, cuda_stream>>>(parent_candidates_ptr,
                                                  lds,
                                                  parent_candidates_size,
                                                  visited_hashmap_ptr,
                                                  hash_bitlen,
                                                  small_hash_bitlen,
                                                  parent_list_ptr,
                                                  ldd,
                                                  parent_list_size,
                                                  terminate_flag);
}

template <class INDEX_T>
RAFT_KERNEL remove_parent_bit_kernel(const std::uint32_t num_queries,
                                     const std::uint32_t num_topk,
                                     INDEX_T* const topk_indices_ptr,  // [ld, num_queries]
                                     const std::uint32_t ld)
{
  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;

  uint32_t i_query = blockIdx.x;
  if (i_query >= num_queries) return;

  for (unsigned i = threadIdx.x; i < num_topk; i += blockDim.x) {
    topk_indices_ptr[i + (ld * i_query)] &= ~index_msb_1_mask;  // clear most significant bit
  }
}

template <class INDEX_T>
void remove_parent_bit(const std::uint32_t num_queries,
                       const std::uint32_t num_topk,
                       INDEX_T* const topk_indices_ptr,  // [ld, num_queries]
                       const std::uint32_t ld,
                       cudaStream_t cuda_stream = 0)
{
  const std::size_t grid_size  = num_queries;
  const std::size_t block_size = 256;
  remove_parent_bit_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
    num_queries, num_topk, topk_indices_ptr, ld);
}

template <class T, class S>
RAFT_KERNEL batched_memcpy_kernel(T* const dst,  // [batch_size, ld_dst]
                                  const uint64_t ld_dst,
                                  const S* const src,  // [batch_size, ld_src]
                                  const uint64_t ld_src,
                                  const uint64_t count,
                                  const uint64_t batch_size)
{
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= count * batch_size) { return; }
  const auto i          = tid % count;
  const auto j          = tid / count;
  dst[i + (ld_dst * j)] = src[i + (ld_src * j)];
}

template <class T, class S>
void batched_memcpy(T* const dst,  // [batch_size, ld_dst]
                    const uint64_t ld_dst,
                    const S* const src,  // [batch_size, ld_src]
                    const uint64_t ld_src,
                    const uint64_t count,
                    const uint64_t batch_size,
                    cudaStream_t cuda_stream)
{
  assert(ld_dst >= count);
  assert(ld_src >= count);
  constexpr uint32_t block_size = 256;
  const auto grid_size          = (batch_size * count + block_size - 1) / block_size;
  batched_memcpy_kernel<T, S>
    <<<grid_size, block_size, 0, cuda_stream>>>(dst, ld_dst, src, ld_src, count, batch_size);
}

// result_buffer (work buffer) for "multi-kernel"
// +--------------------+------------------------------+-------------------+
// | internal_top_k (A) | neighbors of internal_top_k  | internal_topk (B) |
// | <itopk_size>       | <search_width * graph_degree> | <itopk_size>      |
// +--------------------+------------------------------+-------------------+
// |<---                 result_buffer_allocation_size                 --->|
// |<---                       result_buffer_size  --->|                     // Double buffer (A)
//                      |<---  result_buffer_size                      --->| // Double buffer (B)
template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SAMPLE_FILTER_T,
          typename SourceIndexT = IndexT,
          typename OutputIndexT = SourceIndexT>
struct search
  : search_plan_impl<DataT, IndexT, DistanceT, SAMPLE_FILTER_T, SourceIndexT, OutputIndexT> {
  using base_type =
    search_plan_impl<DataT, IndexT, DistanceT, SAMPLE_FILTER_T, SourceIndexT, OutputIndexT>;
  using DATA_T     = typename base_type::DATA_T;
  using INDEX_T    = typename base_type::INDEX_T;
  using DISTANCE_T = typename base_type::DISTANCE_T;

  static_assert(std::is_same_v<DISTANCE_T, float>, "Only float is supported as resulting distance");

  using base_type::algo;
  using base_type::hashmap_max_fill_rate;
  using base_type::hashmap_min_bitlen;
  using base_type::hashmap_mode;
  using base_type::itopk_size;
  using base_type::max_iterations;
  using base_type::max_queries;
  using base_type::min_iterations;
  using base_type::num_random_samplings;
  using base_type::rand_xor_mask;
  using base_type::search_width;
  using base_type::team_size;
  using base_type::thread_block_size;

  using base_type::dim;
  using base_type::graph_degree;
  using base_type::topk;

  using base_type::hash_bitlen;

  using base_type::dataset_size;
  using base_type::hashmap_size;
  using base_type::result_buffer_size;
  using base_type::small_hash_bitlen;
  using base_type::small_hash_reset_interval;

  using base_type::smem_size;

  using base_type::dataset_desc;
  using base_type::dev_seed;
  using base_type::hashmap;
  using base_type::num_executed_iterations;
  using base_type::num_seeds;

  size_t result_buffer_allocation_size;
  rmm::device_uvector<INDEX_T> result_indices;       // results_indices_buffer
  rmm::device_uvector<DISTANCE_T> result_distances;  // result_distances_buffer
  rmm::device_uvector<INDEX_T> parent_node_list;
  rmm::device_uvector<uint32_t> topk_hint;
  rmm::device_uvector<uint32_t> terminate_flag;  // dev_terminate_flag, host_terminate_flag.;
  rmm::device_uvector<uint32_t> topk_workspace;

  // temporary storage for _find_topk
  rmm::device_uvector<float> input_keys_storage;
  rmm::device_uvector<float> output_keys_storage;
  rmm::device_uvector<INDEX_T> input_values_storage;
  rmm::device_uvector<INDEX_T> output_values_storage;

  search(raft::resources const& res,
         search_params params,
         const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
         int64_t dim,
         int64_t dataset_size,
         int64_t graph_degree,
         uint32_t topk)
    : base_type(res, params, dataset_desc, dim, dataset_size, graph_degree, topk),
      result_indices(0, raft::resource::get_cuda_stream(res)),
      result_distances(0, raft::resource::get_cuda_stream(res)),
      parent_node_list(0, raft::resource::get_cuda_stream(res)),
      topk_hint(0, raft::resource::get_cuda_stream(res)),
      topk_workspace(0, raft::resource::get_cuda_stream(res)),
      terminate_flag(0, raft::resource::get_cuda_stream(res)),
      input_keys_storage(0, raft::resource::get_cuda_stream(res)),
      output_keys_storage(0, raft::resource::get_cuda_stream(res)),
      input_values_storage(0, raft::resource::get_cuda_stream(res)),
      output_values_storage(0, raft::resource::get_cuda_stream(res))
  {
    set_params(res);
  }

  void set_params(raft::resources const& res)
  {
    //
    // Allocate memory for intermediate buffer and workspace.
    //
    result_buffer_size            = itopk_size + (search_width * graph_degree);
    result_buffer_allocation_size = result_buffer_size + itopk_size;
    result_indices.resize(result_buffer_allocation_size * max_queries,
                          raft::resource::get_cuda_stream(res));
    result_distances.resize(result_buffer_allocation_size * max_queries,
                            raft::resource::get_cuda_stream(res));

    parent_node_list.resize(max_queries * search_width, raft::resource::get_cuda_stream(res));
    topk_hint.resize(max_queries, raft::resource::get_cuda_stream(res));

    size_t topk_workspace_size = _cuann_find_topk_bufferSize(
      itopk_size, max_queries, result_buffer_size, utils::get_cuda_data_type<DATA_T>());
    RAFT_LOG_DEBUG("# topk_workspace_size: %lu", topk_workspace_size);
    topk_workspace.resize(topk_workspace_size, raft::resource::get_cuda_stream(res));
    terminate_flag.resize(1, raft::resource::get_cuda_stream(res));
    hashmap.resize(hashmap_size, raft::resource::get_cuda_stream(res));
  }

  ~search() {}

  inline void _find_topk(raft::resources const& handle,
                         uint32_t topK,
                         uint32_t sizeBatch,
                         uint32_t numElements,
                         const float* inputKeys,    // [sizeBatch, ldIK,]
                         uint32_t ldIK,             // (*) ldIK >= numElements
                         const INDEX_T* inputVals,  // [sizeBatch, ldIV,]
                         uint32_t ldIV,             // (*) ldIV >= numElements
                         float* outputKeys,         // [sizeBatch, ldOK,]
                         uint32_t ldOK,             // (*) ldOK >= topK
                         INDEX_T* outputVals,       // [sizeBatch, ldOV,]
                         uint32_t ldOV,             // (*) ldOV >= topK
                         void* workspace,
                         bool sort,
                         uint32_t* hints)
  {
    auto stream = raft::resource::get_cuda_stream(handle);

    // _cuann_find_topk right now is limited to a max-k of 1024.
    // RAFT has a matrix::select_k function - which handles arbitrary sized values of k,
    // but doesn't accept strided inputs unlike _cuann_find_topk
    // The multi-kernel search path requires strided access - since its cleverly allocating memory
    // (layout described in the search_plan_impl function below), such that both the
    // neighbors and the internal_topk are adjacent - in a double buffered format.
    // Since this layout doesn't work with the matrix::select_k code - we have to copy
    // over to a contiguous (non-strided) access to handle topk larger than 1024, and
    // potentially also copy back to a strided layout afterwards
    if (topK <= 1024) {
      return _cuann_find_topk(topK,
                              sizeBatch,
                              numElements,
                              inputKeys,
                              ldIK,
                              inputVals,
                              ldIV,
                              outputKeys,
                              ldOK,
                              outputVals,
                              ldOV,
                              workspace,
                              sort,
                              hints,
                              stream);
    }

    if (ldIK > numElements) {
      if (input_keys_storage.size() != sizeBatch * numElements) {
        input_keys_storage.resize(sizeBatch * numElements, stream);
      }
      batched_memcpy(
        input_keys_storage.data(), numElements, inputKeys, ldIK, numElements, sizeBatch, stream);
      inputKeys = input_keys_storage.data();
    }

    if (ldIV > numElements) {
      if (input_values_storage.size() != sizeBatch * numElements) {
        input_values_storage.resize(sizeBatch * numElements, stream);
      }

      batched_memcpy(
        input_values_storage.data(), numElements, inputVals, ldIV, numElements, sizeBatch, stream);
      inputVals = input_values_storage.data();
    }

    if ((ldOK > topK) && (output_keys_storage.size() != sizeBatch * topK)) {
      output_keys_storage.resize(sizeBatch * topK, stream);
    }

    if ((ldOV > topK) && (output_values_storage.size() != sizeBatch * topK)) {
      output_values_storage.resize(sizeBatch * topK, stream);
    }

    cuvs::selection::select_k(
      handle,
      raft::make_device_matrix_view<const float, int64_t>(inputKeys, sizeBatch, numElements),
      raft::make_device_matrix_view<const INDEX_T, int64_t>(inputVals, sizeBatch, numElements),
      raft::make_device_matrix_view<float, int64_t>(
        ldOK > topK ? output_keys_storage.data() : outputKeys, sizeBatch, topK),
      raft::make_device_matrix_view<INDEX_T, int64_t>(
        ldOV > topK ? output_values_storage.data() : outputVals, sizeBatch, topK),
      true,  // select_min
      sort);

    if (ldOK > topK) {
      batched_memcpy(outputKeys, ldOK, output_keys_storage.data(), topK, topK, sizeBatch, stream);
    }

    if (ldOV > topK) {
      batched_memcpy(outputVals, ldOV, output_values_storage.data(), topK, topK, sizeBatch, stream);
    }
  }

  void operator()(
    raft::resources const& res,
    raft::device_matrix_view<const INDEX_T, int64_t, raft::row_major> graph,
    std::optional<raft::device_vector_view<const SourceIndexT, int64_t>> source_indices,
    OutputIndexT* const topk_indices_ptr,  // [num_queries, topk]
    DISTANCE_T* const topk_distances_ptr,  // [num_queries, topk]
    const DATA_T* const queries_ptr,       // [num_queries, dataset_dim]
    const uint32_t num_queries,
    const INDEX_T* dev_seed_ptr,              // [num_queries, num_seeds]
    uint32_t* const num_executed_iterations,  // [num_queries,]
    uint32_t topk,
    SAMPLE_FILTER_T sample_filter)
  {
    // Init hashmap
    cudaStream_t stream      = raft::resource::get_cuda_stream(res);
    const uint32_t hash_size = hashmap::get_size(hash_bitlen);
    set_value_batch(
      hashmap.data(), hash_size, utils::get_max_value<INDEX_T>(), hash_size, num_queries, stream);

    auto source_indices_ptr = source_indices.has_value() ? source_indices->data_handle() : nullptr;

    // Topk hint can not be used when applying a filter
    uint32_t* const top_hint_ptr =
      std::is_same<SAMPLE_FILTER_T, cuvs::neighbors::filtering::none_sample_filter>::value
        ? topk_hint.data()
        : nullptr;
    // Init topk_hint
    if (top_hint_ptr != nullptr && topk_hint.size() > 0) {
      set_value(top_hint_ptr, 0xffffffffu, num_queries, stream);
    }

    // Choose initial entry point candidates at random
    random_pickup_jit(dataset_desc,
                      queries_ptr,
                      num_queries,
                      result_buffer_size,
                      num_random_samplings,
                      rand_xor_mask,
                      dev_seed_ptr,
                      num_seeds,
                      result_indices.data(),
                      result_distances.data(),
                      result_buffer_allocation_size,
                      hashmap.data(),
                      hash_bitlen,
                      stream,
                      // Bound random seed selection to the graph size, not the dataset size.
                      // During iterative / CAGRA-Q build the graph is smaller than the dataset,
                      // so using dataset_size here selects seeds that index past the graph end
                      // (out-of-bounds access). See https://github.com/rapidsai/cuvs/pull/1780.
                      static_cast<IndexT>(graph.extent(0)));

    std::shared_ptr<AlgorithmLauncher> compute_distance_to_child_nodes_launcher =
      make_cagra_multi_kernel_jit_launcher<DATA_T,
                                           INDEX_T,
                                           DISTANCE_T,
                                           SourceIndexT,
                                           sample_filter_jit_tag_t<SAMPLE_FILTER_T>>(
        dataset_desc,
        "compute_distance_to_child_nodes",
        make_cagra_sample_filter_udf_fragment<SourceIndexT>(sample_filter));

    unsigned iter = 0;
    while (1) {
      // Make an index list of internal top-k nodes
      _find_topk(res,
                 itopk_size,
                 num_queries,
                 result_buffer_size,
                 result_distances.data() + (iter & 0x1) * itopk_size,
                 result_buffer_allocation_size,
                 result_indices.data() + (iter & 0x1) * itopk_size,
                 result_buffer_allocation_size,
                 result_distances.data() + (1 - (iter & 0x1)) * result_buffer_size,
                 result_buffer_allocation_size,
                 result_indices.data() + (1 - (iter & 0x1)) * result_buffer_size,
                 result_buffer_allocation_size,
                 topk_workspace.data(),
                 true,
                 top_hint_ptr);

      // termination (1)
      if ((iter + 1 == max_iterations)) {
        iter++;
        break;
      }

      if (iter + 1 >= min_iterations) { set_value<uint32_t>(terminate_flag.data(), 1, stream); }

      // pickup parent nodes
      uint32_t _small_hash_bitlen = 0;
      if ((iter + 1) % small_hash_reset_interval == 0) { _small_hash_bitlen = small_hash_bitlen; }

      pickup_next_parents(result_indices.data() + (1 - (iter & 0x1)) * result_buffer_size,
                          result_buffer_allocation_size,
                          itopk_size,
                          num_queries,
                          hashmap.data(),
                          hash_bitlen,
                          _small_hash_bitlen,
                          parent_node_list.data(),
                          search_width,
                          search_width,
                          terminate_flag.data(),
                          stream);

      // termination (2)
      if (iter + 1 >= min_iterations) {
        if (get_value(terminate_flag.data(), stream)) {
          iter++;
          break;
        }
      }

      // Compute distance to child nodes that are adjacent to the parent node
      compute_distance_to_child_nodes_jit(
        parent_node_list.data(),
        result_indices.data() + (1 - (iter & 0x1)) * result_buffer_size,
        result_distances.data() + (1 - (iter & 0x1)) * result_buffer_size,
        result_buffer_allocation_size,
        search_width,
        dataset_desc,
        graph.data_handle(),
        graph.extent(1),
        source_indices_ptr,
        queries_ptr,
        num_queries,
        hashmap.data(),
        hash_bitlen,
        result_indices.data() + itopk_size,
        result_distances.data() + itopk_size,
        result_buffer_allocation_size,
        sample_filter,
        stream,
        compute_distance_to_child_nodes_launcher);

      iter++;
    }  // while ( 1 )
    auto result_indices_ptr   = result_indices.data() + (iter & 0x1) * result_buffer_size;
    auto result_distances_ptr = result_distances.data() + (iter & 0x1) * result_buffer_size;

    if constexpr (!std::is_same<SAMPLE_FILTER_T,
                                cuvs::neighbors::filtering::none_sample_filter>::value) {
      // Remove parent bit in search results
      remove_parent_bit(num_queries,
                        result_buffer_size,
                        result_indices.data() + (iter & 0x1) * itopk_size,
                        result_buffer_allocation_size,
                        stream);

      std::shared_ptr<AlgorithmLauncher> apply_filter_launcher =
        make_cagra_apply_filter_jit_launcher<DATA_T,
                                             INDEX_T,
                                             DISTANCE_T,
                                             SourceIndexT,
                                             sample_filter_jit_tag_t<SAMPLE_FILTER_T>>(
          dataset_desc, make_cagra_sample_filter_udf_fragment<SourceIndexT>(sample_filter));

      apply_filter_jit<INDEX_T, DISTANCE_T, SourceIndexT, SAMPLE_FILTER_T>(
        source_indices_ptr,
        result_indices.data() + (iter & 0x1) * itopk_size,
        result_distances.data() + (iter & 0x1) * itopk_size,
        result_buffer_allocation_size,
        result_buffer_size,
        num_queries,
        0,
        sample_filter,
        stream,
        apply_filter_launcher);

      result_indices_ptr   = result_indices.data() + (1 - (iter & 0x1)) * result_buffer_size;
      result_distances_ptr = result_distances.data() + (1 - (iter & 0x1)) * result_buffer_size;
      _find_topk(res,
                 itopk_size,
                 num_queries,
                 result_buffer_size,
                 result_distances.data() + (iter & 0x1) * itopk_size,
                 result_buffer_allocation_size,
                 result_indices.data() + (iter & 0x1) * itopk_size,
                 result_buffer_allocation_size,
                 result_distances_ptr,
                 result_buffer_allocation_size,
                 result_indices_ptr,
                 result_buffer_allocation_size,
                 topk_workspace.data(),
                 true,
                 top_hint_ptr);
    } else {
      // Remove parent bit in search results
      remove_parent_bit(
        num_queries, itopk_size, result_indices_ptr, result_buffer_allocation_size, stream);
    }

    // Copy results from working buffer to final buffer
    if (source_indices_ptr != nullptr) {
      auto buf_result_indices = raft::make_device_matrix_view<const INDEX_T, int64_t>(
        result_indices_ptr, num_queries, result_buffer_allocation_size);
      raft::linalg::map_offset(
        res,
        raft::make_device_matrix_view<OutputIndexT, int64_t>(topk_indices_ptr, num_queries, topk),
        [source_indices_ptr, buf_result_indices, topk] __device__(auto offset) {
          auto i = offset / topk;
          auto j = offset % topk;
          return static_cast<OutputIndexT>(source_indices_ptr[buf_result_indices(i, j)]);
        });
    } else {
      batched_memcpy(topk_indices_ptr,
                     topk,
                     result_indices_ptr,
                     result_buffer_allocation_size,
                     topk,
                     num_queries,
                     stream);
    }
    if (topk_distances_ptr) {
      batched_memcpy(topk_distances_ptr,
                     topk,
                     result_distances_ptr,
                     result_buffer_allocation_size,
                     topk,
                     num_queries,
                     stream);
    }

    if (num_executed_iterations) {
      for (std::uint32_t i = 0; i < num_queries; i++) {
        num_executed_iterations[i] = iter;
      }
    }
  }
};

}  // namespace multi_kernel_search
}  // namespace cuvs::neighbors::cagra::detail
