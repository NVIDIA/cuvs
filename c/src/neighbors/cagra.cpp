/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <cstring>
#include <dlpack/dlpack.h>

#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/numpy_serializer.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/common.h>
#include <cuvs/neighbors/cagra.hpp>

#include "../core/exceptions.hpp"
#include "../core/interop.hpp"

#include "cagra.hpp"
#include <fstream>

namespace {

static void _set_graph_build_params(
  std::variant<std::monostate,
               cuvs::neighbors::cagra::graph_build_params::ivf_pq_params,
               cuvs::neighbors::cagra::graph_build_params::nn_descent_params,
               cuvs::neighbors::cagra::graph_build_params::ace_params,
               cuvs::neighbors::cagra::graph_build_params::iterative_search_params>& out_params,
  cuvsCagraIndexParams& params,
  cuvsCagraGraphBuildAlgo algo,
  int64_t n_rows,
  int64_t dim)

{
  auto metric = static_cast<cuvs::distance::DistanceType>((int)params.metric);
  switch (algo) {
    case cuvsCagraGraphBuildAlgo::AUTO_SELECT: break;
    case cuvsCagraGraphBuildAlgo::IVF_PQ: {
      auto pq_params = cuvs::neighbors::cagra::graph_build_params::ivf_pq_params(
        raft::matrix_extent<int64_t>(n_rows, dim), metric);
      if (params.graph_build_params) {
        auto ivf_params = static_cast<cuvsIvfPqParams*>(params.graph_build_params);
        if (ivf_params->ivf_pq_build_params) {
          auto bp                                         = ivf_params->ivf_pq_build_params;
          pq_params.build_params.add_data_on_build        = bp->add_data_on_build;
          pq_params.build_params.n_lists                  = bp->n_lists;
          pq_params.build_params.kmeans_n_iters           = bp->kmeans_n_iters;
          pq_params.build_params.kmeans_trainset_fraction = bp->kmeans_trainset_fraction;
          pq_params.build_params.pq_bits                  = bp->pq_bits;
          pq_params.build_params.pq_dim                   = bp->pq_dim;
          pq_params.build_params.codebook_kind =
            static_cast<cuvs::neighbors::ivf_pq::codebook_gen>(bp->codebook_kind);
          pq_params.build_params.force_random_rotation = bp->force_random_rotation;
          pq_params.build_params.conservative_memory_allocation =
            bp->conservative_memory_allocation;
          pq_params.build_params.max_train_points_per_pq_code = bp->max_train_points_per_pq_code;
        }
        if (ivf_params->ivf_pq_search_params) {
          auto sp                                          = ivf_params->ivf_pq_search_params;
          pq_params.search_params.n_probes                 = sp->n_probes;
          pq_params.search_params.lut_dtype                = sp->lut_dtype;
          pq_params.search_params.internal_distance_dtype  = sp->internal_distance_dtype;
          pq_params.search_params.preferred_shmem_carveout = sp->preferred_shmem_carveout;
        }
        if (ivf_params->refinement_rate > 1.0f) {
          pq_params.refinement_rate = ivf_params->refinement_rate;
        }
      }
      out_params = pq_params;
      break;
    }
    case cuvsCagraGraphBuildAlgo::NN_DESCENT: {
      auto nn_params =
        cuvs::neighbors::nn_descent::index_params(params.intermediate_graph_degree, metric);
      nn_params.max_iterations = params.nn_descent_niter;
      out_params               = nn_params;
      break;
    }
    case cuvsCagraGraphBuildAlgo::ACE: {
      cuvs::neighbors::cagra::graph_build_params::ace_params ace_p;
      if (params.graph_build_params) {
        auto ace_params_c             = static_cast<cuvsAceParams*>(params.graph_build_params);
        ace_p.npartitions         = ace_params_c->npartitions;
        ace_p.ef_construction     = ace_params_c->ef_construction;
        ace_p.build_dir           = std::string(ace_params_c->build_dir);
        ace_p.use_disk            = ace_params_c->use_disk;
        ace_p.max_host_memory_gb  = ace_params_c->max_host_memory_gb;
        ace_p.max_gpu_memory_gb   = ace_params_c->max_gpu_memory_gb;
      }
      out_params = ace_p;
      break;
    }
    case cuvsCagraGraphBuildAlgo::ITERATIVE_CAGRA_SEARCH: {
      cuvs::neighbors::cagra::graph_build_params::iterative_search_params p;
      out_params = p;
      break;
    }
  }
}

template <typename T>
void* _build(cuvsResources_t res, cuvsCagraIndexParams params, DLManagedTensor* dataset_tensor)
{
  auto dataset = dataset_tensor->dl_tensor;

  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto index   = new cuvs::neighbors::cagra::index<T, uint32_t>(*res_ptr);

  auto index_params = cuvs::neighbors::cagra::index_params();
  convert_c_index_params(params, dataset.shape[0], dataset.shape[1], &index_params);

  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    *index            = cuvs::neighbors::cagra::build(*res_ptr, index_params, mds);
  } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    using mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    *index            = cuvs::neighbors::cagra::build(*res_ptr, index_params, mds);
  }
  return index;
}

template <typename T>
void* _from_args(cuvsResources_t res,
                 cuvsDistanceType _metric,
                 DLManagedTensor* graph_tensor,
                 DLManagedTensor* dataset_tensor)
{
  auto metric  = static_cast<cuvs::distance::DistanceType>((int)_metric);
  auto dataset = dataset_tensor->dl_tensor;
  auto graph   = graph_tensor->dl_tensor;
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  void* index = NULL;
  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    if (cuvs::core::is_dlpack_device_compatible(graph)) {
      using graph_mdspan_type = raft::device_matrix_view<uint32_t const, int64_t, raft::row_major>;
      auto graph_mds          = cuvs::core::from_dlpack<graph_mdspan_type>(graph_tensor);
      index = new cuvs::neighbors::cagra::index<T, uint32_t>(*res_ptr, metric, mds, graph_mds);
    } else {
      using graph_mdspan_type = raft::host_matrix_view<uint32_t const, int64_t, raft::row_major>;
      auto graph_mds          = cuvs::core::from_dlpack<graph_mdspan_type>(graph_tensor);
      index = new cuvs::neighbors::cagra::index<T, uint32_t>(*res_ptr, metric, mds, graph_mds);
    }
  } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    using mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    if (cuvs::core::is_dlpack_device_compatible(graph)) {
      using graph_mdspan_type = raft::device_matrix_view<uint32_t const, int64_t, raft::row_major>;
      auto graph_mds          = cuvs::core::from_dlpack<graph_mdspan_type>(graph_tensor);
      index = new cuvs::neighbors::cagra::index<T, uint32_t>(*res_ptr, metric, mds, graph_mds);
    } else {
      using graph_mdspan_type = raft::host_matrix_view<uint32_t const, int64_t, raft::row_major>;
      auto graph_mds          = cuvs::core::from_dlpack<graph_mdspan_type>(graph_tensor);
      index = new cuvs::neighbors::cagra::index<T, uint32_t>(*res_ptr, metric, mds, graph_mds);
    }
  }
  return index;
}

template <typename T>
void _extend(cuvsResources_t res,
             cuvsCagraExtendParams params,
             cuvsCagraIndex index,
             DLManagedTensor* additional_dataset_tensor)
{
  auto dataset   = additional_dataset_tensor->dl_tensor;
  auto index_ptr = reinterpret_cast<cuvs::neighbors::cagra::index<T, uint32_t>*>(index.addr);
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);

  // TODO: use C struct here (see issue #487)
  auto extend_params           = cuvs::neighbors::cagra::extend_params();
  extend_params.max_chunk_size = params.max_chunk_size;

  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(additional_dataset_tensor);
    cuvs::neighbors::cagra::extend(*res_ptr, extend_params, mds, *index_ptr);
  } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    using mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(additional_dataset_tensor);
    cuvs::neighbors::cagra::extend(*res_ptr, extend_params, mds, *index_ptr);
  } else {
    RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
              dataset.dtype.code,
              dataset.dtype.bits);
  }
}

template <typename T, typename IdxT>
void _search(cuvsResources_t res,
             cuvsCagraSearchParams params,
             cuvsCagraIndex index,
             DLManagedTensor* queries_tensor,
             DLManagedTensor* neighbors_tensor,
             DLManagedTensor* distances_tensor,
             cuvsFilter filter)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::cagra::index<T, uint32_t>*>(index.addr);

  auto search_params = cuvs::neighbors::cagra::search_params();
  convert_c_search_params(params, &search_params);

  using queries_mdspan_type   = raft::device_matrix_view<T const, int64_t, raft::row_major>;
  using neighbors_mdspan_type = raft::device_matrix_view<IdxT, int64_t, raft::row_major>;
  using distances_mdspan_type = raft::device_matrix_view<float, int64_t, raft::row_major>;
  auto queries_mds            = cuvs::core::from_dlpack<queries_mdspan_type>(queries_tensor);
  auto neighbors_mds          = cuvs::core::from_dlpack<neighbors_mdspan_type>(neighbors_tensor);
  auto distances_mds          = cuvs::core::from_dlpack<distances_mdspan_type>(distances_tensor);
  if (filter.type == NO_FILTER) {
    cuvs::neighbors::cagra::search(
      *res_ptr, search_params, *index_ptr, queries_mds, neighbors_mds, distances_mds);
  } else if (filter.type == BITSET) {
    using filter_mdspan_type    = raft::device_vector_view<std::uint32_t, int64_t, raft::row_major>;
    auto removed_indices_tensor = reinterpret_cast<DLManagedTensor*>(filter.addr);
    auto removed_indices = cuvs::core::from_dlpack<filter_mdspan_type>(removed_indices_tensor);
    cuvs::core::bitset_view<std::uint32_t, int64_t> removed_indices_bitset(
      removed_indices, index_ptr->dataset().extent(0));
    auto bitset_filter_obj = cuvs::neighbors::filtering::bitset_filter(removed_indices_bitset);
    cuvs::neighbors::cagra::search(*res_ptr,
                                   search_params,
                                   *index_ptr,
                                   queries_mds,
                                   neighbors_mds,
                                   distances_mds,
                                   bitset_filter_obj);
  } else {
    RAFT_FAIL("Unsupported filter type: BITMAP");
  }
}

template <typename T>
void _search(cuvsResources_t res,
             cuvsCagraSearchParams params,
             cuvsCagraIndex index,
             DLManagedTensor* queries_tensor,
             DLManagedTensor* neighbors_tensor,
             DLManagedTensor* distances_tensor,
             cuvsFilter filter)
{
  if (neighbors_tensor->dl_tensor.dtype.code == kDLUInt &&
      neighbors_tensor->dl_tensor.dtype.bits == 32) {
    _search<T, uint32_t>(
      res, params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
  } else if (neighbors_tensor->dl_tensor.dtype.code == kDLInt &&
             neighbors_tensor->dl_tensor.dtype.bits == 64) {
    _search<T, int64_t>(
      res, params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
  } else {
    RAFT_FAIL("neighbors should be of type uint32_t or int64_t");
  }
}

template <typename T>
void _serialize(cuvsResources_t res,
                const char* filename,
                cuvsCagraIndex_t index,
                bool include_dataset)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::cagra::index<T, uint32_t>*>(index->addr);
  cuvs::neighbors::cagra::serialize(*res_ptr, std::string(filename), *index_ptr, include_dataset);
}

template <typename T>
void _serialize_to_hnswlib(cuvsResources_t res, const char* filename, cuvsCagraIndex_t index)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::cagra::index<T, uint32_t>*>(index->addr);
  cuvs::neighbors::cagra::serialize_to_hnswlib(*res_ptr, std::string(filename), *index_ptr);
}

template <typename T>
void* _deserialize(cuvsResources_t res, const char* filename)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto index   = new cuvs::neighbors::cagra::index<T, uint32_t>(*res_ptr);
  cuvs::neighbors::cagra::deserialize(*res_ptr, std::string(filename), index);
  return index;
}

template <typename T>
void* _merge(cuvsResources_t res,
             cuvsCagraIndexParams params,
             cuvsCagraIndex_t* indices,
             size_t num_indices,
             cuvsFilter filter)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  cuvs::neighbors::cagra::index_params params_cpp;

  params_cpp.metric =
    static_cast<cuvs::distance::DistanceType>((int)params.metric);
  params_cpp.intermediate_graph_degree =
    params.intermediate_graph_degree;
  params_cpp.graph_degree = params.graph_degree;

  int64_t total_size = 0;
  int64_t dim        = 0;
  if (params.build_algo == cuvsCagraGraphBuildAlgo::IVF_PQ) {
    auto first_idx_ptr =
      reinterpret_cast<cuvs::neighbors::cagra::index<T, uint32_t>*>(indices[0]->addr);
    dim = first_idx_ptr->dim();
    for (size_t i = 0; i < num_indices; ++i) {
      auto idx_ptr =
        reinterpret_cast<cuvs::neighbors::cagra::index<T, uint32_t>*>(indices[i]->addr);
      total_size += idx_ptr->size();
    }
  }

  _set_graph_build_params(params_cpp.graph_build_params,
                          params,
                          params.build_algo,
                          total_size,
                          dim);

  std::vector<cuvs::neighbors::cagra::index<T, uint32_t>*> index_ptrs;
  index_ptrs.reserve(num_indices);
  for (size_t i = 0; i < num_indices; ++i) {
    auto idx_ptr = reinterpret_cast<cuvs::neighbors::cagra::index<T, uint32_t>*>(indices[i]->addr);
    index_ptrs.push_back(idx_ptr);
  }

  if (filter.type == NO_FILTER) {
    return new cuvs::neighbors::cagra::index<T, uint32_t>(
      cuvs::neighbors::cagra::merge(*res_ptr, params_cpp, index_ptrs));
  } else if (filter.type == BITSET) {
    using filter_mdspan_type    = raft::device_vector_view<std::uint32_t, int64_t, raft::row_major>;
    auto removed_indices_tensor = reinterpret_cast<DLManagedTensor*>(filter.addr);
    auto removed_indices = cuvs::core::from_dlpack<filter_mdspan_type>(removed_indices_tensor);
    cuvs::core::bitset_view<std::uint32_t, int64_t> removed_indices_bitset(
      removed_indices, total_size);
    auto bitset_filter_obj = cuvs::neighbors::filtering::bitset_filter(removed_indices_bitset);
    return new cuvs::neighbors::cagra::index<T, uint32_t>(
      cuvs::neighbors::cagra::merge(*res_ptr, params_cpp, index_ptrs, bitset_filter_obj));
  } else {
    RAFT_FAIL("Unsupported filter type: BITMAP");
  }
}

template <typename T, typename IdxT>
void get_dataset_view(cuvsCagraIndex_t index, DLManagedTensor* dataset)
{
  auto index_ptr = reinterpret_cast<cuvs::neighbors::cagra::index<T, IdxT>*>(index->addr);
  cuvs::core::to_dlpack(index_ptr->dataset(), dataset);
}

template <typename T, typename IdxT>
void get_graph_view(cuvsCagraIndex_t index, DLManagedTensor* graph)
{
  auto index_ptr = reinterpret_cast<cuvs::neighbors::cagra::index<T, IdxT>*>(index->addr);
  cuvs::core::to_dlpack(index_ptr->graph(), graph);
}

// Helper function to populate C IVF-PQ params from C++ params
static void _populate_c_ivf_pq_params(cuvsIvfPqParams* c_ivf_pq,
                                    const cuvs::neighbors::cagra::graph_build_params::ivf_pq_params& cpp_ivf_pq)
{
  // Populate the IVF-PQ build params
  auto& bp = cpp_ivf_pq.build_params;
  c_ivf_pq->ivf_pq_build_params->metric = static_cast<cuvsDistanceType>(bp.metric);
  c_ivf_pq->ivf_pq_build_params->metric_arg = bp.metric_arg;
  c_ivf_pq->ivf_pq_build_params->add_data_on_build = bp.add_data_on_build;
  c_ivf_pq->ivf_pq_build_params->n_lists = bp.n_lists;
  c_ivf_pq->ivf_pq_build_params->kmeans_n_iters = bp.kmeans_n_iters;
  c_ivf_pq->ivf_pq_build_params->kmeans_trainset_fraction = bp.kmeans_trainset_fraction;
  c_ivf_pq->ivf_pq_build_params->pq_bits = bp.pq_bits;
  c_ivf_pq->ivf_pq_build_params->pq_dim = bp.pq_dim;
  c_ivf_pq->ivf_pq_build_params->codebook_kind = static_cast<cuvsIvfPqCodebookGen>(bp.codebook_kind);
  c_ivf_pq->ivf_pq_build_params->force_random_rotation = bp.force_random_rotation;
  c_ivf_pq->ivf_pq_build_params->conservative_memory_allocation = bp.conservative_memory_allocation;
  c_ivf_pq->ivf_pq_build_params->max_train_points_per_pq_code = bp.max_train_points_per_pq_code;

  // Populate the IVF-PQ search params
  auto& sp = cpp_ivf_pq.search_params;
  c_ivf_pq->ivf_pq_search_params->n_probes = sp.n_probes;
  c_ivf_pq->ivf_pq_search_params->lut_dtype = sp.lut_dtype;
  c_ivf_pq->ivf_pq_search_params->internal_distance_dtype = sp.internal_distance_dtype;
  c_ivf_pq->ivf_pq_search_params->preferred_shmem_carveout = sp.preferred_shmem_carveout;

  c_ivf_pq->refinement_rate = cpp_ivf_pq.refinement_rate;
}

// Helper function to populate C struct from C++ index_params
static void _populate_cagra_index_params_from_cpp(cuvsCagraIndexParams_t c_params,
                                                 const cuvs::neighbors::cagra::index_params& cpp_params)
{
  c_params->metric = static_cast<cuvsDistanceType>(cpp_params.metric);
  c_params->intermediate_graph_degree = cpp_params.intermediate_graph_degree;
  c_params->graph_degree = cpp_params.graph_degree;

  // Set build algo and parameters based on the variant
  if (std::holds_alternative<cuvs::neighbors::cagra::graph_build_params::nn_descent_params>(
        cpp_params.graph_build_params)) {
    c_params->build_algo = NN_DESCENT;
    auto nn_params =
      std::get<cuvs::neighbors::cagra::graph_build_params::nn_descent_params>(
        cpp_params.graph_build_params);
    c_params->nn_descent_niter = nn_params.max_iterations;
  } else if (std::holds_alternative<cuvs::neighbors::cagra::graph_build_params::ivf_pq_params>(
               cpp_params.graph_build_params)) {
    c_params->build_algo = IVF_PQ;
    auto ivf_pq_params =
      std::get<cuvs::neighbors::cagra::graph_build_params::ivf_pq_params>(
        cpp_params.graph_build_params);

    _populate_c_ivf_pq_params(static_cast<cuvsIvfPqParams*>(c_params->graph_build_params), ivf_pq_params);
  } else if (std::holds_alternative<cuvs::neighbors::cagra::graph_build_params::ace_params>(
               cpp_params.graph_build_params)) {
    c_params->build_algo = ACE;
    auto ace_params =
      std::get<cuvs::neighbors::cagra::graph_build_params::ace_params>(
        cpp_params.graph_build_params);
    cuvsAceParams* c_ace_params = new cuvsAceParams;
    c_ace_params->npartitions = ace_params.npartitions;
    c_ace_params->ef_construction = ace_params.ef_construction;
    c_ace_params->build_dir = ace_params.build_dir.empty() ? nullptr : strdup(ace_params.build_dir.c_str());
    c_ace_params->use_disk = ace_params.use_disk;
    c_params->graph_build_params = c_ace_params;
  }
}

}  // namespace

namespace cuvs::neighbors::cagra {
void convert_c_index_params(cuvsCagraIndexParams params,
                            int64_t n_rows,
                            int64_t dim,
                            cuvs::neighbors::cagra::index_params* out)
{
  out->metric                    = static_cast<cuvs::distance::DistanceType>((int)params.metric);
  out->intermediate_graph_degree = params.intermediate_graph_degree;
  out->graph_degree              = params.graph_degree;
  _set_graph_build_params(out->graph_build_params, params, params.build_algo, n_rows, dim);

  if (auto* cparams = params.compression; cparams != nullptr) {
    auto compression_params                        = cuvs::neighbors::vpq_params();
    compression_params.pq_bits                     = cparams->pq_bits;
    compression_params.pq_dim                      = cparams->pq_dim;
    compression_params.vq_n_centers                = cparams->vq_n_centers;
    compression_params.kmeans_n_iters              = cparams->kmeans_n_iters;
    compression_params.vq_kmeans_trainset_fraction = cparams->vq_kmeans_trainset_fraction;
    compression_params.pq_kmeans_trainset_fraction = cparams->pq_kmeans_trainset_fraction;
    out->compression.emplace(compression_params);
  }
}
void convert_c_search_params(cuvsCagraSearchParams params,
                             cuvs::neighbors::cagra::search_params* out)
{
  out->max_queries           = params.max_queries;
  out->itopk_size            = params.itopk_size;
  out->max_iterations        = params.max_iterations;
  out->algo                  = static_cast<cuvs::neighbors::cagra::search_algo>(params.algo);
  out->team_size             = params.team_size;
  out->search_width          = params.search_width;
  out->min_iterations        = params.min_iterations;
  out->thread_block_size     = params.thread_block_size;
  out->hashmap_mode          = static_cast<cuvs::neighbors::cagra::hash_mode>(params.hashmap_mode);
  out->hashmap_min_bitlen    = params.hashmap_min_bitlen;
  out->hashmap_max_fill_rate = params.hashmap_max_fill_rate;
  out->num_random_samplings  = params.num_random_samplings;
  out->rand_xor_mask         = params.rand_xor_mask;
  out->persistent            = params.persistent;
  out->persistent_lifetime   = params.persistent_lifetime;
  out->persistent_device_usage = params.persistent_device_usage;
}
}  // namespace cuvs::neighbors::cagra
extern "C" cuvsError_t cuvsCagraIndexCreate(cuvsCagraIndex_t* index)
{
  return cuvs::core::translate_exceptions([=] { *index = new cuvsCagraIndex{}; });
}

extern "C" cuvsError_t cuvsCagraIndexDestroy(cuvsCagraIndex_t index_c_ptr)
{
  return cuvs::core::translate_exceptions([=] {
    auto index = *index_c_ptr;

    if (index.dtype.code == kDLFloat && index.dtype.bits == 32) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::cagra::index<float, uint32_t>*>(index.addr);
      delete index_ptr;
    } else if (index.dtype.code == kDLFloat && index.dtype.bits == 16) {
      auto index_ptr = reinterpret_cast<cuvs::neighbors::cagra::index<half, uint32_t>*>(index.addr);
      delete index_ptr;
    } else if (index.dtype.code == kDLInt && index.dtype.bits == 8) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::cagra::index<int8_t, uint32_t>*>(index.addr);
      delete index_ptr;
    } else if (index.dtype.code == kDLUInt && index.dtype.bits == 8) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::cagra::index<uint8_t, uint32_t>*>(index.addr);
      delete index_ptr;
    }
    delete index_c_ptr;
  });
}

extern "C" cuvsError_t cuvsCagraIndexGetDims(cuvsCagraIndex_t index, int64_t* dim)
{
  return cuvs::core::translate_exceptions([=] {
    auto index_ptr = reinterpret_cast<cuvs::neighbors::cagra::index<float, uint32_t>*>(index->addr);
    *dim           = index_ptr->dim();
  });
}

extern "C" cuvsError_t cuvsCagraIndexGetSize(cuvsCagraIndex_t index, int64_t* size)
{
  return cuvs::core::translate_exceptions([=] {
    auto index_ptr = reinterpret_cast<cuvs::neighbors::cagra::index<float, uint32_t>*>(index->addr);
    *size          = index_ptr->size();
  });
}

extern "C" cuvsError_t cuvsCagraIndexGetGraphDegree(cuvsCagraIndex_t index, int64_t* graph_degree)
{
  return cuvs::core::translate_exceptions([=] {
    auto index_ptr = reinterpret_cast<cuvs::neighbors::cagra::index<float, uint32_t>*>(index->addr);
    *graph_degree  = index_ptr->graph_degree();
  });
}

extern "C" cuvsError_t cuvsCagraIndexGetDataset(cuvsCagraIndex_t index, DLManagedTensor* dataset)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      get_dataset_view<float, uint32_t>(index, dataset);
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      get_dataset_view<half, uint32_t>(index, dataset);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      get_dataset_view<int8_t, uint32_t>(index, dataset);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      get_dataset_view<uint8_t, uint32_t>(index, dataset);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraIndexGetGraph(cuvsCagraIndex_t index, DLManagedTensor* graph)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      get_graph_view<float, uint32_t>(index, graph);
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      get_graph_view<half, uint32_t>(index, graph);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      get_graph_view<int8_t, uint32_t>(index, graph);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      get_graph_view<uint8_t, uint32_t>(index, graph);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraBuild(cuvsResources_t res,
                                      cuvsCagraIndexParams_t params,
                                      DLManagedTensor* dataset_tensor,
                                      cuvsCagraIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = dataset_tensor->dl_tensor;
    index->dtype = dataset.dtype;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      index->addr = reinterpret_cast<uintptr_t>(_build<float>(res, *params, dataset_tensor));
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 16) {
      index->addr = reinterpret_cast<uintptr_t>(_build<half>(res, *params, dataset_tensor));
    } else if (dataset.dtype.code == kDLInt && dataset.dtype.bits == 8) {
      index->addr = reinterpret_cast<uintptr_t>(_build<int8_t>(res, *params, dataset_tensor));
    } else if (dataset.dtype.code == kDLUInt && dataset.dtype.bits == 8) {
      index->addr = reinterpret_cast<uintptr_t>(_build<uint8_t>(res, *params, dataset_tensor));
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraIndexFromArgs(cuvsResources_t res,
                                              cuvsDistanceType metric,
                                              DLManagedTensor* graph_tensor,
                                              DLManagedTensor* dataset_tensor,
                                              cuvsCagraIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = dataset_tensor->dl_tensor;
    index->dtype = dataset.dtype;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      index->addr =
        reinterpret_cast<uintptr_t>(_from_args<float>(res, metric, graph_tensor, dataset_tensor));
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 16) {
      index->addr =
        reinterpret_cast<uintptr_t>(_from_args<half>(res, metric, graph_tensor, dataset_tensor));
    } else if (dataset.dtype.code == kDLInt && dataset.dtype.bits == 8) {
      index->addr =
        reinterpret_cast<uintptr_t>(_from_args<int8_t>(res, metric, graph_tensor, dataset_tensor));
    } else if (dataset.dtype.code == kDLUInt && dataset.dtype.bits == 8) {
      index->addr =
        reinterpret_cast<uintptr_t>(_from_args<uint8_t>(res, metric, graph_tensor, dataset_tensor));
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraExtend(cuvsResources_t res,
                                       cuvsCagraExtendParams_t params,
                                       DLManagedTensor* additional_dataset_tensor,
                                       cuvsCagraIndex_t index_c_ptr)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = additional_dataset_tensor->dl_tensor;
    auto index   = *index_c_ptr;

    if ((dataset.dtype.code == kDLFloat) && (dataset.dtype.bits == 32)) {
      _extend<float>(res, *params, index, additional_dataset_tensor);
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 16) {
      _extend<half>(res, *params, index, additional_dataset_tensor);
    } else if (dataset.dtype.code == kDLInt && dataset.dtype.bits == 8) {
      _extend<int8_t>(res, *params, index, additional_dataset_tensor);
    } else if (dataset.dtype.code == kDLUInt && dataset.dtype.bits == 8) {
      _extend<uint8_t>(res, *params, index, additional_dataset_tensor);
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraSearch(cuvsResources_t res,
                                       cuvsCagraSearchParams_t params,
                                       cuvsCagraIndex_t index_c_ptr,
                                       DLManagedTensor* queries_tensor,
                                       DLManagedTensor* neighbors_tensor,
                                       DLManagedTensor* distances_tensor,
                                       cuvsFilter filter)
{
  return cuvs::core::translate_exceptions([=] {
    auto queries   = queries_tensor->dl_tensor;
    auto neighbors = neighbors_tensor->dl_tensor;
    auto distances = distances_tensor->dl_tensor;

    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(queries),
                 "queries should have device compatible memory");
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(neighbors),
                 "neighbors should have device compatible memory");
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(distances),
                 "distances should have device compatible memory");

    // NB: the dtype of neighbors is checked later in _search function
    RAFT_EXPECTS(distances.dtype.code == kDLFloat && distances.dtype.bits == 32,
                 "distances should be of type float32");

    auto index = *index_c_ptr;
    RAFT_EXPECTS(queries.dtype.code == index.dtype.code, "type mismatch between index and queries");

    if (queries.dtype.code == kDLFloat && queries.dtype.bits == 32) {
      _search<float>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
    } else if (queries.dtype.code == kDLFloat && queries.dtype.bits == 16) {
      _search<half>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
    } else if (queries.dtype.code == kDLInt && queries.dtype.bits == 8) {
      _search<int8_t>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
    } else if (queries.dtype.code == kDLUInt && queries.dtype.bits == 8) {
      _search<uint8_t>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
    } else {
      RAFT_FAIL("Unsupported queries DLtensor dtype: %d and bits: %d",
                queries.dtype.code,
                queries.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraMerge(cuvsResources_t res,
                                      cuvsCagraIndexParams_t params,
                                      cuvsCagraIndex_t* indices,
                                      size_t num_indices,
                                      cuvsFilter filter,
                                      cuvsCagraIndex_t output_index)
{
  return cuvs::core::translate_exceptions([=] {
    // Basic checks on inputs
    RAFT_EXPECTS(indices != nullptr && num_indices > 0, "indices array cannot be null or empty");
    RAFT_EXPECTS(params != nullptr, "params cannot be null");

    // Use first index dtype as reference
    auto dtype = (*indices[0]).dtype;
    for (size_t i = 1; i < num_indices; ++i) {
      RAFT_EXPECTS((*indices[i]).dtype.code == dtype.code && (*indices[i]).dtype.bits == dtype.bits,
                   "All input indices must have the same data type");
      RAFT_EXPECTS((*indices[i]).addr != 0, "All input indices must be built (non-empty)");
    }
    RAFT_EXPECTS(output_index != nullptr, "Output index pointer must not be null");
    output_index->dtype = dtype;  // output index type matches inputs
    // Dispatch based on data type
    if (dtype.code == kDLFloat && dtype.bits == 32) {
      output_index->addr =
        reinterpret_cast<uintptr_t>(_merge<float>(res, *params, indices, num_indices, filter));
    } else if (dtype.code == kDLFloat && dtype.bits == 16) {
      output_index->addr =
        reinterpret_cast<uintptr_t>(_merge<half>(res, *params, indices, num_indices, filter));
    } else if (dtype.code == kDLInt && dtype.bits == 8) {
      output_index->addr =
        reinterpret_cast<uintptr_t>(_merge<int8_t>(res, *params, indices, num_indices, filter));
    } else if (dtype.code == kDLUInt && dtype.bits == 8) {
      output_index->addr =
        reinterpret_cast<uintptr_t>(_merge<uint8_t>(res, *params, indices, num_indices, filter));
    } else {
      RAFT_FAIL("Unsupported index data type: code=%d, bits=%d", dtype.code, dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraIndexParamsCreate(cuvsCagraIndexParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    *params                       = new cuvsCagraIndexParams{.metric                    = L2Expanded,
                                                             .intermediate_graph_degree = 128,
                                                             .graph_degree              = 64,
                                                             .build_algo                = IVF_PQ,
                                                             .nn_descent_niter          = 20};
    (*params)->graph_build_params = new cuvsIvfPqParams{nullptr, nullptr, 1};
  });
}

extern "C" cuvsError_t cuvsCagraIndexParamsDestroy(cuvsCagraIndexParams_t params)
{
  return cuvs::core::translate_exceptions([=] {
    // Delete graph_build_params based on the build algorithm type
    if (params->graph_build_params != nullptr) {
      switch (params->build_algo) {
      case cuvsCagraGraphBuildAlgo::IVF_PQ:
        delete static_cast<cuvsIvfPqParams *>(params->graph_build_params);
        break;
      case cuvsCagraGraphBuildAlgo::ACE: {
        auto ace_params = static_cast<cuvsAceParams *>(params->graph_build_params);
        // Free the allocated build directory string
        if (ace_params->build_dir) { free(const_cast<char*>(ace_params->build_dir)); }
        delete ace_params;
        break;
      }
      case cuvsCagraGraphBuildAlgo::AUTO_SELECT:
      case cuvsCagraGraphBuildAlgo::NN_DESCENT:
      case cuvsCagraGraphBuildAlgo::ITERATIVE_CAGRA_SEARCH:
        // These algorithms don't have separate parameter structs
        break;
      }
    }
    delete params;
  });
}

extern "C" cuvsError_t cuvsCagraCompressionParamsCreate(cuvsCagraCompressionParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    auto ps = cuvs::neighbors::vpq_params();
    *params =
      new cuvsCagraCompressionParams{.pq_bits                     = ps.pq_bits,
                                     .pq_dim                      = ps.pq_dim,
                                     .vq_n_centers                = ps.vq_n_centers,
                                     .kmeans_n_iters              = ps.kmeans_n_iters,
                                     .vq_kmeans_trainset_fraction = ps.vq_kmeans_trainset_fraction,
                                     .pq_kmeans_trainset_fraction = ps.pq_kmeans_trainset_fraction};
  });
}

extern "C" cuvsError_t cuvsCagraCompressionParamsDestroy(cuvsCagraCompressionParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsAceParamsCreate(cuvsAceParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    auto ps = cuvs::neighbors::cagra::graph_build_params::ace_params();

    // Allocate and copy the build directory string
    const char* build_dir = strdup(ps.build_dir.c_str());

    *params = new cuvsAceParams{.npartitions         = ps.npartitions,
                                .ef_construction     = ps.ef_construction,
                                .build_dir           = build_dir,
                                .use_disk            = ps.use_disk,
                                .max_host_memory_gb  = ps.max_host_memory_gb,
                                .max_gpu_memory_gb   = ps.max_gpu_memory_gb};
  });
}

extern "C" cuvsError_t cuvsAceParamsDestroy(cuvsAceParams_t params)
{
  return cuvs::core::translate_exceptions([=] {
    if (params) {
      // Free the allocated build directory string
      if (params->build_dir) { free(const_cast<char*>(params->build_dir)); }
      delete params;
    }
  });
}

extern "C" cuvsError_t cuvsCagraIndexParamsFromHnswParams(cuvsCagraIndexParams_t params,
                                                           int64_t n_rows,
                                                           int64_t dim,
                                                           int M,
                                                           int ef_construction,
                                                           enum cuvsCagraHnswHeuristicType heuristic,
                                                           cuvsDistanceType metric)
{
  return cuvs::core::translate_exceptions([=] {
    auto cpp_metric = static_cast<cuvs::distance::DistanceType>((int)metric);
    auto cpp_heuristic = static_cast<cuvs::neighbors::cagra::hnsw_heuristic_type>((int)heuristic);
    auto cpp_params = cuvs::neighbors::cagra::index_params::from_hnsw_params(
      raft::matrix_extent<int64_t>(n_rows, dim), M, ef_construction, cpp_heuristic, cpp_metric);

    _populate_cagra_index_params_from_cpp(params, cpp_params);
  });
}

extern "C" cuvsError_t cuvsCagraExtendParamsCreate(cuvsCagraExtendParams_t* params)
{
  return cuvs::core::translate_exceptions(
    [=] { *params = new cuvsCagraExtendParams{.max_chunk_size = 0}; });
}

extern "C" cuvsError_t cuvsCagraExtendParamsDestroy(cuvsCagraExtendParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsCagraSearchParamsCreate(cuvsCagraSearchParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    *params = new cuvsCagraSearchParams{
      .itopk_size              = 64,
      .search_width            = 1,
      .hashmap_max_fill_rate   = 0.5,
      .num_random_samplings    = 1,
      .rand_xor_mask           = 0x128394,
      .persistent              = false,
      .persistent_lifetime     = 2,
      .persistent_device_usage = 1.0,
    };
  });
}

extern "C" cuvsError_t cuvsCagraSearchParamsDestroy(cuvsCagraSearchParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsCagraDeserialize(cuvsResources_t res,
                                            const char* filename,
                                            cuvsCagraIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    // read the numpy dtype from the beginning of the file
    std::ifstream is(filename, std::ios::in | std::ios::binary);
    if (!is) { RAFT_FAIL("Cannot open file %s", filename); }
    char dtype_string[4]{};
    if (!is.read(dtype_string, sizeof(dtype_string))) {
      RAFT_FAIL("Invalid or truncated index header in file %s", filename);
    }
    auto dtype =
      raft::numpy_serializer::parse_descr(std::string(dtype_string, sizeof(dtype_string)));
    is.close();

    index->dtype.bits = dtype.itemsize * 8;
    if (dtype.kind == 'f' && dtype.itemsize == 4) {
      index->addr       = reinterpret_cast<uintptr_t>(_deserialize<float>(res, filename));
      index->dtype.code = kDLFloat;
    } else if (dtype.kind == 'e' && dtype.itemsize == 2) {
      index->addr       = reinterpret_cast<uintptr_t>(_deserialize<half>(res, filename));
      index->dtype.code = kDLFloat;
    } else if (dtype.kind == 'i' && dtype.itemsize == 1) {
      index->addr       = reinterpret_cast<uintptr_t>(_deserialize<int8_t>(res, filename));
      index->dtype.code = kDLInt;
    } else if (dtype.kind == 'u' && dtype.itemsize == 1) {
      index->addr       = reinterpret_cast<uintptr_t>(_deserialize<uint8_t>(res, filename));
      index->dtype.code = kDLUInt;
    } else {
      RAFT_FAIL("Unsupported dtype in file %s", filename);
    }
  });
}

extern "C" cuvsError_t cuvsCagraSerialize(cuvsResources_t res,
                                          const char* filename,
                                          cuvsCagraIndex_t index,
                                          bool include_dataset)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      _serialize<float>(res, filename, index, include_dataset);
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      _serialize<half>(res, filename, index, include_dataset);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      _serialize<int8_t>(res, filename, index, include_dataset);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      _serialize<uint8_t>(res, filename, index, include_dataset);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraSerializeToHnswlib(cuvsResources_t res,
                                                   const char* filename,
                                                   cuvsCagraIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      _serialize_to_hnswlib<float>(res, filename, index);
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      _serialize_to_hnswlib<half>(res, filename, index);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      _serialize_to_hnswlib<int8_t>(res, filename, index);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      _serialize_to_hnswlib<uint8_t>(res, filename, index);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}
