/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <optional>
#include <dlpack/dlpack.h>

#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/ivf_sq.h>
#include <cuvs/neighbors/ivf_sq.hpp>

#include "../core/exceptions.hpp"
#include "../core/interop.hpp"

namespace cuvs::neighbors::ivf_sq {
void convert_c_index_params(cuvsIvfSqIndexParams params,
                            cuvs::neighbors::ivf_sq::index_params* out)
{
  out->metric                        = static_cast<cuvs::distance::DistanceType>((int)params.metric);
  out->metric_arg                    = params.metric_arg;
  out->add_data_on_build             = params.add_data_on_build;
  out->n_lists                       = params.n_lists;
  out->kmeans_n_iters                = params.kmeans_n_iters;
  out->max_train_points_per_cluster  = params.max_train_points_per_cluster;
  out->conservative_memory_allocation = params.conservative_memory_allocation;
}
void convert_c_search_params(cuvsIvfSqSearchParams params,
                             cuvs::neighbors::ivf_sq::search_params* out)
{
  out->n_probes = params.n_probes;
}
}  // namespace cuvs::neighbors::ivf_sq

namespace {

using index_type = cuvs::neighbors::ivf_sq::index<uint8_t>;

void _reset_index(cuvsIvfSqIndex_t index)
{
  RAFT_EXPECTS(index != nullptr, "index cannot be null");
  auto index_ptr = reinterpret_cast<index_type*>(index->addr);
  index->addr    = 0;
  index->dtype   = DLDataType{};
  delete index_ptr;
}

template <typename T>
void* _build(cuvsResources_t res, cuvsIvfSqIndexParams params, DLManagedTensor* dataset_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  auto build_params = cuvs::neighbors::ivf_sq::index_params();
  cuvs::neighbors::ivf_sq::convert_c_index_params(params, &build_params);

  auto dataset = dataset_tensor->dl_tensor;

  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type = raft::device_matrix_view<const T, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    return new index_type(cuvs::neighbors::ivf_sq::build(*res_ptr, build_params, mds));
  } else {
    using mdspan_type = raft::host_matrix_view<const T, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    return new index_type(cuvs::neighbors::ivf_sq::build(*res_ptr, build_params, mds));
  }
}

template <typename T>
void _search(cuvsResources_t res,
             cuvsIvfSqSearchParams params,
             cuvsIvfSqIndex index,
             DLManagedTensor* queries_tensor,
             DLManagedTensor* neighbors_tensor,
             DLManagedTensor* distances_tensor,
             cuvsFilter* filter)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_sq::index<uint8_t>*>(index.addr);

  auto search_params = cuvs::neighbors::ivf_sq::search_params();
  cuvs::neighbors::ivf_sq::convert_c_search_params(params, &search_params);

  using queries_mdspan_type   = raft::device_matrix_view<const T, int64_t, raft::row_major>;
  using neighbors_mdspan_type = raft::device_matrix_view<int64_t, int64_t, raft::row_major>;
  using distances_mdspan_type = raft::device_matrix_view<float, int64_t, raft::row_major>;
  auto queries_mds            = cuvs::core::from_dlpack<queries_mdspan_type>(queries_tensor);
  auto neighbors_mds          = cuvs::core::from_dlpack<neighbors_mdspan_type>(neighbors_tensor);
  auto distances_mds          = cuvs::core::from_dlpack<distances_mdspan_type>(distances_tensor);

  if (filter == nullptr || filter->type == NO_FILTER) {
    cuvs::neighbors::ivf_sq::search(
      *res_ptr, search_params, *index_ptr, queries_mds, neighbors_mds, distances_mds);
  } else if (filter->type == BITSET) {
    using filter_mdspan_type    = raft::device_vector_view<std::uint32_t, int64_t, raft::row_major>;
    auto removed_indices_tensor = reinterpret_cast<DLManagedTensor*>(filter->addr);
    auto removed_indices = cuvs::core::from_dlpack<filter_mdspan_type>(removed_indices_tensor);
    cuvs::core::bitset_view<std::uint32_t, int64_t> removed_indices_bitset(removed_indices,
                                                                           index_ptr->size());
    auto bitset_filter_obj = cuvs::neighbors::filtering::bitset_filter(removed_indices_bitset);
    cuvs::neighbors::ivf_sq::search(*res_ptr,
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

void _serialize(cuvsResources_t res, const char* filename, cuvsIvfSqIndex index)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_sq::index<uint8_t>*>(index.addr);
  cuvs::neighbors::ivf_sq::serialize(*res_ptr, std::string(filename), *index_ptr);
}

void* _deserialize(cuvsResources_t res, const char* filename)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto index   = new cuvs::neighbors::ivf_sq::index<uint8_t>(*res_ptr);
  cuvs::neighbors::ivf_sq::deserialize(*res_ptr, std::string(filename), index);
  return index;
}

template <typename T>
void _extend(cuvsResources_t res,
             DLManagedTensor* new_vectors,
             DLManagedTensor* new_indices,
             cuvsIvfSqIndex index)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_sq::index<uint8_t>*>(index.addr);

  bool on_device = cuvs::core::is_dlpack_device_compatible(new_vectors->dl_tensor);
  if (new_indices != nullptr &&
      on_device != cuvs::core::is_dlpack_device_compatible(new_indices->dl_tensor)) {
    RAFT_FAIL("extend inputs must both either be on device memory or host memory");
  }

  if (on_device) {
    using vectors_mdspan_type = raft::device_matrix_view<const T, int64_t, raft::row_major>;
    using indices_mdspan_type = raft::device_vector_view<const int64_t, int64_t>;
    auto vectors_mds          = cuvs::core::from_dlpack<vectors_mdspan_type>(new_vectors);
    std::optional<indices_mdspan_type> indices_mds;
    if (new_indices != nullptr) {
      indices_mds.emplace(cuvs::core::from_dlpack<indices_mdspan_type>(new_indices));
    }
    cuvs::neighbors::ivf_sq::extend(*res_ptr, vectors_mds, indices_mds, index_ptr);
  } else {
    using vectors_mdspan_type = raft::host_matrix_view<const T, int64_t, raft::row_major>;
    using indices_mdspan_type = raft::host_vector_view<const int64_t, int64_t>;
    auto vectors_mds          = cuvs::core::from_dlpack<vectors_mdspan_type>(new_vectors);
    std::optional<indices_mdspan_type> indices_mds;
    if (new_indices != nullptr) {
      indices_mds.emplace(cuvs::core::from_dlpack<indices_mdspan_type>(new_indices));
    }
    cuvs::neighbors::ivf_sq::extend(*res_ptr, vectors_mds, indices_mds, index_ptr);
  }
}

void _get_centers(cuvsIvfSqIndex index, DLManagedTensor* centers)
{
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_sq::index<uint8_t>*>(index.addr);
  cuvs::core::to_dlpack(index_ptr->centers(), centers);
}
}  // namespace

extern "C" cuvsError_t cuvsIvfSqIndexCreate(cuvsIvfSqIndex_t* index)
{
  return cuvs::core::translate_exceptions([=] { *index = new cuvsIvfSqIndex{}; });
}

extern "C" cuvsError_t cuvsIvfSqIndexDestroy(cuvsIvfSqIndex_t index_c_ptr)
{
  return cuvs::core::translate_exceptions([=] {
    _reset_index(index_c_ptr);
    delete index_c_ptr;
  });
}

extern "C" cuvsError_t cuvsIvfSqBuild(cuvsResources_t res,
                                      cuvsIvfSqIndexParams_t params,
                                      DLManagedTensor* dataset_tensor,
                                      cuvsIvfSqIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = dataset_tensor->dl_tensor;

    if (dataset.dtype.code != kDLFloat ||
        (dataset.dtype.bits != 32 && dataset.dtype.bits != 16)) {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }

    _reset_index(index);

    index->dtype.code = dataset.dtype.code;
    index->dtype.bits = dataset.dtype.bits;

    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      index->addr = reinterpret_cast<uintptr_t>(_build<float>(res, *params, dataset_tensor));
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 16) {
      index->addr = reinterpret_cast<uintptr_t>(_build<half>(res, *params, dataset_tensor));
    }
  });
}

static cuvsError_t _cuvsIvfSqSearchImpl(cuvsResources_t res,
                                        cuvsIvfSqSearchParams_t params,
                                        cuvsIvfSqIndex_t index_c_ptr,
                                        DLManagedTensor* queries_tensor,
                                        DLManagedTensor* neighbors_tensor,
                                        DLManagedTensor* distances_tensor,
                                        cuvsFilter* filter)
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

    RAFT_EXPECTS(neighbors.dtype.code == kDLInt && neighbors.dtype.bits == 64,
                 "neighbors should be of type int64_t");
    RAFT_EXPECTS(distances.dtype.code == kDLFloat && distances.dtype.bits == 32,
                 "distances should be of type float32");

    auto index = *index_c_ptr;
    if (queries.dtype.code == kDLFloat && queries.dtype.bits == 32) {
      _search<float>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
    } else if (queries.dtype.code == kDLFloat && queries.dtype.bits == 16) {
      _search<half>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
    } else {
      RAFT_FAIL("Unsupported queries DLtensor dtype: %d and bits: %d",
                queries.dtype.code,
                queries.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsIvfSqSearch(cuvsResources_t res,
                                       cuvsIvfSqSearchParams_t params,
                                       cuvsIvfSqIndex_t index_c_ptr,
                                       DLManagedTensor* queries_tensor,
                                       DLManagedTensor* neighbors_tensor,
                                       DLManagedTensor* distances_tensor,
                                       cuvsFilter filter)
{
  return _cuvsIvfSqSearchImpl(
    res, params, index_c_ptr, queries_tensor, neighbors_tensor, distances_tensor, &filter);
}

extern "C" cuvsError_t cuvsIvfSqIndexParamsCreate(cuvsIvfSqIndexParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    *params = new cuvsIvfSqIndexParams{.metric                         = L2Expanded,
                                       .metric_arg                     = 2.0f,
                                       .add_data_on_build              = true,
                                       .n_lists                        = 1024,
                                       .kmeans_n_iters                 = 20,
                                       .max_train_points_per_cluster    = 256,
                                       .conservative_memory_allocation = false};
  });
}

extern "C" cuvsError_t cuvsIvfSqIndexParamsDestroy(cuvsIvfSqIndexParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsIvfSqSearchParamsCreate(cuvsIvfSqSearchParams_t* params)
{
  return cuvs::core::translate_exceptions(
    [=] { *params = new cuvsIvfSqSearchParams{.n_probes = 20}; });
}

extern "C" cuvsError_t cuvsIvfSqSearchParamsDestroy(cuvsIvfSqSearchParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsIvfSqDeserialize(cuvsResources_t res,
                                            const char* filename,
                                            cuvsIvfSqIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    _reset_index(index);
    index->addr = reinterpret_cast<uintptr_t>(_deserialize(res, filename));
  });
}

extern "C" cuvsError_t cuvsIvfSqSerialize(cuvsResources_t res,
                                          const char* filename,
                                          cuvsIvfSqIndex_t index)
{
  return cuvs::core::translate_exceptions([=] { _serialize(res, filename, *index); });
}

extern "C" cuvsError_t cuvsIvfSqExtend(cuvsResources_t res,
                                       DLManagedTensor* new_vectors,
                                       DLManagedTensor* new_indices,
                                       cuvsIvfSqIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto vectors = new_vectors->dl_tensor;

    if (index->dtype.code == 0 && index->dtype.bits == 0) {
      index->dtype.code = vectors.dtype.code;
      index->dtype.bits = vectors.dtype.bits;
    }

    if (vectors.dtype.code == kDLFloat && vectors.dtype.bits == 32) {
      _extend<float>(res, new_vectors, new_indices, *index);
    } else if (vectors.dtype.code == kDLFloat && vectors.dtype.bits == 16) {
      _extend<half>(res, new_vectors, new_indices, *index);
    } else {
      RAFT_FAIL(
        "Unsupported vectors DLtensor dtype: %d and bits: %d", vectors.dtype.code, vectors.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsIvfSqIndexGetNLists(cuvsIvfSqIndex_t index, int64_t* n_lists)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(index != nullptr, "index cannot be null");
    RAFT_EXPECTS(index->addr != 0, "index must be built before getting n_lists");
    auto index_ptr =
      reinterpret_cast<cuvs::neighbors::ivf_sq::index<uint8_t>*>(index->addr);
    *n_lists = index_ptr->n_lists();
  });
}

extern "C" cuvsError_t cuvsIvfSqIndexGetDim(cuvsIvfSqIndex_t index, int64_t* dim)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(index != nullptr, "index cannot be null");
    RAFT_EXPECTS(index->addr != 0, "index must be built before getting dim");
    auto index_ptr =
      reinterpret_cast<cuvs::neighbors::ivf_sq::index<uint8_t>*>(index->addr);
    *dim = index_ptr->dim();
  });
}

extern "C" cuvsError_t cuvsIvfSqIndexGetSize(cuvsIvfSqIndex_t index, int64_t* size)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(index != nullptr, "index cannot be null");
    RAFT_EXPECTS(index->addr != 0, "index must be built before getting size");
    auto index_ptr =
      reinterpret_cast<cuvs::neighbors::ivf_sq::index<uint8_t>*>(index->addr);
    *size = index_ptr->size();
  });
}

extern "C" cuvsError_t cuvsIvfSqIndexGetCenters(cuvsIvfSqIndex_t index, DLManagedTensor* centers)
{
  return cuvs::core::translate_exceptions([=] { _get_centers(*index, centers); });
}
