
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <dlpack/dlpack.h>

#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/numpy_serializer.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/ivf_flat.h>
#include <cuvs/neighbors/ivf_flat.hpp>

#include "../core/exceptions.hpp"
#include "../core/interop.hpp"

#include <fstream>

namespace cuvs::neighbors::ivf_flat {
void convert_c_index_params(cuvsIvfFlatIndexParams params,
                            cuvs::neighbors::ivf_flat::index_params* out)
{
  out->metric                   = static_cast<cuvs::distance::DistanceType>((int)params.metric),
  out->metric_arg               = params.metric_arg;
  out->add_data_on_build        = params.add_data_on_build;
  out->n_lists                  = params.n_lists;
  out->kmeans_n_iters           = params.kmeans_n_iters;
  out->kmeans_trainset_fraction = params.kmeans_trainset_fraction;
  out->adaptive_centers         = params.adaptive_centers;
  out->conservative_memory_allocation = params.conservative_memory_allocation;
}
void convert_c_search_params(cuvsIvfFlatSearchParams params,
                             cuvs::neighbors::ivf_flat::search_params* out)
{
  out->n_probes = params.n_probes;
}
}  // namespace cuvs::neighbors::ivf_flat

namespace {

template <typename T, typename IdxT>
void* _build(cuvsResources_t res, cuvsIvfFlatIndexParams params, DLManagedTensor* dataset_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  auto build_params = cuvs::neighbors::ivf_flat::index_params();
  cuvs::neighbors::ivf_flat::convert_c_index_params(params, &build_params);

  auto dataset = dataset_tensor->dl_tensor;
  auto dim     = dataset.shape[1];

  auto index = new cuvs::neighbors::ivf_flat::index<T, IdxT>(*res_ptr, build_params, dim);

  using mdspan_type = raft::device_matrix_view<T const, IdxT, raft::row_major>;
  auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);

  cuvs::neighbors::ivf_flat::build(*res_ptr, build_params, mds, *index);

  return index;
}

template <typename T, typename IdxT>
void _search(cuvsResources_t res,
             cuvsIvfFlatSearchParams params,
             cuvsIvfFlatIndex index,
             DLManagedTensor* queries_tensor,
             DLManagedTensor* neighbors_tensor,
             DLManagedTensor* distances_tensor,
             cuvsFilter filter)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_flat::index<T, IdxT>*>(index.addr);

  auto search_params = cuvs::neighbors::ivf_flat::search_params();
  convert_c_search_params(params, &search_params);

  using queries_mdspan_type   = raft::device_matrix_view<T const, IdxT, raft::row_major>;
  using neighbors_mdspan_type = raft::device_matrix_view<IdxT, IdxT, raft::row_major>;
  using distances_mdspan_type = raft::device_matrix_view<float, IdxT, raft::row_major>;
  auto queries_mds            = cuvs::core::from_dlpack<queries_mdspan_type>(queries_tensor);
  auto neighbors_mds          = cuvs::core::from_dlpack<neighbors_mdspan_type>(neighbors_tensor);
  auto distances_mds          = cuvs::core::from_dlpack<distances_mdspan_type>(distances_tensor);

  if (filter.type == NO_FILTER) {
    cuvs::neighbors::ivf_flat::search(
      *res_ptr, search_params, *index_ptr, queries_mds, neighbors_mds, distances_mds);
  } else if (filter.type == BITSET) {
    using filter_mdspan_type    = raft::device_vector_view<std::uint32_t, int64_t, raft::row_major>;
    auto removed_indices_tensor = reinterpret_cast<DLManagedTensor*>(filter.addr);
    auto removed_indices = cuvs::core::from_dlpack<filter_mdspan_type>(removed_indices_tensor);
    cuvs::core::bitset_view<std::uint32_t, int64_t> removed_indices_bitset(removed_indices,
                                                                           index_ptr->size());
    auto bitset_filter_obj = cuvs::neighbors::filtering::bitset_filter(removed_indices_bitset);
    cuvs::neighbors::ivf_flat::search(*res_ptr,
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

template <typename T, typename IdxT>
void _serialize(cuvsResources_t res, const char* filename, cuvsIvfFlatIndex index)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_flat::index<T, IdxT>*>(index.addr);
  cuvs::neighbors::ivf_flat::serialize(*res_ptr, std::string(filename), *index_ptr);
}

template <typename T, typename IdxT>
void* _deserialize(cuvsResources_t res, const char* filename)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto index   = new cuvs::neighbors::ivf_flat::index<T, IdxT>(*res_ptr);
  cuvs::neighbors::ivf_flat::deserialize(*res_ptr, std::string(filename), index);
  return index;
}

template <typename T, typename IdxT>
void _extend(cuvsResources_t res,
             DLManagedTensor* new_vectors,
             DLManagedTensor* new_indices,
             cuvsIvfFlatIndex index)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_flat::index<T, IdxT>*>(index.addr);
  using vectors_mdspan_type = raft::device_matrix_view<T const, IdxT, raft::row_major>;
  using indices_mdspan_type = raft::device_vector_view<IdxT, IdxT>;

  auto vectors_mds = cuvs::core::from_dlpack<vectors_mdspan_type>(new_vectors);
  auto indices_mds = cuvs::core::from_dlpack<indices_mdspan_type>(new_indices);

  cuvs::neighbors::ivf_flat::extend(*res_ptr, vectors_mds, indices_mds, index_ptr);
}

template <typename T, typename IdxT>
void get_centers(cuvsIvfFlatIndex index, DLManagedTensor* centers)
{
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_flat::index<T, IdxT>*>(index.addr);
  cuvs::core::to_dlpack(index_ptr->centers(), centers);
}
}  // namespace

extern "C" cuvsError_t cuvsIvfFlatIndexCreate(cuvsIvfFlatIndex_t* index)
{
  return cuvs::core::translate_exceptions([=] { *index = new cuvsIvfFlatIndex{}; });
}

extern "C" cuvsError_t cuvsIvfFlatIndexDestroy(cuvsIvfFlatIndex_t index_c_ptr)
{
  return cuvs::core::translate_exceptions([=] {
    auto index = *index_c_ptr;

    if (index.dtype.code == kDLFloat && index.dtype.bits == 32) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<float, int64_t>*>(index.addr);
      delete index_ptr;
    } else if (index.dtype.code == kDLFloat && index.dtype.bits == 16) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<half, int64_t>*>(index.addr);
      delete index_ptr;
    } else if (index.dtype.code == kDLInt) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<int8_t, int64_t>*>(index.addr);
      delete index_ptr;
    } else if (index.dtype.code == kDLUInt) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>*>(index.addr);
      delete index_ptr;
    }
    delete index_c_ptr;
  });
}

extern "C" cuvsError_t cuvsIvfFlatBuild(cuvsResources_t res,
                                        cuvsIvfFlatIndexParams_t params,
                                        DLManagedTensor* dataset_tensor,
                                        cuvsIvfFlatIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = dataset_tensor->dl_tensor;

    index->dtype = dataset.dtype;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      index->addr =
        reinterpret_cast<uintptr_t>(_build<float, int64_t>(res, *params, dataset_tensor));
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 16) {
      index->addr =
        reinterpret_cast<uintptr_t>(_build<half, int64_t>(res, *params, dataset_tensor));
    } else if (dataset.dtype.code == kDLInt && dataset.dtype.bits == 8) {
      index->addr =
        reinterpret_cast<uintptr_t>(_build<int8_t, int64_t>(res, *params, dataset_tensor));
    } else if (dataset.dtype.code == kDLUInt && dataset.dtype.bits == 8) {
      index->addr =
        reinterpret_cast<uintptr_t>(_build<uint8_t, int64_t>(res, *params, dataset_tensor));
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsIvfFlatSearch(cuvsResources_t res,
                                         cuvsIvfFlatSearchParams_t params,
                                         cuvsIvfFlatIndex_t index_c_ptr,
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

    RAFT_EXPECTS(neighbors.dtype.code == kDLInt && neighbors.dtype.bits == 64,
                 "neighbors should be of type int64_t");
    RAFT_EXPECTS(distances.dtype.code == kDLFloat && distances.dtype.bits == 32,
                 "distances should be of type float32");

    auto index = *index_c_ptr;
    RAFT_EXPECTS(queries.dtype.code == index.dtype.code, "type mismatch between index and queries");

    if (queries.dtype.code == kDLFloat && queries.dtype.bits == 32) {
      _search<float, int64_t>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
    } else if (queries.dtype.code == kDLFloat && queries.dtype.bits == 16) {
      _search<half, int64_t>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
    } else if (queries.dtype.code == kDLInt && queries.dtype.bits == 8) {
      _search<int8_t, int64_t>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
    } else if (queries.dtype.code == kDLUInt && queries.dtype.bits == 8) {
      _search<uint8_t, int64_t>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
    } else {
      RAFT_FAIL("Unsupported queries DLtensor dtype: %d and bits: %d",
                queries.dtype.code,
                queries.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsIvfFlatIndexParamsCreate(cuvsIvfFlatIndexParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    *params = new cuvsIvfFlatIndexParams{.metric                         = L2Expanded,
                                         .metric_arg                     = 2.0f,
                                         .add_data_on_build              = true,
                                         .n_lists                        = 1024,
                                         .kmeans_n_iters                 = 20,
                                         .kmeans_trainset_fraction       = 0.5,
                                         .adaptive_centers               = false,
                                         .conservative_memory_allocation = false};
  });
}

extern "C" cuvsError_t cuvsIvfFlatIndexParamsDestroy(cuvsIvfFlatIndexParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsIvfFlatSearchParamsCreate(cuvsIvfFlatSearchParams_t* params)
{
  return cuvs::core::translate_exceptions(
    [=] { *params = new cuvsIvfFlatSearchParams{.n_probes = 20}; });
}

extern "C" cuvsError_t cuvsIvfFlatSearchParamsDestroy(cuvsIvfFlatSearchParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsIvfFlatDeserialize(cuvsResources_t res,
                                              const char* filename,
                                              cuvsIvfFlatIndex_t index)
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
      index->addr       = reinterpret_cast<uintptr_t>(_deserialize<float, int64_t>(res, filename));
      index->dtype.code = kDLFloat;
    } else if (dtype.kind == 'e' && dtype.itemsize == 2) {
      index->addr       = reinterpret_cast<uintptr_t>(_deserialize<half, int64_t>(res, filename));
      index->dtype.code = kDLFloat;
      index->dtype.bits = 16;
    } else if (dtype.kind == 'i' && dtype.itemsize == 1) {
      index->addr       = reinterpret_cast<uintptr_t>(_deserialize<int8_t, int64_t>(res, filename));
      index->dtype.code = kDLInt;
    } else if (dtype.kind == 'u' && dtype.itemsize == 1) {
      index->addr = reinterpret_cast<uintptr_t>(_deserialize<uint8_t, int64_t>(res, filename));
      index->dtype.code = kDLUInt;
    } else {
      RAFT_FAIL(
        "Unsupported dtype in file %s itemsize %i kind %i", filename, dtype.itemsize, dtype.kind);
    }
  });
}

extern "C" cuvsError_t cuvsIvfFlatSerialize(cuvsResources_t res,
                                            const char* filename,
                                            cuvsIvfFlatIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      _serialize<float, int64_t>(res, filename, *index);
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      _serialize<half, int64_t>(res, filename, *index);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      _serialize<int8_t, int64_t>(res, filename, *index);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      _serialize<uint8_t, int64_t>(res, filename, *index);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsIvfFlatExtend(cuvsResources_t res,
                                         DLManagedTensor* new_vectors,
                                         DLManagedTensor* new_indices,
                                         cuvsIvfFlatIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      _extend<float, int64_t>(res, new_vectors, new_indices, *index);
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      _extend<half, int64_t>(res, new_vectors, new_indices, *index);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      _extend<int8_t, int64_t>(res, new_vectors, new_indices, *index);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      _extend<uint8_t, int64_t>(res, new_vectors, new_indices, *index);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsIvfFlatIndexGetNLists(cuvsIvfFlatIndex_t index, int64_t* n_lists)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<float, int64_t>*>(index->addr);
      *n_lists = index_ptr->n_lists();
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<half, int64_t>*>(index->addr);
      *n_lists = index_ptr->n_lists();
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<int8_t, int64_t>*>(index->addr);
      *n_lists = index_ptr->n_lists();
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>*>(index->addr);
      *n_lists = index_ptr->n_lists();
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsIvfFlatIndexGetDim(cuvsIvfFlatIndex_t index, int64_t* dim)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<float, int64_t>*>(index->addr);
      *dim = index_ptr->dim();
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<half, int64_t>*>(index->addr);
      *dim = index_ptr->dim();
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<int8_t, int64_t>*>(index->addr);
      *dim = index_ptr->dim();
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>*>(index->addr);
      *dim = index_ptr->dim();
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsIvfFlatIndexGetCenters(cuvsIvfFlatIndex_t index,
                                                  DLManagedTensor* centers)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      get_centers<float, int64_t>(*index, centers);
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      get_centers<half, int64_t>(*index, centers);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      get_centers<int8_t, int64_t>(*index, centers);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      get_centers<uint8_t, int64_t>(*index, centers);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}
