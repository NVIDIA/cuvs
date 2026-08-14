
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <dlpack/dlpack.h>
#include <fstream>

#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/numpy_serializer.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/brute_force.h>
#include <cuvs/neighbors/common.h>
#include <cuvs/neighbors/brute_force.hpp>

#include "../core/exceptions.hpp"
#include "../core/interop.hpp"

namespace {

template <typename T, typename LayoutT = raft::row_major, typename DistT = float>
void* _build(cuvsResources_t res,
             DLManagedTensor* dataset_tensor,
             cuvsDistanceType metric,
             T metric_arg)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  using mdspan_type = raft::device_matrix_view<T const, int64_t, LayoutT>;
  auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);

  cuvs::neighbors::brute_force::index_params params;
  params.metric     = static_cast<cuvs::distance::DistanceType>((int)metric);
  params.metric_arg = metric_arg;

  auto index_on_stack = cuvs::neighbors::brute_force::build(*res_ptr, params, mds);
  auto index_on_heap = new cuvs::neighbors::brute_force::index<T, DistT>(std::move(index_on_stack));
  return index_on_heap;
}

template <typename T, typename QueriesLayoutT = raft::row_major, typename DistT = float>
void _search(cuvsResources_t res,
             cuvsBruteForceIndex index,
             DLManagedTensor* queries_tensor,
             DLManagedTensor* neighbors_tensor,
             DLManagedTensor* distances_tensor,
             cuvsFilter prefilter)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::brute_force::index<T, DistT>*>(index.addr);

  using queries_mdspan_type   = raft::device_matrix_view<T const, int64_t, QueriesLayoutT>;
  using neighbors_mdspan_type = raft::device_matrix_view<int64_t, int64_t, raft::row_major>;
  using distances_mdspan_type = raft::device_matrix_view<DistT, int64_t, raft::row_major>;
  using prefilter_mds_type    = raft::device_vector_view<uint32_t, int64_t>;

  auto queries_mds   = cuvs::core::from_dlpack<queries_mdspan_type>(queries_tensor);
  auto neighbors_mds = cuvs::core::from_dlpack<neighbors_mdspan_type>(neighbors_tensor);
  auto distances_mds = cuvs::core::from_dlpack<distances_mdspan_type>(distances_tensor);

  cuvs::neighbors::brute_force::search_params params;

  if (prefilter.type == NO_FILTER) {
    cuvs::neighbors::brute_force::search(*res_ptr,
                                         params,
                                         *index_ptr,
                                         queries_mds,
                                         neighbors_mds,
                                         distances_mds,
                                         cuvs::neighbors::filtering::none_sample_filter{});
  } else if (prefilter.type == BITMAP) {
    using prefilter_bmp_type = cuvs::core::bitmap_view<uint32_t, int64_t>;
    auto prefilter_ptr       = reinterpret_cast<DLManagedTensor*>(prefilter.addr);
    auto prefilter_mds       = cuvs::core::from_dlpack<prefilter_mds_type>(prefilter_ptr);
    const auto prefilter     = cuvs::neighbors::filtering::bitmap_filter(
      prefilter_bmp_type((uint32_t*)prefilter_mds.data_handle(),
                         queries_mds.extent(0),
                         index_ptr->dataset().extent(0)));
    cuvs::neighbors::brute_force::search(
      *res_ptr, params, *index_ptr, queries_mds, neighbors_mds, distances_mds, prefilter);
  } else if (prefilter.type == BITSET) {
    using prefilter_bst_type = cuvs::core::bitset_view<uint32_t, int64_t>;
    auto prefilter_ptr       = reinterpret_cast<DLManagedTensor*>(prefilter.addr);
    auto prefilter_mds       = cuvs::core::from_dlpack<prefilter_mds_type>(prefilter_ptr);
    const auto prefilter     = cuvs::neighbors::filtering::bitset_filter(
      prefilter_bst_type((uint32_t*)prefilter_mds.data_handle(), index_ptr->dataset().extent(0)));
    cuvs::neighbors::brute_force::search(
      *res_ptr, params, *index_ptr, queries_mds, neighbors_mds, distances_mds, prefilter);
  } else {
    RAFT_FAIL("Unsupported prefilter type");
  }
}

template <typename T, typename DistT = float>
void _serialize(cuvsResources_t res, const char* filename, cuvsBruteForceIndex index)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::brute_force::index<T, DistT>*>(index.addr);
  cuvs::neighbors::brute_force::serialize(*res_ptr, std::string(filename), *index_ptr);
}

template <typename T, typename DistT = float>
void* _deserialize(cuvsResources_t res, const char* filename)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto index   = new cuvs::neighbors::brute_force::index<T, DistT>(*res_ptr);
  cuvs::neighbors::brute_force::deserialize(*res_ptr, std::string(filename), index);
  return index;
}
}  // namespace

extern "C" cuvsError_t cuvsBruteForceIndexCreate(cuvsBruteForceIndex_t* index)
{
  return cuvs::core::translate_exceptions([=] { *index = new cuvsBruteForceIndex{}; });
}

extern "C" cuvsError_t cuvsBruteForceIndexDestroy(cuvsBruteForceIndex_t index_c_ptr)
{
  return cuvs::core::translate_exceptions([=] {
    auto index = *index_c_ptr;

    if ((index.dtype.code == kDLFloat) && index.dtype.bits == 32) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::brute_force::index<float, float>*>(index.addr);
      delete index_ptr;
    } else if ((index.dtype.code == kDLFloat) && index.dtype.bits == 16) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::brute_force::index<half, float>*>(index.addr);
      delete index_ptr;
    }
    delete index_c_ptr;
  });
}

extern "C" cuvsError_t cuvsBruteForceBuild(cuvsResources_t res,
                                           DLManagedTensor* dataset_tensor,
                                           cuvsDistanceType metric,
                                           float metric_arg,
                                           cuvsBruteForceIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = dataset_tensor->dl_tensor;
    index->dtype = dataset.dtype;

    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      if (cuvs::core::is_c_contiguous(dataset_tensor)) {
        index->addr =
          reinterpret_cast<uintptr_t>(_build<float>(res, dataset_tensor, metric, metric_arg));
      } else if (cuvs::core::is_f_contiguous(dataset_tensor)) {
        index->addr = reinterpret_cast<uintptr_t>(
          _build<float, raft::col_major>(res, dataset_tensor, metric, metric_arg));
      } else {
        RAFT_FAIL("dataset input to cuvsBruteForceBuild must be contiguous (non-strided)");
      }
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 16) {
      if (cuvs::core::is_c_contiguous(dataset_tensor)) {
        index->addr =
          reinterpret_cast<uintptr_t>(_build<half>(res, dataset_tensor, metric, metric_arg));
      } else if (cuvs::core::is_f_contiguous(dataset_tensor)) {
        index->addr = reinterpret_cast<uintptr_t>(
          _build<half, raft::col_major>(res, dataset_tensor, metric, metric_arg));
      } else {
        RAFT_FAIL("dataset input to cuvsBruteForceBuild must be contiguous (non-strided)");
      }
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsBruteForceSearch(cuvsResources_t res,
                                            cuvsBruteForceIndex_t index_c_ptr,
                                            DLManagedTensor* queries_tensor,
                                            DLManagedTensor* neighbors_tensor,
                                            DLManagedTensor* distances_tensor,
                                            cuvsFilter prefilter)
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
      if (cuvs::core::is_c_contiguous(queries_tensor)) {
        _search<float>(res, index, queries_tensor, neighbors_tensor, distances_tensor, prefilter);
      } else if (cuvs::core::is_f_contiguous(queries_tensor)) {
        _search<float, raft::col_major>(
          res, index, queries_tensor, neighbors_tensor, distances_tensor, prefilter);
      } else {
        RAFT_FAIL("queries input to cuvsBruteForceSearch must be contiguous (non-strided)");
      }
    } else if (queries.dtype.code == kDLFloat && queries.dtype.bits == 16) {
      if (cuvs::core::is_c_contiguous(queries_tensor)) {
        _search<half>(res, index, queries_tensor, neighbors_tensor, distances_tensor, prefilter);
      } else if (cuvs::core::is_f_contiguous(queries_tensor)) {
        _search<half, raft::col_major>(
          res, index, queries_tensor, neighbors_tensor, distances_tensor, prefilter);
      } else {
        RAFT_FAIL("queries input to cuvsBruteForceSearch must be contiguous (non-strided)");
      }
    } else {
      RAFT_FAIL("Unsupported queries DLtensor dtype: %d and bits: %d",
                queries.dtype.code,
                queries.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsBruteForceDeserialize(cuvsResources_t res,
                                                 const char* filename,
                                                 cuvsBruteForceIndex_t index)
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
      index->dtype.code = kDLFloat;
      index->addr       = reinterpret_cast<uintptr_t>(_deserialize<float>(res, filename));
    } else if (dtype.kind == 'e' && dtype.itemsize == 2) {
      index->dtype.code = kDLFloat;
      index->addr       = reinterpret_cast<uintptr_t>(_deserialize<half>(res, filename));
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsBruteForceSerialize(cuvsResources_t res,
                                               const char* filename,
                                               cuvsBruteForceIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      _serialize<float>(res, filename, *index);
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      _serialize<half>(res, filename, *index);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}
