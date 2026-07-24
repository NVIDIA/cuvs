/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <dlpack/dlpack.h>
#include <type_traits>

#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/tiered_index.h>
#include <cuvs/neighbors/tiered_index.hpp>


#include "../core/exceptions.hpp"
#include "../core/interop.hpp"
#include "cagra.hpp"
#include "c_api_box.hpp"
#include "ivf_flat.hpp"
#include "ivf_pq.hpp"

namespace {
using namespace cuvs::neighbors;

enum class tiered_cagra_dataset_layout : uint8_t { not_applicable, device_padded, device_standard };

struct tiered_c_api_index_box {
  void* index_ptr;
  cuvsTieredIndexANNAlgo algo;
  tiered_cagra_dataset_layout cagra_layout;
  cuvs::neighbors::c_api::detail::owner_record owner_rec;
};

template <typename UpstreamT>
static auto make_tiered_index_box(tiered_index::index<UpstreamT>* ptr,
                                  cuvsTieredIndexANNAlgo algo,
                                  tiered_cagra_dataset_layout cagra_layout =
                                    tiered_cagra_dataset_layout::not_applicable)
  -> tiered_c_api_index_box*
{
  return new tiered_c_api_index_box{
    ptr, algo, cagra_layout, cuvs::neighbors::c_api::detail::make_owner_record(ptr)};
}

static auto require_tiered_index_box(cuvsTieredIndex const& index, const char* null_handle_err)
  -> tiered_c_api_index_box*
{
  auto* box = reinterpret_cast<tiered_c_api_index_box*>(index.addr);
  if (box == nullptr) { RAFT_FAIL("%s", null_handle_err); }
  return box;
}

template <typename Fn>
static void with_tiered_cagra_index_by_layout(tiered_c_api_index_box* box,
                                              const char* null_handle_err,
                                              Fn&& fn)
{
  if (box == nullptr) { RAFT_FAIL("%s", null_handle_err); }
  RAFT_EXPECTS(box->algo == CUVS_TIERED_INDEX_ALGO_CAGRA,
               "with_tiered_cagra_index_by_layout requires CAGRA algo");
  if (box->cagra_layout == tiered_cagra_dataset_layout::device_padded) {
    auto* index_ptr = reinterpret_cast<tiered_index::index<cagra::device_padded_index<float, uint32_t>>*>(
      box->index_ptr);
    fn(index_ptr);
  } else if (box->cagra_layout == tiered_cagra_dataset_layout::device_standard) {
    auto* index_ptr = reinterpret_cast<tiered_index::index<cagra::device_standard_index<float, uint32_t>>*>(
      box->index_ptr);
    fn(index_ptr);
  } else {
    RAFT_FAIL("Tiered CAGRA index layout is not initialized");
  }
}

template <typename UpstreamT>
static auto require_typed_tiered_index(cuvsTieredIndex const& index,
                                       cuvsTieredIndexANNAlgo expected_algo,
                                       const char* null_handle_err,
                                       const char* wrong_algo_err)
  -> tiered_index::index<UpstreamT>*
{
  auto* box = require_tiered_index_box(index, null_handle_err);
  if (box->algo != expected_algo) { RAFT_FAIL("%s", wrong_algo_err); }
  return reinterpret_cast<tiered_index::index<UpstreamT>*>(box->index_ptr);
}

template <typename TieredIndexT>
struct tiered_upstream_type;

template <typename UpstreamT>
struct tiered_upstream_type<tiered_index::index<UpstreamT>> {
  using type = UpstreamT;
};

template <typename T>
void convert_c_index_params(cuvsTieredIndexParams params,
                            int64_t n_rows,
                            int64_t dim,
                            tiered_index::index_params<T>* out)
{
  out->min_ann_rows               = params.min_ann_rows;
  out->create_ann_index_on_extend = params.create_ann_index_on_extend;
  out->metric                     = static_cast<cuvs::distance::DistanceType>((int)params.metric);

  if constexpr (std::is_same_v<T, cagra::index_params>) {
    if (params.cagra_params != NULL) {
      cagra::convert_c_index_params(*params.cagra_params, n_rows, dim, out);
    }
  } else if constexpr (std::is_same_v<T, ivf_flat::index_params>) {
    if (params.ivf_flat_params != NULL) {
      ivf_flat::convert_c_index_params(*params.ivf_flat_params, out);
    }
  } else if constexpr (std::is_same_v<T, ivf_pq::index_params>) {
    if (params.ivf_pq_params != NULL) {
      ivf_pq::convert_c_index_params(*params.ivf_pq_params, out);
    }
  } else {
    RAFT_FAIL("unhandled index params type");
  }
}

template <typename T>
void* _build(cuvsResources_t res, cuvsTieredIndexParams params, DLManagedTensor* dataset_tensor)
{
  auto res_ptr      = reinterpret_cast<raft::resources*>(res);
  using mdspan_type = raft::device_matrix_view<const T, int64_t, raft::row_major>;
  auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);

  auto dataset = dataset_tensor->dl_tensor;
  RAFT_EXPECTS(dataset.ndim == 2, "dataset should be a 2-dimensional tensor");
  RAFT_EXPECTS(dataset.shape != nullptr, "dataset should have an initialized shape");

  switch (params.algo) {
    case CUVS_TIERED_INDEX_ALGO_CAGRA: {
      auto build_params = tiered_index::index_params<cagra::index_params>();
      convert_c_index_params(params, dataset.shape[0], dataset.shape[1], &build_params);
      if (cuvs::neighbors::matrix_row_width_matches_cagra_required(mds)) {
        auto padded_view = cuvs::neighbors::make_device_padded_dataset_view(*res_ptr, mds);
        auto* ptr = new tiered_index::index<cagra::device_padded_index<T, uint32_t>>(
          tiered_index::build(*res_ptr, build_params, padded_view));
        return make_tiered_index_box(
          ptr, CUVS_TIERED_INDEX_ALGO_CAGRA, tiered_cagra_dataset_layout::device_padded);
      }
      auto* ptr = new tiered_index::index<cagra::device_standard_index<T, uint32_t>>(
        tiered_index::build(*res_ptr, build_params, mds));
      return make_tiered_index_box(
        ptr, CUVS_TIERED_INDEX_ALGO_CAGRA, tiered_cagra_dataset_layout::device_standard);
    }
    case CUVS_TIERED_INDEX_ALGO_IVF_FLAT: {
      auto build_params = tiered_index::index_params<ivf_flat::index_params>();
      convert_c_index_params(params, dataset.shape[0], dataset.shape[1], &build_params);
      auto* ptr = new tiered_index::index<ivf_flat::index<T, int64_t>>(
        tiered_index::build(*res_ptr, build_params, mds));
      return make_tiered_index_box(ptr, CUVS_TIERED_INDEX_ALGO_IVF_FLAT);
    }
    case CUVS_TIERED_INDEX_ALGO_IVF_PQ: {
      auto build_params = tiered_index::index_params<ivf_pq::index_params>();
      convert_c_index_params(params, dataset.shape[0], dataset.shape[1], &build_params);
      auto* ptr = new tiered_index::index<ivf_pq::typed_index<T, int64_t>>(
        tiered_index::build(*res_ptr, build_params, mds));
      return make_tiered_index_box(ptr, CUVS_TIERED_INDEX_ALGO_IVF_PQ);
    }
    default: RAFT_FAIL("unsupported tiered index algorithm");
  }
}

template <typename UpstreamT>
void _search(cuvsResources_t res,
             void* params,
             cuvsTieredIndex index,
             DLManagedTensor* queries_tensor,
             DLManagedTensor* neighbors_tensor,
             DLManagedTensor* distances_tensor,
             cuvsFilter filter)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = require_typed_tiered_index<UpstreamT>(
    index, index.algo, "_search: null index handle", "_search: index algorithm mismatch");

  auto search_params = typename UpstreamT::search_params_type();
  if (params != NULL) {
    if constexpr (std::is_same_v<typename UpstreamT::search_params_type, cagra::search_params>) {
      auto c_params = reinterpret_cast<cuvsCagraSearchParams*>(params);
      cagra::convert_c_search_params(*c_params, &search_params);
    }
    if constexpr (std::is_same_v<typename UpstreamT::search_params_type, ivf_flat::search_params>) {
      auto c_params = reinterpret_cast<cuvsIvfFlatSearchParams*>(params);
      ivf_flat::convert_c_search_params(*c_params, &search_params);
    }
    if constexpr (std::is_same_v<typename UpstreamT::search_params_type, ivf_pq::search_params>) {
      auto c_params = reinterpret_cast<cuvsIvfPqSearchParams*>(params);
      ivf_pq::convert_c_search_params(*c_params, &search_params);
    }
  }

  using T = typename UpstreamT::value_type;

  using queries_mdspan_type   = raft::device_matrix_view<T const, int64_t, raft::row_major>;
  using neighbors_mdspan_type = raft::device_matrix_view<int64_t, int64_t, raft::row_major>;
  using distances_mdspan_type = raft::device_matrix_view<T, int64_t, raft::row_major>;
  auto queries_mds            = cuvs::core::from_dlpack<queries_mdspan_type>(queries_tensor);
  auto neighbors_mds          = cuvs::core::from_dlpack<neighbors_mdspan_type>(neighbors_tensor);
  auto distances_mds          = cuvs::core::from_dlpack<distances_mdspan_type>(distances_tensor);

  if (filter.type == NO_FILTER) {
    tiered_index::search(
      *res_ptr, search_params, *index_ptr, queries_mds, neighbors_mds, distances_mds);
  } else if (filter.type == BITSET) {
    using filter_mdspan_type    = raft::device_vector_view<std::uint32_t, int64_t, raft::row_major>;
    auto removed_indices_tensor = reinterpret_cast<DLManagedTensor*>(filter.addr);
    auto removed_indices = cuvs::core::from_dlpack<filter_mdspan_type>(removed_indices_tensor);
    cuvs::core::bitset_view<std::uint32_t, int64_t> removed_indices_bitset(removed_indices,
                                                                           index_ptr->size());
    auto bitset_filter_obj = cuvs::neighbors::filtering::bitset_filter(removed_indices_bitset);

    tiered_index::search(*res_ptr,
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

template <typename UpstreamT>
void _extend(cuvsResources_t res, DLManagedTensor* new_vectors, cuvsTieredIndex index)
{
  auto res_ptr              = reinterpret_cast<raft::resources*>(res);
  auto index_ptr            = require_typed_tiered_index<UpstreamT>(
    index, index.algo, "_extend: null index handle", "_extend: index algorithm mismatch");
  using T                   = typename UpstreamT::value_type;
  using vectors_mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;

  auto vectors_mds = cuvs::core::from_dlpack<vectors_mdspan_type>(new_vectors);

  tiered_index::extend(*res_ptr, vectors_mds, index_ptr);
}
template <typename UpstreamT>
void _merge(cuvsResources_t res,
            cuvsTieredIndexParams params,
            cuvsTieredIndex_t* indices,
            size_t num_indices,
            cuvsTieredIndex_t output_index)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  std::vector<cuvs::neighbors::tiered_index::index<UpstreamT>*> cpp_indices;

  int64_t n_rows = 0, dim = 0;
  tiered_cagra_dataset_layout expected_layout = tiered_cagra_dataset_layout::not_applicable;
  if (indices[0]->algo == CUVS_TIERED_INDEX_ALGO_CAGRA) {
    auto* box0 = require_tiered_index_box(*indices[0], "_merge: null index handle");
    expected_layout = box0->cagra_layout;
  }
  for (size_t i = 0; i < num_indices; ++i) {
    RAFT_EXPECTS(indices[i]->dtype.code == indices[0]->dtype.code,
                 "indices must all have the same dtype");
    RAFT_EXPECTS(indices[i]->dtype.bits == indices[0]->dtype.bits,
                 "indices must all have the same dtype");
    RAFT_EXPECTS(indices[i]->algo == indices[0]->algo,
                 "indices must all have the same index algorithm");
    if (indices[i]->algo == CUVS_TIERED_INDEX_ALGO_CAGRA) {
      auto* boxi = require_tiered_index_box(*indices[i], "_merge: null index handle");
      RAFT_EXPECTS(boxi->cagra_layout == expected_layout,
                   "indices must all have the same CAGRA layout");
    }

    auto idx_ptr = require_typed_tiered_index<UpstreamT>(*indices[i],
                                                         indices[0]->algo,
                                                         "_merge: null index handle",
                                                         "_merge: index algorithm mismatch");
    n_rows += idx_ptr->size();
    if (dim) {
      RAFT_EXPECTS(dim == idx_ptr->dim(), "indices must all have the same dimensionality");
    } else {
      dim = idx_ptr->dim();
    }
    cpp_indices.push_back(idx_ptr);
  }

  auto build_params = tiered_index::index_params<typename UpstreamT::index_params_type>();
  convert_c_index_params(params, n_rows, dim, &build_params);

  auto* ptr = new cuvs::neighbors::tiered_index::index<UpstreamT>(
    cuvs::neighbors::tiered_index::merge(*res_ptr, build_params, cpp_indices));
  auto cagra_layout = tiered_cagra_dataset_layout::not_applicable;
  if (indices[0]->algo == CUVS_TIERED_INDEX_ALGO_CAGRA) {
    auto* box0 = require_tiered_index_box(*indices[0], "_merge: null index handle");
    cagra_layout = box0->cagra_layout;
  }
  auto* box           = make_tiered_index_box(ptr, indices[0]->algo, cagra_layout);
  output_index->addr  = reinterpret_cast<uintptr_t>(box);
  output_index->dtype = indices[0]->dtype;
  output_index->algo  = indices[0]->algo;
}

}  // namespace

extern "C" cuvsError_t cuvsTieredIndexCreate(cuvsTieredIndex_t* index)
{
  return cuvs::core::translate_exceptions([=] { *index = new cuvsTieredIndex{}; });
}

extern "C" cuvsError_t cuvsTieredIndexDestroy(cuvsTieredIndex_t index_c_ptr)
{
  return cuvs::core::translate_exceptions([=] {
    auto index = *index_c_ptr;
    auto* box  = reinterpret_cast<tiered_c_api_index_box*>(index.addr);
    if (box != nullptr) {
      cuvs::neighbors::c_api::detail::destroy_owner_record(box->owner_rec);
      delete box;
    }
    delete index_c_ptr;
  });
}

extern "C" cuvsError_t cuvsTieredIndexBuild(cuvsResources_t res,
                                            cuvsTieredIndexParams_t params,
                                            DLManagedTensor* dataset_tensor,
                                            cuvsTieredIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = dataset_tensor->dl_tensor;
    index->dtype = dataset.dtype;
    index->algo  = params->algo;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      index->addr = reinterpret_cast<uintptr_t>(_build<float>(res, *params, dataset_tensor));
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsTieredIndexSearch(cuvsResources_t res,
                                             void* search_params,
                                             cuvsTieredIndex_t index_c_ptr,
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

    switch (index.algo) {
      case CUVS_TIERED_INDEX_ALGO_CAGRA: {
        auto* box = require_tiered_index_box(index, "cuvsTieredIndexSearch: null index handle");
        with_tiered_cagra_index_by_layout(
          box, "cuvsTieredIndexSearch: null index handle", [&](auto* typed_index) {
            using ann_t = typename tiered_upstream_type<std::remove_reference_t<decltype(*typed_index)>>::type;
            _search<ann_t>(
              res, search_params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
          });
        break;
      }
      case CUVS_TIERED_INDEX_ALGO_IVF_FLAT: {
        _search<ivf_flat::index<float, int64_t>>(
          res, search_params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
        break;
      }
      case CUVS_TIERED_INDEX_ALGO_IVF_PQ: {
        _search<ivf_pq::typed_index<float, int64_t>>(
          res, search_params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
        break;
      }
      default: RAFT_FAIL("unsupported tiered index algorithm");
    }
  });
}

extern "C" cuvsError_t cuvsTieredIndexParamsCreate(cuvsTieredIndexParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    cuvs::neighbors::tiered_index::index_params<cagra::index_params> cpp_params;
    *params = new cuvsTieredIndexParams{
      .metric                     = static_cast<cuvsDistanceType>((int)cpp_params.metric),
      .algo                       = CUVS_TIERED_INDEX_ALGO_CAGRA,
      .min_ann_rows               = cpp_params.min_ann_rows,
      .create_ann_index_on_extend = cpp_params.create_ann_index_on_extend};
  });
}

extern "C" cuvsError_t cuvsTieredIndexParamsDestroy(cuvsTieredIndexParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsTieredIndexExtend(cuvsResources_t res,
                                             DLManagedTensor* new_vectors,
                                             cuvsTieredIndex_t index_c_ptr)
{
  return cuvs::core::translate_exceptions([=] {
    auto index = *index_c_ptr;
    switch (index.algo) {
      case CUVS_TIERED_INDEX_ALGO_CAGRA: {
        auto* box = require_tiered_index_box(index, "cuvsTieredIndexExtend: null index handle");
        with_tiered_cagra_index_by_layout(
          box, "cuvsTieredIndexExtend: null index handle", [&](auto* typed_index) {
            using ann_t = typename tiered_upstream_type<std::remove_reference_t<decltype(*typed_index)>>::type;
            _extend<ann_t>(res, new_vectors, index);
          });
        break;
      }
      case CUVS_TIERED_INDEX_ALGO_IVF_FLAT: {
        _extend<ivf_flat::index<float, int64_t>>(res, new_vectors, index);
        break;
      }
      case CUVS_TIERED_INDEX_ALGO_IVF_PQ: {
        _extend<ivf_pq::typed_index<float, int64_t>>(res, new_vectors, index);
        break;
      }
      default: RAFT_FAIL("unsupported tiered index algorithm");
    }
  });
}

extern "C" cuvsError_t cuvsTieredIndexMerge(cuvsResources_t res,
                                            cuvsTieredIndexParams_t params,
                                            cuvsTieredIndex_t* indices,
                                            size_t num_indices,
                                            cuvsTieredIndex_t output_index)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(num_indices >= 1, "must have at least one index to merge");

    switch (indices[0]->algo) {
      case CUVS_TIERED_INDEX_ALGO_CAGRA: {
        auto* box = require_tiered_index_box(*indices[0], "cuvsTieredIndexMerge: null index handle");
        with_tiered_cagra_index_by_layout(
          box, "cuvsTieredIndexMerge: null index handle", [&](auto* typed_index) {
            using ann_t = typename tiered_upstream_type<std::remove_reference_t<decltype(*typed_index)>>::type;
            _merge<ann_t>(res, *params, indices, num_indices, output_index);
          });
        break;
      }
      case CUVS_TIERED_INDEX_ALGO_IVF_FLAT: {
        _merge<ivf_flat::index<float, int64_t>>(res, *params, indices, num_indices, output_index);
        break;
      }
      case CUVS_TIERED_INDEX_ALGO_IVF_PQ: {
        _merge<ivf_pq::typed_index<float, int64_t>>(
          res, *params, indices, num_indices, output_index);
        break;
      }
      default: RAFT_FAIL("unsupported tiered index algorithm");
    }
  });
}
