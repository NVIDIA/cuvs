/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <dlpack/dlpack.h>

#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/ivf_pq.h>
#include <cuvs/neighbors/ivf_pq.hpp>

#include "../core/exceptions.hpp"
#include "../core/interop.hpp"

namespace cuvs::neighbors::ivf_pq {
void convert_c_index_params(cuvsIvfPqIndexParams params, cuvs::neighbors::ivf_pq::index_params* out)
{
  out->metric                   = static_cast<cuvs::distance::DistanceType>((int)params.metric),
  out->metric_arg               = params.metric_arg;
  out->add_data_on_build        = params.add_data_on_build;
  out->n_lists                  = params.n_lists;
  out->kmeans_n_iters           = params.kmeans_n_iters;
  out->kmeans_trainset_fraction = params.kmeans_trainset_fraction;
  out->pq_bits                  = params.pq_bits;
  out->pq_dim                   = params.pq_dim;
  out->codebook_kind =
    static_cast<cuvs::neighbors::ivf_pq::codebook_gen>((int)params.codebook_kind);
  out->force_random_rotation          = params.force_random_rotation;
  out->conservative_memory_allocation = params.conservative_memory_allocation;
  out->max_train_points_per_pq_code   = params.max_train_points_per_pq_code;
  out->codes_layout = static_cast<cuvs::neighbors::ivf_pq::list_layout>((int)params.codes_layout);
}
void convert_c_search_params(cuvsIvfPqSearchParams params,
                             cuvs::neighbors::ivf_pq::search_params* out)
{
  out->n_probes                 = params.n_probes;
  out->lut_dtype                = params.lut_dtype;
  out->internal_distance_dtype  = params.internal_distance_dtype;
  out->preferred_shmem_carveout = params.preferred_shmem_carveout;
  out->coarse_search_dtype      = params.coarse_search_dtype;
  out->max_internal_batch_size  = params.max_internal_batch_size;
}

}  // namespace cuvs::neighbors::ivf_pq

namespace {

template <typename T, typename IdxT>
void* _build(cuvsResources_t res, cuvsIvfPqIndexParams params, DLManagedTensor* dataset_tensor)
{
  RAFT_EXPECTS(res != 0, "res must not be NULL");
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  auto build_params = cuvs::neighbors::ivf_pq::index_params();
  convert_c_index_params(params, &build_params);

  cuvs::core::validate_dlpack_tensor(dataset_tensor, 2);
  auto dataset = dataset_tensor->dl_tensor;
  auto dim     = dataset.shape[1];

  auto index = new cuvs::neighbors::ivf_pq::index<IdxT>(*res_ptr, build_params, dim);

  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type = raft::device_matrix_view<const T, IdxT, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    cuvs::neighbors::ivf_pq::build(*res_ptr, build_params, mds, index);
  } else {
    using mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    cuvs::neighbors::ivf_pq::build(*res_ptr, build_params, mds, index);
  }

  return index;
}

template <typename T, typename IdxT>
void _search(cuvsResources_t res,
             cuvsIvfPqSearchParams params,
             cuvsIvfPqIndex index,
             DLManagedTensor* queries_tensor,
             DLManagedTensor* neighbors_tensor,
             DLManagedTensor* distances_tensor)
{
  RAFT_EXPECTS(res != 0, "res must not be NULL");
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_pq::index<IdxT>*>(index.addr);

  auto search_params = cuvs::neighbors::ivf_pq::search_params();
  cuvs::neighbors::ivf_pq::convert_c_search_params(params, &search_params);

  using queries_mdspan_type   = raft::device_matrix_view<const T, IdxT, raft::row_major>;
  using neighbors_mdspan_type = raft::device_matrix_view<IdxT, IdxT, raft::row_major>;
  using distances_mdspan_type = raft::device_matrix_view<float, IdxT, raft::row_major>;
  auto queries_mds            = cuvs::core::from_dlpack<queries_mdspan_type>(queries_tensor);
  auto neighbors_mds          = cuvs::core::from_dlpack<neighbors_mdspan_type>(neighbors_tensor);
  auto distances_mds          = cuvs::core::from_dlpack<distances_mdspan_type>(distances_tensor);

  cuvs::neighbors::ivf_pq::search(
    *res_ptr, search_params, *index_ptr, queries_mds, neighbors_mds, distances_mds);
}

template <typename IdxT>
void _serialize(cuvsResources_t res, const char* filename, cuvsIvfPqIndex index)
{
  RAFT_EXPECTS(res != 0, "res must not be NULL");
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_pq::index<IdxT>*>(index.addr);
  cuvs::neighbors::ivf_pq::serialize(*res_ptr, std::string(filename), *index_ptr);
}

template <typename IdxT>
void* _deserialize(cuvsResources_t res, const char* filename)
{
  RAFT_EXPECTS(res != 0, "res must not be NULL");
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto index   = new cuvs::neighbors::ivf_pq::index<IdxT>(*res_ptr);
  cuvs::neighbors::ivf_pq::deserialize(*res_ptr, std::string(filename), index);
  return index;
}

template <typename T, typename IdxT>
void _extend(cuvsResources_t res,
             DLManagedTensor* new_vectors,
             DLManagedTensor* new_indices,
             cuvsIvfPqIndex index)
{
  RAFT_EXPECTS(res != 0, "res must not be NULL");
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_pq::index<IdxT>*>(index.addr);

  bool on_device = cuvs::core::is_dlpack_device_compatible(new_vectors->dl_tensor);
  if (on_device != cuvs::core::is_dlpack_device_compatible(new_indices->dl_tensor)) {
    RAFT_FAIL("extend inputs must both either be on device memory or host memory");
  }

  if (on_device) {
    using vectors_mdspan_type = raft::device_matrix_view<const T, IdxT, raft::row_major>;
    using indices_mdspan_type = raft::device_vector_view<IdxT, IdxT>;
    auto vectors_mds          = cuvs::core::from_dlpack<vectors_mdspan_type>(new_vectors);
    auto indices_mds          = cuvs::core::from_dlpack<indices_mdspan_type>(new_indices);
    cuvs::neighbors::ivf_pq::extend(*res_ptr, vectors_mds, indices_mds, index_ptr);
  } else {
    using vectors_mdspan_type = raft::host_matrix_view<const T, IdxT, raft::row_major>;
    using indices_mdspan_type = raft::host_vector_view<IdxT, IdxT>;
    auto vectors_mds          = cuvs::core::from_dlpack<vectors_mdspan_type>(new_vectors);
    auto indices_mds          = cuvs::core::from_dlpack<indices_mdspan_type>(new_indices);
    cuvs::neighbors::ivf_pq::extend(*res_ptr, vectors_mds, indices_mds, index_ptr);
  }
}

template <typename IdxT>
void _get_centers(cuvsIvfPqIndex index, DLManagedTensor* output)
{
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_pq::index<IdxT>*>(index.addr);

  auto centers = index_ptr->centers();

  // mimic the behaviour of `extract_centers` without performing a copy of the data
  auto strided_centers = raft::make_device_strided_matrix_view(
    centers.data_handle(), centers.extent(0), index_ptr->dim(), index_ptr->dim_ext());

  cuvs::core::to_dlpack(strided_centers, output);
}

template <typename IdxT>
void _get_centers_padded(cuvsIvfPqIndex index, DLManagedTensor* output)
{
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_pq::index<IdxT>*>(index.addr);
  // Return the full padded centers [n_lists, dim_ext] as a contiguous array
  cuvs::core::to_dlpack(index_ptr->centers(), output);
}

template <typename IdxT>
void _get_pq_centers(cuvsIvfPqIndex index, DLManagedTensor* centers)
{
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_pq::index<IdxT>*>(index.addr);
  cuvs::core::to_dlpack(index_ptr->pq_centers(), centers);
}

template <typename IdxT>
void _get_centers_rot(cuvsIvfPqIndex index, DLManagedTensor* centers_rot)
{
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_pq::index<IdxT>*>(index.addr);
  cuvs::core::to_dlpack(index_ptr->centers_rot(), centers_rot);
}

template <typename IdxT>
void _get_rotation_matrix(cuvsIvfPqIndex index, DLManagedTensor* rotation_matrix)
{
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_pq::index<IdxT>*>(index.addr);
  cuvs::core::to_dlpack(index_ptr->rotation_matrix(), rotation_matrix);
}

template <typename IdxT>
void _get_list_sizes(cuvsIvfPqIndex index, DLManagedTensor* list_sizes)
{
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_pq::index<IdxT>*>(index.addr);
  cuvs::core::to_dlpack(index_ptr->list_sizes(), list_sizes);
}

template <typename IdxT>
void _unpack_contiguous_list_data(cuvsResources_t res,
                                  cuvsIvfPqIndex index,
                                  DLManagedTensor* out_codes,
                                  uint32_t label,
                                  uint32_t offset)
{
  RAFT_EXPECTS(res != 0, "res must not be NULL");
  auto index_ptr    = reinterpret_cast<cuvs::neighbors::ivf_pq::index<IdxT>*>(index.addr);
  using mdspan_type = raft::device_matrix_view<uint8_t, uint32_t, raft::row_major>;
  auto mds          = cuvs::core::from_dlpack<mdspan_type>(out_codes);
  auto res_ptr      = reinterpret_cast<raft::resources*>(res);

  cuvs::neighbors::ivf_pq::helpers::codepacker::unpack_contiguous_list_data(
    *res_ptr, *index_ptr, mds.data_handle(), mds.extent(0), label, offset);
}

template <typename IdxT>
void _get_list_indices(cuvsIvfPqIndex index,
                       uint32_t label,
                       DLManagedTensor* out_labels)
{
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_pq::index<IdxT>*>(index.addr);
  if (index_ptr->codes_layout() == cuvs::neighbors::ivf_pq::list_layout::FLAT) {
    auto& list =
      static_cast<cuvs::neighbors::ivf_pq::list_data_flat<IdxT>&>(*index_ptr->lists()[label]);
    cuvs::core::to_dlpack(list.indices.view(), out_labels);
  } else {
    auto& list = static_cast<cuvs::neighbors::ivf_pq::list_data_interleaved<IdxT>&>(
      *index_ptr->lists()[label]);
    cuvs::core::to_dlpack(list.indices.view(), out_labels);
  }
}

template <typename T, typename IdxT>
void _transform(cuvsResources_t res,
                                  cuvsIvfPqIndex index,
             DLManagedTensor* input_dataset,
             DLManagedTensor* output_labels,
             DLManagedTensor* output_dataset)
{
  RAFT_EXPECTS(res != 0, "res must not be NULL");
  auto index_ptr    = reinterpret_cast<cuvs::neighbors::ivf_pq::index<IdxT>*>(index.addr);
  auto res_ptr      = reinterpret_cast<raft::resources*>(res);

  using input_mdspan_type = raft::device_matrix_view<const T, IdxT, raft::row_major>;
  using labels_mdspan_type = raft::device_vector_view<uint32_t, IdxT, raft::row_major>;
  using output_mdspan_type = raft::device_matrix_view<uint8_t, IdxT, raft::row_major>;

  auto input_mds = cuvs::core::from_dlpack<input_mdspan_type>(input_dataset);
  auto labels_mds = cuvs::core::from_dlpack<labels_mdspan_type>(output_labels);
  auto output_mds = cuvs::core::from_dlpack<output_mdspan_type>(output_dataset);

  cuvs::neighbors::ivf_pq::transform(*res_ptr, *index_ptr, input_mds, labels_mds, output_mds);
}

}  // namespace

extern "C" cuvsError_t cuvsIvfPqIndexCreate(cuvsIvfPqIndex_t* index)
{
  return cuvs::core::translate_exceptions([=] { *index = new cuvsIvfPqIndex{}; });
}

extern "C" cuvsError_t cuvsIvfPqIndexDestroy(cuvsIvfPqIndex_t index_c_ptr)
{
  return cuvs::core::translate_exceptions([=] {
    if (index_c_ptr == nullptr) { return; }
    auto index = *index_c_ptr;

    auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_pq::index<int64_t>*>(index.addr);
    delete index_ptr;
    delete index_c_ptr;
  });
}

extern "C" cuvsError_t cuvsIvfPqBuild(cuvsResources_t res,
                                      cuvsIvfPqIndexParams_t params,
                                      DLManagedTensor* dataset_tensor,
                                      cuvsIvfPqIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset      = dataset_tensor->dl_tensor;
    index->dtype.code = dataset.dtype.code;
    index->dtype.bits = dataset.dtype.bits;

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

extern "C" cuvsError_t cuvsIvfPqSearch(cuvsResources_t res,
                                       cuvsIvfPqSearchParams_t params,
                                       cuvsIvfPqIndex_t index_c_ptr,
                                       DLManagedTensor* queries_tensor,
                                       DLManagedTensor* neighbors_tensor,
                                       DLManagedTensor* distances_tensor)
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
      _search<float, int64_t>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor);
    } else if (queries.dtype.code == kDLFloat && queries.dtype.bits == 16) {
      _search<half, int64_t>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor);
    } else if (queries.dtype.code == kDLInt && queries.dtype.bits == 8) {
      _search<int8_t, int64_t>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor);
    } else if (queries.dtype.code == kDLUInt && queries.dtype.bits == 8) {
      _search<uint8_t, int64_t>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor);
    } else {
      RAFT_FAIL("Unsupported queries DLtensor dtype: %d and bits: %d",
                queries.dtype.code,
                queries.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsIvfPqIndexParamsCreate(cuvsIvfPqIndexParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    *params = new cuvsIvfPqIndexParams{.metric                         = L2Expanded,
                                       .metric_arg                     = 2.0f,
                                       .add_data_on_build              = true,
                                       .n_lists                        = 1024,
                                       .kmeans_n_iters                 = 20,
                                       .kmeans_trainset_fraction       = 0.5,
                                       .pq_bits                        = 8,
                                       .pq_dim                         = 0,
                                       .codebook_kind                  = CUVS_IVF_PQ_CODEBOOK_GEN_PER_SUBSPACE,
                                       .force_random_rotation          = false,
                                       .conservative_memory_allocation = false,
                                       .max_train_points_per_pq_code   = 256,
                                       .codes_layout                   = CUVS_IVF_PQ_LIST_LAYOUT_INTERLEAVED};
  });
}

extern "C" cuvsError_t cuvsIvfPqIndexParamsDestroy(cuvsIvfPqIndexParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsIvfPqSearchParamsCreate(cuvsIvfPqSearchParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    *params = new cuvsIvfPqSearchParams{.n_probes                 = 20,
                                        .lut_dtype                = CUDA_R_32F,
                                        .internal_distance_dtype  = CUDA_R_32F,
                                        .coarse_search_dtype      = CUDA_R_32F,
                                        .max_internal_batch_size  = 4096,
                                        .preferred_shmem_carveout = 1.0};
  });
}

extern "C" cuvsError_t cuvsIvfPqSearchParamsDestroy(cuvsIvfPqSearchParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsIvfPqDeserialize(cuvsResources_t res,
                                            const char* filename,
                                            cuvsIvfPqIndex_t index)
{
  return cuvs::core::translate_exceptions(
    [=] { index->addr = reinterpret_cast<uintptr_t>(_deserialize<int64_t>(res, filename)); });
}

extern "C" cuvsError_t cuvsIvfPqSerialize(cuvsResources_t res,
                                          const char* filename,
                                          cuvsIvfPqIndex_t index)
{
  return cuvs::core::translate_exceptions([=] { _serialize<int64_t>(res, filename, *index); });
}

extern "C" cuvsError_t cuvsIvfPqExtend(cuvsResources_t res,
                                       DLManagedTensor* new_vectors,
                                       DLManagedTensor* new_indices,
                                       cuvsIvfPqIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto vectors = new_vectors->dl_tensor;

    // Set the index dtype if not already set (e.g., for view-type indices built from precomputed data)
    if (index->dtype.code == 0 && index->dtype.bits == 0) {
      index->dtype.code = vectors.dtype.code;
      index->dtype.bits = vectors.dtype.bits;
    }

    if (vectors.dtype.code == kDLFloat && vectors.dtype.bits == 32) {
      _extend<float, int64_t>(res, new_vectors, new_indices, *index);
    } else if (vectors.dtype.code == kDLFloat && vectors.dtype.bits == 16) {
      _extend<half, int64_t>(res, new_vectors, new_indices, *index);
    } else if (vectors.dtype.code == kDLInt && vectors.dtype.bits == 8) {
      _extend<int8_t, int64_t>(res, new_vectors, new_indices, *index);
    } else if (vectors.dtype.code == kDLUInt && vectors.dtype.bits == 8) {
      _extend<uint8_t, int64_t>(res, new_vectors, new_indices, *index);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", vectors.dtype.code, vectors.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsIvfPqIndexGetNLists(cuvsIvfPqIndex_t index, int64_t* n_lists)
{
  return cuvs::core::translate_exceptions([=] {
    auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_pq::index<int64_t>*>(index->addr);
    *n_lists       = index_ptr->n_lists();
  });
}

extern "C" cuvsError_t cuvsIvfPqIndexGetDim(cuvsIvfPqIndex_t index, int64_t* dim)
{
  return cuvs::core::translate_exceptions([=] {
    auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_pq::index<int64_t>*>(index->addr);
    *dim           = index_ptr->dim();
  });
}

extern "C" cuvsError_t cuvsIvfPqIndexGetSize(cuvsIvfPqIndex_t index, int64_t* size)
{
  return cuvs::core::translate_exceptions([=] {
    auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_pq::index<int64_t>*>(index->addr);
    *size          = index_ptr->size();
  });
}

extern "C" cuvsError_t cuvsIvfPqIndexGetPqDim(cuvsIvfPqIndex_t index, int64_t* pq_dim)
{
  return cuvs::core::translate_exceptions([=] {
    auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_pq::index<int64_t>*>(index->addr);
    *pq_dim        = index_ptr->pq_dim();
  });
}

extern "C" cuvsError_t cuvsIvfPqIndexGetPqBits(cuvsIvfPqIndex_t index, int64_t* pq_bits)
{
  return cuvs::core::translate_exceptions([=] {
    auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_pq::index<int64_t>*>(index->addr);
    *pq_bits       = index_ptr->pq_bits();
  });
}

extern "C" cuvsError_t cuvsIvfPqIndexGetPqLen(cuvsIvfPqIndex_t index, int64_t* pq_len)
{
  return cuvs::core::translate_exceptions([=] {
    auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_pq::index<int64_t>*>(index->addr);
    *pq_len        = index_ptr->pq_len();
  });
}

extern "C" cuvsError_t cuvsIvfPqIndexGetCenters(cuvsIvfPqIndex_t index, DLManagedTensor* centers)
{
  return cuvs::core::translate_exceptions([=] { _get_centers<int64_t>(*index, centers); });
}

extern "C" cuvsError_t cuvsIvfPqIndexGetCentersPadded(cuvsIvfPqIndex_t index,
                                                      DLManagedTensor* centers)
{
  return cuvs::core::translate_exceptions([=] { _get_centers_padded<int64_t>(*index, centers); });
}

extern "C" cuvsError_t cuvsIvfPqIndexGetPqCenters(cuvsIvfPqIndex_t index,
                                                  DLManagedTensor* pq_centers)
{
  return cuvs::core::translate_exceptions([=] { _get_pq_centers<int64_t>(*index, pq_centers); });
}

extern "C" cuvsError_t cuvsIvfPqIndexGetCentersRot(cuvsIvfPqIndex_t index,
                                                   DLManagedTensor* centers_rot)
{
  return cuvs::core::translate_exceptions([=] { _get_centers_rot<int64_t>(*index, centers_rot); });
}

extern "C" cuvsError_t cuvsIvfPqIndexGetRotationMatrix(cuvsIvfPqIndex_t index,
                                                       DLManagedTensor* rotation_matrix)
{
  return cuvs::core::translate_exceptions(
    [=] { _get_rotation_matrix<int64_t>(*index, rotation_matrix); });
}

extern "C" cuvsError_t cuvsIvfPqBuildPrecomputed(cuvsResources_t res,
                                                  cuvsIvfPqIndexParams_t params,
                                                  uint32_t dim,
                                                  DLManagedTensor* pq_centers_tensor,
                                                  DLManagedTensor* centers_tensor,
                                                  DLManagedTensor* centers_rot_tensor,
                                                  DLManagedTensor* rotation_matrix_tensor,
                                                  cuvsIvfPqIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(res != 0, "res must not be NULL");
    auto res_ptr = reinterpret_cast<raft::resources*>(res);

    auto build_params = cuvs::neighbors::ivf_pq::index_params();
    convert_c_index_params(*params, &build_params);

    // Verify all tensors are on device
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(pq_centers_tensor->dl_tensor),
                 "pq_centers should have device compatible memory");
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(centers_tensor->dl_tensor),
                 "centers should have device compatible memory");
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(centers_rot_tensor->dl_tensor),
                 "centers_rot should have device compatible memory");
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(rotation_matrix_tensor->dl_tensor),
                 "rotation_matrix should have device compatible memory");

    // Verify all tensors are float32
    auto& pq_centers_dl = pq_centers_tensor->dl_tensor;
    auto& centers_dl    = centers_tensor->dl_tensor;
    auto& centers_rot_dl = centers_rot_tensor->dl_tensor;
    auto& rotation_matrix_dl = rotation_matrix_tensor->dl_tensor;

    RAFT_EXPECTS(pq_centers_dl.dtype.code == kDLFloat && pq_centers_dl.dtype.bits == 32,
                 "pq_centers must be float32");
    RAFT_EXPECTS(centers_dl.dtype.code == kDLFloat && centers_dl.dtype.bits == 32,
                 "centers must be float32");
    RAFT_EXPECTS(centers_rot_dl.dtype.code == kDLFloat && centers_rot_dl.dtype.bits == 32,
                 "centers_rot must be float32");
    RAFT_EXPECTS(rotation_matrix_dl.dtype.code == kDLFloat && rotation_matrix_dl.dtype.bits == 32,
                 "rotation_matrix must be float32");

    // Convert DLPack tensors to mdspan views
    using pq_centers_mdspan_type = raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major>;
    using matrix_mdspan_type = raft::device_matrix_view<const float, uint32_t, raft::row_major>;

    auto pq_centers_mds = cuvs::core::from_dlpack<pq_centers_mdspan_type>(pq_centers_tensor);
    auto centers_mds = cuvs::core::from_dlpack<matrix_mdspan_type>(centers_tensor);
    auto centers_rot_mds = cuvs::core::from_dlpack<matrix_mdspan_type>(centers_rot_tensor);
    auto rotation_matrix_mds = cuvs::core::from_dlpack<matrix_mdspan_type>(rotation_matrix_tensor);

    // Build the index
    auto* idx = new cuvs::neighbors::ivf_pq::index<int64_t>(
      cuvs::neighbors::ivf_pq::build(
        *res_ptr, build_params, dim, pq_centers_mds, centers_mds, centers_rot_mds, rotation_matrix_mds));

    index->addr = reinterpret_cast<uintptr_t>(idx);
    // Leave dtype unset (0) - it will be set when extend() is called with actual data
    index->dtype.code = 0;
    index->dtype.bits = 0;
  });
}

extern "C" cuvsError_t cuvsIvfPqIndexGetListSizes(cuvsIvfPqIndex_t index,
                                                  DLManagedTensor* list_sizes)
{
  return cuvs::core::translate_exceptions([=] { _get_list_sizes<int64_t>(*index, list_sizes); });
}

extern "C" cuvsError_t cuvsIvfPqIndexUnpackContiguousListData(cuvsResources_t res,
                                                              cuvsIvfPqIndex_t index,
                                                              DLManagedTensor* out_codes,
                                                              uint32_t label,
                                                              uint32_t offset)
{
  return cuvs::core::translate_exceptions(
    [=] { _unpack_contiguous_list_data<int64_t>(res, *index, out_codes, label, offset); });
}

extern "C" cuvsError_t cuvsIvfPqIndexGetListIndices(cuvsIvfPqIndex_t index,
                                                    uint32_t label,
                                                    DLManagedTensor* out_labels)
{
  return cuvs::core::translate_exceptions(
    [=] { _get_list_indices<int64_t>(*index, label, out_labels); });
}

extern "C" cuvsError_t cuvsIvfPqTransform(cuvsResources_t res,
                                          cuvsIvfPqIndex_t index,
                                          DLManagedTensor* input_dataset,
                                          DLManagedTensor* output_labels,
                                          DLManagedTensor* output_dataset) {
  return cuvs::core::translate_exceptions(
    [=] {
      // Verify all tensors are on device
      RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(input_dataset->dl_tensor),
                   "input_dataset should have device compatible memory");
      RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(output_labels->dl_tensor),
                   "output_labels should have device compatible memory");
      RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(output_dataset->dl_tensor),
                   "output_dataset should have device compatible memory");

      // Verify dtypes of inputs
      auto& output_labels_dl = output_labels->dl_tensor;
      RAFT_EXPECTS(output_labels_dl.dtype.code == kDLUInt && output_labels_dl.dtype.bits == 32,
                   "output_labels must have a uint32 dtype ");
      auto& output_dataset_dl = output_dataset->dl_tensor;
      RAFT_EXPECTS(output_dataset_dl.dtype.code == kDLUInt && output_dataset_dl.dtype.bits == 8,
                   "output_dataset must have a uint8 dtype");

      auto & dataset = input_dataset->dl_tensor;
      if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
        _transform<float, int64_t>(res, *index, input_dataset, output_labels, output_dataset);
      } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 16) {
        _transform<half, int64_t>(res, *index, input_dataset, output_labels, output_dataset);
      } else if (dataset.dtype.code == kDLInt && dataset.dtype.bits == 8) {
        _transform<int8_t, int64_t>(res, *index, input_dataset, output_labels, output_dataset);
      } else if (dataset.dtype.code == kDLUInt && dataset.dtype.bits == 8) {
        _transform<uint8_t, int64_t>(res, *index, input_dataset, output_labels, output_dataset);
      } else {
        RAFT_FAIL("Unsupported input_dataset DLtensor dtype: %d and bits: %d",
                  dataset.dtype.code,
                  dataset.dtype.bits);
      }
    });
}
