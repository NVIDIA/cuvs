/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>

#include <dlpack/dlpack.h>

#include <cuvs/cluster/soar.h>
#include <cuvs/cluster/soar.hpp>
#include <cuvs/core/c_api.h>

#include "../core/exceptions.hpp"
#include "../core/interop.hpp"

namespace {

template <typename LabelT>
void _predict(cuvsResources_t res,
              const cuvsSoarParams& params,
              DLManagedTensor* dataset_tensor,
              DLManagedTensor* centroids_tensor,
              DLManagedTensor* labels_tensor,
              DLManagedTensor* soar_labels_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  using matrix_type = raft::device_matrix_view<const float, int64_t, raft::row_major>;

  auto dataset   = cuvs::core::from_dlpack<matrix_type>(dataset_tensor);
  auto centroids = cuvs::core::from_dlpack<matrix_type>(centroids_tensor);
  auto labels =
    cuvs::core::from_dlpack<raft::device_vector_view<const LabelT, int64_t>>(labels_tensor);
  auto soar_labels =
    cuvs::core::from_dlpack<raft::device_vector_view<LabelT, int64_t>>(soar_labels_tensor);

  // Cluster ids are non-negative and the C++ API takes uint32, so an int32 label array has the
  // same bit pattern and can be handed over without a conversion pass over the labels.
  auto labels_u32 = raft::make_device_vector_view<const uint32_t, int64_t>(
    reinterpret_cast<const uint32_t*>(labels.data_handle()), labels.extent(0));
  auto soar_labels_u32 = raft::make_device_vector_view<uint32_t, int64_t>(
    reinterpret_cast<uint32_t*>(soar_labels.data_handle()), soar_labels.extent(0));

  cuvs::cluster::soar::params cpp_params;
  cpp_params.lambda = params.lambda;

  cuvs::cluster::soar::predict(
    *res_ptr, cpp_params, dataset, centroids, labels_u32, soar_labels_u32);
}

}  // namespace

extern "C" cuvsError_t cuvsSoarParamsCreate(cuvsSoarParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    cuvs::cluster::soar::params cpp_params;
    *params = new cuvsSoarParams{.lambda = cpp_params.lambda};
  });
}

extern "C" cuvsError_t cuvsSoarParamsDestroy(cuvsSoarParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsSoarPredict(cuvsResources_t res,
                                       cuvsSoarParams_t params,
                                       DLManagedTensor* dataset,
                                       DLManagedTensor* centroids,
                                       DLManagedTensor* labels,
                                       DLManagedTensor* soar_labels)
{
  return cuvs::core::translate_exceptions([=] {
    auto labels_dtype      = labels->dl_tensor.dtype;
    auto soar_labels_dtype = soar_labels->dl_tensor.dtype;

    RAFT_EXPECTS(labels_dtype.code == soar_labels_dtype.code &&
                   labels_dtype.bits == soar_labels_dtype.bits,
                 "labels and soar_labels must have the same dtype");

    if (labels_dtype.code == kDLUInt && labels_dtype.bits == 32) {
      _predict<uint32_t>(res, *params, dataset, centroids, labels, soar_labels);
    } else if (labels_dtype.code == kDLInt && labels_dtype.bits == 32) {
      _predict<int32_t>(res, *params, dataset, centroids, labels, soar_labels);
    } else {
      RAFT_FAIL("Unsupported labels DLtensor dtype: %d and bits: %d",
                labels_dtype.code,
                labels_dtype.bits);
    }
  });
}
