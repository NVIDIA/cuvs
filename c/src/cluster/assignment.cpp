/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>

#include <dlpack/dlpack.h>

#include <cuvs/cluster/assignment.h>
#include <cuvs/cluster/assignment.hpp>
#include <cuvs/core/c_api.h>

#include "../core/exceptions.hpp"
#include "../core/interop.hpp"
#include "assignment_convert.hpp"

extern "C" cuvsError_t cuvsAssignmentIndexCreate(cuvsAssignmentIndex_t *index) {
  return cuvs::core::translate_exceptions(
      [=] { *index = new cuvsAssignmentIndex{}; });
}

extern "C" cuvsError_t cuvsAssignmentIndexDestroy(cuvsAssignmentIndex_t index) {
  return cuvs::core::translate_exceptions([=] {
    cuvs::core::release_assignment_index(*index);
    delete index;
  });
}

extern "C" cuvsError_t
cuvsAssignmentParamsCreate(cuvsAssignmentParams_t *params) {
  return cuvs::core::translate_exceptions([=] {
    cuvs::cluster::ivf_assignment_params defaults;
    *params = new cuvsAssignmentParams{
        .kind = CUVS_ASSIGNMENT_IVF,
        .ivf = {.n_lists = defaults.n_lists,
                    .kmeans_n_iters = defaults.kmeans_n_iters,
                    .n_probes = defaults.n_probes},
        .index = nullptr};
  });
}

extern "C" cuvsError_t
cuvsAssignmentParamsDestroy(cuvsAssignmentParams_t params) {
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsAssignmentIndexBuild(cuvsResources_t res,
                                                cuvsAssignmentParams_t params,
                                                DLManagedTensor *centroids,
                                                cuvsDistanceType metric,
                                                cuvsAssignmentIndex_t index) {
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = reinterpret_cast<raft::resources *>(res);
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(centroids->dl_tensor),
                 "centroids should have device compatible memory");
    auto &dl = centroids->dl_tensor;
    RAFT_EXPECTS(dl.dtype.code == kDLFloat && dl.dtype.bits == 32,
                 "centroids must have a float32 dtype");
    using centroids_mdspan_type =
        raft::device_matrix_view<const float, int64_t, raft::row_major>;
    auto centroids_mds =
        cuvs::core::from_dlpack<centroids_mdspan_type>(centroids);

    switch (params->kind) {
    case CUVS_ASSIGNMENT_IVF:
      cuvs::core::store_assignment_index(
          *index,
          cuvs::cluster::build_ivf_assignment_index(
              *res_ptr, cuvs::core::convert_ivf_params(params->ivf),
              centroids_mds,
              static_cast<cuvs::distance::DistanceType>(metric)));
      break;
    default:
      RAFT_FAIL("Unsupported assignment kind: %d",
                static_cast<int>(params->kind));
    }
  });
}
