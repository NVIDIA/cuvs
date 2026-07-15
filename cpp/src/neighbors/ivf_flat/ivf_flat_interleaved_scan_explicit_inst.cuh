/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../detail/ann_utils.cuh"
#include "ivf_flat_interleaved_scan_jit.cuh"
#include <cstdint>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <optional>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <string>

#define CUVS_INST_IVF_FLAT_INTERLEAVED_SCAN(T, IdxT, SampleFilterT)                        \
  template void                                                                            \
  ivfflat_interleaved_scan<T,                                                              \
                           typename cuvs::spatial::knn::detail::utils::config<T>::value_t, \
                           IdxT,                                                           \
                           SampleFilterT>(const index<T, IdxT>& index,                     \
                                          const T* queries,                                \
                                          const uint32_t* coarse_query_results,            \
                                          const uint32_t n_queries,                        \
                                          const uint32_t queries_offset,                   \
                                          cuvs::distance::DistanceType metric,             \
                                          const uint32_t n_probes,                         \
                                          const uint32_t k,                                \
                                          const uint32_t max_samples,                      \
                                          const uint32_t* chunk_indices,                   \
                                          bool select_min,                                 \
                                          SampleFilterT sample_filter,                     \
                                          uint32_t* neighbors,                             \
                                          float* distances,                                \
                                          uint32_t& grid_dim_x,                            \
                                          rmm::cuda_stream_view stream,                    \
                                          const std::optional<std::string>& metric_udf);

#define COMMA ,
